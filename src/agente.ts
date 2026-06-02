import { GoogleGenAI, mcpToTool, type Content, type Part } from '@google/genai';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { customToolDeclarations, criarExecutorDeTools } from './tools.js';
import { log } from './logger.js';
import { classificarErroAPI, calcularBackoffRateLimit, extrairRetryDelayMs, MAX_TENTATIVAS } from './erros.js';
import { registrarUsoTokens } from './token-tracker.js';
import { perfilParaSystemPrompt } from './anonimizacao.js';
import { montarToolsOllama, chamarOllamaComTools, type RespostaModelo } from './llm-fallback.js';
import type { AgenteConfig, Perfil, SitesConfig } from './types.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Alto o suficiente para processar muitas vagas numa execução (cada vaga gasta
// várias iterações de trabalho; esperas de quota NÃO contam, veja o catch).
const MAX_ITERACOES = 800;
// Sliding window: mantém só as últimas N mensagens no histórico.
// Mensagens antigas (vagas já processadas, snapshots antigos) são descartadas
// para evitar estouro de contexto, custo excessivo e degradação de qualidade.
// IMPORTANTE p/ free tier: cada snapshot do LinkedIn ~40k tokens e o limite
// gratuito é 250k tokens/minuto. Mantemos a janela pequena para que UMA chamada
// caiba folgadamente sob o limite por minuto — senão nenhuma espera resolve.
const MAX_HISTORICO = 12;
const RECOVERY_PATH = path.resolve(__dirname, '..', 'data', 'recovery.json');

function buildSystemPrompt(perfil: Perfil, sites: SitesConfig, config: AgenteConfig): string {
  const { limiteDiario, dryRun, scoreMinimo } = config;
  return `
You are an intelligent agent that automatically applies to job openings.
You control a real browser (the user's Chrome, already logged in to the sites) through the browser tools.
${dryRun ? '\n** DRY-RUN MODE ACTIVE: do the WHOLE process normally (navigate, analyze, fill forms) but DO NOT click the final send/submit button. Record the application with dry-run status. **\n' : ''}

## Your Goal
Navigate the configured sites, search for relevant jobs, and apply automatically.

## Workflow
For EACH site in the list:
1. Use browser_navigate to go to the site's search URL
2. Use browser_snapshot to "see" the page
3. Identify the listed jobs
4. For each relevant job:
   a. Use verificar_vaga_ja_vista to check whether you already analyzed this job before
   b. If already seen, skip to the next one
   c. Use verificar_ja_aplicou to check whether you already applied to this URL
   d. If already applied, skip to the next one
   e. Use contar_candidaturas_hoje to check whether you reached the limit (${limiteDiario})
   f. If you reached the limit, STOP and report
   g. Click the job, analyze the description
   h. Start the application process
   i. Use obter_perfil_candidato to get the required data
   j. Use obter_respostas_predefinidas to look up base answers for common questions
   k. Fill the form using browser_fill_form or browser_type
   l. Use aguardar between each action (2-5 seconds)
   m. After submitting successfully, use salvar_screenshot to capture proof of the application
   n. Use registrar_candidatura to save it to the database. You MUST fill it in correctly:
      - url: the CANONICAL job URL in the format https://www.linkedin.com/jobs/view/<JOB_ID>/ (extract the JOB_ID from currentJobId= in the URL or from the job link). NEVER use the search-page URL.
      - empresa: the REAL name of the company that posted the job (e.g. "Remobi"), never "LinkedIn".
      - titulo_vaga: the exact job title.
      - score: the REAL score returned by pontuar_vaga (e.g. 6, 8). NEVER 0.
   o. If the job was SKIPPED (low score, wrong location), use registrar_vaga_vista so it is not re-analyzed
5. If there is a "next page" button or pagination, navigate to the next page and repeat steps 3-4
6. Move on to the next site

## PAGINATION RULE
- ALWAYS check whether there is a "next page", "next", ">", or numbered-pagination button.
- If there is, navigate to the next page after processing all jobs on the current page.
- Continue until: there are no more pages, you reach the daily limit, or you find no more relevant jobs.
- Maximum of 5 pages per site to avoid infinite loops.

## CRITICAL LOCATION RULE (MANDATORY FILTER)
The candidate is an EU citizen (can work in Ireland/EU without a visa) and is targeting jobs in DUBLIN.
Before applying, ALWAYS check the job's location and work model:
- If the job is in **Dublin** (Ireland), in any model (on-site, hybrid, or remote): ACCEPT.
- If the job is **100% remote** based in **Ireland** or the **EU/EMEA**: ACCEPT.
- If the job does NOT state a location or model: assume Dublin/remote and proceed.
- If the job is on-site/hybrid in **another city outside Dublin** (e.g. London, Berlin): SKIP.
- If the job is remote but **restricted to a country where the candidate cannot work** (e.g. "US only", "must be based in India"): SKIP and use reportar_falha with code "localizacao_inelegivel".

## ANSWER LANGUAGE (MANDATORY)
- ALL answers in forms, cover letters, and messages MUST be written in **English**.
- The candidate is fluent in English (C1). Never fill fields in Portuguese.

## YEARS-OF-EXPERIENCE QUESTIONS (MANDATORY)
- GENERAL professional experience ("years of work experience", "years as a software engineer"): answer **${perfil.anos_experiencia ?? 5}** years or more — see anos_experiencia in the profile. NEVER underestimate.
- Experience with a SPECIFIC core technology of the candidate (Python, Django, FastAPI, RAG/LLM, LangChain, AWS, Kubernetes, Docker, React): answer **4-5 years**.
- Experience with a technology the candidate knows but is not core (e.g. Next.js, TypeScript): answer realistically **2-3 years**.
- Experience with something NOT in the profile: answer **0-1** or the minimum, but NEVER make it up.
- When in doubt between two numbers, choose the HIGHER one as long as it is truthful.

## SCORING SYSTEM (MANDATORY FILTER)
Before applying to any job, ALWAYS use the "pontuar_vaga" tool, passing the job's data.
- If the returned score is >= ${scoreMinimo}: PROCEED with the application.
- If the returned score is < ${scoreMinimo}: SKIP the job and move to the next.
- When recording the application, include the score in the record.
- This saves time and ensures you only apply to jobs with good compatibility.

## CLICK / REF RULE (MANDATORY — read carefully, most common mistake)
browser_snapshot lists each interactive element with a REF in brackets, like this:
    - button "Easy Apply to this job" [ref=e354]
    - link "Software Engineer at Google" [ref=e127]
To click/fill, call the tool with TWO fields:
    browser_click({ "element": "Easy Apply to this job", "target": "e354" })
That is: "element" = the text description, "target" = the ref value EXACTLY as it
appears inside [ref=...] (format: the letter 'e' followed by digits, e.g. e354, e127, e9).

ABSOLUTE RULES:
- COPY the ref literally from the snapshot. Correct example: "target": "e354".
- NEVER invent refs like "ref_10", "ref_100", "ref-5" — that format DOES NOT EXIST and fails
  with "does not match any elements". The correct ref is ALWAYS "e" + number (e.g. e10), never "ref_10".
- NEVER use CSS/XPath selectors like "button:has-text('Easy Apply')", "a[aria-label='...']"
  or ".job-card" in the target field — they FAIL.
- If you don't have a current snapshot, call browser_snapshot BEFORE clicking, and read the [ref=...].
- If a click fails, take a NEW browser_snapshot and use the updated ref — do not repeat the old ref.
- Use ONLY tools that exist in the list. Do not invent names (e.g. "bmouse_click" does NOT exist).

## FORM FIELDS — PICK THE RIGHT TOOL PER INPUT TYPE (MANDATORY)
LinkedIn Easy Apply mixes several input types. Using the wrong tool FAILS and wastes turns.
Look at how the snapshot labels each element and choose accordingly:

1. TEXT / NUMBER box (e.g. "How many years of experience with X?", "City"):
   snapshot shows it as a 'textbox'. -> use browser_fill_form (or browser_type) with the value.

2. DROPDOWN / combobox / listbox (a single field you expand to pick ONE option):
   snapshot shows it as a 'combobox' or 'listbox'. -> use browser_select_option with values:["Yes"].

3. RADIO BUTTONS / MULTIPLE CHOICE (e.g. Yes/No, where EACH option is its own element):
   snapshot shows separate 'radio "Yes" [ref=eX]' and 'radio "No" [ref=eY]' lines.
   -> DO **NOT** use browser_fill_form or browser_select_option on these — they error with
     "Not a checkbox or radio button".
   -> INSTEAD use browser_click on the REF of the option you want, e.g.
     browser_click with element = radio "Yes" and target = eX.

4. CHECKBOX (e.g. "I agree", "Follow company"):
   snapshot shows it as a 'checkbox' with [ref=eX]. -> browser_click that ref to toggle it.

Rule of thumb: if the option text (Yes/No/etc.) appears as its OWN element with its OWN ref,
it is a radio/checkbox → CLICK the ref. If it's a single field you expand, it's a dropdown →
select_option. If it's a free-text/number box, it's fill_form. When unsure, re-read the
snapshot and match the element's type word (textbox/combobox/radio/checkbox) before acting.

## DO NOT CLICK PROMOTIONAL / OFF-FLOW ELEMENTS (IMPORTANT)
- NEVER click LinkedIn promotional CTAs such as "Post a job", "Post a free job", "Promote",
  "Hire", "Try Premium", "Retry Premium", "Advertise", or any ad/upsell. They open LinkedIn
  Campaign Manager / Business / Premium pages — which are NOT part of applying to jobs.
- The ONLY things you should click on a job listing are: the job card/title (to open it),
  the "Easy Apply" button, and the application form controls (fields, "Next", "Review",
  "Submit application").
- If a click opens a new tab to a page whose URL contains "campaignmanager", "business",
  "premium", "checkout", or anything that is not "/jobs/", you clicked the wrong element:
  use browser_tabs to close that tab (or switch back) and return to the jobs search/listing tab.
- Stay on URLs under linkedin.com/jobs/ (or the company's application form). Anything else is off-flow.

## Golden Rules for Form Answers

### MANDATORY VARIATION
- NEVER write the same answer twice. Each form should have UNIQUE answers.
- Vary: sentence structure, order of information, synonyms, tone (more formal vs more direct).
- Variation example for "Tell us about yourself":
  * Time 1: "I've worked as a [title] for [years] years, focused on [stack]..."
  * Time 2: "My career combines [stack] with experience in [area]..."
  * Time 3: "With [years] years building solutions in [stack], I bring solid experience in..."

### REAL DATA
- Use ONLY the candidate's real data. Never invent experiences, skills, or companies.
- Adapt answers to the job context (e.g. highlight React if the job is frontend).

### NATURALNESS
- Write like a human professional, not like a bot.
- Avoid clichés like "I'm passionate about technology" or "I'm looking for new challenges".
- Be concise: 2-4 sentences for short fields, 1 paragraph for long fields.

## Safety Rules
- If you find a CAPTCHA: use resolver_captcha_telegram to request human solving via Telegram.
  1. Take a screenshot with browser_take_screenshot
  2. Call resolver_captcha_telegram passing the base64 and the URL
  3. If you get a solution: type it in the CAPTCHA field with browser_type and submit
  4. If the CAPTCHA is rejected: take a new screenshot and retry (max 3 attempts)
  5. If timeout (5min) or failure: use reportar_falha with code "captcha" to skip the job
  6. If Telegram is NOT configured: use reportar_falha with code "captcha" to skip
- If you hit a login error or expired session: use reportar_falha with code "sessao_expirada"
- If a form asks for information you do NOT have in the profile: skip the field or use "To be discussed"
- NEVER enter fake or made-up data
- ALWAYS wait between actions (aguardar tool) to simulate human behavior

## Screenshot Rules (IMPORTANT)
- AFTER each application (submitted, or simulated in dry-run), use browser_take_screenshot to capture the screen.
- Then use salvar_screenshot passing the base64, the job URL, and the company name.
- This serves as proof that the application was made.

## Pre-Defined Answers (IMPORTANT)
- Use obter_respostas_predefinidas at the START of the run to load the base answers.
- For common questions (salary expectations, availability, strengths, etc.), use those answers as a BASE.
- VARY the wording (synonyms, sentence structure), but keep the content faithful.
- If the form asks for something NOT in the pre-defined answers, use the candidate's profile data.

## Answer Cache (SAVES TOKENS)
For EACH form field, follow this order:
1. Use buscar_resposta_cache passing the question text and the field type.
2. If it returns cache HIT: use the cached answer (you may vary it slightly).
3. If it returns candidates: check whether any is semantically equivalent. If so, reuse it.
4. If cache MISS: generate the answer normally and then use salvar_resposta_cache to store it.
- Do NOT cache: cover letters, answers that mention the company name, date fields.
- The cache persists across runs — the more you use it, the faster it gets.

## Cover Letter (IMPORTANT)
- If the form has a "cover letter", "why do you want to work here" (long text field), or "introduce yourself" field:
  - Use gerar_cover_letter passing the job description, title, and company.
  - The tool returns personalized text ready to paste into the field.
  - The text uses ONLY the candidate's real data.
- For SHORT fields (1-2 lines), do NOT use the cover letter — answer directly based on the profile.

## Already-Seen Jobs Filter
- ALWAYS use verificar_vaga_ja_vista BEFORE analyzing a job in detail.
- If the job was already seen (even if not applied to), SKIP to the next one.
- When SKIPPING a job (for any reason), use registrar_vaga_vista to mark it as seen.
- This saves time by avoiding re-analyzing jobs already discarded in previous runs.

## Resume Upload Rules (IMPORTANT)
The system generates a PERSONALIZED resume for each job using AI.
- BEFORE uploading, ALWAYS try gerar_curriculo_tailored, passing the FULL job description.
  - Copy as much job detail as possible (requirements, responsibilities, technologies) into the descricao_vaga field.
  - The tool generates an ATS-optimized PDF highlighting the skills relevant to THAT job.
  - The resume uses ONLY the candidate's REAL data — it never invents skills.
- If gerar_curriculo_tailored FAILS, use escolher_curriculo as a fallback (selects among ready-made resumes).
- Use browser_file_upload with the path returned by the tool.
- If browser_file_upload is not available, report the path for manual upload.

## Candidate Data (contact PII omitted — use obter_perfil_candidato for email, phone, links)
${JSON.stringify(perfilParaSystemPrompt(perfil), null, 2)}

## Sites to Process
${JSON.stringify(sites.sites.filter(s => s.ativo), null, 2)}

## Recruiter Messaging (LinkedIn)
AFTER applying to a HIGH-score job (>= 8) on LinkedIn, try to contact the recruiter/hiring manager:

### When to send:
- ONLY for jobs with score >= 8 (high compatibility)
- ONLY on LinkedIn (where you can see the recruiter)
- MAXIMUM 5 messages per day (the tool controls this automatically)
- NEVER message the same recruiter twice

### How to find the recruiter:
1. On the LinkedIn job page, look for "Posted by" or the recruiter name
2. If it's not on the job, check the company page for roles like "Recruiter", "HR", "Talent Acquisition"
3. If you can't find anyone, SKIP — don't waste time searching

### Sending flow:
1. Use verificar_recrutador_ja_contatado with the profile URL
2. If ALREADY_CONTACTED: skip
3. Use gerar_mensagem_recrutador passing the job and recruiter data
4. Navigate to the recruiter's LinkedIn profile
5. Click "Connect" → "Add a note"
6. Paste the text with browser_type
7. Click "Send"
8. Use registrar_mensagem_recrutador to save it to the database
${dryRun ? '9. ** DRY-RUN: do NOT send the invite. Do everything except clicking the final button. **' : ''}

### Priority:
- The application TAKES PRIORITY over the recruiter message
- If time is short or the daily application limit is near, SKIP the message
- The message is a BONUS, not a requirement

## Failure Classification (IMPORTANT)
When you hit a problem during an application, use the "reportar_falha" tool with the appropriate code.
The system classifies it automatically and decides whether to skip or retry.

### PERMANENT failures (never retry):
- vaga_expirada: The job is no longer available
- captcha: CAPTCHA detected on the page
- sessao_expirada: Session expired, needs re-login
- localizacao_inelegivel: On-site/hybrid job outside Dublin, or remote restricted to a country where the candidate can't work
- ja_aplicou: Candidate already applied (detected by the site, not the database)
- conta_necessaria: Requires signup on a specific platform
- nao_e_vaga: The page is not a job posting
- sso_obrigatorio: Requires SSO login (Google, Microsoft)
- site_bloqueado: The site blocked access
- cloudflare: Anti-bot protection active
- formulario_incompativel: A form you cannot fill
- vaga_interna: Internal employees only
- idioma_incompativel: Requires a language the candidate doesn't have

### RETRYABLE failures (retry, max ${MAX_TENTATIVAS}x):
- timeout: Page took too long to load
- erro_rede: Connection error
- pagina_nao_carregou: Page loaded incompletely
- erro_servidor: 500/502/503 site error
- elemento_nao_encontrado: Button or field disappeared from the page
- erro_upload: Failed to upload resume/file
- erro_mcp: Browser communication error

### How to use:
1. Hit a problem → use reportar_falha with url_vaga + codigo_falha + description
2. If the response says SKIP → move to the next job
3. If the response says RETRY → try the same action again (backoff was already applied)
4. Do NOT try to solve permanent failures — skip and move on

## When Finishing
When you finish all sites or reach the daily limit, write a summary:
- How many applications were made
- At which companies/jobs
- Whether there was any error or block (include the failure codes)
`;
}

// Gemini 3.x exige que TODO functionCall no histórico carregue um
// thought_signature. Tool calls vindos do fallback local (gemma) não têm
// assinatura → o Gemini rejeita o request inteiro (400). Antes de cada chamada
// ao Gemini, convertemos os turnos de tool SEM assinatura em texto puro: o
// contexto é preservado, mas deixam de ser functionCall/functionResponse.
// Turnos nativos do Gemini (com assinatura) passam intactos.
function sanitizarHistoricoParaGemini(history: Content[]): Content[] {
  const out: Content[] = [];
  for (let i = 0; i < history.length; i++) {
    const turn = history[i];
    const parts = turn.parts ?? [];
    const temCallSemAssinatura = parts.some(
      (p) => (p as { functionCall?: unknown }).functionCall &&
             !(p as { thoughtSignature?: unknown }).thoughtSignature,
    );
    if (turn.role === 'model' && temCallSemAssinatura) {
      const txt = parts.map((p) => {
        const part = p as { text?: string; functionCall?: { name?: string; args?: unknown } };
        if (typeof part.text === 'string') return part.text;
        if (part.functionCall) return `[acao: ${part.functionCall.name}(${JSON.stringify(part.functionCall.args ?? {})})]`;
        return '';
      }).filter(Boolean).join('\n');
      out.push({ role: 'model', parts: [{ text: txt || '[acao]' }] });
      // Converte também o functionResponse seguinte (senão fica órfão)
      const next = history[i + 1];
      if (next?.role === 'user' && (next.parts ?? []).some((p) => (p as { functionResponse?: unknown }).functionResponse)) {
        const rtxt = (next.parts ?? []).map((p) => {
          const part = p as { text?: string; functionResponse?: { name?: string; response?: { result?: unknown } } };
          if (part.functionResponse) {
            const r = part.functionResponse.response?.result;
            return `[resultado ${part.functionResponse.name}: ${typeof r === 'string' ? r : JSON.stringify(r)}]`;
          }
          return part.text ?? '';
        }).filter(Boolean).join('\n');
        out.push({ role: 'user', parts: [{ text: rtxt || '[resultado]' }] });
        i++;
      }
      continue;
    }
    out.push(turn);
  }
  return out;
}

export async function executarAgente(
  mcpClient: Client,
  perfil: Perfil,
  sites: SitesConfig,
  config: AgenteConfig,
): Promise<string> {
  const ai = new GoogleGenAI({ apiKey: config.geminiApiKey });
  const executarTool = criarExecutorDeTools(perfil, config.geminiApiKey, config.geminiModel);
  const systemPrompt = buildSystemPrompt(perfil, sites, config);

  const sitesAtivos = sites.sites.filter(s => s.ativo);
  if (sitesAtivos.length === 0) {
    return 'Nenhum site ativo configurado em sites.json. Adicione sites e tente novamente.';
  }

  log('AGENTE', `=== Agent run starting ===`);
  log('AGENTE', `Active sites: ${sitesAtivos.length} → ${sitesAtivos.map(s => s.nome).join(', ')}`);
  log('AGENTE', `Daily limit: ${config.limiteDiario} applications | Minimum score to apply: ${config.scoreMinimo ?? 'n/a'}`);
  log('AGENTE', `Primary model: ${config.geminiModel} | Dry-run: ${config.dryRun ?? false}`);

  // Fallback local (Ollama): só assume enquanto o Gemini está rate-limited.
  const toolsOllama = config.fallbackAtivo ? await montarToolsOllama(mcpClient) : [];
  // Enquanto Date.now() < geminiCooldownUntil, mandamos as chamadas pro Ollama.
  let geminiCooldownUntil = 0;
  if (config.fallbackAtivo) {
    log('AGENTE', `Fallback ENABLED (chain): ${config.fallbackModels.join(' → ')} — used during Gemini cooldowns`);
  }

  // Historico de mensagens para manter contexto entre iteracoes
  const history: Content[] = [];

  // Tentar restaurar estado de uma execucao anterior interrompida
  const estadoRecuperado = carregarRecovery();
  if (estadoRecuperado) {
    log('AGENTE', `Recovering state: ${estadoRecuperado.iteracao} previous iteration(s), last site: ${estadoRecuperado.ultimoSite}`);
  }

  // Mensagem inicial que dispara o agente
  const mensagemInicial = `
Start the application process. Begin with the first site in the list.
Remember: use aguardar between each action, check for duplicates, and vary your answers.
  `.trim();

  history.push({ role: 'user', parts: [{ text: mensagemInicial }] });

  let iteracao = 0;
  let respostaFinal = '';
  let errosConsecutivos = 0;
  let semToolConsecutivas = 0; // respostas seguidas sem tool call (modelo fraco?)

  // ---- Stale-ref guard ----------------------------------------------------
  // The #1 click failure is the model reusing a ref (e123) from an OLD snapshot:
  // refs are regenerated on every snapshot, so once the page changes (navigate,
  // click, scroll), a previously valid ref now points at a DIFFERENT element —
  // hence "clicked AI Engineer but landed on Product Engineer". We track the set
  // of refs from the most recent browser_snapshot and whether the page has
  // mutated since. A ref-based action using an unknown/stale ref is rejected
  // (not sent to the browser) with a message forcing a fresh snapshot first.
  const REF_TOOLS = new Set(['browser_click', 'browser_hover', 'browser_select_option', 'browser_drag', 'browser_type']);
  const MUTATING_TOOLS = new Set(['browser_navigate', 'browser_navigate_back', 'browser_click', 'browser_press_key', 'browser_select_option', 'browser_type', 'browser_fill_form', 'browser_drag', 'browser_tabs']);
  let refsAtuais = new Set<string>(); // refs listed in the latest snapshot
  let snapshotStale = true;           // has the page changed since that snapshot?
  const lerRef = (a: Record<string, unknown>): string => String((a.ref ?? a.target) ?? '');

  // Faz uma chamada ao modelo. Tenta o Gemini (principal). Se o Gemini estiver
  // em cooldown (rate-limit) E o fallback estiver ativo, usa o Ollama local.
  // Numa resposta 429 do Gemini, marca o cooldown e responde a ESTA chamada
  // pelo Ollama. Erros não-429 são propagados para o catch externo tratar.
  async function gerarResposta(): Promise<RespostaModelo> {
    const usarFallbackAgora = config.fallbackAtivo && Date.now() < geminiCooldownUntil;

    if (!usarFallbackAgora) {
      try {
        const response = await ai.models.generateContent({
          model: config.geminiModel,
          contents: sanitizarHistoricoParaGemini(history),
          config: {
            systemInstruction: systemPrompt,
            automaticFunctionCalling: { disable: true },
            tools: [mcpToTool(mcpClient), { functionDeclarations: customToolDeclarations }],
          },
        });
        geminiCooldownUntil = 0; // Gemini respondeu → saudável de novo
        registrarUsoTokens(config.geminiModel, response.usageMetadata, 'agente');
        return {
          content: response.candidates?.[0]?.content,
          functionCalls: (response.functionCalls ?? []).map((fc) => ({
            name: fc.name ?? 'unknown',
            args: (fc.args ?? {}) as Record<string, unknown>,
          })),
          text: response.text ?? '',
          modelo: config.geminiModel,
          usageMetadata: response.usageMetadata,
        };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (config.fallbackAtivo && classificarErroAPI(msg) === 'rate_limit') {
          const delay = extrairRetryDelayMs(msg) ?? 30000;
          geminiCooldownUntil = Date.now() + delay;
          log('WARN', `Gemini rate-limited (429) → switching to fallback chain [${config.fallbackModels.join(' → ')}] for ~${Math.round(delay / 1000)}s. Reason: ${msg.slice(0, 120)}`);
          // segue abaixo para o Ollama
        } else {
          throw e; // rede/fatal, ou rate-limit sem fallback → catch externo
        }
      }
    }

    // Caminho Ollama (fallback) — cadeia: tenta cada modelo em ordem até um
    // responder. Se um modelo de cloud estiver ocupado/indisponível, cai pro
    // próximo. Só lança erro se TODOS falharem.
    let ultimoErro: unknown;
    for (const modelo of config.fallbackModels) {
      try {
        log('AGENTE', `Calling fallback model "${modelo}" (Ollama @ ${config.ollamaUrl}, num_ctx=${config.fallbackNumCtx})...`);
        const t0 = Date.now();
        const resp = await chamarOllamaComTools(history, systemPrompt, toolsOllama, {
          model: modelo,
          baseUrl: config.ollamaUrl,
          numCtx: config.fallbackNumCtx,
        });
        log('AGENTE', `Fallback model "${modelo}" responded in ${Date.now() - t0}ms (${resp.usageMetadata?.totalTokenCount ?? '?'} tokens, ${resp.functionCalls.length} tool call(s))`);
        registrarUsoTokens(modelo, resp.usageMetadata, 'agente-fallback');
        return resp;
      } catch (e) {
        ultimoErro = e;
        log('WARN', `Fallback model "${modelo}" failed (${e instanceof Error ? e.message.slice(0, 120) : e}); trying next in chain...`);
      }
    }
    throw ultimoErro ?? new Error('All fallback models failed');
  }

  while (iteracao < MAX_ITERACOES) {
    iteracao++;
    log('AGENTE', `--- Iteration ${iteracao}/${MAX_ITERACOES} (history: ${history.length} msgs) ---`);

    try {
      const t0 = Date.now();
      const response = await gerarResposta();
      const tokens = response.usageMetadata?.totalTokenCount;
      log('AGENTE', `Model "${response.modelo ?? '?'}" replied in ${Date.now() - t0}ms${tokens != null ? ` | ${tokens} tokens (in: ${response.usageMetadata?.promptTokenCount ?? '?'}, out: ${response.usageMetadata?.candidatesTokenCount ?? '?'})` : ''}`);

      // Reset do contador — iteração bem sucedida
      errosConsecutivos = 0;

      const candidate = response.content;
      if (!candidate) {
        log('WARN', 'Model returned an empty response. Ending run.');
        break;
      }

      // Log the model's own text/reasoning when it includes any (helps debug
      // why a model chose a given action or went off-flow).
      const modelText = (response.text || '').trim();
      if (modelText) {
        log('AGENTE', `Model says: ${modelText.slice(0, 300)}${modelText.length > 300 ? '…' : ''}`);
      }

      // Adiciona resposta do modelo ao historico
      history.push(candidate);

      // Verifica se tem function calls
      const functionCalls = response.functionCalls;

      if (!functionCalls || functionCalls.length === 0) {
        // Resposta sem tool call. NÃO encerra a execução de imediato — modelos
        // fracos (ex: fallback gemma) às vezes respondem só texto no meio do
        // fluxo, e encerrar aqui mataria o run inteiro. Só encerramos se:
        //  - o modelo sinalizar fim explicitamente ("FIM" / "concluí" / limite), OU
        //  - acumular várias respostas vazias seguidas (travou de verdade).
        const textoFinal = (response.text || '').trim();
        const sinalizouFim = /\bFIM\b|conclu[ií]|finaliz|limite di[aá]rio (atingid|alcanç)/i.test(textoFinal);
        semToolConsecutivas++;

        if (sinalizouFim || semToolConsecutivas >= 4) {
          log('AGENTE', `Ending run: ${sinalizouFim ? 'model signalled completion' : `${semToolConsecutivas} consecutive tool-less responses`}.\n${textoFinal}`);
          respostaFinal = textoFinal || 'Run ended.';
          limparRecovery();
          break;
        }

        // Cutuca o modelo a continuar chamando tools
        log('WARN', `Model replied with no tool call (${semToolConsecutivas}/4). Nudging it to continue...`);
        history.push({
          role: 'user',
          parts: [{
            text: 'You did not call any tool. If you have NOT yet reached the daily application limit and there are still relevant jobs to process, CONTINUE the flow by calling the next tool (e.g. browser_snapshot to re-read the page, or browser_navigate to the next search/page). If you are truly done with everything, reply only "FIM".',
          }],
        });
        continue;
      }

      // Houve tool call → reset do contador de respostas vazias
      semToolConsecutivas = 0;

      // Processa function calls em paralelo (Promise.all)
      // Quando o Gemini retorna múltiplas calls numa mesma resposta,
      // ele já considera que são independentes entre si.
      log('AGENTE', `Model requested ${functionCalls.length} tool call(s)${functionCalls.length > 1 ? ' (running in parallel)' : ''}: ${functionCalls.map(fc => fc.name).join(', ')}`);

      // If the model takes a fresh snapshot in THIS same batch, any ref-based
      // call alongside it is allowed (it's reading the new refs) — only enforce
      // the guard when no snapshot accompanies the ref action.
      const batchTemSnapshot = functionCalls.some((fc) => fc.name === 'browser_snapshot');

      const toolResults: Part[] = await Promise.all(
        functionCalls.map(async (fc) => {
          const toolName = fc.name ?? 'unknown';
          const toolArgs = (fc.args ?? {}) as Record<string, unknown>;
          const argsStr = JSON.stringify(toolArgs);
          const isCustom = customToolDeclarations.some(t => t.name === toolName);
          log('TOOL', `→ ${toolName} [${isCustom ? 'custom' : 'playwright-mcp'}] args=${argsStr.substring(0, 200)}${argsStr.length > 200 ? '…' : ''}`);
          const tTool = Date.now();

          let resultado = '';
          let isErro = false;

          // --- Stale-ref guard: reject ref-based actions on outdated refs ---
          const refUsado = lerRef(toolArgs);
          if (REF_TOOLS.has(toolName) && /^e\d+$/.test(refUsado) && !batchTemSnapshot) {
            if (snapshotStale) {
              resultado = `STALE_REF: The page changed since your last browser_snapshot, so ref "${refUsado}" is no longer reliable (refs are regenerated on every snapshot). Call browser_snapshot NOW to get current refs, then use a ref from THAT snapshot. Do NOT reuse old refs.`;
              isErro = true;
            } else if (!refsAtuais.has(refUsado)) {
              const amostra = [...refsAtuais].slice(0, 8).join(', ');
              resultado = `STALE_REF: Ref "${refUsado}" is not present in the current snapshot, so it would click the wrong element. Take a fresh browser_snapshot and pick a ref that actually appears in it. Refs currently available include: ${amostra || '(none — snapshot first)'}.`;
              isErro = true;
            }
          }

          // Verifica se e uma tool customizada ou do MCP
          if (isErro) {
            // guard rejected this call — skip execution entirely
            log('WARN', `✗ ${toolName} BLOCKED by stale-ref guard (ref=${refUsado}); asking model to re-snapshot`);
          } else if (isCustom) {
            resultado = await executarTool(toolName, toolArgs);
          } else {
            // Tool do Playwright MCP — executa via mcpClient
            try {
              const mcpResult = await mcpClient.callTool({
                name: toolName,
                arguments: toolArgs,
              });

              const content = mcpResult.content as Array<{ type: string; text?: string }> | undefined;
              resultado = content
                ?.map((c) => (c.type === 'text' ? c.text : JSON.stringify(c)))
                .join('\n') || 'OK';
            } catch (mcpError) {
              resultado = `MCP_ERROR: ${mcpError instanceof Error ? mcpError.message : String(mcpError)}`;
              isErro = true;
              log('ERRO', `✗ ${toolName} threw: ${resultado}`);
            }
          }

          // Surface tool-reported failures (custom tools return strings starting
          // with ERRO/FALHA/MCP_ERROR) at WARN level so they stand out in logs.
          if (!isErro && /^(ERRO|FALHA|MCP_ERROR|ERROR)/i.test(resultado)) isErro = true;
          const dur = Date.now() - tTool;
          log(isErro ? 'WARN' : 'TOOL', `${isErro ? '✗' : '✓'} ${toolName} done in ${dur}ms → ${resultado.substring(0, 200)}${resultado.length > 200 ? '…' : ''}`);

          return {
            functionResponse: {
              name: fc.name,
              response: { result: resultado },
            },
          } as Part;
        }),
      );

      // Update the stale-ref state AFTER the batch (race-free; done serially in
      // call order). A fresh snapshot refreshes the ref set and clears staleness;
      // any mutating action marks the refs as stale so the next ref-based call
      // is forced to re-snapshot.
      for (let i = 0; i < functionCalls.length; i++) {
        const nome = functionCalls[i].name ?? '';
        const fr = (toolResults[i] as { functionResponse?: { response?: { result?: unknown } } }).functionResponse;
        const res = typeof fr?.response?.result === 'string' ? fr.response.result : '';
        const falhou = /^(ERRO|FALHA|MCP_ERROR|ERROR|STALE_REF)/i.test(res);
        if (nome === 'browser_snapshot' && !falhou) {
          const refs = res.match(/\[ref=(e\d+)\]/g)?.map((m) => m.slice(5, -1)) ?? [];
          refsAtuais = new Set(refs);
          snapshotStale = false;
          log('AGENTE', `Snapshot refs refreshed: ${refsAtuais.size} interactive element(s) now addressable`);
        } else if (MUTATING_TOOLS.has(nome) && !falhou) {
          snapshotStale = true; // page may have changed → old refs no longer trustworthy
        }
      }

      // Envia resultados das tools de volta ao modelo
      history.push({ role: 'user', parts: toolResults });

      // Se a vaga acabou de ser ENVIADA (clique em "Submit application" bem
      // sucedido) e o modelo NÃO registrou a candidatura na mesma rodada,
      // forçamos o registro na próxima — senão a contagem fica errada e o bot
      // pode re-aplicar à mesma vaga (sem dedup).
      const enviouAgora = functionCalls.some((fc) => {
        if (fc.name !== 'browser_click') return false;
        const blob = JSON.stringify(fc.args ?? {}).toLowerCase();
        return blob.includes('submit application') || blob.includes('enviar candidatura');
      });
      const registrouAgora = functionCalls.some((fc) => fc.name === 'registrar_candidatura');
      if (enviouAgora) {
        log('INFO', `🟢 "Submit application" click detected this turn${registrouAgora ? ' (and registrar_candidatura was called)' : ' — forcing registrar_candidatura next turn'}`);
      }
      if (registrouAgora) {
        log('INFO', `📝 Application recorded via registrar_candidatura`);
      }
      if (enviouAgora && !registrouAgora) {
        history.push({
          role: 'user',
          parts: [{
            text: 'You just clicked "Submit application" and the application WAS SENT. BEFORE any other action, call the registrar_candidatura tool NOW with this job\'s REAL data: canonical url (https://www.linkedin.com/jobs/view/<JOB_ID>/), real company, titulo_vaga, and the score that pontuar_vaga returned. Without this the job is not recorded and you may apply again by mistake.',
          }],
        });
      }

      // Sliding window: descarta mensagens antigas se o histórico cresceu demais.
      // Regras da API Gemini sobre a ordem dos turnos:
      //   - um functionCall (turno 'model') deve vir logo após um turno 'user'
      //     ou após um functionResponse;
      //   - um functionResponse (turno 'user') deve vir logo após um functionCall.
      // No nosso loop, o ÚNICO turno 'user' de texto puro é a mensagem inicial
      // (history[0]); todos os outros turnos 'user' são functionResponses. Por
      // isso preservamos sempre history[0] e fazemos a janela recomeçar num
      // turno 'model' (functionCall), que é válido logo após a mensagem inicial.
      if (history.length > MAX_HISTORICO) {
        const head = history[0];
        let body = history.slice(1);
        const excesso = history.length - MAX_HISTORICO;
        body = body.slice(excesso);
        while (body.length > 0 && body[0].role !== 'model') {
          body.shift();
        }
        const removidas = history.length - (1 + body.length);
        history.length = 0;
        history.push(head, ...body);
        log('AGENTE', `Sliding window: dropped ${removidas} old message(s) (history now: ${history.length})`);
      }

      // Compactação de tokens (CRÍTICO para o free tier de 250k tokens/min):
      // cada snapshot do navegador (~40k tokens) e screenshots (base64) ficam no
      // histórico e são REENVIADOS a cada chamada. O agente só precisa do
      // conteúdo das ÚLTIMAS mensagens para agir; resultados de tool antigos já
      // foram "consumidos". Então truncamos resultados grandes de tool em todas
      // as mensagens, exceto as PRESERVAR_INTEGRO mais recentes.
      const PRESERVAR_INTEGRO = 4;
      const LIMITE_RESULT = 2000; // chars (~500 tokens)
      for (let i = 0; i < history.length - PRESERVAR_INTEGRO; i++) {
        const parts = history[i]?.parts;
        if (!parts) continue;
        for (const part of parts) {
          const fr = (part as { functionResponse?: { response?: { result?: unknown } } }).functionResponse;
          const result = fr?.response?.result;
          if (typeof result === 'string' && result.length > LIMITE_RESULT) {
            fr!.response!.result =
              result.slice(0, 200) +
              `\n…[resultado antigo truncado p/ economizar tokens — ${result.length} chars originais]`;
          }
        }
      }

      // Salva estado para recovery a cada 5 iteracoes
      if (iteracao % 5 === 0) {
        salvarRecovery(iteracao, sitesAtivos.map(s => s.nome));
        log('AGENTE', `Recovery checkpoint saved at iteration ${iteracao}`);
      }

    } catch (error) {
      const mensagemErro = error instanceof Error ? error.message : String(error);
      log('ERRO', `Error in iteration ${iteracao}: ${mensagemErro}`);

      const tipoErro = classificarErroAPI(mensagemErro);

      if (tipoErro === 'rate_limit') {
        // Quota (não é falha do agente): honra o retryDelay exato que a API
        // devolve; senão cai no backoff exponencial. NÃO incrementa
        // errosConsecutivos nem consome iteração — apenas espera a quota
        // voltar e tenta de novo. Isso permite "esperar entre rate limits"
        // sem escalar o backoff nem esgotar MAX_ITERACOES.
        const sugerido = extrairRetryDelayMs(mensagemErro);
        const backoff = sugerido ?? calcularBackoffRateLimit(1);
        log('WARN', `Rate limit (quota) with no fallback available. Waiting ${Math.round(backoff / 1000)}s (${sugerido ? 'API retryDelay' : 'default backoff'}) before retrying...`);
        await new Promise(resolve => setTimeout(resolve, backoff));
        iteracao--; // não conta a espera de quota como iteração de trabalho
        continue;
      }

      if (tipoErro === 'rede') {
        errosConsecutivos++;
        if (errosConsecutivos <= MAX_TENTATIVAS) {
          const backoff = 5000 * Math.pow(2, errosConsecutivos - 1);
          log('WARN', `Network error (${errosConsecutivos}/${MAX_TENTATIVAS}). Retrying in ${Math.round(backoff / 1000)}s...`);
          await new Promise(resolve => setTimeout(resolve, backoff));
          continue;
        }
        log('ERRO', `Persistent network error after ${errosConsecutivos} attempts. Ending run.`);
      }

      // Erro fatal ou tentativas esgotadas
      salvarRecovery(iteracao, sitesAtivos.map(s => s.nome));
      respostaFinal = `Error during run: ${mensagemErro}`;
      break;
    }
  }

  if (iteracao >= MAX_ITERACOES) {
    respostaFinal = `Agente atingiu o limite maximo de ${MAX_ITERACOES} iteracoes.`;
  }

  limparRecovery();
  log('AGENTE', `=== Agent run finished after ${iteracao} iteration(s) ===`);
  return respostaFinal;
}

// ========== RECOVERY (persistencia de estado) ==========

interface RecoveryState {
  iteracao: number;
  ultimoSite: string;
  timestamp: string;
}

function salvarRecovery(iteracao: number, sites: string[]): void {
  try {
    const estado: RecoveryState = {
      iteracao,
      ultimoSite: sites[sites.length - 1] || '',
      timestamp: new Date().toISOString(),
    };
    writeFileSync(RECOVERY_PATH, JSON.stringify(estado, null, 2));
  } catch {
    // Silencioso
  }
}

function carregarRecovery(): RecoveryState | null {
  try {
    if (!existsSync(RECOVERY_PATH)) return null;
    const conteudo = readFileSync(RECOVERY_PATH, 'utf-8');
    return JSON.parse(conteudo) as RecoveryState;
  } catch {
    return null;
  }
}

function limparRecovery(): void {
  try {
    if (existsSync(RECOVERY_PATH)) {
      writeFileSync(RECOVERY_PATH, '');
    }
  } catch {
    // Silencioso
  }
}
