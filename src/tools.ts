import { Type, type FunctionDeclaration } from '@google/genai';
import { readFileSync, mkdirSync, existsSync, writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  verificarJaAplicou,
  registrarCandidatura,
  contarCandidaturasHoje,
  listarCandidaturas,
  registrarVagaVista,
  verificarVagaJaVista,
  atualizarScreenshot,
  buscarRespostaCache,
  buscarCandidatasCache,
  salvarRespostaCache,
  sanitizarPergunta,
  verificarRecrutadorJaContatado,
  registrarMensagemRecrutador,
  contarMensagensHoje,
} from './database.js';
import { log } from './logger.js';
import { notificarCandidatura, solicitarResolucaoCaptcha } from './notificacoes.js';
import { gerarCurriculoTailored } from './curriculo-tailored.js';
import { gerarCoverLetter } from './cover-letter.js';
import { gerarMensagemRecrutador } from './mensagem-recrutador.js';
import {
  ehFalhaPermanente,
  ehFalhaRetriavel,
  calcularBackoff,
  FALHAS_PERMANENTES,
  FALHAS_RETRIAVEIS,
  MAX_TENTATIVAS,
} from './erros.js';
import type { Perfil, RespostasPredefinidas } from './types.js';

// Configuração do Gemini passada pelo index.ts na criação do executor
let _geminiApiKey = '';
let _geminiModel = '';

// Mapa de tentativas por URL para controle de retry (adaptado do ApplyPilot: attempts tracking)
const tentativasPorUrl = new Map<string, number>();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

interface CurriculoEntry {
  id: string;
  arquivo: string;
  foco: string;
  usar_quando: string;
}

interface CurriculosConfig {
  fallback: CurriculoEntry;
  curriculos: CurriculoEntry[];
}

function carregarCurriculos(): CurriculosConfig {
  const caminho = path.resolve(__dirname, '..', 'config', 'curriculos.json');
  return JSON.parse(readFileSync(caminho, 'utf-8'));
}

// ========== DEFINICAO DAS TOOLS ==========

export const customToolDeclarations: FunctionDeclaration[] = [
  {
    name: 'obter_perfil_candidato',
    description:
      'Returns all of the candidate\'s personal and professional data to fill out forms and generate personalized responses.',
    parameters: {
      type: Type.OBJECT,
      properties: {},
    },
  },
  {
    name: 'verificar_ja_aplicou',
    description:
      'Checks in the database whether the candidate has already applied to a specific job by its URL. Returns true or false.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        url: {
          type: Type.STRING,
          description: 'Full URL of the job to check',
        },
      },
      required: ['url'],
    },
  },
  {
    name: 'registrar_candidatura',
    description:
      'Records in the database that an application was submitted successfully. Call this AFTER filling out and submitting the form.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        plataforma: {
          type: Type.STRING,
          description: 'Name of the platform (e.g. Gupy, LinkedIn, Vagas.com)',
        },
        titulo_vaga: {
          type: Type.STRING,
          description: 'Job title',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name',
        },
        url: {
          type: Type.STRING,
          description: 'Job URL',
        },
        mensagem_enviada: {
          type: Type.BOOLEAN,
          description: 'Whether a personalized message was sent to the recruiter',
        },
      },
      required: ['plataforma', 'titulo_vaga', 'empresa', 'url'],
    },
  },
  {
    name: 'contar_candidaturas_hoje',
    description:
      'Returns how many applications have already been submitted today. Use this to check whether the daily limit has been reached.',
    parameters: {
      type: Type.OBJECT,
      properties: {},
    },
  },
  {
    name: 'listar_candidaturas_recentes',
    description:
      'Lists the most recent applications submitted for reference and to avoid duplicates.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        limite: {
          type: Type.NUMBER,
          description: 'Number of applications to return (default: 20)',
        },
      },
    },
  },
  {
    name: 'pontuar_vaga',
    description:
      'Evaluates how well a job matches the candidate\'s profile (score from 1 to 10). ALWAYS use this BEFORE deciding whether to apply. If the score is below the configured minimum, SKIP the job.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        titulo_vaga: {
          type: Type.STRING,
          description: 'Job title',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name',
        },
        tecnologias_pedidas: {
          type: Type.STRING,
          description: 'List of technologies/requirements the job asks for',
        },
        senioridade: {
          type: Type.STRING,
          description: 'Seniority level requested (junior, mid-level, senior, etc.)',
        },
        modelo_trabalho: {
          type: Type.STRING,
          description: 'Work model (remote, hybrid, on-site)',
        },
        localizacao: {
          type: Type.STRING,
          description: 'City/state of the job',
        },
      },
      required: ['titulo_vaga', 'tecnologias_pedidas'],
    },
  },
  {
    name: 'escolher_curriculo',
    description:
      'Chooses the most suitable resume for the job based on its description. Returns the path to the correct PDF for upload. ALWAYS use this tool BEFORE uploading a resume.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        descricao_vaga: {
          type: Type.STRING,
          description: 'Summary of the job description (technologies requested, type of role, field of work)',
        },
      },
      required: ['descricao_vaga'],
    },
  },
  {
    name: 'aguardar',
    description:
      'Waits a random amount of time between actions to simulate human behavior. ALWAYS use this between navigation actions.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        min_ms: {
          type: Type.NUMBER,
          description: 'Minimum time in milliseconds (default: 2000)',
        },
        max_ms: {
          type: Type.NUMBER,
          description: 'Maximum time in milliseconds (default: 5000)',
        },
      },
    },
  },
  {
    name: 'salvar_screenshot',
    description:
      'Saves the current page screenshot as proof of the application. Use this AFTER submitting (or simulating in dry-run) the application. Pass the base64 data of the screenshot obtained via browser_take_screenshot.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        url_vaga: {
          type: Type.STRING,
          description: 'Job URL to associate the screenshot with',
        },
        screenshot_base64: {
          type: Type.STRING,
          description: 'Base64 data of the screenshot (obtained via browser_take_screenshot)',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name (used to name the file)',
        },
      },
      required: ['url_vaga', 'screenshot_base64', 'empresa'],
    },
  },
  {
    name: 'verificar_vaga_ja_vista',
    description:
      'Checks whether a job has already been seen/analyzed before (even if it was not applied to). Avoids wasting time re-analyzing jobs that were already discarded. Use this BEFORE analyzing a job in detail.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        url: {
          type: Type.STRING,
          description: 'Job URL to check',
        },
      },
      required: ['url'],
    },
  },
  {
    name: 'registrar_vaga_vista',
    description:
      'Records that a job was seen/analyzed. Use this for jobs that were SKIPPED (low score, wrong location, etc.) so they are not re-analyzed in the future.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        url: {
          type: Type.STRING,
          description: 'Job URL',
        },
        titulo_vaga: {
          type: Type.STRING,
          description: 'Job title',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name',
        },
        plataforma: {
          type: Type.STRING,
          description: 'Platform (Gupy, Vagas.com, etc.)',
        },
        score: {
          type: Type.NUMBER,
          description: 'Calculated score of the job',
        },
        motivo_pulo: {
          type: Type.STRING,
          description: 'Reason the job was skipped',
        },
      },
      required: ['url'],
    },
  },
  {
    name: 'obter_respostas_predefinidas',
    description:
      'Returns the candidate\'s predefined answers to common form questions (salary expectations, availability, strengths, etc.). Use these as a BASE to vary the answers.',
    parameters: {
      type: Type.OBJECT,
      properties: {},
    },
  },
  {
    name: 'gerar_curriculo_tailored',
    description:
      'Generates a personalized PDF resume for the specific job. The resume is rewritten by AI to highlight the skills relevant to THIS job, keeping ONLY the candidate\'s real data. Use this BEFORE uploading the resume. If it fails, fall back to escolher_curriculo.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        descricao_vaga: {
          type: Type.STRING,
          description: 'FULL job description (copy as many details as possible: requirements, responsibilities, technologies, seniority)',
        },
        titulo_vaga: {
          type: Type.STRING,
          description: 'Job title (e.g. Backend Java Developer)',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name',
        },
      },
      required: ['descricao_vaga'],
    },
  },
  {
    name: 'gerar_cover_letter',
    description:
      'Generates a personalized cover letter for the job. Returns text ready to paste into the form field. Use this when the form asks for a "cover letter", "carta de apresentacao", "why do you want to work here" (long field), or "introduce yourself".',
    parameters: {
      type: Type.OBJECT,
      properties: {
        descricao_vaga: {
          type: Type.STRING,
          description: 'Job description (requirements, responsibilities)',
        },
        titulo_vaga: {
          type: Type.STRING,
          description: 'Job title',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name',
        },
      },
      required: ['descricao_vaga', 'empresa', 'titulo_vaga'],
    },
  },
  {
    name: 'buscar_resposta_cache',
    description:
      'Checks the cache to see if this form question has already been answered before. Use this BEFORE generating a new answer. If it returns a cache hit, use the cached answer (you may slightly vary the wording). Saves tokens and ensures consistency.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        pergunta: {
          type: Type.STRING,
          description: 'Text of the form field question/label',
        },
        tipo_campo: {
          type: Type.STRING,
          description: 'Field type: textbox, numeric, dropdown, radio, date, textarea',
        },
      },
      required: ['pergunta', 'tipo_campo'],
    },
  },
  {
    name: 'salvar_resposta_cache',
    description:
      'Saves an answer to the cache to reuse in future forms. Use this AFTER filling a field with a generated answer. Do NOT save: cover letters, answers that mention the company name, or specific date fields.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        pergunta: {
          type: Type.STRING,
          description: 'Text of the field question/label',
        },
        tipo_campo: {
          type: Type.STRING,
          description: 'Field type: textbox, numeric, dropdown, radio, date, textarea',
        },
        resposta: {
          type: Type.STRING,
          description: 'Answer that was used in the field',
        },
        empresa_atual: {
          type: Type.STRING,
          description: 'Company name of the current job (used to validate whether the answer is generic enough to cache)',
        },
      },
      required: ['pergunta', 'tipo_campo', 'resposta'],
    },
  },
  {
    name: 'reportar_falha',
    description:
      'Reports a failure encountered during the application process. Automatically classifies it as PERMANENT (never retry) or RETRIABLE (try again). Use this when you encounter errors such as: expired job, CAPTCHA, timeout, network error, incompatible form, etc. Permanent codes: vaga_expirada, captcha, sessao_expirada, localizacao_inelegivel, ja_aplicou, conta_necessaria, nao_e_vaga, sso_obrigatorio, site_bloqueado, cloudflare, formulario_incompativel, vaga_interna, idioma_incompativel. Retriable codes: timeout, erro_rede, pagina_nao_carregou, erro_servidor, elemento_nao_encontrado, erro_upload, erro_mcp.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        url_vaga: {
          type: Type.STRING,
          description: 'URL of the job where the failure occurred',
        },
        codigo_falha: {
          type: Type.STRING,
          description: 'Failure code (e.g. vaga_expirada, captcha, timeout, erro_rede)',
        },
        descricao: {
          type: Type.STRING,
          description: 'Free-form description of what happened',
        },
        titulo_vaga: {
          type: Type.STRING,
          description: 'Job title (if available)',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name (if available)',
        },
        plataforma: {
          type: Type.STRING,
          description: 'Platform (Gupy, Vagas.com, etc.)',
        },
      },
      required: ['url_vaga', 'codigo_falha', 'descricao'],
    },
  },
  {
    name: 'resolver_captcha_telegram',
    description:
      'Sends a screenshot of a CAPTCHA to Telegram and waits for the human user to solve it. Returns the solution typed by the user. Use this when you encounter a CAPTCHA that blocks the application from progressing. REQUIRES: Telegram configured (.env). Timeout: 5 minutes.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        screenshot_base64: {
          type: Type.STRING,
          description: 'Screenshot of the CAPTCHA in base64 (obtained via browser_take_screenshot)',
        },
        url_vaga: {
          type: Type.STRING,
          description: 'URL of the page where the CAPTCHA appeared',
        },
      },
      required: ['screenshot_base64', 'url_vaga'],
    },
  },
  {
    name: 'gerar_mensagem_recrutador',
    description:
      'Generates a personalized message to send to the job\'s recruiter/hiring manager via LinkedIn. The message is at most 280 characters (connection note). Use this ONLY when: (1) the job has a high score (>= 8), (2) you have identified the recruiter on the job page, and (3) the recruiter has NOT been contacted before. The message uses the candidate\'s REAL data and highlights overlaps with the job.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        nome_recrutador: {
          type: Type.STRING,
          description: 'Name of the recruiter/hiring manager (found on the job page or LinkedIn profile)',
        },
        cargo_recrutador: {
          type: Type.STRING,
          description: 'Job title of the recruiter (Recruiter, HR Manager, Tech Lead, etc.)',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name',
        },
        titulo_vaga: {
          type: Type.STRING,
          description: 'Job title',
        },
        descricao_vaga: {
          type: Type.STRING,
          description: 'Job description (requirements, responsibilities)',
        },
      },
      required: ['nome_recrutador', 'empresa', 'titulo_vaga', 'descricao_vaga'],
    },
  },
  {
    name: 'verificar_recrutador_ja_contatado',
    description:
      'Checks whether a recruiter has already been contacted before (by their LinkedIn profile URL). Use this BEFORE generating a message to avoid sending a duplicate message.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        url_perfil: {
          type: Type.STRING,
          description: 'LinkedIn profile URL of the recruiter',
        },
      },
      required: ['url_perfil'],
    },
  },
  {
    name: 'registrar_mensagem_recrutador',
    description:
      'Records in the database that a message was sent to a recruiter. Use this AFTER successfully sending the connection invite on LinkedIn.',
    parameters: {
      type: Type.OBJECT,
      properties: {
        nome_recrutador: {
          type: Type.STRING,
          description: 'Name of the recruiter',
        },
        cargo_recrutador: {
          type: Type.STRING,
          description: 'Job title of the recruiter',
        },
        empresa: {
          type: Type.STRING,
          description: 'Company name',
        },
        url_perfil: {
          type: Type.STRING,
          description: 'LinkedIn profile URL of the recruiter',
        },
        url_vaga: {
          type: Type.STRING,
          description: 'URL of the associated job',
        },
        titulo_vaga: {
          type: Type.STRING,
          description: 'Job title',
        },
        mensagem: {
          type: Type.STRING,
          description: 'Text of the message that was sent',
        },
        score_vaga: {
          type: Type.NUMBER,
          description: 'Job score (1-10)',
        },
      },
      required: ['nome_recrutador', 'empresa', 'url_perfil', 'mensagem'],
    },
  },
];

// ========== EXECUTOR DAS TOOLS ==========

export function criarExecutorDeTools(perfil: Perfil, geminiApiKey?: string, geminiModel?: string) {
  if (geminiApiKey) _geminiApiKey = geminiApiKey;
  if (geminiModel) _geminiModel = geminiModel;
  return async function executarTool(name: string, args: Record<string, unknown>): Promise<string> {
    switch (name) {
      case 'obter_perfil_candidato': {
        return JSON.stringify(perfil, null, 2);
      }

      case 'verificar_ja_aplicou': {
        const url = args.url as string;
        const jaAplicou = verificarJaAplicou(url);
        return jaAplicou
          ? 'JA_APLICOU: O candidato ja se candidatou a esta vaga. Pule para a proxima.'
          : 'NOVA_VAGA: O candidato ainda nao se candidatou. Pode prosseguir.';
      }

      case 'registrar_candidatura': {
        const score = (args.score as number) || 0;
        const empresa = args.empresa as string;
        const tituloVaga = args.titulo_vaga as string;
        const isDryRun = !!args.dry_run;
        const sucesso = registrarCandidatura({
          plataforma: args.plataforma as string,
          titulo_vaga: tituloVaga,
          empresa,
          url: args.url as string,
          mensagem_enviada: args.mensagem_enviada ? 1 : 0,
          status: isDryRun ? 'dry-run' : 'aplicado',
          score,
        });
        if (sucesso) {
          log('AGENTE', `Application recorded: ${tituloVaga} — ${empresa} (score: ${score})`);
          notificarCandidatura(empresa, tituloVaga, score, isDryRun).catch(() => {});
        }
        return sucesso
          ? 'REGISTRADO: Candidatura salva no banco de dados com sucesso.'
          : 'ERRO: Falha ao registrar candidatura (possivelmente duplicada).';
      }

      case 'contar_candidaturas_hoje': {
        const total = contarCandidaturasHoje();
        return `Total de candidaturas hoje: ${total}`;
      }

      case 'listar_candidaturas_recentes': {
        const limite = (args.limite as number) || 20;
        const candidaturas = listarCandidaturas(limite);
        return JSON.stringify(candidaturas, null, 2);
      }

      case 'pontuar_vaga': {
        const tecsPedidas = (args.tecnologias_pedidas as string).toLowerCase();
        const senioridade = ((args.senioridade as string) || '').toLowerCase();
        const localizacao = ((args.localizacao as string) || '').toLowerCase();
        const modelo = ((args.modelo_trabalho as string) || '').toLowerCase();

        let score = 5; // Base

        // Match de tecnologias (+1 por cada tech que o candidato tem)
        const minhasTechs = perfil.stack_principal.map(s => s.toLowerCase());
        for (const tech of minhasTechs) {
          if (tecsPedidas.includes(tech)) score += 1;
        }

        // Penalidades
        if (senioridade.includes('senior') || senioridade.includes('sênior')) score -= 2;
        if (senioridade.includes('pleno')) score += 1;
        if (senioridade.includes('junior') || senioridade.includes('júnior')) score += 1;

        // Localizacao
        if (localizacao.includes('uberlandia') || localizacao.includes('uberlândia')) {
          score += 1;
        } else if (modelo.includes('presencial') || modelo.includes('hibrido')) {
          score -= 3; // Fora de Uberlandia e nao remoto = penalidade forte
        }
        if (modelo.includes('remoto')) score += 1;

        // Clamp entre 1-10
        score = Math.max(1, Math.min(10, score));

        return JSON.stringify({
          score,
          veredicto: score >= 6 ? 'APLICAR' : 'PULAR',
          motivo: score >= 6
            ? `Score ${score}/10: boa compatibilidade com o perfil.`
            : `Score ${score}/10: baixa compatibilidade. Pule para a proxima vaga.`,
        });
      }

      case 'escolher_curriculo': {
        const descricao = (args.descricao_vaga as string).toLowerCase();
        const config = carregarCurriculos();

        // Mapeamento de palavras-chave para cada curriculo
        const mapeamento: Record<string, string[]> = {
          'backend-java': ['backend', 'back-end', 'back end', 'java', 'api rest', 'apis rest', 'microsservico', 'microservico', 'servidor'],
          'java-enterprise': ['corporativo', 'camunda', 'automacao de processos', 'integracao de sistemas', 'consultoria', 'gestao'],
          'full-stack-backend': ['full stack', 'fullstack', 'full-stack', 'backend', 'java', 'react'],
          'full-stack': ['full stack', 'fullstack', 'full-stack', 'ponta a ponta', 'end to end'],
          'mobile-react-native': ['mobile', 'react native', 'expo', 'ios', 'android', 'aplicativo', 'app mobile'],
        };

        let melhorMatch = '';
        let maiorScore = 0;

        for (const [id, keywords] of Object.entries(mapeamento)) {
          const score = keywords.reduce((acc, kw) => acc + (descricao.includes(kw) ? 1 : 0), 0);
          if (score > maiorScore) {
            maiorScore = score;
            melhorMatch = id;
          }
        }

        // Fallback: se nenhum score ou score muito baixo, usa o curriculo original
        if (maiorScore === 0) {
          const fallbackPath = path.resolve(__dirname, '..', config.fallback.arquivo);
          return JSON.stringify({
            curriculo_escolhido: 'original',
            foco: config.fallback.foco,
            caminho: fallbackPath,
            motivo: 'Nenhum curriculo especifico se encaixou. Usando curriculo original como fallback.',
          });
        }

        const curriculo = config.curriculos.find(c => c.id === melhorMatch);
        if (!curriculo) {
          const fallbackPath = path.resolve(__dirname, '..', config.fallback.arquivo);
          return JSON.stringify({
            curriculo_escolhido: 'original',
            foco: config.fallback.foco,
            caminho: fallbackPath,
            motivo: 'Curriculo especifico nao encontrado. Usando curriculo original como fallback.',
          });
        }

        const caminhoAbsoluto = path.resolve(__dirname, '..', curriculo.arquivo);
        return JSON.stringify({
          curriculo_escolhido: curriculo.id,
          foco: curriculo.foco,
          caminho: caminhoAbsoluto,
          motivo: `Escolhido "${curriculo.foco}" com score ${maiorScore} para a vaga descrita.`,
        });
      }

      case 'aguardar': {
        // Coerção numérica OBRIGATÓRIA: modelos (esp. cloud) às vezes mandam
        // "2000" (string). Sem Number(), "1000" + "2000" vira concatenação e o
        // setTimeout dorme por horas. Também limitamos o teto a 10s por segurança.
        const TETO = 10000;
        let min = Number(args.min_ms);
        let max = Number(args.max_ms);
        if (!Number.isFinite(min)) min = 2000;
        if (!Number.isFinite(max)) max = 5000;
        min = Math.max(0, Math.min(min, TETO));
        max = Math.max(min, Math.min(max, TETO));
        const tempo = Math.floor(Math.random() * (max - min + 1)) + min;
        await new Promise((resolve) => setTimeout(resolve, tempo));
        return `Aguardou ${tempo}ms com sucesso.`;
      }

      case 'salvar_screenshot': {
        const screenshotsDir = path.resolve(__dirname, '..', 'screenshots');
        if (!existsSync(screenshotsDir)) {
          mkdirSync(screenshotsDir, { recursive: true });
        }

        const empresaNome = (args.empresa as string).replace(/[^a-zA-Z0-9]/g, '_').substring(0, 30);
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
        const nomeArquivo = `${timestamp}_${empresaNome}.png`;
        const caminhoCompleto = path.join(screenshotsDir, nomeArquivo);

        try {
          const base64Data = args.screenshot_base64 as string;
          const buffer = Buffer.from(base64Data, 'base64');
          writeFileSync(caminhoCompleto, buffer);
          atualizarScreenshot(args.url_vaga as string, caminhoCompleto);
          log('TOOL', `Screenshot saved: ${nomeArquivo}`);
          return `Screenshot salvo com sucesso em: ${caminhoCompleto}`;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          log('ERRO', `Failed to save screenshot: ${msg}`);
          return `ERRO ao salvar screenshot: ${msg}`;
        }
      }

      case 'verificar_vaga_ja_vista': {
        const url = args.url as string;
        const jaVista = verificarVagaJaVista(url);
        return jaVista
          ? 'JA_VISTA: Esta vaga ja foi analisada anteriormente. Pule para a proxima.'
          : 'NOVA: Esta vaga ainda nao foi vista. Pode analisar.';
      }

      case 'registrar_vaga_vista': {
        const sucesso = registrarVagaVista({
          url: args.url as string,
          titulo_vaga: args.titulo_vaga as string | undefined,
          empresa: args.empresa as string | undefined,
          plataforma: args.plataforma as string | undefined,
          score: args.score as number | undefined,
          motivo_pulo: args.motivo_pulo as string | undefined,
        });
        return sucesso
          ? 'REGISTRADO: Vaga marcada como vista.'
          : 'ERRO: Falha ao registrar vaga vista.';
      }

      case 'obter_respostas_predefinidas': {
        try {
          const caminho = path.resolve(__dirname, '..', 'config', 'respostas.json');
          const conteudo = readFileSync(caminho, 'utf-8');
          const respostas = JSON.parse(conteudo) as RespostasPredefinidas;
          // Remove campos internos
          const { _comentario, _todo_preencher, ...respostasUteis } = respostas as Record<string, unknown>;
          return JSON.stringify(respostasUteis, null, 2);
        } catch {
          return 'ERRO: Arquivo config/respostas.json nao encontrado.';
        }
      }

      case 'buscar_resposta_cache': {
        const pergunta = args.pergunta as string;
        const tipoCampo = args.tipo_campo as string;

        // 1. Busca exact match / substring match
        const cacheHit = buscarRespostaCache(pergunta, tipoCampo);
        if (cacheHit) {
          log('TOOL', `Cache HIT (exact): "${pergunta.substring(0, 50)}..." → "${cacheHit.resposta.substring(0, 50)}..." (used ${cacheHit.vezes_usada}x)`);
          return JSON.stringify({
            encontrado: true,
            metodo: 'exact',
            resposta: cacheHit.resposta,
            vezes_usada: cacheHit.vezes_usada,
            instrucao: 'Use esta resposta. Pode variar levemente a forma de escrever mas mantenha o conteudo.',
          });
        }

        // 2. Busca candidatas por palavras-chave para matching semântico
        const candidatas = buscarCandidatasCache(pergunta);
        if (candidatas.length > 0) {
          log('TOOL', `Cache: ${candidatas.length} candidate(s) found for semantic matching`);
          return JSON.stringify({
            encontrado: false,
            candidatas: candidatas.map(c => ({
              pergunta_original: c.pergunta_sanitizada,
              resposta: c.resposta,
              tipo: c.tipo_campo,
            })),
            instrucao: 'Nenhum match exato. Verifique se alguma candidata e semanticamente equivalente. Se sim, reutilize a resposta (variando a forma). Se nao, gere uma resposta nova e salve no cache.',
          });
        }

        log('TOOL', `Cache MISS: "${pergunta.substring(0, 50)}..."`);
        return JSON.stringify({
          encontrado: false,
          candidatas: [],
          instrucao: 'Nenhuma resposta no cache. Gere uma resposta nova e salve no cache apos preencher o campo.',
        });
      }

      case 'salvar_resposta_cache': {
        const pergunta = args.pergunta as string;
        const tipoCampo = args.tipo_campo as string;
        const resposta = args.resposta as string;
        const empresaAtual = (args.empresa_atual as string) || '';

        // Regra do AIHawk: não cachear respostas que mencionam o nome da empresa
        if (empresaAtual && resposta.toLowerCase().includes(empresaAtual.toLowerCase())) {
          log('TOOL', `Cache SKIP: response mentions "${empresaAtual}" (too specific to cache)`);
          return 'NAO_CACHEADO: Resposta menciona o nome da empresa e e especifica demais para reutilizar em outras vagas.';
        }

        const sucesso = salvarRespostaCache(pergunta, tipoCampo, resposta);
        if (sucesso) {
          log('TOOL', `Cache SAVE: "${sanitizarPergunta(pergunta).substring(0, 50)}..." → "${resposta.substring(0, 50)}..."`);
          return 'CACHEADO: Resposta salva no cache para reutilizacao futura.';
        }
        return 'JA_EXISTE: Essa pergunta ja existe no cache.';
      }

      case 'reportar_falha': {
        const urlVaga = args.url_vaga as string;
        const codigoFalha = args.codigo_falha as string;
        const descricaoFalha = args.descricao as string;

        if (ehFalhaPermanente(codigoFalha)) {
          // Falha permanente: registra como vista e nunca mais tenta
          // (ApplyPilot usa attempts=99 como sentinela; nós registramos em vagas_vistas)
          registrarVagaVista({
            url: urlVaga,
            titulo_vaga: (args.titulo_vaga as string) || undefined,
            empresa: (args.empresa as string) || undefined,
            plataforma: (args.plataforma as string) || undefined,
            motivo_pulo: `PERMANENTE:${codigoFalha} — ${descricaoFalha}`,
          });
          tentativasPorUrl.delete(urlVaga);
          log('FALHA', `PERMANENT [${codigoFalha}]: ${descricaoFalha} — ${urlVaga}`);

          return JSON.stringify({
            tipo: 'PERMANENTE',
            acao: 'PULAR',
            codigo: codigoFalha,
            mensagem: `Falha permanente (${codigoFalha}). Vaga registrada como vista — nunca sera retentada. Passe para a proxima vaga.`,
          });
        }

        if (ehFalhaRetriavel(codigoFalha)) {
          const tentativasAtuais = (tentativasPorUrl.get(urlVaga) || 0) + 1;
          tentativasPorUrl.set(urlVaga, tentativasAtuais);

          if (tentativasAtuais >= MAX_TENTATIVAS) {
            // Esgotou tentativas — trata como permanente
            registrarVagaVista({
              url: urlVaga,
              titulo_vaga: (args.titulo_vaga as string) || undefined,
              empresa: (args.empresa as string) || undefined,
              plataforma: (args.plataforma as string) || undefined,
              motivo_pulo: `ESGOTADO:${codigoFalha} — ${tentativasAtuais} tentativas — ${descricaoFalha}`,
            });
            tentativasPorUrl.delete(urlVaga);
            log('FALHA', `EXHAUSTED [${codigoFalha}]: ${tentativasAtuais}/${MAX_TENTATIVAS} attempts — ${urlVaga}`);

            return JSON.stringify({
              tipo: 'ESGOTADO',
              acao: 'PULAR',
              codigo: codigoFalha,
              tentativas: tentativasAtuais,
              mensagem: `Maximo de ${MAX_TENTATIVAS} tentativas atingido para esta vaga. Passe para a proxima.`,
            });
          }

          const backoffMs = calcularBackoff(tentativasAtuais);
          log('FALHA', `RETRIABLE [${codigoFalha}]: attempt ${tentativasAtuais}/${MAX_TENTATIVAS}, backoff ${backoffMs}ms — ${urlVaga}`);

          // Aguarda backoff antes de liberar o agente para retentar
          await new Promise(resolve => setTimeout(resolve, backoffMs));

          return JSON.stringify({
            tipo: 'RETRIAVEL',
            acao: 'RETENTAR',
            codigo: codigoFalha,
            tentativa_atual: tentativasAtuais,
            max_tentativas: MAX_TENTATIVAS,
            backoff_aplicado_ms: backoffMs,
            mensagem: `Falha retriavel (${codigoFalha}). Tentativa ${tentativasAtuais}/${MAX_TENTATIVAS}. Backoff de ${Math.round(backoffMs / 1000)}s ja aplicado. Tente novamente agora.`,
          });
        }

        // Código desconhecido — trata como permanente por segurança
        log('FALHA', `UNKNOWN [${codigoFalha}]: ${descricaoFalha} — ${urlVaga}`);
        registrarVagaVista({
          url: urlVaga,
          motivo_pulo: `DESCONHECIDO:${codigoFalha} — ${descricaoFalha}`,
        });

        return JSON.stringify({
          tipo: 'DESCONHECIDO',
          acao: 'PULAR',
          codigo: codigoFalha,
          mensagem: `Codigo de falha desconhecido (${codigoFalha}). Pule esta vaga por seguranca.`,
        });
      }

      case 'gerar_mensagem_recrutador': {
        const nomeRecrutador = args.nome_recrutador as string;
        const cargoRecrutador = (args.cargo_recrutador as string) || '';
        const empresa = args.empresa as string;
        const tituloVaga = args.titulo_vaga as string;
        const descricaoVaga = args.descricao_vaga as string;

        // Verifica limite diário de mensagens (max 5 por dia)
        const mensagensHoje = contarMensagensHoje();
        if (mensagensHoje >= 5) {
          return JSON.stringify({
            sucesso: false,
            motivo: 'LIMITE_DIARIO',
            mensagem: `Limite diario de mensagens a recrutadores atingido (${mensagensHoje}/5). Nao envie mais mensagens hoje.`,
          });
        }

        if (!_geminiApiKey || !_geminiModel) {
          return 'ERRO: Configuracao do Gemini nao disponivel para gerar mensagem.';
        }

        try {
          const resultado = await gerarMensagemRecrutador(
            _geminiApiKey,
            _geminiModel,
            perfil,
            nomeRecrutador,
            cargoRecrutador,
            empresa,
            tituloVaga,
            descricaoVaga,
          );

          log('TOOL', `Recruiter message ${resultado.fonte === 'cache' ? '(cache)' : '(new)'}: ${nomeRecrutador} — ${empresa}`);

          return JSON.stringify({
            sucesso: true,
            texto: resultado.texto,
            caracteres: resultado.texto.length,
            fonte: resultado.fonte,
            instrucao: 'Use este texto como nota ao enviar convite de conexao no LinkedIn. Passos: (1) va ao perfil do recrutador, (2) clique em "Conectar", (3) clique em "Adicionar nota", (4) cole o texto com browser_type, (5) clique em "Enviar". Apos sucesso, use registrar_mensagem_recrutador.',
          });
        } catch (error) {
          const msg = error instanceof Error ? error.message : String(error);
          log('ERRO', `Recruiter message failed: ${msg}`);
          return `ERRO: ${msg}. Pule o envio de mensagem para este recrutador.`;
        }
      }

      case 'verificar_recrutador_ja_contatado': {
        const urlPerfil = args.url_perfil as string;
        const jaContatado = verificarRecrutadorJaContatado(urlPerfil);
        return jaContatado
          ? 'JA_CONTATADO: Este recrutador ja recebeu uma mensagem anteriormente. Pule.'
          : 'NOVO: Este recrutador ainda nao foi contatado. Pode prosseguir.';
      }

      case 'registrar_mensagem_recrutador': {
        const sucesso = registrarMensagemRecrutador({
          nome_recrutador: args.nome_recrutador as string,
          cargo_recrutador: (args.cargo_recrutador as string) || undefined,
          empresa: args.empresa as string,
          url_perfil: args.url_perfil as string,
          url_vaga: (args.url_vaga as string) || undefined,
          titulo_vaga: (args.titulo_vaga as string) || undefined,
          mensagem: args.mensagem as string,
          score_vaga: (args.score_vaga as number) || undefined,
        });

        if (sucesso) {
          log('AGENTE', `Message recorded: ${args.nome_recrutador} — ${args.empresa}`);
          return 'REGISTRADO: Mensagem para recrutador salva no banco de dados.';
        }
        return 'ERRO: Falha ao registrar (recrutador possivelmente ja contatado).';
      }

      case 'resolver_captcha_telegram': {
        const screenshotB64 = args.screenshot_base64 as string;
        const urlCaptcha = args.url_vaga as string;

        log('FALHA', `CAPTCHA detected at: ${urlCaptcha}. Requesting resolution via Telegram...`);

        try {
          const solucao = await solicitarResolucaoCaptcha(screenshotB64, urlCaptcha);

          if (solucao) {
            return JSON.stringify({
              sucesso: true,
              solucao,
              instrucao: 'Digite esta solucao no campo do CAPTCHA usando browser_type e depois submeta o formulario. Se o CAPTCHA rejeitar a solucao, tire outro screenshot e chame esta tool novamente (max 3 tentativas).',
            });
          }

          return JSON.stringify({
            sucesso: false,
            motivo: 'timeout',
            instrucao: 'Nenhuma solucao recebida em 5 minutos. Use reportar_falha com codigo "captcha" para pular esta vaga.',
          });
        } catch (error) {
          const msg = error instanceof Error ? error.message : String(error);
          log('ERRO', `CAPTCHA resolver: ${msg}`);
          return JSON.stringify({
            sucesso: false,
            motivo: msg,
            instrucao: 'Falha ao solicitar resolucao. Use reportar_falha com codigo "captcha" para pular esta vaga.',
          });
        }
      }

      case 'gerar_cover_letter': {
        const descricao = args.descricao_vaga as string;
        const empresa = args.empresa as string;
        const titulo = args.titulo_vaga as string;

        if (!_geminiApiKey || !_geminiModel) {
          return 'ERRO: Configuracao do Gemini nao disponivel para gerar cover letter.';
        }

        try {
          const resultado = await gerarCoverLetter(
            _geminiApiKey,
            _geminiModel,
            perfil,
            descricao,
            empresa,
            titulo,
          );

          log('TOOL', `Cover letter ${resultado.fonte === 'cache' ? '(cache)' : '(new)'} for ${titulo} — ${empresa}`);

          return JSON.stringify({
            sucesso: true,
            texto: resultado.texto,
            fonte: resultado.fonte,
            instrucao: 'Cole este texto no campo de carta de apresentacao do formulario. Voce pode fazer pequenos ajustes de tom se necessario.',
          });
        } catch (error) {
          const msg = error instanceof Error ? error.message : String(error);
          log('ERRO', `Cover letter failed: ${msg}`);
          return `ERRO: ${msg}. Escreva uma resposta curta baseada no perfil do candidato como fallback.`;
        }
      }

      case 'gerar_curriculo_tailored': {
        const descricao = args.descricao_vaga as string;
        const titulo = (args.titulo_vaga as string) || '';
        const empresa = (args.empresa as string) || '';

        if (!_geminiApiKey || !_geminiModel) {
          log('ERRO', 'Tailored resume: API key or model not configured');
          return 'ERRO: Configuracao do Gemini nao disponivel. Use escolher_curriculo como fallback.';
        }

        try {
          const resultado = await gerarCurriculoTailored(
            _geminiApiKey,
            _geminiModel,
            perfil,
            descricao,
          );

          log('TOOL', `Tailored resume ${resultado.fonte === 'cache' ? '(cache)' : '(new)'}: ${resultado.caminhoPDF}`);

          return JSON.stringify({
            sucesso: true,
            caminho: resultado.caminhoPDF,
            caminho_html: resultado.caminhoHTML,
            fonte: resultado.fonte,
            motivo: `Curriculo personalizado ${resultado.fonte === 'cache' ? 'recuperado do cache' : 'gerado com sucesso'} para: ${titulo || 'vaga'} ${empresa ? `na ${empresa}` : ''}.`,
          });
        } catch (error) {
          const msg = error instanceof Error ? error.message : String(error);
          log('ERRO', `Tailored resume failed: ${msg}`);
          return `ERRO: ${msg}. Use escolher_curriculo como fallback.`;
        }
      }

      default:
        return `ERRO: Tool "${name}" nao reconhecida.`;
    }
  };
}
