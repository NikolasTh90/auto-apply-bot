// Fallback local (Ollama) para o loop agentico principal.
//
// Objetivo: enquanto o Gemini está rate-limited, em vez de ficar ocioso
// esperando a quota voltar, deixamos um modelo LOCAL (ex: gemma4:e2b-mlx via
// Ollama) dirigir o navegador. O Gemini continua sendo o principal (mais
// rápido e mais forte); o Ollama só assume durante a janela de cooldown.
//
// O histórico do agente é mantido no formato do Gemini (Content[] com parts
// text/functionCall/functionResponse). Aqui convertemos para o formato de
// mensagens do Ollama (/api/chat) e convertemos a resposta de volta para o
// formato do Gemini, para que o resto do loop não precise saber qual modelo
// respondeu.

import type { Content, Part } from '@google/genai';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { customToolDeclarations } from './tools.js';
import type { UsageMetadata } from './token-tracker.js';

// ---------- Tipos do Ollama ----------
interface OllamaTool {
  type: 'function';
  function: { name: string; description?: string; parameters?: unknown };
}
interface OllamaToolCall {
  function: { name: string; arguments: Record<string, unknown> | string };
}
interface OllamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: OllamaToolCall[];
  tool_name?: string;
}

// Resultado normalizado (mesma forma que o loop espera do Gemini)
export interface RespostaModelo {
  content: Content | undefined;            // turno do 'model' p/ empilhar no history
  functionCalls: Array<{ name: string; args: Record<string, unknown> }>;
  text: string;
  usageMetadata?: UsageMetadata;
  modelo?: string;                         // which model produced this response (for logging)
}

// ---------- Construção das tools no formato Ollama ----------
// Combina as tools customizadas do bot + as tools do Playwright MCP.
export async function montarToolsOllama(mcpClient: Client): Promise<OllamaTool[]> {
  const tools: OllamaTool[] = customToolDeclarations.map((d) => ({
    type: 'function' as const,
    function: {
      name: (d as { name?: string }).name ?? '',
      description: (d as { description?: string }).description,
      parameters: (d as { parameters?: unknown }).parameters,
    },
  }));

  try {
    const mcp = await mcpClient.listTools();
    for (const t of mcp.tools ?? []) {
      tools.push({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: t.inputSchema,
        },
      });
    }
  } catch {
    // se não listar, segue só com as customizadas
  }
  return tools;
}

// ---------- Conversão history (Gemini) -> messages (Ollama) ----------
function historyParaMensagensOllama(history: Content[], systemPrompt: string): OllamaMessage[] {
  const msgs: OllamaMessage[] = [{ role: 'system', content: systemPrompt }];

  for (const turno of history) {
    const parts: Part[] = turno.parts ?? [];

    if (turno.role === 'model') {
      const textParts = parts.filter((p) => typeof (p as { text?: string }).text === 'string');
      const callParts = parts.filter((p) => (p as { functionCall?: unknown }).functionCall);
      const texto = textParts.map((p) => (p as { text: string }).text).join('\n');
      const toolCalls: OllamaToolCall[] = callParts.map((p) => {
        const fc = (p as { functionCall: { name?: string; args?: Record<string, unknown> } }).functionCall;
        return { function: { name: fc.name ?? '', arguments: fc.args ?? {} } };
      });
      msgs.push({
        role: 'assistant',
        content: texto,
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }

    // role === 'user' : texto puro OU functionResponses
    for (const p of parts) {
      const part = p as {
        text?: string;
        functionResponse?: { name?: string; response?: { result?: unknown } };
      };
      if (typeof part.text === 'string') {
        msgs.push({ role: 'user', content: part.text });
      } else if (part.functionResponse) {
        const result = part.functionResponse.response?.result;
        msgs.push({
          role: 'tool',
          tool_name: part.functionResponse.name,
          content: typeof result === 'string' ? result : JSON.stringify(result),
        });
      }
    }
  }
  return msgs;
}

// ---------- Chamada principal ao Ollama com tools ----------
export async function chamarOllamaComTools(
  history: Content[],
  systemPrompt: string,
  tools: OllamaTool[],
  opts: { model: string; baseUrl: string; numCtx: number },
): Promise<RespostaModelo> {
  const messages = historyParaMensagensOllama(history, systemPrompt);

  // Timeout defensivo: se um modelo de cloud travar, abortamos para a cadeia
  // poder cair pro próximo modelo em vez de congelar o run inteiro.
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 120_000);
  let res: Response;
  try {
    res = await fetch(`${opts.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: opts.model,
        messages,
        tools,
        stream: false,
        // Desliga o modo "thinking": respostas mais diretas e tool calls mais
        // limpos; evita o modelo divagar/entrar em loop ruminando.
        think: false,
        options: { num_ctx: opts.numCtx, temperature: 0 },
      }),
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    throw new Error(`Ollama fallback erro ${res.status}: ${await res.text()}`);
  }

  const data = (await res.json()) as {
    message: { content?: string; tool_calls?: OllamaToolCall[] };
    prompt_eval_count?: number;
    eval_count?: number;
  };

  const texto = data.message.content?.trim() ?? '';
  const rawCalls = data.message.tool_calls ?? [];

  const functionCalls = rawCalls.map((tc) => {
    let args: Record<string, unknown> = {};
    const a = tc.function.arguments;
    if (typeof a === 'string') {
      try { args = JSON.parse(a); } catch { args = {}; }
    } else if (a && typeof a === 'object') {
      args = a;
    }
    return { name: tc.function.name, args };
  });

  // Monta o turno 'model' no formato Gemini para empilhar no history
  const parts: Part[] = [];
  if (texto) parts.push({ text: texto });
  for (const fc of functionCalls) parts.push({ functionCall: { name: fc.name, args: fc.args } });
  const content: Content = { role: 'model', parts };

  const promptTokens = data.prompt_eval_count ?? 0;
  const outputTokens = data.eval_count ?? 0;

  return {
    content,
    functionCalls,
    text: texto,
    modelo: opts.model,
    usageMetadata: {
      promptTokenCount: promptTokens,
      candidatesTokenCount: outputTokens,
      totalTokenCount: promptTokens + outputTokens,
    } as UsageMetadata,
  };
}
