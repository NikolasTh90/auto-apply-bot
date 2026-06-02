// Multi-LLM Adapter Pattern — adaptado do AIHawk (6 providers).
//
// O AIHawk usa LangChain como abstração. Nós usamos uma interface leve
// sem dependências extras: fetch nativo para Ollama e OpenAI-compatible,
// SDK nativo para Gemini.
//
// Arquitetura:
// - Agente principal: sempre Gemini (precisa de tool calling + MCP)
// - Módulos auxiliares (cover letter, currículo, mensagem): usam o provider
//   configurado em LLM_AUX_PROVIDER (pode ser Ollama = custo zero)
//
// Providers suportados:
// - gemini: Google Gemini via @google/genai (padrão)
// - ollama: Modelos locais via Ollama REST API (custo zero)
// - openai: OpenAI ou qualquer API compatível (ex: Together, Groq)

import { GoogleGenAI } from '@google/genai';
import { log } from './logger.js';
import { registrarUsoTokens, type UsageMetadata } from './token-tracker.js';

// ========== INTERFACE ==========

export interface LLMResponse {
  text: string;
  usageMetadata?: UsageMetadata;
}

export interface LLMProvider {
  readonly nome: string;
  generate(prompt: string): Promise<LLMResponse>;
}

// ========== GEMINI PROVIDER ==========

class GeminiProvider implements LLMProvider {
  readonly nome = 'gemini';
  private ai: GoogleGenAI;

  constructor(private apiKey: string, private model: string) {
    this.ai = new GoogleGenAI({ apiKey });
  }

  async generate(prompt: string): Promise<LLMResponse> {
    const response = await this.ai.models.generateContent({
      model: this.model,
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
    });

    return {
      text: response.text?.trim() ?? '',
      usageMetadata: response.usageMetadata as UsageMetadata | undefined,
    };
  }
}

// ========== OLLAMA PROVIDER ==========
// Usa a REST API nativa do Ollama (sem dependências extras).
// Endpoint: POST /api/chat
// O Ollama retorna prompt_eval_count e eval_count para tracking de tokens.

class OllamaProvider implements LLMProvider {
  readonly nome = 'ollama';

  constructor(private model: string, private baseUrl: string) {}

  async generate(prompt: string): Promise<LLMResponse> {
    const url = `${this.baseUrl}/api/chat`;

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        messages: [{ role: 'user', content: prompt }],
        stream: false,
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Ollama erro ${response.status}: ${body}`);
    }

    const data = await response.json() as {
      message: { content: string };
      prompt_eval_count?: number;
      eval_count?: number;
    };

    const promptTokens = data.prompt_eval_count ?? 0;
    const outputTokens = data.eval_count ?? 0;

    return {
      text: data.message.content.trim(),
      usageMetadata: {
        promptTokenCount: promptTokens,
        candidatesTokenCount: outputTokens,
        totalTokenCount: promptTokens + outputTokens,
      },
    };
  }
}

// ========== OPENAI-COMPATIBLE PROVIDER ==========
// Funciona com: OpenAI, Together AI, Groq, Mistral, vLLM, etc.
// Também funciona com Ollama via endpoint /v1/chat/completions.

class OpenAICompatProvider implements LLMProvider {
  readonly nome = 'openai';

  constructor(
    private model: string,
    private apiKey: string,
    private baseUrl: string,
  ) {}

  async generate(prompt: string): Promise<LLMResponse> {
    const url = `${this.baseUrl}/chat/completions`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.4,
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`OpenAI-compat erro ${response.status}: ${body}`);
    }

    const data = await response.json() as {
      choices: Array<{ message: { content: string } }>;
      usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
    };

    return {
      text: data.choices[0].message.content.trim(),
      usageMetadata: {
        promptTokenCount: data.usage?.prompt_tokens,
        candidatesTokenCount: data.usage?.completion_tokens,
        totalTokenCount: data.usage?.total_tokens,
      },
    };
  }
}

// ========== FACTORY + SINGLETON ==========

let providerAux: LLMProvider | null = null;
let providerFallback: LLMProvider | null = null;

/**
 * Configura o provider auxiliar (usado por cover letter, currículo, mensagem recrutador).
 * Chamar uma vez no index.ts após carregar as env vars.
 */
export function configurarLLMAux(config: {
  provider: string;
  model: string;
  geminiApiKey?: string;
  geminiModel?: string;
  ollamaUrl?: string;
  openaiApiKey?: string;
  openaiBaseUrl?: string;
}): void {
  providerAux = criarProvider(
    config.provider,
    config.model,
    config.geminiApiKey,
    config.ollamaUrl,
    config.openaiApiKey,
    config.openaiBaseUrl,
  );

  // Fallback: sempre Gemini (se disponível e não for o provider principal)
  if (config.provider !== 'gemini' && config.geminiApiKey) {
    providerFallback = new GeminiProvider(config.geminiApiKey, config.geminiModel || 'gemini-2.5-pro');
  }

  log('INFO', `Auxiliary LLM: ${providerAux.nome} (${config.model})${providerFallback ? ` — fallback: ${providerFallback.nome}` : ''}`);
}

function criarProvider(
  tipo: string,
  model: string,
  geminiApiKey?: string,
  ollamaUrl?: string,
  openaiApiKey?: string,
  openaiBaseUrl?: string,
): LLMProvider {
  switch (tipo) {
    case 'gemini':
      if (!geminiApiKey) throw new Error('GEMINI_API_KEY necessária para provider gemini');
      return new GeminiProvider(geminiApiKey, model);

    case 'ollama':
      return new OllamaProvider(model, ollamaUrl || 'http://localhost:11434');

    case 'openai':
      return new OpenAICompatProvider(
        model,
        openaiApiKey || '',
        openaiBaseUrl || 'https://api.openai.com/v1',
      );

    default:
      throw new Error(`Provider LLM desconhecido: "${tipo}". Use: gemini, ollama, ou openai.`);
  }
}

/**
 * Gera texto usando o provider auxiliar configurado.
 * Se falhar e houver fallback, tenta o fallback.
 *
 * @param prompt - Texto do prompt
 * @param contexto - Identificador para tracking de tokens (ex: 'cover_letter')
 * @returns Texto gerado
 */
export async function gerarTextoAux(prompt: string, contexto: string): Promise<LLMResponse> {
  if (!providerAux) {
    throw new Error('LLM auxiliar não configurado. Chame configurarLLMAux() primeiro.');
  }

  try {
    const response = await providerAux.generate(prompt);
    registrarUsoTokens(
      `${providerAux.nome}:aux`,
      response.usageMetadata,
      contexto,
    );
    return response;
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    log('WARN', `Auxiliary LLM (${providerAux.nome}) failed: ${msg}`);

    // Tenta fallback
    if (providerFallback) {
      log('INFO', `Trying fallback: ${providerFallback.nome}...`);
      const response = await providerFallback.generate(prompt);
      registrarUsoTokens(
        `${providerFallback.nome}:fallback`,
        response.usageMetadata,
        `${contexto}_fallback`,
      );
      return response;
    }

    throw error;
  }
}

/**
 * Retorna o nome do provider auxiliar atual.
 */
export function obterNomeProviderAux(): string {
  return providerAux?.nome ?? 'nao_configurado';
}
