export interface Experiencia {
  empresa: string;
  cargo: string;
  periodo: string;
  descricao: string;
}

export interface Perfil {
  nome: string;
  email: string;
  telefone: string;
  linkedin: string;
  github: string;
  portfolio: string;
  curriculo_path: string;
  titulo_profissional: string;
  anos_experiencia: number;
  stack_principal: string[];
  resumo_profissional: string;
  pretensao_salarial: string;
  modelo_trabalho: string[];
  cidade: string;
  disponibilidade: string;
  palavras_chave_busca: string[];
  experiencias?: Experiencia[];
  bancos_de_dados?: string[];
  metodologias?: string[];
  informacoes_extras?: Record<string, string>;
}

export interface Site {
  nome: string;
  urls_busca: string[];
  tipo: 'portal_vagas' | 'linkedin' | 'empresa_direta';
  instrucoes: string;
  ativo: boolean;
}

export interface SitesConfig {
  sites: Site[];
}

export interface Candidatura {
  id?: number;
  plataforma: string;
  titulo_vaga: string;
  empresa: string;
  url: string;
  data_aplicacao: string;
  mensagem_enviada: number;
  status: string;
  score?: number;
  screenshot_path?: string;
}

export interface AgenteConfig {
  geminiApiKey: string;
  geminiModel: string;
  cdpEndpoint: string;
  limiteDiario: number;
  delayMin: number;
  delayMax: number;
  dryRun: boolean;
  scoreMinimo: number;
  dashboardPort: number;
  // Telegram (opcional)
  telegramBotToken: string;
  telegramChatId: string;
  // Email (opcional)
  smtpHost: string;
  smtpPort: number;
  smtpUser: string;
  smtpPass: string;
  emailDestinatario: string;
  // Cron (desativado por padrão)
  cronAtivo: boolean;
  cronHorario: string;
  // Multi-LLM (opcional — provider auxiliar para cover letter, currículo, mensagem)
  llmAuxProvider: string;
  llmAuxModel: string;
  ollamaUrl: string;
  openaiApiKey: string;
  openaiBaseUrl: string;
  // Fallback (Ollama) — usado SÓ enquanto o Gemini está rate-limited.
  // É uma CADEIA: tenta cada modelo em ordem até um responder.
  fallbackAtivo: boolean;
  fallbackModels: string[];
  fallbackNumCtx: number;
}

export interface RespostasPredefinidas {
  [chave: string]: string | Record<string, string>;
}
