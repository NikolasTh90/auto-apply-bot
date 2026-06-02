// Classificação de falhas adaptada do ApplyPilot
// https://github.com/nicognaW/ApplyPilot
//
// ApplyPilot usa 13 tipos de falha permanente e attempts=99 como sentinela.
// Nós melhoramos com: backoff exponencial, categorias em português,
// e integração com o banco de vagas_vistas.

// ========== FALHAS PERMANENTES ==========
// Nunca retentar — a vaga ou situação não vai mudar.
export const FALHAS_PERMANENTES = new Set([
  'vaga_expirada',            // Vaga fechada ou não existe mais
  'captcha',                  // CAPTCHA detectado (sem handler ainda)
  'sessao_expirada',          // Login/sessão expirou — precisa relogar
  'localizacao_inelegivel',   // Presencial/híbrido fora de Uberlândia
  'ja_aplicou',               // Candidato já se candidatou a esta vaga
  'conta_necessaria',         // Precisa criar conta em plataforma específica
  'nao_e_vaga',               // Página não é uma vaga de emprego
  'sso_obrigatorio',          // Requer SSO (Google, Microsoft, etc.)
  'site_bloqueado',           // Site bloqueou acesso (ban, IP block)
  'cloudflare',               // Proteção Cloudflare/anti-bot ativa
  'formulario_incompativel',  // Formulário que o bot não consegue preencher
  'vaga_interna',             // Vaga exclusiva para funcionários internos
  'idioma_incompativel',      // Vaga exige idioma que o candidato não tem
]);

// ========== FALHAS RETRIÁVEIS ==========
// Vale tentar de novo — problema pode ser temporário.
export const FALHAS_RETRIAVEIS = new Set([
  'timeout',                  // Timeout de rede
  'erro_rede',                // Erro de conexão (DNS, TCP, etc.)
  'pagina_nao_carregou',      // Página não carregou corretamente
  'erro_servidor',            // HTTP 500, 502, 503, 504
  'elemento_nao_encontrado',  // Elemento desapareceu da página (race condition)
  'erro_upload',              // Falha no upload de currículo/arquivo
  'erro_mcp',                 // Erro de comunicação com Playwright MCP
]);

export const MAX_TENTATIVAS = 3;

export function ehFalhaPermanente(codigoFalha: string): boolean {
  return FALHAS_PERMANENTES.has(codigoFalha);
}

export function ehFalhaRetriavel(codigoFalha: string): boolean {
  return FALHAS_RETRIAVEIS.has(codigoFalha);
}

// Backoff exponencial: 5s → 15s → 45s
// Melhoria sobre o ApplyPilot que não tem backoff nenhum.
export function calcularBackoff(tentativa: number): number {
  return 5000 * Math.pow(3, tentativa - 1);
}

// Classifica erros de API/rede capturados no catch do agente
export function classificarErroAPI(mensagem: string): 'rate_limit' | 'rede' | 'fatal' {
  const msg = mensagem.toLowerCase();

  if (msg.includes('429') || msg.includes('rate_limit') || msg.includes('resource_exhausted')) {
    return 'rate_limit';
  }

  if (
    msg.includes('econnrefused') ||
    msg.includes('econnreset') ||
    msg.includes('etimedout') ||
    msg.includes('enotfound') ||
    msg.includes('fetch failed') ||
    msg.includes('network') ||
    msg.includes('socket hang up') ||
    msg.includes('503') ||
    msg.includes('502') ||
    msg.includes('504')
  ) {
    return 'rede';
  }

  return 'fatal';
}

// Delay com backoff para rate limit (30s base, 60s, 120s)
export function calcularBackoffRateLimit(tentativa: number): number {
  return 30000 * Math.pow(2, tentativa - 1);
}

// Extrai o tempo de espera sugerido pela API Gemini num erro 429.
// Formatos vistos: "Please retry in 34.238s", "retryDelay":"36s".
// Retorna o delay em ms (com folga de 2s) ou null se nao encontrar.
export function extrairRetryDelayMs(mensagem: string): number | null {
  const m =
    mensagem.match(/retry in (\d+(?:\.\d+)?)s/i) ||
    mensagem.match(/"?retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)s/i);
  if (!m) return null;
  const segundos = parseFloat(m[1]);
  if (!isFinite(segundos)) return null;
  return Math.round(segundos * 1000) + 2000;
}
