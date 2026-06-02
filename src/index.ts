import 'dotenv/config';
import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { conectarPlaywrightMCP, desconectarMCP } from './mcp-client.js';
import { inicializarBanco, fecharBanco, registrarExecucao, contarCandidaturasHoje } from './database.js';
import { executarAgente } from './agente.js';
import { iniciarDashboard } from './dashboard.js';
import { inicializarLogger, log } from './logger.js';
import { configurarTelegram, notificarResumo, notificarErro } from './notificacoes.js';
import { configurarEmail, enviarRelatorioEmail, gerarHTMLRelatorio } from './email.js';
import { iniciarCron, pararCron } from './cron.js';
import { exibirResumoTokens, obterCustoTotal, obterTokensTotal, obterTotalChamadas, resetarTracker } from './token-tracker.js';
import { configurarLLMAux } from './llm-adapter.js';
import type { Perfil, SitesConfig, AgenteConfig } from './types.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function carregarConfig<T>(arquivo: string): T {
  const caminho = path.resolve(__dirname, '..', 'config', arquivo);
  try {
    const conteudo = readFileSync(caminho, 'utf-8');
    return JSON.parse(conteudo) as T;
  } catch (error) {
    console.error(`\nERRO: Could not load ${caminho}`);
    console.error('Check that the file exists and is in valid JSON format.\n');
    process.exit(1);
  }
}

function validarEnv(): AgenteConfig {
  const geminiApiKey = process.env.GEMINI_API_KEY;
  if (!geminiApiKey) {
    console.error('\nERRO: GEMINI_API_KEY not defined.');
    console.error('Copy .env.example to .env and fill in your key.\n');
    process.exit(1);
  }

  return {
    geminiApiKey,
    geminiModel: process.env.GEMINI_MODEL || 'gemini-2.5-pro',
    cdpEndpoint: process.env.CDP_ENDPOINT || 'http://localhost:9222',
    limiteDiario: parseInt(process.env.LIMITE_DIARIO || '10', 10),
    delayMin: parseInt(process.env.DELAY_MIN || '2000', 10),
    delayMax: parseInt(process.env.DELAY_MAX || '5000', 10),
    dryRun: process.env.DRY_RUN === 'true',
    scoreMinimo: parseInt(process.env.SCORE_MINIMO || '6', 10),
    dashboardPort: parseInt(process.env.DASHBOARD_PORT || '3000', 10),
    // Telegram
    telegramBotToken: process.env.TELEGRAM_BOT_TOKEN || '',
    telegramChatId: process.env.TELEGRAM_CHAT_ID || '',
    // Email
    smtpHost: process.env.SMTP_HOST || '',
    smtpPort: parseInt(process.env.SMTP_PORT || '587', 10),
    smtpUser: process.env.SMTP_USER || '',
    smtpPass: process.env.SMTP_PASS || '',
    emailDestinatario: process.env.EMAIL_DESTINATARIO || '',
    // Cron
    cronAtivo: process.env.CRON_ATIVO === 'true',
    cronHorario: process.env.CRON_HORARIO || '09:00',
    // Multi-LLM
    llmAuxProvider: process.env.LLM_AUX_PROVIDER || 'gemini',
    llmAuxModel: process.env.LLM_AUX_MODEL || process.env.GEMINI_MODEL || 'gemini-2.5-pro',
    ollamaUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
    openaiApiKey: process.env.OPENAI_API_KEY || '',
    openaiBaseUrl: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
    // Fallback local durante rate-limit do Gemini
    fallbackAtivo: process.env.OLLAMA_FALLBACK_ENABLED === 'true',
    fallbackModels: (process.env.OLLAMA_FALLBACK_MODEL || 'nemotron-3-super:cloud')
      .split(',').map((s) => s.trim()).filter(Boolean),
    fallbackNumCtx: parseInt(process.env.OLLAMA_FALLBACK_NUM_CTX || '65536', 10),
  };
}

async function executarFluxoPrincipal(config: AgenteConfig): Promise<void> {
  // Carrega configuracoes
  const perfil = carregarConfig<Perfil>('perfil.json');
  log('INFO', `Profile loaded: ${perfil.nome} (${perfil.titulo_profissional})`);

  const sites = carregarConfig<SitesConfig>('sites.json');
  const sitesAtivos = sites.sites.filter(s => s.ativo);
  log('INFO', `Sites: ${sitesAtivos.length} active of ${sites.sites.length} total`);

  // Inicializa banco de dados
  inicializarBanco();
  const candidaturasHoje = contarCandidaturasHoje();
  log('INFO', `Applications today: ${candidaturasHoje}/${config.limiteDiario}`);

  if (candidaturasHoje >= config.limiteDiario && !config.dryRun) {
    log('WARN', 'Daily application limit already reached. Try again tomorrow.');
    fecharBanco();
    return;
  }

  // Inicia dashboard web
  iniciarDashboard(config.dashboardPort);

  // Conecta ao Playwright MCP (que conecta ao Chrome)
  let mcpClient;
  try {
    mcpClient = await conectarPlaywrightMCP(config.cdpEndpoint);
  } catch (error) {
    log('ERRO', 'Could not connect to Chrome.');
    log('ERRO', 'Make sure Chrome is open with: google-chrome --remote-debugging-port=9222');
    await notificarErro('Falha ao conectar ao Chrome. Verifique se o CDP esta ativo.');
    fecharBanco();
    process.exit(1);
  }

  // Executa o agente
  const erros: string[] = [];
  let resultado = '';

  try {
    resultado = await executarAgente(mcpClient, perfil, sites, config);
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    erros.push(msg);
    log('ERRO', `FATAL ERROR: ${msg}`);
    await notificarErro(msg);
  }

  // Registra execucao
  const candidaturasDepois = contarCandidaturasHoje();
  const novasCandidaturas = candidaturasDepois - candidaturasHoje;

  registrarExecucao(
    novasCandidaturas,
    sitesAtivos.map(s => s.nome),
    erros,
  );

  // Resumo final
  log('INFO', '='.repeat(60));
  log('INFO', config.dryRun ? '  EXECUTION SUMMARY (DRY-RUN)' : '  EXECUTION SUMMARY');
  log('INFO', '='.repeat(60));
  log('INFO', `  Mode: ${config.dryRun ? 'DRY-RUN (simulation)' : 'PRODUCTION'}`);
  log('INFO', `  New applications: ${novasCandidaturas}`);
  log('INFO', `  Total today: ${candidaturasDepois}/${config.limiteDiario}`);
  log('INFO', `  Sites processed: ${sitesAtivos.map(s => s.nome).join(', ')}`);
  log('INFO', `  Errors: ${erros.length > 0 ? erros.join('; ') : 'None'}`);
  log('INFO', `  Token cost:        $${obterCustoTotal().toFixed(4)} USD (${obterTokensTotal().toLocaleString('pt-BR')} tokens, ${obterTotalChamadas()} calls)`);
  log('INFO', `  Dashboard: http://localhost:${config.dashboardPort}`);

  // Resumo detalhado de tokens
  exibirResumoTokens();

  if (resultado) {
    log('AGENTE', `Report:\n${resultado}`);
  }

  // Notificacoes
  await notificarResumo(novasCandidaturas, erros.length, config.dryRun);

  // Relatorio por email
  await enviarRelatorioEmail(
    `Job Bot — ${novasCandidaturas} candidatura(s) ${config.dryRun ? '(DRY-RUN)' : ''}`,
    gerarHTMLRelatorio({
      total: novasCandidaturas,
      empresas: [], // O agente registra no banco, aqui só o resumo
      erros,
      dryRun: config.dryRun,
      scoresMedio: 0,
    }),
  );

  // Cleanup
  await desconectarMCP();
  fecharBanco();
  log('INFO', 'Agent finished successfully.');
}

async function main() {
  // Inicializa logger antes de tudo
  inicializarLogger();

  log('INFO', '='.repeat(60));
  log('INFO', '  JOB BOT - Intelligent Application Agent');
  log('INFO', '='.repeat(60));

  // Valida ambiente
  const config = validarEnv();
  log('INFO', `Model: ${config.geminiModel}`);
  log('INFO', `CDP: ${config.cdpEndpoint}`);
  log('INFO', `Daily limit: ${config.limiteDiario}`);
  log('INFO', `Minimum score: ${config.scoreMinimo}/10`);

  if (config.dryRun) {
    log('WARN', '*** DRY-RUN MODE ACTIVE — no application will actually be submitted ***');
  }

  // Configura notificacoes (opcionais)
  configurarTelegram(config.telegramBotToken, config.telegramChatId);
  configurarEmail(config.smtpHost, config.smtpPort, config.smtpUser, config.smtpPass, config.emailDestinatario);

  // Configura LLM auxiliar (cover letter, currículo, mensagem recrutador)
  configurarLLMAux({
    provider: config.llmAuxProvider,
    model: config.llmAuxModel,
    geminiApiKey: config.geminiApiKey,
    geminiModel: config.geminiModel,
    ollamaUrl: config.ollamaUrl,
    openaiApiKey: config.openaiApiKey,
    openaiBaseUrl: config.openaiBaseUrl,
  });

  // Verifica se é modo cron ou execução única
  if (config.cronAtivo) {
    log('INFO', `CRON mode enabled. Time: ${config.cronHorario}`);
    log('INFO', 'The bot will keep running and execute automatically at the configured time.');
    log('INFO', 'Press Ctrl+C to stop.');

    iniciarCron(config.cronHorario, () => {
      resetarTracker();
      return executarFluxoPrincipal(config);
    });

    // Mantém o processo vivo
    process.on('SIGINT', () => {
      log('INFO', 'Received SIGINT. Stopping cron...');
      pararCron();
      fecharBanco();
      process.exit(0);
    });
  } else {
    // Execução única
    await executarFluxoPrincipal(config);
  }
}

main().catch((error) => {
  log('ERRO', `Unexpected error: ${error}`);
  fecharBanco();
  process.exit(1);
});
