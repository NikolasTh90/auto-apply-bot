import { log } from './logger.js';

// ============================================================
// AGENDAMENTO (CRON)
// Desativado por padrão. Para ativar:
// 1. No .env, defina: CRON_ATIVO=true
// 2. Configure o horário: CRON_HORARIO=09:00 (formato HH:MM)
// 3. O bot executará automaticamente no horário configurado
// ============================================================

let cronTimer: ReturnType<typeof setInterval> | null = null;

export function iniciarCron(horario: string, executar: () => Promise<void>): void {
  const [hora, minuto] = horario.split(':').map(Number);

  if (isNaN(hora) || isNaN(minuto) || hora < 0 || hora > 23 || minuto < 0 || minuto > 59) {
    log('ERRO', `Cron: Invalid time "${horario}". Use HH:MM format (e.g. 09:00)`);
    return;
  }

  log('INFO', `Cron: Scheduled to run daily at ${horario}`);

  // Verifica a cada minuto se é hora de executar
  cronTimer = setInterval(async () => {
    const agora = new Date();
    if (agora.getHours() === hora && agora.getMinutes() === minuto) {
      log('INFO', 'Cron: Starting scheduled run...');
      try {
        await executar();
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        log('ERRO', `Cron: Error during scheduled run — ${msg}`);
      }
    }
  }, 60_000); // Verifica a cada 60 segundos
}

export function pararCron(): void {
  if (cronTimer) {
    clearInterval(cronTimer);
    cronTimer = null;
    log('INFO', 'Cron: Schedule stopped.');
  }
}
