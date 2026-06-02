import { log } from './logger.js';

// ============================================================
// TODO (precisa da sua configuração):
// 1. Instale o nodemailer: npm install nodemailer
// 2. Preencha as variáveis SMTP no .env:
//    SMTP_HOST=smtp.gmail.com
//    SMTP_PORT=587
//    SMTP_USER=seu-email@gmail.com
//    SMTP_PASS=sua-senha-de-app (gere em https://myaccount.google.com/apppasswords)
//    EMAIL_DESTINATARIO=seu-email@gmail.com
// ============================================================

interface EmailConfig {
  host: string;
  port: number;
  user: string;
  pass: string;
  destinatario: string;
}

let emailConfig: EmailConfig | null = null;

export function configurarEmail(host: string, port: number, user: string, pass: string, destinatario: string): void {
  if (!host || !user || !pass || !destinatario) {
    log('WARN', 'Email: Incomplete SMTP configuration. Email reports disabled.');
    return;
  }
  emailConfig = { host, port, user, pass, destinatario };
  log('INFO', 'Email: Configured successfully.');
}

export async function enviarRelatorioEmail(assunto: string, corpoHTML: string): Promise<boolean> {
  if (!emailConfig) {
    log('WARN', 'Email: Not configured. Skipping send.');
    return false;
  }

  try {
    // Import dinâmico para não quebrar se nodemailer não estiver instalado
    const nodemailer = await import('nodemailer');

    const transporter = nodemailer.createTransport({
      host: emailConfig.host,
      port: emailConfig.port,
      secure: emailConfig.port === 465,
      auth: {
        user: emailConfig.user,
        pass: emailConfig.pass,
      },
    });

    await transporter.sendMail({
      from: `"Job Bot" <${emailConfig.user}>`,
      to: emailConfig.destinatario,
      subject: assunto,
      html: corpoHTML,
    });

    log('INFO', `Email: Report sent to ${emailConfig.destinatario}`);
    return true;
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    log('ERRO', `Email: Failed to send — ${msg}`);
    return false;
  }
}

export function gerarHTMLRelatorio(dados: {
  total: number;
  empresas: string[];
  erros: string[];
  dryRun: boolean;
  scoresMedio: number;
}): string {
  const modo = dados.dryRun ? 'DRY-RUN (simulação)' : 'PRODUÇÃO';
  const listaEmpresas = dados.empresas.length > 0
    ? dados.empresas.map(e => `<li>${e}</li>`).join('')
    : '<li>Nenhuma candidatura realizada</li>';
  const listaErros = dados.erros.length > 0
    ? dados.erros.map(e => `<li style="color:#e74c3c">${e}</li>`).join('')
    : '<li style="color:#2ecc71">Nenhum erro</li>';

  return `
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 24px;">
  <div style="max-width: 600px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <h1 style="color: #2c3e50; margin-bottom: 8px;">Job Bot — Relatório</h1>
    <p style="color: #7f8c8d; margin-top: 0;">Modo: <strong>${modo}</strong></p>

    <div style="background: #ecf0f1; border-radius: 8px; padding: 16px; margin: 16px 0;">
      <h2 style="margin: 0; font-size: 48px; color: #3498db;">${dados.total}</h2>
      <p style="margin: 4px 0 0; color: #7f8c8d;">candidaturas realizadas</p>
      <p style="margin: 4px 0 0; color: #7f8c8d;">Score médio: ${dados.scoresMedio.toFixed(1)}/10</p>
    </div>

    <h3 style="color: #2c3e50;">Empresas</h3>
    <ul>${listaEmpresas}</ul>

    <h3 style="color: #2c3e50;">Erros</h3>
    <ul>${listaErros}</ul>

    <hr style="border: none; border-top: 1px solid #ecf0f1; margin: 24px 0;">
    <p style="color: #bdc3c7; font-size: 12px;">Gerado automaticamente pelo Job Bot</p>
  </div>
</body>
</html>`;
}
