import { appendFileSync, mkdirSync, existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LOGS_DIR = path.resolve(__dirname, '..', 'logs');

let logFilePath: string;

export function inicializarLogger(): void {
  if (!existsSync(LOGS_DIR)) {
    mkdirSync(LOGS_DIR, { recursive: true });
  }

  const agora = new Date();
  const nomeArquivo = `execucao_${agora.toISOString().slice(0, 10)}_${agora.toTimeString().slice(0, 8).replace(/:/g, '-')}.log`;
  logFilePath = path.join(LOGS_DIR, nomeArquivo);

  log('INFO', `Logger initialized → ${logFilePath}`);
}

export function log(nivel: 'INFO' | 'WARN' | 'ERRO' | 'TOOL' | 'AGENTE' | 'FALHA', mensagem: string): void {
  const timestamp = new Date().toISOString();
  const linha = `[${timestamp}] [${nivel}] ${mensagem}\n`;

  // Sempre imprime no console
  if (nivel === 'ERRO' || nivel === 'FALHA') {
    process.stderr.write(linha);
  } else {
    process.stdout.write(linha);
  }

  // Salva no arquivo se o logger foi inicializado
  if (logFilePath) {
    try {
      appendFileSync(logFilePath, linha);
    } catch {
      // Silencioso — não quebrar o fluxo por erro de log
    }
  }
}

export function obterCaminhoLog(): string {
  return logFilePath || '';
}
