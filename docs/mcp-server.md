# MCP Server — i18n Content Checker

## Overview

MCP (Model Context Protocol) server that exposes the i18n content scanner as tools for AI agents. Allows scanning web pages for untranslated content directly from VS Code Copilot Chat or any MCP-compatible client.

**Transport:** stdio
**SDK:** `@modelcontextprotocol/sdk` v1.29
**Entry point:** `src/mcp/server.ts`

## Architecture

The MCP server acts as a thin client that communicates with the running Next.js app via HTTP. It does **not** access the database or Payload CMS directly.

```
┌──────────────────┐      stdio       ┌─────────────────┐      HTTP       ┌──────────────────┐
│  AI Agent        │ ◄──────────────► │  MCP Server     │ ──────────────► │  Next.js App     │
│  (VS Code, etc.) │                  │  (src/mcp/)     │                 │  (localhost:3000)│
└──────────────────┘                  └─────────────────┘                 └──────────────────┘
```

**Prerequisite:** The Next.js app must be running (`pnpm dev`) for the MCP server to function.

## Tools

| Tool | Description | Annotations |
|------|-------------|-------------|
| `scan_pages` | Scan URL(s) for untranslated content across locales | `readOnlyHint: true`, `openWorldHint: true` |
| `get_config` | Get current scanner configuration | `readOnlyHint: true`, `openWorldHint: false` |
| `list_models` | List available Ollama models | `readOnlyHint: true`, `openWorldHint: false` |
| `list_locales` | List available locale codes | `readOnlyHint: true`, `openWorldHint: false` |

All tools are **read-only** (`destructiveHint: false`). No tool modifies state or data.

## Setup

### 1. VS Code (automatic)

The server is pre-configured in `.vscode/mcp.json`. To activate:

1. Start the app: `pnpm dev`
2. Open Command Palette → **MCP: List Servers**
3. `i18n-content-checker` should appear — start it
4. Use tools in Copilot Chat via `@` mention

### 2. Manual

```bash
# Requires the Next.js app running on localhost:3000
SCANNER_BASE_URL=http://localhost:3000 pnpm mcp
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCANNER_BASE_URL` | `http://localhost:3000` | Base URL of the running Next.js app |

## Logging

All tool invocations are logged to **stderr** as structured JSON:

```json
{"timestamp":"2026-05-26T15:30:00.000Z","level":"info","tool":"scan_pages","message":"Scanning 2 URL(s) across all locale(s)"}
```

Logs go to stderr to avoid interfering with the stdio MCP protocol on stdout.

## Compliance — AI Engineering Standards

Verified against [AI Engineering Standards](https://talkdesk.atlassian.net/wiki/spaces/IN/pages/6118349321/AI+Engineering+Standards) (InfoSec, Secção 6) and [TDD Talkdesk MCP](https://talkdesk.atlassian.net/wiki/spaces/TET/pages/6379143473/20260506+TDD+Talkdesk+MCP).

### Section 6 — MCP Requirements

| Requisito | Estado | Notas |
|-----------|--------|-------|
| **6.1** Apenas vendors/serviços aprovados | ✅ | Usa `@modelcontextprotocol/sdk` (protocolo open-source standard) |
| **6.2** Least privilege — sem secrets, customer data, admin access | ✅ | O server não acede a secrets, credenciais, ou dados de clientes. Apenas faz scan de páginas públicas e lê configuração local |
| **6.3** Documentado | ✅ | Este documento |
| **6.3** Access controls, logging, monitoring | ✅ | Logging estruturado (JSON) para stderr em todas as tool invocations |
| **6.4** Context injection = external API risk | ✅ | O server apenas chama APIs internas da própria app (localhost) |
| **6.5** Contexto mínimo necessário | ✅ | Cada tool expõe apenas os dados necessários para a sua função |

### Tool Annotations (TDD Talkdesk MCP)

Todas as tools declaram annotations conforme requisitos do TDD (~30% das rejeições são por annotations em falta):

- **`readOnlyHint`**: `true` em todas (nenhuma modifica estado)
- **`destructiveHint`**: `false` em todas
- **`openWorldHint`**: `true` apenas em `scan_pages` (faz fetch a URLs externas); `false` nas restantes

### Data Classification

Seguindo a tabela de classificação do AI Engineering Standards:

| Dado | Tier | Justificação |
|------|------|--------------|
| URLs de páginas públicas | T1 — Public | Conteúdo web público |
| Texto extraído das páginas | T1 — Public | Conteúdo já publicado |
| Configuração do scanner (model, excluded terms) | T2 — Internal | Preferências internas sem impacto se divulgadas |
| Resultados do scan | T2 — Internal | Análise de qualidade interna |

Nenhum dado T3 (Confidential) ou T4 (Restricted) é processado pelo MCP server.

### Pendente

- [ ] **Aprovação Security** (6.1/6.3): Revisão pela equipa de Security antes de uso em produção
- [ ] **Monitorização centralizada**: Considerar integração com Dynatrace para logs em produção
