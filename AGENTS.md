# i18n Content Checker v2

## Tech Stack

- **Framework**: Next.js 16 (App Router, Turbopack) + React 19 + TypeScript (strict)
- **CMS / Backend**: Payload CMS v3 (`@payloadcms/next`) + PostgreSQL
- **UI**: Tailwind CSS 4 + shadcn/ui + Radix primitives
- **Scanner**: cheerio (HTML parsing) + Ollama (local LLM for example extraction)
- **Testing**: Vitest (integration) + Playwright (e2e)
- **Package manager**: pnpm (do NOT use npm or yarn)

## Project Structure

```
src/
├── app/
│   ├── (frontend)/[locale]/   # Public website (i18n routes)
│   │   ├── [...slug]/         # CMS-managed pages (Payload layout builder)
│   │   └── posts/             # Blog posts
│   ├── (payload)/             # Payload CMS admin panel
│   └── api/scanner/           # Scanner API routes (scan, config, models)
├── scanner/                   # Scanner feature (self-contained)
│   ├── components/            # Scanner UI components (client-side)
│   ├── lib/                   # Server-side logic (heuristic, LLM, URL utils)
│   ├── services/              # Client-side fetch wrappers for scanner API
│   └── types.ts               # Scanner-specific types
├── mcp/                       # MCP server (see docs/mcp-server.md)
├── collections/               # Payload collection definitions
├── globals/                   # Payload globals (Header, Footer, AiTranslation)
├── i18n/config.ts             # Locale configuration (en, pt, es, fr)
├── proxy.ts                   # Middleware for locale routing
└── payload.config.ts          # Payload CMS configuration
```

## Commands

```bash
pnpm dev           # Start dev server (Turbopack)
pnpm build         # Production build
pnpm lint          # ESLint
pnpm test          # Run all tests (integration + e2e)
pnpm test:int      # Vitest integration tests only
pnpm test:e2e      # Playwright e2e tests only
pnpm mcp           # Start MCP server (requires app running)
pnpm seed          # Seed the database
pnpm dev:fresh     # Reset DB schema + start dev
```

## Conventions

### Code Style

- **No semicolons**, single quotes, trailing commas, 100 char width, 2-space indent (see `.prettierrc.json`)
- Path alias: `@/*` → `src/*`
- Files: `kebab-case.ts` / `PascalCase.tsx` for components
- TypeScript strict mode — avoid `any`

### Architecture Boundaries

| What you're adding     | Where it goes                                  |
| ---------------------- | ---------------------------------------------- |
| Payload collection     | `src/collections/<Name>.ts`                    |
| Payload global         | `src/globals/<Name>/config.ts`                 |
| Scanner UI component   | `src/scanner/components/`                      |
| Scanner server logic   | `src/scanner/lib/`                             |
| Scanner API endpoint   | `src/app/api/scanner/<name>/route.ts`          |
| Scanner client service | `src/scanner/services/`                        |
| CMS page/frontend      | `src/app/(frontend)/[locale]/`                 |
| Shared types           | `src/scanner/types.ts` (scanner) or co-located |
| MCP server             | `src/mcp/`                                     |

### Scanner Detection (Hybrid)

1. **Heuristic** (`language-heuristic.ts`) — instant, deterministic English word-frequency analysis
2. **LLM** (`ollama-detector.ts`) — samples chunks via Ollama for illustrative examples
3. If heuristic returns 0% untranslated, LLM is skipped

### API Routes

- Scanner API uses **NDJSON streaming** for long-running scan responses
- API routes are server-side only — never import from `services/` in API routes
- Services (`scanner/services/`) are client-side fetch wrappers — no business logic

### i18n

- Locales: `en` (default), `pt`, `es`, `fr` — defined in `src/i18n/config.ts`
- Frontend routes: `(frontend)/[locale]/`
- Payload fields use `localized: true` for translatable content

## Key Documentation

- [MCP Server](docs/mcp-server.md) — Architecture, tools, compliance checklist
