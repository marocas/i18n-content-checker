<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.

<!-- END:nextjs-agent-rules -->

# Project Architecture

## Tech Stack

- **Next.js 16** (App Router, Turbopack) + **React 19** + **TypeScript**
- **MUI 9** (Material UI) — uses slot-based API (`params.slotProps.input`, not `params.InputProps`)
- **Ollama** — local LLM for example extraction
- **cheerio** — server-side HTML parsing
- **pnpm** — package manager (do NOT use npm or yarn)

## Folder Structure

```
src/
├── app/              # Next.js App Router — pages and API routes only
│   ├── page.tsx      # Main page (client component, orchestrates state)
│   ├── layout.tsx    # Root layout
│   └── api/          # Server-side API routes (Next.js Route Handlers)
│       ├── config/   # GET/PUT config.json persistence
│       ├── models/   # GET available Ollama models
│       └── scan/     # POST scan locales (NDJSON streaming response)
├── components/       # React UI components ("use client")
├── services/         # Client-side fetch wrappers for API routes
├── lib/              # Server-side utilities and core logic
│   ├── types.ts      # Shared TypeScript interfaces
│   ├── ollama-detector.ts   # LLM integration (example extraction)
│   ├── language-heuristic.ts # Fast English detection heuristic
│   └── url-utils.ts  # URL manipulation helpers
└── theme.ts          # MUI theme configuration
```

## Conventions

### Where to put new code

| What you're adding        | Where it goes                    |
| ------------------------- | -------------------------------- |
| New page                  | `src/app/<route>/page.tsx`       |
| New API endpoint          | `src/app/api/<name>/route.ts`    |
| React component           | `src/components/<Name>.tsx`      |
| Client-side API call      | `src/services/<name>-service.ts` |
| Server-side utility/logic | `src/lib/<name>.ts`              |
| Shared types/interfaces   | `src/lib/types.ts`               |

### Rules

- **API routes** (`src/app/api/`) run server-side only. Never import from `services/` in API routes.
- **Services** (`src/services/`) are client-side fetch wrappers. They call API routes and return typed data. Never put business logic here.
- **Lib** (`src/lib/`) is for server-side logic. Never use `fetch()` to call our own API routes from lib — that's what services are for on the client.
- **Components** are `"use client"` React components. They receive data via props or call services directly. Keep them focused on presentation.
- **Types** shared between client and server live in `src/lib/types.ts`.
- Use **NDJSON streaming** for long-running API responses (see `scan/route.ts`).
- Config is persisted to `config.json` at project root via the config API route.

### MUI 9 Gotchas

- Autocomplete: use `params.slotProps.input` (NOT `params.InputProps`)
- Typography: put `fontWeight` in `sx` prop, not as a direct prop
- Always check MUI 9 migration notes before using MUI components

### Detection Architecture

The scanner uses a **hybrid approach**:

1. **Heuristic** (`language-heuristic.ts`) — instant, deterministic English word-frequency analysis for the untranslated percentage
2. **LLM** (`ollama-detector.ts`) — samples 1–2 text chunks via Ollama for illustrative example sentences
3. If heuristic returns 0% untranslated, the LLM is skipped entirely

### Naming

- Files: `kebab-case.ts` / `kebab-case.tsx`
- Components: `PascalCase.tsx`
- Services: `<name>-service.ts`
- Interfaces/types: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
