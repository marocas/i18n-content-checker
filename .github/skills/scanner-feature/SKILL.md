---
name: scanner-feature
description: 'i18n: Use when adding a new scanner feature, endpoint, or detection capability. Covers creating API routes, client services, UI components, and types following the hybrid detection pattern (heuristic + LLM).'
argument-hint: 'Feature name, e.g. "batch export" or "locale comparison"'
---

# Scanner Feature Skill

## When to Use

- Adding a new scanner API endpoint
- Creating a new scanner UI component
- Adding a new detection or analysis capability
- Extending scanner types

## Architecture Overview

The scanner is a self-contained feature within the Payload CMS app:

```
src/scanner/
├── components/     # Client-side React components ('use client')
├── lib/            # Server-side logic (heuristic, LLM, URL utils)
├── services/       # Client-side fetch wrappers for API routes
└── types.ts        # All scanner-specific types

src/app/api/scanner/
└── <name>/route.ts # API route handlers (server-side only)
```

## Procedure

### 1. Define types

Add interfaces to `src/scanner/types.ts`. Keep scanner types together — don't scatter across files.

```typescript
// src/scanner/types.ts
export interface MyFeatureRequest {
  // input params
}

export interface MyFeatureResult {
  // output shape
}
```

### 2. Create the API route

Create `src/app/api/scanner/<name>/route.ts`. API routes are **server-side only**.

Rules:

- Never import from `src/scanner/services/` (those are client-side wrappers)
- Import server logic from `src/scanner/lib/`
- For long-running operations, use **NDJSON streaming** (see `scan/route.ts` as reference)
- Validate input before processing
- Return proper HTTP status codes

```typescript
// src/app/api/scanner/<name>/route.ts
import type { MyFeatureRequest } from '@/scanner/types'
import { NextRequest } from 'next/server'

export async function POST(request: NextRequest): Promise<Response> {
  const body: MyFeatureRequest = await request.json()

  // Validate input
  if (!body.requiredField) {
    return Response.json({ error: 'Missing required field' }, { status: 400 })
  }

  // Process and return
  const result = await processFeature(body)
  return Response.json(result)
}
```

For streaming responses (long-running operations):

```typescript
const stream = new ReadableStream({
  async start(controller) {
    const encoder = new TextEncoder()
    for (const item of items) {
      const result = await process(item)
      controller.enqueue(encoder.encode(JSON.stringify(result) + '\n'))
    }
    controller.close()
  },
})

return new Response(stream, {
  headers: {
    'Content-Type': 'application/x-ndjson',
    'Transfer-Encoding': 'chunked',
  },
})
```

### 3. Create the client service

Create `src/scanner/services/<name>-service.ts`. Services are **client-side fetch wrappers only** — no business logic.

```typescript
// src/scanner/services/<name>-service.ts
import type { MyFeatureRequest, MyFeatureResult } from '@/scanner/types'

export async function myFeature(request: MyFeatureRequest): Promise<MyFeatureResult> {
  const response = await fetch('/api/scanner/<name>', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`Request failed: ${response.statusText}`)
  }

  return response.json()
}
```

For NDJSON streaming responses, use the callback pattern from [scan-service.ts](../../src/scanner/services/scan-service.ts).

### 4. Create the UI component

Create `src/scanner/components/<Name>.tsx`. Components are `'use client'`.

Rules:

- Use shadcn/ui + Radix primitives (not MUI)
- Import UI primitives from `@/components/ui/`
- Use `useCallback` for event handlers
- Keep components focused — extract sub-components when needed

```typescript
'use client'

import { Button } from '@/components/ui/button'
import { useCallback, useState } from 'react'

export function MyFeature() {
  const [loading, setLoading] = useState(false)

  const handleAction = useCallback(async () => {
    setLoading(true)
    try {
      // call service
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div>
      <Button onClick={handleAction} disabled={loading}>
        {loading ? 'Processing...' : 'Run'}
      </Button>
    </div>
  )
}
```

### 5. Add MCP tool (if applicable)

If the feature should be accessible via MCP, add a tool to `src/mcp/server.ts`:

- All tools must have `readOnlyHint`/`destructiveHint` annotations
- Use `openWorldHint: true` only if the tool fetches external URLs
- Add logging via the `log()` helper

### 6. Update types export

Ensure new types are exported from `src/scanner/types.ts` so both API routes and services can import them.

## Checklist

- [ ] Types added to `src/scanner/types.ts`
- [ ] API route created at `src/app/api/scanner/<name>/route.ts`
- [ ] Client service created at `src/scanner/services/<name>-service.ts`
- [ ] UI component created at `src/scanner/components/<Name>.tsx`
- [ ] MCP tool added (if applicable)
- [ ] No cross-boundary imports (services ≠ API routes)
