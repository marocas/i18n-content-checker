#!/usr/bin/env node
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { z } from 'zod'

import type { AppConfig, LocaleConfig, LocaleScanResult, OllamaModel } from '../scanner/types'

const BASE_URL = process.env.SCANNER_BASE_URL ?? 'http://localhost:3000'
const FETCH_TIMEOUT_MS = 120_000

const server = new McpServer({
  name: 'i18n-scanner',
  version: '1.0.0',
})

// ── Logging ──────────────────────────────────────────────────

function log(level: 'info' | 'warn' | 'error', tool: string, message: string) {
  const entry = JSON.stringify({
    timestamp: new Date().toISOString(),
    level,
    tool,
    message,
  })
  process.stderr.write(entry + '\n')
}

// ── Helpers ──────────────────────────────────────────────────

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)

  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return (await res.json()) as T
  } finally {
    clearTimeout(timeout)
  }
}

async function readNdjsonStream(response: Response): Promise<LocaleScanResult[]> {
  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response stream')

  const decoder = new TextDecoder()
  const results: LocaleScanResult[] = []
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (!line.trim()) continue
      const parsed = JSON.parse(line)
      if (parsed._streamError) throw new Error(parsed.message)
      results.push(parsed as LocaleScanResult)
    }
  }

  if (buffer.trim()) {
    const parsed = JSON.parse(buffer)
    if (parsed._streamError) throw new Error(parsed.message)
    results.push(parsed as LocaleScanResult)
  }

  return results
}

// ── Active Scans Registry ────────────────────────────────────

const activeScans = new Map<string, AbortController>()
let scanCounter = 0

function generateScanId(): string {
  return `scan_${++scanCounter}_${Date.now()}`
}

function formatResults(results: LocaleScanResult[], sourceUrls: string[]): string {
  if (results.length === 0) return 'No results.'

  const lines: string[] = []

  for (const sourceUrl of sourceUrls) {
    lines.push(`# Source: ${sourceUrl}`, '')

    const urlResults = results.filter((r) => r.url.includes(new URL(sourceUrl).pathname))
    const group = urlResults.length > 0 ? urlResults : results

    for (const r of group) {
      const header = `## ${r.locale} — ${r.url}`
      const status =
        r.status === 'clean'
          ? '✅ Clean (no untranslated content)'
          : r.status === 'error'
            ? `❌ Error: ${r.errorMessage}`
            : `⚠️ ${r.untranslatedPercent}% untranslated content detected`

      lines.push(header, status)

      if (r.examples.length > 0) {
        lines.push('', '**Examples:**')
        for (const ex of r.examples) {
          lines.push(`- "${ex.text}" (English words: ${ex.englishWords.join(', ')})`)
        }
      }

      lines.push('')
    }
  }

  return lines.join('\n')
}

interface ScannerBlockDefaults {
  locales: LocaleConfig[]
  excludedTerms: string[]
}

async function fetchBlockDefaults(): Promise<ScannerBlockDefaults> {
  try {
    return await fetchJson<ScannerBlockDefaults>('/api/scanner/locales')
  } catch (error) {
    log('warn', 'fetch_block_defaults', error instanceof Error ? error.message : 'Unknown error')
    return { locales: [], excludedTerms: [] }
  }
}

function describeLocales(locales: LocaleConfig[]): string {
  if (locales.length === 0) {
    return 'Locale codes to check. Defaults to the locales configured on the Scanner block.'
  }

  return `Locale codes to check. Defaults to all Scanner block locales. Present these as selectable options to the user: ${locales.map((locale) => `${locale.flag} ${locale.code} (${locale.label})`).join(', ')}`
}

function createLocalesSchema(locales: LocaleConfig[]) {
  if (locales.length === 0) {
    return z.array(z.string()).optional().describe(describeLocales(locales))
  }

  const localeCodes = locales.map((locale) => locale.code) as [string, ...string[]]
  return z.array(z.enum(localeCodes)).optional().describe(describeLocales(locales))
}

function registerTools(blockDefaults: ScannerBlockDefaults) {
  const availableLocales = blockDefaults.locales

  // ── Tools ────────────────────────────────────────────────────

  server.registerTool(
    'scan_pages',
    {
      title: 'Scan Pages for Untranslated Content',
      description:
        'Scan one or more URLs for untranslated (English) content in the specified locales. Returns a detailed report per URL/locale combination.',
      inputSchema: {
        urls: z.array(z.url()).describe('URLs to scan (English version of the pages)'),
        locales: createLocalesSchema(availableLocales),
        useLLM: z
          .boolean()
          .optional()
          .describe('Use LLM (Ollama) for example extraction. Defaults to current config.'),
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        openWorldHint: true,
      },
    },
    async ({ urls, locales, useLLM }) => {
      const scanId = generateScanId()
      log(
        'info',
        'scan_pages',
        `[${scanId}] Scanning ${urls.length} URL(s) across ${locales?.length ?? 'all'} locale(s)`,
      )

      // Fetch user config + block defaults, merge excludedTerms
      const config = await fetchJson<AppConfig>('/api/scanner/config')
      const currentDefaults = await fetchBlockDefaults()
      const fallbackLocales =
        currentDefaults.locales.length > 0 ? currentDefaults.locales : availableLocales
      const selectedLocales = locales?.length
        ? locales
        : fallbackLocales.map((locale) => locale.code)

      // User overrides take priority; fall back to block defaults
      const excludedTerms =
        config.excludedTerms.length > 0 ? config.excludedTerms : currentDefaults.excludedTerms

      const body = {
        urls,
        locales: selectedLocales,
        excludedTerms,
        model: config.model,
        useLLM: useLLM ?? config.useLLM,
      }

      const controller = new AbortController()
      activeScans.set(scanId, controller)
      const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)

      try {
        const response = await fetch(`${BASE_URL}/api/scanner/scan`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        })

        if (!response.ok) {
          return {
            content: [
              {
                type: 'text' as const,
                text: `Scan failed: ${response.status} ${response.statusText}`,
              },
            ],
          }
        }

        const results = await readNdjsonStream(response)
        log('info', 'scan_pages', `[${scanId}] Scan complete: ${results.length} result(s)`)
        const report = formatResults(results, urls)

        return { content: [{ type: 'text' as const, text: report }] }
      } catch (err) {
        if (
          controller.signal.aborted &&
          !(err instanceof Error && err.message.includes('timeout'))
        ) {
          log('info', 'scan_pages', `[${scanId}] Scan cancelled`)
          return {
            content: [{ type: 'text' as const, text: `Scan ${scanId} was cancelled.` }],
          }
        }
        throw err
      } finally {
        clearTimeout(timeout)
        activeScans.delete(scanId)
      }
    },
  )

  server.registerTool(
    'get_config',
    {
      title: 'Get Scanner Configuration',
      description: 'Get the current scanner configuration (model, excluded terms, LLM toggle).',
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        openWorldHint: false,
      },
    },
    async () => {
      log('info', 'get_config', 'Fetching scanner configuration')
      const config = await fetchJson<AppConfig>('/api/scanner/config')
      const text = [
        `**Model:** ${config.model || '(none)'}`,
        `**Use LLM:** ${config.useLLM ? 'Yes' : 'No'}`,
        `**Excluded terms:** ${config.excludedTerms.length ? config.excludedTerms.join(', ') : '(none)'}`,
      ].join('\n')

      return { content: [{ type: 'text' as const, text }] }
    },
  )

  server.registerTool(
    'list_models',
    {
      title: 'List Ollama Models',
      description: 'List available Ollama models for LLM-powered example extraction.',
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        openWorldHint: false,
      },
    },
    async () => {
      log('info', 'list_models', 'Listing available Ollama models')
      const data = await fetchJson<{ models: OllamaModel[]; error?: string }>('/api/scanner/models')

      if (data.error) {
        return { content: [{ type: 'text' as const, text: `Ollama unavailable: ${data.error}` }] }
      }

      if (data.models.length === 0) {
        return { content: [{ type: 'text' as const, text: 'No models available.' }] }
      }

      const text = data.models
        .map((m) => `- **${m.name}** (${m.parameterSize}, ${(m.size / 1e9).toFixed(1)} GB)`)
        .join('\n')

      return { content: [{ type: 'text' as const, text }] }
    },
  )

  server.registerTool(
    'list_locales',
    {
      title: 'List Available Locales',
      description: 'List all available locale codes that can be scanned.',
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        openWorldHint: false,
      },
    },
    async () => {
      log('info', 'list_locales', 'Listing available locales')
      const defaults = await fetchBlockDefaults()

      if (defaults.locales.length === 0) {
        return {
          content: [{ type: 'text' as const, text: 'No Scanner block locales configured.' }],
        }
      }

      const text = defaults.locales
        .map((locale) => `- ${locale.flag} **${locale.code}** — ${locale.label}`)
        .join('\n')
      return { content: [{ type: 'text' as const, text }] }
    },
  )

  server.registerTool(
    'cancel_scan',
    {
      title: 'Cancel Scan',
      description:
        'Cancel an in-progress scan. If no scanId is provided, cancels all active scans.',
      inputSchema: {
        scanId: z
          .string()
          .optional()
          .describe('The scan ID to cancel. Omit to cancel all active scans.'),
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: true,
        openWorldHint: false,
      },
    },
    async ({ scanId }) => {
      if (scanId) {
        const controller = activeScans.get(scanId)
        if (!controller) {
          return {
            content: [{ type: 'text' as const, text: `No active scan found with ID: ${scanId}` }],
          }
        }
        controller.abort()
        activeScans.delete(scanId)
        log('info', 'cancel_scan', `Cancelled scan: ${scanId}`)
        return { content: [{ type: 'text' as const, text: `Scan ${scanId} cancelled.` }] }
      }

      const count = activeScans.size
      if (count === 0) {
        return { content: [{ type: 'text' as const, text: 'No active scans to cancel.' }] }
      }

      for (const [id, controller] of activeScans) {
        controller.abort()
        activeScans.delete(id)
      }
      log('info', 'cancel_scan', `Cancelled ${count} active scan(s)`)
      return {
        content: [{ type: 'text' as const, text: `Cancelled ${count} active scan(s).` }],
      }
    },
  )
}

// ── Start ────────────────────────────────────────────────────

async function main() {
  const blockDefaults = await fetchBlockDefaults()
  registerTools(blockDefaults)

  const transport = new StdioServerTransport()
  await server.connect(transport)
}

main().catch((err) => {
  console.error('MCP server failed to start:', err)
  process.exit(1)
})
