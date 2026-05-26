import { detectEnglishWithLLM, isOllamaConnectionError } from '@/scanner/lib/ollama-detector'
import { buildLocalizedUrl } from '@/scanner/lib/url-utils'
import type { LocaleScanResult, ScanRequest } from '@/scanner/types'
import * as cheerio from 'cheerio'
import { NextRequest } from 'next/server'

const FETCH_TIMEOUT_MS = 15_000

async function fetchPageText(url: string): Promise<string> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) i18n-content-checker/1.0',
        Accept: 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.5',
      },
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`)
    }

    const html = await response.text()
    const $ = cheerio.load(html)

    $('script, style, noscript, svg, meta, link, head').remove()
    $('header, nav, footer').remove()

    const main = $('main.main').first()
    const root = main.length ? main : $('body')

    const BLOCK_TAGS =
      'p, h1, h2, h3, h4, h5, h6, li, td, th, div, section, article, blockquote, figcaption, dt, dd'
    root.find(BLOCK_TAGS).each((_, el) => {
      $(el).append('\n')
    })

    return root
      .text()
      .replace(/[^\S\n]+/g, ' ')
      .replace(/\n{2,}/g, '\n')
      .trim()
  } finally {
    clearTimeout(timeout)
  }
}

async function scanLocale(
  url: string,
  locale: string,
  model: string,
  excludedTerms: string[],
  useLLM: boolean,
): Promise<LocaleScanResult> {
  const localizedUrl = buildLocalizedUrl(url, locale)

  try {
    const pageText = await fetchPageText(localizedUrl)
    const detection = await detectEnglishWithLLM(pageText, locale, model, excludedTerms, useLLM)

    return {
      locale,
      url: localizedUrl,
      status: detection.untranslatedPercent > 0 ? 'english_found' : 'clean',
      untranslatedPercent: detection.untranslatedPercent,
      examples: detection.examples,
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error'
    return {
      locale,
      url: localizedUrl,
      status: 'error',
      untranslatedPercent: 0,
      examples: [],
      errorMessage: message,
    }
  }
}

export async function POST(request: NextRequest): Promise<Response> {
  const body: ScanRequest = await request.json()
  const { urls, locales, excludedTerms, model, useLLM } = body

  if (!urls?.length || !locales?.length) {
    return Response.json({ results: [] }, { status: 400 })
  }

  for (const u of urls) {
    try {
      new URL(u)
    } catch {
      return Response.json({ results: [] }, { status: 400 })
    }
  }

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()

      for (const url of urls) {
        for (const locale of locales) {
          try {
            const result = await scanLocale(url, locale, model, excludedTerms, useLLM)
            controller.enqueue(encoder.encode(JSON.stringify(result) + '\n'))
          } catch (err) {
            if (isOllamaConnectionError(err)) {
              const errorEvent = {
                _streamError: 'ollama_unavailable',
                message:
                  'Could not connect to Ollama. Make sure it is running with `ollama serve`.',
              }
              controller.enqueue(encoder.encode(JSON.stringify(errorEvent) + '\n'))
              controller.close()
              return
            }
            const message = err instanceof Error ? err.message : 'Unknown error'
            const errorResult: LocaleScanResult = {
              locale,
              url: '',
              status: 'error',
              untranslatedPercent: 0,
              examples: [],
              errorMessage: message,
            }
            controller.enqueue(encoder.encode(JSON.stringify(errorResult) + '\n'))
          }
        }
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
}
