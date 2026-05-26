import type { LocaleScanResult, ScanRequest } from '@/scanner/types'

export class OllamaUnavailableError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'OllamaUnavailableError'
  }
}

export class ScanCancelledError extends Error {
  constructor() {
    super('Scan cancelled')
    this.name = 'ScanCancelledError'
  }
}

export async function scanLocales(
  request: ScanRequest,
  onResult: (result: LocaleScanResult) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/scanner/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Scan failed: ${response.statusText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response stream')

  const decoder = new TextDecoder()
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
      if (parsed._streamError === 'ollama_unavailable') {
        throw new OllamaUnavailableError(parsed.message)
      }
      onResult(parsed)
    }
  }

  if (buffer.trim()) {
    const parsed = JSON.parse(buffer)
    if (parsed._streamError === 'ollama_unavailable') {
      throw new OllamaUnavailableError(parsed.message)
    }
    onResult(parsed)
  }
}
