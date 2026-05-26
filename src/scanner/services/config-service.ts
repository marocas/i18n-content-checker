import type { AppConfig } from '@/scanner/types'

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch('/api/scanner/config')
  return res.json()
}

export async function saveConfig(partial: Partial<AppConfig>): Promise<AppConfig> {
  const res = await fetch('/api/scanner/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial),
  })
  return res.json()
}
