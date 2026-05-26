'use client'

import { Input } from '@/components/ui/input'
import { fetchModels } from '@/scanner/services/models-service'
import type { OllamaModel } from '@/scanner/types'
import { Loader2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

interface ModelSelectorProps {
  value: string
  onChange: (model: string) => void
  disabled?: boolean
}

function formatSize(bytes: number): string {
  const gb = bytes / 1e9
  if (gb >= 1) return `${gb.toFixed(1)} GB`
  return `${(bytes / 1e6).toFixed(0)} MB`
}

export function ModelSelector({ value, onChange, disabled }: ModelSelectorProps) {
  const [models, setModels] = useState<OllamaModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m)
        if (m.length > 0 && !value) {
          onChange(m[0].name)
        }
      })
      .catch(() => {
        setError('Could not connect to Ollama. Make sure it is running with `ollama serve`.')
      })
      .finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  return (
    <div ref={ref} className="relative">
      <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Ollama Model
      </label>

      {error && (
        <div className="mb-2 rounded-md border border-yellow-500/50 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-600 dark:text-yellow-400">
          {error}
        </div>
      )}

      <div className="relative">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setOpen(true)}
          disabled={disabled || loading}
          placeholder={loading ? 'Loading models…' : 'Select or type a model name'}
        />
        {loading && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          </div>
        )}
      </div>

      {open && models.length > 0 && (
        <div className="absolute z-50 mt-1 w-full rounded-md border bg-popover shadow-md">
          {models.map((model) => (
            <button
              key={model.name}
              type="button"
              onClick={() => {
                onChange(model.name)
                setOpen(false)
              }}
              className="flex w-full items-center justify-between px-3 py-2 text-sm hover:bg-accent"
            >
              <span>{model.name}</span>
              <span className="text-xs text-muted-foreground">
                {model.parameterSize} · {formatSize(model.size)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
