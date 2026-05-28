'use client'

import { Input } from '@/components/ui/input'
import { cn } from '@/utilities/ui'
import { X } from 'lucide-react'
import { useState, type KeyboardEvent } from 'react'

interface UrlInputProps {
  value: string[]
  onChange: (value: string[]) => void
  disabled?: boolean
  label?: string
  placeholder?: string
}

function isValidUrl(value: string): boolean {
  try {
    new URL(value)
    return true
  } catch {
    return false
  }
}

export function UrlInput({ value, onChange, disabled, label, placeholder }: UrlInputProps) {
  const [inputValue, setInputValue] = useState('')

  const addUrl = () => {
    const trimmed = inputValue.trim()
    if (!trimmed) return
    if (!isValidUrl(trimmed)) return
    if (value.includes(trimmed)) return

    onChange([...value, trimmed])
    setInputValue('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      addUrl()
    }
    if (e.key === 'Backspace' && !inputValue && value.length > 0) {
      onChange(value.slice(0, -1))
    }
  }

  const removeUrl = (url: string) => {
    onChange(value.filter((v) => v !== url))
  }

  return (
    <div>
      <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {label || 'English URLs to scan'}
      </label>
      <div
        className={cn(
          'flex min-h-[44px] flex-wrap items-center gap-1.5 rounded-md border bg-background px-3 py-2',
          disabled && 'opacity-50',
        )}
      >
        {value.map((url) => (
          <span
            key={url}
            className="inline-flex items-center gap-1 rounded-md bg-secondary px-2 py-0.5 text-xs text-secondary-foreground"
          >
            {url}
            <button
              type="button"
              onClick={() => removeUrl(url)}
              disabled={disabled}
              className="ml-0.5 rounded-sm hover:bg-destructive/20"
            >
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={
            value.length === 0
              ? placeholder || 'https://www.example.com/pricing (press Enter to add)'
              : 'Add another URL…'
          }
          className="min-w-[200px] flex-1 border-0 p-0 shadow-none focus-visible:ring-0 focus-visible:outline-none"
        />
      </div>
    </div>
  )
}
