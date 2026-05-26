'use client'

import type { LocaleConfig } from '@/scanner/types'
import { cn } from '@/utilities/ui'

interface LocaleSelectorProps {
  locales: LocaleConfig[]
  selectedLocales: string[]
  onChange: (locales: string[]) => void
  disabled?: boolean
}

export function LocaleSelector({
  locales,
  selectedLocales,
  onChange,
  disabled,
}: LocaleSelectorProps) {
  const toggleLocale = (code: string) => {
    if (selectedLocales.includes(code)) {
      onChange(selectedLocales.filter((l) => l !== code))
    } else {
      onChange([...selectedLocales, code])
    }
  }

  return (
    <div>
      {/* Not localized — locale codes are identifiers, not translatable */}
      <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Locales to scan
      </label>
      <div className="flex flex-wrap gap-2">
        {locales.map((locale) => {
          const isSelected = selectedLocales.includes(locale.code)
          return (
            <button
              key={locale.code}
              type="button"
              onClick={() => !disabled && toggleLocale(locale.code)}
              disabled={disabled}
              className={cn(
                'rounded-full border px-3 py-1 text-sm transition-colors',
                isSelected
                  ? 'border-primary bg-primary text-primary-foreground'
                  : 'border-border bg-background text-foreground hover:bg-accent',
                disabled && 'cursor-not-allowed opacity-50',
              )}
            >
              {locale.flag} {locale.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
