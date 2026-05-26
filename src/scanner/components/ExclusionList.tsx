'use client'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/utilities/ui'
import { ChevronDown, ChevronUp, X } from 'lucide-react'
import { useState, type KeyboardEvent } from 'react'

interface ExclusionListProps {
  terms: string[]
  onChange: (terms: string[]) => void
  disabled?: boolean
}

export function ExclusionList({ terms, onChange, disabled }: ExclusionListProps) {
  const [inputValue, setInputValue] = useState('')
  const [expanded, setExpanded] = useState(false)

  const addTerms = () => {
    const newTerms = inputValue
      .split(',')
      .map((t) => t.trim())
      .filter((t) => t.length > 0 && !terms.includes(t))

    if (newTerms.length === 0) return
    onChange([...terms, ...newTerms])
    setInputValue('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      addTerms()
    }
  }

  const removeTerm = (term: string) => {
    onChange(terms.filter((t) => t !== term))
  }

  return (
    <div>
      <div className="mb-2 flex items-center">
        {/* Not localized — excluded terms are English identifiers used for filtering */}
        <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Excluded terms ({terms.length})
        </label>
        <div className="flex-1" />
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="rounded p-1 text-muted-foreground hover:bg-accent"
        >
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>
      </div>
      <div className="flex gap-2">
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Add term and press Enter or comma..."
          className="flex-1"
        />
        <Button onClick={addTerms} disabled={disabled || inputValue.trim().length === 0} size="sm">
          Add
        </Button>
      </div>
      {expanded && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {terms.map((term) => (
            <span
              key={term}
              className={cn(
                'inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs',
                disabled && 'opacity-50',
              )}
            >
              {term}
              <button
                type="button"
                onClick={() => removeTerm(term)}
                disabled={disabled}
                className="rounded-sm hover:bg-destructive/20"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
