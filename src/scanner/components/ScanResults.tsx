'use client'

import type { LocaleConfig, LocaleScanResult } from '@/scanner/types'
import { cn } from '@/utilities/ui'
import {
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Copy,
  ExternalLink,
  XCircle,
} from 'lucide-react'
import { useState } from 'react'

interface ScanResultsProps {
  results: LocaleScanResult[]
  locales: LocaleConfig[]
}

function StatusBadge({ status }: { status: LocaleScanResult['status'] }) {
  switch (status) {
    case 'clean':
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-green-500/10 px-2 py-0.5 text-xs font-medium text-green-600 dark:text-green-400">
          <CheckCircle className="h-3 w-3" /> Clean
        </span>
      )
    case 'english_found':
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-600 dark:text-yellow-400">
          <AlertTriangle className="h-3 w-3" /> English found
        </span>
      )
    case 'error':
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-red-500/10 px-2 py-0.5 text-xs font-medium text-red-600 dark:text-red-400">
          <XCircle className="h-3 w-3" /> Error
        </span>
      )
  }
}

function ProgressBar({ result }: { result: LocaleScanResult }) {
  if (result.status === 'error') return null

  const translatedPct = 100 - result.untranslatedPercent
  const color =
    translatedPct === 100 ? 'bg-green-500' : translatedPct >= 80 ? 'bg-yellow-500' : 'bg-red-500'

  return (
    <div className="px-4 pb-3">
      <div className="mb-1 flex justify-between text-xs text-muted-foreground">
        <span>{result.untranslatedPercent}% untranslated</span>
        <span className="font-semibold">{translatedPct}% translated</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn('h-full rounded-full transition-all', color)}
          style={{ width: `${translatedPct}%` }}
        />
      </div>
    </div>
  )
}

function highlightEnglishWords(text: string, englishWords: string[]): React.ReactNode {
  if (englishWords.length === 0) return text

  const escapedWords = englishWords.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const regex = new RegExp(`\\b(${escapedWords.join('|')})\\b`, 'gi')
  const parts = text.split(regex)

  return parts.map((part, i) => {
    const isMatch = englishWords.some((w) => w.toLowerCase() === part.toLowerCase())
    if (isMatch) {
      return (
        <span
          key={i}
          className="rounded bg-yellow-500/30 px-0.5 font-semibold text-yellow-700 dark:text-yellow-300"
        >
          {part}
        </span>
      )
    }
    return part
  })
}

function LocaleResultRow({
  result,
  locales,
}: {
  result: LocaleScanResult
  locales: LocaleConfig[]
}) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const locale = locales.find((l) => l.code === result.locale)

  const copyReport = () => {
    const lines = [
      `Locale: ${result.locale} (${result.url})`,
      `Status: ${result.status}`,
      `Untranslated: ${result.untranslatedPercent}%`,
      '',
    ]
    for (const sentence of result.examples) {
      lines.push(`- "${sentence.text}"`)
      lines.push(`  English words: ${sentence.englishWords.join(', ')}`)
    }
    navigator.clipboard.writeText(lines.join('\n'))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div
      className={cn(
        'overflow-hidden rounded-lg border',
        result.status === 'english_found' && 'border-yellow-500/50',
        result.status === 'error' && 'border-red-500/50',
      )}
    >
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-accent/50"
      >
        <span className="text-lg leading-none">{locale?.flag}</span>
        <span className="min-w-[50px] font-semibold">{result.locale}</span>
        <a
          href={result.url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:underline"
        >
          {result.url} <ExternalLink className="h-3 w-3" />
        </a>
        <div className="flex-1" />
        <StatusBadge status={result.status} />
        {result.examples.length > 0 && (
          <span className="text-xs text-muted-foreground">
            {result.examples.length} example{result.examples.length !== 1 ? 's' : ''}
          </span>
        )}
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      <ProgressBar result={result} />

      {expanded && (
        <div className="border-t px-4 pb-4">
          {result.status === 'error' && (
            <p className="py-2 text-sm text-red-600 dark:text-red-400">{result.errorMessage}</p>
          )}
          {result.examples.length > 0 && (
            <>
              <div className="flex justify-end pt-2">
                <button
                  type="button"
                  onClick={copyReport}
                  className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-accent"
                >
                  <Copy className="h-3 w-3" />
                  {copied ? 'Copied!' : 'Copy report'}
                </button>
              </div>
              {result.examples.map((sentence, i) => (
                <div key={i} className={cn('py-2', i < result.examples.length - 1 && 'border-b')}>
                  <p className="text-sm leading-relaxed">
                    {highlightEnglishWords(sentence.text, sentence.englishWords)}
                  </p>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {sentence.englishWords.map((word, j) => (
                      <span
                        key={`${word}-${j}`}
                        className="rounded border border-yellow-500/50 px-1.5 py-0.5 text-[10px] text-yellow-600 dark:text-yellow-400"
                      >
                        {word}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </>
          )}
          {result.status === 'clean' && (
            <p className="py-2 text-sm text-green-600 dark:text-green-400">
              No English content detected on this page.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export function ScanResults({ results, locales }: ScanResultsProps) {
  if (results.length === 0) return null

  const englishCount = results.filter((r) => r.status === 'english_found').length
  const errorCount = results.filter((r) => r.status === 'error').length
  const cleanCount = results.filter((r) => r.status === 'clean').length

  return (
    <div>
      <div className="mb-3 flex items-center gap-3">
        <h2 className="text-lg font-semibold">Results</h2>
        <span className="rounded-full border border-green-500/50 px-2 py-0.5 text-xs text-green-600 dark:text-green-400">
          {cleanCount} clean
        </span>
        {englishCount > 0 && (
          <span className="rounded-full border border-yellow-500/50 px-2 py-0.5 text-xs text-yellow-600 dark:text-yellow-400">
            {englishCount} with English
          </span>
        )}
        {errorCount > 0 && (
          <span className="rounded-full border border-red-500/50 px-2 py-0.5 text-xs text-red-600 dark:text-red-400">
            {errorCount} errors
          </span>
        )}
      </div>
      <div className="flex flex-col gap-3">
        {results.map((result) => (
          <LocaleResultRow
            key={`${result.locale}-${result.url}`}
            result={result}
            locales={locales}
          />
        ))}
      </div>
    </div>
  )
}
