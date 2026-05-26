'use client'

import { Button } from '@/components/ui/button'
import { ExclusionList } from '@/scanner/components/ExclusionList'
import { LocaleSelector } from '@/scanner/components/LocaleSelector'
import { ModelSelector } from '@/scanner/components/ModelSelector'
import { ScanResults } from '@/scanner/components/ScanResults'
import { UrlInput } from '@/scanner/components/UrlInput'
import { fetchConfig, saveConfig } from '@/scanner/services/config-service'
import { OllamaUnavailableError, scanLocales } from '@/scanner/services/scan-service'
import type { LocaleScanResult, ScannerConfigProps } from '@/scanner/types'
import { AlertCircle, Info, Search } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'

function useDebouncedSave(value: unknown, key: string, delayMs = 500): void {
  const timer = useRef<ReturnType<typeof setTimeout>>(null)
  const isInitial = useRef(true)

  useEffect(() => {
    if (isInitial.current) {
      isInitial.current = false
      return
    }
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => {
      saveConfig({ [key]: value }).catch(() => {})
    }, delayMs)
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [value, key, delayMs])
}

interface ScannerPageProps {
  config: ScannerConfigProps
}

export function ScannerPage({ config }: ScannerPageProps) {
  const [urls, setUrls] = useState<string[]>([])
  const [selectedLocales, setSelectedLocales] = useState(config.locales.map((l) => l.code))
  const [excludedTerms, setExcludedTerms] = useState(config.excludedTerms)
  const [selectedModel, setSelectedModel] = useState('')
  const [useLLM, setUseLLM] = useState(config.llmToggle.defaultEnabled)
  const [results, setResults] = useState<LocaleScanResult[]>([])
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [configLoaded, setConfigLoaded] = useState(false)
  const [ollamaToast, setOllamaToast] = useState(false)

  useEffect(() => {
    fetchConfig()
      .then((c) => {
        console.log('Fetched config:', c)

        if (c.model) setSelectedModel(c.model)
        if (typeof c.useLLM === 'boolean') setUseLLM(c.useLLM)
        // Only update excludedTerms if API returns non-empty array (user preferences)
        // Empty array from API means no user override, keep block defaults
        if (Array.isArray(c.excludedTerms) && c.excludedTerms.length > 0) {
          setExcludedTerms(c.excludedTerms)
        }
      })
      .catch(() => {})
      .finally(() => setConfigLoaded(true))
  }, [])

  useDebouncedSave(selectedModel, 'model')
  useDebouncedSave(excludedTerms, 'excludedTerms')
  useDebouncedSave(useLLM, 'useLLM')

  const handleScan = useCallback(async () => {
    if (urls.length === 0) {
      setError('Add at least one URL to scan (press Enter after typing)')
      return
    }
    if (selectedLocales.length === 0) {
      setError('Select at least one locale to scan')
      return
    }

    setError(null)
    setScanning(true)
    setResults([])

    try {
      await scanLocales(
        { urls, locales: selectedLocales, excludedTerms, model: selectedModel, useLLM },
        (result) => setResults((prev) => [...prev, result]),
      )
    } catch (err) {
      if (err instanceof OllamaUnavailableError) {
        setOllamaToast(true)
        setTimeout(() => setOllamaToast(false), 8000)
      } else {
        setError(err instanceof Error ? err.message : 'Scan failed')
      }
    } finally {
      setScanning(false)
    }
  }, [urls, selectedLocales, excludedTerms, selectedModel, useLLM])

  return (
    <div className="container py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">i18n Content Checker</h1>
        <p className="mt-1 text-muted-foreground">
          Scan localized pages for untranslated English content
        </p>
      </div>

      <div className="flex flex-col gap-6">
        {/* URL input + scan button */}
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <UrlInput
              value={urls}
              onChange={setUrls}
              disabled={scanning}
              label={config.urlInput.label}
              placeholder={config.urlInput.placeholder}
            />
          </div>
          <Button
            onClick={handleScan}
            disabled={scanning || urls.length === 0 || (useLLM && !selectedModel)}
            size="lg"
            className="h-[44px]"
          >
            <Search className="mr-2 h-4 w-4" />
            {scanning ? 'Scanning...' : 'Scan pages'}
          </Button>
        </div>

        {/* Locale selector */}
        <LocaleSelector
          locales={config.locales}
          selectedLocales={selectedLocales}
          onChange={setSelectedLocales}
          disabled={scanning}
        />

        {/* Exclusion list */}
        <ExclusionList terms={excludedTerms} onChange={setExcludedTerms} disabled={scanning} />

        {/* Model selector */}
        {configLoaded && (
          <ModelSelector value={selectedModel} onChange={setSelectedModel} disabled={scanning} />
        )}

        {/* LLM toggle */}
        <label className="flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={useLLM}
            onChange={(e) => setUseLLM(e.target.checked)}
            disabled={scanning}
            className="h-4 w-4 rounded border-border"
          />
          <span className="text-sm">{config.llmToggle.label}</span>
          <span className="text-xs text-muted-foreground">{config.llmToggle.hint}</span>
        </label>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 rounded-md border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm text-red-600 dark:text-red-400">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
            <button
              type="button"
              onClick={() => setError(null)}
              className="ml-auto text-red-600 hover:text-red-800 dark:text-red-400"
            >
              ×
            </button>
          </div>
        )}

        {/* Scanning progress */}
        {scanning && (
          <div className="h-1 w-full overflow-hidden rounded-full bg-secondary">
            <div className="h-full w-1/3 animate-pulse rounded-full bg-primary" />
          </div>
        )}

        {/* Info box */}
        {config.howItWorks && (
          <div className="flex items-start gap-2 rounded-md border border-blue-500/30 bg-blue-500/5 px-4 py-3 text-sm text-muted-foreground">
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-500" />
            <span>{config.howItWorks}</span>
          </div>
        )}

        {/* Results */}
        <ScanResults results={results} locales={config.locales} />
      </div>

      {/* Ollama toast */}
      {ollamaToast && (
        <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-lg bg-red-600 px-4 py-3 text-sm text-white shadow-lg">
          Could not connect to Ollama. Make sure it is running with{' '}
          <code className="rounded bg-red-700 px-1">ollama serve</code>.
        </div>
      )}
    </div>
  )
}
