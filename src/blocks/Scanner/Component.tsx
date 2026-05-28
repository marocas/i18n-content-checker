import React from 'react'

import { ScannerPage } from '@/scanner/components/ScannerPage'
import type { LocaleConfig, ScannerConfigProps } from '@/scanner/types'

interface PayloadLocaleRow {
  id?: string
  code: string
  flag: string
  label: string
}

interface ScannerBlockProps {
  locales?: PayloadLocaleRow[] | unknown
  excludedTerms?: string[]
  urlInputLabel?: string
  urlInputPlaceholder?: string
  llmToggleLabel?: string
  llmToggleHint?: string
  llmToggleDefault?: boolean
  howItWorks?: string
}

function parseLocales(raw: unknown): LocaleConfig[] {
  if (!Array.isArray(raw)) return []
  return raw.map((item) => ({
    code: item.code ?? '',
    label: item.label ?? '',
    flag: item.flag ?? '',
  }))
}

export const ScannerBlock: React.FC<ScannerBlockProps> = (props) => {
  const hasExcludedTerms = Array.isArray(props.excludedTerms) && props.excludedTerms.length > 0
  const config: ScannerConfigProps = {
    locales: parseLocales(props.locales),
    excludedTerms: hasExcludedTerms ? (props.excludedTerms as string[]) : [],
    urlInput: {
      label: props.urlInputLabel ?? 'English URLs to scan',
      placeholder:
        props.urlInputPlaceholder ?? 'https://www.example.com/pricing (press Enter to add)',
    },
    llmToggle: {
      label: props.llmToggleLabel ?? 'Use LLM for deeper analysis',
      hint: props.llmToggleHint ?? '(slower but more thorough)',
      defaultEnabled: props.llmToggleDefault ?? false,
    },
    howItWorks: props.howItWorks,
  }

  return <ScannerPage config={config} />
}
