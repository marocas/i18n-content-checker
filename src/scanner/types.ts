export interface LocaleConfig {
  code: string
  label: string
  flag: string
}

export interface ScanRequest {
  urls: string[]
  locales: string[]
  excludedTerms: string[]
  model: string
  useLLM: boolean
}

export interface FlaggedSentence {
  text: string
  englishWords: string[]
}

export interface DetectionResult {
  untranslatedPercent: number
  examples: FlaggedSentence[]
}

export interface LocaleScanResult {
  locale: string
  url: string
  status: 'clean' | 'english_found' | 'error'
  untranslatedPercent: number
  examples: FlaggedSentence[]
  errorMessage?: string
}

export interface AppConfig {
  model: string
  excludedTerms: string[]
  useLLM: boolean
}

export interface OllamaModel {
  name: string
  size: number
  parameterSize: string
}

export interface ScannerConfigProps {
  locales: LocaleConfig[]
  excludedTerms: string[]
  urlInput: {
    label: string
    placeholder: string
  }
  llmToggle: {
    label: string
    hint: string
    defaultEnabled: boolean
  }
  howItWorks?: string
}
