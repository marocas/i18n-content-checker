export interface LocaleConfig {
  code: string;
  label: string;
  flag: string;
}

export const AVAILABLE_LOCALES: LocaleConfig[] = [
  { code: "pt-pt", label: "Portuguese (PT)", flag: "🇵🇹" },
  { code: "pt-br", label: "Portuguese (BR)", flag: "🇧🇷" },
  { code: "it-it", label: "Italian", flag: "🇮🇹" },
  { code: "de-de", label: "German", flag: "🇩🇪" },
  { code: "fr-fr", label: "French", flag: "🇫🇷" },
  { code: "es-es", label: "Spanish", flag: "🇪🇸" },
];

export const DEFAULT_EXCLUDED_TERMS = [
  "JavaScript",
  "CSS",
  "HTML",
  "CRM",
  "API",
  "SaaS",
  "AI",
  "Cloud",
];

export interface ScanRequest {
  url: string;
  locales: string[];
  excludedTerms: string[];
  model: string;
  useLLM: boolean;
}

export interface FlaggedSentence {
  text: string;
  englishWords: string[];
}

export interface DetectionResult {
  untranslatedPercent: number;
  examples: FlaggedSentence[];
}

export interface LocaleScanResult {
  locale: string;
  url: string;
  status: "clean" | "english_found" | "error";
  untranslatedPercent: number;
  examples: FlaggedSentence[];
  errorMessage?: string;
}

export interface ScanResponse {
  results: LocaleScanResult[];
}
