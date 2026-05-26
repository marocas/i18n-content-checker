export const localeConfig = [
  { code: 'en', label: 'English' },
  { code: 'pt', label: 'Português' },
  { code: 'es', label: 'Español' },
  { code: 'fr', label: 'Français' },
] as const

export const locales = localeConfig.map((l) => l.code)
export type Locale = (typeof localeConfig)[number]['code']
export const defaultLocale: Locale = 'en'
