import type { Block } from 'payload'

export const Scanner: Block = {
  slug: 'scanner',
  interfaceName: 'ScannerBlock',
  labels: {
    singular: 'i18n Scanner',
    plural: 'i18n Scanners',
  },
  fields: [
    {
      name: 'locales',
      type: 'json',
      // Note: Not localized — these are locale codes (pt-pt, en, etc.), not translatable content
      label: 'Locales to Scan',
      required: true,
      admin: {
        components: {
          Field: '@/blocks/Scanner/LocalesInput#LocalesInput',
        },
      },
      defaultValue: [
        { code: 'pt-pt', label: 'Portuguese (PT)', flag: '🇵🇹' },
        { code: 'pt-br', label: 'Portuguese (BR)', flag: '🇧🇷' },
        { code: 'it-it', label: 'Italian', flag: '🇮🇹' },
        { code: 'de-de', label: 'German', flag: '🇩🇪' },
        { code: 'fr-fr', label: 'French', flag: '🇫🇷' },
        { code: 'es-es', label: 'Spanish', flag: '🇪🇸' },
      ],
    },
    {
      name: 'excludedTerms',
      type: 'json',
      // Note: Not localized — these are English identifiers (JavaScript, CSS, etc.) used for filtering, always in English
      label: 'Excluded Terms',
      admin: {
        components: {
          Field: '@/blocks/Scanner/TagsInput#TagsInput',
        },
      },
      defaultValue: ['JavaScript', 'CSS', 'HTML', 'CRM', 'API', 'SaaS', 'AI', 'Cloud'],
    },
    {
      type: 'row',
      fields: [
        {
          name: 'urlInputLabel',
          type: 'text',
          label: 'URL Input Label',
          defaultValue: 'English URLs to scan',
          admin: { width: '50%' },
        },
        {
          name: 'urlInputPlaceholder',
          type: 'text',
          label: 'URL Input Placeholder',
          defaultValue: 'https://www.example.com/pricing (press Enter to add)',
          admin: { width: '50%' },
        },
      ],
    },
    {
      type: 'row',
      fields: [
        {
          name: 'llmToggleLabel',
          type: 'text',
          label: 'LLM Toggle Label',
          defaultValue: 'Use LLM for deeper analysis',
          admin: { width: '40%' },
        },
        {
          name: 'llmToggleHint',
          type: 'text',
          label: 'LLM Toggle Hint',
          defaultValue: '(slower but more thorough)',
          admin: { width: '40%' },
        },
        {
          name: 'llmToggleDefault',
          type: 'checkbox',
          label: 'LLM enabled by default',
          defaultValue: false,
          admin: {
            width: '20%',
            style: { alignSelf: 'flex-end', paddingBottom: '0.75rem' },
          },
        },
      ],
    },
    {
      name: 'howItWorks',
      type: 'textarea',
      label: 'How it Works (info box)',
      defaultValue:
        'How this works: Pages are fetched server-side. The scanner extracts visible text, removes HTML, and flags English-looking words not present in the target language. Results may vary based on page structure and JavaScript rendering.',
      admin: {
        description: 'Displayed as an info banner on the scanner page.',
      },
    },
  ],
}
