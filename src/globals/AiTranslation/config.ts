import type { GlobalConfig } from 'payload'

export const AiTranslation: GlobalConfig = {
  slug: 'ai-translation',
  label: 'AI Translation',
  access: {
    read: () => true,
  },
  admin: {
    group: 'Settings',
  },
  fields: [
    {
      name: 'llmHost',
      type: 'text',
      label: 'LLM Host URL',
      required: true,
      admin: {
        placeholder: 'http://localhost:11434',
        description: 'The base URL of the Ollama or OpenAI-compatible API server.',
      },
    },
    {
      name: 'llmModel',
      type: 'text',
      label: 'LLM Model',
      required: true,
      admin: {
        placeholder: 'llama3.1',
        description: 'The model to use for translations (e.g. llama3.1, mistral, gpt-4o).',
      },
    },
    {
      name: 'llmApiKey',
      type: 'text',
      label: 'API Key',
      admin: {
        placeholder: 'ollama',
        description: 'API key for the LLM service. Use "ollama" for local Ollama.',
      },
    },
    {
      name: 'translationPrompt',
      type: 'textarea',
      label: 'Translation Prompt',
      admin: {
        placeholder:
          'Translate the following JSON array from locale "{{localeFrom}}" to locale "{{localeTo}}". Return ONLY a valid JSON array of strings with no markdown, no code fences, no explanation. Input: {{texts}}',
        description:
          'Prompt template for translations. Use {{localeFrom}}, {{localeTo}}, and {{texts}} as placeholders.',
      },
    },
  ],
}
