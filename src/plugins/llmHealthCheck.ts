import type { Plugin } from 'payload'

export const llmHealthCheck: Plugin = (incomingConfig) => {
  return {
    ...incomingConfig,
    onInit: async (payload) => {
      if (incomingConfig.onInit) await incomingConfig.onInit(payload)

      const settings = await payload.findGlobal({ slug: 'ai-translation' })

      const host = settings.llmHost || process.env.LLM_HOST || 'http://localhost:11434'

      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 5000)

        const res = await fetch(`${host}/api/tags`, { signal: controller.signal })
        clearTimeout(timeout)

        if (!res.ok) throw new Error(`Status ${res.status}`)

        const data = await res.json()
        const models = data.models?.map((m: { name: string }) => m.name) || []

        payload.logger.info(`✓ LLM connected at ${host} (models: ${models.join(', ') || 'none'})`)
      } catch (error) {
        payload.logger.warn(
          `⚠ LLM not reachable at ${host} — auto-translation will not work. Start Ollama or update the host in Translation Settings.`,
        )
      }
    },
  }
}
