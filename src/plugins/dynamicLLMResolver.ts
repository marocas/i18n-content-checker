import type { TranslateResolver } from '@payload-enchants/translator'

export const dynamicLLMResolver: TranslateResolver = {
  key: 'dynamic-llm',
  resolve: async ({ localeFrom, localeTo, texts, req }) => {
    req.payload.logger.info(
      `[AI Translate] ${localeFrom} → ${localeTo} | ${texts.length} texts: ${JSON.stringify(texts.slice(0, 3))}${texts.length > 3 ? '...' : ''}`,
    )

    if (texts.length === 0) {
      return { success: true, translatedTexts: [] }
    }

    const settings = await req.payload.findGlobal({ slug: 'ai-translation' })

    const host = settings.llmHost || 'http://localhost:11434'
    const model = settings.llmModel || 'llama3.1'
    const apiKey = settings.llmApiKey || 'ollama'

    req.payload.logger.info(
      `[AI Translate] Config from CMS → host: "${host}", model: "${model}", apiKey: "${apiKey ? '***' : 'empty'}"`,
    )

    const promptTemplate =
      settings.translationPrompt ||
      'Translate the following JSON array from locale "{{localeFrom}}" to locale "{{localeTo}}". Return ONLY a valid JSON array of strings with no markdown, no code fences, no explanation. Input: {{texts}}'

    const prompt = promptTemplate
      .replace('{{localeFrom}}', localeFrom)
      .replace('{{localeTo}}', localeTo)
      .replace('{{texts}}', JSON.stringify(texts))

    req.payload.logger.info(
      `[AI Translate] Calling ${host}/v1/chat/completions with model: ${model}`,
    )

    const response = await fetch(`${host}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.1,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      req.payload.logger.error(`[AI Translate] Request failed (${response.status}): ${errorText}`)
      return { success: false }
    }

    const data = await response.json()
    const content = data.choices?.[0]?.message?.content

    req.payload.logger.info(`[AI Translate] Raw LLM response: ${content?.substring(0, 200)}`)

    if (!content) {
      req.payload.logger.error('[AI Translate] LLM returned empty response')
      return { success: false }
    }

    // Parse the JSON array from the response
    let parsed: unknown
    try {
      parsed = JSON.parse(content.trim())
    } catch (e) {
      req.payload.logger.error(`[AI Translate] Failed to parse JSON: ${content.substring(0, 300)}`)
      return { success: false }
    }

    if (!Array.isArray(parsed) || parsed.length !== texts.length) {
      req.payload.logger.error(
        `[AI Translate] Array mismatch. Expected ${texts.length}, got ${Array.isArray(parsed) ? parsed.length : typeof parsed}. Response: ${JSON.stringify(parsed).substring(0, 300)}`,
      )
      return { success: false }
    }

    req.payload.logger.info(`[AI Translate] Success — translated ${parsed.length} texts`)
    return { success: true, translatedTexts: parsed }
  },
}
