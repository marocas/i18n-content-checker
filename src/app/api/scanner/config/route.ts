import type { AppConfig } from '@/scanner/types'
import configPromise from '@payload-config'
import { NextRequest } from 'next/server'
import { getPayload } from 'payload'

const CONFIG_KEY = 'scanner-user-config'

export async function GET(): Promise<Response> {
  const payload = await getPayload({ config: configPromise })

  try {
    const pref = await payload.find({
      collection: 'payload-preferences',
      where: { key: { equals: CONFIG_KEY } },
      limit: 1,
    })

    const stored = pref.docs[0]?.value as Partial<AppConfig> | undefined
    const config: AppConfig = {
      model: stored?.model ?? '',
      excludedTerms: stored?.excludedTerms ?? [],
      useLLM: stored?.useLLM ?? false,
    }
    return Response.json(config)
  } catch {
    return Response.json({ model: '', excludedTerms: [], useLLM: false })
  }
}

export async function PUT(request: NextRequest): Promise<Response> {
  const body = await request.json()
  const payload = await getPayload({ config: configPromise })

  try {
    const existing = await payload.find({
      collection: 'payload-preferences',
      where: { key: { equals: CONFIG_KEY } },
      limit: 1,
    })

    const stored = existing.docs[0]?.value as Partial<AppConfig> | undefined
    const current: AppConfig = {
      model: stored?.model ?? '',
      excludedTerms: stored?.excludedTerms ?? [],
      useLLM: stored?.useLLM ?? false,
    }

    const config: AppConfig = {
      model: typeof body.model === 'string' ? body.model : current.model,
      excludedTerms: Array.isArray(body.excludedTerms) ? body.excludedTerms : current.excludedTerms,
      useLLM: typeof body.useLLM === 'boolean' ? body.useLLM : current.useLLM,
    }

    if (existing.docs[0]) {
      await payload.update({
        collection: 'payload-preferences',
        id: existing.docs[0].id,
        data: { value: config as unknown as Record<string, unknown> },
      })
    } else {
      await payload.create({
        collection: 'payload-preferences',
        data: { key: CONFIG_KEY, value: config } as never,
      })
    }

    return Response.json(config)
  } catch {
    // Silently fail — user preferences are non-critical
    return Response.json({ model: '', excludedTerms: [], useLLM: false })
  }
}
