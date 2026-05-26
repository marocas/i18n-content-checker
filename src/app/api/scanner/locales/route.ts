import { defaultLocale } from '@/i18n/config'
import type { LocaleConfig } from '@/scanner/types'
import configPromise from '@payload-config'
import { unstable_cache } from 'next/cache'
import { getPayload } from 'payload'

function parseLocales(raw: unknown): LocaleConfig[] {
  if (!Array.isArray(raw)) return []

  return raw.flatMap((item) => {
    if (!item || typeof item !== 'object') return []

    const locale = item as Partial<LocaleConfig>

    return [
      {
        code: typeof locale.code === 'string' ? locale.code : '',
        label: typeof locale.label === 'string' ? locale.label : '',
        flag: typeof locale.flag === 'string' ? locale.flag : '',
      },
    ].filter((entry) => entry.code.length > 0)
  })
}

const getScannerLocales = unstable_cache(
  async (): Promise<LocaleConfig[]> => {
    const payload = await getPayload({ config: configPromise })

    const pages = await payload.find({
      collection: 'pages',
      draft: false,
      limit: 1,
      locale: defaultLocale,
      overrideAccess: false,
      pagination: false,
      where: {
        slug: {
          equals: 'scanner',
        },
      },
    })

    const page = pages.docs[0]
    const scannerBlock = Array.isArray(page?.layout)
      ? page.layout.find((block) => block?.blockType === 'scanner')
      : null

    if (!scannerBlock) return []

    return parseLocales(scannerBlock.locales)
  },
  ['scanner-block-locales'],
  { revalidate: 300 },
)

export async function GET(): Promise<Response> {
  try {
    const locales = await getScannerLocales()
    return Response.json(locales)
  } catch {
    return Response.json([])
  }
}
