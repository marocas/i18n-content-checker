import type { Page, ScannerBlock } from '@/payload-types'
import configPromise from '@payload-config'
import { getPayload } from 'payload'

/**
 * Seeds a Scanner page with default block config.
 * Run with: npx tsx src/scripts/seed-scanner-config.ts
 */
async function seed() {
  const payload = await getPayload({ config: configPromise })

  const existing = await payload.find({
    collection: 'pages',
    where: { slug: { equals: 'scanner' } },
    limit: 1,
  })

  const scannerBlock: ScannerBlock = {
    blockType: 'scanner',
    locales: [
      { code: 'pt-pt', label: 'Portuguese (PT)', flag: '🇵🇹' },
      { code: 'pt-br', label: 'Portuguese (BR)', flag: '🇧🇷' },
      { code: 'it-it', label: 'Italian', flag: '🇮🇹' },
      { code: 'de-de', label: 'German', flag: '🇩🇪' },
      { code: 'fr-fr', label: 'French', flag: '🇫🇷' },
      { code: 'es-es', label: 'Spanish', flag: '🇪🇸' },
    ],
    excludedTerms: ['JavaScript', 'CSS', 'HTML', 'CRM', 'API', 'SaaS', 'AI', 'Cloud'],
    urlInputLabel: 'English URLs to scan',
    urlInputPlaceholder: 'https://www.example.com/pricing (press Enter to add)',
    llmToggleLabel: 'Use LLM for deeper analysis',
    llmToggleHint: '(slower but more thorough)',
    llmToggleDefault: false,
  }

  const pageData: Pick<Page, 'title' | 'slug' | 'hero' | 'layout' | '_status'> = {
    title: 'Scanner',
    slug: 'scanner',
    hero: {
      type: 'none',
    },
    layout: [scannerBlock],
    _status: 'published',
  }

  const existingPage = existing.docs[0]

  if (existingPage) {
    await payload.update({
      collection: 'pages',
      id: existingPage.id,
      data: pageData,
    })
    console.log(`Updated scanner page (ID: ${existingPage.id})`)
  } else {
    const page = await payload.create({
      collection: 'pages',
      data: pageData,
    })
    console.log(`Created scanner page (ID: ${page.id})`)
  }

  process.exit(0)
}

seed().catch((err) => {
  console.error('Seed failed:', err)
  process.exit(1)
})
