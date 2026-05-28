import { seed } from '@/endpoints/seed'
import config from '@/payload.config'
import 'dotenv/config'
import { getPayload } from 'payload'

const run = async () => {
  const payload = await getPayload({ config })

  try {
    const req = { payload } as any
    await seed({ payload, req })
  } catch (error) {
    // Revalidation errors are expected when running outside Next.js runtime
    if (error instanceof Error && error.message.includes('Invariant')) {
      payload.logger.info('Seed completed (revalidation skipped — not in Next.js runtime)')
    } else {
      console.error('Seed failed:', error)
      process.exit(1)
    }
  }

  process.exit(0)
}

run()
