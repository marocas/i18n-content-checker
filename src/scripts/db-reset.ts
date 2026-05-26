import config from '@payload-config'
import 'dotenv/config'
import { getPayload } from 'payload'

const SCHEMA = 'i18n_scanner'

const payload = await getPayload({ config })

try {
  await payload.db.drizzle.execute(`DROP SCHEMA IF EXISTS ${SCHEMA} CASCADE`)
  await payload.db.drizzle.execute(`CREATE SCHEMA ${SCHEMA}`)
  console.log('Schema reset OK')
} catch (e) {
  console.error(e)
  process.exit(1)
} finally {
  process.exit(0)
}
