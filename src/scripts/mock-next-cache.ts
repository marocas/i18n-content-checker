// Mock for next/cache — used when running seed script outside Next.js runtime
export function revalidatePath() {}
export function revalidateTag() {}
export function unstable_cache(fn: Function) {
  return fn
}
