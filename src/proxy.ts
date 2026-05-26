import { defaultLocale, locales } from '@/i18n/config'
import { NextRequest, NextResponse } from 'next/server'

function getLocaleFromPathname(pathname: string): string | undefined {
  const segments = pathname.split('/')
  return locales.find((locale) => segments[1] === locale)
}

function getPreferredLocale(request: NextRequest): string {
  const acceptLanguage = request.headers.get('accept-language')
  if (!acceptLanguage) return defaultLocale

  const preferred = acceptLanguage
    .split(',')
    .map((lang) => lang.split(';')[0].trim().substring(0, 2))
    .find((lang) => locales.includes(lang as (typeof locales)[number]))

  return preferred || defaultLocale
}

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Skip admin, api, _next, and static files
  if (
    pathname.startsWith('/admin') ||
    pathname.startsWith('/api') ||
    pathname.startsWith('/next') ||
    pathname.startsWith('/_next') ||
    pathname.startsWith('/favicon') ||
    pathname.includes('.')
  ) {
    return NextResponse.next()
  }

  const pathnameLocale = getLocaleFromPathname(pathname)

  // If locale is already in the URL, continue
  if (pathnameLocale) {
    return NextResponse.next()
  }

  // Redirect to the preferred locale
  const locale = getPreferredLocale(request)
  const newUrl = new URL(`/${locale}${pathname}`, request.url)
  newUrl.search = request.nextUrl.search

  return NextResponse.redirect(newUrl)
}

export const config = {
  matcher: ['/((?!_next|admin|api|next|favicon|.*\\..*).*)'],
}
