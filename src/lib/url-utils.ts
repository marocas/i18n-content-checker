/**
 * Builds localized URLs by inserting the locale code after the domain.
 * Example: https://www.example.com/pricing + "pt-pt" => https://www.example.com/pt-pt/pricing
 */
export function buildLocalizedUrl(englishUrl: string, locale: string): string {
  const url = new URL(englishUrl);
  const pathSegments = url.pathname.split("/").filter(Boolean);
  url.pathname = `/${locale}/${pathSegments.join("/")}`;
  return url.toString();
}
