---
description: 'i18n: Scan a URL for untranslated content using the i18n Content Checker MCP server. Returns a report per locale.'
argument-hint: 'URL to scan, e.g. https://www.example.com/pricing'
tools: ['i18n-content-checker/*']
---

Scan the provided URL for untranslated (English) content.

## Steps

1. Use the `scan_pages` MCP tool with the URL from the user's argument
2. If no locales were specified, scan all available locales (use `list_locales` to show which ones)
3. Present the results clearly:
   - For each locale, show the status (clean / % untranslated / error)
   - If examples were found, list them with the flagged English words
   - Summarize with an overall assessment

## Output Format

Use a table for the summary, then detail any issues found:

| Locale | Status    | Untranslated |
| ------ | --------- | ------------ |
| pt-pt  | ✅ Clean  | 0%           |
| fr-fr  | ⚠️ Issues | 12%          |

Then list examples for locales with issues.
