import { detectEnglishWithLLM } from "@/lib/ollama-detector";
import { LocaleScanResult, ScanRequest } from "@/lib/types";
import { buildLocalizedUrl } from "@/lib/url-utils";
import * as cheerio from "cheerio";
import { NextRequest } from "next/server";

const FETCH_TIMEOUT_MS = 15_000;

async function fetchPageText(url: string): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) i18n-content-checker/1.0",
        Accept: "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.5",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }

    const html = await response.text();
    const $ = cheerio.load(html);

    // Remove script, style, and metadata elements
    $("script, style, noscript, svg, meta, link, head").remove();

    // Remove navigation chrome — focus on main content only
    $("header, nav, footer").remove();

    // Prefer <main> or <article> content; fall back to full body
    const main = $("main, article, [role='main']").first();
    const root = main.length ? main : $("body");

    // Extract visible text
    return root.text().replace(/\s+/g, " ").trim();
  } finally {
    clearTimeout(timeout);
  }
}

async function scanLocale(
  url: string,
  locale: string,
  model: string,
  excludedTerms: string[],
  useLLM: boolean,
): Promise<LocaleScanResult> {
  const localizedUrl = buildLocalizedUrl(url, locale);

  try {
    const pageText = await fetchPageText(localizedUrl);
    const detection = await detectEnglishWithLLM(
      pageText,
      locale,
      model,
      excludedTerms,
      useLLM,
    );

    return {
      locale,
      url: localizedUrl,
      status: detection.untranslatedPercent > 0 ? "english_found" : "clean",
      untranslatedPercent: detection.untranslatedPercent,
      examples: detection.examples,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return {
      locale,
      url: localizedUrl,
      status: "error",
      untranslatedPercent: 0,
      examples: [],
      errorMessage: message,
    };
  }
}

export async function POST(request: NextRequest): Promise<Response> {
  const body: ScanRequest = await request.json();
  const { url, locales, excludedTerms, model, useLLM } = body;

  if (!url || !locales?.length) {
    return Response.json({ results: [] }, { status: 400 });
  }

  try {
    new URL(url);
  } catch {
    return Response.json({ results: [] }, { status: 400 });
  }

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();

      for (const locale of locales) {
        const result = await scanLocale(url, locale, model, excludedTerms, useLLM);
        controller.enqueue(encoder.encode(JSON.stringify(result) + "\n"));
      }

      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson",
      "Transfer-Encoding": "chunked",
    },
  });
}
