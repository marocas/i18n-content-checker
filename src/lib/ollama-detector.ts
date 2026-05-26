import { Ollama } from "ollama";
import { detectLanguageHeuristic, HeuristicResult } from "./language-heuristic";
import { DetectionResult, FlaggedSentence } from "./types";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const LLM_TIMEOUT_MS = 120_000;
const MAX_CHUNK_LENGTH = 3_000;
const MAX_EXAMPLES = 5;

const SYSTEM_PROMPT = `You are a translation verification assistant. You receive text scraped from a localized web page and a target language code.

Your task: find up to ${MAX_EXAMPLES} sentences that are **clearly written in English** and were NOT translated to the target language.

Rules:
- A sentence is untranslated ONLY if its grammar and structure are English. A sentence in the target language that contains a few English loanwords is NOT untranslated.
- Cognates and loanwords commonly used in the target language (e.g. "software", "marketing", "online", "engagement", "customer") are acceptable — do NOT flag sentences just because they contain these.
- Brand names, product names, and technical acronyms are NOT English — ignore them.
- If the user provides excluded terms, ignore those terms entirely.
- For "englishWords", list ONLY the distinctly English words — NOT words that also exist in the target language.
- Be conservative: when in doubt, do NOT flag a sentence. Returning an empty list is better than a false positive.
- Return ONLY valid JSON in this exact format:

{"examples": [{"text": "exact English sentence from the input", "englishWords": ["key", "English", "words"]}]}

- "text" must be copied verbatim from the input — do NOT paraphrase or fabricate sentences.
- If no English sentences are found, return: {"examples": []}
- Do not include any explanation or text outside the JSON.`;

function buildUserPrompt(
  text: string,
  locale: string,
  excludedTerms: string[],
): string {
  const exclusions =
    excludedTerms.length > 0
      ? `\nExcluded terms (ignore these): ${excludedTerms.join(", ")}`
      : "";

  return `Target language: ${locale}
Find up to ${MAX_EXAMPLES} sentences that are in English instead of ${locale}.${exclusions}

Text to analyze:
${text}`;
}

function parseLLMResponse(
  content: string,
  sourceText: string,
): FlaggedSentence[] {
  const parsed = JSON.parse(content);

  if (!Array.isArray(parsed.examples)) return [];

  const sourceLower = sourceText.toLowerCase();

  return (
    parsed.examples
      .filter(
        (s: { text?: unknown; englishWords?: unknown }) =>
          typeof s.text === "string" &&
          Array.isArray(s.englishWords) &&
          s.englishWords.length > 0,
      )
      // Drop hallucinated sentences — text must appear in the source
      .filter((s: { text: string }) =>
        sourceLower.includes(s.text.toLowerCase()),
      )
      .slice(0, MAX_EXAMPLES)
      .map((s: { text: string; englishWords: string[] }) => ({
        text: s.text,
        // Trust the LLM's word identification — don't re-filter through the heuristic word list
        englishWords: s.englishWords.filter(
          (w: unknown): w is string => typeof w === "string" && w.length > 0,
        ),
      }))
      .filter((s: { englishWords: string[] }) => s.englishWords.length > 0)
  );
}

function splitIntoChunks(text: string): string[] {
  const sentences = text
    .split(/(?<=[.!?\n])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 3);

  const chunks: string[] = [];
  let current = "";

  for (const sentence of sentences) {
    if (current.length + sentence.length + 1 > MAX_CHUNK_LENGTH && current) {
      chunks.push(current);
      current = sentence;
    } else {
      current = current ? `${current} ${sentence}` : sentence;
    }
  }

  if (current) chunks.push(current);
  return chunks;
}

async function fetchExamplesFromLLM(
  text: string,
  locale: string,
  model: string,
  excludedTerms: string[],
): Promise<FlaggedSentence[]> {
  const host = process.env.OLLAMA_HOST ?? "http://127.0.0.1:11434";
  const ollama = new Ollama({ host });

  // Send all chunks with reduced size to minimise hallucination
  const allChunks = splitIntoChunks(text);

  const allExamples: FlaggedSentence[] = [];

  for (const chunk of allChunks) {
    const response = await ollama.chat({
      model,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: buildUserPrompt(chunk, locale, excludedTerms),
        },
      ],
      format: "json",
      stream: false,
      options: { temperature: 0 },
    });

    allExamples.push(...parseLLMResponse(response.message.content, text));
    if (allExamples.length >= MAX_EXAMPLES) break;
  }

  // Deduplicate and limit
  const seen = new Set<string>();
  const unique: FlaggedSentence[] = [];
  for (const ex of allExamples) {
    if (!seen.has(ex.text)) {
      seen.add(ex.text);
      unique.push(ex);
    }
    if (unique.length >= MAX_EXAMPLES) break;
  }

  return unique;
}

export async function detectEnglishWithLLM(
  text: string,
  locale: string,
  model: string,
  excludedTerms: string[],
  useLLM = false,
): Promise<DetectionResult> {
  // Fast heuristic for the percentage and all flagged sentences
  const heuristic: HeuristicResult = detectLanguageHeuristic(text);

  // Filter excluded terms from displayed englishWords (case-insensitive)
  const singleWordExcluded = new Set<string>();
  const multiWordExcluded: string[][] = [];
  for (const term of excludedTerms) {
    const words = term.toLowerCase().split(/\s+/).filter(Boolean);
    if (words.length === 1) singleWordExcluded.add(words[0]);
    else if (words.length > 1) multiWordExcluded.push(words);
  }

  function filterExcluded(examples: FlaggedSentence[]): FlaggedSentence[] {
    return examples
      .map((s) => {
        const wordsLower = new Set(s.englishWords.map((w) => w.toLowerCase()));
        const multiMatched = new Set<string>();
        for (const termWords of multiWordExcluded) {
          if (termWords.every((tw) => wordsLower.has(tw))) {
            for (const tw of termWords) multiMatched.add(tw);
          }
        }
        return {
          text: s.text,
          englishWords: s.englishWords.filter(
            (w) =>
              !singleWordExcluded.has(w.toLowerCase()) &&
              !multiMatched.has(w.toLowerCase()),
          ),
        };
      })
      .filter((s) => s.englishWords.length > 0);
  }

  function recalculatePercent(filteredExamples: FlaggedSentence[]): number {
    if (heuristic.totalSentences === 0) return 0;
    return Math.round(
      (filteredExamples.length / heuristic.totalSentences) * 100,
    );
  }

  // All heuristic-flagged sentences
  const heuristicExamples: FlaggedSentence[] = heuristic.flaggedSentences.map(
    (s) => ({ text: s.text, englishWords: s.englishWords }),
  );

  if (!useLLM) {
    // Without the LLM, trust the heuristic alone
    if (heuristic.untranslatedPercent === 0) {
      return { untranslatedPercent: 0, examples: [] };
    }
    const filtered = filterExcluded(heuristicExamples);
    return {
      untranslatedPercent: recalculatePercent(filtered),
      examples: filtered,
    };
  }

  // Enrich with LLM — may catch sentences the heuristic missed
  try {
    const llmExamples = await fetchExamplesFromLLM(
      text,
      locale,
      model,
      excludedTerms,
    );

    // Merge: heuristic first, then LLM extras not already found
    const seenTexts = new Set(
      heuristicExamples.map((e) => e.text.toLowerCase()),
    );
    const merged = [...heuristicExamples];
    for (const ex of llmExamples) {
      if (!seenTexts.has(ex.text.toLowerCase())) {
        seenTexts.add(ex.text.toLowerCase());
        merged.push(ex);
      }
    }

    const filteredMerged = filterExcluded(merged);

    const untranslatedPercent =
      filteredMerged.length > 0
        ? Math.max(recalculatePercent(filteredMerged), 1)
        : 0;

    return {
      untranslatedPercent,
      examples: filteredMerged,
    };
  } catch (err) {
    // Re-throw connection errors so callers can show a clear message
    if (isOllamaConnectionError(err)) {
      throw err;
    }
    // Other LLM failures — still return heuristic results
    const filtered = filterExcluded(heuristicExamples);
    return {
      untranslatedPercent: recalculatePercent(filtered),
      examples: filtered,
    };
  }
}

export function isOllamaConnectionError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const msg = err.message.toLowerCase();
  return (
    msg.includes("econnrefused") ||
    msg.includes("fetch failed") ||
    (msg.includes("connect") && msg.includes("refused"))
  );
}
