import { Ollama } from "ollama";
import { detectLanguageHeuristic, HeuristicResult } from "./language-heuristic";
import { DetectionResult, FlaggedSentence } from "./types";

const LLM_TIMEOUT_MS = 120_000;
const MAX_CHUNK_LENGTH = 6_000;
const MAX_EXAMPLES = 5;

const SYSTEM_PROMPT = `You are a translation verification assistant. You receive text from a localized web page and a target language code.

Your task: find up to ${MAX_EXAMPLES} representative sentences that are clearly in English (not translated to the target language).

Rules:
- Only flag sentences whose main content is English, not the target language.
- Cognates and loanwords commonly used in the target language (e.g. "software", "marketing", "online") are acceptable — do not flag these.
- Brand names, product names, and technical acronyms are NOT English — ignore them.
- If the user provides excluded terms, ignore those terms entirely.
- Return ONLY valid JSON in this exact format:

{"examples": [{"text": "exact English sentence found", "englishWords": ["key", "English", "words"]}]}

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

function parseLLMResponse(content: string): FlaggedSentence[] {
  const parsed = JSON.parse(content);

  if (!Array.isArray(parsed.examples)) return [];

  return parsed.examples
    .filter(
      (s: { text?: unknown; englishWords?: unknown }) =>
        typeof s.text === "string" &&
        Array.isArray(s.englishWords) &&
        s.englishWords.length > 0,
    )
    .slice(0, MAX_EXAMPLES)
    .map((s: { text: string; englishWords: string[] }) => ({
      text: s.text,
      englishWords: s.englishWords.filter(
        (w: unknown) => typeof w === "string",
      ),
    }));
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

  // Sample up to 2 chunks spread across the text for diverse examples
  const allChunks = splitIntoChunks(text);
  const sampled =
    allChunks.length <= 2
      ? allChunks
      : [allChunks[0], allChunks[Math.floor(allChunks.length / 2)]];

  const allExamples: FlaggedSentence[] = [];

  for (const chunk of sampled) {
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

    allExamples.push(...parseLLMResponse(response.message.content));
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
): Promise<DetectionResult> {
  // Fast heuristic for the percentage — milliseconds, not minutes
  const heuristic: HeuristicResult = detectLanguageHeuristic(text);

  // If fully translated, skip LLM entirely
  if (heuristic.untranslatedPercent === 0) {
    return { untranslatedPercent: 0, examples: [] };
  }

  // Use LLM only for illustrative examples
  const timeout = setTimeout(() => {}, LLM_TIMEOUT_MS);
  try {
    const examples = await fetchExamplesFromLLM(
      text,
      locale,
      model,
      excludedTerms,
    );
    return {
      untranslatedPercent: heuristic.untranslatedPercent,
      examples,
    };
  } catch {
    // If LLM fails, still return the heuristic percentage
    return {
      untranslatedPercent: heuristic.untranslatedPercent,
      examples: [],
    };
  } finally {
    clearTimeout(timeout);
  }
}
