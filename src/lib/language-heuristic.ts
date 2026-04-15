/**
 * Lightweight sentence-level language detection using common word lists.
 * Returns the percentage of sentences that appear to be in English
 * rather than the expected target language.
 */

const MIN_SENTENCE_LENGTH = 10;
const MIN_WORD_LENGTH = 2;

// High-frequency English function words — these rarely appear in other languages.
// Deliberately excludes cognates and loanwords.
export const ENGLISH_MARKERS = new Set([
  "the",
  "and",
  "that",
  "have",
  "for",
  "not",
  "with",
  "you",
  "this",
  "but",
  "his",
  "from",
  "they",
  "been",
  "said",
  "each",
  "which",
  "their",
  "will",
  "other",
  "about",
  "many",
  "then",
  "them",
  "these",
  "some",
  "would",
  "into",
  "than",
  "its",
  "over",
  "such",
  "after",
  "should",
  "also",
  "most",
  "could",
  "where",
  "just",
  "those",
  "before",
  "between",
  "does",
  "through",
  "while",
  "being",
  "when",
  "what",
  "there",
  "how",
  "were",
  "are",
  "was",
  "can",
  "had",
  "here",
  "more",
  "why",
  "did",
  "get",
  "has",
  "our",
  "out",
  "who",
  "may",
  "she",
  "her",
  "him",
  "own",
  "any",
  "without",
  "whether",
  "every",
  "because",
  "customers",
  "customer",
  "help",
  "like",
  "make",
  "only",
  "even",
  "including",
  "across",
  "need",
  "still",
  "better",
  "within",
  "ensure",
  "understand",
  "often",
  "reduce",
  "improve",
  "deliver",
  "support",
  "experience",
  "experiences",
  "service",
  "services",
]);

// Minimum ratio of English marker words in a sentence to classify it as English.
const ENGLISH_THRESHOLD = 0.15;

function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?\n])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length >= MIN_SENTENCE_LENGTH);
}

function tokenize(sentence: string): string[] {
  return sentence
    .toLowerCase()
    .replace(/[^a-záàâãéèêíïóôõúüçñ\s'-]/gi, " ")
    .split(/\s+/)
    .filter((w) => w.length >= MIN_WORD_LENGTH);
}

export interface HeuristicResult {
  untranslatedPercent: number;
  totalSentences: number;
  englishSentences: number;
  flaggedSentences: { text: string; englishWords: string[] }[];
}

export function detectLanguageHeuristic(text: string): HeuristicResult {
  const sentences = splitSentences(text);
  if (sentences.length === 0) {
    return {
      untranslatedPercent: 0,
      totalSentences: 0,
      englishSentences: 0,
      flaggedSentences: [],
    };
  }

  const flaggedSentences: { text: string; englishWords: string[] }[] = [];

  for (const sentence of sentences) {
    const words = tokenize(sentence);
    if (words.length < 3) continue;

    const englishWords = words.filter((w) => ENGLISH_MARKERS.has(w));
    if (englishWords.length / words.length >= ENGLISH_THRESHOLD) {
      const uniqueWords = [...new Set(englishWords)];
      flaggedSentences.push({ text: sentence, englishWords: uniqueWords });
    }
  }

  const untranslatedPercent = Math.round(
    (flaggedSentences.length / sentences.length) * 100,
  );

  return {
    untranslatedPercent,
    totalSentences: sentences.length,
    englishSentences: flaggedSentences.length,
    flaggedSentences,
  };
}
