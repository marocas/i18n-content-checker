/**
 * Lightweight sentence-level language detection using common word lists.
 * Returns the percentage of sentences that appear to be in English
 * rather than the expected target language.
 */

const MIN_SENTENCE_LENGTH = 10;
const MIN_WORD_LENGTH = 2;

// High-frequency English function words — these rarely appear in other languages.
// Deliberately excludes cognates and loanwords.
const ENGLISH_MARKERS = new Set([
  "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his", "from", "they", "been", "said", "each", "which",
  "their", "will", "other", "about", "many", "then", "them", "these", "some", "would", "into", "than", "its", "over", "such", "after",
  "should", "also", "most", "could", "where", "just", "those", "before", "between", "does", "through", "while", "being", "when",
  "what", "there", "how", "were", "are", "was", "can", "had", "here", "more", "why", "did", "get", "has","our", "out", "who", "may",
  "she", "her", "him", "own", "any", "without", "whether", "every", "because", "customers", "customer","help", "like", "make", "only",
  "even", "including", "across", "need", "still", "better", "within", "ensure", "understand", "often", "reduce", "improve", "deliver",
  "support", "experience", "experiences", "service", "services",
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

function isEnglishSentence(sentence: string): boolean {
  const words = tokenize(sentence);
  if (words.length < 3) return false;

  const englishCount = words.filter((w) => ENGLISH_MARKERS.has(w)).length;
  return englishCount / words.length >= ENGLISH_THRESHOLD;
}

export interface HeuristicResult {
  untranslatedPercent: number;
  totalSentences: number;
  englishSentences: number;
}

export function detectLanguageHeuristic(text: string): HeuristicResult {
  const sentences = splitSentences(text);
  if (sentences.length === 0) {
    return { untranslatedPercent: 0, totalSentences: 0, englishSentences: 0 };
  }

  const englishSentences = sentences.filter(isEnglishSentence).length;
  const untranslatedPercent = Math.round(
    (englishSentences / sentences.length) * 100,
  );

  return {
    untranslatedPercent,
    totalSentences: sentences.length,
    englishSentences,
  };
}
