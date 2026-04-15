# i18n Content Checker

A Next.js web application that scans localized web pages for untranslated English content. Enter a URL, select target locales, and the tool will detect English text that may have been left untranslated.

## Supported Locales

- 🇵🇹 Portuguese (PT)
- 🇧🇷 Portuguese (BR)
- 🇮🇹 Italian
- 🇩🇪 German
- 🇫🇷 French
- 🇪🇸 Spanish

## Prerequisites

- [Volta](https://volta.sh/) (recommended for automatic Node.js/pnpm version management)
- Node.js 20.x
- pnpm 10.x
- [Ollama](https://ollama.com/) running locally

### Node.js & pnpm

The easiest way is to install [Volta](https://volta.sh/), which automatically manages Node.js and pnpm versions per project:

```bash
# Install Volta
curl https://get.volta.sh | bash
```

Once installed, Volta reads the pinned versions from `package.json` and downloads them automatically — no manual setup needed.

```bash
# Install Node.js and pnpm
volta install node@20
volta install pnpm@10
```

Verify your setup:

```bash
node -v   # should be v20.x
pnpm -v   # should be v10.x
```

## Ollama Setup

This tool uses [Ollama](https://ollama.com/) to run a local LLM that identifies example untranslated sentences. A fast heuristic handles the translation percentage — the LLM only provides illustrative examples.

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.com/download
```

### 2. Start the Ollama server

```bash
ollama serve

# Or download open Ollama application
```

By default Ollama runs on `http://localhost:11434`. To use a different host, set the `OLLAMA_HOST` environment variable in `.env.local`:

```env
OLLAMA_HOST=http://localhost:11434
```

### 3. Pull a model

```bash
ollama pull gemma3:4b
```

### Recommended Models

| Model                 | Size   | Speed               | Accuracy | Best for                  |
| --------------------- | ------ | ------------------- | -------- | ------------------------- |
| **gemma3:4b**         | 4.3 GB | Fast (~15s/chunk)   | Good     | Daily use, quick scans    |
| **gemma4:e2b**        | 5.1 GB | Medium (~25s/chunk) | Better   | Balanced speed/quality    |
| **gemma4:latest**     | 8 GB   | Slower (~40s/chunk) | Best     | Thorough analysis         |
| **translategemma:4b** | 4.3 GB | Fast                | Good     | Translation-focused tasks |

> **Tip:** Start with `gemma3:4b` for fast iteration. Use a larger model when you need higher-quality examples or are reviewing important pages.

> **Note:** Speed depends on your hardware. Times above are approximate for Apple Silicon (M-series). GPU acceleration is used automatically when available.

### Pulling multiple models

You can pull several models and switch between them in the UI:

```bash
ollama pull gemma3:4b
ollama pull gemma4:e2b
ollama pull gemma4:latest
```

The model selector in the app will list all locally available models.

### How detection works

The tool uses a **hybrid approach**:

1. **Heuristic** (instant) — Analyzes English function-word frequency per sentence to calculate the untranslated percentage. Fast, deterministic, and accurate.
2. **LLM** (seconds) — Samples 1–2 text chunks and asks the model to pick out representative English sentences as examples. Skipped entirely if the heuristic finds 0% untranslated.

This means the percentage you see is always from the heuristic (reliable), while the example sentences come from the LLM (contextual).

## Getting Started

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Scripts

| Command              | Description                               |
| -------------------- | ----------------------------------------- |
| `pnpm dev`           | Start development server                  |
| `pnpm build`         | Build for production                      |
| `pnpm start`         | Start production server                   |
| `pnpm lint`          | Run ESLint                                |
| `pnpm commit`        | Create a conventional commit (Commitizen) |
| `pnpm release`       | Bump version & generate CHANGELOG         |
| `pnpm release:minor` | Release as minor version bump             |
| `pnpm release:major` | Release as major version bump             |
| `pnpm release:first` | Generate the first release                |

## Commit Convention

This project uses [Commitizen](https://github.com/commitizen/cz-cli) with the [Conventional Changelog](https://www.conventionalcommits.org/) format. Use `pnpm commit` instead of `git commit` to get an interactive prompt.

## Releasing

Versioning and changelog generation are handled by [standard-version](https://github.com/conventional-changelog/standard-version):

```bash
# First release
pnpm release:first

# Subsequent releases (auto-detects bump from commits)
pnpm release
```

This will:

1. Bump the version in `package.json`
2. Update `CHANGELOG.md`
3. Create a version commit and git tag
