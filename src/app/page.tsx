"use client";

import ExclusionList from "@/components/ExclusionList";
import LocaleSelector from "@/components/LocaleSelector";
import ModelSelector from "@/components/ModelSelector";
import ScanResults from "@/components/ScanResults";
import ThemeModeToggle from "@/components/ThemeModeToggle";
import UrlInput from "@/components/UrlInput";
import {
  AVAILABLE_LOCALES,
  DEFAULT_EXCLUDED_TERMS,
  LocaleScanResult,
} from "@/lib/types";
import { fetchConfig, saveConfig } from "@/services/config-service";
import { OllamaUnavailableError, scanLocales } from "@/services/scan-service";
import SearchIcon from "@mui/icons-material/Search";
import {
  Alert,
  Box,
  Button,
  Container,
  FormControlLabel,
  LinearProgress,
  Snackbar,
  Switch,
  Tooltip,
  Typography,
} from "@mui/material";
import { useEffect, useRef, useState } from "react";

function useDebouncedSave(value: unknown, key: string, delayMs = 500): void {
  const timer = useRef<ReturnType<typeof setTimeout>>(null);
  const isInitial = useRef(true);

  useEffect(() => {
    if (isInitial.current) {
      isInitial.current = false;
      return;
    }
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      saveConfig({ [key]: value }).catch(() => {});
    }, delayMs);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [value, key, delayMs]);
}

export default function Home() {
  const [urls, setUrls] = useState<string[]>([]);
  const [selectedLocales, setSelectedLocales] = useState(
    AVAILABLE_LOCALES.map((l) => l.code),
  );
  const [excludedTerms, setExcludedTerms] = useState(DEFAULT_EXCLUDED_TERMS);
  const [selectedModel, setSelectedModel] = useState("");
  const [useLLM, setUseLLM] = useState(false);
  const [results, setResults] = useState<LocaleScanResult[]>([]);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [configLoaded, setConfigLoaded] = useState(false);
  const [ollamaToast, setOllamaToast] = useState(false);

  // Load saved config on mount
  useEffect(() => {
    fetchConfig()
      .then((config) => {
        if (config.model) setSelectedModel(config.model);
        if (typeof config.useLLM === "boolean") setUseLLM(config.useLLM);
        if (Array.isArray(config.excludedTerms)) {
          setExcludedTerms(config.excludedTerms);
        }
      })
      .catch(() => {})
      .finally(() => setConfigLoaded(true));
  }, []);

  // Auto-save when model or terms change
  useDebouncedSave(selectedModel, "model");
  useDebouncedSave(excludedTerms, "excludedTerms");
  useDebouncedSave(useLLM, "useLLM");

  const handleScan = async () => {
    if (urls.length === 0) {
      setError("Add at least one URL to scan (press Enter after typing)");
      return;
    }
    if (selectedLocales.length === 0) {
      setError("Select at least one locale to scan");
      return;
    }

    setError(null);
    setScanning(true);
    setResults([]);

    try {
      await scanLocales(
        {
          urls,
          locales: selectedLocales,
          excludedTerms,
          model: selectedModel,
          useLLM,
        },
        (result) => setResults((prev) => [...prev, result]),
      );
    } catch (err) {
      if (err instanceof OllamaUnavailableError) {
        setOllamaToast(true);
      } else {
        setError(err instanceof Error ? err.message : "Scan failed");
      }
    } finally {
      setScanning(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          mb: 4,
        }}
      >
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }} gutterBottom>
            i18n Content Checker
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Scan localized pages for untranslated English content
          </Typography>
        </Box>
        <ThemeModeToggle />
      </Box>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <Box sx={{ display: "flex", gap: 1, alignItems: "flex-end" }}>
          <Box sx={{ flex: 1 }}>
            <UrlInput value={urls} onChange={setUrls} disabled={scanning} />
          </Box>
          <Button
            variant="contained"
            size="large"
            startIcon={<SearchIcon />}
            onClick={handleScan}
            disabled={
              scanning || urls.length === 0 || (useLLM && !selectedModel)
            }
            sx={{ px: 4, height: 56 }}
          >
            {scanning ? "Scanning..." : "Scan pages"}
          </Button>
        </Box>

        <Box>
          <LocaleSelector
            selectedLocales={selectedLocales}
            onChange={setSelectedLocales}
            disabled={scanning}
          />
        </Box>

        <Box>
          <ExclusionList
            terms={excludedTerms}
            onChange={setExcludedTerms}
            disabled={scanning}
          />
        </Box>

        <Box>
          {configLoaded && (
            <ModelSelector
              value={selectedModel}
              onChange={setSelectedModel}
              disabled={scanning}
            />
          )}
        </Box>

        <Box>
          <Tooltip title="Use Ollama LLM to find additional untranslated sentences the heuristic may miss. Slower but more thorough.">
            <FormControlLabel
              control={
                <Switch
                  checked={useLLM}
                  onChange={(e) => setUseLLM(e.target.checked)}
                  disabled={scanning}
                />
              }
              label="Use LLM for deeper analysis"
            />
          </Tooltip>
        </Box>

        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {scanning && <LinearProgress />}

        <Alert severity="info" variant="outlined" sx={{ fontSize: "0.85rem" }}>
          <strong>How this works:</strong> Pages are fetched server-side. The
          scanner extracts visible text, removes HTML, and flags English-looking
          words not present in the target language. Results may vary based on
          page structure and JavaScript rendering.
        </Alert>

        <ScanResults results={results} />
      </Box>

      <Snackbar
        open={ollamaToast}
        autoHideDuration={8000}
        onClose={() => setOllamaToast(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={() => setOllamaToast(false)}
          severity="error"
          variant="filled"
          sx={{ width: "100%" }}
        >
          Could not connect to Ollama. Make sure it is running with{" "}
          <code>ollama serve</code>.
        </Alert>
      </Snackbar>
    </Container>
  );
}
