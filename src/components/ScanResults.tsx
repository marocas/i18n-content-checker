"use client";

import { AVAILABLE_LOCALES, LocaleScanResult } from "@/lib/types";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ErrorIcon from "@mui/icons-material/Error";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import WarningIcon from "@mui/icons-material/Warning";
import {
  Box,
  Button,
  Chip,
  Collapse,
  IconButton,
  LinearProgress,
  Link,
  Paper,
  Typography,
} from "@mui/material";
import { useState } from "react";

interface ScanResultsProps {
  results: LocaleScanResult[];
}

function StatusChip({ status }: { status: LocaleScanResult["status"] }) {
  switch (status) {
    case "clean":
      return (
        <Chip
          icon={<CheckCircleIcon />}
          label="Clean"
          color="success"
          size="small"
        />
      );
    case "english_found":
      return (
        <Chip
          icon={<WarningIcon />}
          label="English found"
          color="warning"
          size="small"
        />
      );
    case "error":
      return (
        <Chip icon={<ErrorIcon />} label="Error" color="error" size="small" />
      );
  }
}

function highlightEnglishWords(
  text: string,
  englishWords: string[],
): React.ReactNode {
  if (englishWords.length === 0) return text;

  const escapedWords = englishWords.map((w) =>
    w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  );
  const regex = new RegExp(`\\b(${escapedWords.join("|")})\\b`, "gi");
  const parts = text.split(regex);

  return parts.map((part, i) => {
    const isMatch = englishWords.some(
      (w) => w.toLowerCase() === part.toLowerCase(),
    );
    if (isMatch) {
      return (
        <Box
          key={i}
          component="span"
          sx={{
            backgroundColor: "warning.dark",
            color: "warning.contrastText",
            borderRadius: "3px",
            px: 0.5,
            fontWeight: 600,
          }}
        >
          {part}
        </Box>
      );
    }
    return part;
  });
}

function TranslationCoverage({ result }: { result: LocaleScanResult }) {
  if (result.status === "error") return null;

  const translatedPct = 100 - result.untranslatedPercent;

  const color =
    translatedPct === 100
      ? "success"
      : translatedPct >= 80
        ? "warning"
        : "error";

  return (
    <Box sx={{ px: 2, pb: 1.5 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          {result.untranslatedPercent}% untranslated
        </Typography>
        <Typography
          variant="caption"
          sx={{ fontWeight: 600 }}
          color={`${color}.main`}
        >
          {translatedPct}% translated
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={translatedPct}
        color={color}
        sx={{ height: 6, borderRadius: 3 }}
      />
    </Box>
  );
}

function LocaleResultRow({ result }: { result: LocaleScanResult }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const locale = AVAILABLE_LOCALES.find((l) => l.code === result.locale);

  const copyReport = () => {
    const lines = [
      `Locale: ${result.locale} (${result.url})`,
      `Status: ${result.status}`,
      `Untranslated: ${result.untranslatedPercent}%`,
      "",
    ];
    for (const sentence of result.examples) {
      lines.push(`- "${sentence.text}"`);
      lines.push(`  English words: ${sentence.englishWords.join(", ")}`);
    }
    navigator.clipboard.writeText(lines.join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Paper
      variant="outlined"
      sx={{
        overflow: "hidden",
        borderColor:
          result.status === "english_found"
            ? "warning.main"
            : result.status === "error"
              ? "error.main"
              : "divider",
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 2,
          px: 2,
          py: 1.5,
          cursor: "pointer",
          "&:hover": { backgroundColor: "action.hover" },
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Typography sx={{ fontSize: "1.2rem", lineHeight: 1 }}>
          {locale?.flag}
        </Typography>
        <Typography sx={{ fontWeight: 600, minWidth: 50 }}>
          {result.locale}
        </Typography>
        <Link
          href={result.url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          sx={{
            fontSize: "0.85rem",
            color: "text.secondary",
            textDecoration: "none",
            "&:hover": { textDecoration: "underline" },
            display: "flex",
            alignItems: "center",
            gap: 0.5,
          }}
        >
          {result.url} <OpenInNewIcon sx={{ fontSize: 14 }} />
        </Link>
        <Box sx={{ flex: 1 }} />
        <StatusChip status={result.status} />
        {result.examples.length > 0 && (
          <Typography variant="body2" color="text.secondary">
            {result.examples.length} example
            {result.examples.length !== 1 ? "s" : ""}
          </Typography>
        )}
        <IconButton size="small">
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>
      <TranslationCoverage result={result} />
      <Collapse in={expanded}>
        <Box
          sx={{ px: 2, pb: 2, borderTop: "1px solid", borderColor: "divider" }}
        >
          {result.status === "error" && (
            <Typography color="error" sx={{ py: 1 }}>
              {result.errorMessage}
            </Typography>
          )}
          {result.examples.length > 0 && (
            <>
              <Box sx={{ display: "flex", justifyContent: "flex-end", pt: 1 }}>
                <Button
                  size="small"
                  startIcon={<ContentCopyIcon />}
                  onClick={copyReport}
                >
                  {copied ? "Copied!" : "Copy report"}
                </Button>
              </Box>
              {result.examples.map((sentence, i) => (
                <Box
                  key={i}
                  sx={{
                    py: 1,
                    borderBottom:
                      i < result.examples.length - 1 ? "1px solid" : "none",
                    borderColor: "divider",
                  }}
                >
                  <Typography variant="body2" sx={{ lineHeight: 1.8 }}>
                    {highlightEnglishWords(
                      sentence.text,
                      sentence.englishWords,
                    )}
                  </Typography>
                  <Box
                    sx={{
                      display: "flex",
                      gap: 0.5,
                      mt: 0.5,
                      flexWrap: "wrap",
                    }}
                  >
                    {sentence.englishWords.map((word, j) => (
                      <Chip
                        key={`${word}-${j}`}
                        label={word}
                        size="small"
                        color="warning"
                        variant="outlined"
                        sx={{ fontSize: "0.75rem" }}
                      />
                    ))}
                  </Box>
                </Box>
              ))}
            </>
          )}
          {result.status === "clean" && (
            <Typography color="success.main" sx={{ py: 1 }}>
              No English content detected on this page.
            </Typography>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
}

export default function ScanResults({ results }: ScanResultsProps) {
  if (results.length === 0) return null;

  const englishCount = results.filter(
    (r) => r.status === "english_found",
  ).length;
  const errorCount = results.filter((r) => r.status === "error").length;
  const cleanCount = results.filter((r) => r.status === "clean").length;

  return (
    <Box>
      <Box sx={{ display: "flex", gap: 2, mb: 2, alignItems: "center" }}>
        <Typography variant="h6">Results</Typography>
        <Chip
          label={`${cleanCount} clean`}
          color="success"
          size="small"
          variant="outlined"
        />
        {englishCount > 0 && (
          <Chip
            label={`${englishCount} with English`}
            color="warning"
            size="small"
            variant="outlined"
          />
        )}
        {errorCount > 0 && (
          <Chip
            label={`${errorCount} errors`}
            color="error"
            size="small"
            variant="outlined"
          />
        )}
      </Box>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
        {results.map((result) => (
          <LocaleResultRow key={result.locale} result={result} />
        ))}
      </Box>
    </Box>
  );
}
