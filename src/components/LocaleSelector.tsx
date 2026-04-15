"use client";

import { AVAILABLE_LOCALES } from "@/lib/types";
import { Box, Chip } from "@mui/material";

interface LocaleSelectorProps {
  selectedLocales: string[];
  onChange: (locales: string[]) => void;
  disabled?: boolean;
}

export default function LocaleSelector({
  selectedLocales,
  onChange,
  disabled,
}: LocaleSelectorProps) {
  const toggleLocale = (code: string) => {
    const isSelected = selectedLocales.includes(code);
    if (isSelected) {
      onChange(selectedLocales.filter((l) => l !== code));
    } else {
      onChange([...selectedLocales, code]);
    }
  };

  return (
    <>
      <label
        style={{
          fontSize: "0.75rem",
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "#aaa",
          marginBottom: 8,
          display: "block",
        }}
      >
        Locales to scan
      </label>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
        {AVAILABLE_LOCALES.map((locale) => {
          const isSelected = selectedLocales.includes(locale.code);
          return (
            <Chip
              key={locale.code}
              label={`${locale.flag} ${locale.label}`}
              onClick={() => !disabled && toggleLocale(locale.code)}
              variant={isSelected ? "filled" : "outlined"}
              color={isSelected ? "primary" : "default"}
              disabled={disabled}
              sx={{
                fontWeight: isSelected ? 600 : 400,
                cursor: disabled ? "not-allowed" : "pointer",
              }}
            />
          );
        })}
      </Box>
    </>
  );
}
