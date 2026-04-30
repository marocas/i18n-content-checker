"use client";

import LanguageIcon from "@mui/icons-material/Language";
import { Autocomplete, Chip, InputAdornment, TextField } from "@mui/material";
import { useState } from "react";

interface UrlInputProps {
  value: string[];
  onChange: (value: string[]) => void;
  disabled?: boolean;
}

function isValidUrl(value: string): boolean {
  try {
    new URL(value);
    return true;
  } catch {
    return false;
  }
}

export default function UrlInput({ value, onChange, disabled }: UrlInputProps) {
  const [inputValue, setInputValue] = useState("");

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
        English URLs to scan
      </label>
      <Autocomplete
        multiple
        freeSolo
        options={[]}
        value={value}
        inputValue={inputValue}
        disabled={disabled}
        onInputChange={(_e, newValue) => setInputValue(newValue)}
        onChange={(_e, newValue) => {
          const urls = newValue.filter((v) => isValidUrl(v));
          onChange(urls);
          if (newValue.length > urls.length) {
            // Keep invalid text in the input so the user can fix it
            const rejected = newValue.find((v) => !isValidUrl(v));
            if (rejected) setInputValue(rejected);
          }
        }}
        renderTags={(tagValue, getTagProps) =>
          tagValue.map((option, index) => {
            const { key, ...rest } = getTagProps({ index });
            return <Chip key={key} label={option} size="small" {...rest} />;
          })
        }
        renderInput={(params) => (
          <TextField
            {...params}
            placeholder={
              value.length === 0
                ? "https://www.example.com/pricing (press Enter to add)"
                : "Add another URL…"
            }
            slotProps={{
              ...params.slotProps,
              input: {
                ...params.slotProps.input,
                startAdornment: (
                  <>
                    <InputAdornment position="start">
                      <LanguageIcon sx={{ color: "text.secondary" }} />
                    </InputAdornment>
                    {params.slotProps.input.startAdornment}
                  </>
                ),
              },
            }}
            sx={{
              "& .MuiOutlinedInput-root": {
                backgroundColor: "background.paper",
              },
            }}
          />
        )}
      />
    </>
  );
}
