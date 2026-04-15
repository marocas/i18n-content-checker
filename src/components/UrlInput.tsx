"use client";

import LanguageIcon from "@mui/icons-material/Language";
import { InputAdornment, TextField } from "@mui/material";

interface UrlInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export default function UrlInput({ value, onChange, disabled }: UrlInputProps) {
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
        English URL to scan
      </label>
      <TextField
        fullWidth
        placeholder="https://www.example.com/pricing"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        slotProps={{
          input: {
            startAdornment: (
              <InputAdornment position="start">
                <LanguageIcon sx={{ color: "text.secondary" }} />
              </InputAdornment>
            ),
          },
        }}
        sx={{
          "& .MuiOutlinedInput-root": {
            backgroundColor: "background.paper",
          },
        }}
      />
    </>
  );
}
