"use client";

import CloseIcon from "@mui/icons-material/Close";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  Box,
  Button,
  Chip,
  Collapse,
  IconButton,
  TextField,
} from "@mui/material";
import { KeyboardEvent, useState } from "react";

interface ExclusionListProps {
  terms: string[];
  onChange: (terms: string[]) => void;
  disabled?: boolean;
}

export default function ExclusionList({
  terms,
  onChange,
  disabled,
}: ExclusionListProps) {
  const [inputValue, setInputValue] = useState("");
  const [expanded, setExpanded] = useState(false);

  const addTerms = () => {
    const newTerms = inputValue
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t.length > 0 && !terms.includes(t));

    if (newTerms.length === 0) return;

    onChange([...terms, ...newTerms]);
    setInputValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addTerms();
    }
  };

  const removeTerm = (term: string) => {
    onChange(terms.filter((t) => t !== term));
  };

  return (
    <>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          mb: 1,
        }}
      >
        <label
          style={{
            fontSize: "0.75rem",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            color: "#aaa",
            display: "block",
          }}
        >
          Excluded terms ({terms.length})
        </label>
        <Box sx={{ flex: 1 }} />
        <IconButton size="small" onClick={() => setExpanded(!expanded)}>
          {expanded ? (
            <ExpandLessIcon fontSize="small" />
          ) : (
            <ExpandMoreIcon fontSize="small" />
          )}
        </IconButton>
      </Box>
      <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
        <TextField
          fullWidth
          size="small"
          placeholder="Add term and press Enter or comma..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          sx={{
            "& .MuiOutlinedInput-root": {
              backgroundColor: "background.paper",
            },
          }}
        />
        <Button
          variant="contained"
          onClick={addTerms}
          disabled={disabled || inputValue.trim().length === 0}
          sx={{ whiteSpace: "nowrap" }}
        >
          Add
        </Button>
      </Box>
      <Collapse in={expanded}>
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
          {terms.map((term) => (
            <Chip
              key={term}
              label={term}
              onDelete={() => removeTerm(term)}
              deleteIcon={<CloseIcon />}
              disabled={disabled}
              variant="outlined"
            />
          ))}
        </Box>
      </Collapse>
    </>
  );
}
