"use client";

import {
  fetchModels,
  OllamaModel,
} from "@/services/models-service";
import {
  Alert,
  Autocomplete,
  CircularProgress,
  TextField,
} from "@mui/material";
import { useEffect, useState } from "react";

interface ModelSelectorProps {
  value: string;
  onChange: (model: string) => void;
  disabled?: boolean;
}

function formatSize(bytes: number): string {
  const gb = bytes / 1e9;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(bytes / 1e6).toFixed(0)} MB`;
}

export default function ModelSelector({
  value,
  onChange,
  disabled,
}: ModelSelectorProps) {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels()
      .then((models) => {
        setModels(models);
        if (models.length > 0 && !value) {
          onChange(models[0].name);
        }
      })
      .catch(() => {
        setError(
          "Could not connect to Ollama. Make sure it is running with `ollama serve`.",
        );
      })
      .finally(() => {
        setLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
        Ollama Model
      </label>

      {error && (
        <Alert severity="warning" sx={{ mb: 1 }}>
          {error}
        </Alert>
      )}

      <Autocomplete
        freeSolo
        disabled={disabled || loading}
        options={models.map((m) => m.name)}
        value={value}
        onChange={(_e, newValue) => onChange(newValue ?? "")}
        onInputChange={(_e, newValue) => onChange(newValue ?? "")}
        getOptionLabel={(option) => option}
        renderOption={(props, option) => {
          const model = models.find((m) => m.name === option);
          return (
            <li {...props} key={option}>
              {option}
              {model && (
                <span
                  key={`model-${model.name}`}
                  style={{
                    marginLeft: "auto",
                    paddingLeft: 16,
                    fontSize: "0.8rem",
                    color: "#999",
                  }}
                >
                  {model.parameterSize} · {formatSize(model.size)}
                </span>
              )}
            </li>
          );
        }}
        renderInput={(params) => (
          <TextField
            {...params}
            placeholder={
              loading ? "Loading models…" : "Select or type a model name"
            }
            size="small"
            slotProps={{
              ...params.slotProps,
              input: {
                ...params.slotProps.input,
                endAdornment: (
                  <>
                    {loading && <CircularProgress size={18} />}
                    {params.slotProps.input.endAdornment}
                  </>
                ),
              },
            }}
          />
        )}
      />
    </>
  );
}
