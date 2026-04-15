export interface OllamaModel {
  name: string;
  size: number;
  parameterSize: string;
}

interface ModelsResponse {
  models: OllamaModel[];
  error?: string;
}

export async function fetchModels(): Promise<OllamaModel[]> {
  const res = await fetch("/api/models");
  const data: ModelsResponse = await res.json();

  if (!res.ok) {
    throw new Error(data.error ?? "Failed to fetch models");
  }

  return data.models;
}
