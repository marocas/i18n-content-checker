export interface AppConfig {
  model: string;
  excludedTerms: string[];
  useLLM: boolean;
}

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch("/api/config");
  return res.json();
}

export async function saveConfig(
  partial: Partial<AppConfig>,
): Promise<AppConfig> {
  const res = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(partial),
  });
  return res.json();
}
