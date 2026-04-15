import { DEFAULT_EXCLUDED_TERMS } from "@/lib/types";
import fs from "fs/promises";
import { NextRequest } from "next/server";
import path from "path";

const CONFIG_PATH = path.join(process.cwd(), "config.json");

interface AppConfig {
  model: string;
  excludedTerms: string[];
  useLLM: boolean;
}

const DEFAULT_CONFIG: AppConfig = {
  model: "",
  excludedTerms: DEFAULT_EXCLUDED_TERMS,
  useLLM: false,
};

async function readConfig(): Promise<AppConfig> {
  try {
    const raw = await fs.readFile(CONFIG_PATH, "utf-8");
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_CONFIG, ...parsed };
  } catch {
    return DEFAULT_CONFIG;
  }
}

async function writeConfig(config: AppConfig): Promise<void> {
  await fs.writeFile(CONFIG_PATH, JSON.stringify(config, null, 2) + "\n");
}

export async function GET(): Promise<Response> {
  const config = await readConfig();
  return Response.json(config);
}

export async function PUT(request: NextRequest): Promise<Response> {
  const body = await request.json();
  const current = await readConfig();

  const updated: AppConfig = {
    model: typeof body.model === "string" ? body.model : current.model,
    excludedTerms: Array.isArray(body.excludedTerms)
      ? body.excludedTerms.filter((t: unknown) => typeof t === "string")
      : current.excludedTerms,
    useLLM: typeof body.useLLM === "boolean" ? body.useLLM : current.useLLM,
  };

  await writeConfig(updated);
  return Response.json(updated);
}
