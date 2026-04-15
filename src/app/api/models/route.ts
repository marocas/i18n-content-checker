import { NextResponse } from "next/server";
import { Ollama } from "ollama";

export async function GET() {
  const host = process.env.OLLAMA_HOST ?? "http://127.0.0.1:11434";

  try {
    const ollama = new Ollama({ host });
    const { models } = await ollama.list();

    return NextResponse.json({
      models: models.map((m) => ({
        name: m.name,
        size: m.size,
        parameterSize: m.details.parameter_size,
      })),
    });
  } catch {
    return NextResponse.json(
      {
        error:
          "Could not connect to Ollama. Make sure it is running with `ollama serve`.",
        models: [],
      },
      { status: 503 },
    );
  }
}
