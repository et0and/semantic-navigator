/**
 * Copilot-backed labelling provider.
 *
 * Uses the GitHub Copilot chat completions endpoint with JSON schema
 * enforcement (response_format: json_schema) to produce structured labels,
 * mirroring the Python `responses.parse(text_format=Labels)` calls.
 */

import { z } from "zod"
import { getCopilotToken } from "./auth.ts"
import type { EmbedEntry } from "./embed.ts"

// ---------------------------------------------------------------------------
// Zod schemas (mirrors Python Pydantic models)
// ---------------------------------------------------------------------------

const LabelSchema = z.object({
  overarchingTheme: z.string(),
  distinguishingFeature: z.string(),
  label: z.string(),
})

const LabelsSchema = z.object({
  labels: z.array(LabelSchema),
})

export type Label = z.infer<typeof LabelSchema>
export type Labels = z.infer<typeof LabelsSchema>

// ---------------------------------------------------------------------------
// Copilot endpoint constants
// ---------------------------------------------------------------------------
const COPILOT_COMPLETIONS_URL =
  "https://api.githubcopilot.com/chat/completions"

// ---------------------------------------------------------------------------
// Provider config
// ---------------------------------------------------------------------------
export interface CopilotConfig {
  model: string
  /** Passed in from the authentication layer */
  onVerification: (url: string, code: string) => void
}

// ---------------------------------------------------------------------------
// Core completion call
// ---------------------------------------------------------------------------

interface ChatMessage {
  role: "system" | "user" | "assistant"
  content: string
}

async function chatComplete(
  config: CopilotConfig,
  messages: ChatMessage[]
): Promise<string> {
  const token = await getCopilotToken(config.onVerification)

  const body = {
    model: config.model,
    messages,
    temperature: 0,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "Labels",
        strict: true,
        schema: {
          type: "object",
          properties: {
            labels: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  overarchingTheme: { type: "string" },
                  distinguishingFeature: { type: "string" },
                  label: { type: "string" },
                },
                required: ["overarchingTheme", "distinguishingFeature", "label"],
                additionalProperties: false,
              },
            },
          },
          required: ["labels"],
          additionalProperties: false,
        },
      },
    },
  }

  let lastError: unknown
  for (let attempt = 0; attempt < 3; attempt++) {
    const resp = await fetch(COPILOT_COMPLETIONS_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "Editor-Version": "vscode/1.95.0",
        "Editor-Plugin-Version": "copilot/1.246.0",
        "User-Agent": "GitHubCopilotChat/0.22.4",
        "Openai-Intent": "conversation-panel",
      },
      body: JSON.stringify(body),
    })

    if (!resp.ok) {
      const errBody = await resp.text().catch(() => "")
      lastError = new Error(`Copilot API error: HTTP ${resp.status} ${resp.statusText} — ${errBody}`)
      // Retry on 5xx
      if (resp.status >= 500) {
        await Bun.sleep(1000 * (attempt + 1))
        continue
      }
      throw lastError
    }

    const data = (await resp.json()) as {
      choices: Array<{ message: { content: string } }>
    }
    const content = data.choices[0]?.message?.content
    if (content === undefined) throw new Error("Empty Copilot response")
    return content
  }

  throw lastError
}

// ---------------------------------------------------------------------------
// Parse + validate with fallback
// ---------------------------------------------------------------------------

function parseLabels(raw: string): Labels {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error(`Copilot returned non-JSON: ${raw.slice(0, 200)}`)
  }
  return LabelsSchema.parse(parsed)
}

// ---------------------------------------------------------------------------
// Public labelling API
// ---------------------------------------------------------------------------

/**
 * Label individual files within a leaf cluster.
 * Mirrors: `responses.parse(model, input="Label each file in 3 to 7 words...", text_format=Labels)`
 */
export async function labelFiles(
  config: CopilotConfig,
  entries: EmbedEntry[]
): Promise<Label[]> {
  const renderedEntries = entries
    .map((e) => `# File: ${e.path}\n\n${e.text}`)
    .join("\n\n")

  const prompt =
    `Label each file in 3 to 7 words. Don't include file path/names in descriptions.\n\n` +
    renderedEntries

  const raw = await chatComplete(config, [
    {
      role: "system",
      content:
        "You label source code files and documents with brief, descriptive phrases. " +
        "Respond only with valid JSON matching the provided schema.",
    },
    { role: "user", content: prompt },
  ])

  const result = parseLabels(raw)
  return result.labels
}

/**
 * Label clusters from their child tree labels.
 * Mirrors: `responses.parse(model, input="Label each cluster in 2 words...", text_format=Labels)`
 */
export async function labelClusters(
  config: CopilotConfig,
  clusterLabels: string[][]
): Promise<Label[]> {
  const rendered = clusterLabels
    .map((labels) => `# Cluster\n\n${labels.join("\n")}`)
    .join("\n\n")

  const prompt =
    `Label each cluster in 2 words. Don't include file path/names in labels.\n\n` +
    rendered

  const raw = await chatComplete(config, [
    {
      role: "system",
      content:
        "You label clusters of source code files with very short, descriptive phrases. " +
        "Respond only with valid JSON matching the provided schema.",
    },
    { role: "user", content: prompt },
  ])

  const result = parseLabels(raw)
  return result.labels
}
