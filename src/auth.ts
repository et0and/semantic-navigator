/**
 * GitHub Copilot OAuth authentication.
 *
 * Flow:
 *  1. GitHub device flow → GitHub OAuth token (scope: read:user, copilot)
 *  2. Exchange GitHub OAuth token → Copilot access token (short-lived)
 *  3. Cache Copilot token with expiry under ~/.config/semantic-navigator/
 *
 * References:
 *  - https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#device-flow
 *  - Copilot token endpoint: https://api.github.com/copilot_internal/v2/token
 */

import { createOAuthDeviceAuth } from "@octokit/auth-oauth-device"
import path from "node:path"
import fs from "node:fs"
import os from "node:os"

const GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"

const COPILOT_TOKEN_URL =
  "https://api.github.com/copilot_internal/v2/token"

const TOKEN_CACHE_DIR = path.join(
  os.homedir(),
  ".config",
  "semantic-navigator"
)
const TOKEN_CACHE_FILE = path.join(TOKEN_CACHE_DIR, "auth.json")

interface AuthCache {
  githubToken: string
  copilotToken: string
  expiresAt: number
}

function readCache(): AuthCache | null {
  try {
    const raw = fs.readFileSync(TOKEN_CACHE_FILE, "utf-8")
    return JSON.parse(raw) as AuthCache
  } catch {
    return null
  }
}

function writeCache(cache: AuthCache): void {
  fs.mkdirSync(TOKEN_CACHE_DIR, { recursive: true })
  fs.writeFileSync(TOKEN_CACHE_FILE, JSON.stringify(cache, null, 2), "utf-8")
}

async function acquireGithubToken(
  onVerification: (url: string, code: string) => void
): Promise<string> {
  const auth = createOAuthDeviceAuth({
    clientType: "oauth-app",
    clientId: GITHUB_CLIENT_ID,
    scopes: ["read:user"],
    onVerification: (verification) => {
      onVerification(verification.verification_uri, verification.user_code)
    },
  })

  const result = await auth({ type: "oauth" })
  return result.token
}

interface CopilotTokenResponse {
  token: string
  expires_at: number
}

async function fetchCopilotToken(githubToken: string): Promise<{ token: string; expiresAt: number }> {
  const resp = await fetch(COPILOT_TOKEN_URL, {
    method: "GET",
    headers: {
      Authorization: `token ${githubToken}`,
      "Editor-Version": "vscode/1.95.0",
      "Editor-Plugin-Version": "copilot/1.246.0",
      "User-Agent": "GitHubCopilotChat/0.22.4",
    },
  })

  if (!resp.ok) {
    throw new Error(
      `Failed to fetch Copilot token: HTTP ${resp.status} ${resp.statusText}`
    )
  }

  const data = (await resp.json()) as CopilotTokenResponse
  return { token: data.token, expiresAt: data.expires_at }
}

export async function getCopilotToken(
  onVerification: (url: string, code: string) => void
): Promise<string> {
  const now = Math.floor(Date.now() / 1000)
  const cache = readCache()

  if (cache !== null && cache.expiresAt - now > 60) {
    return cache.copilotToken
  }

  let githubToken = cache?.githubToken ?? null

  if (githubToken === null) {
    githubToken = await acquireGithubToken(onVerification)
  }

  let copilotResult: { token: string; expiresAt: number }
  try {
    copilotResult = await fetchCopilotToken(githubToken)
  } catch (err) {
    // GitHub token may have been revoked — start fresh
    if (cache !== null) {
      githubToken = await acquireGithubToken(onVerification)
      copilotResult = await fetchCopilotToken(githubToken)
    } else {
      throw err
    }
  }

  writeCache({
    githubToken,
    copilotToken: copilotResult.token,
    expiresAt: copilotResult.expiresAt,
  })

  return copilotResult.token
}

export function clearAuthCache(): void {
  try {
    fs.unlinkSync(TOKEN_CACHE_FILE)
  } catch {
    // Already gone :)
  }
}
