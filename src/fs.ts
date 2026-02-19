import path from "node:path"
import fs from "node:fs"

// Default glob patterns to exclude from file discovery
export const DEFAULT_EXCLUDES = [
  "node_modules",
  ".git",
  "dist",
  "build",
  "target",
  ".next",
  ".nuxt",
  ".output",
  "__pycache__",
  ".cache",
  "coverage",
  ".turbo",
]

export interface FsOptions {
  maxFileBytes: number
  excludePatterns: string[]
  readConcurrency: number
  maxFiles: number
}

export interface FileEntry {
  /** Path relative to the root directory passed to the CLI */
  relativePath: string
  content: string
}

function shouldExclude(relPath: string, excludePatterns: string[]): boolean {
  const parts = relPath.split("/")
  return excludePatterns.some((pat) =>
    parts.some((part) => part === pat || part.startsWith(pat + "/"))
  )
}

/** List paths tracked in the git index, relative to `directory`. */
async function listGitTrackedPaths(directory: string): Promise<string[]> {
  const result = await Bun.spawn(
    ["git", "ls-files", "-z", "--full-name"],
    {
      cwd: directory,
      stdout: "pipe",
      stderr: "pipe",
    }
  )

  const exitCode = await result.exited

  if (exitCode !== 0) {
    throw new Error("git ls-files failed")
  }

  const raw = await new Response(result.stdout).text()
  return raw.split("\0").filter(Boolean)
}

/** Walk a directory non-recursively (top-level files only), like Python fallback. */
function listDirectoryFiles(directory: string): string[] {
  const entries = fs.readdirSync(directory, { withFileTypes: true })
  return entries
    .filter((e) => e.isFile())
    .map((e) => e.name)
}

async function readFile(
  directory: string,
  relativePath: string,
  maxFileBytes: number
): Promise<FileEntry | null> {
  const absolutePath = path.join(directory, relativePath)

  let stat: fs.Stats
  try {
    stat = fs.statSync(absolutePath)
  } catch {
    return null
  }

  // Skip directories (e.g. git submodules or symlinks-to-directories)
  if (stat.isDirectory()) return null

  // Skip very large files
  if (stat.size > maxFileBytes) return null

  const file = Bun.file(absolutePath)

  let bytes: ArrayBuffer
  try {
    bytes = await file.arrayBuffer()
  } catch {
    return null
  }

  // Decode as UTF-8; skip binary files
  let content: string
  try {
    content = new TextDecoder("utf-8", { fatal: true }).decode(bytes)
  } catch {
    return null
  }

  return { relativePath, content }
}

/** Resolve the git root for a given directory, or null if not in a git repo. */
async function resolveGitRoot(directory: string): Promise<string | null> {
  const result = await Bun.spawn(
    ["git", "rev-parse", "--show-toplevel"],
    {
      cwd: directory,
      stdout: "pipe",
      stderr: "pipe",
    }
  )
  const exitCode = await result.exited
  if (exitCode !== 0) return null
  const out = await new Response(result.stdout).text()
  return out.trim()
}

export async function discoverFiles(
  directory: string,
  opts: FsOptions,
  onProgress?: (done: number, total: number) => void
): Promise<FileEntry[]> {
  const absDir = path.resolve(directory)

  // --- Path discovery ---
  let relativePaths: string[]

  const gitRoot = await resolveGitRoot(absDir)

  if (gitRoot !== null) {
    // Get all tracked paths relative to git root, then filter to those under
    // our target directory (handles monorepo sub-directory invocation).
    const allTracked = await listGitTrackedPaths(gitRoot)
    const relToGitRoot = path.relative(gitRoot, absDir)

    relativePaths = allTracked
      .map((p) => {
        // Make path relative to our target directory
        if (relToGitRoot === "") return p
        if (p.startsWith(relToGitRoot + "/")) {
          return p.slice(relToGitRoot.length + 1)
        }
        return null
      })
      .filter((p): p is string => p !== null)
  } else {
    // Non-git: only top-level files (matches Python fallback behaviour)
    relativePaths = listDirectoryFiles(absDir)
  }

  // --- Apply exclude patterns ---
  relativePaths = relativePaths.filter(
    (p) => !shouldExclude(p, opts.excludePatterns)
  )

  // --- Enforce max-files cap ---
  if (relativePaths.length > opts.maxFiles) {
    relativePaths = relativePaths.slice(0, opts.maxFiles)
  }

  const total = relativePaths.length

  // --- Read files with concurrency limit ---
  const results: FileEntry[] = []
  let done = 0

  for (let i = 0; i < total; i += opts.readConcurrency) {
    const batch = relativePaths.slice(i, i + opts.readConcurrency)
    const entries = await Promise.all(
      batch.map((p) => readFile(absDir, p, opts.maxFileBytes))
    )
    for (const entry of entries) {
      if (entry !== null) results.push(entry)
    }
    done += batch.length
    onProgress?.(done, total)
  }

  return results
}
