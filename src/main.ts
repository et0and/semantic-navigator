/**
 * CLI entry point and pipeline orchestrator for semantic-navigator.
 *
 * Pipeline:
 *   1. Parse CLI flags (commander)
 *   2. Optionally handle --logout
 *   3. Init UI
 *   4. Discover + read files  → progress: "reading"
 *   5. Tokenise + chunk        → (synchronous, fast)
 *   6. Embed chunks            → progress: "embedding"
 *   7. Spectral cluster        → progress: "clustering"
 *   8. Build labelled tree     → progress: "labelling"
 *   9. Hand tree to UI
 */

import { Command } from "commander"
import path from "node:path"

import { discoverFiles, DEFAULT_EXCLUDES, type FsOptions } from "./fs.ts"
import { chunkFile } from "./tokenize.ts"
import { embedChunks, DEFAULT_EMBEDDING_MODEL, type EmbedOptions, type EmbedEntry } from "./embed.ts"
import { splitCluster, type Cluster } from "./cluster.ts"
import { buildTree } from "./tree.ts"
import { clearAuthCache, getCopilotToken } from "./auth.ts"
import { SemanticNavigatorUI, type ProgressState } from "./ui.ts"
import type { CopilotConfig } from "./labels.ts"

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

const program = new Command()
  .name("ovid")
  .description("Browse a repository's files by semantic meaning")
  .argument("[directory]", "Directory to analyse (default: current working directory)", ".")
  .option("--completion-model <model>", "Copilot model to use for labelling", "gpt-5-mini")
  .option("--max-files <n>", "Maximum number of files to index", (v) => parseInt(v, 10), 2000)
  .option("--max-file-bytes <n>", "Skip files larger than this many bytes", (v) => parseInt(v, 10), 1_000_000)
  .option("--exclude-glob <pattern...>", "Glob patterns to exclude (repeatable)", DEFAULT_EXCLUDES)
  .option("--read-concurrency <n>", "Concurrent file reads", (v) => parseInt(v, 10), 64)
  .option("--embed-batch-size <n>", "Chunks per embedding batch", (v) => parseInt(v, 10), 32)
  .option("--embed-concurrency <n>", "Concurrent embedding batches", (v) => parseInt(v, 10), 2)
  .option("--logout", "Clear cached GitHub / Copilot credentials and exit")
  .helpOption("-h, --help", "Show help")

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

let _ui: SemanticNavigatorUI | undefined

async function main(): Promise<void> {
  program.parse(process.argv)

  const opts = program.opts<{
    completionModel: string
    maxFiles: number
    maxFileBytes: number
    excludeGlob: string[]
    readConcurrency: number
    embedBatchSize: number
    embedConcurrency: number
    logout: boolean | undefined
  }>()

  // --- --logout shortcut ---
  if (opts.logout) {
    clearAuthCache()
    console.log("Logged out: cached credentials removed.")
    process.exit(0)
  }

  const directory = path.resolve(program.args[0] ?? ".")

  // ---------------------------------------------------------------------------
  // Init UI first so all progress is rendered in-terminal
  // ---------------------------------------------------------------------------

  const ui = new SemanticNavigatorUI()
  _ui = ui
  await ui.init()

  // We'll wire console errors through the UI destroy → exit pattern
  const fatal = (msg: string): never => {
    ui.destroy()
    console.error(`\nError: ${msg}`)
    process.exit(1)
  }

  // ---------------------------------------------------------------------------
  // Step 1: Auth — preload Copilot token before we start the pipeline so the
  // device-flow prompt appears cleanly before the TUI takes over the terminal.
  // ---------------------------------------------------------------------------

  // The device-flow callback: rendered as a status line in the header while
  // the user completes the browser flow.
  const onVerification = (url: string, code: string): void => {
    ui.updateProgress({
      phase: "reading",
      done: 0,
      total: 0,
      message: `Visit ${url} and enter code: ${code}`,
    })
  }

  // Eagerly authenticate so the token is warm before we need it for labelling.
  // If this throws we want the TUI torn down cleanly.
  let copilotToken: string
  try {
    copilotToken = await getCopilotToken(onVerification)
  } catch (err) {
    fatal(`Authentication failed: ${err instanceof Error ? err.message : String(err)}`)
  }

  const copilotConfig: CopilotConfig = {
    model: opts.completionModel,
    onVerification,
  }

  // ---------------------------------------------------------------------------
  // Step 2: Discover + read files
  // ---------------------------------------------------------------------------

  ui.updateProgress({ phase: "reading", done: 0, total: 0 })

  const fsOpts: FsOptions = {
    maxFileBytes: opts.maxFileBytes,
    excludePatterns: opts.excludeGlob,
    readConcurrency: opts.readConcurrency,
    maxFiles: opts.maxFiles,
  }

  let files: Awaited<ReturnType<typeof discoverFiles>> | undefined
  try {
    files = await discoverFiles(directory, fsOpts, (done, total) => {
      ui.updateProgress({ phase: "reading", done, total })
    })
  } catch (err) {
    fatal(`File discovery failed: ${err instanceof Error ? err.message : String(err)}`)
  }
  // fatal() is typed as never, so execution only reaches here if try succeeded
  const resolvedFiles = files!

  if (resolvedFiles.length === 0) {
    fatal("No files found in the specified directory.")
  }

  // ---------------------------------------------------------------------------
  // Step 3: Tokenise + chunk (synchronous, fast — no progress bar needed)
  // ---------------------------------------------------------------------------

  ui.updateProgress({ phase: "embedding", done: 0, total: resolvedFiles.length })

  const chunks = resolvedFiles
    .map((f) => chunkFile(f.relativePath, f.content))
    .filter((c): c is NonNullable<typeof c> => c !== null)

  if (chunks.length === 0) {
    fatal("No embeddable chunks produced from the discovered files.")
  }

  // ---------------------------------------------------------------------------
  // Step 4: Embed chunks
  // ---------------------------------------------------------------------------

  const embedOpts: EmbedOptions = {
    model: DEFAULT_EMBEDDING_MODEL,
    batchSize: opts.embedBatchSize,
    concurrency: opts.embedConcurrency,
  }

  let embedEntriesRaw: EmbedEntry[] | undefined
  try {
    embedEntriesRaw = await embedChunks(chunks, embedOpts, (done, total) => {
      ui.updateProgress({ phase: "embedding", done, total })
    })
  } catch (err) {
    fatal(`Embedding failed: ${err instanceof Error ? err.message : String(err)}`)
  }
  const embedEntries = embedEntriesRaw!

  if (embedEntries.length === 0) {
    fatal("Embedding produced no results.")
  }

  // ---------------------------------------------------------------------------
  // Step 5: Spectral clustering (CPU-bound, synchronous)
  // ---------------------------------------------------------------------------

  ui.updateProgress({
    phase: "clustering",
    done: 0,
    total: embedEntries.length,
    message: `Clustering ${embedEntries.length} files…`,
  })

  const rootCluster: Cluster = { entries: embedEntries }

  // splitCluster is synchronous; yield to the event loop first so the UI has
  // a chance to render the "Clustering…" status line.
  await Bun.sleep(0)

  // ---------------------------------------------------------------------------
  // Step 6: Build labelled tree
  // ---------------------------------------------------------------------------

  ui.updateProgress({
    phase: "labelling",
    done: 0,
    total: 0,
    message: "Labelling…",
  })

  const rootLabel = path.basename(path.resolve(directory))

  let treeRaw: Awaited<ReturnType<typeof buildTree>> | undefined
  try {
    treeRaw = await buildTree(copilotConfig, rootLabel, rootCluster, (msg) => {
      ui.updateProgress({ phase: "labelling", done: 0, total: 0, message: msg })
    })
  } catch (err) {
    fatal(`Labelling failed: ${err instanceof Error ? err.message : String(err)}`)
  }
  const tree = treeRaw!

  // ---------------------------------------------------------------------------
  // Step 7: Hand the tree to the UI
  // ---------------------------------------------------------------------------

  ui.setTree(tree)

  // The UI event loop keeps the process alive until the user presses q/Esc.
}

main().catch((err) => {
  _ui?.destroy()
  console.error("Unexpected error:", err)
  process.exit(1)
})
