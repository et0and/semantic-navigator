/**
 * Two-layer result cache for semantic-navigator.
 *
 * Layer 1 — Embedding cache (~/.cache/semantic-navigator/embeddings.json)
 *   Content-addressed: maps sha256(text)[0:16] → number[]
 *   Per-entry granularity: only re-embed chunks whose text changed.
 *
 * Layer 2 — Tree cache (~/.cache/semantic-navigator/trees/<fingerprint>.json)
 *   Keyed by sha256(model + sorted(path:contentHash pairs)).
 *   A single changed file invalidates the whole tree, forcing a fresh
 *   cluster + label run, but embeddings are still reused from layer 1.
 */

import { createHash } from "node:crypto"
import { mkdirSync, existsSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { homedir } from "node:os"
import type { Tree } from "./tree.ts"

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const CACHE_DIR = join(homedir(), ".cache", "semantic-navigator")
const EMBED_CACHE_PATH = join(CACHE_DIR, "embeddings.json")
const TREES_DIR = join(CACHE_DIR, "trees")

function ensureDirs(): void {
  mkdirSync(CACHE_DIR, { recursive: true })
  mkdirSync(TREES_DIR, { recursive: true })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Short hex digest: sha256(text).slice(0, 16) */
export function textHash(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 16)
}

/**
 * Fingerprint for the tree cache: sha256(model + sorted path:contentHash pairs).
 * `fileHashes` is a map from relative path to sha256(file content).
 */
export function treeFingerprint(model: string, fileHashes: Map<string, string>): string {
  const entries = Array.from(fileHashes.entries())
    .map(([p, h]) => `${p}:${h}`)
    .sort()
    .join("\n")
  return createHash("sha256").update(model + "\n" + entries).digest("hex")
}

// ---------------------------------------------------------------------------
// Layer 1: Embedding cache
// ---------------------------------------------------------------------------

type EmbedCacheMap = Record<string, number[]>

let _embedCache: EmbedCacheMap | null = null

function loadEmbedCache(): EmbedCacheMap {
  if (_embedCache !== null) return _embedCache
  if (existsSync(EMBED_CACHE_PATH)) {
    try {
      _embedCache = JSON.parse(readFileSync(EMBED_CACHE_PATH, "utf-8")) as EmbedCacheMap
    } catch {
      _embedCache = {}
    }
  } else {
    _embedCache = {}
  }
  return _embedCache
}

/** Look up a cached embedding by text content. Returns null on miss. */
export function getCachedEmbedding(text: string): Float32Array | null {
  const cache = loadEmbedCache()
  const key = textHash(text)
  const vec = cache[key]
  if (vec === undefined) return null
  return Float32Array.from(vec)
}

/** Store an embedding in the in-memory cache (call flushEmbedCache to persist). */
export function setCachedEmbedding(text: string, embedding: Float32Array): void {
  const cache = loadEmbedCache()
  const key = textHash(text)
  cache[key] = Array.from(embedding)
}

/** Persist the in-memory embedding cache to disk. */
export function flushEmbedCache(): void {
  if (_embedCache === null) return
  ensureDirs()
  writeFileSync(EMBED_CACHE_PATH, JSON.stringify(_embedCache), "utf-8")
}

// ---------------------------------------------------------------------------
// Layer 2: Tree cache
// ---------------------------------------------------------------------------

/** Look up a cached Tree by fingerprint. Returns null on miss. */
export function getCachedTree(fingerprint: string): Tree | null {
  const treePath = join(TREES_DIR, `${fingerprint}.json`)
  if (!existsSync(treePath)) return null
  try {
    return JSON.parse(readFileSync(treePath, "utf-8")) as Tree
  } catch {
    return null
  }
}

/** Persist a Tree to the tree cache. */
export function setCachedTree(fingerprint: string, tree: Tree): void {
  ensureDirs()
  const treePath = join(TREES_DIR, `${fingerprint}.json`)
  writeFileSync(treePath, JSON.stringify(tree), "utf-8")
}
