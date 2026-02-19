import { get_encoding, type Tiktoken } from "tiktoken"

// For all-MiniLM-L6-v2 the real token limit is 256 word-piece tokens, but
// we chunk by *cl100k* tokens here to stay consistent with the Python port
// which chunked by the OpenAI embedding model's tokenizer. The actual BERT
// tokenizer produces more tokens per word, so this is a conservative upper
// bound that still keeps chunks well within the model's real limit.
export const MAX_TOKENS_PER_EMBED = 8192

// How many chunks we send to the embedding model per batch (mirrors Python).
export const MAX_TOKENS_PER_BATCH_EMBED = 300_000

export type TokenEncoding = Tiktoken

let _encoding: Tiktoken | null = null

export function getTokenEncoding(): Tiktoken {
  if (_encoding === null) {
    // cl100k_base is the closest widely-available encoding; used as a
    // conservative proxy for chunking any text content.
    _encoding = get_encoding("cl100k_base")
  }
  return _encoding
}

export interface Chunk {
  /** Path of the originating file (relative to root) */
  path: string
  /**
   * The text that will be embedded — includes a `path:\n\n` prefix followed
   * by (up to) the first MAX_TOKENS_PER_EMBED tokens of the file content.
   */
  text: string
}

/**
 * Split file content into at most one chunk (mirrors Python's [:1] slice).
 * Returns null for empty files.
 */
export function chunkFile(path: string, content: string): Chunk | null {
  const enc = getTokenEncoding()

  const prefix = `${path}:\n\n`
  const prefixTokens = enc.encode(prefix)
  const contentTokens = enc.encode(content)

  const maxContentTokens = MAX_TOKENS_PER_EMBED - prefixTokens.length
  if (maxContentTokens <= 0) return null

  const trimmedContentTokens = contentTokens.slice(0, maxContentTokens)

  // Decode back to text
  const prefixBuf = enc.decode(prefixTokens)
  const contentBuf = enc.decode(trimmedContentTokens)

  const text =
    new TextDecoder().decode(prefixBuf) +
    new TextDecoder().decode(contentBuf)

  if (trimmedContentTokens.length === 0) return null

  return { path, text }
}

/**
 * Partition an array of chunks into batches that stay within the
 * MAX_TOKENS_PER_BATCH_EMBED token budget.
 */
export function batchChunks(chunks: Chunk[]): Chunk[][] {
  const maxPerBatch = Math.floor(MAX_TOKENS_PER_BATCH_EMBED / MAX_TOKENS_PER_EMBED)
  const batches: Chunk[][] = []
  for (let i = 0; i < chunks.length; i += maxPerBatch) {
    batches.push(chunks.slice(i, i + maxPerBatch))
  }
  return batches
}
