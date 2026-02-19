import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers"
import type { Chunk } from "./tokenize.ts"

export const DEFAULT_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2"

// Embedding dimension for all-MiniLM-L6-v2
const EMBEDDING_DIM = 384

export interface EmbedEntry {
  /** Path relative to root (same as Chunk.path) */
  path: string
  /** The chunked text that was embedded */
  text: string
  /** L2-normalised embedding vector */
  embedding: Float32Array
}

export interface EmbedOptions {
  model: string
  batchSize: number
  concurrency: number
}

let _pipe: FeatureExtractionPipeline | null = null

async function getEmbedPipeline(model: string): Promise<FeatureExtractionPipeline> {
  if (_pipe === null) {
    _pipe = (await pipeline("feature-extraction", model, {
      // Use float32 for full precision on CPU
      dtype: "fp32",
    }) as unknown) as FeatureExtractionPipeline
  }
  return _pipe
}

/** Mean-pool a raw [seqLen × dim] tensor output into a single [dim] vector. */
function meanPool(data: Float32Array, seqLen: number, dim: number): Float32Array {
  const pooled = new Float32Array(dim)
  for (let t = 0; t < seqLen; t++) {
    for (let d = 0; d < dim; d++) {
      pooled[d] = (pooled[d] ?? 0) + (data[t * dim + d] ?? 0)
    }
  }
  for (let d = 0; d < dim; d++) {
    pooled[d]! /= seqLen
  }
  return pooled
}

/** L2-normalise a vector in place; returns the same array. */
function l2Normalise(v: Float32Array): Float32Array {
  let norm = 0
  for (let i = 0; i < v.length; i++) norm += v[i]! * v[i]!
  norm = Math.sqrt(norm)
  if (norm > 1e-12) {
    for (let i = 0; i < v.length; i++) v[i]! /= norm
  }
  return v
}

async function embedBatch(
  pipe: FeatureExtractionPipeline,
  texts: string[]
): Promise<Float32Array[]> {
  // @huggingface/transformers returns a Tensor with shape [batch, seqLen, dim]
  const output = await pipe(texts, { pooling: "mean", normalize: true })

  // output.data is a flat Float32Array of shape [batch * dim]
  const data = output.data as Float32Array
  const batchSize = texts.length
  const dim = data.length / batchSize

  const results: Float32Array[] = []
  for (let i = 0; i < batchSize; i++) {
    // When pooling + normalize are handled by the pipeline we can slice directly
    const vec = data.slice(i * dim, (i + 1) * dim) as Float32Array
    results.push(vec)
  }
  return results
}

/**
 * Embed all chunks using the local model, with batching + concurrency limits.
 * Calls `onProgress(done, total)` after each batch completes.
 */
export async function embedChunks(
  chunks: Chunk[],
  opts: EmbedOptions,
  onProgress?: (done: number, total: number) => void
): Promise<EmbedEntry[]> {
  if (chunks.length === 0) return []

  const pipe = await getEmbedPipeline(opts.model)

  const batches: Chunk[][] = []
  for (let i = 0; i < chunks.length; i += opts.batchSize) {
    batches.push(chunks.slice(i, i + opts.batchSize))
  }

  const entries: EmbedEntry[] = new Array(chunks.length)
  let chunkIndex = 0
  let done = 0

  for (let i = 0; i < batches.length; i += opts.concurrency) {
    const concurrentBatches = batches.slice(i, i + opts.concurrency)
    const startIndex = chunkIndex

    const batchResults = await Promise.all(
      concurrentBatches.map((batch) =>
        embedBatch(pipe, batch.map((c) => c.text))
      )
    )

    let offset = startIndex
    for (let b = 0; b < concurrentBatches.length; b++) {
      const batch = concurrentBatches[b]!
      const embeddings = batchResults[b]!
      for (let j = 0; j < batch.length; j++) {
        const chunk = batch[j]!
        entries[offset] = {
          path: chunk.path,
          text: chunk.text,
          embedding: embeddings[j]!,
        }
        offset++
      }
      chunkIndex += batch.length
      done += batch.length
      onProgress?.(done, chunks.length)
    }
  }

  return entries
}

/** Compute cosine distance between two L2-normalised vectors: 1 - dot(a,b) */
export function cosineDist(a: Float32Array, b: Float32Array): number {
  let dot = 0
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!
  return 1 - dot
}

export { EMBEDDING_DIM }
