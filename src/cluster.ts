import type { EmbedEntry } from "./embed.ts"

export const MAX_CLUSTERS = 20
export const MAX_LEAVES = 20

export interface Cluster {
  entries: EmbedEntry[]
}

type Matrix = Float64Array[] 

function matFromEmbeds(entries: EmbedEntry[]): Matrix {
  return entries.map((e) => Float64Array.from(e.embedding))
}

/** L2-normalise each row of a matrix in-place. */
function normaliseRows(m: Matrix): Matrix {
  for (const row of m) {
    let norm = 0
    for (let i = 0; i < row.length; i++) norm += row[i]! * row[i]!
    norm = Math.sqrt(norm)
    if (norm > 1e-12) {
      for (let i = 0; i < row.length; i++) row[i]! /= norm
    }
  }
  return m
}

function cosDist(a: Float64Array, b: Float64Array): number {
  let dot = 0
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!
  return Math.max(0, 1 - dot)
}

/**
 * Brute-force k nearest neighbours (cosine distance).
 * Returns { distances, indices } each of shape [N][k].
 * Acceptable for N ≤ ~10k on a laptop.
 */
function knn(
  normalized: Matrix,
  k: number
): { distances: Float64Array[]; indices: Int32Array[] } {
  const N = normalized.length
  const distances: Float64Array[] = []
  const indices: Int32Array[] = []

  for (let i = 0; i < N; i++) {
    // Compute distances to all other points
    const dists: Array<[number, number]> = []
    for (let j = 0; j < N; j++) {
      if (j === i) continue
      dists.push([cosDist(normalized[i]!, normalized[j]!), j])
    }
    // Partial sort: we only need the k smallest
    dists.sort((a, b) => a[0] - b[0])
    const kNearest = dists.slice(0, k)
    distances.push(Float64Array.from(kNearest.map((x) => x[0])))
    indices.push(Int32Array.from(kNearest.map((x) => x[1])))
  }

  return { distances, indices }
}

/**
 * Count connected components in an undirected k-NN connectivity graph.
 * Uses union-find.
 */
function connectedComponents(indices: Int32Array[], N: number): number {
  const parent = Int32Array.from({ length: N }, (_, i) => i)

  function find(x: number): number {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]!]!
      x = parent[x]!
    }
    return x
  }

  function union(a: number, b: number): void {
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) parent[ra] = rb
  }

  for (let i = 0; i < N; i++) {
    for (const j of indices[i]!) {
      union(i, j)
    }
  }

  const roots = new Set<number>()
  for (let i = 0; i < N; i++) roots.add(find(i))
  return roots.size
}

/**
 * Build the degree diagonal `dd` from sparse affinity triplets, then return
 * a sparse matvec closure for the *negated* normalised Laplacian (-L_norm).
 *
 * L_norm = I - D^{-1/2} A D^{-1/2}
 * -L_norm = D^{-1/2} A D^{-1/2} - I
 *
 * The matvec avoids allocating an N×N dense matrix — at N=2000, k_sparse≈7,
 * this reduces the cost per multiply from O(N²)=4M to O(N·k)=14k ops.
 */
function buildNormLaplacianSparseMatvec(
  sparseAffinity: Array<{ i: number; j: number; v: number }>,
  N: number
): { matvec: (v: Float64Array) => Float64Array; dd: Float64Array } {
  // Accumulate row sums (degree) for normalisation
  const degree = new Float64Array(N)
  for (const { i, j, v } of sparseAffinity) {
    degree[i]! += v
    if (i !== j) degree[j]! += v
  }

  const dd = new Float64Array(N)
  for (let i = 0; i < N; i++) {
    dd[i] = degree[i]! > 1e-12 ? 1 / Math.sqrt(degree[i]!) : 0
  }

  // Pre-compute normalised weights once so the closure stays cheap.
  // w_ij = v * dd[i] * dd[j]  (the off-diagonal contribution to A_norm)
  const normAffinity = sparseAffinity.map(({ i, j, v }) => ({
    i,
    j,
    w: v * dd[i]! * dd[j]!,
  }))

  // Matvec for -L_norm = A_norm - I
  // result[i] = -v[i] + sum_j w_ij * v[j]   (using symmetry)
  const matvecFn = (vec: Float64Array): Float64Array => {
    const out = new Float64Array(N)
    // Start from -I · vec
    for (let i = 0; i < N; i++) out[i] = -vec[i]!
    // Add symmetric A_norm contributions
    for (const { i, j, w } of normAffinity) {
      out[i]! += w * vec[j]!
      if (i !== j) out[j]! += w * vec[i]!
    }
    return out
  }

  return { matvec: matvecFn, dd }
}

/**
 * Symmetric QR algorithm (Francis double-shift, implicit).
 * Returns { values, vectors } where vectors is column-major (vectors[j] = j-th eigenvector).
 *
 * This operates on dense matrices — fine for N ≤ ~500.  For larger N we use
 * a power-iteration / deflation approach to extract only the smallest
 * `maxK` eigenpairs.
 */

/** Dot product of two arrays */
function dot(a: Float64Array, b: Float64Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!
  return s
}

/** Subtract projection: a -= (dot(a,b)/dot(b,b)) * b, in place */
function subtractProjection(a: Float64Array, b: Float64Array): void {
  const scale = dot(a, b) / (dot(b, b) + 1e-15)
  for (let i = 0; i < a.length; i++) a[i]! -= scale * b[i]!
}

/** Normalise a vector in place, return its norm */
function normaliseVec(v: Float64Array): number {
  const n = Math.sqrt(dot(v, v))
  if (n > 1e-12) for (let i = 0; i < v.length; i++) v[i]! /= n
  return n
}

/**
 * Randomised power-iteration with deflation to extract the `k` eigenpairs
 * corresponding to the *smallest* eigenvalues of a symmetric matrix.
 *
 * Instead of a dense matrix, accepts a sparse `matvecFn` closure so that the
 * per-iteration cost is O(N·k_sparse) rather than O(N²).  The closure should
 * implement multiplication by the *negated* Laplacian (-L_norm), whose top
 * eigenvalues correspond to L_norm's bottom ones (matching the Python code
 * which does `laplacian *= -1`).
 */
function topKEigenpairs(
  matvecFn: (v: Float64Array) => Float64Array,
  n: number,
  k: number,
  maxIter = 300,
  tol = 1e-6
): { values: Float64Array<ArrayBuffer>; vectors: Float64Array<ArrayBuffer>[] } {
  const N = n
  const rng = seededRng(42)

  const vectors: Float64Array<ArrayBuffer>[] = []
  const values = new Float64Array(k)

  for (let idx = 0; idx < k; idx++) {
    // Random start
    let v = Float64Array.from({ length: N }, () => rng() * 2 - 1) as Float64Array<ArrayBuffer>
    normaliseVec(v)

    // Deflate against already-found vectors
    for (const u of vectors) subtractProjection(v, u)
    normaliseVec(v)

    let lambda = 0
    for (let iter = 0; iter < maxIter; iter++) {
      const Mv = matvecFn(v) as Float64Array<ArrayBuffer>

      // Deflate
      for (const u of vectors) subtractProjection(Mv, u)

      const newLambda = dot(v, Mv)
      const norm = normaliseVec(Mv)

      if (norm < 1e-14) break

      const diff = Math.abs(newLambda - lambda)
      lambda = newLambda
      v = Mv

      if (iter > 10 && diff < tol) break
    }

    vectors.push(v)
    values[idx] = lambda
  }

  return { values, vectors }
}

/** Deterministic sign flip: each eigenvector's sign is chosen so that the
 *  component with the largest absolute value is positive (matches sklearn). */
function deterministicSignFlip(vectors: Float64Array[]): void {
  for (const v of vectors) {
    let maxAbs = 0
    let maxSign = 1
    for (const x of v) {
      if (Math.abs(x) > maxAbs) {
        maxAbs = Math.abs(x)
        maxSign = x >= 0 ? 1 : -1
      }
    }
    if (maxSign < 0) {
      for (let i = 0; i < v.length; i++) v[i]! *= -1
    }
  }
}

/** Simple seeded pseudo-random number generator (mulberry32). */
function seededRng(seed: number): () => number {
  let s = seed
  return () => {
    s |= 0
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Euclidean distance squared */
function distSq(a: Float64Array, b: Float64Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i]! - b[i]!) ** 2
  return s
}

function kmeans(
  points: Float64Array[],
  k: number,
  maxIter = 300,
  seed = 0
): Int32Array {
  const N = points.length
  if (k >= N) return Int32Array.from({ length: N }, (_, i) => i % k)

  const dim = points[0]!.length
  const rng = seededRng(seed)

  // K-means++ initialisation
  const centroids: Float64Array[] = []
  centroids.push(points[Math.floor(rng() * N)]!)

  for (let c = 1; c < k; c++) {
    const dists = points.map((p) =>
      Math.min(...centroids.map((cent) => distSq(p, cent)))
    )
    const total = dists.reduce((a, b) => a + b, 0)
    let r = rng() * total
    let chosen = 0
    for (let i = 0; i < N; i++) {
      r -= dists[i]!
      if (r <= 0) {
        chosen = i
        break
      }
    }
    centroids.push(Float64Array.from(points[chosen]!))
  }

  const labels = new Int32Array(N)

  for (let iter = 0; iter < maxIter; iter++) {
    // Assignment
    let changed = false
    for (let i = 0; i < N; i++) {
      let bestDist = Infinity
      let bestLabel = 0
      for (let c = 0; c < k; c++) {
        const d = distSq(points[i]!, centroids[c]!)
        if (d < bestDist) {
          bestDist = d
          bestLabel = c
        }
      }
      if (labels[i] !== bestLabel) {
        labels[i] = bestLabel
        changed = true
      }
    }

    if (!changed) break

    // Update centroids
    for (let c = 0; c < k; c++) {
      const newCentroid = new Float64Array(dim)
      let count = 0
      for (let i = 0; i < N; i++) {
        if (labels[i] === c) {
          for (let d = 0; d < dim; d++) newCentroid[d]! += points[i]![d]!
          count++
        }
      }
      if (count > 0) {
        for (let d = 0; d < dim; d++) newCentroid[d]! /= count
        centroids[c] = newCentroid
      }
    }
  }

  return labels
}

const MINI_BATCH_THRESHOLD = 512
const MINI_BATCH_SIZE = 128
const MINI_BATCH_ITERS = 120
const KMEANS_MAX_ITER = 60
const KMEANS_RETRIES = 2
const MINI_BATCH_RETRIES = 2

interface ClusterState {
  entries: EmbedEntry[]
  points: Float64Array[]
}

function countLabels(labels: Int32Array, k: number): Int32Array {
  const counts = new Int32Array(k)
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i]
    if (label !== undefined) counts[label] = (counts[label] ?? 0) + 1
  }
  return counts
}

function nearestCentroid(point: Float64Array, centroids: Float64Array[]): number {
  let best = 0
  let bestDist = Infinity
  for (let c = 0; c < centroids.length; c++) {
    const d = distSq(point, centroids[c]!)
    if (d < bestDist) {
      bestDist = d
      best = c
    }
  }
  return best
}

function assignLabels(points: Float64Array[], centroids: Float64Array[]): Int32Array {
  const labels = new Int32Array(points.length)
  for (let i = 0; i < points.length; i++) {
    labels[i] = nearestCentroid(points[i]!, centroids)
  }
  return labels
}

function initRandomCentroids(
  points: Float64Array[],
  k: number,
  rng: () => number
): Float64Array[] {
  const N = points.length
  const centroids: Float64Array[] = []
  const used = new Set<number>()
  for (let c = 0; c < k; c++) {
    let idx = Math.floor(rng() * N)
    for (let attempts = 0; attempts < 4 && used.has(idx); attempts++) {
      idx = Math.floor(rng() * N)
    }
    used.add(idx)
    centroids.push(Float64Array.from(points[idx]!))
  }
  return centroids
}

function miniBatchKmeans(
  points: Float64Array[],
  k: number,
  rng: () => number,
  opts: { batchSize: number; maxIter: number }
): Int32Array {
  const N = points.length
  if (N === 0) return new Int32Array()

  const dim = points[0]!.length
  const centroids = initRandomCentroids(points, k, rng)
  const counts = new Int32Array(k)
  const batchSize = Math.min(opts.batchSize, N)

  for (let iter = 0; iter < opts.maxIter; iter++) {
    for (let b = 0; b < batchSize; b++) {
      const idx = Math.floor(rng() * N)
      const point = points[idx]!
      const c = nearestCentroid(point, centroids)
      counts[c] = (counts[c] ?? 0) + 1
      const centroid = centroids[c]!
      const eta = 1 / (counts[c] ?? 1)
      for (let d = 0; d < dim; d++) {
        centroid[d]! = centroid[d]! + eta * (point[d]! - centroid[d]!)
      }
    }
  }

  return assignLabels(points, centroids)
}

function splitByProjection(points: Float64Array[], rng: () => number): Int32Array {
  const N = points.length
  const labels = new Int32Array(N)
  if (N <= 1) return labels

  const dim = points[0]!.length
  const a = Math.floor(rng() * N)
  let b = Math.floor(rng() * N)
  if (b === a) b = (a + 1) % N

  const pa = points[a]!
  const pb = points[b]!
  const dir = new Float64Array(dim)
  for (let d = 0; d < dim; d++) dir[d]! = pa[d]! - pb[d]!

  let min = Infinity
  let max = -Infinity
  const proj = new Float64Array(N)
  for (let i = 0; i < N; i++) {
    const p = points[i]!
    let dot = 0
    for (let d = 0; d < dim; d++) dot += p[d]! * dir[d]!
    proj[i] = dot
    if (dot < min) min = dot
    if (dot > max) max = dot
  }

  const threshold = (min + max) / 2
  for (let i = 0; i < N; i++) labels[i] = proj[i]! <= threshold ? 0 : 1

  const counts = countLabels(labels, 2)
  if ((counts[0] ?? 0) === 0 || (counts[1] ?? 0) === 0) {
    const mid = Math.floor(N / 2)
    for (let i = 0; i < N; i++) labels[i] = i < mid ? 0 : 1
  }

  return labels
}

function chooseBisectLabels(points: Float64Array[], rng: () => number): Int32Array {
  const N = points.length
  if (N <= 1) return new Int32Array(N)

  const useMiniBatch = N >= MINI_BATCH_THRESHOLD
  const retries = useMiniBatch ? MINI_BATCH_RETRIES : KMEANS_RETRIES

  for (let attempt = 0; attempt <= retries; attempt++) {
    const seed = Math.floor(rng() * 1_000_000_000)
    const labels = useMiniBatch
      ? miniBatchKmeans(points, 2, seededRng(seed), {
          batchSize: MINI_BATCH_SIZE,
          maxIter: MINI_BATCH_ITERS,
        })
      : kmeans(points, 2, KMEANS_MAX_ITER, seed)

    const counts = countLabels(labels, 2)
    const left = counts[0] ?? 0
    const right = counts[1] ?? 0
    if (left > 0 && right > 0) return labels
  }

  return splitByProjection(points, rng)
}

function bisectCluster(
  cluster: ClusterState,
  rng: () => number
): { left: ClusterState; right: ClusterState } {
  const { entries, points } = cluster
  const N = entries.length
  if (N <= 1) {
    return {
      left: { entries, points },
      right: { entries: [], points: [] },
    }
  }

  let labels = chooseBisectLabels(points, rng)
  let leftEntries: EmbedEntry[] = []
  let rightEntries: EmbedEntry[] = []
  let leftPoints: Float64Array[] = []
  let rightPoints: Float64Array[] = []

  for (let i = 0; i < N; i++) {
    if (labels[i] === 0) {
      leftEntries.push(entries[i]!)
      leftPoints.push(points[i]!)
    } else {
      rightEntries.push(entries[i]!)
      rightPoints.push(points[i]!)
    }
  }

  if (leftEntries.length === 0 || rightEntries.length === 0) {
    const mid = Math.floor(N / 2)
    leftEntries = entries.slice(0, mid)
    rightEntries = entries.slice(mid)
    leftPoints = points.slice(0, mid)
    rightPoints = points.slice(mid)
  }

  return {
    left: { entries: leftEntries, points: leftPoints },
    right: { entries: rightEntries, points: rightPoints },
  }
}

/**
 * Recursively split a Cluster into sub-clusters using bisecting k-means.
 * Returns [input] when the cluster is small enough to be a leaf.
 */
export function splitCluster(input: Cluster): Cluster[] {
  const N = input.entries.length

  if (N <= MAX_LEAVES) return [input]

  const normalized = normaliseRows(matFromEmbeds(input.entries))
  const rng = seededRng(42)

  const work: ClusterState[] = [{ entries: input.entries, points: normalized }]
  const leaves: Cluster[] = []

  while (work.length > 0) {
    const cluster = work.pop()!
    if (cluster.entries.length <= MAX_LEAVES) {
      leaves.push({ entries: cluster.entries })
      continue
    }

    const { left, right } = bisectCluster(cluster, rng)
    if (right.entries.length > 0) work.push(right)
    if (left.entries.length > 0) work.push(left)
  }

  return leaves
}
