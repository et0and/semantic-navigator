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

/**
 * Recursively split a Cluster into sub-clusters using spectral clustering.
 * Returns [input] when the cluster is small enough to be a leaf.
 */
export function splitCluster(input: Cluster): Cluster[] {
  const N = input.entries.length

  if (N <= MAX_LEAVES) return [input]

  const normalized = normaliseRows(matFromEmbeds(input.entries))

  // --- Precompute all pairwise distances once (O(N²)) ---
  // Each row is sorted ascending so we can slice any k cheaply.
  const allDistances: Array<Array<[number, number]>> = Array.from({ length: N }, () => [])
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (j === i) continue
      allDistances[i]!.push([cosDist(normalized[i]!, normalized[j]!), j])
    }
    allDistances[i]!.sort((a, b) => a[0] - b[0])
  }

  /** Slice sorted rows to get k-NN result for any k in O(N·k). */
  function knnFromPrecomputed(k: number): { distances: Float64Array[]; indices: Int32Array[] } {
    const distances: Float64Array[] = []
    const indices: Int32Array[] = []
    for (let i = 0; i < N; i++) {
      const row = allDistances[i]!.slice(0, k)
      distances.push(Float64Array.from(row.map((x) => x[0])))
      indices.push(Int32Array.from(row.map((x) => x[1])))
    }
    return { distances, indices }
  }

  // --- Adaptive k-NN: find smallest k that gives 1 connected component ---
  const candidateKs: number[] = []
  for (let n = 0; ; n++) {
    const k = Math.round(Math.exp(n))
    if (k >= N) break
    candidateKs.push(k)
  }
  candidateKs.push(Math.floor(N / 2))

  let chosenK = candidateKs[candidateKs.length - 1]!
  let chosenKnnResult: { distances: Float64Array[]; indices: Int32Array[] } | null = null

  for (const k of candidateKs) {
    const knnResult = knnFromPrecomputed(k)
    const nComponents = connectedComponents(knnResult.indices, N)
    if (nComponents === 1) {
      chosenK = k
      chosenKnnResult = knnResult
      break
    }
  }

  if (chosenKnnResult === null) {
    chosenKnnResult = knnFromPrecomputed(chosenK)
  }

  const { distances, indices } = chosenKnnResult

  // --- Build affinity matrix (sparse triplets) ---
  // σ[i] = distance to Kth nearest neighbour
  const sigmas = distances.map((d) => d[d.length - 1]!)

  const sparseAffinity: Array<{ i: number; j: number; v: number }> = []

  for (let i = 0; i < N; i++) {
    for (let n = 0; n < chosenK; n++) {
      const j = indices[i]![n]!
      const d = distances[i]![n]!
      const sigma_i = sigmas[i]!
      const sigma_j = sigmas[j]!
      const denom = Math.max(sigma_i * sigma_j, 1e-12)
      const v = Math.exp(-(d * d) / denom)
      sparseAffinity.push({ i, j, v })
    }
  }

  // --- Sparse normalised Laplacian matvec ---
  // Avoids building an N×N dense matrix; matvec cost is O(N·k_sparse) vs O(N²).
  const { matvec: negLMatvec, dd } = buildNormLaplacianSparseMatvec(sparseAffinity, N)

  const k = Math.min(MAX_CLUSTERS + 1, N)
  const { values: rawValues, vectors } = topKEigenpairs(negLMatvec, N, k)

  // Eigenvalues were of -L; flip sign back to get L eigenvalues
  const eigenvalues = Float64Array.from(rawValues, (v) => -v)

  // Sort by eigenvalue ascending (smallest first), skip index 0
  const sortedIdx = Array.from({ length: k }, (_, i) => i).sort(
    (a, b) => eigenvalues[a]! - eigenvalues[b]!
  )

  const sortedEigenvalues = Float64Array.from(sortedIdx, (i) => eigenvalues[i]!)
  const sortedVectors = sortedIdx.map((i) => vectors[i]!)

  deterministicSignFlip(sortedVectors)

  // --- Eigengap heuristic (skip λ₀ ≈ 0) ---
  // n_clusters = argmax(diff(eigenvalues[1:])) + 2
  let maxGap = -Infinity
  let nClusters = 2
  for (let i = 1; i < sortedEigenvalues.length - 1; i++) {
    const gap = sortedEigenvalues[i + 1]! - sortedEigenvalues[i]!
    if (gap > maxGap) {
      maxGap = gap
      nClusters = i + 1 // 1-indexed + 1 for the off-by-one vs Python
    }
  }
  nClusters = Math.max(2, Math.min(nClusters, MAX_CLUSTERS))

  // --- Spectral embeddings: use eigenvectors 1..nClusters (skip 0) ---
  // Build [N × nClusters] matrix, normalise each row
  const spectralPoints: Float64Array[] = Array.from({ length: N }, () =>
    new Float64Array(nClusters)
  )
  for (let c = 0; c < nClusters; c++) {
    const vec = sortedVectors[c + 1] // skip smallest (index 0)
    if (vec === undefined) break
    for (let i = 0; i < N; i++) {
      // Divide by dd[i] (matches Python `wide_spectral_embeddings = eigenvectors.T / dd`)
      spectralPoints[i]![c] = (vec[i]! / dd[i]!)
    }
  }
  // L2-normalise each row
  for (const row of spectralPoints) {
    let norm = 0
    for (const v of row) norm += v * v
    norm = Math.sqrt(norm)
    if (norm > 1e-12) for (let d = 0; d < row.length; d++) row[d]! /= norm
  }

  // --- K-means ---
  const labels = kmeans(spectralPoints, nClusters)

  // Group entries by cluster label, preserving order
  const groups = new Map<number, EmbedEntry[]>()
  for (let i = 0; i < N; i++) {
    const label = labels[i]!
    if (!groups.has(label)) groups.set(label, [])
    groups.get(label)!.push(input.entries[i]!)
  }

  return Array.from(groups.values()).map((entries) => ({ entries }))
}
