/**
 * Tree data structure and labelling pipeline.
 * Direct port of the Python Tree/to_pattern/label_nodes/tree functions.
 */

import { splitCluster, type Cluster } from "./cluster.ts"
import { labelFiles, labelClusters, type CopilotConfig } from "./labels.ts"

// ---------------------------------------------------------------------------
// Tree type
// ---------------------------------------------------------------------------

export interface Tree {
  /** Display label (e.g. "src/components/*.tsx: UI Components") */
  label: string
  /** All leaf file paths beneath this node */
  files: string[]
  /** Child nodes (empty for leaves) */
  children: Tree[]
}

// ---------------------------------------------------------------------------
// Pattern helper (port of Python to_pattern)
// ---------------------------------------------------------------------------

/**
 * Given a list of file paths, return a compact pattern string like
 * `"src/components/*.tsx: "` when all files share a common prefix/suffix,
 * or `""` if there is no meaningful shared pattern.
 */
export function toPattern(files: string[]): string {
  if (files.length === 0) return ""

  // Longest common prefix
  let prefix = files[0]!
  for (const f of files) {
    while (!f.startsWith(prefix)) prefix = prefix.slice(0, -1)
    if (prefix === "") break
  }

  // Longest common suffix (reverse trick from Python)
  const reversed = files.map((f) => f.slice(prefix.length).split("").reverse().join(""))
  let suffix = reversed[0]!
  for (const r of reversed) {
    while (!r.startsWith(suffix)) suffix = suffix.slice(0, -1)
    if (suffix === "") break
  }
  // Re-reverse
  suffix = suffix.split("").reverse().join("")

  const middles = files.map((f) => {
    const core = f.slice(prefix.length)
    return suffix.length > 0 ? core.slice(0, core.length - suffix.length) : core
  })
  const hasStar = middles.some((m) => m.length > 0)
  const star = hasStar ? "*" : ""

  if (prefix) {
    if (suffix) return `${prefix}${star}${suffix}: `
    return `${prefix}${star}: `
  } else {
    if (suffix) return `${star}${suffix}: `
    return ""
  }
}

// ---------------------------------------------------------------------------
// Collect all file paths from a list of Tree nodes
// ---------------------------------------------------------------------------

export function collectFiles(trees: Tree[]): string[] {
  return trees.flatMap((t) => t.files)
}

// ---------------------------------------------------------------------------
// Recursive labelling pipeline (port of label_nodes + tree)
// ---------------------------------------------------------------------------

/**
 * Recursively label a Cluster, returning a list of Tree nodes.
 *
 * @param config    Copilot config
 * @param cluster   Current cluster to process
 * @param depth     Recursion depth (0 = root level; shows progress)
 * @param onStatus  Optional callback for status messages
 */
export async function labelNodes(
  config: CopilotConfig,
  cluster: Cluster,
  depth: number,
  onStatus?: (msg: string) => void
): Promise<Tree[]> {
  const children = splitCluster(cluster)

  if (children.length === 1) {
    // Leaf cluster: label each file individually
    const entries = cluster.entries
    let labels = await labelFiles(config, entries)

    // Guard: align label count with entry count (Copilot may return fewer)
    if (labels.length < entries.length) {
      const missing = entries.length - labels.length
      labels = [
        ...labels,
        ...Array.from({ length: missing }, () => ({
          overarchingTheme: "",
          distinguishingFeature: "",
          label: "unlabelled",
        })),
      ]
    }

    return entries.map((entry, i) => ({
      label: `${entry.path}: ${labels[i]!.label}`,
      files: [entry.path],
      children: [],
    }))
  }

  // Internal node: recurse into each child cluster, then label the clusters
  const childTreeLists = await Promise.all(
    children.map((child) => labelNodes(config, child, depth + 1, onStatus))
  )

  if (depth === 0) {
    onStatus?.(`Labelling ${children.length} clusters…`)
  }

  // Build input to cluster labeller: for each child, pass its leaf labels
  const clusterLabelInputs = childTreeLists.map((trees) =>
    trees.map((t) => t.label)
  )

  let clusterLabels = await labelClusters(config, clusterLabelInputs)

  // Guard: align label count
  if (clusterLabels.length < children.length) {
    const missing = children.length - clusterLabels.length
    clusterLabels = [
      ...clusterLabels,
      ...Array.from({ length: missing }, () => ({
        overarchingTheme: "",
        distinguishingFeature: "",
        label: "unlabelled",
      })),
    ]
  }

  return clusterLabels.map((clusterLabel, i) => {
    const trees = childTreeLists[i]!
    const files = collectFiles(trees)
    const pattern = toPattern(files)
    return {
      label: `${pattern}${clusterLabel.label}`,
      files,
      children: trees,
    }
  })
}

/**
 * Build the root Tree for a directory.
 */
export async function buildTree(
  config: CopilotConfig,
  rootLabel: string,
  cluster: Cluster,
  onStatus?: (msg: string) => void
): Promise<Tree> {
  const children = await labelNodes(config, cluster, 0, onStatus)
  return {
    label: rootLabel,
    files: collectFiles(children),
    children,
  }
}
