/**
 * OpenTUI core UI for semantic-navigator.
 *
 * Renders an interactive, keyboard-navigable tree view using the OpenTUI
 * imperative core API.
 *
 * Controls:
 *   ↑ / k      Move cursor up
 *   ↓ / j      Move cursor down
 *   ← / h      Collapse node
 *   → / l      Expand node
 *   Enter       Toggle expand/collapse
 *   q / Esc     Quit
 *   ?           Toggle help overlay
 */

import {
  createCliRenderer,
  BoxRenderable,
  TextRenderable,
  ScrollBoxRenderable,
  type CliRenderer,
} from "@opentui/core"

import type { Tree } from "./tree.ts"

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
const THEME = {
  bg:           "#1a1b26",
  headerBg:     "#24283b",
  footerBg:     "#24283b",
  border:       "#414868",
  text:         "#c0caf5",
  dimText:      "#565f89",
  cursorBg:     "#2d3f76",
  cursorText:   "#7dcfff",
  countColor:   "#e0af68",
  expandColor:  "#9ece6a",
  patternColor: "#7aa2f7",
  statusColor:  "#bb9af7",
}

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

// ---------------------------------------------------------------------------
// Flat node model for scroll-based rendering
// ---------------------------------------------------------------------------

interface FlatNode {
  tree: Tree
  depth: number
  isExpanded: boolean
  isLeaf: boolean
}

function buildFlatList(
  tree: Tree,
  depth: number,
  expanded: Set<string>,
  out: FlatNode[] = []
): FlatNode[] {
  const isLeaf = tree.children.length === 0
  const isExpanded = !isLeaf && expanded.has(nodeKey(tree, depth))
  out.push({ tree, depth, isExpanded, isLeaf })
  if (isExpanded) {
    for (const child of tree.children) {
      buildFlatList(child, depth + 1, expanded, out)
    }
  }
  return out
}

function nodeKey(tree: Tree, depth: number): string {
  return `${depth}::${tree.label}`
}

// ---------------------------------------------------------------------------
// Progress / status UI shown before tree is ready
// ---------------------------------------------------------------------------

export interface ProgressState {
  phase: "reading" | "embedding" | "clustering" | "labelling" | "working" | "done"
  done: number
  total: number
  message?: string
}

// ---------------------------------------------------------------------------
// Main UI class
// ---------------------------------------------------------------------------

export class SemanticNavigatorUI {
  private renderer!: CliRenderer
  private rootBox!: BoxRenderable
  private headerText!: TextRenderable
  private scrollBox!: ScrollBoxRenderable
  private footerText!: TextRenderable

  // Tree state
  private tree: Tree | null = null
  private flatNodes: FlatNode[] = []
  private expanded: Set<string> = new Set()
  private cursorIndex = 0

  // Progress state (shown before tree is ready)
  private progress: ProgressState = { phase: "reading", done: 0, total: 0 }
  private spinnerTimer: ReturnType<typeof setInterval> | null = null
  private spinnerIndex = 0

  // Row renderables (reused by rebuildRows)
  private rowRenderables: TextRenderable[] = []
  private rowIds: string[] = []

  // Help overlay
  private helpVisible = false
  private helpBox!: BoxRenderable

  async init(): Promise<void> {
    this.renderer = await createCliRenderer({
      targetFps: 30,
      exitOnCtrlC: true,
    })

    const W = this.renderer.width
    const H = this.renderer.height

    // Root container
    this.rootBox = new BoxRenderable(this.renderer, {
      id: "root",
      width: W,
      height: H,
      flexDirection: "column",
      backgroundColor: THEME.bg,
    })
    this.renderer.root.add(this.rootBox)

    // Header
    const headerBox = new BoxRenderable(this.renderer, {
      id: "header",
      width: "100%",
      height: 1,
      backgroundColor: THEME.headerBg,
      flexDirection: "row",
      paddingLeft: 1,
      paddingRight: 1,
    })
    this.headerText = new TextRenderable(this.renderer, {
      id: "header-text",
      content: " ovid ",
      fg: THEME.statusColor,
    })
    headerBox.add(this.headerText)
    this.rootBox.add(headerBox)

    // Scroll box for tree rows
    this.scrollBox = new ScrollBoxRenderable(this.renderer, {
      id: "scrollbox",
      width: "100%",
      height: H - 2,
    })
    this.rootBox.add(this.scrollBox)

    // Footer
    const footerBox = new BoxRenderable(this.renderer, {
      id: "footer",
      width: "100%",
      height: 1,
      backgroundColor: THEME.footerBg,
      paddingLeft: 1,
      paddingRight: 1,
    })
    this.footerText = new TextRenderable(this.renderer, {
      id: "footer-text",
      content: "↑↓ navigate  ←→/Enter toggle  q quit  ? help",
      fg: THEME.dimText,
    })
    footerBox.add(this.footerText)
    this.rootBox.add(footerBox)

    // Help overlay (hidden by default)
    this.helpBox = new BoxRenderable(this.renderer, {
      id: "help",
      position: "absolute",
      left: Math.floor(W / 2) - 20,
      top: Math.floor(H / 2) - 7,
      width: 40,
      height: 14,
      border: true,
      borderStyle: "rounded",
      borderColor: THEME.border,
      backgroundColor: THEME.headerBg,
      title: " Help ",
      titleAlignment: "center",
      padding: 1,
      zIndex: 10,
    })
    this.helpBox.visible = false

    const helpLines = [
      ["↑ / k", "Move cursor up"],
      ["↓ / j", "Move cursor down"],
      ["→ / l / Enter", "Expand node"],
      ["← / h / Enter", "Collapse node"],
      ["q / Esc", "Quit"],
      ["?", "Toggle this help"],
    ]
    for (const [keys, desc] of helpLines) {
      const row = new BoxRenderable(this.renderer, {
        flexDirection: "row",
        marginBottom: 0,
      })
      const keyText = new TextRenderable(this.renderer, {
        content: (keys ?? "").padEnd(16),
        fg: THEME.cursorText,
        width: 16,
      })
      const descText = new TextRenderable(this.renderer, {
        content: desc ?? "",
        fg: THEME.text,
      })
      row.add(keyText)
      row.add(descText)
      this.helpBox.add(row)
    }
    this.rootBox.add(this.helpBox)

    // Keyboard handler
    this.renderer.keyInput.on("keypress", (key) => {
      if (this.helpVisible) {
        if (key.name === "escape" || key.name === "?" || key.name === "q") {
          this.toggleHelp()
        }
        return
      }

      switch (key.name) {
        case "up":
        case "k":
          this.moveCursor(-1)
          break
        case "down":
        case "j":
          this.moveCursor(1)
          break
        case "right":
        case "l":
          this.expandCurrent()
          break
        case "left":
        case "h":
          this.collapseCurrent()
          break
        case "return":
        case "enter":
          this.toggleCurrent()
          break
        case "q":
        case "escape":
          this.renderer.destroy()
          process.exit(0)
          break
        case "?":
          this.toggleHelp()
          break
      }
    })

    // Show loading state
    this.renderProgress()

    // Start the render loop — without this, no setTimeout is ever scheduled
    // and the process exits as soon as the async pipeline completes.
    this.renderer.start()
  }

  // ---------------------------------------------------------------------------
  // Progress display (before tree is ready)
  // ---------------------------------------------------------------------------

  updateProgress(state: ProgressState): void {
    this.progress = state
    if (state.phase === "working") {
      this.startSpinner()
    } else {
      this.stopSpinner()
    }
    this.renderProgress()
  }

  private renderProgress(): void {
    const { phase, done, total, message } = this.progress
    const pct = total > 0 ? Math.round((done / total) * 100) : 0

    let phaseLabel: string
    switch (phase) {
      case "reading":    phaseLabel = "Reading files"; break
      case "embedding":  phaseLabel = "Embedding"; break
      case "clustering": phaseLabel = "Clustering"; break
      case "labelling":  phaseLabel = "Labelling"; break
      case "working":    phaseLabel = "Working"; break
      default:           phaseLabel = "Done"; break
    }

    const statusLine = total > 0
      ? `${phaseLabel}: ${done}/${total} (${pct}%)`
      : message ?? phaseLabel

    this.headerText.content = ` ovid ${statusLine}`

    // Clear rows and show a single status line
    this.clearRows()
    const id = "status-line"
    const spinner = phase === "working"
      ? `${SPINNER_FRAMES[this.spinnerIndex % SPINNER_FRAMES.length] ?? "•"} `
      : ""
    const statusText = new TextRenderable(this.renderer, {
      id,
      content: `  ${spinner}${statusLine}…`,
      fg: THEME.statusColor,
    })
    this.scrollBox.add(statusText)
    this.rowRenderables = [statusText]
    this.rowIds = [id]
  }

  private startSpinner(): void {
    if (this.spinnerTimer !== null) return
    this.spinnerTimer = setInterval(() => {
      if (this.progress.phase !== "working") return
      this.spinnerIndex = (this.spinnerIndex + 1) % SPINNER_FRAMES.length
      this.renderProgress()
    }, 80)
  }

  private stopSpinner(): void {
    if (this.spinnerTimer === null) return
    clearInterval(this.spinnerTimer)
    this.spinnerTimer = null
    this.spinnerIndex = 0
  }

  // ---------------------------------------------------------------------------
  // Tree display (after tree is ready)
  // ---------------------------------------------------------------------------

  setTree(tree: Tree): void {
    this.stopSpinner()
    this.tree = tree

    // Auto-expand root and its direct children
    this.expanded.add(nodeKey(tree, 0))

    this.rebuildFlatList()
    this.renderRows()

    this.headerText.content =
      ` ovid ${tree.label} (${tree.files.length} files)`
  }

  private rebuildFlatList(): void {
    if (this.tree === null) return
    this.flatNodes = buildFlatList(this.tree, 0, this.expanded)
  }

  private clearRows(): void {
    for (const id of this.rowIds) {
      this.scrollBox.remove(id)
    }
    for (const r of this.rowRenderables) {
      r.destroy()
    }
    this.rowRenderables = []
    this.rowIds = []
  }

  private renderRows(): void {
    this.clearRows()

    const W = this.renderer.width - 3 // scrollbar takes ~3 cols

    for (let i = 0; i < this.flatNodes.length; i++) {
      const node = this.flatNodes[i]!
      const isCursor = i === this.cursorIndex
      const indent = "  ".repeat(node.depth)

      let chevron: string
      if (node.isLeaf) {
        chevron = "  "
      } else if (node.isExpanded) {
        chevron = "▾ "
      } else {
        chevron = "▸ "
      }

      const rawLabel = node.tree.label
      const colonIdx = rawLabel.lastIndexOf(": ")
      let displayLabel: string
      if (!node.isLeaf && colonIdx > 0) {
        const count = ` (${node.tree.files.length})`
        displayLabel = `${indent}${chevron}${rawLabel}${count}`
      } else {
        displayLabel = `${indent}${chevron}${rawLabel}`
      }

      // Truncate to terminal width
      if (displayLabel.length > W) {
        displayLabel = displayLabel.slice(0, W - 1) + "…"
      }

      const id = `row-${i}`
      const row = new TextRenderable(this.renderer, {
        id,
        content: displayLabel,
        fg: isCursor ? THEME.cursorText : (node.isLeaf ? THEME.text : THEME.expandColor),
        bg: isCursor ? THEME.cursorBg : THEME.bg,
        width: W,
      })
      this.scrollBox.add(row)
      this.rowRenderables.push(row)
      this.rowIds.push(id)
    }
  }

  // ---------------------------------------------------------------------------
  // Navigation
  // ---------------------------------------------------------------------------

  private moveCursor(delta: number): void {
    const newIndex = Math.max(
      0,
      Math.min(this.flatNodes.length - 1, this.cursorIndex + delta)
    )
    if (newIndex === this.cursorIndex) return
    this.cursorIndex = newIndex
    this.renderRows()
    this.scrollToCursor()
  }

  private scrollToCursor(): void {
    this.scrollBox.scrollTo(this.cursorIndex)
  }

  private expandCurrent(): void {
    const node = this.flatNodes[this.cursorIndex]
    if (node === undefined || node.isLeaf) return
    const key = nodeKey(node.tree, node.depth)
    if (this.expanded.has(key)) return
    this.expanded.add(key)
    this.rebuildFlatList()
    this.renderRows()
  }

  private collapseCurrent(): void {
    const node = this.flatNodes[this.cursorIndex]
    if (node === undefined || node.isLeaf) return
    const key = nodeKey(node.tree, node.depth)
    if (!this.expanded.has(key)) return
    this.expanded.delete(key)
    this.rebuildFlatList()
    this.renderRows()
  }

  private toggleCurrent(): void {
    const node = this.flatNodes[this.cursorIndex]
    if (node === undefined || node.isLeaf) return
    const key = nodeKey(node.tree, node.depth)
    if (this.expanded.has(key)) {
      this.expanded.delete(key)
    } else {
      this.expanded.add(key)
    }
    this.rebuildFlatList()
    this.renderRows()
  }

  private toggleHelp(): void {
    this.helpVisible = !this.helpVisible
    this.helpBox.visible = this.helpVisible
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  destroy(): void {
    this.stopSpinner()
    this.renderer.destroy()
  }
}
