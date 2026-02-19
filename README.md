# ovid

Semantic project navigator for local repos. Builds a labeled tree of files by meaning and renders it in a TUI.

Port of the original Python project by [Gabriella Gonzalez](https://github.com/Gabriella439/semantic-navigator)

Key differences in this port:

- TypeScript + Bun CLI, OpenTUI renderer
- Local embeddings via `@huggingface/transformers` (`Xenova/all-MiniLM-L6-v2`)
- Labels generated through GitHub Copilot device flow (no OpenAI API key)
- Two-layer cache under `~/.cache/semantic-navigator`

**What it does**

- Reads tracked files (git) or top-level files (non-git)
- Embeds file chunks locally with `Xenova/all-MiniLM-L6-v2`
- Clusters files with bisecting k-means
- Labels clusters and leaves via GitHub Copilot
- Renders an interactive tree (OpenTUI)

**How it works**

- Discover files → chunk text → embed → cluster → label → display
- Embeddings are cached in `~/.cache/semantic-navigator/embeddings.json`
- Trees are cached in `~/.cache/semantic-navigator/trees/<fingerprint>.json`

**Usage**

- `bun run src/main.ts [directory]`
- `bun run src/main.ts --help`

**Notes**

- First run prompts for GitHub device flow to access Copilot.
