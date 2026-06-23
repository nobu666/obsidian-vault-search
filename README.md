# obsidian-vault-search

Hybrid (keyword + semantic) local search over your Obsidian / Markdown vault — a single ~300-line Python CLI that feeds your AI agent's recall. No plugins, no cloud, no heavy framework. Just the standard library and a local [Ollama](https://ollama.com) embedding model.

```
$ vault-search "that postgres connection pool issue"
daily/2024-11-08.md:42       [emb+kw]  fixed pool exhaustion — bumped max connections and put pgbouncer in front ...
notes/postgres-tuning.md:15  [emb]     connection pooling: prefer transaction mode for short-lived queries ...
projects/api-rewrite.md:88   [kw]      TODO: revisit db pool sizing under load
```

## Why

Keyword search (`grep`/`rg`) is high precision, low recall: it misses notes that say the same thing in different words, and it can't find a dated daily-log file (`メモ/2026-06-23.md`) unless you remember the date. Embedding search fixes recall but drifts on precision. This tool fuses both with [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), so you get semantic recall *and* exact-term precision in one ranked list.

It's built to be called by a coding agent (Claude Code, etc.) at the start of a turn, so the agent can pull relevant past notes into context before answering — but it works fine as a plain CLI too.

## How it differs

- **vs Smart Connections / Copilot (Obsidian plugins):** those live inside the Obsidian GUI. This is a CLI, so an external agent or a shell script can drive it.
- **vs khoj:** khoj is great but heavyweight (a long-running server, many dependencies). This is one file, stdlib only, brute-force cosine — readable in a sitting and trivial to audit.
- **Fully local & private:** the only network call is to your local Ollama. Note contents never leave the machine.

## Requirements

- Python 3.8+ (no third-party packages)
- [Ollama](https://ollama.com) running locally
- An embedding model: `ollama pull bge-m3` (multilingual, 1024-dim; works well for English, Japanese, and mixed notes)

## Install

```bash
git clone https://github.com/nobu666/obsidian-vault-search.git
cd obsidian-vault-search
chmod +x vault-search
ollama pull bge-m3
# optional: put it on your PATH
ln -s "$PWD/vault-search" /usr/local/bin/vault-search
```

## Usage

```bash
# point it at your vault (defaults to ~/Documents/Obsidian/Vault)
export VAULT_DIR="$HOME/path/to/your/vault"

vault-search index            # build / update the index (incremental, by mtime)
vault-search index --rebuild  # wipe and rebuild from scratch
vault-search "your question in natural language"
vault-search "..." -k 12      # return more results (default 8)
vault-search selfcheck        # run internal tests (no Ollama needed)
```

Output is one result per line: `relative/path.md:line  [emb+kw]  snippet`. The tag shows which signal matched — `emb` (semantic), `kw` (keyword), or both.

The first index of a large vault takes a while (embedding is the bottleneck — roughly 1000 files in ~10 minutes). After that, `index` only re-embeds changed files and runs in well under a second.

### Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `VAULT_DIR` | `~/Documents/Obsidian/Vault` | Vault root to index |
| `VAULT_SEARCH_DB` | `~/.cache/vault-search/index.db` | SQLite index location (safe to delete; regenerable) |
| `VAULT_SEARCH_MODEL` | `bge-m3` | Ollama embedding model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |

> If you change the model, run `vault-search index --rebuild` — embeddings of different dimensions can't be compared.

## Use it as your agent's memory recall

The point of the CLI shape is that an agent can call it. For example, with Claude Code, add a rule that tells the agent to refresh the index and search the vault before answering:

```
Before answering, recall from my notes:
1. run: vault-search index          # keep the index fresh (sub-second when nothing changed)
2. run: vault-search "<the user's question, rephrased>"
3. read the top hits, then answer with that context in mind
```

Now the agent surfaces relevant past decisions, daily logs, and knowledge notes on its own — even when your wording today doesn't match what you wrote months ago.

## How it works

1. **Chunk** each `.md` by heading; split overly long sections on paragraph boundaries.
2. **Embed** each chunk via Ollama, store the L2-normalized vector + text in SQLite.
3. **Search** = cosine similarity over all chunks (brute force; ~75 ms for ~3000 chunks) ∪ case-insensitive substring keyword match, fused with RRF.

Incremental indexing compares file mtimes; deleted files are pruned on the next `index`.

## Limitations

- Brute-force cosine is fine into the low tens of thousands of chunks; beyond that you'd want a real vector index.
- Keyword matching is substring-based, so single-character CJK queries are weak — semantic recall covers that gap.
- Line numbers for sub-split long sections point at the section start, not the exact sub-chunk.
- Optimized for recall over precision: it errs toward surfacing a few extra candidates.

## License

[MIT](LICENSE)
