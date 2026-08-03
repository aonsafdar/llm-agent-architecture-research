# Agent tooling setup (literature + repos)

## Which config matters for *you*

| How you run the agent | Where MCP must be configured |
|------------------------|------------------------------|
| **the CLI agent / `agent` CLI in tmux** on the login node (your autoresearch loop) | **[the CLI agent MCP](https://docs.agent.com/en/docs/agent-code/mcp)** — `agent mcp add`, **`~/.agent.json`**, and/or **project `.mcp.json`** in this repo. |
| **Cursor** chat only | **`~/.cursor/mcp.json`** — does **not** apply to the tmux `agent` CLI. |

If the autonomous agent only runs **inside tmux via the CLI agent**, **`~/.cursor/mcp.json` is irrelevant** unless you also use Cursor. Configure MCP for **the CLI agent** on the **same machine** where tmux runs (`agent mcp list`, `/mcp` inside a session).

---

## 1. the CLI agent MCP (primary — tmux on login node)

Official guide: **[Connect the CLI agent to tools via MCP](https://docs.agent.com/en/docs/agent-code/mcp)**.

### Quick adds (HTTP)

```bash
# DeepWiki — GitHub repo wikis / structure
agent mcp add --transport http deepwiki https://<wiki-server>/mcp

# alphaXiv — papers MCP (needs API key; create one in your alphaXiv account)
export ALPHAXIV_API_KEY="your-key-here"   # put in ~/.bashrc or session before agent
agent mcp add --transport http alphaxiv https://<lit-server>/mcp/v1 \
  --header "Authorization: Bearer ${ALPHAXIV_API_KEY}"
```

If `--header` does not expand env vars in your CLI version, paste the key once (avoid committing it) or use **project `.mcp.json`** below with `"Bearer ${ALPHAXIV_API_KEY}"`.

### arXiv MCP (stdio — local `uv`)

Current **arxiv-mcp-server** (0.3.x) needs **Python ≥3.11**; **`dwc-vit`** is often **3.10**, so use **`uv tool run`** from that conda env (see [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server)):

```bash
conda activate /path/to/conda-env
pip install uv
uv tool install arxiv-mcp-server
mkdir -p /path/to/scratch/.arxiv-mcp-server/papers

agent mcp add --transport stdio arxiv-mcp-server -- \
  /path/to/conda-env/bin/uv tool run arxiv-mcp-server \
  --storage-path /path/to/scratch/.arxiv-mcp-server/papers
```

### Hugging Face MCP

Open **[<hub>/settings/mcp](https://<hub>/settings/mcp)** while logged in. If they offer **the CLI agent** instructions, use those. Otherwise use **`agent mcp add-json`** with the JSON they provide (often `type` + `url` + `headers`), or add the equivalent **`agent mcp add --transport http ...`** line. **Do not commit** raw tokens; prefer env vars in `.mcp.json` (`${VAR}` — [supported in project `.mcp.json`](https://docs.agent.com/en/docs/agent-code/mcp#environment-variable-expansion-in-mcpjson)).

### Manage & verify

```bash
agent mcp list
agent mcp get deepwiki
# Inside an interactive the CLI agent session:
#   /mcp
```

---

## 2. Project `.mcp.json` (this repo)

A **project-scoped** [`.mcp.json`](../.mcp.json) at the repo root is checked in so every clone shares the same **server list**; **secrets** stay in environment variables (e.g. `ALPHAXIV_API_KEY`). After pulling, run **`agent mcp reset-project-choices`** once if the agent asks for approval.

**Before starting the loop in tmux:**

```bash
export ALPHAXIV_API_KEY="..."   # required for alphaXiv MCP (401 without it)
export PATH="/path/to/conda-env/bin:$PATH"  # if needed for uv
```

---

## 3. GitHub CLI (`gh`)

Not an MCP server; available to the agent as **shell commands** if `gh` is on `PATH` and **`gh auth login`** was run on the login node.

```bash
gh auth status
```

---

## 4. Optional: Cursor-only MCP

If you **also** use Cursor, configure **`~/.cursor/mcp.json`** separately. Same URLs (DeepWiki, alphaXiv with Bearer header if the UI supports it, etc.) — see Cursor MCP docs. This does **not** wire tools into tmux `agent`.

---

## 5. Verification checklist (the CLI agent loop)

- [ ] On the login node: `agent mcp list` shows **deepwiki**, **alphaxiv**, **arxiv-mcp-server**, and **Hugging Face** (if added).
- [ ] `ALPHAXIV_API_KEY` set when using alphaXiv (endpoint returns **401** without auth).
- [ ] `gh auth status` OK.
- [ ] Agent prompt still points at **`program.md`** + **`autoresearch/notes.md`**.

**Notes discipline:** Literature in `notes.md` = **motivation / context**; **Slurm + metrics** = validation. See **`program.md` § Traceability in `autoresearch/notes.md`**.
