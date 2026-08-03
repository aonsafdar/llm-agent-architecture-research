# Agent / autonomous research

1. Read **`RESEARCH_BRIEF.md`** first — phase, budget, datasets, autonomous loop, **token efficiency**. Skim **`program.md`** only if needed. For **`autoresearch/notes.md`**, read the **tail** (recent Phase 2 entries), not the full file every time — see `RESEARCH_BRIEF.md` → Token efficiency.
2. **Phase 2 is active:** ~20M params, 5–8 GFLOPs, CIFAR-100 primary (100ep screening), ImageNet when ready. See `RESEARCH_BRIEF.md`.
3. If you run as **the CLI agent in tmux**, MCP tools come from **the CLI agent** config ([`autoresearch/TOOLING.md`](autoresearch/TOOLING.md), [`.mcp.json`](.mcp.json)); **`~/.cursor/mcp.json` does not apply** to that agent.
4. **More context:** optional paper PDF — **`Research-Context/5210_Optimized.pdf`** when you need full research framing.
5. Reference config: **`configs/cpvit_minimal.yaml`** (read-only template).
6. Working copy: **`configs/autoresearch_experiment.yaml`** — edit this.
7. **Budget check:** `python tools/compute_flops_ptflops.py --repo . --yaml configs/autoresearch_experiment.yaml`
8. Submit with **`sbatch`** (see `RESEARCH_BRIEF.md` — do **not** use `submit_and_wait.sh`). A **standalone watcher** (`slurm/job_watcher.sh`) nudges tmux when jobs finish; **do not** poll `squeue` in a loop.
9. **W&B:** optional quick check; do not pull full run history into the session.
10. Logs: **`grep 'Max accuracy'`** on **`logs/cpvit-minimal-<jobid>.out`** — do not dump whole logs.
11. If val **improves**: **commit + `git push`**. Else reset/discard. **Then** next hypothesis. Up to **2 GPUs**: CIFAR + ImageNet in parallel per `RESEARCH_BRIEF.md`.
12. Notes: **`autoresearch/notes.md`** — per hypothesis: direction, literature (2024+ where possible), result. See **`RESEARCH_BRIEF.md` → Token efficiency** for caps (1 search, 1 paper abstract, narrow code reads).

**Human:** Set `CONDA_ENV` in `slurm/cpvit_minimal.slurm`. Prefer a **cheaper/faster model** and shorter **extended thinking** in the CLI agent settings if usage limits bite.
