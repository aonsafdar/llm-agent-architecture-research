# Autonomous architecture research — channel-primary ViT (channel attention / axis decoupling)

This file is the **human-authored “research program”** for an AI agent, in the spirit of [karpathy/autoresearch](https://github.com/karpathy/autoresearch): propose hypotheses, edit config and (when needed) `models/cls_cvt.py`, run **GPU training only via Slurm**, log to **Weights & Biases**, and **commit + push to GitHub** when validation improves.

**Companion:** [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) — one-page entry path. **Tooling (MCP + `gh`):** [`autoresearch/TOOLING.md`](autoresearch/TOOLING.md). If you run the agent as **the CLI agent in tmux** on the login node, MCP is configured for **the CLI agent** ([docs](https://docs.agent.com/en/docs/agent-code/mcp)), including project [`.mcp.json`](.mcp.json)—not only Cursor.

---

## Research focus

- **Central idea:** **Channel attention** with **axis decoupling** (channels as tokens, spatial dimension as features; dimension swapping in the backbone). Architecture optimization should strengthen this story—better attention geometry, inductive biases, mixing, stability—not unrelated tricks.
- **Heuristics to explore:** stage depth vs width, `DIM_EMBED` / `NUM_HEADS`, QKV projection style (`QKV_PROJ_METHOD`), patch embed (`PATCH_EMBED_METHOD`), kernels / stride / padding, drop-path, small well-scoped changes in `Attention` / `Block` in `cls_cvt.py` (e.g. normalization, gated biases, projection layout)—each tied to a clear hypothesis.

**Paper / extended context (optional):** For notation, motivation, and full research framing beyond this file, read the PDF at **`Research-Context/5210_Optimized.pdf`** (repo root). Example absolute path on this machine: `/path/to/repo/Research-Context/5210_Optimized.pdf`. Use it when a hypothesis should align tightly with the published write-up.

---

## Phase 1 (now): CIFAR-10, compact compute

| Constraint | Target |
|------------|--------|
| **Params** | **≤ ~6M** (stay in the ~5–6M band unless you intentionally document a one-off) |
| **FLOPs** | **≤ ~4 GFLOPs** (fvcore, 224×224 CIFAR-10 pipeline) |
| **Epochs** | **100** per run (default in `configs/cpvit_minimal.yaml`) |
| **Dataset** | CIFAR-10, `IMG_SIZE: 224` |

**Before committing an architecture change**, verify budget on the **experiment** config:

```bash
python tools/compute_flops_ptflops.py --repo . --yaml configs/autoresearch_experiment.yaml
```

Check the printed **Params** and **GFLOPs**. If you exceed the envelope, shrink the hypothesis or justify in the commit message.

## Phase 1b (human-enabled): literature-grounded novelty + harder benchmarks

**When to enter:** Human stops the loop after **~50 hypotheses** (or explicitly switches mode). The agent **resumes** using **`autoresearch/notes.md`** (champion config, last hypothesis id, RESUME block—see below) **plus** this section and **`autoresearch/TOOLING.md`**.

**Central constraint (unchanged):** Ideas must stay aligned with **channel attention / axis decoupling**. New attention mechanisms, tokenization, or residual paths are **in scope** if they **preserve or sharpen** that story—not generic ViT clones unrelated to channel-as-token geometry.

**External grounding (optional tools):** Wire these for **the CLI agent** on the machine where you run **`agent` in tmux** ([`autoresearch/TOOLING.md`](autoresearch/TOOLING.md), [`.mcp.json`](.mcp.json)). `~/.cursor/mcp.json` does **not** affect the tmux CLI.

- **DeepWiki MCP** — structured Q&A / wiki content for **public GitHub repos** (compare block layouts, attention variants).
- **alphaXiv MCP** — `https://api.alphaxiv.org/mcp/v1` with **`Authorization: Bearer ${ALPHAXIV_API_KEY}`** (export key before starting the agent; see alphaXiv account / API settings).
- **Hugging Face MCP** — papers, Hub models/datasets, library docs (cite titles + URLs in notes when an idea is literature-derived).
- **arXiv MCP** ([arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server)) — search/download/read arXiv PDFs locally via **`uv tool run`** (see `autoresearch/TOOLING.md`).
- **GitHub CLI (`gh`)** — browse/clone reference repos; cross-check implementations against our `models/cls_cvt.py`.

Retrieval **does not** replace correctness checks: validate tensor shapes, training stability, param/FLOP budget, and run order (one Slurm job, wait, parse). Follow **`program.md` § Traceability in `autoresearch/notes.md`**: every hypothesis records **why** the direction was chosen and **literature context** separately from **metric outcomes**.

**Datasets (optional, if CIFAR-10 is saturated for ranking hypotheses):**

- Root on this cluster: **`/path/to/data/data`** (CIFAR-100, Tiny ImageNet, etc., as laid out there).
- Set **`DATA.DATASET`** to `cifar100` or `tiny-imagenet` when the codebase supports it (see **`backup/configs/cvt/`** for YAML examples, e.g. `cvt_13_224_cifar100.yaml`, `cvt_13_224_tiny_imagenet.yaml`).
- Pass **`DATA_DIR`** with `sbatch --export=ALL,CFG,RUN_TAG,DATA_DIR,...` — the Slurm script passes `--data-path "$DATA_DIR"` to `main.py`. Point it at the correct subdirectory under the scratch tree (layout may be `.../data/cifar100`, `.../data/tiny-imagenet-200`, etc.—**verify on disk** before submitting).
- **Document every protocol change** in `autoresearch/notes.md` (dataset + epochs + seed); do not mix numbers across protocols in one “best” table without labeling columns.

**Long runs (300 epochs):**

- Default screen remains **100 epochs** for comparability.
- The agent **may** schedule **300 epochs** for **at most one or two champion configs** when logs/W&B show **clear underfitting at 100** *and* the human budget allows—state the rationale and **separate** W&B run group (e.g. `refine-300ep`). Never use 300-ep results to beat 100-ep rows without an explicit comparison table.

**Original research expectations:**

- Prefer **small, testable edits** in `models/cls_cvt.py` with a **hypothesis + failure mode** written in notes before Slurm.
- Use MCP/`gh` to **avoid reinventing** broken patterns and to **record** prior art in notes as **context**; **experiments** judge whether an idea helps here.

---

## Phase 2 (later): scalability

- Scale toward **~20–30M parameters** and train on **ImageNet-1k** to validate scaling—**out of scope** until the human extends the program beyond Phase 1b. Keep compact configs in the Phase-1 envelope unless explicitly scaling.

---

## Hypothesis checkpoint & resume (required discipline)

- **Count:** Maintain the latest hypothesis id in **`autoresearch/notes.md`** (e.g. `### H50 checkpoint` with date, best val acc, branch, job id).
- **After ~50 hypotheses (human stop):** Agent (or human) adds a **RESUME** block at the top or bottom of `notes.md`:
  - Champion **`MODEL.NAME` / branch / commit** (or path to saved YAML).
  - **Protocol:** dataset, epochs, seed, aug recipe summary.
  - **Next direction:** e.g. “Phase 1b — explore X under channel attention; MCP enabled.”
- **Next session first actions:** Read `program.md` (this file), `AGENT_INSTRUCTIONS.md`, **`autoresearch/TOOLING.md`** if using MCP, then **`autoresearch/notes.md`** RESUME block, then continue **one job at a time**.

---

## Bootstrap (first session)

1. **Environment:** In `slurm/cpvit_minimal.slurm`, set `CONDA_ENV` to a path that exists on **compute nodes**.
2. **Weights & Biases:** On the login node, `wandb login` once (or `WANDB_API_KEY`). Slurm sets **`WANDB=1`** by default; runs use `WANDB_NAME=$RUN_TAG`, `WANDB_RUN_GROUP=$SLURM_JOB_NAME`. Override `WANDB_PROJECT` if needed (default `anon-project`).
3. **Data:** CIFAR-10 at `data/cifar10` (torchvision may download if policy allows).
4. **Working config:** `cp configs/cpvit_minimal.yaml configs/autoresearch_experiment.yaml` — edit only the copy.
5. **Logs:** Slurm stdout **`logs/cpvit-minimal-<jobid>.out`**. Also inspect the same run in the W&B UI (curves, `val/acc1`, etc.).
6. **Baseline:** After the first 100-epoch reference run, record `Max accuracy:` (and optional best val loss from W&B) in `autoresearch/notes.md`. **`SEED=0`** by default in `config.py` unless you override with `--opts SEED <n>` for a reason.

---

## Your role

Maximize **CIFAR-10 validation top-1 accuracy** under the **Phase-1** constraints and the **structural** rules below. Prefer ideas that improve **channel attention** and **inductive bias** (locality, mixing, stable attention) over opaque capacity dumps.

---

## Traceability in `autoresearch/notes.md` (rationale vs proof)

For **each** logged hypothesis (every run), the notes entry must make two things explicit:

1. **Direction rationale** — Why this experiment *now*: what gap, ablation, or story beat it follows from (e.g. “H7 removed drop path; test if mild drop path returns under reduced aug”).
2. **Literature context** — What you **consulted** for *motivation*, not as evidence the idea works here. Use **arXiv IDs**, paper **titles + URLs**, Hugging Face paper links, GitHub repos, or our PDF **`Research-Context/5210_Optimized.pdf`**. If nothing external applied, state **`Literature: none — internal ablation / exploratory`** so the trace is honest.

**Important:** Citations and MCP retrieval document **intellectual context and why the direction was chosen**. They are **not** proof of correctness or of a val-acc win. **Validation** is always **implementation checks** (shapes, stability, param/FLOP budget) plus **Slurm + W&B metrics**. Do not argue “this is correct because paper X says so”—argue “we tried this because X motivates the inductive bias; metrics show …”.

Phase **1b** expects richer literature lines (MCP, `gh`, arXiv); Phase **1** may lean on prior notes + the project PDF, but the same **rationale + literature** structure still applies.

---

## Cluster policy (critical)

- **Do not** train on GPU or stress the **login node**.
- **Do** use `sbatch slurm/cpvit_minimal.slurm` (with `CFG`, `RUN_TAG`, optional `OPTS`).
- **One job at a time:** submit a run, **wait until that job finishes** (queue empty for that job id), **then** parse results, commit/push or reset, and only **then** start the next experiment. Do not stack overlapping training jobs.
- Login node: edit files, `git`, `wandb login`, `sbatch`, **wait/poll**, `grep`/`tail` logs, `python tools/compute_flops_ptflops.py`, optional `OMP_NUM_THREADS=2` / `MKL_NUM_THREADS=2` for small local scripts.

---

## Baseline (reference file)

**`configs/cpvit_minimal.yaml`** — channel-primary stack, **no MLP**, **no tail**, **no DW shortcut**, GAP + linear head. Do not overwrite; copy to `configs/autoresearch_experiment.yaml` for experiments.

---

## Experiment loop

1. **Branch:** `git checkout main && git pull`, then `git checkout -b autoresearch/my-hypothesis-tag` (use a real tag, not a placeholder).

2. **One hypothesis per iteration** — document in commit message when you keep a change. In **`autoresearch/notes.md`**, record **direction rationale** and **literature context** for that hypothesis (see **§ Traceability in `autoresearch/notes.md`**). Papers/MCP explain *why you chose the direction*; **metrics** prove whether it helped.

3. **Implement**
   - **Primary:** `configs/autoresearch_experiment.yaml` (unique `MODEL.NAME` per run).
   - **Secondary:** `models/cls_cvt.py` — small, motivated edits only.
   - **Epochs:** Default **100** (in YAML). For a quick smoke test only:  
     `OPTS='--opts TRAIN.EPOCHS 10'` — do not compare smoke runs to full 100-epoch numbers.

4. **Check compute** (after editing YAML, before Slurm):

   ```bash
   python tools/compute_flops_ptflops.py --repo . --yaml configs/autoresearch_experiment.yaml
   ```

5. **Submit exactly one Slurm job and wait for it to finish** — from repo root:

   ```bash
   export CFG=configs/autoresearch_experiment.yaml
   export RUN_TAG=autoresearch-$(date +%Y%m%d-%H%M%S)-my-hypothesis-tag
   JOBID=$(sbatch --parsable --export=ALL,CFG,RUN_TAG slurm/cpvit_minimal.slurm)
   # If your Slurm has no --parsable: JOBID=$(sbatch --export=ALL,CFG,RUN_TAG slurm/cpvit_minimal.slurm | awk '{print $NF}')

   echo "Waiting for job ${JOBID} to complete..."
   while [[ -n $(squeue -h -j "${JOBID}" 2>/dev/null) ]]; do sleep 30; done
   ```

   Then confirm the job did not fail (e.g. `sacct -j "${JOBID}" --format=State --noheader | head -1`, or check **`logs/cpvit-minimal-${JOBID}.out`** for `ERROR` / missing `Done:`).

   W&B is on by default; `RUN_TAG` becomes the run name. Optional: `export WANDB_PROJECT=my-project` before `sbatch`.

   **Do not** submit another training job until this one has finished and you have parsed step 6.

6. **Parse results**
   - Log file: `Max accuracy:` / per-epoch `Accuracy of the network...`
   - **W&B:** `val/acc1`, training loss, and any custom logs.

7. **Keep + publish (if improved)**  
   For the **same protocol** (100 epochs, same `SEED` policy), if **val accuracy** beats your recorded baseline (or you accept a justified tradeoff documented in the commit):
   - `git add` only intended files.
   - `git commit -m "autoresearch: <hypothesis> (val acc …, params …, GFLOPs …)"`
   - **`git push -u origin HEAD`** (or `git push origin <branch>`) so GitHub has the improvement.

   If **not improved:** `git reset --hard HEAD` (or discard edits to the experiment config) and try a new hypothesis.

8. **Repeat** from step 2 (or 3) — **only after** the previous job completed and you finished steps 6–7. Never overlap runs.

---

## What you may change

- **Config:** `MODEL.SPEC` fields that respect **no MLP / no tail** (see below), `TRAIN` (within 100-epoch default unless smoke), regularization, architecture sizes **within the Phase-1 param/FLOP envelope**.
- **Code:** `models/cls_cvt.py` — attention, projections, normalization, token layout—aligned with **channel attention / axis decoupling**.

---

## What you must NOT change (Phase 1 compact loop)

- **No standard ViT spatial FFN in the backbone:** keep **`MLP_RATIO`** at **0** for backbone stages (no conventional width-MLP on spatial self-attention blocks in the CA stack). **Channel-attention FFN** via **`CA_MLP_RATIO`** is **allowed** when it fits the axis-decoupling story and budget.
- **No tail:** **`SA_TAIL_DEPTH: [0,0,0]`** (extend with extra zeros if `NUM_STAGES > 3`); no ViT / spatial self-attention tail unless the human explicitly changes the program.
- **Dataset:** default **CIFAR-10** for Phase 1 comparability; Phase 1b may use **CIFAR-100** or **Tiny ImageNet** from the path in § Phase 1b with notes updated.
- **No login-node training.**

(DW shortcuts and other knobs exist in schema; follow the active phase in this file and `notes.md`.)

---

## Success metrics

| Metric | Where | Better |
|--------|--------|--------|
| Val top-1 | Log + W&B `val/acc1` | Higher |
| Params / GFLOPs | `compute_flops_ptflops.py` | Within Phase-1 envelope |

Record baselines and best scores in **`autoresearch/notes.md`**.
