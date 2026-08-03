# channel-primary ViT Autonomous Architecture Research

Read `program.md` for the full research program. Read `autoresearch/notes.md` for experimental history and current best results.

## What this project is

Channel-Primary Vision Transformer: channels are tokens, spatial dimensions are features; dimension swapping in the backbone enables **pure channel attention** without a spatial self-attention tail. The research goal is to **maximize validation accuracy** by innovating on the channel-attention architecture itself — not by hyperparameter tuning or augmentation sweeps.

**Paper context:** `Research-Context/5210_Optimized.pdf` contains the full research framing, notation, and motivation.

## Current best

**Phase 1 results (5.8M params, ~1.2 GFLOPs, CIFAR-10):**
- **100-epoch champion:** H64 — **96.59%** val top-1. Triple auxiliary supervision (stages 0+1+2), CPE 5×5, SwiGLU MLP, learned attention temperature, DW shortcuts, Layer Scale.
- **300-epoch validation:** H61 (H56 arch) — **97.25%** val top-1 with Mixup/CutMix/LS re-enabled.

**Architecture:** 4-stage [64→128→256→384], DEPTH=[1,2,3,7], CA_MLP_RATIO=[8,8,4,1], DW_SHORTCUT_BACKBONE=true, Layer Scale 1e-4, NUM_HEADS=[1,1,1,1], CPE 5×5, SwiGLU, triple aux loss (0.1, 0.2, 0.4).

**66 hypotheses completed.** See `autoresearch/notes.md` for the full log (H1–H67+). Phase 1 CIFAR-10 work is complete.

---

## Phase 3: ImageNet-1K Structural Innovation

You are now in **Phase 3**. Phases 1 (CIFAR-10, 5.8M) and 2 (CIFAR-100, 22M) are complete. The primary benchmark is now **ImageNet-1K 224×224**. The goal is bold structural innovation in channel attention — not more incremental config sweeps.

### Ultimate goal

**Prove that pure channel attention — without any spatial self-attention — can match or beat SOTA on ImageNet-1K at ~20M params.** Current SOTA for efficient ViTs at this scale is ~83–84% top-1 (e.g. FastViT, EfficientViT-MIT, SHViT). These models all rely on spatial self-attention. Achieving competitive results with channel attention alone would be a significant and publishable finding.

**The channel-attention mechanism is the core contribution.** Every architectural change you make should improve *how channels attend to each other* — richer channel interactions, better cross-stage channel information flow, smarter channel grouping/factorization, stronger spatial context injection *into* channel tokens, etc. Do not chase accuracy through tricks unrelated to channel attention (e.g. generic augmentation sweeps, distillation from spatial-attention teachers, or bolt-on spatial processing that undermines the pure-CA story). If you hit a ceiling, the right response is to innovate on channel attention itself, not to work around it.

CIFAR-100 is the fast iteration benchmark. **ImageNet-1K is the final validation target.**

### Budget

| Constraint | Target |
|------------|--------|
| **Params** | **~20M** (15–22M acceptable range) |
| **FLOPs** | **5–8 GFLOPs** (check before every submit) |
| **Epochs** | **100** for CIFAR-100 screening, **300** for champion validation |
| **GPUs** | **2 available** — run CIFAR-100 iteration + ImageNet validation in parallel |

### Dataset

| Dataset | Path | Classes | Role |
|---------|------|---------|------|
| CIFAR-100 | `/path/to/data/data/cifar100` | 100 | **Primary iteration benchmark.** Fast (~2h/run at 100ep). All hypothesis screening here. |
| ImageNet-1K | `/path/to/imagenet` | 1000 | **Ultimate validation target.** 1.28M train / 50K val images, 224×224. Run champion architectures here. |
| Tiny-ImageNet | `/path/to/data/data/tiny-imagenet-200` | 200 | Secondary generalizability check. 64×64 native (resize to 224). |
| CIFAR-10 | `data/cifar10` (repo-local) | 10 | Legacy reference only. Do not use for new experiments. |

When switching datasets, update `DATA.DATASET`, `DATA.DATA_PATH`, `MODEL.NUM_CLASSES`, and `DATA.IMG_SIZE` in the experiment YAML.

### ImageNet-1K protocol — PRIMARY BENCHMARK (effective immediately)

**CIFAR-100 is no longer the primary iteration benchmark. All hypothesis screening now uses ImageNet-1K at 224×224.**

#### 100-epoch screening (hypothesis iteration)
- **Use for every new hypothesis** — this is the new fast-iteration benchmark. ~2 days per run on any GPU.
- **Any 2 GPUs** — any GPU type is acceptable for 100ep screening. No need to wait for a specific accelerator type.
- **Recipe**: `BATCH_SIZE=256` per GPU (512 total), `LR=5e-4`, `WD=0.05`, 100ep, warmup=10ep, Mixup=0.4, CutMix=0.5, RandAugment rand-m7-mstd0.5-inc1, RE=0.1, LS=0.1, DROP_PATH=[0,0.05,0.1,0.2].
- **Submit 2-GPU** via `--wrap` with `torchrun --nproc_per_node=2`. If 2-GPU fails on launch (exits in <60s), fall back to single GPU immediately (`--nproc_per_node=1`, `--local_rank 0`, `BS=512`).
- **2-GPU launch requirement**: `main.py` requires `--local_rank` explicitly — torchrun sets it as env var only. Always include `--local_rank $LOCAL_RANK` in the launch command.
- **Baseline to beat**: establish a clean ImageNet-1K 100ep baseline first (current champion P2-H26 arch, recipe above). Log as `IN-B100`.
- **Commit if improved** vs the ImageNet 100ep baseline. Discard otherwise.

#### 300-epoch validation (champions only)
- Only for architectures that improved at 100ep. Use GPU nodes, 2 GPUs, full DeiT recipe (Mixup=0.8, CutMix=1.0, rand-m9, RE=0.25, warmup=20ep, LS=0.1). Target **≥85% top-1**.
- Reference config: `configs/imagenet_2gpu.yaml`. Update arch to match current champion before submitting.

#### Log format
Log ImageNet results in `autoresearch/notes.md` under `### ImageNet-1K` section. Track ep25/ep50/ep100 accuracy to monitor trajectory.

### Epoch policy

- **100 epochs on ImageNet-1K**: default for screening all new hypotheses.
- **300 epochs on ImageNet-1K**: only for confirmed champions (improved at 100ep). Use GPU, full DeiT augs.
- **CIFAR-100**: no longer used for screening. May be used for quick sanity checks only if ImageNet queue is full.
- **Document the training recipe for every run** in `autoresearch/notes.md`.

### What carries forward from Phase 1

The following architectural innovations were validated at 5.8M on CIFAR-10 and should form the starting point:
- **SwiGLU MLP** (H54: +0.12pp)
- **CPE 5×5** (H51-52: +0.26pp cumulative)
- **Deep supervision / auxiliary losses** (H46/50/64: +0.71pp cumulative)
- **Learned attention temperature** (H47: +0.12pp)
- **Layer Scale 1e-4** (H30: +0.01pp, but important for training stability)
- **DW shortcuts 3×3** (H6: +0.81pp)

Use these as the baseline architecture, scaled to ~20M params. How you scale (wider, deeper, more stages, or combinations) is your decision — research the tradeoffs.

### Constraints (unchanged)

- **No spatial self-attention tail**: `SA_TAIL_DEPTH` stays all zeros. The story is pure channel attention.
- **Central idea**: every change must make channel attention itself more powerful. Ask yourself: "does this improve how channels attend to each other, or is it a generic trick that any architecture could use?" If the latter, skip it.
- **Accuracy is evidence, not the goal.** The goal is the strongest possible channel-attention mechanism. Accuracy on CIFAR-100/ImageNet validates whether your channel-attention innovations actually work.
- **No login-node training.**
- **Commit + push on improvement** to the current branch. Discard and move on otherwise.

---

## Research direction

**You are a researcher, not an engineer.** Your hypotheses should be driven by ideas from recent literature, not by config ablations. Changing kernel sizes, toggling flags, and adjusting widths are engineering — they produce diminishing returns. Real breakthroughs come from understanding what recent papers have discovered about attention mechanisms and adapting those insights to the channel-attention domain.

### How to generate hypotheses (follow this order)

1. **Start with literature.** Before designing any hypothesis, search arxiv (2024–2026) for recent advances in attention mechanisms, token mixing, channel interactions, or efficient vision transformers. Use `search_papers` and `read_paper` MCP tools. Read **1–2 abstracts** per hypothesis (see Token efficiency — do not pull in many papers per run).
2. **Find a mechanism.** Identify a specific mechanism, technique, or insight from a recent paper that could be adapted to channel-token attention. Examples: "This paper uses X for spatial tokens — how would X work for channel tokens?" or "This paper shows Y improves attention quality — can Y be applied to channel-to-channel attention maps?"
3. **Adapt to channel attention.** Design your hypothesis around adapting the mechanism to channel tokens. Explain *why* the mechanism works differently (or better) for channels vs spatial tokens.
4. **Then implement.** Only after steps 1–3 should you write code.

**Do NOT:** generate hypotheses by tweaking the previous run's config (bigger kernel, fewer blocks, wider dims). If your hypothesis can be described as "change parameter X from A to B," it is engineering, not research. Research hypotheses introduce *new mechanisms* or *structural changes* inspired by understanding from literature.

### What has already been exhausted (do NOT retry these)

The following directions have been systematically explored and failed across 10+ hypotheses. Do not revisit them:
- **Attention mechanism tweaks**: sigmoid attention, differential attention, value residual, gated output, grouped CA, register tokens, channel pair bias, multi-head CA — **all failed**. The softmax single-head CA is at its optimum for the current architecture.
- **CPE schedule tuning**: interval variations (3, 4), per-stage extension to stage-1 — exhausted. CPE is good as-is.
- **Q/K/V kernel sizes at stage-2**: 5×5 at stage-2 consistently hurts.
- **DW shortcut kernel sizes**: 3×3 is optimal, larger hurts.
- **Incremental width/depth changes**: minor embedding dim tweaks, slight depth rebalancing — all noise-level.

**The next hypothesis must introduce a genuinely new structural idea** not in the list above.

### Bold structural directions

The list below is **illustrative, not prescriptive**. Any hypothesis that passes the novelty test below qualifies as bold. Use these examples as seeds for thinking — do not treat them as a closed menu. Novel ideas you synthesise from literature (2024+) or from patterns across our own 90+ runs are equally welcome and, in fact, encouraged.

**Novelty test — every hypothesis must pass this:**
A hypothesis is *bold* if it introduces:
- a new mechanism for how channels attend to each other (new CA formulation), **or**
- a new structural component not present in the champion (new building block or new topology), **or**
- a direct response to a pattern observed in our own failure record (e.g. "every SDPA modification failed in §3 — try replacing SDPA entirely with an SSM scan").

A hypothesis **fails** the test if it can be described as "change parameter X from A to B" without introducing a new mechanism or component. Config edits alone do not qualify, regardless of which knob is touched.

Seed examples (non-exhaustive — feel free to go beyond them):

1. **Funneling architecture — extra stages to extreme spatial compression**
   Add 1–2 stages beyond the current 7×7, reducing the spatial map to 4×4, 2×2, or even 1×1. At each scale, channel attention captures qualitatively different relationships: 7×7 = semantic spatial patterns, 2×2 = quadrant-level co-activation, 1×1 = pure global channel recalibration (SE-Net-like). This creates a true *channel attention hierarchy* spanning local-to-global. The final stages are computationally cheap (tiny spatial maps) so you can make them wide (768–1024 dim) without blowing the budget. Combined with all-stage cross-fusion, every scale feeds the classifier.

2. **Factorised / low-rank channel attention**
   The C×C attention matrix (640×640 = 409K entries) is expensive and potentially redundant. Factorise into C×r and r×C projections (r << C, e.g. r=64), creating a bottleneck channel interaction graph. This forces the model to learn a compact basis of inter-channel relationships.

3. **Sparse top-k channel attention**
   Instead of every channel attending to all C others (dense softmax), each channel attends to only its top-k most correlated channels (k=32–64). This removes spurious weak inter-channel correlations and forces the model to learn which channels are genuinely informative for each other.

4. **Hierarchical local + global channel attention (within-group + cross-group)**
   Split channels into G groups. Within each group, run full dense CA (local channel interactions). Across groups, run a lightweight global CA on group-summary tokens (group GAP). Fine-grained within-group specialisation plus coarse cross-group routing. Distinct from failed grouped-only CA (H17, P2-H3) because the cross-group step is added.

5. **Linear / kernel / SSM channel attention**
   Replace softmax(QK^T)V with a kernel approximation (random Fourier features, linear attention φ(Q)·(φ(K)^T·V)) or an SSM scan (Mamba/S4-style) to compute channel interactions in O(C) or O(C·d) instead of O(C²). Different inductive bias entirely.

**Self-audit before submitting any hypothesis:** in the notes.md entry, explicitly state (i) which novelty category it falls into (new mechanism / new component / failure-pattern response / other-bold), or (ii) a one-line justification of why it's genuinely new. If you cannot write this line, the hypothesis is not bold enough.

### General research areas for literature search

- **"efficient attention" OR "linear attention" vision 2025**
- **"token mixing" OR "state space model" vision 2025** — Mamba/RWKV for channel mixing
- **"channel attention" hierarchy OR pyramid 2024**
- **"sparse attention" top-k vision 2024**
- **"funnel transformer" OR "progressive downsampling" attention 2024**

### Literature quality requirements

**Every hypothesis must be grounded in a specific paper or mechanism from 2024+.** This is non-negotiable.

1. **Search arxiv for papers from 2024–2026** using the MCP tools. Use broad technique queries (e.g. "efficient attention mechanism 2025"), not narrow architecture queries.
2. **Read the abstract and key sections** of at least one relevant paper using `read_paper`. Extract the specific mechanism that inspires your hypothesis.
3. **In your notes.md entry**, clearly state: "Inspired by [Paper Name] (arXiv:XXXX.XXXXX, 2024): [specific mechanism]. Adapted for channel tokens by [explanation]."
4. **If no relevant 2024+ paper exists** for an idea, that is fine — but note that you searched and explain why the idea is novel. Purely first-principles hypotheses are allowed but should be the exception, not every hypothesis.

---

## Tools available

### Literature (MCP servers — configured in `.mcp.json`)
- **arxiv-mcp-server**: `search_papers`, `download_paper`, `read_paper` — search for attention mechanisms, normalization, ViT architecture papers
- **deepwiki**: ask questions about any GitHub repo — use to understand implementations in repos like `rwightman/pytorch-image-models`, `facebookresearch/deit`, `microsoft/CvT`, etc.
- **huggingface**: search models, datasets, papers on HuggingFace

Use these **before** designing a hypothesis: search for prior art, read relevant papers, understand how others implemented the idea. Cite papers in your commit messages when relevant.

### Code
- **`models/cls_cvt.py`** — the model. This is your primary innovation surface. Read it thoroughly before proposing changes.
- **`config.py`** — register new YAML keys here when adding architectural features.
- **`configs/autoresearch_experiment.yaml`** — the mutable experiment config. Keep `configs/cpvit_minimal.yaml` as read-only reference.
- **`tools/compute_flops_ptflops.py`** — budget check before every Slurm submit.

### Compute
- GPU training via `sbatch slurm/cpvit_minimal.slurm` only. Never train on the login node.
- **2 GPUs available.** You may run up to 2 concurrent jobs (e.g. one CIFAR-100 screening + one ImageNet validation).
- For CIFAR-100 iteration: submit → let turn end → watcher auto-nudges → parse → commit/discard → next.
- **Do NOT poll squeue in loops.** A standalone watcher (`slurm/job_watcher.sh`) runs in a separate tmux window and handles job completion detection for all jobs.

---

## Token efficiency (hard limits — usage caps matter)

Rate limits burn fast on long autonomous runs. **Treat tokens as a budget.** Research quality comes from correct experiments and clear `notes.md` entries, not from long reasoning traces or huge file reads.

### Session start

At the start of each new session (i.e., after a restart or long pause), read these **three files in full — once**:
1. **`RESEARCH_BRIEF.md`** — this file (full read, it is short).
2. **`autoresearch/notes.md`** — the FULL file. This is the complete experimental record of 90+ hypotheses. You need the full history to avoid repeating failures and to build on established knowledge. Do not tail it.
3. **`autoresearch/channel_attention_insights.md`** — the empirical synthesis. Compact (~6 pages). Contains empirical laws, validated mechanisms, exhausted directions, and open frontiers. Read it every session start.

After the session start reads, **do not re-read these files every turn** — summarize key points in memory. Only re-read specific sections if you need to check a detail.

- **`program.md`:** read once per session or when scope is unclear.
- **Never** paste large chunks of `notes.md`, logs, or search results into your replies — summarize in 3–5 bullets max when needed.

### Per hypothesis (caps)

| Action | Limit |
|--------|--------|
| Literature | **1** `search_papers` query; pick **1** paper; **`read_paper` abstract (or first page) only** unless you must implement a non-obvious detail |
| Code | **`grep` / narrow `read_file` range** on `cls_cvt.py` — do not re-read the entire file each hypothesis |
| Config | Diff against previous YAML or read only changed keys — avoid full duplicate reads |
| Logs | **`grep 'Max accuracy'`** on `logs/cpvit-minimal-<jobid>.out` — never `cat` multi‑MB logs |
| DeepWiki / Hugging Face | **Skip** unless arxiv search fails; they are expensive |
| Bash | One command per check where possible; **no** `squeue` / watch loops |

### Writing and commits

- **`notes.md`:** stay thorough on *hypothesis, mechanism, result, and one literature line* — that is the durable record. Do not duplicate the same story in three places in one entry.
- **Commit messages:** one informative line is enough; no multi-paragraph essays unless the change is huge.

### After job submit

- **Stop.** Do not “think out loud” at length, re-plan the whole program, or re-read unrelated files. End the turn and wait for the watcher.

### Human-side controls (outside this file)

- In **the CLI agent / CLI**, prefer a **faster / cheaper model** for routine iteration if available; reserve the strongest model for hard debugging.
- Disable or shorten **extended thinking** if it multiplies token use without improving Slurm outcomes.

---

## Autonomous loop

**You run autonomously.** Do not stop between hypotheses. Do not ask for permission. Do not wait for user input. Run hypotheses back-to-back in a continuous loop. **Never end your turn while the loop has work to do.**

1. **Search literature first** (arxiv, deepwiki). Find a recent (2024+) paper with a mechanism to adapt. Read the abstract/intro. This is the most important step — do not skip it.
2. Design the hypothesis around the mechanism found in step 1. Write it up in your head before touching code.
3. Read `models/cls_cvt.py` and understand the relevant code path.
4. Implement: edit `cls_cvt.py` + register new config keys in `config.py` + update `configs/autoresearch_experiment.yaml`.
5. Check budget: `python tools/compute_flops_ptflops.py --repo . --yaml configs/autoresearch_experiment.yaml`
6. Submit the job with plain sbatch:
   ```
   sbatch --export=ALL,CFG=configs/autoresearch_experiment.yaml,RUN_TAG=<tag> slurm/cpvit_minimal.slurm
   ```
   **Do NOT use `submit_and_wait.sh`.** A standalone job watcher (`slurm/job_watcher.sh`) runs in a separate tmux window and will automatically nudge you when any job finishes. **After submitting, let your turn end.** Do NOT poll squeue — that wastes tokens.
7. When the watcher nudges you, parse `logs/cpvit-minimal-<jobid>.out` for `Max accuracy:`.
8. Update `autoresearch/notes.md` with hypothesis, motivation (including paper references), result.
9. If improved: commit + `git push`. If not: discard/revert changes.
10. **Immediately go to step 1 (literature search) for the next hypothesis. Do NOT stop.**

**Only pause and ask the user if a job fails catastrophically** (crash, OOM, missing data, repeated Slurm failures). Normal negative results are not failures — discard and move on.

**If your turn ends while waiting for a job**, the standalone watcher in the `watcher` tmux window will automatically send you a message when it finishes. Just resume the loop from where you left off.
