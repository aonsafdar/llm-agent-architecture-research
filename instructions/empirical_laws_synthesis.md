# Channel Attention Insights — channel-primary ViT Empirical Synthesis

*Read this document ONCE at every session start. It captures the essential empirical knowledge accumulated across 90+ hypotheses (H1–P2-H32). Use it to avoid repeating failures and to guide the next hypothesis.*

---

## 1. What makes channel attention work — core empirical laws

**Law 1: FFN is load-bearing.** Pure CA stacking without MLP caps at ~70% on CIFAR-10 (baseline). H39 (ratio 0 at stage 3): −1.07pp. The SwiGLU MLP is not a helper — it is part of the representation capacity. Never remove or over-reduce it.

**Law 2: Single-head is better than multi-head.** H3 and H58 both independently confirmed this (−1.13pp, −0.36pp). Reason: in channel-token mode, each "head" reduces the per-token feature vector from H×W features to H×W/num_heads. Fewer spatial samples per channel token → weaker channel-to-channel comparison. Use `NUM_HEADS=[1,1,1,1]` always.

**Law 3: Spatial compactness improves channel attention.** H14 showed keeping stage-3 at 14×14 (stride=1) hurts vs 7×7. At smaller spatial maps, each channel's feature vector (of size H×W) is more globally informative — it aggregates the entire semantic response of that channel. Stage-3 at 7×7 is optimal.

**Law 4: Top-heavy depth distribution.** H15, H20, H26, H36, H62 all confirm: concentrate depth in the final stage (dim=384/576/640, 7×7). The deepest, most semantic CA is the most productive. Moving blocks earlier consistently hurts. Current optimal: [1,2,3,7] (Phase 1) → [1,2,5,10] (Phase 2).

**Law 5: Softmax competition is beneficial for channels.** Sigmoid attention (P2-H18, −0.21pp), differential attention (P2-H8, −0.91pp), and register tokens (P2-H29, −0.67pp) all hurt. Channels compete for attention weight via softmax; this competition forces channel specialization. Do not break it.

**Law 6: Do not modify the attention computation itself.** Every attempt to change what happens inside SDPA has failed: sigmoid attn, differential attn, value residual, gated output (H48, P2-H24), register tokens, channel pair bias, QK-norm, multi-head. The mechanism to improve is the *input to attention* (spatial context richness) and the *architecture around attention* (CPE, shortcuts, cross-stage fusion) — not attention itself.

---

## 2. What improves channel attention — validated mechanisms

### Spatial context enrichment for channel tokens

**CPE 5×5 (H51/H52): +0.26pp.** 5×5 DWConv at each stage entry injects local spatial context into channel tokens before attention. Each channel "knows where it is active" before comparing with other channels. 7×7 hurts (over-smooths); 3×3 is suboptimal. This is the single most important spatial enhancement.

**Periodic CPE re-injection (P2-H10/H26): +0.08pp + 0.07pp.** Stage-3 at interval=4 and stage-2 at interval=2. After multiple attention blocks, channel tokens' spatial activation patterns drift. Periodic CPE re-applies the spatial fingerprint mid-stage. Optimal schedule: `CPE_REPEAT_INTERVAL=[0,0,2,4]`. Denser intervals (3, or adding stage-1) hurt.

**5×5 Q/K kernel at stage-3 (P2-H11): +0.28pp.** Increasing Q/K DW conv from 3×3 to 5×5 at stage-3 (7×7 map) gives each channel token richer spatial evidence for computing channel similarity. 5×5 covers 71% of the 7×7 map. Does NOT generalise to stage-2 or stage-1 (confirmed twice: P2-H13, P2-H28).

**DW shortcuts 3×3 (H6): +0.81pp.** Local spatial shortcuts provide complementary gradient paths. 3×3 is optimal; 5×5 hurts once CPE is present (CPE and large DW kernels are redundant — H56 showed reverting to 3×3 gave +0.25pp after CPE was added).

### Gradient flow and training signal

**Deep supervision — triple aux (H46/H50/H64): +0.71pp cumulative.** Aux heads at stages 0, 1, 2 with weights 0.1, 0.2, 0.4. Gradient flow to early CA blocks is a real bottleneck. Stage-0 supervision (after 1 block) is surprisingly useful (+0.11pp). Optimal weights: (0.1, 0.2, 0.4). Stronger weights (0.15, 0.3, 0.4) or intra-stage mid-block supervision (H67, P2-H9) both hurt — supervision should be at stage boundaries only, not mid-block.

**Learned attention temperature (H47): +0.12pp.** Per-head log-parameterised scalar τ (init=0, i.e. τ=1 at start). Different stages have vastly different spatial feature dimensions (H×W varies 56²→7²) → different natural attention sharpness. Temperature allows each stage to calibrate independently. Gated output (H48) is NOT equivalent — it hurt. Temperature calibrates sharpness; gate controls magnitude — different concepts.

**Layer Scale 1e-4 (H30): stability.** CaiT-style per-block output scaling. Prevents early residuals from dominating. Init=1e-4 is the correct conservative value for this architecture (1e-5 too small, 1e-3 too large).

### Multi-scale aggregation

**Cross-stage GAP fusion (P2-H4/H22): +0.38pp + 0.12pp.** All-stage aggregation: collect GAP of stages 0, 1, 2, 3 → project to classifier dim → add with scale init=0. This is at the classifier, NOT at any attention block input. Spatial input fusion (P2-H16) hurt — inject only at the classifier. Scale init=0 ensures safe convergence.

**SwiGLU MLP (H54): +0.12pp.** Gated feedforward (SiLU(Wg·x) * Wu·x) selectively amplifies channel features. The gating is complementary to channel attention — attention routes, SwiGLU gates. Use hidden dim = 2/3 of GELU equivalent to maintain param count.

---

## 3. Exhausted directions — never retry

| Direction | Experiments | Outcome | Core reason |
|-----------|-------------|---------|-------------|
| Multi-head CA | H3, H58 | −1.13pp, −0.36pp | Splits spatial feature dim → weaker channel comparison |
| Channel pair bias | H43, P2-H21 | −0.13pp, −0.29pp | Unconstrained C×C params overfit; no structural benefit |
| Grouped channel CA | H3, P2-H3, P2-H17 | All negative | Cross-group interactions are essential |
| Sigmoid attention | P2-H18 | −0.21pp | Softmax competition is beneficial |
| Differential attention | P2-H8 | −0.91pp | Halving spatial features destroys Q/K signal |
| Value residual | P2-H19 | −0.07pp | No attention sink problem in practice |
| Dual-scale GAP attention | P2-H20 | −0.02pp | 5×5 Q/K already captures global info at 7×7 |
| Gated attention output | H48, P2-H24 | −0.07pp, −0.17pp | Gate suppresses useful cross-channel signals |
| Register tokens | P2-H29 | −0.67pp | Disrupts C×C attention matrix |
| QK-norm | H42 | −0.88pp | Constrains attention score range |
| RMSNorm | H57 | −0.23pp | LayerNorm mean-centering is important for channel tokens |
| 5×5 Q/K at stage-2 | P2-H13, P2-H28 | −0.34pp, −0.57pp | 14×14 map: 5×5 is local, blurs local patterns |
| DW shortcut >3×3 | H44, H49, P2-H14 | −0.22pp, −0.11pp, +0.01pp (noise) | 3×3 optimal with CPE; larger is redundant |
| 7×7 CPE kernel | H53, P2-H31 | −0.26pp, −0.34pp | Full-map CPE over-smooths; 5×5 is the sweet spot |
| dw_sep at stages 1/2 | H18, H38 | −0.64pp, −0.30pp | dw_only downsampling is better there |
| Extra depth at stage-0 | H19, H36 | −0.56pp, −0.31pp | One block at 56×56 is sufficient |
| CPE interval variations | P2-H27, P2-H30 | −0.63pp, −0.86pp | Optimal: [0,0,2,4]. Adding stage-1 or denser stage-3 hurts |
| Intra-stage mid-block aux | H67, P2-H9 | −0.22pp, +0.01pp (noise) | Supervision at stage boundaries only |
| Sparse top-k CA (stage-3 only) | H3-1 | −0.17pp (ImageNet) | Softmax tail is NOT noise at C=640; pruning pairwise channel interactions loses useful signal. Whole "restrict-CA" family likely dead end |
| Dual-resolution CA (fine + coarse parallel, init-0 ramp) | H3-2 | −0.15pp (ImageNet) | Adding a parallel coarse CA pathway alongside fine CA didn't help — coarse path either stayed near-zero or competed with cross-stage fusion. Minor CA-mechanism tweaks plateau. Move to new pathways (cross-stage CA bridges) or different mixing operators (linear/SSM) |

---

## 4. Current champion architecture (Phase 2, P2-H26)

| Parameter | Value |
|-----------|-------|
| **DIM_EMBED** | [96, 192, 384, 640] |
| **DEPTH** | [1, 2, 5, 10] |
| **CA_MLP_RATIO** | [8, 8, 4, 1] |
| **NUM_HEADS** | [1, 1, 1, 1] |
| **SwiGLU** | true |
| **CPE_KERNEL_SIZE** | 5 |
| **CPE_REPEAT_INTERVAL** | [0, 0, 2, 4] |
| **KERNEL_QKV** | [3, 3, 3, 5] (5×5 at stage-3 only) |
| **DW_SHORTCUT_BACKBONE** | true, KERNEL=[3,3,3,3] |
| **CROSS_STAGE_FUSION** | true (stages 1+2) |
| **CROSS_STAGE_FUSION_S0** | true (adds stage 0) |
| **AUX_LOSS_WEIGHT_S0/S1/S2** | 0.1 / 0.2 / 0.4 |
| **AUX_MID_BLOCK_IDX** | 5 (mid-stage-3) |
| **LAYER_SCALE** | 1e-4 |
| **ATTN_TEMP** | true (learned per head) |
| **PATCH_EMBED_METHOD** | ['dw_sep','dw_only','dw_only','dw_sep'] |

**Performance:** 83.37% CIFAR-100 100ep (P2-H26), 85.07% CIFAR-100 300ep (P2-H15 arch), ~22M params, ~3.5 GFLOPs. **Primary benchmark (Phase 3): ImageNet-1K 100ep.**

---

## 5. Key insights from failures — understanding the channel attention regime

**Why does spatial context injection help but attention modification hurt?** Channel attention at stage-3 operates on 640 tokens, each a vector of 49 spatial features (7×7). The *quality of this vector* determines channel comparison quality — CPE and Q/K kernels improve this vector. But SDPA itself is already doing the right thing: it correctly identifies which channels co-activate. Modifying SDPA (sigmoid, differential, gated, register) disrupts a mechanism that works.

**Why does multi-scale fusion help at the classifier but not at the input?** GAP fusion at the classifier (P2-H4, H22) adds complementary multi-scale channel statistics. Spatial input fusion (P2-H16) injects stage-2's 14×14 feature maps into stage-3's input and hurts: the semantic abstraction levels conflict. Lesson: cross-scale feature reuse should happen at the end (classifier), not at intermediate inputs.

**Why does drop path regime flip between Phase 1 and Phase 2?** Phase 1 (5.8M, CIFAR-10, 100ep): model underfits → drop path hurts. Phase 2 (22M, CIFAR-100, 100ep): model near-capacity → drop path helps. Regularization decisions must be re-made at every new scale/dataset.

**Why does multi-head CA consistently hurt?** Not because multi-head attention is wrong in principle, but because channel-token CA is already "multi-view" — the spatial feature vector (dim=H×W) is the natural feature space, and single-head uses all of it. Splitting heads means each head only sees H×W/num_heads spatial features per channel token, degrading Q/K quality. A different form of "multi-view" (e.g. different spatial receptive fields, or hierarchical scales) might work — just not head-based splitting.

---

## 6. Phase 3 — open frontiers (bold structural directions)

**This list is non-exhaustive and meant to seed thinking, not to constrain it.** Any structural idea from recent literature (2024+) or from cross-referencing the failure patterns in §3 is equally welcome. The examples below have not been tried; each represents a qualitatively new direction. Use them as inspiration, not as a closed menu.

**Novelty criterion** (apply to every hypothesis): a bold hypothesis introduces (a) a new channel-attention mechanism, (b) a new structural component/topology, or (c) a direct response to a failure pattern from §3 (e.g. "every SDPA tweak failed → replace SDPA entirely"). Hypotheses describable as "change parameter X from A to B" do not qualify, regardless of the knob.

1. **Funneling: extra stages (4×4, 2×2, 1×1).** Current stage-3 is 7×7. Additional stages compress spatial maps further. At 1×1, channel attention becomes pure SE-Net-style global recalibration with learned Q/K. A novel, deep funnel creates a true local→global CA hierarchy. Could allow wider dims (768+) at the cheap tiny stages.

2. **Factorised / low-rank channel attention.** Replace C×C attention with two projections C→r and r→C (r<<C, e.g. r=64). Forces compact inter-channel relationship basis. Reduces redundancy in the current 640×640=409K attention matrix.

3. **Sparse top-k channel attention.** Each channel attends to only its top-k most correlated channels (k=32–64). Removes spurious weak correlations that may dominate the softmax. Allows C to scale further without quadratic cost.

4. **Hierarchical group CA (within-group + cross-group).** NOT the same as failed grouped CA (P2-H3/H17 had no cross-group step). New: dense CA within G groups, then a lightweight cross-group CA on G summary tokens. Gives fine-grained within-group specialisation AND coarse cross-group routing.

5. **Mamba/SSM for channel mixing.** State space models (Mamba, S4) can mix channels in O(C log C) with long-range memory. Replaces SDPA in channel-token mode with an SSM scan. Different inductive bias: sequential channel processing vs all-pairs attention.

---

*Last updated: Phase 3 transition, 2026-03-29. Reflect H1–P2-H32 (90+ runs).*
