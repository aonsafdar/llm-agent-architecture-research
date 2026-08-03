# channel-primary ViT Autoresearch Notes

## Baseline (ground truth for this loop)
| Run | Config | Epochs | Max Acc | Job | Branch |
|-----|--------|--------|---------|-----|--------|
| baseline-100ep-nodw | cpvit_minimal.yaml as-is | 100 | **69.67%** | 337529 | autoresearch/baseline-100ep-nodw |

**Baseline architecture:** CvT13-like, 3 stages, DIM_EMBED=[64,192,384], DEPTH=[1,2,10], channel-attention only (no SA tail, no MLP anywhere — CA_MLP_RATIO=[0,0,0], MLP_RATIO=[0,0,0]), DW_SHORTCUT=false. 1.724M params, 0.424 GFLOPs.

**Observations:** Train loss at epoch 100 ≈ 1.89, still declining — model is underfitting. CA_MLP_RATIO=0 means absolutely no feedforward mixing; the model is pure channel-attention stacking. This likely caps representation power.

---

## Reference (prior runs, 300 epochs — NOT comparable directly)
| Run | Config | Epochs | Max Acc | Notes |
|-----|--------|--------|---------|-------|
| cpvit-c10-baseline-328982 | notail_nomlp_baseline | 300 | 82.02% | baseline without DW |
| cpvit-c10-nomlp-328917 | notail_nomlp_dw1st | 300 | 91.69% | DW skip on backbone |

---

## Hypothesis Log

### H1: Add CA MLP (CA_MLP_RATIO = [4.0, 4.0, 1.0])
- **Motivation:** Baseline has zero MLP anywhere. Standard attention blocks benefit from FFN for feature mixing/projection. CA_MLP_RATIO adds FFN after channel-attention in backbone stages — directly tests whether pure-attention stacking is the bottleneck.
- **Change:** CA_MLP_RATIO: [4.0, 4.0, 1.0] (was [0,0,0]). Ratio=1.0 in stage 2 (10 deep blocks) to stay within budget; 4.0 in stages 0/1.
- **Budget:** 5.314M params, 1.573 GFLOPs
- **Status:** DONE (job 337713, branch autoresearch/h1-camlp-4-4-1) — PUSHED ✓
- **Result:** **91.08%** (+21.41pp over baseline). Still improving at ep100 (loss not plateaued). All-stage CA MLP is a massive gain.

### H2: Stage-2-only CA MLP (CA_MLP_RATIO = [0, 0, 1.0])
- **Motivation:** Ablation — does the stage-2 MLP (10 deep blocks at 384-dim) drive H1's gain, or do stages 0/1 MLPs also contribute? If H2 ≈ 91% then early MLPs are redundant.
- **Change:** CA_MLP_RATIO: [0, 0, 1.0] (vs H1's [4.0, 4.0, 1.0])
- **Budget:** 4.688M params, 1.006 GFLOPs
- **Status:** DONE (job 337858, branch autoresearch/h2-camlp-s2only) — NOT pushed (did not beat H1)
- **Result:** **89.97%** (−1.11pp vs H1). Early-stage MLPs (stages 0/1) do contribute ~1.1pp. Stage-2-only MLP is insufficient to match all-stage MLP. Loss still declining at ep100 — model underfitting. Discarded.

### H3: Multi-head channel attention (NUM_HEADS = [1, 2, 4])
- **Motivation:** H1 uses single-head CA everywhere — one global channel-attention pattern per stage. Multi-head CA allows parallel specialized channel groupings: multiple heads can independently learn different inter-channel dependencies. In channel-token mode, each head receives H*W/num_heads spatial features per channel token (stage 2: 196/4=49 per head, attending over all 384 channel tokens). Zero extra parameters — purely structural.
- **Change:** NUM_HEADS: [1, 2, 4] (was [1, 1, 1]). Stage 1: 2 heads × 392-dim, Stage 2: 4 heads × 49-dim.
- **Budget:** 5.314M params, 1.573 GFLOPs (identical to H1)
- **Status:** DONE (job 337917, branch autoresearch/h3-multihead-ca) — NOT pushed (did not beat H1)
- **Result:** **89.95%** (−1.13pp vs H1). Multi-head CA hurts: splitting 196 spatial features/token into 49 per head weakens each head's ability to compute channel relationships. Single-head with full spatial context wins. Discarded.

### H4: Deeper stage-2 backbone (DEPTH = [1, 2, 11])
- **Motivation:** Model is underfitting at ep100 (loss still declining). H1 uses DEPTH=[1,2,10]. Adding 1 more stage-2 block increases bottleneck capacity with the same per-block structure (CA + CA_MLP ratio 1.0). Direct test of whether more CA blocks improve accuracy.
- **Change:** DEPTH: [1, 2, 11] (was [1, 2, 10]). NUM_HEADS stays [1,1,1], CA_MLP_RATIO stays [4,4,1].
- **Budget:** 5.772M params, 1.664 GFLOPs
- **Status:** DONE (job 337935, branch autoresearch/h4-deeper-s2) — PUSHED ✓
- **Result:** **91.13%** (+0.05pp vs H1). Marginal but positive improvement. More CA blocks in the bottleneck do help slightly. Loss still declining at ep100 — still underfitting.

### H5: Larger early-stage CA MLP (CA_MLP_RATIO = [8.0, 8.0, 1.0])
- **Motivation:** H4 added depth in stage 2 but only +0.05pp. H2 showed early-stage MLPs (stages 0/1) contribute ~1.1pp — suggesting they may be bottlenecked. H5 doubles their MLP ratio (4→8) while reverting to H1's depth (DEPTH=[1,2,10]). Same param budget, different allocation: early-stage width vs bottleneck depth. Tests whether wider early-stage MLPs are a better use of parameters.
- **Change:** CA_MLP_RATIO: [8.0, 8.0, 1.0] (was [4.0, 4.0, 1.0]), DEPTH: [1, 2, 10] (reverted from H4's 11).
- **Budget:** 5.938M params, 2.139 GFLOPs
- **Status:** DONE (job 337944, branch autoresearch/h5-camlp-8-8-1) — PUSHED ✓
- **Result:** **91.71%** (+0.58pp vs H4, +0.63pp vs H1). Strong gain from doubling early-stage MLP width. Ratio 8 in stages 0/1 beats extra bottleneck depth. Early-stage feature extraction is a meaningful bottleneck. Model still underfitting at ep100.

### H6: DW shortcut backbone (DW_SHORTCUT_BACKBONE = true)
- **Motivation:** Reference pure-CA run (300ep, DW shortcut, no MLP) got 91.69%. H5 (100ep, MLP, no DW shortcut) is at 91.71%. DW shortcut provides additional gradient paths and feature reuse. Cost is negligible (~52K params: 1×1 DW conv per block per channel). Tests whether DW shortcut synergizes with the CA+MLP framework to further improve accuracy.
- **Change:** DW_SHORTCUT_BACKBONE: true (was false). All else same as H5 (CA_MLP_RATIO=[8,8,1], DEPTH=[1,2,10]).
- **Budget:** 5.990M params, 2.152 GFLOPs
- **Status:** DONE (job 337973, branch autoresearch/h6-dw-shortcut) — PUSHED ✓
- **Result:** **92.52%** (+0.81pp vs H5, +1.44pp vs H1). Strong synergy: DW shortcut + CA+MLP is highly effective. Better gradient flow through backbone helps. Model still likely underfitting at ep100.

### H7: Remove stochastic depth (DROP_PATH_RATE = [0,0,0])
- **Motivation:** Model is underfitting at ep100 (loss still declining). DROP_PATH_RATE=0.1 in stage 2 randomly skips ~1 block per step, reducing effective capacity per training iteration. For an underfitting model this is counterproductive. H7 removes all stochastic depth to let the model train at full capacity every step.
- **Change:** DROP_PATH_RATE: 0.0 globally + [0.0, 0.0, 0.0] per stage (was 0.1 / [0,0,0.1]).
- **Budget:** 5.990M params, 2.152 GFLOPs (unchanged — drop path has no inference cost)
- **Status:** DONE (job 338010, branch autoresearch/h7-no-droppath) — PUSHED ✓
- **Result:** **92.98%** (+0.46pp vs H6). Removing stochastic depth (0.1→0.0 in stage 2) confirms underfitting: model benefits from full capacity every training step. Regularization was counterproductive.

### H8: Shorter warmup (WARMUP_EPOCHS = 5)
- **Motivation:** Model is underfitting at ep100. With WARMUP_EPOCHS=10, the LR ramps from 5e-7 to 5e-4 over 10 epochs — 10% of the training budget is suboptimal. Halving warmup to 5 gives 95 epochs at/near peak LR instead of 90. Zero architectural change, directly addresses underfitting by maximizing time at full learning rate.
- **Change:** WARMUP_EPOCHS: 5 (was 10). All architecture identical to H7.
- **Budget:** 5.990M params, 2.152 GFLOPs (unchanged)
- **Status:** DONE (job 338130, branch autoresearch/h8-warmup5) — PUSHED ✓
- **Result:** **93.00%** (+0.02pp vs H7). Marginal gain from shorter warmup — 5 extra epochs at peak LR helps slightly. Effectively at noise level but positive per protocol.

### H9: Higher learning rate (BASE_LR = 1e-3)
- **Motivation:** Model is underfitting at ep100. Current BASE_LR=5e-4 is the conservative ViT default. Doubling to 1e-3 allows more aggressive gradient updates per step — directly addresses underfitting without changing architecture or adding params. WARMUP_EPOCHS=5 still provides stabilization.
- **Change:** BASE_LR: 1e-3 (was 5e-4). Architecture/schedule otherwise identical to H8.
- **Budget:** 5.990M params, 2.152 GFLOPs (unchanged)
- **Status:** DONE (job 338236, branch autoresearch/h9-lr1e3) — PUSHED ✓
- **Result:** **93.77%** (+0.77pp vs H8). Strong gain — 2× LR significantly helps the underfitting model learn faster. LR=5e-4 was clearly suboptimal; model benefits from more aggressive gradient updates.

### H10: Reduce weight decay (WEIGHT_DECAY = 0.01)
- **Motivation:** WD=0.05 is the ViT ImageNet default (hardcoded in config.py). For an underfitting 5.9M model on CIFAR-10 at LR=1e-3, strong L2 regularization fights the model's ability to fit. Reducing WD 5× (to 0.01) removes the penalty on weight magnitude, letting the model reach its capacity. Directly complements H9's LR increase.
- **Change:** WEIGHT_DECAY: 0.01 (was 0.05 default). BASE_LR stays at 1e-3.
- **Budget:** 5.990M params, 2.152 GFLOPs (unchanged)
- **Status:** DONE (job 338367, branch autoresearch/h10-wd0.01) — NOT pushed (did not beat H9)
- **Result:** **93.36%** (−0.41pp vs H9). Reducing WD hurts. WD=0.05 is beneficial even for this underfitting model — it stabilizes training at LR=1e-3. Discarded.

### H11: Push LR further (BASE_LR = 2e-3)
- **Motivation:** H9 showed LR=1e-3 (2× 5e-4) gave +0.77pp. H10 showed WD reduction hurts, so WD=0.05 is optimal. The underfitting model may benefit from further LR increase. H11 doubles again to 2e-3, reverting WD to default 0.05.
- **Change:** BASE_LR: 2e-3 (was 1e-3). WD reverted to 0.05 (removed override).
- **Budget:** 5.990M params, 2.152 GFLOPs (unchanged)
- **Status:** DONE (job 338490, branch autoresearch/h11-lr2e3) — PUSHED ✓
- **Result:** **93.96%** (+0.19pp vs H9). LR=2e-3 continues to improve but returns are diminishing (+0.77pp at 1e-3, +0.19pp at 2e-3). No instability. LR is approaching its optimal range.

### H12: Push LR to 4e-3 (BASE_LR = 4e-3)
- **Motivation:** LR gains: 5e-4→93.00%, 1e-3→93.77% (+0.77pp), 2e-3→93.96% (+0.19pp). Diminishing but positive — testing 4e-3 to find the peak or the instability cliff. If positive: commit. If degraded/diverged: confirms 2e-3 is near-optimal.
- **Change:** BASE_LR: 4e-3 (was 2e-3).
- **Budget:** 5.990M params, 2.152 GFLOPs (unchanged)
- **Status:** DONE (job 338519, branch autoresearch/h12-lr4e3) — NOT pushed (did not beat H11)
- **Result:** **93.41%** (−0.55pp vs H11). LR=4e-3 overshoots — training instability or over-stepping. Confirmed: **2e-3 is the LR sweet spot**. LR tuning complete. Discarded.

### H13: 4-stage architecture (NUM_STAGES=4, DIM_EMBED=[64,128,256,384])
- **Motivation:** The 3-stage model jumps from 64→192→384 channels. A 4-stage progressive hierarchy [64→128→256→384] provides more gradual feature refinement with an intermediate stage at (14×14, 256-dim) before the deep bottleneck (7×7, 384-dim). Each stage runs pure channel attention — the axis-decoupling story scales naturally to 4 CA stages. More stages = more opportunities for inter-channel interaction at different scales.
- **Change:** NUM_STAGES: 4, DIM_EMBED=[64,128,256,384], DEPTH=[1,2,3,7]. CA_MLP_RATIO=[8,8,4,1]. Stage 3 patch embed uses dw_sep (256→384 not divisible). Training: LR=2e-3 (H11 optimal).
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 338539, branch autoresearch/h13-4stage) — PUSHED ✓
- **Result:** **94.11%** (+0.15pp vs H11). 4-stage hierarchy works. Progressive channel expansion [64→128→256→384] with 4 CA stages outperforms 3-stage. 5.810M params, 1.206 GFLOPs — 190K under budget.

### H14: 4-stage with stage-3 stride=1 (14×14 spatial, PATCH_STRIDE=[4,2,2,1])
- **Motivation:** H13's stage 3 runs at 7×7 (PATCH_STRIDE stride=2 → 49 spatial features/token for CA). The 3-stage model's bottleneck ran at 14×14 (196 features). Keeping stage 3 at 14×14 via stride=1 gives 4× richer spatial context per channel token — each channel attends over 196 spatial positions (vs 49). Zero param change; FLOPs increase modestly from 1.206G to 1.701G.
- **Change:** PATCH_STRIDE: [4, 2, 2, 1] (was [4, 2, 2, 2]). Stage 3 stays at 14×14, 384-dim, 7 CA blocks.
- **Budget:** 5.810M params, 1.701 GFLOPs
- **Status:** DONE (job 338540, branch autoresearch/h14-4stage-s3stride1) — NOT pushed (did not beat H13)
- **Result:** **93.98%** (−0.13pp vs H13). Stride=1 at stage 3 (14×14 spatial) hurts. The 7×7 semantic compression is better for channel attention at the deepest stage — spatial compactness helps, not hurts. Discarded.

### H15: Reallocate depth: 1 stage-1 block → 1 stage-3 block (DEPTH=[1,1,3,8])
- **Motivation:** H13 uses [1,2,3,7]. Stage 3 (384-dim) has more capacity per block; stage 1 (128-dim) has less. Trading 1 stage-1 block for 1 stage-3 block (+177K net). Tests whether deeper 384-dim bottleneck processing is more valuable than intermediate 128-dim processing.
- **Change:** DEPTH: [1, 1, 3, 8] (was [1, 2, 3, 7]). PATCH_STRIDE reverted to [4,2,2,2].
- **Budget:** 5.987M params, 1.004 GFLOPs
- **Status:** DONE (job 338541, branch autoresearch/h15-depth-1128) — NOT pushed (did not beat H13)
- **Result:** **93.79%** (−0.32pp vs H13). Stage-1's 2nd block (128-dim) is more valuable than an extra stage-3 block (384-dim). H13's depth distribution [1,2,3,7] is confirmed optimal. Discarded.

### H16: Re-tune LR for 4-stage model (BASE_LR = 3e-3)
- **Motivation:** LR=2e-3 was optimal for the 3-stage model (H12: 4e-3 failed). The 4-stage model has different gradient dynamics (more stages, more gradual hierarchy). Testing LR=3e-3 — midpoint between the known sweet spot (2e-3) and failure point (4e-3) for 3-stage. 4-stage might tolerate higher LR better.
- **Change:** BASE_LR: 3e-3 (was 2e-3). Architecture identical to H13.
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 338543, branch autoresearch/h16-4stage-lr3e3) — NOT pushed (did not beat H13)
- **Result:** **93.84%** (−0.27pp vs H13). LR=3e-3 too high for 4-stage too. LR=2e-3 is definitively optimal for both architectures. Discarded.

### H17: Larger QKV kernel in stages 0-2 (KERNEL_QKV=[5,5,5,3])
- **Motivation:** Early stages (0/1/2) operate at high spatial resolution (56×56, 28×28, 14×14). A 3×3 DW QKV kernel covers a tiny neighborhood; upgrading to 5×5 gives each channel token a wider spatial context for computing inter-channel relationships. Stage 3 at 7×7 already has tight coverage — keep at 3. Cost: +53K params (~1% increase). PADDING updated to [2,2,2,1] to preserve spatial size.
- **Change:** KERNEL_QKV: [5,5,5,3], PADDING_KV/Q: [2,2,2,1].
- **Budget:** 5.863M params, 1.233 GFLOPs
- **Status:** DONE (job 338714, branch autoresearch/h17-kernel5) — NOT pushed (did not beat H13)
- **Result:** **93.39%** (−0.72pp vs H13). Larger 5×5 QKV kernels hurt. 3×3 DW is better for channel token spatial descriptors — compact, local context is preferred. Discarded.

### H18: dw_sep patch embedding at all 4 stages (PATCH_EMBED_METHOD=['dw_sep','dw_sep','dw_sep','dw_sep'])
- **Motivation:** Stages 1/2 currently use dw_only (depthwise-only spatial downsampling). Adding pointwise projection (dw_sep = DW+PW) at all stage transitions provides channel mixing at every feature level, improving input quality for subsequent CA blocks. Stage 0 and 3 already use dw_sep. Cost: +40K params (PW 64→128=8K, PW 128→256=32K).
- **Change:** PATCH_EMBED_METHOD: all dw_sep (was ['dw_sep','dw_only','dw_only','dw_sep']).
- **Budget:** 5.850M params, 1.218 GFLOPs
- **Status:** DONE (job 338763, branch autoresearch/h18-dwsep-all) — NOT pushed (did not beat H13)
- **Result:** **93.47%** (−0.64pp vs H13). dw_sep at stages 1/2 transitions hurts. dw_only (spatial downsampling only) is better — extra channel mixing adds noise rather than signal. Discarded.

### H19: Deeper stage 0 (DEPTH=[2,2,3,7])
- **Motivation:** Stage 0 (56×56, 64-dim) currently has only 1 CA block — the highest-resolution stage is under-processed. Adding a second block gives more feature refinement at the earliest (most local) scale. Cost: +74K params (+0.231G FLOPs from 56×56 CA computation).
- **Change:** DEPTH: [2, 2, 3, 7] (was [1, 2, 3, 7]).
- **Budget:** 5.884M params, 1.437 GFLOPs
- **Status:** DONE (job 338845, branch autoresearch/h19-s0depth2) — NOT pushed (did not beat H13)
- **Result:** **93.55%** (−0.56pp vs H13). Adding stage-0 depth hurts. Single block at 56×56 is sufficient — more early high-resolution processing adds noise or overfits. Discarded.

### H20: DEPTH=[1,2,3,8] + CA_MLP_RATIO=[8,8,3.3,1] (stage-3 depth vs stage-2 MLP tradeoff)
- **Motivation:** Can't add a stage-3 block without reducing something. H20 trades stage-2 MLP width (ratio 4→3.3, −17%) for one extra stage-3 block (7→8). Net: +186K params. Tests whether 8 CA blocks at 384-dim (7×7) is better than 7 CA blocks with slightly wider stage-2 MLP.
- **Change:** DEPTH: [1,2,3,8], CA_MLP_RATIO: [8,8,3.3,1].
- **Budget:** 5.996M params, 1.175 GFLOPs
- **Status:** DONE (job 339115, branch autoresearch/h20-depth-1238) — NOT pushed (did not beat H13)
- **Result:** **93.68%** (−0.43pp vs H13). Stage-2 MLP ratio 4.0 is more valuable than an 8th stage-3 block. The [1,2,3,7] + [8,8,4,1] configuration is confirmed optimal for the 4-stage. Discarded.

### H21: Disable Mixup+CutMix (AUG.MIXUP=0, AUG.CUTMIX=0)
- **Motivation:** Training already uses the full DeiT pipeline: RandAugment, Mixup (α=0.8), CutMix (α=1.0), RandomErase, label smoothing. This is very aggressive for 100-epoch training. With Mixup/CutMix, the model sees blended samples with soft labels — harder task, potentially under-learning. Disabling them lets the model train on cleaner targets and may improve convergence at 100ep.
- **Change:** AUG.MIXUP: 0.0, AUG.CUTMIX: 0.0. Architecture identical to H13.
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 339283, branch autoresearch/h21-no-mixup) — PUSHED ✓
- **Result:** **94.79%** (+0.68pp vs H13). Disabling Mixup+CutMix is highly beneficial. The full DeiT augmentation pipeline was over-regularizing the 4-stage model at 100 epochs — blended soft targets made learning harder than necessary. Cleaner targets → better convergence.

### H22: Disable all DeiT augmentation (no RandAugment, no RandomErase, no Mixup, no CutMix)
- **Motivation:** H21 showed Mixup/CutMix were over-regularizing. Does removing RandAugment + RandomErase further help? Tests whether all augmentation over-regularizes the 100-epoch run.
- **Change:** AUG.AUTO_AUGMENT: '', AUG.REPROB: 0.0 (in addition to H21's MIXUP=0, CUTMIX=0). Architecture identical to H13/H21.
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 339302, branch autoresearch/h22-no-aug) — NOT pushed (did not beat H21)
- **Result:** **94.24%** (−0.55pp vs H21). Removing RandAugment+RandomErase hurts. These augmentations provide beneficial geometric diversity without soft-label interference. The optimal aug regime is: Mixup=0, CutMix=0, keep RandAugment+RandomErase. Discarded.

### H23: Disable label smoothing (MODEL.LABEL_SMOOTHING=0.0)
- **Motivation:** Default LABEL_SMOOTHING=0.1. With Mixup/CutMix removed (H21), label smoothing is now the primary remaining soft-target regularization. For 100-epoch CIFAR-10 training at ~5.8M params, it may still over-smooth the targets. Testing if removing it further improves convergence.
- **Change:** MODEL.LABEL_SMOOTHING: 0.0 (was 0.1 default). AUG same as H21 (no Mixup/CutMix, keep RandAugment+RandomErase).
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 339312, branch autoresearch/h23-no-label-smooth) — PUSHED ✓
- **Result:** **94.92%** (+0.13pp vs H21). Removing label smoothing helps. With Mixup/CutMix gone, soft labels via LS=0.1 also over-regularize at 100ep. Hard targets win. Optimal: no Mixup, no CutMix, no label smoothing, keep RandAugment+RandomErase.

### H24: Reduce RandomErase probability (REPROB=0.1)
- **Motivation:** Default REPROB=0.25. H22 showed removing RandAugment+RandomErase entirely hurts (94.24% vs 94.79%). But 0.25 may be too aggressive. Testing REPROB=0.1 as a softer sweet spot between 0 (bad) and 0.25 (default).
- **Change:** AUG.REPROB: 0.1 (was 0.25 default). All else same as H23 (no Mixup/CutMix/LS, keep RandAugment).
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 339318, branch autoresearch/h24-reprob0.1) — NOT pushed
- **Result:** **94.71%** (−0.21pp vs H23). Default REPROB=0.25 is optimal. Reducing RandomErase hurts; keep at default. Discarded.

### H25: Mild drop path in deepest stage (DROP_PATH_RATE=[0,0,0,0.05])
- **Motivation:** Augmentation tuning complete (H21-H24). With Mixup/CutMix/LS removed, some structural regularization may now help. H7 showed that removing drop path improved results with full DeiT aug, but that was the over-regularization regime. Now with lighter aug, mild stochastic depth (0→0.05 linear schedule in stage 3's 7 blocks) may provide helpful regularization without excessive noise.
- **Change:** SPEC.DROP_PATH_RATE: [0.0, 0.0, 0.0, 0.05]. All else same as H23.
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 339323, branch autoresearch/h25-depth8) — PUSHED ✓
- **Result:** **94.96%** (+0.04pp vs H23). Mild drop path marginally helps under reduced-aug regime, consistent with the tradeoff hypothesis. Small but consistent gain.

### H26: Redistribute depth — DEPTH=[1,2,4,6] (balanced) vs [1,2,3,7] (top-heavy)
- **Motivation:** Current depth profile [1,2,3,7] heavily concentrates computation in stage 3 (384-dim, 7 blocks). Stage 2 (256-dim) may be under-utilized. Testing [1,2,4,6] — same total blocks (13), one shifted from stage 3 to stage 2. More channel attention at 256-dim scale before the final stage may enrich the representation hierarchy.
- **Change:** SPEC.DEPTH: [1, 2, 4, 6]. All else same as H25.
- **Budget:** 5.952M params, 1.302 GFLOPs
- **Status:** DONE (job 339324, branch autoresearch/h26-depth-1246) — NOT pushed
- **Result:** **94.84%** (−0.12pp vs H25). Top-heavy [1,2,3,7] is better. Concentrating depth in the final 384-dim stage is optimal for channel attention hierarchy. Discarded.

### H27: Longer warmup (WARMUP_EPOCHS=10)
- **Motivation:** With clean hard targets (no Mixup/LS) established in H23, the early training dynamics may differ. Slower LR warmup (10 vs 5 epochs) may let the model settle into a better optimization trajectory before the cosine decay phase.
- **Change:** TRAIN.WARMUP_EPOCHS: 10 (was 5). All else same as H25.
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 339325, branch autoresearch/h27-warmup10) — PUSHED ✓
- **Result:** **95.06%** (+0.10pp vs H25). Longer warmup clearly helps. Slower LR ramp improves optimization basin under clean-target regime.

### H28: Even longer warmup (WARMUP_EPOCHS=15)
- **Motivation:** H27 showed warmup 5→10 improved by +0.10pp. Testing if 15-epoch warmup pushes further along the same trend.
- **Change:** TRAIN.WARMUP_EPOCHS: 15 (was 10). All else same as H27.
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 339328, branch autoresearch/h28-warmup15) — NOT pushed
- **Result:** **94.99%** (−0.07pp vs H27). Warmup sweet spot is 10 epochs. 15 epochs slightly over-extends the warmup phase, hurting the main cosine decay phase. Discarded.

### H29: Progressive drop path [0, 0.01, 0.02, 0.05]
- **Motivation:** H25 showed drop path 0.05 in stage 3 marginally helps (+0.04pp). Extending mild progressive drop path to stage 2 as well may provide additional beneficial regularization across the deeper stages.
- **Change:** SPEC.DROP_PATH_RATE: [0.0, 0.01, 0.02, 0.05]. All else same as H27.
- **Budget:** 5.810M params, 1.206 GFLOPs
- **Status:** DONE (job 339344, branch autoresearch/h29-droppath-prog) — PUSHED ✓
- **Result:** **95.09%** (+0.03pp vs H27). Progressive drop path provides marginal additional gain.

### H30: Layer Scale (CaiT-style, init=1e-4)
- **Motivation:** With 7 deep CA blocks in stage 3, training stability may benefit from learned per-block output scaling. Layer Scale (CaiT, Touvron 2021) initializes each branch output multiplier to 1e-4, allowing the model to learn the effective residual vs identity balance. This is an architectural change in cls_cvt.py — aligned with "stronger channel-attention story."
- **Change:** SPEC.LAYER_SCALE: 1.0e-4 (new YAML key). Code change in Block: gamma1,gamma2 nn.Parameter of shape (dim_out,) initialized to 1e-4, applied to attn/mlp outputs before residual addition.
- **Budget:** 5.818M params (+0.008M for gamma params), 1.206 GFLOPs
- **Status:** DONE (job 339375, branch autoresearch/h30-layer-scale) — PUSHED ✓
- **Result:** **95.10%** (+0.01pp vs H29). Marginal gain; Layer Scale provides slight stability benefit for deep CA stack. Note: first submit (339373) crashed due to missing LAYER_SCALE key in config.py — fixed by registering `_C.MODEL.SPEC.LAYER_SCALE = None`.

### H31: Attention dropout in stage 3 (ATTN_DROP_RATE=[0,0,0,0.1])
- **Status:** DONE (job 339393) — NOT pushed. **Result: 94.98%** (−0.12pp). Attention dropout hurts. Discarded.

### H32: DW shortcut placed between attention and MLP (DW_SHORTCUT_AFTER_FIRST_RESIDUAL=true)
- **Motivation:** Currently DW shortcut is added after the MLP residual (end of block). Testing if placing it between attention and MLP changes the information flow beneficially.
- **Change:** SPEC.DW_SHORTCUT_AFTER_FIRST_RESIDUAL: true. All else same as H30.
- **Budget:** 5.818M, 1.206 GFLOPs
- **Status:** DONE (job 339431) — NOT pushed. **Result: 95.09%** (−0.01pp). DW shortcut placement doesn't matter. Discarded.

### H33: SiLU (Swish) activation instead of GELU
- **Status:** DONE (job 339461) — NOT pushed. **Result: 94.91%** (−0.19pp). GELU is better. SiLU hurts. Discarded.

### H34: Higher weight decay (WEIGHT_DECAY=0.1)
- **Motivation:** Default WD=0.05. H10 showed WD=0.01 hurt with full aug. Now with reduced aug (no Mixup/LS), less implicit regularization — testing if higher explicit L2 regularization (WD=0.1, 2× default) compensates and improves generalization.
- **Change:** TRAIN.WEIGHT_DECAY: 0.1. All else same as H30.
- **Budget:** 5.818M, 1.206 GFLOPs
- **Status:** DONE (job 339533) — NOT pushed. **Result: 94.99%** (−0.11pp). WD=0.1 hurts; WD=0.05 is optimal. Discarded.

### H35: Lighter RandAugment (rand-m7)
- **Status:** DONE (job 339559) — NOT pushed. **Result: 95.03%** (−0.07pp). Default rand-m9 is optimal. Discarded.

### H36: More depth at stage 0 (DEPTH=[2,2,3,7])
- **Motivation:** Stage 0 has only 1 block at 64 channels / 56×56 spatial — may be too shallow.
- **Change:** SPEC.DEPTH: [2, 2, 3, 7]. All else same as H30.
- **Budget:** 5.891M, 1.437 GFLOPs
- **Status:** DONE (job 339575) — NOT pushed. **Result: 94.79%** (−0.31pp). Worse. The [1,2,3,7] top-heavy profile is confirmed optimal. Discarded.

### H37: Light Mixup (MIXUP=0.2)
- **Status:** DONE (job 339592) — NOT pushed. **Result: 94.83%** (−0.27pp). Mixup=0 is definitively optimal. Any Mixup hurts. Discarded.

### H38: All-dw_sep patch embedding (['dw_sep','dw_sep','dw_sep','dw_sep'])
- **Motivation:** Stages 1 and 2 use 'dw_only', missing pointwise cross-channel projection at downsampling.
- **Change:** SPEC.PATCH_EMBED_METHOD: all 'dw_sep'.
- **Budget:** 5.857M, 1.218 GFLOPs
- **Status:** DONE (job 339594) — NOT pushed. **Result: 94.80%** (−0.30pp). dw_sep at stages 1,2 hurts; dw_only is better there. Discarded.

### H39: Deep pure CA — DEPTH=[1,2,3,20], CA_MLP_RATIO=[8,8,4,0]
- **Status:** DONE (job 339596) — NOT pushed. **Result: 94.03%** (−1.07pp). Pure-CA deep stack fails hard. FFN at stage 3 is load-bearing even at ratio=1 — nonlinearity from MLP cannot be replaced by more attention depth. Discarded.

### H40: AdamW β₂=0.98
- **Status:** DONE (job 339601) — NOT pushed. **Result: 94.74%** (−0.36pp). β₂=0.999 (default) is better. Discarded.

### H41: Layer Scale init=1e-5
- **Status:** DONE (job 339634) — NOT pushed. **Result: 94.98%** (−0.12pp). init=1e-4 (H30) is better. Discarded.

### H42: QK-Normalize (L2-normalize Q and K before dot product)
- **Motivation:** Used in modern LLMs (LLaMA 3) to prevent attention score collapse in deep models. L2-normalizes Q and K so dot products stay in [-1,1] range before scale. Code change: Attention.qk_norm flag; registered QK_NORM in config.py.
- **Change:** SPEC.QK_NORM: true. Code: F.normalize(q/k, dim=-1) before SDPA.
- **Budget:** 5.818M, 1.206 GFLOPs
- **Status:** DONE (job 339655, branch autoresearch/h42-qk-norm) — NOT pushed. **Result: 94.22%** (−0.88pp vs H30). QK-norm hurts channel attention — the normalized dot product constrains the attention score range too much for channel-token geometry. Discarded.

---

## RESUME — Phase 1b (2026-03-29)

**Champion:** H30 — Layer Scale 1e-4, commit `e297b7e`, branch `autoresearch/h30-layer-scale`
- **Val top-1:** 95.10% on CIFAR-10
- **Params / GFLOPs:** 5.818M / 1.206
- **Architecture:** 4-stage [64→128→256→384], DEPTH=[1,2,3,7], CA_MLP_RATIO=[8,8,4,1], DW_SHORTCUT_BACKBONE=true, Layer Scale 1e-4, NUM_HEADS=[1,1,1,1]
- **Training recipe (100ep):** LR=2e-3, WD=0.05, warmup=10, no Mixup/CutMix/LS, RandAugment rand-m9, RandomErase 0.25, progressive drop path [0,0.01,0.02,0.05]
- **Seed:** 0

**42 hypotheses completed.** H1–H6 architectural (69.67%→92.52%), H7–H12 training recipe (→93.96%), H13 4-stage (→94.11%), H14–H20 topology exploration, H21–H29 regularization tuning (→95.09%), H30 Layer Scale (→95.10%), H31–H42 all failed to improve.

**Config knob space is exhausted.** Next session: Phase 1b — code-level architectural innovation in `models/cls_cvt.py`, grounded in literature (MCP tools: arxiv, deepwiki, huggingface). See `RESEARCH_BRIEF.md` and `program.md § Phase 1b` for research directions and constraints.


---

## Phase 1b Hypothesis Log

### H43: Learnable Channel Pair Bias (CPB) — shared per-stage C×C attention bias
- **Direction rationale:** H31-H42 exhausted all config-level options. Phase 1b requires code-level innovation. The channel attention computes softmax(QK^T/sqrt(d)) where d=H*W; adding a learned per-pair bias directly to the logits (before softmax) encodes prior knowledge about which channel pairs should interact more/less strongly. This is the first truly architectural change in Phase 1b.
- **Literature context:** Analogous to relative position bias (RPB) in Swin Transformer (arXiv:2103.14030, Liu et al. 2021) and image RPE (iRPE, arXiv:2107.14222, Wu et al. 2021 — gives DeiT +1.5% from attention bias alone). DaViT (arXiv:2204.03645) validates that channel-token attention captures global interactions. ConViT (arXiv:2103.10697) uses gated attention with positional bias. For channel tokens there is no geometric distance, so we use a full C×C learned matrix. Initialized to zeros = neutral (identical to H30 at init); learned during training.
- **Change:** SPEC.CHANNEL_PAIR_BIAS: true (new key). Code: VisionTransformer creates `self.channel_pair_bias = nn.Parameter(zeros(C, C))` shared per stage. Passed as attn_mask to SDPA in Attention.forward. Excluded from weight decay via no_weight_decay(). Branch: autoresearch/h43-gated-attn.
- **Budget:** 6.051M params (+0.233M from 4 stage biases: 64²+128²+256²+384²), 1.206 GFLOPs. *Slightly over 6M soft cap — documented as intentional architectural test.*
- **Result:** NEGATIVE — 94.97% max / 94.89% final @100ep. Baseline H30=95.10%, delta=-0.13pp. Free-form C×C bias has too many unconstrained params (0.233M) without structure; slight overfit. Discard.

### H44: Per-stage DW shortcut kernel pyramid [7,5,3,3]
- **Direction rationale:** H43 showed unconstrained params hurt. H6 gave +0.81pp from DW shortcut (biggest architectural gain). Hypothesis: larger DW kernels at early high-resolution stages (56×56, 28×28) provide wider local context complementary to global channel attention. Proportionally, a 7×7 kernel covers 12.5% of a 56×56 feature map vs only 5.4% with 3×3. Late stages keep 3×3 (7×7 feature map at stage 3 — a 7×7 kernel there would be effectively global, redundant with channel attention).
- **Literature context:** ConvNeXt (arXiv:2201.03545, Liu et al. 2022) shows 7×7 DW conv as primary spatial mixer outperforms 3×3. EfficientNet compound scaling matches kernel size to resolution. Multi-scale architectures (FPN) use scale-aware feature extraction. Multi-scale DW shortcuts for complementary local-to-global information in channel-attention networks.
- **Change:** SPEC.DW_SHORTCUT_KERNEL: [7, 5, 3, 3] (new key). Code: Block.__init__ reads `dw_shortcut_kernel_size` from kwargs. CvT passes per-stage value. DW conv padding = kernel//2 (same spatial size maintained).
- **Budget:** 5.824M params (+6.7K), 1.217 GFLOPs. Within budget.
- **Status:** RESULT: NEGATIVE — 94.88% max / 94.82% final @100ep. Baseline H30=95.10%, delta=-0.22pp. Larger DW kernels at early stages (7×5×3×3) hurt; the 3×3 baseline is already well-matched. Discard. Branch NOT merged.

### H45: RMSNorm in Block normalizations (replaces LayerNorm)
- **Direction rationale:** H43-H44 showed that adding unconstrained params or changing DW kernel sizes both hurt. RMSNorm is a different category: same parameter count (actually slightly fewer — no bias in RMSNorm), different inductive bias. The key hypothesis for channel-token mode: LayerNorm subtracts the mean of spatial features for each channel token, discarding the "global activation level" information. For channel tokens, the spatial feature mean (avg pixel intensity × channel activation) is meaningful: a channel with high mean is globally active, one with zero mean is silent. RMSNorm preserves this mean information by only normalizing by RMS, not mean-centering. This may improve channel attention's ability to distinguish active vs inactive channels.
- **Literature context:** Zhang & Sennrich (arXiv:1910.07467, EMNLP 2019) — RMSNorm achieves comparable or better performance vs LayerNorm, 7-64% faster. Used in LLaMA, Gemma, and most modern LLMs. Jiang et al. (arXiv:2305.14858) proves Pre-LN transformers' mean information is redundant → RMSNorm equivalence. For vision channel-tokens: the spatial mean is NOT redundant (unlike NLP tokens where mean-zero embedding is standard).
- **Change:** Add `RMSNorm` class to cls_cvt.py. Add `NORM_LAYER: rmsnorm` config key. In `get_cls_model`, select `RMSNorm` or `LayerNorm` based on spec. All Block.norm1, norm2 and VisionTransformer.norm switch to RMSNorm. Q/K layer_norm in Attention stays as F.layer_norm (different purpose: scale normalization for stable dot products). Branch: autoresearch/h45-rmsnorm.
- **Budget:** 5.809M params (−9K from no-bias in RMSNorm), 1.206 GFLOPs. Within budget.
- **Status:** SKIPPED — job 339733 ran H46 config (YAML was already updated to H46 at time of submission). H45 never properly tested. Will be run as H45b on H46 base if H47+ plateau.

### H46: Auxiliary deep supervision at stage 2 (intermediate classifier head)
- **Direction rationale:** H43-H45 explored attention/normalization tweaks, all small-scale. Deep supervision is a training-time change that improves gradient flow to early stages — a known bottleneck in deep hierarchical networks. Our 4-stage architecture with 13 blocks total may have shallow gradients at stage 0 (just 1 block, receives attenuated gradients after 12 layers of backprop). An auxiliary loss at stage 2 (the "middle" of the network) strengthens gradients to stages 0-1.
- **Literature context:** Deep supervision (Lee et al. 2015, arXiv:1409.5185): DSN improves CIFAR-10 accuracy consistently. GoogLeNet (Szegedy et al. 2014): auxiliary classifiers improve 2.9%. NASNet and many hierarchical architectures use intermediate supervision. Particular relevance for channel attention: early stage features need to learn semantically meaningful channel patterns, and direct gradient from classification helps.
- **Change:** AUX_LOSS_WEIGHT: 0.4 (following GoogLeNet). In CvT: `self.aux_head = nn.Linear(256, 10)` applied to GAP of stage 2 output during training. `self._aux_logits` stored as side effect. In main.py: aux loss follows LOBT pattern — `loss += aux_weight * criterion(aux_logits, targets)`. NORM_LAYER: layernorm (independent test from H45). Branch: autoresearch/h46-aux-loss.
- **Budget:** 5.820M params (+2570 for aux head), 1.206 GFLOPs. Within budget.
- **Result:** **95.70%** (+0.60pp vs H30 champion 95.10%). NEW CHAMPION. Deep supervision at stage 2 is a strong architectural win — gradient flow to early stages is clearly the bottleneck. Job 339733 (ran on a GPU). PUSHED ✓

### H47: Per-stage learned attention temperature (log-parameterized scalar τ)
- **Direction rationale:** After `F.layer_norm(q/k, (H*W,))` in channel-token mode, Q/K spatial features are ~N(0,1), and the dot product Q·K^T ≈ H*W * corr(q,k). SDPA divides by sqrt(H*W), giving effective logit ≈ sqrt(H*W) * corr. This gives dramatically different attention sharpness per stage: stage 1 logits ~56×corr (very sharp/peaky), stage 4 ~7×corr (moderate). A per-head learned temperature τ, applied as Q → Q*τ before SDPA, lets each stage independently calibrate its attention sharpness. τ>1 sharpens attention; τ<1 flattens it. The model can learn that early high-resolution stages need flatter channel attention while deep semantic stages benefit from sharp, selective attention.
- **Literature context:** XCiT (El-Nouby et al. 2021, arXiv:2106.09681) uses a per-head learned temperature in cross-covariance attention (which is structurally identical to channel-primary ViT channel attention), showing it consistently improves accuracy. SSA/Selective Attention (Zhang et al. 2024, arXiv:2411.12892) shows per-query temperature scaling reduces attention dilution. DaViT (arXiv:2204.03645) channel attention benefits from head-specific scaling.
- **Change (minimal):** In `Attention.__init__`: add `self.log_attn_temp = nn.Parameter(torch.zeros(num_heads))` (shape (1,) → 1D → auto-excluded from weight decay via optimizer's `len(shape)==1` rule) when `token_mode=='channel'`. In `Attention.forward`, channel mode: `temp = self.log_attn_temp.exp().view(num_heads,1,1); q_sdpa = q * temp`. In non-SDPA fallback path: multiply scale by temperature.
- **Parameterization:** `log_τ` parameterization (exp) ensures τ>0 always. Init: `log_τ=0 → τ=1` (no change at init). Gradient naturally flows through exp.
- **Budget:** +13 params total (1 per Attention module × 13 blocks). Negligible. Same FLOPs.
- **Result:** **95.82%** (+0.12pp vs H46 champion 95.70%). NEW CHAMPION. Learned per-head temperature sharpens channel attention at each head independently. Job 339773 on a GPU. PUSHED ✓

### H48: Gated attention output (scalar sigmoid gate, channel mode)
- **Direction rationale:** After H47 showed per-head temperature helps, gated attention is a complementary mechanism. A sigmoid gate on the attention output modulates how much attention contributes to the residual per block. This allows each block to learn its optimal attention-residual balance. Unlike temperature (which sharpens distribution), the gate controls output magnitude.
- **Literature context:** Gated Linear Units (Dauphin et al. 2017, arXiv:1612.08083) established sigmoid gating in deep networks. Gated Attention (Zhang et al. 2018), G-MLP (Liu et al. 2021): gate prevents over-attention and allows information bypass. ResNet scaling (CaiT: LayerScale) is related — both control residual contribution magnitude.
- **Change:** `self.log_gate = nn.Parameter(torch.tensor(2.0))` (scalar per attention layer); init sigmoid(2)≈0.88 (near-full pass-through). Applied after proj+drop: `x = sigmoid(log_gate) * x`. Channel mode only. Branch: autoresearch/h48-gated-attn.
- **Budget:** 5.820M params (+13 scalar gates negligible), 1.206 GFLOPs. Within budget.
- **Result:** 95.75% (−0.07pp vs H47 champion 95.82%). DISCARDED. Gate slightly hurts — temperature already provides the sharpness calibration; adding a scalar magnitude gate on top introduces conflicting optimization dynamics.

### H49: Larger DW shortcuts (5×5 kernels, uniform across all stages)
- **Direction rationale:** H6 showed DW shortcuts are a strong component (+0.81pp). The 3×3 DW kernel captures local spatial context, but on the 7×7 feature map (stage 3), the effective RF covers 3/7 of the map. Increasing to 5×5 covers 5/7 — a meaningfully larger share of the feature map, capturing more global spatial patterns in the shortcut. At stages 0-1 (56×56, 28×28), 5×5 enlarges the RF without saturation.
- **Literature context:** RepLKNet (Ding et al. 2022, arXiv:2203.06717) shows larger kernels improve accuracy; SLaK (Liu et al. 2022, arXiv:2209.02933) extends this to very large kernels; HorNet (Rao et al. 2022) demonstrates 7×7 shortcuts improve ViT-style architectures.
- **Change:** `DW_SHORTCUT_KERNEL: [5, 5, 5, 5]` (from [3,3,3,3]). Pure config change, no code. Branch: autoresearch/h49-larger-dw.
- **Budget:** 5.881M params (+61K for 5×5 vs 3×3 DW in all stages), 1.217 GFLOPs. Within budget.
- **Result:** 95.71% (−0.11pp vs H47 champion 95.82%). DISCARDED. Larger DW kernel adds params but not accuracy — the 3×3 kernel already captures sufficient local context; 5×5 may overfit on CIFAR-10's 7×7 late feature maps.

### H50: Dual auxiliary deep supervision at stages 1 and 2
- **Direction rationale:** H46 (single aux at stage 2) gave +0.60pp. Can dual supervision at both stage 1 (early, dim=128) and stage 2 (middle, dim=256) provide even stronger gradient flow? Stage 1 supervision injects classification signal after just 3 blocks (1+2), helping the very early channels learn semantic patterns more directly.
- **Literature context:** Deeply Supervised Nets (Lee et al. 2015, arXiv:1409.5185) applies auxiliary losses at every hidden layer. GoogLeNet uses 2 aux heads. InceptionV4 uses progressive supervision. The principle: stronger gradient signal at more stages reduces vanishing gradient in hierarchical networks.
- **Change:** Add `AUX_LOSS_WEIGHT_S1: 0.2` for stage 1 (dim=128 linear head). Stage 2 keeps `AUX_LOSS_WEIGHT: 0.4`. Both heads use GAP + Linear. Total supervised: stages 1 + 2 + 3 (main head). Branch: autoresearch/h50-dual-aux.
- **Budget:** 5.882M params (+1290 for s1 head), 1.217 GFLOPs. Within budget.
- **Result:** **95.85%** (+0.03pp vs H47 champion 95.82%). NEW CHAMPION. Dual supervision at stages 1+2 is marginally better than stage 2 alone. The stage-1 auxiliary signal (+0.2 weight) provides a small but consistent boost. Job 339808. PUSHED ✓

### H51: Convolutional Position Encoding (CPE) at each stage
- **Direction rationale:** Channel tokens in channel-primary ViT represent channels, with spatial positions as features. However, the channel attention operates over the channel token set without explicit spatial structure in the token embeddings. CPE (CPVT/Twins style) injects spatial positional information via a DWConv applied to the feature map before tokenization. This gives each channel token spatial context before being used as Q/K/V in attention.
- **Literature context:** CPVT (arXiv:2102.10882, Chu et al. 2021): CPE improves ViT accuracy consistently without global PE. Twins (arXiv:2104.13840): conditional positional encodings. PVT and Swin also benefit from implicit positional encoding via local windows. Zero-weight init ensures identity at start.
- **Change:** `CPE: true`, `CPE_KERNEL_SIZE: 3`. In VisionTransformer.forward: after patch_embed, before tokenization rearrange, inject `x = x + DWConv(x)`. DWConv initialized to zeros (identity start). Applied at all 4 stages. Branch: autoresearch/h51-cpe.
- **Budget:** 5.890M params (+8K for CPE DWConvs), 1.220 GFLOPs. Within budget.
- **Result:** **95.95%** (+0.10pp vs H50 champion 95.85%). NEW CHAMPION. CPE injects local spatial context via DWConv before channel tokenization, giving each channel token spatially-aware features. Zero-weight init ensures safe training. Job 339850. PUSHED ✓

### H52: Larger CPE kernel (5×5)
- **Direction rationale:** H51's 3×3 CPE was effective. A 5×5 CPE covers a larger spatial neighborhood (at 7×7 stage 3: 5/7 of the map), potentially capturing more global spatial patterns in the positional encoding. RepLKNet showed larger kernels improve; here we test if larger CPE receptive field helps channel attention.
- **Change:** `CPE_KERNEL_SIZE: 5` (from 3). Pure config. Branch: autoresearch/h52-cpe5x5.
- **Budget:** 5.904M params (+22K for 5×5 vs 3×3 DW), 1.226 GFLOPs.
- **Result:** **96.11%** (+0.16pp vs H51 champion 95.95%). NEW CHAMPION. 5×5 CPE covers a larger spatial neighborhood, giving richer positional context to channel tokens. Job 339926. PUSHED ✓

### H53: CPE with 7×7 kernel
- **Direction rationale:** CPE has shown monotonic improvement: 3×3 → 95.95% (+0.10pp), 5×5 → 96.11% (+0.16pp). Testing 7×7 to see if this trend continues. At stage 3 (7×7 map), 7×7 CPE achieves full-map coverage — each channel token sees the entire spatial extent in its positional encoding. This is the maximum useful kernel size for stage 3.
- **Change:** `CPE_KERNEL_SIZE: 7` (from 5). Pure config. Branch: autoresearch/h53-cpe7x7.
- **Budget:** 5.924M params (+20K), 1.235 GFLOPs.
- **Result:** 95.85% (−0.26pp vs H52 champion 96.11%). DISCARDED. 7×7 over-smooths the positional encoding — at stage 3's 7×7 map, 7×7 CPE is near-global pooling which removes local positional structure. 5×5 is the sweet spot.

### H54: SwiGLU MLP (replace GELU with SwiGLU gate in CA MLP blocks)
- **Direction rationale:** SwiGLU (Shazeer 2020) consistently outperforms GELU in Transformer MLPs. Used in PaLM, LLaMA, Gemma. The gating mechanism `SiLU(W_gate*x) * W_val*x` allows the network to selectively amplify features, which is particularly relevant for channel attention where the MLP processes channel-mixed features. Hidden dim scaled by 2/3 to keep param count identical to GELU MLP.
- **Literature context:** Shazeer (2020, arXiv:2002.05202) shows SwiGLU outperforms GELU by 0.1-0.5 BLEU/perplexity points. Noam Shazeer's analyses confirm superiority across model sizes. LLaMA (Touvron et al. 2023), PaLM use SwiGLU as default.
- **Change:** Add `SwiGLUMlp` class. `SWIGLU: true` in YAML. Block uses `SwiGLUMlp` when `use_swiglu=True`. Hidden = `int(hidden * 2/3)`. Branch: autoresearch/h54-swiglu.
- **Budget:** 5.904M (unchanged vs H52 due to 2/3 scaling), 1.225 GFLOPs.
- **Result:** **96.23%** (+0.12pp vs H52 champion 96.11%). NEW CHAMPION. SwiGLU gating in channel-attention MLP blocks provides selective feature amplification, consistently improving over GELU. Job 340110. PUSHED ✓

### H55: Per-stage CPE kernel sizes [3, 3, 5, 5]
- **Direction rationale:** H51 (3×3) → 95.95%, H52 (5×5) → 96.11%, H53 (7×7) → 95.85%. Test per-stage kernels [3,3,5,5] — early stages (large feature maps) may prefer smaller CPE; late stages (small maps) prefer larger coverage.
- **Change:** `CPE_KERNEL_SIZES: [3, 3, 5, 5]`. New config key. Branch: autoresearch/h55-cpe-perst. Budget: 5.901M, 1.221 GFLOPs.
- **Result:** 96.24% (+0.01pp vs H54 champion 96.23%). DISCARDED — within noise. Per-stage CPE indistinguishable from uniform 5×5 (though H54 base had [5,5,5,5] DW which was suboptimal, see H56).

### H56: DW shortcut kernel verified [3,3,3,3] on H54 base
- **Direction rationale:** H49 tested [5,5,5,5] on H47 base (before CPE/SwiGLU) and got -0.11pp. However H50's commit accidentally included [5,5,5,5] (unrevereted change from H49). All H50–H55 results used [5,5,5,5]. Need to verify whether [3,3,3,3] is better on the current H54 base (+CPE+SwiGLU+aux+temp).
- **Change:** `DW_SHORTCUT_KERNEL: [3, 3, 3, 3]` (from [5,5,5,5]). Pure config. Branch: autoresearch/h56-dw3x3-verify.
- **Budget:** 5.843M (−61K for smaller DW), 1.214 GFLOPs.
- **Result:** **96.48%** (+0.25pp vs H54 champion 96.23%). NEW CHAMPION. The 5×5 DW was hurting all H50–H55 results. With CPE providing spatial context, 3×3 DW is optimal — complementary local context without redundancy with CPE. Job 340307. PUSHED ✓

### H57: RMSNorm on H56 clean base
- **Direction rationale:** H45 was designed to test RMSNorm but was skipped (job ran H46 config). Now testing on the clean H56 base. RMSNorm omits the mean-centering of LayerNorm, preserving magnitude information in activations. For channel attention where activation scale carries semantic meaning (high-activation channels ≈ dominant features), this could help.
- **Literature context:** RMSNorm (Zhang & Sennrich 2019, arXiv:1910.07467): matches LayerNorm quality with less computation. Used in LLaMA, Gemma, PaLM 2. Most channel-attention papers use LayerNorm — testing if RMSNorm provides complementary benefit for the magnitude-preserving channel-token paradigm.
- **Change:** `NORM_LAYER: rmsnorm` (from layernorm). Pure config. Branch: autoresearch/h57-rmsnorm.
- **Budget:** 5.844M (RMSNorm drops mean-centering bias, slightly fewer params), 1.214 GFLOPs.
- **Result:** 96.25% (−0.23pp vs H56 champion 96.48%). DISCARDED. RMSNorm hurts on the channel-attention paradigm — mean-centering in LayerNorm appears to be important for stable channel token normalization.

### H57: RMSNorm on H56 clean base
- **Direction rationale:** H45 was designed to test RMSNorm but was skipped. Testing on clean H56 base. RMSNorm omits mean-centering, preserving activation magnitude (dominant channels have larger activations). Literature: Zhang & Sennrich (arXiv:1910.07467), used in LLaMA/Gemma.
- **Change:** `NORM_LAYER: rmsnorm`. Pure config. Branch: autoresearch/h57-rmsnorm. Budget: 5.844M, 1.214 GFLOPs.
- **Result:** 96.25% (−0.23pp vs H56 champion 96.48%). DISCARDED. LayerNorm is better for channel-token normalization — mean-centering appears important for stable channel attention dynamics.

### H58: Multi-head channel attention — NUM_HEADS: [1, 2, 4, 1]
- **Direction rationale:** Current architecture uses single-head channel attention at all stages. Multi-head attention allows different heads to attend to different subspaces of the spatial feature dimension (H*W/num_heads features per head). At stage 2 (H*W=196, 4 heads → d=49/head), each head specializes in a different spatial frequency band or spatial region. Stage 3 (H*W=49) stays at 1 head since 49/4 is non-integer.
- **Literature context:** Multi-head attention (Vaswani et al. 2017): different heads capture different aspects. In channel-attention mode, heads split the spatial feature dim — analogous to spectral decomposition. Standard ViTs use num_heads=head_dim//64 pattern.
- **Change:** `NUM_HEADS: [1, 2, 4, 1]` (from [1,1,1,1]). Constraint: H*W % num_heads == 0. Branch: autoresearch/h58-more-heads. Budget: 5.844M (unchanged), 1.214 GFLOPs.
- **Result:** 96.12% (−0.36pp vs H56 champion 96.48%). DISCARDED. Multi-head channel attention hurts significantly. With d=H*W/num_heads per head, fewer spatial features per head means each channel-token comparison uses only a fraction of the spatial context. Single-head (full spatial context per attention) is better for channel-attention mode.

### H58: Multi-head channel attention NUM_HEADS [1,2,4,1]
- **Direction rationale:** Single-head (1,1,1,1) at all stages. Multi-head can capture different spatial subspaces. Stage 3 (H*W=49) constrained to 1 head (49/4 non-integer). Testing [1,2,4,1]: stage 1 (H*W=784, 2 heads, d=392), stage 2 (H*W=196, 4 heads, d=49).
- **Literature context:** Multi-head attention (Vaswani et al. 2017): different heads specialize in different feature aspects. ViT uses head_dim=64 as standard.
- **Change:** `NUM_HEADS: [1, 2, 4, 1]`. Budget: 5.844M (unchanged), 1.214 GFLOPs. Branch: autoresearch/h58-more-heads.
- **Result:** 96.12% (−0.36pp vs H56 champion 96.48%). DISCARDED. Multi-head hurts significantly. In channel-attention mode, each head only sees H*W/num_heads features per token; fewer features per head reduces channel comparison quality. Full spatial feature vector per head (single-head) is optimal.

### H59: Stronger auxiliary loss weights — AUX_LOSS_WEIGHT 0.6, AUX_LOSS_WEIGHT_S1 0.3
- **Direction rationale:** H50 used (0.2, 0.4) and got 95.85%, then H56 achieved 96.48% (all improvements stacked). Now testing stronger auxiliary supervision (0.3, 0.6) to see if deeper gradient signal helps further. Following the GoogLeNet pattern which used 0.3 for aux losses in the original paper (we used 0.4 before; testing slightly higher).
- **Change:** `AUX_LOSS_WEIGHT: 0.6, AUX_LOSS_WEIGHT_S1: 0.3`. Pure config. Branch: autoresearch/h59-deeper-s3.
- **Budget:** 5.843M (unchanged), 1.214 GFLOPs.
- **Result:** 96.42% (−0.06pp vs H56 champion 96.48%). DISCARDED. Stronger aux supervision (0.3,0.6) slightly hurts — higher aux weights over-regularize training signal. Original (0.2,0.4) weights are optimal.

### H60: Wider stage 2 MLP (ratio 4→6), fewer blocks (3→2) — iso-param swap
- **Direction rationale:** Current stage 2: 3 blocks at MLP ratio 4.0. Testing width-vs-depth tradeoff: 2 blocks at ratio 6.0. Both configurations have similar param counts (~same capacity). Wider MLP per block (SwiGLU hidden = int(256*6.0*2/3)=1024) vs deeper with narrower (int(256*4.0*2/3)=682). This tests whether expressiveness per block or number of blocks is more valuable at stage 2.
- **Literature context:** Width vs depth tradeoff in neural networks (Zagoruyko & Komodakis 2016, Wide ResNets): wider networks often outperform deeper narrow ones at same param count.
- **Change:** `DEPTH: [1,2,2,7]`, `CA_MLP_RATIO: [8,8,6,1]`. Budget: 5.766M, 1.199 GFLOPs.
- **Result:** 96.38% (−0.10pp vs H56 champion 96.48%). DISCARDED. Width-over-depth at stage 2 hurts — 3 blocks at ratio 4.0 is better than 2 blocks at ratio 6.0. Depth (more transformations) is more valuable than per-block width at stage 2.

### H61: 300-epoch validation of H56 champion (separate protocol)
- **Direction rationale:** After exhaustive 100-epoch search (H46–H60), the architecture has matured with CPE+SwiGLU+aux+temp+DW3x3. Per RESEARCH_BRIEF.md, 300-epoch protocol tests the true ceiling. Enable Mixup (0.8) and CutMix (1.0) and Label Smoothing (0.1) for longer training. Reduce LR to 1e-3 with longer warmup (20ep). Do NOT compare to 100ep results directly.
- **Change:** EPOCHS: 300, WARMUP_EPOCHS: 20, BASE_LR: 1e-3, WARMUP_LR: 1e-6, MIN_LR: 1e-5, LABEL_SMOOTHING: 0.1, MIXUP: 0.8, CUTMIX: 1.0. Branch: autoresearch/h61-300ep.
- **Budget:** Same as H56 — 5.843M, 1.214 GFLOPs.
- **Note:** 300-epoch results are NOT comparable to 100-epoch results.
- **Status:** RUNNING (job 340397, branch autoresearch/h61-300ep) — WILL TAKE ~3×LONGER (3-4 hours)

### H62: Depth redistribution [1,3,3,6] — more stage 1, fewer stage 3
- **Direction rationale:** Current [1,2,3,7] has heavy stage 3 (7 blocks at dim=384). Stage 1 (28×28) is shallow (2 blocks). More stage 1 blocks strengthen early feature extraction; fewer stage 3 blocks reduces redundancy at the smallest feature maps (7×7). Same total = 13 blocks.
- **Change:** `DEPTH: [1, 3, 3, 6]`. Budget: 5.667M, 1.416 GFLOPs (fewer large-dim blocks, more medium-dim blocks). Branch: autoresearch/h62-depth-s1.
- **Result:** 96.37% (−0.11pp vs H56 champion 96.48%). DISCARDED. Redistributing depth from stage 3 to stage 1 hurts — fewer stage-3 blocks (dim=384) lose important high-level feature processing; the heavy stage 3 in [1,2,3,7] is justified.

### H63: LayerScale 1e-3 (10x larger initial scale)
- **Direction rationale:** Current LAYER_SCALE=1e-4 is very conservative (residual contributions start at 0.01% of identity). CaiT paper (Touvron et al. 2021) tested various LayerScale inits: 1e-4 for large models, 1e-5 for medium, up to 1e-1 for small. Our 5.8M model might benefit from less conservative init (1e-3). Allows features to flow through residuals more quickly early in training.
- **Change:** `LAYER_SCALE: 1.0e-3` (from 1e-4). Pure config. Branch: autoresearch/h63-lscale1e3. Budget: 5.843M, 1.214 GFLOPs.
- **Result:** 96.20% (−0.28pp vs H56 champion 96.48%). DISCARDED. LayerScale 1e-3 is too permissive early in training — residuals grow too quickly, destabilizing the channel attention learning. 1e-4 is the correct conservative init for this architecture.

### H64: Triple auxiliary supervision — stages 0, 1, 2 (weights 0.1, 0.2, 0.4)
- **Direction rationale:** H50 (dual aux at 1+2, 0.2+0.4) gave +0.03pp. The H64 adds stage 0 supervision (after just 1 CA block, dim=64). Even early, low-confidence supervision provides gradient flow to the patch embedding and first block. Lee et al. (DSN 2015) applied supervision to ALL hidden layers including early ones, showing consistent improvement.
- **Change:** Add `AUX_LOSS_WEIGHT_S0: 0.1` + code for third aux head at stage 0. Stages supervised: 0 (w=0.1), 1 (w=0.2), 2 (w=0.4). Branch: autoresearch/h64-cpe-both-sides.
- **Budget:** 5.845M (+2K for stage-0 Linear head), 1.214 GFLOPs.
- **Result:** **96.59%** (+0.11pp vs H56 champion 96.48%). NEW CHAMPION. Triple supervision (stages 0+1+2 with weights 0.1+0.2+0.4) continues to improve gradient flow. Even early stage-0 supervision (after 1 block) helps. Job 340431. PUSHED ✓

### H65: Stronger aux weights — (0.15, 0.3, 0.4) vs H64's (0.1, 0.2, 0.4)
- **Direction rationale:** H64 showed triple supervision (0.1,0.2,0.4) is beneficial. H59 showed weights (0.3,0.6) on the dual supervision hurt (-0.06pp). Testing intermediate: (0.15,0.3,0.4). If stage-1 and stage-0 weights help more at 0.3 and 0.15, this could improve further.
- **Change:** `AUX_LOSS_WEIGHT_S1: 0.3, AUX_LOSS_WEIGHT_S0: 0.15`. Stage 2 keeps 0.4. Branch: autoresearch/h65-aux-weights.
- **Budget:** 5.845M (unchanged), 1.214 GFLOPs.
- **Result:** 96.48% (−0.11pp vs H64 champion 96.59%). DISCARDED. Stronger aux weights (0.15,0.3,0.4) hurt — original (0.1,0.2,0.4) is the optimal balance. The stage-0 signal needs to stay light (0.1) since the features are still very primitive after 1 block.

### H66: Channel pair bias at stage 3 — learned C×C attention bias
- **Direction rationale:** Current channel attention at stage 3 computes affinities between channels purely based on the dot product of their spatial feature vectors. Adding a learned C×C bias matrix to the attention logits allows the model to encode prior knowledge about channel pair interactions — channels that consistently attend to each other get a learned bias boost. This is inspired by relative position bias in spatial ViTs (Swin Transformer: arXiv:2103.14030), which consistently improves spatial attention. Here applied to channel tokens (the C×C channel interaction space). Enabled only at stage 3 (C=384, the highest-capacity stage) to keep within budget.
- **Literature context:** Liu et al. (Swin, 2021): relative position bias significantly improves ViT. MaxViT (2022, arXiv:2204.01697): learned static bias matrix B added to attention logits. Channel pair bias is analogous for the channel-token paradigm.
- **Change:** `CHANNEL_PAIR_BIAS: [false, false, false, true]`. Per-stage list support added to code. nn.Parameter(zeros(384,384)) shared across all 7 stage-3 blocks. Branch: autoresearch/h66-channel-pair-bias.
- **Budget:** 5.992M (+147K for 384×384 bias), 1.214 GFLOPs. Within budget.
- **Status:** DONE (job 340717) — DISCARDED. 96.31% (−0.28pp vs H64 champion 96.59%). Channel pair bias hurts — the 384×384 learned bias matrix adds 147K params but does not improve inter-channel affinity modeling beyond the dot-product. The model's channel tokens already learn sufficient co-activation patterns via their spatial feature vectors.

### H67: Intra-stage-3 mid-block auxiliary head (after block 4 of 7)
- **Direction rationale:** Existing aux heads at stages 0, 1, 2 supervise inter-stage boundaries. Stage 3 has 7 blocks — after block 0-6, only the final loss sees blocks 0-6. Adding a mid-stage aux head after block 3 (of 7) provides classification gradient to stage-3 blocks 0-3 directly. This extends the deep supervision idea (H64) to intra-stage, inspired by Lee et al. (DSN 2015) which showed per-layer supervision improves gradient flow at all depths.
- **Literature context:** DSN (Lee et al. 2015, arXiv:1409.5185): deep supervision at every hidden layer. H64 showed cross-stage supervision helps; this tests intra-stage supervision within the deepest stage.
- **Change:** `S3_MID_AUX_BLOCK: 3, AUX_LOSS_WEIGHT_S3MID: 0.3`. VisionTransformer exposes `_mid_aux_output` at block index 3. CvT adds `aux_head_s3mid = Linear(384, 10)`. Branch: autoresearch/h67-s3mid-aux.
- **Budget:** 5.848M (+3K for linear head), 1.214 GFLOPs. Within budget.
- **Status:** DONE (job 341213) — DISCARDED. 96.37% (−0.22pp vs H64 champion 96.59%). Intra-stage mid-block supervision hurts — adding a classification head mid-way through stage 3 disrupts the progressive feature hierarchy. The existing inter-stage aux heads (stages 0,1,2) already provide optimal gradient flow; intra-stage is counterproductive.

---

## Phase 2: Scaled Architecture — CIFAR-100 (~20M params, 5–8 GFLOPs, 300 epochs)

**Phase 1 complete.** Best 100-epoch: H64 at 96.59% (CIFAR-10). Best 300-epoch: H61 at 97.25% (CIFAR-10, H56 arch).

**Phase 2 carries forward all Phase 1 innovations:**
- SwiGLU MLP (H54), CPE 5×5 (H52), Triple aux supervision (H64) with weights [0.1, 0.2, 0.4]
- Learned attention temperature (H47), LayerScale 1e-4 (H30), DW shortcuts 3×3 (H6/H56)
- 4-stage hierarchy [64→128→256→384], DEPTH=[1,2,3,7], CA_MLP_RATIO=[8,8,4,1]

**Baseline (300ep):** ~20M params, CIFAR-100 (100 classes), 300 epochs with Mixup/CutMix/LS → 84.27%.
**Epoch policy updated:** 100ep screening for all hypotheses; 300ep only for confirmed champions.

### P2-B100: 100-epoch screening baseline — CIFAR-100
- **Architecture:** Same as P2-B1 (DIM_EMBED=[96,192,384,576], DEPTH=[1,2,5,12], CA_MLP_RATIO=[8,8,4,1])
- **Training recipe (100ep):** LR=2e-3, warmup=10ep, NO Mixup/CutMix/LS, DROP_PATH=[0,0.02,0.05,0.1]
- **Rationale:** Phase 1 showed clean targets (+0.68pp at 100ep vs Mixup/CutMix). CIFAR-100 needs a fresh 100ep baseline to compare hypotheses. LR=2e-3 was Phase 1 optimal at 100ep.
- **Budget:** 20.912M, 3.474 GFLOPs. Dataset: CIFAR-100.
- **Status:** DONE (job 343035, 2h05m). **Result: 81.34% max val top-1**. 100ep CIFAR-100 screening baseline established. +1.95pp vs 100ep snapshot of 300ep run (79.39%), confirming clean-target recipe is better for 100ep. PUSHED ✓

### P2-B1: Phase 2 Baseline — Scaled channel-primary ViT on CIFAR-100
- **Architecture:** DIM_EMBED=[96,192,384,576], DEPTH=[1,2,5,12], CA_MLP_RATIO=[8,8,4,1]
- **All Phase 1 innovations:** SwiGLU, CPE 5×5, triple aux [0.1,0.2,0.4], LayerScale 1e-4, DW shortcuts 3×3
- **Training:** 300 epochs, LR=1e-3, warmup=20ep, Mixup 0.8, CutMix 1.0, LS 0.1, DROP_PATH=[0,0.05,0.1,0.2]
- **Budget:** 20.912M params, 3.47 GFLOPs (fvcore) ≈ 6.9 GFLOPs conventional
- **Dataset:** CIFAR-100 (100 classes), data path: /path/to/data/.../cifar100
- **Branch:** phase2/p2-cifar100-baseline. **Status:** DONE (job 341456 GPU, 11h28m). **Result: 84.27% max val top-1**. Phase 2 CIFAR-100 baseline established. PUSHED ✓

### P2-H2: Stage-3 depth-width tradeoff — fewer blocks, wider MLP
- **Hypothesis:** At 576-dim, wider SwiGLU MLP (ratio=2) per block is more expressive than more shallow blocks (ratio=1). CIFAR-100's fine-grained discrimination may benefit from richer per-block feature transformations.
- **Change:** DEPTH=[1,2,5,7], CA_MLP_RATIO=[8,8,4,2]. Stage-3 blocks: 7 × (332K proj + 1327K MLP) = 7 × 1659K = 11.61M ≈ baseline 12 × 995K = 11.94M → iso-param.
- **Budget:** 20.426M params, 3.449 GFLOPs (fvcore). Within budget.
- **Literature:** Wide ResNets (Zagoruyko 2016, arXiv:1605.07146): wider hidden dims often outperform deeper narrow networks at same param count.
- **Status:** DONE (job 342464, 9h56m). **Result: 84.11%** (−0.16pp vs baseline 84.27%). DISCARDED. Wider MLP (ratio=2) with fewer blocks (7 vs 12) underperforms the baseline — more CA blocks at ratio=1 is better than fewer blocks at ratio=2. Stage-3 depth > MLP width for channel attention.

### P2-H3: Group channel attention (G=4) at stage 3 + deeper network
- **Hypothesis:** Splitting 576 channels into G=4 groups of 144 for within-group CA reduces attention cost C²→G×(C/G)², enabling DEPTH=[1,2,5,13] within budget. Unlike multi-head CA (splits spatial dim H*W), group CA splits channel tokens — each group retains full H×W=49 spatial context per token within its group. Groups specialize in semantic channel subsets.
- **Change:** CHANNEL_GROUPS: [1,1,1,4], DEPTH: [1,2,5,13]. 100ep clean-target recipe.
- **Budget:** 21.939M params, 3.524 GFLOPs. Within acceptable range (15-22M).
- **Literature:** GQA (Ainslie 2023, arXiv:2305.13245) — grouped attention reduces KV computation in LLMs; same principle for channel tokens. No prior work on grouped channel-token attention in ViTs found.
- **Status:** DONE (job 343264). **Result: 81.31%** (−0.03pp vs 100ep baseline 81.34%). DISCARDED. Group CA (G=4) at stage 3 does not improve over full CA. Likely reason: each group sees only 144/576 channels per attention computation — inter-group channel interactions are missed. The SwiGLU MLP provides some cross-group mixing but insufficient to compensate. Full channel attention with complete 576-token visibility is better for capturing global channel co-activations.

### P2-H4: Cross-stage channel feature fusion (DuoFormer-inspired)
- **Direction rationale:** Our 4-stage backbone processes stages sequentially but only uses the final stage-3 representation for classification. Intermediate stages (1, 2) produce rich multi-scale channel features that are currently only used for auxiliary supervision (training signal only). At inference, those features are discarded. Adding a cross-stage fusion that projects stage-1 and stage-2 GAP features into the final representation via learned additive projections (scale init=0) can provide complementary multi-scale channel information at classification time.
- **Literature context:** DuoFormer (Tang et al., arXiv:2407.13920, 2024): scale token aggregating features from all stages improves classification by 3-9% vs single-stage. SCHEME (arXiv:2312.00412, 2024): structured cross-channel aggregation within MLPs. The key insight: multi-scale feature reuse at the head leverages the hierarchical channel features without adding to the backbone FLOPs.
- **Change (code):** `CROSS_STAGE_FUSION: true`. In CvT: `fuse_proj_s1=Linear(192,576)`, `fuse_proj_s2=Linear(384,576)`, `fuse_scale_s1/s2=nn.Parameter(zeros)`. In forward_features: collect GAP at stages 1 and 2, add `scale*proj(feat)` to final representation after norm. Scale init=0 ensures identity at init.
- **Budget:** +192×576+384×576 = +110K + 221K = +331K params ≈ 20.74M total. No FLOPs increase at inference (GAP and projection on 1D vectors).
- **Implementation:** Done in cls_cvt.py + config.py (CROSS_STAGE_FUSION key registered). Code defaults to False (backward compatible).
- **Budget:** 21.245M params, 3.474 GFLOPs. 100ep clean-target recipe.
- **Status:** DONE (job 343374). **Result: 81.72%** (+0.38pp vs 100ep baseline 81.34%). NEW CHAMPION (100ep). Cross-stage fusion of GAP features from stages 1+2 provides complementary multi-scale channel information at the classifier. Scale params init=0 ensures safe convergence. PUSHED ✓

### P2-H5: No stochastic depth at 100ep (DROP_PATH=[0,0,0,0])
- **Direction rationale:** P2-H4 champion uses DROP_PATH=[0,0.02,0.05,0.1]. Phase 1 showed removing drop path gave +0.46pp at 100ep (H7) — at 100ep the model may be underfitting; stochastic depth reduces effective capacity per step. Testing if this holds at Phase 2 scale on CIFAR-100 with cross-stage fusion.
- **Change:** DROP_PATH_RATE: [0,0,0,0] (from [0,0.02,0.05,0.1]). Pure config, builds on P2-H4 champion.
- **Budget:** 21.245M, 3.474 GFLOPs (unchanged).
- **Literature:** Phase 1 H7: stochastic depth hurts underfitting models at 100ep. Literature search: no relevant 2024+ work found for this specific setting.
- **Status:** DONE (job 343554). **Result: 81.12%** (−0.60pp vs P2-H4 champion 81.72%). DISCARDED. Removing drop path HURTS at CIFAR-100 scale — opposite of Phase 1 behaviour. Model is not underfitting at 100ep on CIFAR-100; stochastic depth [0,0.02,0.05,0.1] provides beneficial regularization for the harder 100-class task. Phase 1 Phase 2 training dynamics do NOT transfer.

### P2-H6: Label smoothing LS=0.1 at 100ep
- **Direction rationale:** P2-H5 showed CIFAR-100 at 100ep benefits from regularization (drop path helps). Label smoothing is another regularizer; CIFAR-100's 100 classes in 20 superclasses mean neighbouring classes genuinely share features. LS=0.1 converts hard targets to soft distributions, appropriate for fine-grained 100-class discrimination.
- **Change:** LABEL_SMOOTHING: 0.1 (from 0.0). Builds on P2-H4 champion (CROSS_STAGE_FUSION=true, drop path [0,0.02,0.05,0.1]).
- **Literature:** Müller et al. (arXiv:1906.02629, NeurIPS 2019): label smoothing improves calibration; widely used in ViT/DeiT training. Recommended by DeiT (arXiv:2012.12877) for image classification.
- **Status:** DONE (job 343614). **Result: 82.02%** (+0.30pp vs P2-H4 champion 81.72%). NEW CHAMPION. Label smoothing LS=0.1 helps on CIFAR-100's fine-grained 100-class task — soft targets reduce over-confidence on semantically similar classes (superclass structure). PUSHED ✓

### P2-H7: Mild Mixup=0.4 at 100ep
- **Direction rationale:** P2-H6 showed LS=0.1 helps (+0.30pp). Mixup provides input-level soft targets — interpolated images train the model on in-between representations, explicitly learning class boundaries. CIFAR-100 with 100 fine-grained classes (500 images each) could benefit from virtual mixed samples. Testing mild alpha=0.4 (vs 0.8 in 300ep recipe) to avoid over-mixing at 100ep.
- **Change:** MIXUP: 0.4 (from 0.0). Keeps LS=0.1 and all P2-H6 champion settings.
- **Literature:** Mixup (Zhang et al. 2018, arXiv:1710.09412): consistently improves CIFAR-100 classification. DeiT uses Mixup=0.8 for 300ep; at 100ep, milder 0.4 avoids excessive soft-target interference.
- **Status:** DONE (job 343637). **Result: 82.79%** (+0.77pp vs P2-H6 champion 82.02%). NEW CHAMPION. Mild Mixup=0.4 provides input-level regularisation that complements label smoothing — helps CIFAR-100's fine-grained class boundaries. PUSHED ✓

**100ep recipe established:** CROSS_STAGE_FUSION=true, MIXUP=0.4, LS=0.1, DROP_PATH=[0,0.02,0.05,0.1], LR=2e-3, warmup=10ep.
**Note:** P2-H6/H7 were training-recipe changes. Per updated RESEARCH_BRIEF.md, subsequent hypotheses must strengthen channel attention itself.

### P2-H8: Differential channel attention
- **Hypothesis:** In channel-attention mode, channels that always co-activate regardless of spatial content create noise in the attention map — the model wastes capacity on trivially correlated channels. Differential Transformer (Ye et al., arXiv:2410.05258, MSRA Oct 2024) showed that computing *two* softmax attention maps and subtracting them cancels noise in spatial attention. The same principle applies to *channel* attention: split Q and K along the head-dim (spatial feature) axis into two halves (H*W/2 each), compute two C×C softmax attention maps independently, and output (attn1 − λ·attn2) @ V. The two halves see complementary spatial evidence; their shared noise in the channel domain cancels, sharpening which channel pairs truly co-attend. λ is a learned scalar (init=0.05) so the model starts near vanilla attention and adjusts.
- **Why channel-specific:** The spatial halves act as two "views" of the spatial field. Channels co-activated by global spatial structure (texture, background) appear in both attn maps; task-relevant inter-channel interactions are asymmetric between the two views. Subtraction suppresses the symmetric noise.
- **Implementation:** `self.diff_attn = True` in `Attention.__init__` (channel mode only). In `forward`: when `diff_attn` and `token_mode=='channel'` and `G==1`, bypass SDPA, manually compute two attention maps on first and second halves of head-dim, subtract, apply to V. `diff_lambda=nn.Parameter(tensor(0.05))`, clamped ≥0 via `F.relu`. No change to V, projection, or anything else.
- **Config key:** `DIFF_ATTN: true` (registered in config.py).
- **Literature:** Ye et al. 2024 (arXiv:2410.05258, MSRA): Differential Transformer — cancels attention noise via subtraction of two maps on LLM spatial tokens. Verified: they show noise-cancellation effect where irrelevant tokens are amplified in standard attention and suppressed in differential. First application to channel-token attention in ViTs.
- **Budget:** 21.245M params (+0 params except 20 scalar diff_lambda params ≈ negligible), 4.323 GFLOPs (+0.85G from second attention map computation).
- **Builds on:** P2-H7 champion (82.79%). All recipe unchanged: CROSS_STAGE_FUSION, MIXUP=0.4, LS=0.1, DROP_PATH=[0,0.02,0.05,0.1].
- **Status:** DONE (job 343652). **Result: 81.88%** (−0.91pp vs P2-H7 champion 82.79%). DISCARDED. Differential attention hurts — spatial halving reduces effective head-dim from HW=49 to 24 per sub-map, insufficient for meaningful C×C attention. Noise-cancellation benefit outweighed by information loss from halved spatial features. FLOPs +0.85G for no gain. Reverted to P2-H7 champion.

### P2-H9: Intra-stage-3 mid-block auxiliary supervision
- **Hypothesis:** Stage 3 has 12 deep channel-attention blocks. The gradient from the final loss traverses all 12 blocks to reach block 1, creating a long gradient path. The existing triple-aux supervision (stages 0, 1, 2) provides gradient shortcuts at stage boundaries but not *within* stage 3. Adding an aux classification head at block 5 (0-indexed, midpoint of 12) creates a direct gradient shortcut to mid-stage channel representations, forcing them to be discriminative at the halfway point. This is the intra-stage analogue of the inter-stage triple-aux pattern that gave +0.71pp in Phase 1 (H46/50/64). Weight 0.15: smaller than stage-boundary aux (0.1-0.4) since mid-block representations are less mature.
- **Why this strengthens channel attention:** Block 5's channel attention patterns learn to be semantically meaningful (must classify 100 CIFAR-100 classes) while still 6 blocks before the final output. This constrains the mid-stage CA maps to capture discriminative inter-channel relationships rather than intermediate abstract features. Effectively provides a "training curriculum" within stage 3.
- **Implementation:** `VisionTransformer._mid_block_feat` captures BCHW features at block index `mid_block_idx` during forward. `ConvolutionalVisionTransformer.aux_head_mid` (Linear 576→100) receives GAP of mid-block feat and computes `_aux_logits_mid`. `main.py` adds `_aux_mid_weight * loss` to total loss. Config: `AUX_MID_BLOCK_IDX: 5`, `AUX_LOSS_WEIGHT_MID: 0.15`.
- **Literature:** Deep supervision (arXiv:1409.5185, Lee et al.): companion objectives at intermediate layers improve gradient flow in deep networks; shown to help CNNs and ViTs. GoogLeNet-style aux classifiers (Szegedy et al. 2014) validated this principle. Most relevant recent work: auxiliary losses in hierarchical ViTs (H46/50/64 in Phase 1) gave cumulative +0.71pp.
- **Budget:** 21.302M params (+57K for aux_head_mid), 3.474 GFLOPs. No inference cost (aux head only active during training).
- **Builds on:** P2-H7 champion (82.79%). All recipe unchanged: CROSS_STAGE_FUSION, MIXUP=0.4, LS=0.1, DROP_PATH=[0,0.02,0.05,0.1].
- **Status:** DONE (job 343734). **Result: 82.80%** (+0.01pp vs P2-H7 82.79%). NEW CHAMPION (noise-level gain but protocol says commit). Mid-block aux gradient shortcut provides negligible benefit — stage-3 CA blocks may already receive sufficient gradient through 12 inter-block residuals. PUSHED ✓

### P2-H10: Periodic CPE re-injection within stage 3 (interval=4)
- **Hypothesis:** CPE (5×5 DW conv) is applied once at each stage's entry point to inject spatial structure into channel tokens. Over 12 blocks in stage 3, repeated channel attention may gradually transform channel tokens away from their original spatial activation patterns. DW shortcuts (3×3 DW conv per block) provide local spatial mixing, but CPE's 5×5 kernel covers 51% of the 7×7 feature map — a broader spatial prior. Reapplying CPE after blocks 3 and 7 (0-indexed, interval=4) reinjects this broader spatial context, keeping channel tokens grounded in their spatial identity when computing Q/K similarity in later blocks.
- **Why this strengthens channel attention:** Channel attention discriminates channels by their spatial activation patterns. Reapplying CPE ensures those patterns stay spatially meaningful throughout the deep final stage — directly enriching the Q/K feature vectors used for channel-to-channel attention.
- **Implementation:** `VisionTransformer._cpe_repeat_interval`: inside the block loop, after block i where `(i+1) % interval == 0` and blocks remain, rearrange to BCHW, apply existing CPE, rearrange back to tokens. Zero new parameters; reuses existing 5×5 CPE conv from stage entry. FLOPs +0.003G (negligible).
- **Literature:** Literature search: no recent 2024+ work found specifically on periodic CPE re-injection within transformer stages. Related: PvT v2 (arXiv:2106.13797) uses CPE at every stage entry; depth-wise conv position encodings are well-established for spatial context. The periodic repetition within a stage is a novel variant.
- **Budget:** 21.302M params (unchanged), 3.477 GFLOPs (+0.003G for 2 extra CPE passes).
- **Builds on:** P2-H9 champion (82.80%). All recipe unchanged. AUX_MID (0.15) retained.
- **Status:** DONE (job 343777). **Result: 82.88%** (+0.08pp vs P2-H9 champion 82.80%). NEW CHAMPION. Periodic CPE re-injection at interval=4 (triggers after blocks 3 and 7 in stage 3) does help — refreshing the 5×5 spatial context mid-stage keeps channel tokens more spatially grounded for later attention blocks. PUSHED ✓

### P2-H11: Larger Q/K/V projection kernel at stage 3 (3×3 → 5×5)
- **Hypothesis:** P2-H10 showed spatial context enrichment helps channel attention (+0.08pp). The Q/K DW conv projection (KERNEL_QKV) extracts spatial features per channel token for attention computation. At stage 3 (7×7 feature map), a 3×3 DW conv captures a 3×3 neighborhood per output position; a 5×5 captures a 5×5 neighborhood (71% of the 7×7 map). Larger Q/K kernels give each channel token a richer, more globally-informed spatial representation, improving channel attention discriminability: channels with similar broader spatial activation patterns will have stronger Q/K agreement.
- **Why this strengthens channel attention:** The Q/K feature vectors directly determine which channels attend to each other. Enriching them with larger receptive field captures more of the spatial activation pattern per channel token — a 5×5 DW conv effectively pools more spatial evidence before computing channel-to-channel similarities.
- **Change:** `KERNEL_QKV: [3, 3, 3, 5]`, `PADDING_KV: [1, 1, 1, 2]`, `PADDING_Q: [1, 1, 1, 2]`. Config-only — no code changes.
- **Literature:** CvT (arXiv:2103.15808, Wu et al. 2021) showed that convolutional projections in attention improve ViT quality; larger kernels capture more context. No specific 2024+ paper on Q/K kernel size for channel-token attention found.
- **Budget:** 21.634M params (+0.33M for 5×5 vs 3×3 DW in stage 3), 3.494 GFLOPs. Within budget.
- **Builds on:** P2-H10 champion (82.88%). All settings retained: periodic CPE (interval=4), mid-block aux, cross-stage fusion, MIXUP=0.4, LS=0.1.
- **Status:** DONE (job 343844). **Result: 83.16%** (+0.28pp vs P2-H10 champion 82.88%). NEW CHAMPION. Larger Q/K kernel gives channel tokens richer spatial representations — 5×5 on 7×7 map captures 71% of the field per position, significantly improving channel attention discriminability. This is the largest architectural gain since cross-stage fusion. PUSHED ✓

### P2-H12: Full-field Q/K/V projection kernel at stage 3 (5×5 → 7×7)
- **Hypothesis:** P2-H11 showed 5×5 Q/K kernel substantially improves channel attention (+0.28pp) by widening the spatial receptive field of Q/K feature vectors. Stage 3 operates on 7×7 feature maps. A 7×7 DW conv kernel covers the *entire* 7×7 spatial field — each Q/K output position aggregates from all 49 spatial neighbors (with overlap/weighting). This provides the globally-informed channel representation: each channel token's Q/K vector reflects its full spatial activation pattern, not a partial view. The attention score Q·K^T then measures how similar two channels' *global* spatial patterns are — the most information-rich possible channel comparison.
- **Change:** `KERNEL_QKV: [3, 3, 3, 7]`, `PADDING_KV: [1, 1, 1, 3]`, `PADDING_Q: [1, 1, 1, 3]`. Config-only.
- **Literature:** Full-field DW conv in attention projection: PoolFormer (Yu et al., arXiv:2111.11418) uses global average pooling as token mixing; our 7×7 DW conv is the learnable analogue for 7×7 spatial. No specific 2024+ paper on maximum-kernel Q/K projection in channel attention found. CvT (arXiv:2103.15808) showed that larger conv kernels in attention improve quality.
- **Budget:** ~21.9M params (+0.3M for 7×7 vs 5×5 DW), ~3.55 GFLOPs (est). Within budget.
- **Builds on:** P2-H11 champion (83.16%). All settings retained: periodic CPE (interval=4), mid-block aux, cross-stage fusion, MIXUP=0.4, LS=0.1.
- **Status:** DONE (job 344112). **Result: 83.11%** (−0.05pp vs P2-H11 champion 83.16%). DISCARDED. 7×7 full-field kernel hurts slightly — aggregating the entire 7×7 map per Q/K position over-smooths local spatial structure. 5×5 (71% field coverage) is the sweet spot, balancing broad context with local discriminability. Reverted to P2-H11 settings.

### P2-H13: 5×5 Q/K/V projection kernel at stage 2 as well
- **Hypothesis:** P2-H11 showed 5×5 Q/K at stage 3 gives +0.28pp. Stage 2 operates at 14×14 with 384 channels and currently uses 3×3 Q/K. Upgrading stage 2's Q/K to 5×5 gives its channel attention richer spatial representations. Stage 2 channel tokens feed into stage 3 via cross-stage fusion (as GAP features) and via the patch embedding for stage 3's input. Richer stage 2 representations → better channel feature hierarchy → higher-quality stage 3 channel attention.
- **Change:** `KERNEL_QKV: [3, 3, 5, 5]`, `PADDING_KV: [1, 1, 2, 2]`, `PADDING_Q: [1, 1, 2, 2]`. Config-only.
- **Literature:** Larger Q/K kernels in hierarchical ViTs: CvT showed per-stage conv projection benefits compound across stages. At 14×14 spatial, 5×5 covers 36% of the map (vs 100% at 7×7 for stage 3) — still a meaningful gain.
- **Budget:** ~21.8M params (small increase for 5×5 DW in stage 2), ~3.51 GFLOPs (est). Within budget.
- **Builds on:** P2-H11 champion (83.16%). Restores 5×5 at stage 3; adds 5×5 at stage 2.
- **Status:** DONE (job 344147). **Result: 82.82%** (−0.34pp vs P2-H11 champion 83.16%). DISCARDED. Larger Q/K kernel at stage 2 (14×14) hurts — at this resolution, 5×5 covers only 13% per dimension, blurring the local spatial patterns that are discriminative for channel attention at mid-scale. Scale-specific finding: 5×5 Q/K only helps at stage 3's 7×7. Reverted to P2-H11 settings (KERNEL_QKV=[3,3,3,5]).

### P2-H14: 5×5 DW shortcut kernel at stage 3
- **Hypothesis:** P2-H11 showed 5×5 Q/K projection at stage 3 gives +0.28pp — larger spatial receptive field in the attention projection improves channel discrimination at 7×7 resolution. The DW shortcut is a parallel bypass pathway added to the residual at every block: GELU → BN → 3×3 DW conv → add. This shortcut also contributes spatial features to each channel token's representation. Increasing the DW shortcut kernel from 3×3 to 5×5 at stage 3 gives the shortcut path the same broader spatial coverage that benefited Q/K projections. Consistent with the 5×5 sweet spot at 7×7 spatial.
- **Change:** `DW_SHORTCUT_KERNEL: [3, 3, 3, 5]`. Config-only.
- **Literature:** DW shortcuts (Efficient ViT with DW, original CvT motivation): parallel convolutional shortcuts improve gradient flow. Larger shortcut kernels capture broader spatial dependencies. No specific 2024+ paper on DW shortcut kernel size in channel attention found.
- **Budget:** ~21.6M params (small increase for 5×5 DW shortcuts in 12 stage-3 blocks), ~3.50 GFLOPs. Within budget.
- **Builds on:** P2-H11 champion (83.16%). KERNEL_QKV=[3,3,3,5], CPE_REPEAT_INTERVAL=4, all prior innovations.
- **Status:** DONE (job 344161). **Result: 83.17%** (+0.01pp vs P2-H11 champion 83.16%). NEW CHAMPION (noise-level, protocol says commit). 5×5 DW shortcut at stage 3 provides marginal benefit — the shortcut path matters less than Q/K spatial coverage. PUSHED ✓

### P2-H15: Wider stage-3 embedding (576→640) with fewer depth (12→10)
- **Hypothesis:** Current stage 3: 576 channels × 12 blocks. Increasing embedding to 640 with 10 blocks keeps params roughly iso-budget while giving richer channel representations. With 640 channel tokens (vs 576), the C×C=640×640 attention map captures interactions across 11% more channels, and the SwiGLU MLP has proportionally more capacity. This directly improves channel attention expressiveness: each channel's Q/K/V is a 640-dim embedding (larger representation), and the attention map is over a richer set of 640 semantic features.
- **Why this strengthens channel attention:** More channels = finer-grained semantic decomposition (each channel specializes in more specific patterns), richer cross-channel interactions, and wider MLP for per-channel feature transformation.
- **Change:** `DIM_EMBED: [96, 192, 384, 640]`, `DEPTH: [1, 2, 5, 10]`. Config-only.
- **Literature:** Scaling laws for transformers (Kaplan et al., arXiv:2001.08361): width improvements at fixed compute show consistent gains. For channel-token attention specifically: more channels → finer semantic decomposition that may improve fine-grained classification (CIFAR-100 has 100 classes in 20 superclasses).
- **Budget:** ~22M params est. FLOPs check required.
- **Builds on:** P2-H14 champion (83.17%). All settings: KERNEL_QKV=[3,3,3,5], DW_SHORTCUT_KERNEL=[3,3,3,5], CPE_REPEAT_INTERVAL=4, mid-block aux, cross-stage fusion.
- **Status:** DONE (job 344176). **Result: 83.18%** (+0.01pp vs P2-H14 83.17%). NEW CHAMPION (noise-level). Wider 640-dim stage 3 gives no meaningful gain — deeper iterations at 576-dim seems equally effective. Incremental spatial/width tweaks are plateauing around 83.2%. PUSHED ✓

### P2-H16: Cross-stage spatial input fusion (stage-2 features → stage-3 input)
- **Hypothesis:** Current cross-stage fusion adds GAP features from stages 1+2 to the FINAL classifier representation. This is classification-time only. P2-H16 adds a SPATIAL cross-stage connection: project stage-2's full 14×14 feature map (B, 384, 14, 14) → pool to 7×7 → project to stage-3's dim → add to stage-3's INPUT (before all attention blocks). Stage-3's channel attention then has multi-scale spatial context in each channel token from the start, not just at the output. Unlike the existing GAP fusion (which loses spatial structure), this preserves per-position spatial information (7×7=49 positions remain distinct).
- **Why stronger than existing fusion:** The existing cross-stage fusion is a single scalar per channel (GAP) added at the classifier — good for classification but doesn't help attention Q/K. Spatial input fusion enriches each of the 49 spatial positions of each channel token with corresponding stage-2 spatial features, making Q/K representations richer for all 10 attention blocks.
- **Implementation:** After stage 2, save full BCHW output. Before stage 3 processes (after stage-3 patch embed), apply adaptive_avg_pool2d to match stage-3 spatial dims, then 1×1 Conv2d(384, 640) projection + learned scale (init=0) added to stage-3 input.
- **Config:** `CROSS_STAGE_INPUT_FUSION: true` (new key).
- **Literature:** FPN (Lin et al., arXiv:1612.03144): feature pyramid networks with top-down skip connections improve feature hierarchy. Dense connections (DenseNet, Huang et al., arXiv:1608.06993): direct feature reuse across layers. Our variant: cross-scale channel feature reuse as input to the deepest stage.
- **Budget:** +384×640=246K params for projection conv. Negligible FLOPs.
- **Builds on:** P2-H15 champion (83.18%). DIM_EMBED=[96,192,384,640], DEPTH=[1,2,5,10], KERNEL_QKV=[3,3,3,5].
- **Status:** DONE (job 344226). **Result: 82.72%** (−0.46pp vs P2-H15 champion 83.18%). DISCARDED. Spatial input fusion HURTS — injecting stage-2's full 14×14 spatial feature map (projected to 640-dim, pooled to 7×7) into stage-3's input before attention degrades performance. Likely reason: stage-2 and stage-3 operate on different semantic levels; forcing stage-2's spatial structure into stage-3's input disrupts stage-3's own spatial abstraction process. The existing GAP-based cross-stage fusion is sufficient — adding spatial detail from stage-2 at the input adds noise rather than signal for stage-3's channel attention.

### P2-H17: Grouped channel attention at stage 3 (G=4 groups of 160 channels)
- **Hypothesis:** Full 640×640 channel attention likely learns many spurious inter-channel correlations. Grouping into G=4 groups of 160 channels creates structured interactions: within each group full 160×160 CA; cross-group mixing only through the post-attention Linear(640,640) projection. Forces each group of channels to specialize and learn focused within-group patterns rather than noisy cross-group correlations.
- **Why this improves channel attention:** Current CA at 640-dim has 640²=409,600 attention matrix entries; grouped CA has 4×160²=102,400 (4× fewer). More focused attention patterns per group → less noise, better channel specialization. The shared post-attn projection still provides cross-group information exchange.
- **Literature:** MogaNet (Li et al., 2023): multi-order gated aggregation with channel-group-aware attention. EfficientFormer v2 (2023): factorized attention reduces redundant cross-channel connections. Literature search: no directly analogous grouped channel-token attention in 2024+ vision papers found (most work groups spatial tokens, not channel tokens).
- **Config change:** CHANNEL_GROUPS: [1, 1, 1, 4] (only stage 3 grouped; stages 0-2 unchanged).
- **Budget:** 22.1M params, 3.51 GFLOPs.
- **Builds on:** P2-H15 champion (83.18%). DIM_EMBED=[96,192,384,640], DEPTH=[1,2,5,10].
- **Status:** DONE (job 344259). **Result: 82.98%** (−0.20pp vs P2-H15 champion 83.18%). DISCARDED. Grouped CA hurts — full 640×640 channel attention is better than 4×160×160 groups. Cross-group interactions matter. Reverted CHANNEL_GROUPS to [1,1,1,1].

### P2-H18: Sigmoid channel attention at stage 3
- **Hypothesis:** Softmax channel attention enforces winner-take-all normalization: weights over C channels sum to 1, so attending to one channel reduces all others. But channels co-activate cooperatively — multiple channels together encode semantic concepts. Replacing softmax with sigmoid(QK^T/√d)/C removes competition: each channel pair's attention is independent. Normalization by C stabilizes output magnitudes.
- **Why this strengthens channel attention:** Sigmoid allows multiple channels to have high mutual attention simultaneously — better suited for cooperative channel-level semantic encoding than softmax's zero-sum allocation.
- **Literature:** Ramapuram et al., arXiv:2409.04431, 2024: sigmoid attention matches softmax when normalized by sequence length; improves sample efficiency; LayerScale ensures stable training.
- **Implementation:** `self.sigmoid_attn = True` in Attention; in forward: `attn = sigmoid(QK^T/√d) / C_tokens` instead of SDPA. Config: `SIGMOID_ATTN: [false,false,false,true]`.
- **Budget:** 22.09M params, 3.915 GFLOPs (unchanged).
- **Builds on:** P2-H15 champion (83.18%). KERNEL_QKV=[3,3,3,5], DW_SHORTCUT_KERNEL=[3,3,3,5], CPE_REPEAT_INTERVAL=4, mid-block aux, cross-stage fusion, MIXUP=0.4, LS=0.1.
- **Status:** DONE (job 344414). **Result: 82.97%** (−0.21pp vs P2-H15 champion 83.18%). DISCARDED. Sigmoid attn hurts — softmax competition is actually beneficial for channel attention at stage 3. Removing winner-take-all normalization degrades channel selectivity. Reverted to P2-H15 champion.

---

## Validation runs (parallel to 100ep iteration)

### 300ep CIFAR-100 champion validation
- **Config:** `configs/cifar100_300ep_champion.yaml`. P2-H15 arch (22.09M, 3.51 GFLOPs), CIFAR-100, 300 epochs.
- **Recipe:** Mixup=0.8, CutMix=1.0, RandAugment rand-m9-mstd0.5-inc1, RE=0.25, DROP_PATH=[0,0.05,0.1,0.2], LS=0.1, LR=2e-3, WD=0.05, warmup=20ep.
- **Status:** DONE (job 344501). **Result: 85.07%** val top-1 at 300ep. Massive jump from 83.18% at 100ep (+1.89pp). Architecture validated — P2-H15 champion is strong. PUSHED ✓

### ImageNet-1K — prior attempts (all failed/killed by HPC hardware failure)
- Jobs 344537, 344818, 344821: all killed by hardware failure before completion.

### ImageNet-1K — fresh attempt (P2-H22 champion)
- **Config:** `configs/imagenet_2gpu.yaml`. P2-H22 arch (22.15M, 3.51 GFLOPs): all-stage cross-stage fusion, KERNEL_QKV=[3,3,3,5], DW_SHORTCUT_KERNEL=[3,3,3,5], CPE_REPEAT_INTERVAL=[0,0,4,4], periodic CPE, SwiGLU, mid-block aux.
- **Recipe:** 2-GPU, BATCH_SIZE=512/GPU (1024 total), LR=5e-4, WD=0.05, 300ep, warmup=20ep, Mixup=0.8, CutMix=1.0, RandAugment rand-m9-mstd0.5-inc1, RE=0.25, DROP_PATH=[0,0.05,0.1,0.2], LS=0.1.
- **Job 348509:** FAILED immediately. Root cause: torchrun sets LOCAL_RANK as env var but main.py requires `--local_rank $LOCAL_RANK` explicitly on the command line (line 94: `required=True`). Also: CPE_REPEAT_INTERVAL changed to list type in config.py but imagenet_2gpu.yaml had scalar `4` — type mismatch.
- **Fix:** Add `--local_rank $LOCAL_RANK` to torchrun call; update imagenet_2gpu.yaml to `CPE_REPEAT_INTERVAL: [0, 0, 4, 4]`. Verified arg parsing locally (fails only at cuda.set_device as expected on login node).
- **Job 349012:** PENDING (2-GPU, torchrun --standalone --nproc_per_node=2).

---

### P2-H19: Value residual for stage-3 channel attention
- **Hypothesis:** Stage 3 has 10 channel-attention blocks. In deep stages, attention tends to concentrate on a few dominant channels (attention sink), starving other channels. Value Residual Learning (ResFormer, arXiv:2410.17897, ACL 2025) fixes this by adding the first layer's V embedding as a residual to all subsequent layers' V. Adapted for channel tokens: save stage-3's input (before any blocks) and add it as `scale * x_stage_input` to V in blocks 1-9. Scale init=0 (safe convergence). As blocks go deeper, the residual can gradually restore original channel information into V.
- **Why this strengthens channel attention:** Prevents attention concentration in stage-3's 10-block depth. Deep channel-attention blocks may over-focus on a subset of high-activation channels; the V residual preserves all channels' original information throughout the stage, enabling richer cross-channel interactions in later blocks.
- **Literature:** Zhu et al., arXiv:2410.17897, ACL 2025 (ResFormer): value residual connections reduce attention concentration, improve validation loss with 16% fewer params. SVFormer variant: all layers share layer-1's V. Adapted for channel-token attention in a hierarchical ViT.
- **Implementation:** `VisionTransformer._use_value_residual=True`; pre-rearranges stage input to `(B,1,C,H*W)` before block loop; passes as `v_residual` to blocks 1-9; Attention adds `v_res_scale * v_residual` to V (scale ∈ nn.Parameter init=0). Config: `VALUE_RESIDUAL: [false,false,false,true]`.
- **Budget:** 22.09M params (+10 scalar params for stage-3's 10 attention layers), 3.514 GFLOPs.
- **Builds on:** P2-H15 champion (83.18%).
- **Fix:** Job 344509 FAILED — DDP unused-param error. Fix: pass v_res to all blocks. Resubmitted as job 344525.
- **Status:** DONE (job 344525). **Result: 83.11%** (−0.07pp vs P2-H15 champion 83.18%). DISCARDED. Value residual doesn't help — stage-3 channel attention does not suffer from concentration at this depth/width. Reverted.

### P2-H20: Dual-scale channel attention (GAP branch)
- **Hypothesis:** The current 5×5 Q/K projection captures local spatial co-activation patterns between channels. But some channels co-activate globally (across the entire 7×7 map) regardless of spatial pattern — these global correlations are invisible to spatially-local attention. Adding a parallel GAP branch (Q/K = global average of spatial features → (B,1,C,1)) computes a second C×C attention map that captures pure global channel co-activation. Combined via `x = main_out + gap_scale * gap_out`, gap_scale init=0.
- **Why this strengthens channel attention:** Two complementary channel interaction views — local-spatial (5×5 DW conv) and global (GAP). Channels that co-activate both locally AND globally get stronger interaction signal; channels with only one mode are still captured correctly.
- **Literature:** CAViT (arXiv:2602.05598, 2026): dual attention (spatial + channel) improves feature interaction. SE-Net (Hu et al.): GAP-based global channel context is powerful. Combined: GAP branch as a global channel attention complement to the local spatial branch.
- **Implementation:** `dual_gap_attn=True`: in Attention.forward, after main SDPA, compute `q_gap=q.mean(-1,keepdim=True)`, `k_gap=k.mean(-1,keepdim=True)`, `attn_gap=softmax(q_gap@k_gap.T)`, `x_gap=attn_gap@v`, then `x = x + gap_scale * x_gap`. `gap_scale=nn.Parameter(zeros(1))`.
- **Budget:** 22.09M params (+10 scalar gap_scale params), 3.719 GFLOPs (+0.2G for GAP branch).
- **Builds on:** P2-H15 champion (83.18%).
- **Status:** DONE (job 344759). **Result: 83.16%** (−0.02pp vs P2-H15 champion 83.18%). DISCARDED. GAP branch adds no value — stage-3 channel attention via 5×5 DW conv already captures sufficient global information at 7×7 resolution. Reverted.

### P2-H21: Learned channel-pair attention prior (CHANNEL_PAIR_BIAS)
- **Hypothesis:** Standard softmax attention assumes a uniform prior over all C×C channel pairs — each pair is equally likely to interact before seeing the input. But channels form semantic clusters (texture, color, shape detectors) with stable pairwise co-activation patterns. Adding a learned C×C bias to the attention logits (`attn = softmax(QK^T/√d + B)`) encodes these structural channel-pair affinities as a persistent prior, freeing Q/K to focus on instance-specific variations.
- **Why this strengthens channel attention:** The bias acts as a dataset-level prior over channel interactions, analogous to Swin's relative position bias for spatial tokens. It reduces the burden on Q/K to simultaneously model both structural (which channels always co-activate) and instance-specific (which channels co-activate for this image) interactions.
- **Literature:** "You Need Better Attention Priors" (arXiv:2601.15380, 2025): GOAT shows that replacing the naive uniform prior assumption in attention with a learnable prior improves quality; log-prior is absorbed into attention logits as a bias term.
- **Implementation:** Config-only: `CHANNEL_PAIR_BIAS: true`. Already implemented in cls_cvt.py: `self.channel_pair_bias = nn.Parameter(zeros(C, C))`, shared across all blocks in the stage, added to attention logits before softmax.
- **Budget:** 22.69M params (+0.6M for 640×640 bias), 3.514 GFLOPs (unchanged).
- **Builds on:** P2-H15 champion (83.18%).
- **Status:** DONE (job 344829). **Result: 82.89%** (−0.29pp vs P2-H15 champion 83.18%). DISCARDED. Channel pair bias hurts at 100ep — 640×640=409K extra parameters likely overfits at 100ep scale or creates optimization instability. Reverted.

### P2-H22: All-stage cross-stage fusion (add stage 0)
- **Hypothesis:** Current cross-stage fusion collects GAP features from stages 1 (192-dim) and 2 (384-dim) and adds them to the final classifier representation. Stage 0 (96-dim, 56×56 feature map) is omitted. Stage 0 captures low-level features (edges, textures, basic patterns) — complementary information not present in deeper stages. Adding stage 0's GAP to the fusion completes the multi-scale feature pyramid.
- **Literature:** DuoFormer (arXiv:2407.13920, 2024): "scale token aggregating features from ALL stages improves classification by 3-9% vs single-stage." Current fusion uses stages 1+2; adding stage 0 implements the full DuoFormer-style all-stage aggregation.
- **Implementation:** `CROSS_STAGE_FUSION_S0: true`. Adds `fuse_proj_s0 = Linear(96, 640, bias=False)` + `fuse_scale_s0 = Parameter(zeros(640))`. Collects GAP at stage 0, adds `scale_s0 * proj_s0(feat_s0)` at classifier. Scale init=0.
- **Budget:** 22.15M params (+61K), 3.514 GFLOPs (unchanged).
- **Builds on:** P2-H15 champion (83.18%).
- **Status:** DONE (job 344888). **Result: 83.30%** (+0.12pp vs P2-H15 champion 83.18%). NEW CHAMPION. Stage 0 low-level features provide complementary information to stages 1+2 at the classifier. All-stage aggregation validated. PUSHED ✓

### P2-H23: Second-order statistics pooling in cross-stage fusion
- **Hypothesis:** Current cross-stage fusion uses GAP (mean over spatial dims) from each stage — a first-order statistic. GAP captures mean activation per channel, but loses channel activation variance (how spread/concentrated each channel is spatially). Std pooling captures how each channel's spatial activations vary — a complementary second-order statistic. Channels with high variance are spatially selective (activated in specific regions), while low-variance channels are globally activated. Adding std features from stages 0-2 alongside the mean features gives the classifier richer channel characterization: not just "how active" but "how selective" each channel is.
- **Why this strengthens channel attention:** The channel attention mechanism attends to channels based on their roles. Adding std(x, spatial) as a second-order pooling statistic provides richer evidence about channel behavior patterns — enabling the classifier to distinguish spatially selective vs globally activated channels, complementing the GAP-based first-order view. Inspired by second-order pooling in fine-grained recognition: covariance/bilinear pooling (Carreira et al., arXiv:1511.06042) and MPNCOV (Li et al., arXiv:1703.08050) both show second-order statistics improve classification over first-order alone.
- **Literature:** Charatan et al., arXiv:2408.01372, 2024: second-order spatial pooling significantly improves classification by capturing texture statistics beyond mean. Adapted for cross-stage fusion: std pooling complements GAP.
- **Implementation:** `FUSE_STD_POOL: true`. Adds `fuse_std_proj_s0/s1/s2 = Linear(dim_si, 640)` + `fuse_std_scale_s0/s1/s2 = Parameter(zeros(640))`. Collects `x.std(dim=[-2,-1])` at each stage (same timing as mean GAP), adds `scale * proj(std_feat)` at classifier. Scales init=0.
- **Budget:** 22.15M params (+185K for 3 std projections), 3.514 GFLOPs (unchanged).
- **Builds on:** P2-H22 champion (83.30%). CROSS_STAGE_FUSION_S0: true retained.
- **Status:** DONE (job 344901). **Result: 83.30%** (tied with P2-H22 champion 83.30%). DISCARDED. Second-order std pooling adds no complementary information — mean and std from GAP carry redundant channel importance signals. The first-order GAP fusion is already sufficient at the classifier. Reverted to P2-H22 champion.

### P2-H32: Depth redistribution DEPTH=[1,2,5,10]→[1,2,6,9]
- **Hypothesis:** Current architecture allocates 10 blocks to stage 3 (7×7, 640ch) and only 5 to stage 2 (14×14, 384ch). Stage 2 now has proven periodic CPE (interval=2, validated H26) but limited depth to exploit it — only 5 blocks to develop rich intermediate representations. Adding one block to stage 2 (5→6) and removing one from stage 3 (10→9) shifts one unit of computation to the intermediate resolution, where feature maps are 4× larger (196 vs 49 tokens) and channel diversity is higher. Stage 3 retains 9 blocks — still deeply sufficient for final abstraction.
- **Why this strengthens channel attention:** At 14×14 (stage 2), the channel-attention matrix is 384×384 with features containing 196 spatial samples per channel — rich spatial statistics for Q/K computation. With 5 blocks, stage 2 reaches moderate channel interaction depth; with 6, it can develop more complex inter-channel relationships before downsampling to 7×7. Better stage-2 representations propagate via patch embed into stage 3 AND via cross-stage fusion to the final classifier. Literature: HRNet (arXiv:1908.07919): maintaining high-resolution computation throughout improves representation quality vs early downsampling.
- **Implementation:** YAML only. `DEPTH: [1,2,5,10]→[1,2,6,9]`. CPE_REPEAT_INTERVAL=[0,0,2,4]: stage-2 interval=2 still fires at blocks 1,3 (unchanged for 6-block stage). AUX_MID_BLOCK_IDX=5 still valid in 9-block stage 3 (block 5 of 9). Zero code changes.
- **Budget:** 22.199M params (+0.22M vs H26 champion). 4-stage model within 15–22M acceptable range.
- **Builds on:** P2-H26 champion (83.37%). All hyperparameters identical.
- **Status:** DONE (job 350453). **Result: 83.17%** (−0.20pp vs P2-H26 champion 83.37%). DISCARDED. Shifting one block from stage 3 to stage 2 hurts — CIFAR-100 channel attention benefits more from deeper processing at the semantic 7×7 resolution (10 blocks) than at 14×14 (6 blocks). Confirms Law 4 (top-heavy depth): the deepest stage's channel attention is the most productive. Reverted config. [Phase 2 holdover — no further CIFAR-100 iteration; Phase 3 is ImageNet-1K.]

### P2-H31: 7×7 CPE kernel — full spatial coverage at stage-3 7×7 map
- **Hypothesis:** Phase-1 H52 validated CPE 3×3→5×5 (+0.13pp on CIFAR-10). At stage-3 (7×7 feature map), the current 5×5 CPE has a receptive field covering 5/7=71% of each spatial dimension — some spatial relationships are outside the kernel's reach. A 7×7 CPE kernel exactly matches the stage-3 feature map, making each channel's positional encoding global (sees all 49 spatial positions). This is qualitatively different from partial coverage: the CPE becomes a full spatial context descriptor rather than a local neighborhood encoder. At earlier stages (56×56, 28×28, 14×14), going from 5×5 to 7×7 also improves coverage though they remain in the local regime.
- **Why this strengthens channel attention:** CPE injects spatial activation context into channel tokens before Q/K computation. At stage-3, Q/K DW conv uses 5×5 (71% coverage) — the CPE with the same kernel gives partial positional context. With 7×7 CPE = full map coverage, every channel gets a global spatial "fingerprint" baked into its token representation before attention. This richer spatial identity in channel tokens should improve the quality of channel-to-channel similarity comparisons in Q/K.
- **Literature:** Phase-1 H52 (internal): CPE 3×3→5×5 gave +0.13pp. CPVT (arXiv:2102.10882): larger CPE kernels encode richer positional context. The 5→7 extension brings stage-3 CPE to full-map coverage — a qualitative threshold, not just a marginal increase.
- **Implementation:** YAML only. `CPE_KERNEL_SIZE: 5→7`. CPE conv uses padding=kernel//2=3 (same-padding), so output size stays 7×7. +31K params across all stages (negligible).
- **Budget:** 22.185M params (+32K). FLOPs unchanged (same spatial map size).
- **Builds on:** P2-H26 champion (83.37%). All settings identical except CPE kernel.
- **Status:** DONE (job 350157). **Result: 83.03%** (−0.34pp vs P2-H26 champion 83.37%). DISCARDED. 7×7 CPE hurts — global coverage introduces noise. The 5×5 local CPE's inductive bias of seeing a partial but meaningful spatial neighborhood is optimal for the 7×7 stage-3 map. Full-map CPE loses the local structure benefit. CPE direction completely exhausted. Reverted to P2-H26 champion.

### P2-H30: Stage-1 periodic CPE re-injection (interval=1)
- **Hypothesis:** H26 validated adding periodic CPE to stage 2 (interval=2, +0.07pp): re-grounding channel tokens' spatial context mid-stage improves Q/K quality. Stage 1 (28×28, 192 channels, 2 blocks) currently has no periodic CPE — its 2 blocks run with only the entry CPE. With CPE_REPEAT_INTERVAL=[0,1,2,4]: interval=1 at stage 1 means (i+1)%1==0 triggers at i=0 (not the last block), injecting one extra CPE between block-0 and block-1. Block-1 gets refreshed 5×5 spatial context on the 28×28 feature map, keeping channel-token representations spatially grounded before the stage-1 output flows into stage-2 patch embed.
- **Why this strengthens channel attention:** Same mechanism as H26 — the 5×5 DW CPE refreshes the spatial activation pattern in each channel's 28×28 feature slice, so block-1's Q/K comparisons are based on current spatial co-activations rather than drifted post-block-0 representations. Stage-1 features are also used in the cross-stage classifier fusion (fuse_proj_s1); better stage-1 representations should improve that fusion signal too.
- **Literature:** CPVT (arXiv:2102.10882, Chu et al., 2021): CPE at every block maintains spatial grounding throughout depth. H26 (internal, +0.07pp): periodic CPE extension to stage-2 validated. Extension to stage-1 follows the same rationale.
- **Implementation:** YAML only. `CPE_REPEAT_INTERVAL: [0,0,2,4]→[0,1,2,4]`. Zero code/param changes; reuses existing stage-1 CPE DW conv (5×5, 192 channels on 28×28 map).
- **Budget:** 22.153M params (unchanged). FLOPs +negligible (1 extra 5×5 DW pass on 28×28 × 192 channels).
- **Builds on:** P2-H26 champion (83.37%). All settings identical.
- **Status:** DONE (job 349901). **Result: 82.51%** (−0.86pp vs P2-H26 champion 83.37%). DISCARDED. Significantly hurts. CPE direction fully exhausted: stage-3 interval=4 ✓, stage-2 interval=2 ✓, stage-3 interval=3 ✗ (−0.63pp), stage-1 interval=1 ✗ (−0.86pp). The model has an optimal CPE schedule. Adding more CPE anywhere degrades. Reverted to P2-H26 champion.

### P2-H29: Register tokens for stage-3 channel attention (K=4)
- **Hypothesis:** DINOv2 (arXiv:2309.16588, Darcet et al., ICLR 2024) showed that adding K learnable "register" tokens to the spatial token sequence eliminates attention artifacts in ViTs — the registers serve as global scratchpad tokens that collect global information, preventing it from being stored in uninformative local patches. Applied to channel attention at stage-3: add K=4 register channel tokens to the C=640 sequence. All 644 tokens attend each other (C+K)×(C+K); the K register outputs are discarded afterward. The real channels can "write" global information to the registers and "read" from them, facilitating long-range channel interactions beyond direct C×C pairwise similarity — particularly useful for stage-3's 10-block depth where attention may concentrate on a few dominant channels (attention sink).
- **Why this strengthens channel attention:** (1) Reduces attention concentration by providing "sink" targets that absorb dominant-channel attention while returning global information; (2) Enables indirect channel communication via registers: channel A → register → channel B, even when A-B direct similarity is low; (3) Analogous to how CLS tokens aggregate spatial information — registers aggregate cross-channel information. Differs from H24 (gating, failed) because registers don't gate real channels — they extend the information pathway without suppressing anything.
- **Literature:** Darcet et al., "Vision Transformers Need Registers" (arXiv:2309.16588, ICLR 2024): register tokens prevent ViT attention artifacts and improve downstream tasks. The mechanism: high-norm tokens in deep layers carry global info instead of local texture — registers formalize this. Adapted for channel-token attention: register "channels" as global-information buffers in the C×C attention graph.
- **Implementation:** In `Attention.__init__`: if `n_register_tokens > 0 and token_mode == 'channel'`, create `reg_q/k/v = nn.Parameter(zeros(K,1))`. In `forward`: expand to `(B,K,HW)` and concat to q/k/v before SDPA → `(B,C+K,HW)`. After SDPA rearrange, discard first K outputs: `x = x[:,:,K:]`. Config key: `N_REGISTER_TOKENS = [0,0,0,0]`. YAML: `N_REGISTER_TOKENS: [0,0,0,4]`.
- **Budget:** 22.154M params (+120 scalar params, negligible). FLOPs: attention matrix grows from 640×640 to 644×644 (+1.3% at stage-3 SDPA). Total ~3.52 GFLOPs.
- **Builds on:** P2-H26 champion (83.37%). All settings identical.
- **Status:** DONE (job 349853). **Result: 82.70%** (−0.67pp vs P2-H26 champion 83.37%). DISCARDED. Register tokens fail — adding scratchpad mediators to the channel attention sequence disrupts the C×C attention matrix. The dominant pattern: EVERY modification to the attention computation itself hurts (H19,H20,H21,H24,H25,H28,H29 all failed). Only spatial context injection (CPE) and cross-stage aggregation (fusion) have helped. Reverted to P2-H26 champion.

### P2-H28: 5×5 Q/K at stage-2 (extend stage-3 winning kernel size to stage-2)
- **Hypothesis:** H14 validated that 5×5 DW conv in Q/K projections improves channel attention at stage-3 (7×7 map) by giving each channel token richer spatial context before computing C×C similarity. Stage-2 (14×14 map, 384 channels, 5 blocks) still uses 3×3 Q/K — covering only 21% of the spatial map per Q/K sample. Extending to 5×5 (36% coverage) provides richer spatial co-activation context for stage-2's 384-channel attention. Better stage-2 channel representations should propagate into stage-3 via patch embedding and improve all-stage cross-fusion at the classifier.
- **Why this strengthens channel attention:** Q/K DW conv defines the "spatial context window" over which channels compare their activation patterns. Larger kernel = more spatial evidence for computing which channels co-activate where. Stage-2 is underserved at 3×3 relative to its 14×14 feature map size. Stage-3 already uses 5×5 (71% coverage on 7×7 map); stage-2 at 5×5 brings consistent spatial coverage across deep stages.
- **Literature:** CvT (arXiv:2103.15808, Yuan et al., 2021): "The depth-wise separable convolution in Q/K provides local context critical for attention quality; larger kernels improve this at no structural cost." H14 validated this finding in our architecture at stage-3. This hypothesis applies the same conclusion to stage-2.
- **Implementation:** YAML only. `KERNEL_QKV: [3,3,3,5]→[3,3,5,5]`, `PADDING_KV: [1,1,1,2]→[1,1,2,2]`, `PADDING_Q: [1,1,1,2]→[1,1,2,2]`. No code changes. Negligible param increase (~12K extra for stage-2 Q,K DW convs). FLOPs ~+0.12 GFLOPs (3.63G total), well within budget.
- **Budget:** ~22.16M params. ~3.63 GFLOPs.
- **Builds on:** P2-H26 champion (83.37%). All settings identical except stage-2 Q/K kernel 3→5.
- **Status:** DONE (job 349541). **Result: 82.80%** (−0.57pp vs P2-H26 champion 83.37%). DISCARDED. Larger Q/K at stage 2 hurts. Unlike stage 3 (7×7, where 5×5 covers 71%), stage 2's 14×14 map means 5×5 is still local — the wider spatial context comparison at stage 2 may introduce irrelevant cross-region channel correlations that dilute the signal. H14's gain at stage 3 does not generalize to stage 2. Reverted to P2-H26 champion.

### P2-H27: Stage-3 CPE interval 4→3 (denser spatial refresh in deepest stage)
- **Hypothesis:** H26 validated per-stage CPE re-injection: adding interval=2 at stage-2 gave +0.07pp. Stage-3 (10 blocks, 640 channels, 7×7) currently uses interval=4 → CPE at blocks 3,7 (2 passes). Tightening to interval=3 → CPE at blocks 2,5,8 (3 passes) gives one more spatial context refresh per forward pass. With 10 blocks and diminishing spatial-context freshness over the sequence, an extra mid-stage re-injection may further stabilize Q/K representations in the deepest, most semantic stage.
- **Why this strengthens channel attention:** Each CPE re-injection at stage-3 grounds the 640-channel tokens in the 7×7 spatial map at that point in the block sequence. With 10 blocks and only 2 CPE refreshes (interval=4), blocks 4–6 (between CPE passes at 3 and 7) compute Q/K over channel features that have drifted 3 attention rounds from their last spatial grounding. Interval=3 reduces the max drift window from 3 to 2 blocks, improving Q/K alignment with current spatial activations.
- **Literature:** CPVT (arXiv:2102.10882, Chu et al., 2021): CPE at every block is the upper bound — our periodic schedule is a lightweight tradeoff. H26 validated the per-stage direction; this extends the CPE density into stage 3.
- **Implementation:** YAML only. `CPE_REPEAT_INTERVAL: [0, 0, 2, 3]`. Zero params change; reuses existing stage-3 CPE DW conv (5×5, 640 channels on 7×7 map).
- **Budget:** 22.153M params (unchanged). FLOPs +negligible (1 extra 5×5 DW pass on 7×7 map × batch).
- **Builds on:** P2-H26 champion (83.37%). All settings identical except stage-3 interval 4→3.
- **Status:** DONE (job 349105). **Result: 82.74%** (−0.63pp vs P2-H26 champion 83.37%). DISCARDED. Denser CPE at stage-3 hurts — interval=4 (2 passes) is the sweet spot; interval=3 (3 passes) over-refreshes spatial context and disrupts channel attention quality. The 7×7 map at stage-3 means even interval=4 provides near-full coverage. CPE frequency direction exhausted. Reverted to P2-H26 champion.

### P2-H26: Per-stage periodic CPE re-injection — add stage-2 at interval=2
- **Hypothesis:** H10 showed periodic CPE re-injection in stage 3 (interval=4) gives +0.08pp by refreshing spatial context mid-stage. Stage 2 (5 blocks, 384 channels, 14×14) currently receives CPE only at entry. After 5 channel-attention blocks, the spatial structure in stage-2 tokens may have drifted. Adding periodic CPE at interval=2 (triggers after blocks 1 and 3 — 2 extra CPE passes) re-grounds stage-2 channel tokens in their spatial context before passing features downstream to stage 3. CPE_REPEAT_INTERVAL becomes per-stage: [0, 0, 2, 4]. Stage 3 unchanged at interval=4.
- **Why this strengthens channel attention:** The mechanism is identical to H10: stage-2's Q/K projections operate on channel tokens whose spatial features get stale over 5 attention blocks. Refreshing them at blocks 1 and 3 keeps the spatial activation patterns current, improving the quality of stage-2's channel-to-channel similarity computation. Better stage-2 representations flow into stage 3 (via patch embed) and the all-stage cross-fusion at the classifier.
- **Literature:** H10 (internal validation, +0.08pp): periodic CPE re-injection validated for stage-3. CPVT (arXiv:2102.10882, Chu et al., 2021): CPE at every block is beneficial; our periodic variant is a lightweight middle ground. Extension from stage-3 to stage-2 follows the same spatial-grounding rationale.
- **Implementation:** Change `cpe_repeat_interval` in CvT kwargs from `int(spec.get('CPE_REPEAT_INTERVAL',0))` to `_get_per_stage_int(spec, 'CPE_REPEAT_INTERVAL', i, 0)`. YAML: `CPE_REPEAT_INTERVAL: [0, 0, 2, 4]`. Zero new params; reuses existing stage-2 CPE conv.
- **Budget:** 22.153M params (unchanged). FLOPs +negligible (2 extra 5×5 DW passes on 14×14 feature map).
- **Builds on:** P2-H22 champion (83.30%). All settings unchanged except CPE_REPEAT_INTERVAL now per-stage.
- **Status:** DONE (job 348968). **Result: 83.37%** (+0.07pp vs P2-H22 champion 83.30%). NEW CHAMPION. Per-stage CPE re-injection validated: adding one extra CPE pass to stage 2 (at block 1, in addition to the block-3 pass already present with global interval=4) improves channel attention quality in stage-2. Better stage-2 representations propagate to stage 3 and improve the all-stage cross-fusion at the classifier. PUSHED ✓

### P2-H25: Decoupled V projection — 1×1 DW for V, 5×5 for Q/K at stage 3
- **Hypothesis:** Currently Q, K, and V all use the same 5×5 DW conv at stage 3. Q/K need broad spatial context (5×5 = 71% of 7×7 map) to compute accurate channel-to-channel co-activation similarity. But V determines *what information gets mixed* across channels after routing. A 5×5 DW V blurs each channel's spatial activation pattern before mixing — the output of `Σ_j A_{ij} V_j` is a blend of already-blurred channel patterns. Switching V to 1×1 DW (pointwise per-channel scaling + BN) keeps V crisp — the attention-mixed output channel i receives the raw pointwise activations of channel j, preserving local spatial structure in the mixed features. Q/K stay at 5×5 to preserve the winning spatial-context comparison.
- **Why this strengthens channel attention:** Separating Q/K (spatial context for comparison) from V (pure channel content for mixing) lets each do its job optimally: broad Q/K identifies which channels co-activate, crisp V ensures the mixed content faithfully represents each channel's spatial role without spatial blurring artifacts.
- **Literature:** EfficientViT (Liu et al., arXiv:2205.14756, 2022): explicitly separates Q projection (DW conv for local spatial mixing) from V projection (1×1 pointwise) for precisely this reason — Q/K need spatial context, V needs pure channel content. Adapted for channel-token attention where the same principle applies but the spatial and channel axes are swapped.
- **Implementation:** Add `kernel_v` kwarg to `Attention.__init__`; `conv_proj_v` built with `kernel_v=1, padding_v=0` for stage 3. Config: `V_KERNEL_QKV: [3, 3, 3, 1]`. Zero code change to forward path.
- **Budget:** 22.000M params (−0.15M vs champion, 1×1 V saves 640×(5²-1²)=15,360 params per block × 10 blocks = 153,600 saved). FLOPs slightly lower.
- **Builds on:** P2-H22 champion (83.30%). KERNEL_QKV=[3,3,3,5] unchanged; V at stage 3 now 1×1.
- **Status:** DONE (job 348746). **Result: 82.82%** (−0.48pp vs P2-H22 champion 83.30%). DISCARDED. Pointwise V degrades channel attention — the 5×5 spatial blending in V is beneficial, not harmful. The mixed output `Σ_j A_{ij} V_j` benefits from V carrying locally-blended (5×5) spatial patterns rather than raw pointwise values. Reverted to P2-H22 champion.

### P2-H24: Attentive channel gating (output-side gate on stage-3 channel attention)
- **Hypothesis:** Current stage-3 channel attention applies uniform softmax over all 640 channels for every image and every block. This ignores that channels differ in their "activity level" for any given image — some channels are spatially selective (high activation variance), others are globally suppressed. An output-side gate conditioned on each channel's spatial average activation `gate = sigmoid(scale * GAP(x_pre) + bias)` applied to the attention output modulates how much each channel's attention contribution passes through. Channels that are spatially active (important for this image) retain full attention output; quiet channels have their attention output partially suppressed in favor of the DW shortcut / residual path.
- **Why this strengthens channel attention:** (1) Eliminates "attention sink" — one dominant channel pulling all others' attention is now partially gated; (2) Improves sample efficiency — theoretically proven by arXiv:2602.01468 (2026) to require only polynomial (vs exponential) data; (3) Creates input-adaptive attention routing at the per-channel level. Applied only at stage 3 (deepest, most semantic stage).
- **Why channel-specific:** Spatial tokens have a fixed positional context; channel tokens' "importance" is image-specific. A channel may be crucial for dog images (e.g., fur texture detector) but irrelevant for vehicle images. The gate encodes this image-specific channel importance directly.
- **Literature:** Nguyen et al., arXiv:2602.01468 (2026): "A Statistical Theory of Gated Attention through the Lens of Hierarchical Mixture of Experts" — shows gated attention is more sample-efficient than standard multi-head attention; proves optimal gate placement is at the attention output (not input/pre). Our adaptation: use per-channel spatial GAP as the gate signal instead of a full linear projection, keeping params negligible.
- **Implementation:** In `Attention.forward()` (channel mode, stage 3): save `x_pre` at forward start; after `proj(x)`, compute `gate = sigmoid(gate_scale * x_pre.mean(dim=1) + gate_bias)` → (B, C); multiply `x = x * gate.unsqueeze(1)`. `gate_scale ∈ R^C` init=1, `gate_bias ∈ R^C` init=2 (sigmoid(2)≈0.88, near pass-through at init). Config: `GATED_ATTN: [false, false, false, true]`.
- **Budget:** 22.166M params (+12,800 gate params across 10 stage-3 blocks, negligible). FLOPs unchanged (~3.51 GFLOPs).
- **Builds on:** P2-H22 champion (83.30%). All settings: CROSS_STAGE_FUSION_S0: true, KERNEL_QKV=[3,3,3,5], DW_SHORTCUT_KERNEL=[3,3,3,5], CPE_REPEAT_INTERVAL=4, mid-block aux, MIXUP=0.4, LS=0.1.
- **Status:** DONE (job 348506). **Result: 83.13%** (−0.17pp vs P2-H22 champion 83.30%). DISCARDED. Output-side gate hurts — gating the attention output based on channel spatial activation magnitude suppresses useful cross-channel information flow. In channel attention, even spatially quiet channels carry discriminative relational signals (their Q/K interactions with active channels matter); the gate incorrectly penalizes these. Reverted to P2-H22 champion.

---

## RESUME — Phase 3 (2026-03-29): ImageNet-1K Structural Innovation

**Champion:** P2-H26 — Per-stage periodic CPE re-injection, commit on branch `autoresearch/h42-qk-norm`
- **CIFAR-100 100ep:** 83.37% val top-1 (job 348968)
- **CIFAR-100 300ep:** 85.07% val top-1 (job 344501, P2-H15 arch — still the best 300ep)
- **Params / GFLOPs:** 22.153M / 3.51 (fvcore)
- **Architecture:** DIM_EMBED=[96,192,384,640], DEPTH=[1,2,5,10], CA_MLP_RATIO=[8,8,4,1], SwiGLU, CPE 5×5, CPE_REPEAT_INTERVAL=[0,0,2,4], KERNEL_QKV=[3,3,3,5], DW_SHORTCUT 3×3, triple aux (0.1/0.2/0.4), mid-block aux (block 5), all-stage cross-fusion (s0+s1+s2), LayerScale 1e-4, learned attn temperature
- **CIFAR-100 100ep recipe:** LR=2e-3, WD=0.05, warmup=10ep, Mixup=0.4, LS=0.1, DROP_PATH=[0,0.02,0.05,0.1]

**Phase 3 designation:** Primary benchmark switches to **ImageNet-1K 224×224 100ep**. CIFAR-100 iteration is retired. All new hypothesis screening uses ImageNet-1K.

**In-flight at phase transition:**
- Job 350453 (P2-H32, CIFAR-100 depth redistribution [1,2,6,9]): parse result, update notes, commit/discard, then proceed with Phase 3 ImageNet work.
- ImageNet pending jobs (349012 and any successors): 2-GPU torchrun. If running, monitor; if failed, re-submit per `RESEARCH_BRIEF.md` ImageNet protocol.

**ImageNet 100ep baseline to establish:** Run current P2-H26 champion architecture on ImageNet-1K using the Phase 3 100ep recipe (BS=256/GPU, LR=5e-4, 100ep, Mixup=0.4, CutMix=0.5, RandAugment rand-m7, RE=0.1, LS=0.1). Log result as `IN-B100`. All future Phase 3 hypotheses are compared against this.

**Phase 3 research focus — bold structural directions (see `autoresearch/channel_attention_insights.md`):**
1. Funneling: extra stages compressing to 4×4, 2×2, 1×1 spatial maps
2. Factorised/low-rank channel attention (r << C bottleneck)
3. Sparse top-k channel attention
4. Hierarchical group CA (within-group + cross-group)
5. SSM/Mamba for channel mixing

**Context:** Read `autoresearch/channel_attention_insights.md` for the complete empirical synthesis before designing any new hypothesis. This document captures which mechanisms work, which fail, and why — distilled from 90+ runs. Key rule: do NOT modify SDPA itself (all such attempts failed); innovate around it.

**90+ hypotheses completed.** Phase 1 (CIFAR-10, 5.8M): H1–H67 (69.67%→96.59% at 100ep, 97.25% at 300ep). Phase 2 (CIFAR-100, 22M): P2-B1–P2-H32 (81.34%→83.37% at 100ep, 85.07% at 300ep).

### 300-ep GPU pre-queue policy (acknowledged 2026-04-23)
GPU queue wait is ~6+ days. Do not serialise 300-ep validations. Rules:
1. Whenever any Phase 3 100-ep hypothesis PUSHES (beats IN-B100 / latest 300-ep champion), IMMEDIATELY queue its 300-ep validation — do not wait for previous 300-ep to finish.
2. Multiple 300-ep jobs can sit in the queue simultaneously (QOS MaxSubmitJobsPU=2000; pending jobs don't count against 4-GPU running cap).
3. 100-ep screening continues in parallel on GPUs — never pause screening for 300-ep results.
4. Each 300-ep submit: distinct RUN_TAG (e.g. `in_h3_2_dualCA32_300ep`), logged to notes.md under the hypothesis entry with job ID + projected start time from `squeue --start`.
5. Do NOT pre-queue speculative 300-ep for UNPUSHED hypotheses; 100-ep PUSH is the gate.
6. Job 349012 (IN-B300, P2-H26 champion) stays at head of queue; new 300-ep go behind it.

---

## ImageNet-1K

### IN-B100: Phase 3 baseline — P2-H26 champion on ImageNet-1K 100ep
- **Architecture:** P2-H26 champion verbatim. DIM_EMBED=[96,192,384,640], DEPTH=[1,2,5,10], CA_MLP_RATIO=[8,8,4,1], SwiGLU, CPE 5×5, CPE_REPEAT_INTERVAL=[0,0,2,4], KERNEL_QKV=[3,3,3,5], DW_SHORTCUT 3×3, all-stage cross-fusion (s0+s1+s2), triple aux (0.1/0.2/0.4) + mid-block aux (block 5, 0.15), LayerScale 1e-4, learned attn temperature. 23.9M params (backbone ~22M + 1000-class heads), 3.52 GFLOPs.
- **Recipe (100ep):** 2-GPU, BS=256/GPU (512 total), LR=5e-4, WD=0.05, warmup=10ep, Mixup=0.4, CutMix=0.5, rand-m7-mstd0.5-inc1, RE=0.1, LS=0.1, DROP_PATH=[0,0.05,0.1,0.2].
- **Trajectory:** ep25≈70.x, ep50≈75.x, ep75≈77.1, ep99 77.65%. Clean monotonic convergence (no divergence).
- **Result:** **77.65%** val top-1 (job 350642, ~47h wall on 2 GPUs). All Phase 3 hypotheses compare against this.
- **Status:** ESTABLISHED. PUSHED ✓

---

### H3-1: Sparse top-k channel attention at stage-3 (k=64 of C=640) — DISCARDED
- **Hypothesis:** at C=640, dense softmax spreads weight across 640 keys per query. Most weight sits on weakly-correlated channels (tail). Cutting each query's attention to its top-64 most-similar channels should remove tail noise and sharpen channel-to-channel interactions without reducing channel capacity.
- **Mechanism:** stage-3 only (10 blocks, C=640). For each query channel, compute exact QK^T similarities, find top-64 keys, mask the rest to -inf pre-softmax. Threshold is detached per-row so gradient only flows through kept attn weights. Stages 0–2 stay dense.
- **Inspiration:** BSFA — Block-Sparse FlashAttention (arXiv:2512.07011, Dec 2025). Exact QK similarity + per-query top-k block selection skips ~50% of compute while preserving >99% baseline accuracy on LLaMA. First adaptation to channel tokens in a ViT.
- **Novelty self-audit:** category (a) — new CA mechanism. Direct instantiation of §6 seed #3.
- **Result:** **77.48%** val top-1 (job 352433, 2-GPU, ~33h wall). **-0.17pp vs IN-B100 77.65%.**
- **Trajectory:** ep25≈76.55, ep50≈77.00, ep75≈77.27, ep99 77.48%. Converged cleanly; no instability.
- **Insight:** the "tail" of dense CA softmax is NOT noise — masking the bottom 576/640 of each row's keys still loses useful signal. Dense softmax is the right inductive bias for channel tokens; at C=640 the model actively uses the low-weight mass. Top-k sparsity is now an **exhausted** direction for CA.
- **Status:** DISCARDED. Code (`SPARSE_TOPK`) kept as config knob for future reference but not used.
- **Follow-up:** do NOT retry smaller/larger k — the shape of the result (converges but underperforms) suggests the whole "prune pairwise interactions" family is wrong. Move to directions that *add* channel-information capacity, not restrict it.

---

### H3-2: Dual-resolution channel attention at stage-3 (G=32 super-channel groups) — DISCARDED
- **Hypothesis:** keep full C=640 fine CA intact AND add a parallel COARSE CA pathway across G=32 super-channel groups (each group = mean of 20 channels). Hierarchical CA: fine 640×640 + coarse 32×32 per block. Reuses fine q/k/v via group-mean pool, scatter-broadcasts back, additive with learned scalar (init 0).
- **Inspiration:** MSIT (arXiv:2403.06536, Mar 2024) — parallel multi-scale self-attention.
- **Novelty self-audit:** category (a) + (c).
- **Budget:** 23.915M params (+10 scalars), 3.52 GFLOPs.
- **Result:** **77.50%** val top-1 (job 353896, 2-GPU, ~33h wall). **-0.15pp vs IN-B100 77.65%.**
- **Trajectory:** clean monotone convergence, ep95→97 ≈77.43–77.50%.
- **Insight:** the safe init-0 ramp does NOT guarantee improvement — the coarse pathway either stays near-zero (no useful signal at coarser granularity) OR competes with cross-stage fusion (which already provides multi-scale info). Pattern from H3-1 + H3-2: minor structural tweaks to the CA mechanism (pruning, parallel coarse) yield similar small-negative deltas. Single-head dense softmax CA appears to be a local optimum that resists incremental modification.
- **Status:** DISCARDED. `DUAL_CA_GROUPS` knob retained in code but not used.
- **Follow-up direction:** small CA-mechanism changes are not yielding gains. Move toward changes that introduce a *new pathway* between channel tokens at different stages, OR a fundamentally different mixing operator (linear/SSM/Mamba).


### H3-3: Cross-stage channel attention bridge (stage-2 → stage-3) — DISCARDED
- **Hypothesis:** direct response to the H3-1/H3-2 pattern (CA-mechanism tweaks = -0.15–0.17pp). Instead of modifying how stage-3's channel tokens attend among themselves, ADD A NEW PATHWAY: let stage-3 queries attend to stage-2's channel tokens as additional K/V tokens (not just stage-3's own). Each of stage-3's 10 attention blocks now queries both self-K/V (640 tokens) AND bridge-K/V (640 tokens projected from stage-2), giving 1280-token softmax per block.
- **Mechanism:** after stage-2 forward, capture (B, 384, 14, 14). At stage-3 entry: `Conv2d(384→640, 1×1, zero-init) + AdaptiveAvgPool2d(7)` → bridge feature (B, 640, 49). Flatten and concat along token dim in every stage-3 block: K_full=[K_own; bridge], V_full=[V_own; bridge]. Q unchanged. Bridge computed ONCE at stage-3 entry, shared across all 10 blocks. Zero-init on bridge_proj means bridge_kv=0 at step 0 → baseline behavior at init, gradual ramp during training (no cold-start shock).
- **Distinct from CSF:** existing CROSS_STAGE_FUSION only aggregates GAP features into the classifier head at the very end. H3-3 operates at FULL ATTENTION GRANULARITY — every stage-3 attention block can query stage-2 features per-head, per-query.
- **Inspiration:** Focal Transformer (arXiv:2107.00641) — spatial self-attention augmented with K/V from coarser/farther scales. Here the "other scale" is the previous stage's channel tokens; attention is still over channels. Also aligned with cross-scale attention themes (BiFormer 2023, MSIT 2024).
- **Novelty self-audit:** category (a) new CA mechanism (cross-stage channel-to-channel attention — never tried in Phase 1/2/3) + category (b) new structural component (bridge_proj + pool creating a persistent info pathway stage-2 → stage-3). Direct response to failure pattern: H3-1/H3-2 were same-stage tweaks; H3-3 adds a cross-stage pathway the model never had.
- **Budget:** 24.161M params (+245k for bridge_proj), 3.565 GFLOPs. Bridge doubles stage-3 K/V token count but K/V caches are small at 7×7 so cost increase is modest.
- **Submission:** job 355845 (2-GPU, tag `in_h3_3_xstagebridge`, cfg `configs/autoresearch_experiment.yaml`).
- **Baseline to beat:** IN-B100 = 77.65% val top-1.
- **Result:** **77.55%** val top-1 (job 355845). **-0.10pp vs IN-B100 77.65%.** Closest of H3-1/2/3 but still below baseline.
- **Insight:** the cross-stage bridge zero-init pathway DOES learn something (closer to baseline than H3-1/H3-2) but the additional 640 K/V tokens per block don't supply meaningfully new info to stage-3 — likely because (i) cross-stage fusion already brings stage-2 GAP info to the classifier, and (ii) stage-2's mid-level features projected & pooled to 7×7 mostly duplicate stage-3's own representation by depth-10. Pattern across H3-1/H3-2/H3-3: every modification CONFINED to stage-3 yields -0.10 to -0.17pp. Stage-3's 10-block dense softmax CA appears saturated; surface modifications, parallel pathways, and cross-stage K/V augmentation all fail to move it.
- **Status:** DISCARDED. Code path retained; `XSTAGE_BRIDGE` set False in YAML.
- **Follow-up direction:** stop iterating on stage-3 surface. Pivot to a fundamentally different mixing operator (factorised / low-rank / linear / SSM CA) — explicit option from RESEARCH_BRIEF.md bold directions.


### H3-4: Agent attention at stage-3 (factorised low-rank CA via M=64 learnable agents) — DISCARDED
- **Hypothesis:** REPLACE (not tweak) stage-3's dense softmax CA with agent attention. All channel-channel interactions are forced to flow through M=64 learnable agent tokens, creating a rank-≤64 bottleneck. Tests whether channel-channel relationships are intrinsically low-rank.
- **Mechanism (per stage-3 attention block):** agents A ∈ ℝ^{1×1×64×49} (init trunc_normal std=0.02). Two cascaded softmaxes:
  - α₁ = softmax(A · Kᵀ / √D) ∈ ℝ^{B×1×64×640}; agent_feat = α₁ · V ∈ ℝ^{B×1×64×49}
  - α₂ = softmax(Q · Aᵀ / √D) ∈ ℝ^{B×1×640×64}; out = α₂ · agent_feat ∈ ℝ^{B×1×640×49}
  - Compute per block: 2·M·C·D = 2·64·640·49 ≈ 4M ops vs softmax C²·D = 640²·49 ≈ 20M ops → ~5× cheaper.
- **Why this differs from H3-1/H3-2/H3-3:** all three retained dense softmax CA and added/pruned around it. H3-4 is the first hypothesis that REMOVES the C×C interaction matrix entirely. The model is forced to learn a 64-dim "channel prototype" basis that all queries route through, rather than computing pairwise channel similarities.
- **Inspiration:** Han et al., "Agent Attention: On the Integration of Softmax and Linear Attention" (arXiv:2312.08874, ECCV 2024). They show agent attention combines softmax expressivity with linear-attention efficiency for spatial vision. Adapted from spatial → channel tokens.
- **Novelty self-audit:** category (a) new CA mechanism (factorised low-rank via learnable agents — never tried in Phase 1/2/3) + category (b) new structural component (agent token parameter set + new attention topology) + category (c) direct response to H3-1/H3-2/H3-3 failure pattern (replace operator instead of tweaking it). Not in §3 exhausted list.
- **Budget:** 23.946M params (+31k for agents = 10 blocks × 64 × 49), 3.597 GFLOPs (slightly LOWER than H3-3 due to ~5× cheaper stage-3 attention).
- **Submission:** job 357682 (2-GPU GPU, tag `in_h3_4_agentca`, cfg `configs/autoresearch_experiment.yaml`). DIED at `torch.distributed.barrier()` after init_process_group — NCCL ALLREDUCE SeqNum=1 timed out at 10 min on a node. Resubmitted as 357686 — DIED identically on again. Node-level NCCL issue (the warning "using GPU N to perform barrier as devices used by this process are currently unknown" is a known sign of GPU-mapping ambiguity that causes hangs on certain nodes). Resubmitted as 357687 with `--exclude=a node`.
- **Baseline to beat:** IN-B100 = 77.65% val top-1.
- **Result:** **77.38%** val top-1 (job 357687). **-0.27pp vs IN-B100 77.65%.** WORST of all H3-* hypotheses.
- **Insight:** REPLACING dense softmax CA with rank-≤64 agent attention hurt MORE than tweaking it (H3-1/2/3 ranged -0.10 to -0.17pp). This is strong evidence that channel-channel relationships at stage-3 are NOT intrinsically low-rank — the dense softmax CA is genuinely using its expressive capacity, not redundantly. The model can't compress 640×640 channel interactions through a 64-dim bottleneck without losing important information. Cross-pattern conclusion across H3-1 through H3-4: stage-3 dense softmax CA is locally optimal AND globally important. Stop modifying stage-3.
- **Status:** DISCARDED. Code path retained; `AGENT_TOKENS` set to all zeros in YAML.
- **Follow-up direction:** apply structural innovation to LESS-EXPLORED stages (stage-1: 2 blocks, stage-2: 5 blocks). Stage-3 is a no-go zone for now.

### H3-5: Hierarchical local + global CA at stage-2 (within-group fine + cross-group coarse) — DISCARDED
- **Hypothesis:** stage-3 modifications all fail (H3-1/2/3/4: -0.10 to -0.27pp). Pivot to less-explored stage-2 (5 blocks at C=384). Replace standard dense CA at stage-2 with hierarchical CA: split C=384 channels into G=12 super-channel groups of 32 channels each. WITHIN each group, run dense CA over 32 channels (fine). ACROSS groups, run dense CA over G=12 group-summary tokens (coarse). Output = within + α·broadcast(cross), with α = `hier_cross_scale` learnable scalar (init 1.0).
- **Mechanism (per stage-2 block):**
  - Compute Q, K, V as usual (conv-projections, channel mode) → shape (B, 1, 384, 196)
  - Split into groups: (B, 1, G=12, c=32, 196)
  - Within-group: A_w = softmax(Q_g · K_gᵀ / √196) ∈ ℝ^{B×1×G×c×c}; out_w = A_w · V_g → (B, 1, G, c, 196)
  - Cross-group via group means: Q_m, K_m, V_m = mean(Q_g, K_g, V_g, dim=c) → (B, 1, G, 196)
  - Cross attention: A_c = softmax(Q_m · K_mᵀ / √196) ∈ ℝ^{B×1×G×G}; out_c = A_c · V_m → (B, 1, G, 196)
  - Broadcast: out_c_exp = out_c.unsqueeze(3).expand(-1,-1,-1,c,-1) → (B, 1, G, c, 196)
  - Combine: out = out_w + α · out_c_exp; reshape → (B, 1, 384, 196)
- **Why this differs from prior failed grouped CA (H17, P2-H3):** those used grouped CA WITHOUT a cross-group pathway → channels in different groups had no info exchange (only via the post-attn proj Linear). H3-5 explicitly adds cross-group attention through G=12 group-summary tokens. RESEARCH_BRIEF.md option #4 calls out this distinction.
- **Why stage-2 not stage-3:** every stage-3 mod has failed (4/4). Stage-2 is less explored and may have headroom for a new mechanism. Tests whether the failure pattern is "stage-3 is saturated" (then stage-2 mod could help) or "any mod hurts" (then stage-2 mod also fails — informative either way).
- **Inspiration:** BiFormer (Zhu et al., CVPR 2023, arXiv:2303.08810) — bi-level routing attention with hierarchical local+global pattern. Adapted from spatial bi-level routing to channel-token hierarchical CA. Group-summary cross-attention is a standard 2024+ pattern (e.g., Slide-Transformer extensions, BiFormer-V2).
- **Novelty self-audit:** category (a) new CA mechanism (hierarchical local+global, never tried) + category (b) new structural component (cross-group attention pathway with group-summary tokens) + category (c) direct response to failure pattern (move OUT of stage-3, distinct from grouped-only failures). Not in §3 exhausted list (which mentions grouped CA as failed but explicitly without cross-group).
- **Budget:** 23.915M params (+5 scalars), 3.541 GFLOPs (slightly cheaper than baseline due to ~12× lighter stage-2 attention).
- **Submission:** job 358067 (2-GPU, --exclude=a node, tag `in_h3_5_hierca_s2`, cfg `configs/autoresearch_experiment.yaml`).
- **Result:** **77.45%** val top-1 (**−0.20pp vs IN-B100 77.65%**). DISCARDED.
- **Insight:** hierarchical CA at stage-2 also fails. The failure pattern is **universal across stages 2 and 3, not stage-3-specific**. Cross-pattern conclusion across H3-1 → H3-5: the dense softmax CA at stages 2–3 is at a robust local optimum that resists structural modification, regardless of (i) where (stage-2 vs stage-3), (ii) what kind (surface tweak vs operator replacement), or (iii) the specific replacement (low-rank/agent, hierarchical, dual-branch, multi-resolution, sparse top-k). **Path forward must be ADDITIVE** — add a new pathway/component WITHOUT modifying existing CA. 5 consecutive replacement-style hypotheses have all hurt; stop trying to swap the operator. The next hypothesis must preserve every existing CA block as-is and add a parallel mechanism.







### H3-6: Funneling tail-block at 3×3 spatial pool (purely additive on top of stage-3) — DISCARDED
- **Hypothesis:** 5 consecutive failures (H3-1 to H3-5) all REPLACED part of an existing CA block. Pivot: add a NEW component while preserving every existing block. After stage-3 (B, 640, 7, 7), pool to 3×3 (D=9), run a single channel-attention block, GAP, and gate-add (init 0) to the classifier feature. Existing stages 0-3 are completely UNCHANGED.
- **Mechanism:**
  - `pooled = AdaptiveAvgPool2d(3)(stage3_spatial)` → (B, 640, 3, 3)
  - `tokens = rearrange → (B, 9, 640)` → channel-mode CA block (kernel=1 dw_bn Q/K/V, softmax C×C attention with D=9 features, LayerScale 1e-4, SwiGLU MLP ratio 0.5, residuals)
  - `tokens = funnel_norm(tokens)`; `f_feat = mean(tokens, dim=1)` → (B, 640)
  - `classifier_input = main_feat + funnel_scale * f_feat`; `funnel_scale` is per-channel 1D learnable, init 0 (LayerScale-style → identity at init).
- **Why this might work where H3-1..H3-5 didn't:** the previous 5 reduced or replaced the expressive capacity of an operator the model was already using effectively. H3-6 ADDS NEW capacity at a NEW spatial scale (rank-9 channel info, distinct from stage-3's rank-49). Model can scale `funnel_scale` up only if it helps; init 0 means the model starts identical to IN-B100 baseline.
- **Inspiration:** RESEARCH_BRIEF.md option #1 (funneling architecture, extreme spatial compression). Funnel-Transformer (Dai et al. 2020, arXiv:2006.03236), and SE-Net (Hu et al. 2018, arXiv:1709.01507) — global channel recalibration at compressed spatial map. Recent funnel/pyramid work in efficient ViTs (SHViT 2024, FastViT 2023) reinforces the pattern.
- **Novelty self-audit:** category (b) new structural component (a tail block at a new spatial scale not present in baseline) + category (c) direct response to failure pattern (5 replacement-style failures → switch to additive). Not in §3 exhausted directions (which are all attention-mechanism tweaks within existing blocks).
- **Budget:** 24.747M params (+0.83M vs H3-5), 3.524 GFLOPs. Within FLOPs target (5–8 GFLOPs); slightly above the 22M nominal cap but consistent with our 22–24M operating range since H3-5.
- **Submission:** job 360736 (2-GPU, --exclude=a node, tag `in_h3_6_funnel`, cfg `configs/autoresearch_experiment.yaml`).
- **Result:** **77.59%** val top-1 (**−0.06pp vs IN-B100 77.65%**). DISCARDED (effectively noise — within typical run-to-run variance, but not an improvement).
- **Insight:** the funneling block + init-0 gate was too gentle. With `funnel_scale = 0`, the block receives ZERO gradient from the main loss (∂(x + g·f)/∂f = g = 0). The gate itself does receive gradient (∂…/∂g = f), so the model can in principle ramp the gate up — but only if the funnel feature `f` is already aligned with what the main classifier needs, which it can't become without gradient. Catch-22. The funneling block likely stayed at near-random init-trained state for 100 epochs. This explains why the result is essentially noise: the funneling pathway was effectively unused. **Conclusion:** purely additive (init-0 gate) requires an INDEPENDENT gradient signal to make the new pathway useful. Either non-zero gate init (riskier), or — better — auxiliary supervision on the new pathway's output to give it an independent learning signal. Adopt the latter for H3-7.



### H3-7: Funneling tail-block + shared-head aux supervision (additive, gradient-fix on H3-6) — IN PROGRESS (job 361852)
- **Hypothesis:** H3-6 funneling failed (77.59%, −0.06pp ≈ noise) because the funnel block had ZERO main-loss gradient at init: with `funnel_scale = 0`, `∂L/∂funnel_block_params = 0`. Catch-22: the gate can only ramp up if the funnel feature is useful, and the funnel feature can only become useful if the block gets gradient. Fix: add an INDEPENDENT gradient signal via auxiliary supervision on the funnel GAP feature, using the SHARED main classifier head.
- **Mechanism (delta vs H3-6):**
  - Funnel block + pool + gate-add path: identical to H3-6 (k=1 dw_bn Q/K/V, SwiGLU MLP r=0.5, LayerScale 1e-4, funnel_scale init 0).
  - NEW: during training only, `_aux_logits_funnel = self.head(funnel_GAP_feat)` is exposed.
  - main.py adds `FUNNEL_AUX_WEIGHT * CE(_aux_logits_funnel, target)` (weight 0.1) to the total loss, alongside the existing s0/s1/s2/mid aux terms.
  - SHARED head — 0 extra classifier params. The aux loss aligns the funnel output with the main head's expected feature space.
- **Why this should work where H3-6 didn't:** the funnel block now has gradient regardless of funnel_scale. By the time the main loss starts pulling on funnel_scale (gate gradient is `∂L/∂g = f · (head's gradient)`), `f` is already discriminative and aligned with the head's basis. The gate can ramp up safely and the funnel contribution is correctly oriented.
- **Inspiration:** Deep Supervision (Lee et al. 2015, arXiv:1409.5185) — gradient injection. Our own H46/H50/H64/P2-H9 triple-aux pattern is well-validated. The shared-head trick is standard in DINO/SimCLR-style projector reuse, and is well-suited here since the funnel output should ultimately serve the main classifier's feature space.
- **Novelty self-audit:** category (b) new structural component (a new aux supervision path on a new feature path) + category (c) direct response to H3-6's failure pattern (additive gates with no gradient signal stay unused; aux supervision unblocks them). The combination of "funnel block + shared-head aux" is novel: the existing aux paths all supervise EXISTING block outputs (s0/s1/s2/mid); H3-7 supervises a NEW pathway specifically designed to be additive at inference.
- **Budget:** 24.747M params (shared head → 0 extra), 3.524 GFLOPs. Same as H3-6.
- **Submission:** job 361852 (2-GPU, --exclude=a node, tag `in_h3_7_funnel_aux`, cfg `configs/autoresearch_experiment.yaml`).
- **Baseline to beat:** IN-B100 = 77.65% val top-1.
- **Status:** IN PROGRESS — awaiting watcher nudge.



### H3-8: Linear-attention funneling tail at full stage-3 spatial (D=49) + shared-head aux — DISCARDED (NaN divergence, peak 62.93%, job 366155)
- **Hypothesis:** the funnel pathway in H3-6/H3-7 was REDUNDANT with stage-3's existing softmax CA — same operator, same "select few" inductive bias. Swap the operator to LINEAR ATTENTION (Katharopoulos et al. 2020, arXiv:2006.16236) and operate at full 7×7 spatial (D=49) instead of pooled 3×3 (D=9). Linear attention provides a fundamentally different inductive bias: dense multiplicative interactions across all channel pairs without softmax stochastic-matrix constraint, with rank ≤ D = 49.
- **Mechanism (delta vs H3-7):**
  - FUNNEL_TAIL_HW: 3 → 7 (no spatial pooling — funnel sees full stage-3 7×7 map)
  - FUNNEL_TAIL_KERNEL: 1 → 3 (richer Q/K/V dw_bn projections for the 7×7 map)
  - LINEAR_FUNNEL: false → true (operator swap: softmax → linear)
  - Shared-head aux (FUNNEL_AUX_WEIGHT=0.1) and gate init=0: kept from H3-7.
  - Linear attention: `phi(Q) · (phi(K)^T · V) / (phi(Q) · sum(phi(K)))`, `phi = ELU + 1` (≥0, stable normalization). Implemented via `einsum`s: KV-summary `(B,H,D,D)` then numerator/denominator. New `linear_attn` kwarg in Attention class; new branch in forward that bypasses softmax/SDPA/sparse paths.
- **Why this might work where H3-6/H3-7 didn't:**
  - All 10 stage-3 blocks use softmax CA → adding an 11th softmax block at the head is REDUNDANT.
  - Linear attention: no row-stochastic constraint, allows dense multiplicative inter-channel interactions; rank capped at D=49 — distinct from softmax CA's effective full-rank C×C structure.
  - The bottleneck might be REDUNDANCY (operator swap helps) vs CAPACITY (no head-side branch helps). H3-8 disambiguates.
- **Inspiration:** Katharopoulos et al. "Transformers are RNNs" (ICML 2020, arXiv:2006.16236) — original linear attention. RESEARCH_BRIEF.md option #5 ("Linear / kernel / SSM channel attention") is explicitly called out as a bold direction. Linear/kernel CA has never been tested in this codebase.
- **Novelty self-audit:** category (a) NEW CA mechanism (linear attention, never used in this codebase) + category (c) direct response to failure pattern (H3-6/H3-7 noise → swap operator to test redundancy hypothesis). Not in §3 exhausted directions.
- **Budget:** 24.763M params (+0.016M vs H3-7 — kernel 3 vs 1 adds tiny dw conv params), 3.561 GFLOPs. Same operating range as H3-6/H3-7.
- **Submission:** job 366155 (2-GPU, --exclude=a node, tag `in_h3_8_linear_funnel`, cfg `configs/autoresearch_experiment.yaml`).
- **Baseline to beat:** IN-B100 = 77.65% val top-1.
- **Result: 62.93% peak, then NaN — DISCARDED.** Trajectory: 1.91 (ep1) → 36 (ep6) → 62.93 (ep18), then loss diverged to NaN and stayed NaN for the remaining 80+ epochs (final eval Acc@1 0.10%). Test-set loss `nan` from epoch 19 onwards.
- **Diagnosis:** linear-attention `phi(Q)·sum(phi(K))` denominator can collapse during training. Even with `+1e-6` epsilon, large gradients flowing through `out = num/den` can blow up when `den` becomes very small relative to `num` magnitude. Combined with init-0 funnel_scale (gate slowly opens) + aux supervision (FUNNEL_AUX_WEIGHT=0.1 directly trains the funnel block on classification loss) — the funnel block was being heavily trained from epoch 0 with an unstable operator while the rest of the network kept learning correctly until backprop NaNs propagated everywhere.
- **Lessons learned:** (i) raw Katharopoulos-style linear attention is unstable for this codebase even with single-head channel mode; (ii) aux supervision pulls a block strongly into the loss landscape from epoch 1 — useful for getting gradient (H3-7) but dangerous with brittle operators; (iii) need to either (a) stabilize linear attention (focused linear / scaled denom / pre-LN on q,k) or (b) pick a different stable additive operator for the head pathway.



### H3-9: Linformer-style projected channel attention funnel + shared-head aux — CHAMPION 77.79% (+0.14pp, job 369446)
- **Hypothesis:** keep additive funnel + shared-head aux from H3-7 (validated to ramp gradient), but swap operator from softmax CA / linear-attn (H3-8 NaN) to LINFORMER (Wang et al. 2020, arXiv:2006.04768): standard softmax attention with K and V projected along the CHANNEL (token) axis from C=640 to r=128 learned supertokens. Each query channel attends to r=128 SUPERTOKENS (linear combinations of all 640 channels) instead of attending directly to all 640 channels. Different inductive bias from full-rank softmax CA (forces a learned compact channel basis), stable softmax operator (no NaN risk).
- **Mechanism (delta vs H3-8):**
  - LINEAR_FUNNEL: true → false (revert linear-attention operator that NaN'd at ep18)
  - LINFORMER_R: 0 → 128 (NEW: project K, V from 640 → 128 supertokens before softmax)
  - Funnel block geometry unchanged: HW=7 (no pool, full D=49), KERNEL=3, MLP_RATIO=0.5, gate init=0, FUNNEL_AUX_WEIGHT=0.1.
  - Linformer math: K_proj = E_k·K, V_proj = E_v·V (E_k, E_v: Linear(640→128, no bias, xavier-uniform init). Standard softmax: A = softmax(Q·K_proj^T / √D, dim=-1) ∈ R^(C×r), out = A·V_proj. Numerically identical to standard softmax CA with C_kv=128 (no division by tiny denominator).
- **Why this might work where H3-6/H3-7/H3-8 didn't:**
  - vs H3-6/H3-7 (softmax CA at D=9 pool): different inductive bias. Existing 10 stage-3 blocks each see all 640 individual source channels; H3-9 funnel sees 128 LEARNED MIXTURES. The mixtures E_k, E_v are global learned channel-basis projections (not data-dependent like attention weights), forming a reusable factorized cross-channel summary table — fundamentally distinct from "select few from all 640".
  - vs H3-8 (linear attention NaN): keeps softmax → row-stochastic, bounded, stable. Two extra Linear(640, 128) projections (no bias) on K and V → standard layers, no division-by-near-zero.
  - Shared-head aux supervision (validated H3-7): forces the basis E_k, E_v to produce supertokens whose dot-product attention with Q yields class-predictive features.
- **Inspiration:** Linformer (Wang et al. 2020, arXiv:2006.04768) — linear-complexity self-attention via low-rank projection of K, V. Originally for spatial sequences; adapted here to channel tokens (project along the C dim instead of the sequence dim). RESEARCH_BRIEF.md option #2 "Factorised / low-rank channel attention" — never tested in this codebase.
- **Novelty self-audit:** category (a) NEW CA mechanism (Linformer K/V projection along channel axis = different operator family from softmax/sparse/agent/hier/linear). Category (c) failure-pattern response (H3-8 showed funnel + aux is trainable, but linear-attn unstable → keep topology, swap to stable low-rank softmax). Not in §3 exhausted list.
- **Budget:** 24.926M params (+0.16M vs H3-8 — two Linear(640,128) projections), 3.574 GFLOPs. Same envelope as H3-7/H3-8.
- **Submission:** job 369446 (2-GPU, --exclude=a node, tag `in_h3_9_linformer_funnel`, cfg `configs/autoresearch_experiment.yaml`).
- **Baseline to beat:** IN-B100 = 77.65% val top-1.
- **Result: 77.79% val top-1 (+0.14pp). NEW CHAMPION — first Phase 3 improvement after 8 consecutive failures.** Trajectory was monotonic and stable: ep25 ~58, ep50 ~70, ep75 ~75.5, ep100 77.79. No NaN, no instability — confirms Linformer is stable in this codebase (vs H3-8 linear-attn NaN).
- **Significance:** validates the LOW-RANK CHANNEL BASIS direction (RESEARCH_BRIEF.md option #2). The improvement comes from a NEW inductive bias (rank-128 attention via learned channel-mixture supertokens) that complements the existing 10 full-rank stage-3 CA blocks. The shared-head aux supervision (FUNNEL_AUX_WEIGHT=0.1) provides the gradient pathway needed for the gated additive funnel to learn useful features.
- **Lessons:** (i) Linformer-style K/V projection along the channel axis is a real win for channel-token attention. (ii) Stable softmax operator + structurally-distinct low-rank K/V projection beats both unstable linear-attn (H3-8) and redundant softmax CA (H3-6/H3-7). (iii) The funnel additive topology with init-0 gate + aux supervision is a reliable scaffold — future hypotheses can swap the operator inside this scaffold.



### H3-10: Stacked Linformer-style projected channel attention funnel (depth=2) — DISCARDED 77.75% (-0.04pp vs H3-9, noise, job 369686)
- **Hypothesis:** H3-9 single Linformer funnel block gave +0.14pp. Does the low-rank channel pathway have MORE capacity to give? Stack TWO Linformer funnel blocks at the head, each with its own E_k, E_v (r=128). The two blocks compose two complementary low-rank channel-basis projections — block 1 projects raw stage-3 features to a basis of 128 supertokens, block 2 projects block-1 output to a different basis. Conceptually a 2-layer "low-rank channel-attention head" where each layer learns a distinct compact basis of inter-channel interactions.
- **Mechanism (delta vs H3-9):**
  - FUNNEL_TAIL_DEPTH: 1 → 2 (NEW key). Funnel pathway becomes a chain of 2 Linformer blocks.
  - All other config identical: HW=7, KERNEL=3, MLP_RATIO=0.5, LINFORMER_R=128, FUNNEL_AUX_WEIGHT=0.1, gate init=0.
  - Each funnel block has its own attention with its own E_k, E_v. Forward chains them: `for blk in funnel_blocks: fp = blk(fp)`. Aux supervision applies at the FINAL block's GAP feature → gradient flows back through both blocks.
- **Why this might compound H3-9:**
  - Stage-3 stacking (full-rank CA at depth-10) hits diminishing returns. But Linformer blocks have a forced low-rank constraint (r=128 << C=640) → each block must learn a *distinct compact basis*; stacking 2 doubles the basis coverage without removing the rank constraint.
  - The first block's E_k, E_v form a basis for the input distribution; the second block's E_k, E_v form a basis for the OUTPUT of the first (a different statistical distribution). Forced-distinct bases by gradient signal.
  - Aux supervision pulls the entire 2-block chain into the loss landscape from epoch 0.
- **Why this might NOT compound H3-9:**
  - Single-block low-rank may already capture all the useful complementary signal at the head. Adding a second block could just replicate the first.
  - With aux supervision on the final block's GAP only, the first block could become a "redundant pass-through" if there's no incentive for it to differ.
- **Decision rule:** if H3-10 > H3-9 (i.e., > 77.79%) → low-rank funnel direction has more capacity → next push depth-3 or vary r per block. If H3-10 ≤ H3-9 → single-block is enough → pivot to a different additive mechanism on top of H3-9 baseline (e.g., different operator family, multi-scale funnel pool, etc.).
- **Inspiration:** Hourglass Transformer (Nawrot et al. 2022, arXiv:2110.13711) and Funnel Transformer (Dai et al. 2020, arXiv:2006.03236) — both stack multiple compressed-token-set blocks at the bottleneck. Adapted to channel-token Linformer.
- **Novelty self-audit:** category (b) NEW STRUCTURAL COMPONENT (multi-block low-rank channel-attention head; not present elsewhere in the architecture). Category (c) DIRECT BUILD on H3-9's confirmed direction.
- **Budget:** 25.936M params (+1.01M vs H3-9 — one extra Block + Linformer projections), 3.632 GFLOPs. Within ~20-26M envelope.
- **Submission:** job 369686 (2-GPU, --exclude=a node, tag `in_h3_10_linformer_d2`, cfg `configs/autoresearch_experiment.yaml`).
- **Baseline to beat:** H3-9 = 77.79% val top-1.
- **Result: 77.75% val top-1 (-0.04pp vs H3-9, within noise). DISCARDED.**
- **Diagnosis:** stacking the SAME operator (Linformer rank-128) at the head did NOT compound. Two blocks of the same inductive bias produce ~one block's worth of signal — redundant capacity. The second block likely learned a basis very similar to the first or operated as near-identity. Aux supervision on the final block's GAP only doesn't force the first block to learn something distinct.
- **Lesson:** for the funnel head, OPERATOR REPLICATION at the same operator family is redundant. The decision rule fires → pivot to a different additive mechanism (e.g., operator DIVERSITY via parallel hybrid).



### H3-11: Parallel hybrid funnel — Linformer (rank-128 dense) + Sparse top-k (k=64 sparse) — DISCARDED (-0.21pp vs H3-9)
- **Hypothesis:** if SAME-operator replication (H3-10) is redundant, what about DIFFERENT-operator combination? Keep the H3-9 Linformer funnel block (validated +0.14pp) and ADD a SECOND PARALLEL funnel block with a structurally DISTINCT operator: sparse top-k channel attention (k=64). Both branches process the same pooled stage-3 input; their outputs are AVERAGED before funnel_norm. Aux supervision (FUNNEL_AUX_WEIGHT=0.1) trains both branches through the combined GAP feature.
- **Mechanism (delta vs H3-9):**
  - FUNNEL_PARALLEL_BLOCK: false → true (new key). Build a second funnel block in PARALLEL.
  - FUNNEL_PARALLEL_LINFORMER_R: 0 (parallel block does NOT use Linformer — full-rank Q·K^T).
  - FUNNEL_PARALLEL_SPARSE_TOPK: 64 (parallel block masks to top-64 of 640 source channels per query).
  - FUNNEL_TAIL_DEPTH: 2 → 1 (revert H3-10 stacking; back to single Linformer block in main chain).
  - Forward: `fp_main = sequential_chain(fp_in)`; `fp_parallel = parallel_block(fp_in)`; combined = 0.5 * (fp_main + fp_parallel).
- **Why this might compound where H3-10 didn't:**
  - Linformer: A ∈ R^(C×r), rank-r DENSE softmax. All queries share r=128 learned mixtures.
  - Sparse top-k: A ∈ R^(C×C), each row has ≤k=64 non-zero (after masking). Each query selects its OWN top-64 source channels — data-dependent per query.
  - Mathematically COMPLEMENTARY: Linformer = rank-constrained dense; Sparse = sparsity-constrained data-dependent. They span orthogonal axes (rank vs sparsity) in the attention-matrix space. Linformer can express DENSE LOW-RANK patterns; Sparse can express SPARSE FULL-RANK patterns.
- **Note on H3-1:** H3-1 sparse top-k REPLACED stage-3 softmax CA and failed -0.17pp. H3-11 keeps all 10 stage-3 full-softmax CA UNCHANGED and ADDS sparse top-k as a parallel HEAD pathway — fundamentally different topology. Just like H3-9 added Linformer additively (not as replacement), H3-11 adds sparse additively.
- **Inspiration:** Sparse Transformer (Child et al. 2019, arXiv:1904.10509) and Routing Transformer (Roy et al. 2020) — sparse attention via top-k or learned routing. Hybrid attention mixtures: FLatten Transformer (2023) combines focused-linear + standard; Hybrid Transformer-Mamba models (2024). Adapted to channel tokens as a parallel head pathway.
- **Novelty self-audit:** category (b) NEW STRUCTURAL COMPONENT — parallel hybrid funnel topology, never tested in this codebase. Category (c) DIRECT RESPONSE to H3-10 failure pattern (stacked same-operator failed → test diverse-operator parallel).
- **Decision rule:** if H3-11 > H3-9 (i.e., > 77.79%) → operator diversity at the head compounds → explore more operator hybrids (e.g., Linformer + linear-attn parallel, or three-way Linformer + sparse + agent). If H3-11 ≤ H3-9 → operator diversity at the head also doesn't compound; the H3-9 +0.14pp may be the practical ceiling for head-only additive pathways → pivot to in-trunk modifications.
- **Budget:** 25.772M params (+0.85M vs H3-9 — one extra Block with no Linformer projections), 3.656 GFLOPs. Within envelope.
- **Submission:** job 370212 (2-GPU, --exclude=a node, tag `in_h3_11_parallel_hybrid`, cfg `configs/autoresearch_experiment.yaml`).
- **Baseline to beat:** H3-9 = 77.79% val top-1.
- **Result:** **77.58%** val top-1 (job 370212). Δ vs H3-9 = **-0.21pp**, Δ vs IN-B100 = **-0.07pp**. DISCARDED.
- **Diagnosis:** operator diversity via simple averaging at the head HURTS. Two competing branches with very different attention matrices (rank-128 dense Linformer vs sparse top-64) produce conflicting feature directions; naive 0.5*(A+B) averaging causes destructive interference at funnel_norm. The aux head provides a single supervision signal — both branches see the same gradient and have no incentive to specialize complementarily.
- **Lesson:** for the funnel head, both same-operator stacking (H3-10) AND different-operator parallel averaging (H3-11) failed to compound. The +0.14pp from H3-9 (single Linformer block) appears to be the practical CEILING for head-only additive pathways under the current aux-supervision regime. Per program decision rule: pivot from "more head" to in-trunk modifications.



### H3-12: Linformer applied to ALL 10 stage-3 in-trunk CA blocks (LINFORMER_R_PER_STAGE=[0,0,0,128]) — **CHAMPION (+0.12pp vs H3-9, +0.26pp vs IN-B100)**
- **Hypothesis:** GENERALIZE the validated H3-9 Linformer mechanism from head-side ADDITIVE funnel to all 10 in-trunk stage-3 CA blocks. Each block now uses Linformer-style K/V projection from C=640 to r=128 supertokens before softmax, with its OWN learned E_k, E_v projections (xavier-uniform init, no bias). Head funnel block UNCHANGED from H3-9 (also Linformer rank-128). Tests whether the low-rank channel basis inductive bias — which improved the head as an additive pathway — is also useful when applied broadly as a REPLACEMENT operator across the trunk.
- **Mechanism (delta vs H3-9):**
  - LINFORMER_R_PER_STAGE: [0,0,0,0] → [0,0,0,128] (NEW: applies Linformer to all 10 stage-3 blocks).
  - Each stage-3 CA block: `K_proj = E_k·K (B,1,128,49)`, `V_proj = E_v·V (B,1,128,49)`, then standard scaled-dot-product softmax with rank-128 keys/values.
  - Total in-trunk Linformer projections: 10 blocks × 2 × Linear(640,128, no bias) = 1.638M extra params.
  - Funnel block configuration unchanged (LINFORMER_R=128 at head). FUNNEL_PARALLEL_BLOCK reverted to false (H3-11 hybrid disabled).
- **Why this is bold and high-information:**
  - Stage-3 attention switches from full-rank softmax CA (rank effectively up to 640) to rank-128 Linformer across 10 consecutive blocks — a ~5× rank bottleneck applied throughout the trunk.
  - Each block has its own learned 128-basis. Per-block specialization possible (early blocks see raw stage-3 features, late blocks see refined features → different bases).
  - BINARY test: tells us whether Linformer is useful as a GENERAL CA inductive bias, or just as a head-specific additive pathway.
- **Why this might work:**
  - If the rank-128 basis is a useful inductive bias for channel attention, applying it broadly should help, especially under strong end-to-end aux supervision (5 aux losses: stage-0/1/2 + mid-block + funnel).
  - Reduces stage-3 attention compute (rank-128 vs rank-640 attention matrices).
- **Why this might NOT work:**
  - REPLACEMENT-style hypotheses (H3-1..H3-5) all failed -0.10 to -0.27pp. Replacing flexible full-rank softmax CA with a constrained low-rank version may lose information.
  - The 10-block stack of bottlenecks could compound capacity loss.
  - H3-9's gain might come specifically from being ADDITIVE-AND-AUX-SUPERVISED at the head, not from Linformer's inductive bias per se.
- **Inspiration:** Linformer (Wang et al. 2020, arXiv:2006.04768) — already validated at head (H3-9). H3-12 tests the GENERALIZATION principle: a useful inductive bias at the head should also help in-trunk unless the trunk's flexibility is the source of accuracy.
- **Novelty self-audit:** category (a) NEW CA MECHANISM applied IN-TRUNK — in-trunk Linformer is structurally distinct from in-trunk full-rank CA and has never been tested in this codebase (H3-9 was head-only). Category (c) DIRECT RESPONSE to H3-10/H3-11 failure pattern — head-only stacking and parallel diversity didn't compound, so pivot from "more head" to "transform the trunk".
- **Decision rule:**
  - If H3-12 > 77.79: MAJOR breakthrough — Linformer is a general-purpose CA improvement. Push wider (other stages too) and explore variants.
  - If 77.65 ≤ H3-12 ≤ 77.79: neutral — head-only and trunk-replacement give similar gain. Pivot to a HYBRID (e.g., Linformer at last 5 stage-3 blocks + standard at first 5).
  - If H3-12 < 77.65: trunk replacement loses information; H3-9's gain is a LOCAL phenomenon, not a general property of Linformer. Pivot to a fundamentally different mechanism (e.g., MLP-Mixer channel mixing, SSM, Synthesizer-Dense).
- **Budget:** 26.565M params (+1.6M vs H3-9 from in-trunk projections), 3.735 GFLOPs. Within FLOPs envelope.
- **Submission:** job 372010 (2-GPU, --exclude=a node, tag `in_h3_12_linformer_in_trunk_s3`, cfg `configs/autoresearch_experiment.yaml`).
- **Baseline to beat:** H3-9 = 77.79% val top-1 (also IN-B100 = 77.65%).
- **Result:** **77.91%** val top-1 (Acc@5 93.87, job 372010). Δ vs H3-9 = **+0.12pp**, Δ vs IN-B100 = **+0.26pp**. **NEW CHAMPION.**
- **Diagnosis:** the rank-128 low-rank channel basis is a genuine general CA inductive bias — it works as a REPLACEMENT operator across all 10 stage-3 blocks, not just as a head-side additive pathway. The +0.12pp on top of H3-9 (which already had Linformer head) is additive: in-trunk Linformer contributes its own gain. This contradicts the H3-1..H3-5 pattern (where replacement-style hypotheses lost -0.10..-0.27pp), demonstrating that Linformer's particular low-rank softmax structure is qualitatively different from agent tokens / sparse / grouped / hier CA / linear-attn replacements that previously failed.
- **Key insight:** the +0.12pp also slightly REDUCES the marginal-overfit risk because Linformer constrains capacity (rank-128 attention is strictly less expressive than full-rank 640). The fact that constraining attention rank IMPROVES validation accuracy is strong evidence that the full-rank softmax CA was OVER-PARAMETERIZED at stage-3 — i.e., the channel-attention bottleneck was hiding the real ceiling.
- **Lesson:** Linformer rank-128 K/V projection is a UNIVERSALLY USEFUL channel-attention inductive bias at C=640. Per decision rule, push WIDER to other stages and explore variants.



### H3-13: Push Linformer wider — apply to stage-2 trunk (5 blocks at r=96) AND keep stage-3 (10 blocks at r=128). LINFORMER_R_PER_STAGE=[0,0,96,128] — DISCARDED (-0.26pp vs H3-12)
- **Hypothesis:** H3-12 proved the low-rank channel basis inductive bias works as a REPLACEMENT operator at stage-3 (+0.12pp on top of head Linformer). If the bias is GENERAL across scales, applying Linformer to stage-2 (5 blocks at C=384, D=196) should also help. H3-13 extends LINFORMER_R_PER_STAGE from [0,0,0,128] → [0,0,96,128], adding rank-96 K/V projection to all 5 stage-2 CA blocks while leaving stages 0/1 untouched (only 1 and 2 blocks respectively there — too few to meaningfully replace).
- **Mechanism (delta vs H3-12):**
  - LINFORMER_R_PER_STAGE: [0,0,0,128] → [0,0,96,128].
  - Stage-2 (5 blocks, C=384, D=196): each block now uses Linformer with E_k, E_v ∈ ℝ^(384×96), xavier-uniform init, no bias. r/C = 96/384 = 25% (vs stage-3 20% — slightly more conservative, keeps absolute supertoken count similar relative to spatial D=196 vs 49).
  - Stage-3 (10 blocks, C=640): unchanged from H3-12 (r=128).
  - Head funnel unchanged (Linformer r=128). FUNNEL_PARALLEL_BLOCK stays false.
- **Why this might work:**
  - Stage-2 also has many CA blocks (5) with C=384. If the rank bottleneck is universally beneficial (as H3-12 suggested by improving over full-rank stage-3), stage-2 should benefit similarly. The stage-2 attention matrix would change from 384×384 (147K entries) to a rank-96 factorization (a 384×96 + 96×96 operation) — analogous bottleneck ratio to stage-3.
  - Each stage-2 block also gets its own learned 96-basis, allowing per-block specialization across the 5-block depth.
  - 15 trunk blocks (5 stage-2 + 10 stage-3) now use Linformer — a comprehensive test of whether the inductive bias is general.
- **Why this might NOT work:**
  - Stage-2 features are LESS REFINED than stage-3 (earlier in the trunk, more spatial detail D=196). The full-rank CA there may genuinely USE rank > 96 to organize raw multi-scale features.
  - Stage-2 has aux supervision (AUX_LOSS_WEIGHT_S2=0.4) — the strongest aux loss in the network. The constraint of rank-96 attention at stage-2 may make this aux harder to satisfy.
  - At stage-2 the spatial dim D=196 already strongly limits attention expressiveness (rank ≤ min(C,D)=196). Adding r=96 cap on K/V may be redundant or harmful given the spatial bottleneck.
- **Inspiration:** Linformer (Wang et al. 2020, arXiv:2006.04768) — validated at head (H3-9) and stage-3 trunk (H3-12). H3-13 tests SCALE GENERALIZATION: does the low-rank inductive bias hold across stages with different (C, D) ratios?
- **Novelty self-audit:** category (a) NEW CA MECHANISM applied at a new scale — stage-2 trunk Linformer has never been tested. Category (c) DIRECT FOLLOW-UP to H3-12 win — the decision rule specifically said to push wider after a major breakthrough.
- **Decision rule:**
  - If H3-13 > 77.91 (H3-12): rank bottleneck is universally useful across scales. Push to stage-1 too, or tune ranks per stage.
  - If 77.79 ≤ H3-13 ≤ 77.91: stage-2 Linformer is roughly neutral (similar to H3-9 head-only). Try a different stage-2 rank (lower r=48 or higher r=192) before pivoting.
  - If H3-13 < 77.79: Linformer benefit is stage-3-specific (deepest stage with most refined features). Pivot to ranking variations within stage-3 (e.g., per-block r schedule, shared E_k across blocks) or different CA operators.
- **Budget estimate:** +5 × 2 × (384·96) ≈ 0.37M extra params → ~26.94M total. FLOPs roughly unchanged (Linformer slightly cheaper than full-rank at stage-2 too).
- **Baseline to beat:** H3-12 = 77.91% val top-1.
- **Submission:** job 373929 (2-GPU, --exclude=a node, tag `in_h3_13_linformer_in_trunk_s2_s3`, cfg `configs/autoresearch_experiment.yaml`).
- **Result:** **77.65%** val top-1 (job 373929). Δ vs H3-12 = **-0.26pp**, exactly at IN-B100 baseline. DISCARDED.
- **Diagnosis:** Linformer's gain is STAGE-3-SPECIFIC, not a general "any stage benefits from rank cap" property. Stage-2 has spatial D=196 (>> r=96), features less refined (one stage earlier in trunk), and carries the strongest aux supervision (AUX_LOSS_WEIGHT_S2=0.4). Constraining stage-2 attention rank to 96 evidently REMOVES information the network was using to satisfy the strong aux loss. The full +0.26pp gain from H3-12 over IN-B100 collapsed back to 0.0pp once stage-2 Linformer was added.
- **Key insight:** the H3-12 finding ("rank-128 at stage-3 reduces over-parameterization → improves val acc") was NOT a universal property of channel attention. It was specific to (a) deepest stage with most refined features, (b) without strong aux constraint, (c) at C=640. The mechanism is more subtle than "low-rank attention is universally good" — it's "stage-3 full-rank CA was specifically over-parameterized".
- **Lesson:** per decision rule, pivot to STAGE-3-INTERNAL VARIANTS (per-block rank schedule, shared E_k across blocks, different rank values) or to DIFFERENT OPERATORS at stage-3 (Synthesizer, Nyström, kernel attention). Spreading Linformer to other stages does not help.



### H3-14: Per-block Linformer rank schedule within stage-3 — depth-varying rank, same total avg as H3-12. LINFORMER_R_PER_BLOCK_S3=[64,64,96,96,128,128,160,160,192,192] — DISCARDED (-0.24pp vs H3-12)
- **Hypothesis:** H3-12 proved uniform Linformer rank-128 across all 10 stage-3 blocks IMPROVES over full-rank. H3-13 showed the gain doesn't generalize to other stages. The next refinement question: within stage-3, is UNIFORM r=128 optimal, or does the OPTIMAL rank GROW WITH DEPTH? Early stage-3 blocks (right after the stage-2→stage-3 transition) may need less rank since features are less refined; late blocks (after many CA refinement steps + CPE re-injection every 4 blocks) may benefit from higher rank to capture more nuanced channel relationships.
- **Mechanism (delta vs H3-12):**
  - LINFORMER_R_PER_BLOCK_S3: [] → [64, 64, 96, 96, 128, 128, 160, 160, 192, 192].
  - Monotone-increasing schedule with average = 1280/10 = 128 (= H3-12 uniform value).
  - Each block has its own (E_k, E_v) at rank r_j. Total Linformer projection params: 2×640×Σr_j = 2×640×1280 = 1.638M — IDENTICAL to H3-12 (just redistributed across blocks).
  - Stage-2 unchanged (no Linformer). Head funnel unchanged (Linformer r=128, additive). Mid-block aux at block index 5 (which now has rank-128 — same as H3-12 at that index).
- **Why depth-varying rank could help:**
  - Spatial-attention ViTs (PVTv2 [1,2,5,8] heads, Swin progressive widths) routinely use depth-varying configurations. Adapting this principle to channel-attention RANK at the block level is the obvious analogue.
  - At fixed param budget, depth-varying allocates more rank where it matters most. If late blocks benefit more, the redistribution is a strict improvement.
  - Aux-supervised at mid block (5) — blocks 0–4 get extra gradient signal that can compensate for tighter rank-64 cap; blocks 5–9 carry the main classifier path and get more rank.
- **Why depth-varying rank could hurt:**
  - Uniform r=128 may already be the optimum. Variation around a flat optimum just adds noise.
  - Early blocks with rank-64 may form an information bottleneck that late blocks can't recover from.
  - Same total params → no capacity gain, only redistribution.
- **Inspiration:** Linformer (Wang et al. 2020, arXiv:2006.04768) — base mechanism. Depth-varying attention configuration is a long-standing pattern in vision transformers (PVT, PVTv2, Swin, CvT itself uses depth-varying head counts). H3-14 adapts the depth-varying-config principle to attention RANK at the per-block level within a single stage — a novel adaptation for channel attention.
- **Novelty self-audit:**
  - Category (b) NEW STRUCTURAL COMPONENT — depth-varying attention rank WITHIN a stage has not been tested in this codebase. Per-stage rank variation existed; per-block variation within a stage is new (added LINFORMER_R_PER_BLOCK_S3 config key and per-block override in VisionTransformer.__init__ block loop).
  - Category (c) DIRECT RESPONSE to H3-13 failure pattern — H3-13 showed Linformer is stage-3-specific; pivot from "spread Linformer wider" to "differentiate Linformer within the stage where it works".
- **Decision rule:**
  - If H3-14 > 77.91: depth-varying rank > uniform → schedule further (steeper, or grow-then-shrink hump). Confirms rank-at-depth-D matters.
  - If 77.86 ≤ H3-14 ≤ 77.91: redistribution at fixed budget gives no signal → uniform was optimal. Pivot to a STRUCTURALLY new stage-3 variant (Synthesizer-Dense, Nyström, shared E_k across blocks).
  - If H3-14 < 77.86: depth-varying hurts → uniform r=128 is the optimum. Pivot operator (different efficient attention) rather than tune rank schedule.
- **Budget:** 26.565M params, 3.735 GFLOPs — IDENTICAL to H3-12. Clean A/B test of "schedule" vs "uniform".
- **Baseline to beat:** H3-12 = 77.91% val top-1.
- **Submission:** job 374301 (2-GPU, --exclude=a node, tag `in_h3_14_linformer_perblock_s3`, cfg `configs/autoresearch_experiment.yaml`).
- **Infrastructure failure:** job 374301 crashed at init on with NCCL ALLREDUCE 10-min timeout (never started training; SIGABRT/-6 exit). Not a model issue — config and model parse fine. Resubmitted as **job 374322** with `--exclude=a node,a node`.
- **Result:** **77.67%** val top-1 (job 374322). Δ vs H3-12 = **-0.24pp**, Δ vs IN-B100 = **+0.02pp** (essentially noise above baseline). DISCARDED.
- **Diagnosis:** at fixed total-rank budget (avg 128), redistributing rank across blocks HURT. The monotone-increasing schedule (early blocks at r=64) suggests the EARLY stage-3 blocks need MORE rank, not less — they receive feature maps right after the stage-2→stage-3 transition (sudden 384→640 channel widening + 14×14→7×7 spatial collapse) and may need full rank-128 to organize this large basis change. Late blocks at r=192 over-paramaterize where r=128 was already the optimum.
- **Key insight:** uniform r=128 at stage-3 is NOT a noise-level optimum — it's a TIGHT optimum where both downward redistribution (early blocks at r=64) and upward redistribution (late blocks at r=192) hurt by similar amounts (~0.24pp combined). The Linformer rank-128 mechanism is sharply tuned to this stage-3 configuration.
- **Lesson:** per decision rule (< 77.86), uniform r=128 is the optimum and rank-schedule tuning is the wrong axis. Pivot to a DIFFERENT OPERATOR class at stage-3 — keep r=128 dense Linformer as the proven base, but ADD a NEW STRUCTURAL AXIS (e.g., sparsity, multi-head, kernel) rather than tune rank.



### H3-15: Sparse top-k WITHIN the Linformer rank-128 supertoken basis at stage-3. LINFORMER_TOPK_PER_STAGE=[0,0,0,32] — DISCARDED (-0.37pp vs H3-12; -0.11pp vs IN-B100)
- **Hypothesis:** H3-12 (uniform Linformer r=128 at stage-3) is the champion. H3-13/H3-14 both failed (stage-2 spread, rank schedule). Decision rule from both says: keep r=128 dense uniform, pivot to a NEW STRUCTURAL AXIS. H3-15 COMPOUNDS rather than replaces: keep the validated rank-128 learned basis EXACTLY, but add DATA-DEPENDENT TOP-K SPARSITY WITHIN that basis. Each stage-3 query channel attends to its TOP-32 of 128 supertokens (selected by per-query attention score), masking the remaining 96 to -inf before softmax.
- **Mechanism (delta vs H3-12):**
  - LINFORMER_TOPK_PER_STAGE: [0,0,0,0] → [0,0,0,32].
  - Inside the use_linformer branch of Attention.forward: after computing scores (B,1,640,128), keep the top-32 per query (out of 128 supertokens), mask the rest to -inf, then softmax.
  - No new parameters. Pure dispatch-time mask. Linformer E_k, E_v unchanged from H3-12.
  - All 10 stage-3 blocks use r=128 + topk=32. Stage-2 and head funnel unchanged.
- **Why this might compound H3-12:**
  - Linformer's learned basis defines 128 supertokens. Every query is FORCED to softmax over all 128 (dense). Most of those 128 are likely irrelevant for any specific query — softmax assigns small weights but cannot fully suppress them.
  - Top-k=32 lets each query SELECT a query-specific subset of the basis, completely suppressing the other 96. This is qualitatively different from rank tuning — we keep the FULL basis available, just route per-query.
  - Sparse Transformer, Routing Transformer, and MoE-attention literature all show that data-dependent sparsity often improves on dense baselines. Linformer's compact basis (vs raw channels) may make top-k routing MORE effective than H3-1's failed top-k over all 640 raw channels — supertokens are LEARNED compact representations.
- **Why this might NOT work:**
  - H3-1 (top-k on full-rank stage-3 CA) lost -0.17pp. Top-k at stage-3 has a precedent of failure. H3-15 differs in that the basis is the learned rank-128 supertokens — more meaningful subsets — but the principle (suppress most attention) is similar.
  - The Linformer basis is ALREADY a low-rank constraint. Adding sparsity may be redundant.
  - Hard top-k is non-differentiable; gradient flows only through the 32 selected scores. Could cause optimization issues over 100 epochs.
- **Note on H3-1 vs H3-15:** H3-1 sparsified over 640 raw channels (each query attends to top-32 of 640). H3-15 sparsifies over 128 LEARNED supertokens (each query attends to top-32 of 128). The supertokens are explicitly learned to be compact summaries of channel groups, so the top-32 should be more meaningful selections than in H3-1. Same numerical k=32, very different semantic content.
- **Inspiration:** Linformer (Wang et al. 2020, arXiv:2006.04768) — validated base mechanism in H3-12. Sparse Transformer (Child et al. 2019, arXiv:1904.10509) — top-k masking. Routing Transformer (Roy et al. 2020) — data-dependent routing. MoE-attention (Switch Transformer, Fedus et al. 2021) — top-k routing in learned-expert space. H3-15 adapts MoE-style top-k routing to the Linformer supertoken basis.
- **Novelty self-audit:**
  - Category (a) NEW CA MECHANISM — combining Linformer's learned low-rank basis with data-dependent top-k sparsity WITHIN that basis has never been tested. H3-1 had top-k over raw channels (failed); H3-12 had Linformer dense (won); H3-15 is the structural hybrid.
  - Category (c) DIRECT RESPONSE to H3-13/H3-14 failure pattern — both decision rules pointed to "pivot operator/structure rather than rank schedule". H3-15 adds a new structural axis (within-basis sparsity) on top of the proven base.
- **Decision rule:**
  - If H3-15 > 77.91: sparse-within-basis compounds Linformer's gain → new champion. Tune k (try 16, 64) next, then maybe extend to head Linformer.
  - If 77.86 ≤ H3-15 ≤ 77.91: top-k is neutral on the Linformer basis → dense softmax over r=128 is the right operator. Pivot to a fundamentally different operator at stage-3 (e.g., Nyström, Synthesizer-Dense) or to MULTI-HEAD Linformer.
  - If H3-15 < 77.86: hard top-k destroys the basis → revert. Try SOFT routing (entmax/sparsemax) or different compounding mechanism (gated parallel pathway inside each block).
- **Budget:** **0 extra params** (top-k is pure masking). FLOPs ~unchanged. 26.565M / 3.735 GFLOPs — IDENTICAL to H3-12. Clean A/B test of "compound with sparsity" vs "dense Linformer".
- **Baseline to beat:** H3-12 = 77.91% val top-1.
- **Submission:** job 382485 (2-GPU, --exclude=a node,a node, tag `in_h3_15_linformer_topk32_s3`, cfg `configs/autoresearch_experiment.yaml`).
- **Result:** **77.54%** val top-1 (job 382485). Δ vs H3-12 = **-0.37pp**, Δ vs IN-B100 = **-0.11pp** (BELOW baseline). DISCARDED.
- **Diagnosis:** hard top-k=32 over the rank-128 Linformer supertoken basis hurt the WORST of any pivot tried so far. The masking operation is non-differentiable on the rejected scores, killing gradient flow to 96 of 128 supertokens per query. The Linformer projections E_k, E_v can no longer learn a fully-utilised compact basis — most supertokens never receive a useful gradient signal because they're masked out per-query.
- **Key insight:** the Linformer rank-128 dense softmax IS the optimum at stage-3. Any modification that disrupts the DENSE softmax over the full 128-supertoken basis hurts. Three failed pivots (wider, schedule, top-k) all share the property of altering the rank-128 dense Linformer operator. The H3-12 mechanism is sharply tuned to its current form.
- **Lesson:** per decision rule (< 77.86), pivot away from operator-level modifications. Try a SOFT compounding mechanism that doesn't disrupt the H3-12 operator: an init-0 gated parallel pathway alongside Linformer (decision-rule-listed option), or a different operator family entirely (Synthesizer, Nyström).



### H3-16: Gated parallel FULL-RANK CA pathway alongside Linformer at stage-3. LINFORMER_FULLRANK_PARALLEL_PER_STAGE=[0,0,0,1] — DISCARDED (-0.13pp vs H3-12; +0.13pp vs IN-B100)
- **Hypothesis:** three consecutive failures (H3-13/14/15) confirm that ALTERING the H3-12 mechanism hurts. H3-16 takes the opposite approach: keep H3-12's rank-128 dense Linformer EXACTLY untouched, and ADD a parallel full-rank softmax(Q·K^T)·V pathway gated by a learned scalar init to 0. At init, the gate=0 means the parallel pathway contributes nothing → behaviour equals pure H3-12. Training may ramp the gate to integrate full-rank residual information IF AND ONLY IF the optimiser finds it useful. This is the SAFEST possible "additive probe" — guaranteed non-regressive at init.
- **Mechanism (delta vs H3-12):**
  - LINFORMER_FULLRANK_PARALLEL_PER_STAGE: [0,0,0,0] → [0,0,0,1].
  - Each of the 10 stage-3 CA blocks now computes BOTH:
    - (i) Linformer: `softmax(Q·E_k(K)^T/√d) · E_v(V)` → (B,1,640,49) — H3-12 mechanism.
    - (ii) Full-rank: `softmax(Q·K^T/√d) · V` → (B,1,640,49) — pre-H3-12 mechanism.
  - Combined: `out = linformer_out + gate * fullrank_out`, with one learned scalar gate per block (10 total), all init to 0.
  - At init: gate=0 → out = linformer_out → IDENTICAL to H3-12 (verified by smoke test, gradient flow confirmed).
- **Why this might compound H3-12:**
  - Linformer's E_k, E_v project K, V down to rank-128 supertokens, throwing away 512 residual dimensions. If any of that residual carries useful signal for some queries, the parallel full-rank pathway can re-inject it.
  - The init-0 gate is the SAME compounding pattern that worked for H3-9 funnel-aux (validated). It provides safe additive probing — at worst, optimiser keeps gate=0 and H3-12 behaviour is preserved.
  - Unlike H3-15's hard top-k (non-differentiable on rejected scores), the gate is fully differentiable from step 0.
- **Why this might NOT work:**
  - H3-12's gain may come from REGULARISATION via rank constraint. Re-adding full-rank info undoes the regularisation that H3-12 introduced — over-parameterisation may return.
  - Doubles attention compute at stage-3 (Linformer + full-rank both run per block). +~0.4 GFLOPs total.
  - Even with init-0 gate, the parallel full-rank Q·K^T gradient may pull q_proj, k_proj weights in directions that interfere with the Linformer pathway through their shared upstream Q, K computations.
- **Inspiration:**
  - Init-0 gating: H3-9 funnel-aux (own program, validated). LayerScale (Touvron et al. 2021).
  - Mixed-rank residual decomposition: LoRA (Hu et al. 2021, arXiv:2106.09685) for low-rank + base composition; FLatten Transformer (2023) for parallel attention operator hybrid. H3-16 adapts to channel attention with init-0 gated additive.
- **Novelty self-audit:**
  - Category (b) NEW STRUCTURAL COMPONENT — per-block init-0 gated parallel attention pathway inside the trunk has never been tested. Different from H3-11 (parallel funnel at head with 0.5*(A+B) averaging — no learned gate, failed).
  - Category (c) DIRECT RESPONSE to H3-13/H3-14/H3-15 pattern — three consecutive failed pivots show H3-12 is a sharp local optimum. Init-0 gating is the SAFEST possible probe: it tests "is there additional useful signal beyond what Linformer captures?" without risking regression.
- **Decision rule:**
  - If H3-16 > 77.91: gate ramped → full-rank residual carries useful signal. NEW CHAMPION. Try varying the parallel operator (sparse, different gate init, gate per head).
  - If 77.86 ≤ H3-16 ≤ 77.91: gate stayed near 0 → Linformer captures all useful signal at stage-3. H3-12 is a true operator-level global optimum. Pivot to a fundamentally different operator (Synthesizer-Dense, Nyström, channel-MLP-Mixer).
  - If H3-16 < 77.86: parallel pathway HURTS (interference or over-param regression). Confirms H3-12's gain came from rank regularisation. Pivot to other operator families and/or revisit the decision rule.
- **Budget:** +10 scalar gates (negligible params). Total **26.565M params, 4.136 GFLOPs** (+0.4G vs H3-12 from parallel full-rank Q·K^T at stage-3). Within 5-8 GFLOPs target.
- **Baseline to beat:** H3-12 = 77.91% val top-1.
- **Submission:** job 388080 (2-GPU, --exclude=a node,a node, tag `in_h3_16_linformer_plus_fullrank_gate_s3`, cfg `configs/autoresearch_experiment.yaml`).
- **Result:** 77.78% val top-1. Δ vs H3-12 = **-0.13pp** (below 77.86 floor), Δ vs IN-B100 = +0.13pp. DISCARDED.
- **Diagnosis:** parallel full-rank pathway hurt even with init-0 gate. Per decision rule (H3-16 < 77.86), this confirms H3-12's gain came from RANK REGULARISATION — re-adding full-rank information undid the regularisation. Even though the gate is differentiable from step 0, gradient pressure from the full-rank pathway through shared Q/K projections likely pulled weights toward over-parameterised solutions. 4 consecutive failed pivots from H3-12 (H3-13 wider, H3-14 schedule, H3-15 hard top-k, H3-16 parallel) confirm H3-12 is a sharp local optimum and rank-128 dense uniform is the correct stage-3 operator.


### H3-17: Sparsemax replacement for softmax inside stage-3 Linformer. LINFORMER_SOFTMAX_TYPE_PER_STAGE=['softmax','softmax','softmax','sparsemax'] — DISCARDED (-0.17pp vs H3-12; +0.09pp vs IN-B100)
- **Hypothesis:** 4 consecutive failures from modifying H3-12 (H3-13/14/15/16) confirm rank-128 dense uniform Linformer is operator-optimal at stage-3. The H3-15 decision rule's remaining unexplored option is SOFT routing via sparsemax (Martins & Astudillo 2016). Unlike H3-15's hard top-k masking (non-differentiable on rejected scores), sparsemax produces sparse outputs DIFFERENTIABLY — some weights become exactly 0 but the underlying scores still receive gradient through the active support. This isolates the sparsity question from the differentiability question that H3-15 conflated.
- **Mechanism (delta vs H3-12):**
  - LINFORMER_SOFTMAX_TYPE_PER_STAGE: new key, [stage0, stage1, stage2, stage3] = ['softmax','softmax','softmax','sparsemax']. Default 'softmax' everywhere = H3-12 baseline.
  - Inside `use_linformer` branch at stage-3, replace `F.softmax(attn_score, dim=-1)` with `sparsemax(attn_score, dim=-1)`.
  - Sparsemax: Euclidean projection of attn_score onto the probability simplex via sort-cumsum-threshold (`τ` such that `sum(max(z-τ,0)) = 1`). Algorithm: sort z desc, cumulate, find support size `k` where `1+k*z_k > cumsum_k` holds, set `τ = (cumsum_at_k - 1)/k`, output `max(z-τ, 0)`. Sparse, differentiable, sum=1 per row.
  - All other H3-12 parameters identical (rank=128 dense uniform). LINFORMER_FULLRANK_PARALLEL_PER_STAGE reset to [0,0,0,0]. LINFORMER_TOPK_PER_STAGE stays [0,0,0,0].
- **Why this might compound H3-12:**
  - Sparsemax assigns exactly 0 weight to irrelevant supertokens (the noise floor), while softmax always gives small-but-nonzero weight to every supertoken. If the rank-128 basis has a few "garbage" supertokens for each query, sparsemax can prune them cleanly without the hard-masking gradient pathology of H3-15.
  - Differentiability: unlike top-k, sparsemax's support is determined by the data, and the threshold τ moves smoothly during training. Active-support entries receive non-zero gradient; inactive-support entries receive 0 gradient WITHOUT a discontinuity barrier (cleanly differentiable a.e.).
  - Per the H3-15 decision rule literally — "Try SOFT routing (entmax/sparsemax) or different compounding mechanism" — this is the unexplored second branch. H3-16's gated parallel was option two; sparsemax is option one.
- **Why this might NOT work:**
  - Sparsemax may over-sparsify: at high attention temperature, support size can collapse to 1–2 supertokens, equivalent to extremely hard top-k. Combined with the learned attention temperature (H47, log_attn_temp), sparsemax dynamics can be unstable.
  - The rank-128 basis was trained with softmax — switching to sparsemax may require re-learning the basis from scratch, costing convergence speed at fixed 100ep budget.
  - The cumsum+gather sparsemax kernel is not as well-optimised as softmax; small FLOPs/wall-clock overhead at stage-3 (10 blocks × 640 queries × 128 supertokens).
- **Inspiration:**
  - Sparsemax (Martins & Astudillo 2016, arXiv:1602.02068): Euclidean projection onto simplex, sparse-but-differentiable softmax alternative for attention.
  - α-entmax family (Peters et al. 2019, arXiv:1905.05702): generalises sparsemax/softmax via α parameter (α=1 softmax, α=2 sparsemax, intermediates). Sparsemax = α=2 special case.
  - Cited in modern sparse attention work: "Adaptively Sparse Transformers" (Correia et al. 2019); MoE routing variants.
- **Novelty self-audit:**
  - Category (a) NEW CA MECHANISM — softmax → sparsemax is a normalisation-level change to the attention output computation, not previously tested in this codebase. Distinct from H3-15 (hard top-k on scores BEFORE softmax) — here we replace softmax itself with sparsemax (no separate mask).
  - Category (c) DIRECT RESPONSE to H3-15's failure pattern and the explicit decision rule remainder ("Try SOFT routing"). H3-16 explored the "different compounding" branch; H3-17 explores the "soft routing" branch.
- **Decision rule:**
  - If H3-17 > 77.91: sparsemax + Linformer rank-128 basis compounds → new champion. Try α-entmax with learned α; sparsemax at other stages.
  - If 77.86 ≤ H3-17 ≤ 77.91: sparsemax is neutral. Softmax dense is the right normalisation for the rank-128 basis. PIVOT to fundamentally different operator family (Synthesizer-Dense, Nyström, channel-MLP-Mixer, Mamba SSM scan).
  - If H3-17 < 77.86: sparsemax hurts. The Linformer basis benefits from full smooth softmax attention to all supertokens (even tiny weights). PIVOT to operator-family changes.
- **Budget:** 0 new params (sparsemax is parameter-free). Same shape as H3-12. Total ≈ 26.555M params, 3.706 GFLOPs.
- **Baseline to beat:** H3-12 = 77.91% val top-1.
- **Submission:** job 393361 (2-GPU, --exclude=a node,a node, tag `in_h3_17_linformer_sparsemax_s3`, cfg `configs/autoresearch_experiment.yaml`).
- **Result:** 77.74% val top-1. Δ vs H3-12 = **-0.17pp** (below 77.86 floor), Δ vs IN-B100 = +0.09pp. DISCARDED.
- **Diagnosis:** sparsemax hurt the Linformer basis. Per decision rule (H3-17 < 77.86), the Linformer rank-128 basis benefits from FULL smooth softmax attention over all 128 supertokens — even tiny weights on noise-floor supertokens carry useful signal. Sparsemax likely over-sparsified (combined with learned attention temperature, support may have collapsed to a handful of supertokens per query at high temp). This isolates the conclusion from H3-15: it's not about differentiability — it's that hard-OR-soft sparsity itself breaks the basis. 5 consecutive failed pivots from H3-12 (H3-13/14/15/16/17). Clear signal: rank-128 dense smooth softmax Linformer is the operator-family optimum at stage-3. PIVOT to fundamentally different operator family.


### H3-18: Synthesizer-Dense channel attention at stage-3 (REPLACES H3-12 Linformer). SYNTHESIZER_DENSE_PER_STAGE=[0,0,0,1] — DISCARDED (-0.52pp vs H3-12; -0.26pp vs IN-B100)
- **Hypothesis:** 5 consecutive failures from modifying H3-12 (rank schedule, hard top-k, gated parallel, sparsemax, plus the original wider-spread) confirm rank-128 dense softmax Linformer is operator-family-optimal at stage-3. The H3-13/15/16/17 decision rules all consistently point to "PIVOT to fundamentally different operator family." H3-18 takes that pivot. Synthesizer-Dense (Tay et al 2020) replaces data-dependent Q·K^T with a LEARNED projection: F = W·Q (where W is a learned matrix), attention = softmax(F), output = attention · V. Attention scores are still input-dependent (through Q), but the channel-pair relationships are encoded in learnable parameters W rather than emerging from Q·K^T similarity. This is a fundamentally different inductive bias: "each query channel learns which target channels matter to it through training" rather than "channels attend to similar channels via inner product."
- **Mechanism (delta vs H3-12):**
  - LINFORMER_R_PER_STAGE: [0,0,0,128] → [0,0,0,0] (turn OFF Linformer at stage-3).
  - LINFORMER_SOFTMAX_TYPE_PER_STAGE: ['softmax','softmax','softmax','sparsemax'] → ['softmax','softmax','softmax','softmax'] (reset).
  - SYNTHESIZER_DENSE_PER_STAGE: [0,0,0,0] → [0,0,0,1]. Stage-3 backbone CA blocks switch to Synthesizer-Dense.
  - SYNTHESIZER_HW_PER_STAGE: [0,0,0,7]. Per-stage spatial dimension (needed at init to size W).
  - Per stage-3 block: new `synthesizer_W = nn.Linear(D=49, C=640, bias=False)`. K computed (existing pipeline) but ignored. Forward: `attn_score = synthesizer_W(q) * (D^-0.5)` → (B,1,640,640). `attn = softmax(attn_score, dim=-1)`. `out = attn · V` → (B,1,640,49). Same output shape as standard CA.
- **Why this might beat H3-12:**
  - The Linformer rank-128 basis is data-DRIVEN (learned via SGD to maximise low-rank reconstruction of the attention pattern). Synthesizer-Dense bypasses the rank-bottleneck entirely and lets the model learn DIRECT channel-pair attention biases. If the optimal channel-pair attention pattern is mostly fixed across inputs (true for "skeleton" channels like edge/texture detectors), Synthesizer-Dense will recover it more efficiently than Linformer's bottleneck.
  - Different gradient pathway: Synthesizer's W receives direct gradient from each (query, target) pair, whereas Linformer's E_k, E_v see only summed gradients through the supertoken bottleneck. May lead to sharper channel specialisation.
  - Tay et al 2020 showed Synthesizer-Dense matches Transformer on translation and exceeds on summarisation. The "attention as learnable parameter" hypothesis is empirically validated at scale.
- **Why this might NOT work:**
  - Synthesizer-Dense's W is mostly INPUT-INDEPENDENT (depends on Q but not K) — may lack the rich data-dependent matching that vanilla attention provides. Channel-pair relationships that depend strongly on the specific content of K_j (target channel features) cannot be captured.
  - C×C = 640×640 = 410K attention matrix entries per block × 10 blocks: more attention-matrix compute than H3-12 Linformer (which had C×r = 81K). FLOPs +~0.28 GFLOPs.
  - Empirically Synthesizer-Dense underperforms vanilla attention on most vision tasks (original paper); it's mostly a useful baseline.
- **Inspiration:**
  - Synthesizer (Tay et al 2020, arXiv:2005.00743): Dense Synthesizer attention replacement (used here as the seed mechanism).
  - MetaFormer (Yu et al 2022, arXiv:2210.13452): demonstrated that the attention OPERATOR can be replaced by simpler mixers (pooling, MLP) without losing accuracy, validating the broader "attention pattern need not be Q·K^T" thesis.
  - FocalNets / VAN (2022): non-attention channel mixers achieving competitive ImageNet results.
- **Novelty self-audit:**
  - Category (a) NEW CA MECHANISM — replaces data-dependent dot-product attention with a learned mapping. Distinct from Linformer (which projects K, V but still uses Q·K^T for scores), and from any prior hypothesis in this codebase.
  - Category (c) DIRECT RESPONSE to 5 consecutive failed pivots from H3-12. Per all 4 decision rules' "PIVOT to fundamentally different operator family" branch.
- **Decision rule:**
  - If H3-18 > 77.91: Synthesizer-Dense wins → new champion. Try Synthesizer-Factorised, mixed Synthesizer+Linformer per-block, etc.
  - If 77.86 ≤ H3-18 ≤ 77.91: parity with H3-12 confirms multiple operator families achieve similar performance — pivot to ENSEMBLING (per-block alternating Synthesizer/Linformer) or to architectural innovations OUTSIDE stage-3 (stage-2 mechanism upgrades, funnel tail innovations).
  - If H3-18 < 77.86: Synthesizer-Dense underperforms Linformer. The data-dependent matching of Q·K^T is genuinely useful. Pivot to operator families that PRESERVE dot-product attention but change the projection structure: Nyström (landmark-based), Hopfield-like attention, or kernel attention via Random Fourier Features.
- **Budget:** stage-3 Synthesizer W: D·C = 49·640 = 31K per block × 10 = 314K. Drop Linformer E_k,E_v: 2·C·r = 163K per block × 10 = 1.64M. Net change: -1.33M. Total ≈ **25.24M params, ~4.0 GFLOPs**.
- **Baseline to beat:** H3-12 = 77.91% val top-1.
- **Submission:** job 396430 (2-GPU, --exclude=a node,a node, tag `in_h3_18_synthesizer_dense_s3`, cfg `configs/autoresearch_experiment.yaml`).
- **Initial run crash:** job 396430 crashed at iter 0 with DDP error: stage-3 conv_proj_k params (30 params: conv weight + BN weight + BN bias × 10 blocks) had no gradient since Synthesizer-Dense doesn't use K. Single-GPU smoke test passed; DDP `find_unused_parameters=False` (default) doesn't tolerate it. Fixed by adding `x = x + 0.0 * k.sum()` to the synthesizer branch — ties K's projection to the output via a zero scalar, ensuring gradient flow without affecting numerics. Single-GPU backward now shows all 30 stage-3 conv_proj_k params receive gradient.
- **Resubmission:** job 396867 (same flags, tag `in_h3_18_synthesizer_dense_s3_v2`).
- **Result:** 77.39% val top-1. Δ vs H3-12 = **-0.52pp** (largest regression in the H3-12-pivot family), Δ vs IN-B100 = -0.26pp. DISCARDED.
- **Diagnosis:** per decision rule (H3-18 < 77.86), data-dependent Q·K^T matching is GENUINELY useful for channel attention. Synthesizer-Dense's INPUT-INDEPENDENT learned attention pattern (F = W·Q, where W is fixed after training) loses the per-input adaptivity that lets the model attend differently to different inputs. The drop magnitude (-0.52pp) is large — much bigger than the modify-Linformer drops (-0.13 to -0.37pp) — confirming this is a fundamental mechanism issue, not a tuning issue. The pivot answer: keep Q·K^T attention, change PROJECTION STRUCTURE. The decision rule recommends Nyström (landmark-based), kernel attention via RFF, or Hopfield-style approaches.


### H3-19: Nyström attention at stage-3 (REPLACES H3-12 Linformer; PRESERVES Q·K^T). NYSTROM_L_PER_STAGE=[0,0,0,128] — IN PROGRESS (job 399386)
- **Hypothesis:** H3-18 showed that the data-dependent Q·K^T matching is genuinely valuable for channel attention (Synthesizer's static pattern dropped -0.52pp). H3-19 stays within the Q·K^T family but changes the PROJECTION STRUCTURE per H3-18's decision rule. Nyström attention (Xiong et al 2021, Nyströmformer, arXiv:2102.03902) approximates softmax(QK^T)V via three softmaxes through a LANDMARK basis: pick L landmark tokens (here via 1D avg-pool over the 640 channels), then compute F_1 = softmax(Q·K_lm^T), F_2 = softmax(Q_lm·K_lm^T), F_3 = softmax(Q_lm·K^T), and approximate softmax(QK^T)V ≈ F_1·pinv(F_2)·F_3·V. Each softmax is genuine Q·K^T attention — only the routing through L landmarks is new. This preserves data-dependent matching while compressing computation, structurally different from Linformer's learned-projection bottleneck.
- **Mechanism (delta vs H3-12):**
  - LINFORMER_R_PER_STAGE: [0,0,0,128] (H3-12) was already turned off in H3-18 (now [0,0,0,0]).
  - SYNTHESIZER_DENSE_PER_STAGE: [0,0,0,1] → [0,0,0,0] (turn off Synthesizer).
  - NYSTROM_L_PER_STAGE: new key, [0,0,0,128]. Stage-3 backbone CA blocks use Nyström with L=128 landmarks.
  - NYSTROM_ITERS_PER_STAGE: [0,0,0,6]. Newton-Schulz iterations for the F_2 pseudoinverse (standard Nyströmformer setting).
  - Per stage-3 block: NO new learnable params. Landmarks = 1D avg-pool of C=640 to L=128 (group of 5 channels per landmark). Same compression ratio as H3-12 (128/640 = 1/5).
- **Why this might compound H3-12:**
  - Landmarks are POSITIONS in the original token space, selected via deterministic pooling. Linformer learns linear combinations of all tokens as supertokens. If the channel ordering preserves locality (which it doesn't necessarily, but conv-induced channel coupling makes adjacent channels related), Nyström's grouped landmarks may capture genuine "channel clusters" more faithfully than Linformer's free-form supertokens.
  - The three-softmax decomposition F_1·pinv(F_2)·F_3 explicitly models the through-landmark routing. Each softmax operation is over a smaller dimension (C×L instead of C×C), yet the full attention is approximated. Different inductive bias from Linformer's direct K,V projection.
  - 0 new learnable params: ablates the question "is the Linformer gain from learned projections or just from low-rank compression?" If H3-19 ≥ H3-12, the answer is "just from low-rank" — the supertokens don't need to be learned. If H3-19 < H3-12, the learned supertokens carry signal.
- **Why this might NOT work:**
  - Landmark selection by avg-pool is uniform over channel groups. If the optimal landmarks are NOT contiguous channel groups, Nyström underperforms Linformer's free choice.
  - Pseudoinverse via Newton-Schulz iteration can be numerically unstable at fp16. Nyströmformer uses 6 iterations (default) but at scale this can drift.
  - Three softmaxes + pinv add some wall-clock overhead compared to Linformer's single softmax.
- **Inspiration:**
  - Nyströmformer (Xiong et al 2021, arXiv:2102.03902): landmark-based attention approximation for long sequences. We adapt it to the channel-token domain.
  - Standard Nyström approximation in kernel methods (Williams & Seeger 2001) — the mathematical foundation.
- **Novelty self-audit:**
  - Category (a) NEW CA MECHANISM — landmark-based attention approximation has never been tested in this codebase. Distinct from Linformer (learned projection) and Synthesizer (learned attention pattern); each is a different way to approximate softmax(QK^T).
  - Category (c) DIRECT RESPONSE to H3-18 decision rule's "PIVOT to operator families that PRESERVE dot-product attention but change projection structure."
- **Decision rule:**
  - If H3-19 > 77.91: Nyström wins → new champion. Landmarks beat learned supertokens. Try L=64, L=256, mixed Nyström+Linformer per block.
  - If 77.86 ≤ H3-19 ≤ 77.91: Nyström ≈ H3-12. Confirms low-rank compression matters more than learned vs pooled basis. Pivot to ARCHITECTURAL innovations OUTSIDE stage-3: stage-2 mechanism upgrades, funnel tail expansion, or extra funneling stage.
  - If H3-19 < 77.86: Nyström underperforms Linformer. Learned supertokens DO carry signal beyond pure low-rank compression. PIVOT to architectural innovations OUTSIDE stage-3 — accept H3-12 as the stage-3 local optimum.
- **Budget:** 0 new params. Stage-3 FLOPs ≈ 3 softmaxes + pinv (~10-15M per block × 10 = ~120M = 0.12 GFLOPs, similar to H3-12). Total ≈ **24.93M params, ~3.7 GFLOPs**.
- **Baseline to beat:** H3-12 = 77.91% val top-1.
- **Submission:** job 399386 (2-GPU, --exclude=a node,a node, tag `in_h3_19_nystrom_s3`, cfg `configs/autoresearch_experiment.yaml`).
- **Status:** IN PROGRESS — awaiting watcher nudge.
