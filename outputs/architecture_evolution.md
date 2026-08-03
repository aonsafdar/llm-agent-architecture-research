# channel-primary ViT: Architecture Evolution Report

**Channel-Primary Vision Transformer — From Baseline to Champion**

---

## 1. The Core Idea

channel-primary ViT is a Vision Transformer where the standard roles of **spatial** and **channel** dimensions are swapped. In a conventional ViT, spatial locations are tokens and channels are features. In channel-primary ViT, **channels are tokens** and **spatial positions are features**. This enables pure *channel attention* — the model learns which channels should attend to each other by comparing their spatial activation patterns — without any spatial self-attention.

The research goal: demonstrate that a pure channel-attention architecture can be competitive with spatial self-attention models on vision benchmarks.

---

## 2. Baseline vs Current Champion: At a Glance

| Property | **Original Baseline** | **Current Champion (P2-H21)** |
|---|---|---|
| **Stages** | 3 | 4 |
| **Channel dims** | [64, 192, 384] | [96, 192, 384, 640] |
| **Depth (blocks/stage)** | [1, 2, 10] | [1, 2, 5, 10] |
| **CA MLP ratio** | [0, 0, 0] — *none* | [8, 8, 4, 1] |
| **DW shortcut** | ✗ | ✓ (3×3, all stages) |
| **CPE (positional enc.)** | ✗ | ✓ (5×5, periodic re-injection) |
| **SwiGLU MLP** | ✗ (GELU) | ✓ |
| **Auxiliary supervision** | ✗ | ✓ (stages 0+1+2, weights 0.1/0.2/0.4) |
| **Learned attn temperature** | ✗ | ✓ (per-head, log-parameterized) |
| **Layer Scale** | ✗ | ✓ (init 1e-4) |
| **Q/K/V kernel** | 3×3 uniform | 3×3 (stages 0–2), 5×5 (stage 3) |
| **Cross-stage fusion** | ✗ | ✓ (all stages GAP → classifier) |
| **Channel pair bias** | ✗ | ✓ (stage 3, learned C×C prior) |
| **Parameters** | 1.72M | 22.7M |
| **GFLOPs** | 0.42 | 3.51 |
| **CIFAR-10 (100ep)** | 69.67% | 96.59% (Phase 1) |
| **CIFAR-100 (100ep)** | — | 83.30% |
| **CIFAR-100 (300ep)** | — | **85.07%** |

---

## 3. Most Impactful Innovations

### 3.1 CA MLP: Feed-Forward Network After Channel Attention (+21.4pp)

**What changed:** The original baseline stacked pure channel-attention layers with no feed-forward network (FFN). A SwiGLU-gated MLP was added after each attention block across all stages, with ratio tuned per stage ([8, 8, 4, 1]).

**Why it matters:** In Transformer theory, the attention mechanism aggregates information across tokens (channels), but the FFN provides the per-token nonlinear transformation that builds expressive representations. Without it, the model can only permute and re-weight channel activations — it cannot learn complex nonlinear channel-specific features. Adding CA MLP immediately unlocked 21pp of accuracy.

**What it does to channel attention:** The MLP operates *after* attention, refining each channel's representation independently once global inter-channel context has been gathered. This is the standard Transformer recipe now applied in the channel domain.

**Literature:** Original Transformer architecture (Vaswani et al., 2017).

---

### 3.2 DW Shortcut Backbone: Local Spatial Mixing Per Block (+0.81pp)

**What changed:** A depthwise 3×3 convolution residual path was added to every attention block in the backbone (alongside the attention + MLP residuals).

**Why it matters:** Pure channel attention is *permutation-invariant over spatial positions* — it treats all spatial locations equally when computing Q/K features. The DW shortcut injects local spatial structure directly into the residual stream: each block can exploit nearby pixel relationships without disturbing the global channel attention. This is a complementary inductive bias — channel attention provides global inter-channel relationships, while the DW shortcut provides local spatial coherence.

**What it does to channel attention:** Enriches the residual features with local spatial context at every block, so subsequent channel attention blocks operate on spatially-grounded representations rather than purely global ones.

**Literature:** ConvNeXt (Liu et al., 2022, arXiv:2201.03545) established 3×3 depthwise conv as an effective spatial mixer in ViT-like architectures.

---

### 3.3 Auxiliary Deep Supervision: Gradient Shortcuts to Early Stages (+0.74pp cumulative)

**What changed:** Independent classification heads were added at stages 0, 1, and 2 (in addition to the main head at stage 3), with auxiliary loss weights of 0.1, 0.2, and 0.4. These heads are active only during training.

**Why it matters:** In a 4-stage deep hierarchy, gradients from the final loss must propagate back through all 10+ blocks to reach stage 0. This gradient path degrades, leaving early-stage channel attention blocks poorly supervised. The auxiliary heads inject classification signal directly at each stage, forcing early channels to learn semantically meaningful features from the start.

**What it does to channel attention:** Gives every CA stage a direct learning objective. Stage-0 channels (after just 1 block) must organize themselves to predict class labels — this forces meaningful channel specialization at the earliest scale. The multi-scale gradient scaffolding improves the entire channel hierarchy.

**Literature:** Deeply Supervised Nets (Lee et al., 2015, arXiv:1409.5185); GoogLeNet auxiliary classifiers (Szegedy et al., 2014).

---

### 3.4 Convolutional Position Encoding (CPE 5×5): Spatial Identity for Channel Tokens (+0.26pp)

**What changed:** A 5×5 depthwise convolution (initialized to zero/identity) was added at each stage entry, applied to the 2D feature map before channel tokenization. Periodic re-injection was also added within stage 3 (after blocks 3 and 7).

**Why it matters:** Channel tokens represent entire channels across all spatial positions. When computing Q/K similarities between channels, the features must encode *where* each channel is active. Without explicit positional encoding, channel tokens lack spatial identity — two channels with the same spatial activation magnitude but different spatial distributions look identical. CPE injects local spatial context (via a learned DW conv) into every channel token before attention.

**What it does to channel attention:** Makes Q/K feature vectors spatially informed. Channel A attends strongly to Channel B not just because their overall activations are similar, but because their *spatial patterns* are similar. This is the key inductive bias that makes channel attention semantically meaningful rather than just activation-level.

**Literature:** CPVT (Chu et al., 2021, arXiv:2102.10882); Twins (Xiaohan et al., 2021, arXiv:2104.13840) — conditional positional encodings for ViTs.

---

### 3.5 SwiGLU MLP: Gated Feature Selection (+0.12pp)

**What changed:** The GELU-activated MLP in each block was replaced with a SwiGLU-gated MLP: `output = SiLU(W_gate · x) ⊙ (W_val · x)`. The hidden dimension is scaled by 2/3 to maintain the same parameter count.

**Why it matters:** SwiGLU introduces a learned multiplicative gate that selectively amplifies or suppresses features based on the input. Unlike GELU (which applies a fixed nonlinearity), SwiGLU can route information adaptively — if the gate activates strongly for a feature, it passes through; otherwise it is suppressed. This is particularly valuable in the channel domain where features vary widely in relevance per input.

**What it does to channel attention:** The MLP that follows channel attention can now selectively emphasize the channels whose inter-dependencies were most informative, and suppress noise from weakly-relevant channel interactions.

**Literature:** Shazeer (2020, arXiv:2002.05202); used in LLaMA, PaLM, Gemma.

---

### 3.6 Learned Attention Temperature (+0.12pp)

**What changed:** A per-head log-parameterized scalar temperature `τ = exp(log_τ)` was added to scale Q before SDPA: `Q' = Q × τ`. Initialized to `log_τ = 0` (τ = 1, no change at start).

**Why it matters:** In channel-primary ViT, channel attention operates at different spatial resolutions per stage: stage 0 at 56×56 (3136 spatial features per token), stage 3 at 7×7 (49 features). The effective attention logit magnitude scales with `√(H×W)`, producing dramatically different sharpness across stages. Stage 0 attention is naturally very sharp (competitive); stage 3 is naturally soft. A fixed temperature cannot be optimal for all stages simultaneously. The learned temperature allows each head to independently calibrate its attention distribution.

**What it does to channel attention:** Gives each stage the optimal trade-off between sharp (selective) and flat (diffuse) channel attention. Early stages may prefer softer attention to integrate global context; later semantic stages can benefit from sharper, more selective channel interactions.

**Literature:** XCiT (El-Nouby et al., 2021, arXiv:2106.09681) — per-head temperature in cross-covariance attention; SSA (Zhang et al., 2024, arXiv:2411.12892).

---

### 3.7 Cross-Stage Channel Fusion: Multi-Scale Channel Context at Classifier (+0.38pp in Phase 2)

**What changed:** At inference time, global average pooled (GAP) features from stages 0, 1, and 2 are projected and added to the final stage-3 representation before classification. Learned scale parameters (init=0) ensure safe convergence from identity.

**Why it matters:** Each stage captures channel features at a different spatial resolution and semantic level. Stage 1 (28×28) captures local texture-level channel interactions; stage 3 (7×7) captures high-level semantic channel patterns. The final classifier previously used only stage-3 features. Fusing multi-scale channel representations at the head gives the classifier a richer view of the channel hierarchy.

**What it does to channel attention:** Elevates the role of earlier-stage channel attention from "training signal only" (via auxiliary loss) to "inference signal also." Channel features learned at every scale contribute to the final prediction.

**Literature:** DuoFormer (Tang et al., 2024, arXiv:2407.13920) — scale token aggregation from all stages improves classification by 3–9%.

---

### 3.8 Channel Pair Bias: Learned Inter-Channel Affinity Prior (+0.12pp in Phase 2)

**What changed:** A learned C×C bias matrix `B` (initialized to zeros) was added to the attention logits at stage 3: `attn = softmax(QK^T/√d + B)`. The bias is shared across all blocks in stage 3 and excluded from weight decay.

**Why it matters:** Standard softmax attention assumes a *uniform prior* over all channel pairs before seeing the input. But real networks develop stable inter-channel relationships — certain channels co-activate reliably (e.g., color and texture channels), while others rarely interact. A learned bias matrix encodes these structural affinities as a persistent prior, freeing Q/K to focus on *instance-specific* channel interactions rather than relearning stable priors at every forward pass.

**What it does to channel attention:** Separates the two roles of channel attention: (1) stable structural priors (handled by the bias) and (2) input-dependent channel interactions (handled by Q/K). This is the channel-domain analogue of relative position bias in spatial ViTs.

**Literature:** Swin Transformer (Liu et al., 2021, arXiv:2103.14030) — relative position bias for spatial tokens; GOAT (2025, arXiv:2601.15380) — learnable attention priors.

---

## 4. Architecture Diagram

```
Input Image (224×224×3)
       │
 ┌─────▼──────┐
 │  Stage 0   │  96-dim │ 56×56 │ 1 block
 │  CPE 5×5   │  CA + SwiGLU MLP + DW shortcut
 │  Aux Head 0│  → Aux loss (w=0.1, training only)
 └─────┬──────┘
       │ patch embed (stride 2)
 ┌─────▼──────┐
 │  Stage 1   │  192-dim │ 28×28 │ 2 blocks
 │  CPE 5×5   │  CA + SwiGLU MLP + DW shortcut
 │  Aux Head 1│  → Aux loss (w=0.2, training only)
 └─────┬──────┘  GAP → cross-stage fusion ─────────────────┐
       │ patch embed (stride 2)                             │
 ┌─────▼──────┐                                             │
 │  Stage 2   │  384-dim │ 14×14 │ 5 blocks                │
 │  CPE 5×5   │  CA(3×3 QKV) + SwiGLU MLP + DW shortcut   │
 │  Aux Head 2│  → Aux loss (w=0.4, training only)         │
 └─────┬──────┘  GAP → cross-stage fusion ─────────────────┤
       │ patch embed (stride 2)                             │
 ┌─────▼──────────────────────────┐                        │
 │  Stage 3   │  640-dim │ 7×7   │                        │
 │  CPE 5×5 (+ re-inject at 3,7) │  10 blocks             │
 │  CA(5×5 QKV) + Pair Bias      │                        │
 │  + Temp τ + SwiGLU + DW       │                        │
 └─────┬──────────────────────────┘                        │
       │ GAP                                               │
 ┌─────▼──────────────────────────────────────────────────▼─┐
 │           Cross-Stage Fusion Head                         │
 │   stage3_feat + scale1·proj(stage1_feat)                  │
 │              + scale2·proj(stage2_feat)                   │
 │              + scale0·proj(stage0_feat)                   │
 └─────────────────────┬─────────────────────────────────────┘
                       │
               Linear(640 → num_classes)
```

*All channel attention (CA) blocks: channels are tokens, spatial positions are features. No spatial self-attention anywhere.*

---

## 5. Key Structural Principles Validated

1. **Channel attention needs FFN.** Pure attention stacking without MLP is severely underpowered. The MLP is not optional — it provides the nonlinear transformation that turns attended channel mixtures into discriminative features.

2. **Local + global is better than global alone.** DW shortcuts and CPE inject local spatial structure that channel attention (global by nature) cannot model. The combination systematically outperforms either alone.

3. **Gradient flow is a critical bottleneck.** In a 4-stage deep hierarchy, early-stage channel attention is poorly supervised without auxiliary losses. The +0.74pp from auxiliary supervision shows that architectural innovation is not enough — training dynamics must be addressed explicitly.

4. **Spatial awareness matters for channel tokens.** CPE (+0.26pp) confirms that channel tokens need to know *where* they are active, not just *how much*. Without spatial identity, channel attention conflates structurally different activation patterns.

5. **Multi-scale channel features are complementary.** Cross-stage fusion (+0.38pp) shows that each stage's channel attention captures information at a different level of abstraction that is not redundant — all scales contribute to the final classification.

6. **Single-head attention is optimal for channels.** Unlike spatial ViTs where multi-head attention reliably helps, channel attention prefers single-head (full spatial context per token). Splitting the spatial feature dimension across heads impoverishes each head's ability to compare channel activation patterns.

---

## 6. Results Summary

| Benchmark | Protocol | Result |
|---|---|---|
| CIFAR-10 | 100ep, 5.84M params | **96.59%** top-1 |
| CIFAR-10 | 300ep, 5.84M params | **97.25%** top-1 |
| CIFAR-100 | 100ep, 22.7M params | **83.30%** top-1 |
| CIFAR-100 | 300ep, 22.7M params | **85.07%** top-1 |
| ImageNet-1K | 300ep, 22.7M params | *In progress* |

**Baseline for comparison:** CIFAR-10 100ep starting point: 69.67%. Total improvement: +26.9pp from architectural and training innovations, all within the pure channel-attention framework.

---

*Generated: April 2026 | channel-primary ViT Autonomous Architecture Research*
