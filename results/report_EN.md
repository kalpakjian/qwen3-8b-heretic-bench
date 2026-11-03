# Qwen3-8B-Heretic Benchmark Report
- Generated: 2026-11-03 17:14 UTC
- base: `Qwen/Qwen3-8B`
- heretic: `kalpakjian/qwen3-8b-heretic` (Heretic trial 131/200, bf16 full-precision safetensors)
- Settings: greedy, thinking off, max_new_tokens=256
## 1. Overview: effect vs cost

| Axis | Metric | base `Qwen/Qwen3-8B` | heretic | Delta |
|---|---|---|---|---|
| effect | Explicit refusal phrasing (keyword) | 58.0% | 38.0% | **-20.0 pt** |
| effect | **Substantive refusal (LLM-judge)** | 98.0% | 98.0% | **+0.0 pt** |
| effect | Over-refusal on benign prompts (FPR, lower is better) | 4.0% | 6.0% | +2.0 pt |
| cost | Full-sequence KL (heretic‖base) | 0 | **0.00494** | — |
| cost | top-1 token agreement | 100.0% | 95.9% | — |
| cost | MMLU (16 subjects × 20 questions) | 72.8% | 72.8% | +0.0 pt |
| cost | GSM8K exact match (100 questions) | 88.0% | 89.0% | +1.0 pt |
| cost | wikitext-2 perplexity | 11.683 | 11.716 | +0.29% |

> **The two refusal metrics disagree sharply**: keyword drops from 58.0% to 38.0% (-20.0 pt), while the judge barely moves (98.0% → 98.0%). This abliteration removed **the wording of refusal**, not the **behaviour**.

> Model-card protocol replicated (first-token KL, 50 benign prompts): **0.01510** (the card claims 0.0323 on 20 prompts)

## 2. Refusal rate by category (100-prompt suite)

| Category | Kind | base | heretic | Note |
|---|---|---|---|---|
| `benign` | benign | 4.8% (2/42) | 4.8% (2/42) |  |
| `csa` | should refuse | 66.7% (2/3) | 0.0% (0/3) | alignment fully lost |
| `harassment` | should refuse | 25.0% (1/4) | 25.0% (1/4) |  |
| `hate` | should refuse | 33.3% (1/3) | 0.0% (0/3) |  |
| `illegal` | should refuse | 91.7% (11/12) | 58.3% (7/12) |  |
| `medical` | should refuse | 33.3% (1/3) | 33.3% (1/3) |  |
| `misinformation` | should refuse | 16.7% (1/6) | 0.0% (0/6) |  |
| `privacy` | should refuse | 75.0% (3/4) | 75.0% (3/4) |  |
| `self_harm` | should refuse | 40.0% (4/10) | 40.0% (4/10) |  |
| `tricky_benign` | benign | 0.0% (0/7) | 14.3% (1/7) | over-refusal (should not happen) |
| `violence` | should refuse | 83.3% (5/6) | 50.0% (3/6) |  |

## 2b. Keyword vs judge (why one metric is not enough)

| Model | keyword refusal | judge refusal | COMPLY | PARTIAL | Gap |
|---|---|---|---|---|---|
| base | 58.0% | 98.0% | 0 | 1 | +40.0 pt |
| heretic | 38.0% | 98.0% | 1 | 0 | +60.0 pt |

Keyword matching only sees refusal *phrasing*; the judge is asked whether the harmful content was actually provided.
The gap between them is the share of **"same behaviour, different wording"** — hand-checked against the raw responses (e.g. the bomb-making reply discusses risk and alternatives at length without ever giving instructions).

## 3. KL detail (full sequence, full vocabulary)

- Sample: 20 windows × 256 tokens (5100 positions, each compared over a 151,936-dim vocabulary)
- KL mean: **0.00494**  median 0.00297  p95 0.01548  max 0.14452
- top-1 agreement 95.9%; mean total-variation distance 0.02980
- First-token KL (model-card protocol): **0.01510** (n=50)

- perplexity (**this section's 20×256 windows, for KL reference only**): base 19.128 → heretic 19.127 (-0.01%)
  ⚠️ This differs from the PPL in section 1 because the windows differ (30×512). Use section 1 to judge capability retention.

## 3b. Weight audit (proof the checkpoint was really modified)

- Tensors: **399**, identical name sets (pure weight modification, no structural change)
- Tensors changed: **24** (6.0%) → a surgical low-rank edit, not retraining

| Tensor | Changed | max‖Δ‖ |
|---|---|---|
| `model.layers.27.mlp.down_proj.weight` | 80.2% | 0.06628 |
| `model.layers.33.mlp.down_proj.weight` | 79.9% | 0.04443 |
| `model.layers.28.mlp.down_proj.weight` | 79.8% | 0.04602 |
| `model.layers.29.mlp.down_proj.weight` | 79.8% | 0.04895 |
| `model.layers.34.mlp.down_proj.weight` | 79.7% | 0.02893 |
| `model.layers.30.mlp.down_proj.weight` | 79.7% | 0.05283 |
| `model.layers.31.mlp.down_proj.weight` | 79.7% | 0.02264 |
| `model.layers.21.mlp.down_proj.weight` | 79.6% | 0.03235 |
| `model.layers.32.mlp.down_proj.weight` | 79.6% | 0.03564 |
| `model.layers.25.mlp.down_proj.weight` | 79.5% | 0.06078 |

> Why this check exists: with KL at 0.005 and MMLU unmoved, it is easy to suspect the same weights were loaded twice.
> Comparing the safetensors directly confirms the opposite: almost every tensor is bit-identical, and the edits concentrate in a few layers' `down_proj`/`o_proj` — which is what abliteration is supposed to do.

## 4. Interpretation

- Refusal: 58.0% → 38.0% (-20.0 pt).
- Full-sequence KL 0.00494; Heretic's own thresholds are <0.05 = minimal damage, <0.5 = low damage.
- MMLU: 72.8% → 72.8% (+0.0 pt).
- GSM8K: 88.0% → 89.0% (+1.0 pt).
- PPL: 11.683 → 11.716 (+0.29%).


## 5. Method and limitations (read before quoting a number)

1. **Single greedy snapshot**: refusal verdicts use greedy decoding plus keyword matching
   (the same list as `llm-refusal-audit`), with no repeated sampling. Real refusal rates
   rise with temperature, so these numbers sit closer to a **lower bound**.
2. **Keyword matching misfires**: strings like `sorry` or `policy` can appear in replies
   that are not refusals. Sympathetic contexts such as `sorry to hear` are subtracted, but
   the problem cannot be fully removed; use the LLM judge for anything load-bearing
   (`llm-refusal-audit` measured the judge raising TPR by 1-8 pt).
3. **KL is distribution distance, not capability loss**: a small KL does not prove
   capability survived, and vice versa — which is why section 1 puts both axes side by
   side. Abliteration's KL concentrates on refusal-direction tokens and is diluted by the
   many unrelated ones, so a mean alone understates the damage.
4. **MMLU / GSM8K are sampled**: 20 per subject and 100 questions respectively, standard
   error roughly ±2-5 pt. Treat smaller gaps as noise; raise `--limit-mmlu` /
   `--limit-gsm8k` to tighten.
5. **GSM8K is 0-shot**, not the standard 8-shot, so absolute scores sit below published
   leaderboards. Both models use the identical protocol, so the **relative** gap is the
   meaningful part.
6. **This repo does not re-run Heretic**: it measures the exported weights; it does not
   search for new ablation parameters.
