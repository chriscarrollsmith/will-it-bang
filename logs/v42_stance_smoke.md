# v42 LM agree/disagree-vs-parent stance smoke (2026-07-25)

## Setup

- Dataset: ~5.5k rows / 102 accounts
- Labeler: `label_stance.py` (reply→parent pairs; parent text from CA dump)
- Raters:
  - **v42a/b:** local `Qwen/Qwen2.5-1.5B-Instruct` (no Modal token initially)
  - **v42c–f:** Modal `Qwen/Qwen3-8B` on A10G (~4.6 pairs/s)
- Cache: `stance_cache_Qwen__Qwen3-8B.parquet` (3,428 ok / 3,429)
- Train: `--stance-labels --stance-cache …` vs control

## Calibration

| rater | n | agree mean/p50/≥7 | disagree mean/p50/≥7 |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 508 | 5.41 / **8.0** / **56%** | 3.79 / 2.0 / 27% |
| Qwen3-8B (500) | 500 | 3.65 / 3.0 / 37% | 0.97 / 0.0 / 8% |
| Qwen3-8B (dense) | 3428 | 3.66 / 3.0 / 37% | 0.73 / 0.0 / 6% |

8B is better calibrated (disagree mostly 0; fewer false high-agree). Spot checks of agree≥7 look plausible; a few disagree≥8 still look wrong.

## Correlation vs exposure rate (labeled replies)

| rater / slice | agree ρ | disagree ρ | net ρ | conflict ρ |
|---|---:|---:|---:|---:|
| 1.5B n=508 | +0.057 | +0.012 | +0.029 | +0.172 |
| 8B n=500 | **+0.091** | +0.096 | +0.048 | +0.035 |
| **8B dense n=3428** | **+0.021** | +0.046 | +0.007 | +0.001 |

Dense 8B fails the ~0.06 content gate (500-row lift was sampling noise). Within-account agree wmean ≈0.05 on dense set.

Coverage when dense: **62%** of rows / **80.5%** of replies.

## Small-scale train

| run | labels | wa_nr | wa_rp | global_nr | rmse |
|---|---|---:|---:|---:|---:|
| v42e control | — | 0.148 | **0.247** | 0.394 | 0.988 |
| v42b 1.5B sparse | 508 | 0.148 | 0.221 | 0.425 | 0.976 |
| v42d 8B sparse | 500 | 0.148 | 0.221 | 0.390 | 0.988 |
| v42f 8B dense | 3428 | 0.148 | 0.218 | — | — |

Stance cols absent from top gains in all ablations. **Verdict:** leave `--stance-labels` **opt-in**; do not promote. Better rater improves calibration but not target correlation or wa_rp on this smoke set.
