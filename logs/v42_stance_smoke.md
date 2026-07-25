# v42 LM agree/disagree-vs-parent stance smoke (2026-07-25)

## Setup

- Dataset: ~5.5k rows / 102 accounts
- Labeler: `uv run python label_stance.py --local --limit 500`  
  Model: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) on CPU (~0.46 pairs/s)  
  Cache: `stance_cache_Qwen__Qwen2.5-1.5B-Instruct.parquet`
- Modal / Qwen3-8B not available in this cloud env (no Modal token); same harness supports `uv run modal run label_stance.py`
- Train: `--stance-labels --stance-cache …` vs control

## Coverage / calibration

| | |
|---|---:|
| ok labels | 508 / 508 |
| dataset coverage | 9.2% (smoke limit; among replies 11.9%) |
| agree mean / p50 / ≥7 | 5.41 / **8.0** / **56.5%** |
| disagree mean / p50 / ≥7 | 3.79 / 2.0 / 27.4% |

Prompt asked for most ordinary replies in 0–3; 1.5B rater is **overconfident on agree** (mode 8) — same noisy-labeler pattern as v17 provoke 1.5B.

## Correlation vs exposure rate (labeled replies)

| feature | spearman(rate) | within-account wmean |
|---|---:|---:|
| `agree` | +0.057 | +0.104 |
| `disagree` | +0.012 | −0.041 |
| `stance_net` | +0.029 | +0.075 |
| `stance_conflict` | **+0.172** | +0.135 |

`agree` sits on the ~0.06 content-gate line; `conflict` pierces but is entangled with the overconfident dual-mid scores. Spot checks: some disagree≥8 look wrong (e.g. light jokes).

## Small-scale train (sparse join; most rows `has_stance=0`)

| run | wa_nr_wmean | wa_rp_wmean | global_nr | rmse_log |
|---|---:|---:|---:|---:|
| v42a control | 0.148 | **0.247** | 0.394 | 0.988 |
| v42b + `--stance-labels` | 0.148 | 0.221 | **0.425** | **0.976** |

Stance cols absent from top gains. **Verdict:** leave `--stance-labels` **opt-in**; do not promote. Revisit only with a better-calibrated rater (Modal Qwen3-8B) and denser coverage — not a scoreboard bet on this 1.5B smoke.
