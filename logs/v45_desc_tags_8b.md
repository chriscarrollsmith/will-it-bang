# v45 denser descriptive LM tags — Qwen3-8B (in progress)

## Setup

- Dataset: medium prepare ~23.5k / 315 accounts (0% media via `--skip-media`)
- Labeler: `uv run modal run label_desc.py --nonreply-only` → `Qwen/Qwen3-8B` on A10G
- Cache: `desc_cache_Qwen__Qwen3-8B.parquet`
- Train: `--desc-labels --desc-cache …` vs control

## 500-row calibration smoke (before densify)

| | 1.5B (v43) | 8B (v45) |
|---|---:|---:|
| humor mean / ≥7 | 4.60 / **45%** | 3.00 / **18%** |
| controversy mean / ≥7 | 5.55 / **45%** | 2.65 / **16%** |
| narrative mean / ≥7 | 5.58 / 28% | 1.87 / 11% |
| cta / news / dunk ≥7 | 0.6% / 0.2% / 3.4% | 2.0% / 1.0% / 12% |

8B is substantially better calibrated on humor/controversy (prompt asked for most posts in 0–3).

### Spearman vs exposure rate (500 labeled NR)

| feature | ρ_rate | wa_wmean |
|---|---:|---:|
| humor | **0.117** | 0.063 |
| controversy | 0.030 | 0.088 |
| narrative | **0.165** | 0.029 |
| cta | 0.013 | −0.207 |
| news | 0.077 | 0.007 |
| dunk_bait | 0.012 | 0.077 |
| desc_max | 0.085 | 0.081 |

Humor / narrative clear the ~0.06 content gate → densifying remaining NR (~3,975).

## Dense + train

_(pending densify)_
