# v45 denser descriptive LM tags — Qwen3-8B (2026-07-26)

## Setup

- Dataset: medium prepare 23,456 / 315 accounts (0% media via `--skip-media`)
- Labeler: `uv run modal run label_desc.py --nonreply-only` → `Qwen/Qwen3-8B` A10G (~2.1 tw/s)
- Cache: `desc_cache_Qwen__Qwen3-8B.parquet` — **4,473 ok / 4,475 NR** (19.1% of rows)
- Train: `--desc-labels --desc-cache …` vs same-rebuild control

## 500-row calibration smoke

| | 1.5B (v43) | 8B (v45) |
|---|---:|---:|
| humor mean / ≥7 | 4.60 / **45%** | 3.00 / **18%** |
| controversy mean / ≥7 | 5.55 / **45%** | 2.65 / **16%** |
| narrative mean / ≥7 | 5.58 / 28% | 1.87 / 11% |

8B is substantially better calibrated on humor/controversy.

### Spearman vs exposure rate (500 labeled NR)

| feature | ρ_rate | wa_wmean |
|---|---:|---:|
| humor | **0.117** | 0.063 |
| narrative | **0.165** | 0.029 |
| controversy | 0.030 | 0.088 |
| desc_max | 0.085 | 0.081 |

Humor / narrative cleared ~0.06 → densified remaining NR.

## Dense coverage / pierce (all NR)

| feature | ρ_rate | wa_wmean | mean | ≥7 |
|---|---:|---:|---:|---:|
| humor | **0.077** | 0.052 | 2.92 | 17.1% |
| controversy | 0.057 | 0.097 | 2.67 | 16.5% |
| narrative | **0.137** | **0.112** | 1.79 | 9.8% |
| cta | 0.014 | 0.028 | 0.29 | 2.3% |
| news | 0.021 | 0.045 | 0.47 | 0.9% |
| dunk_bait | 0.020 | 0.052 | 1.84 | 14.0% |
| desc_max | 0.098 | 0.092 | 4.95 | 43.5% |

Narrative remains the strongest dense pierce; humor weakens vs the 500-row smoke (0.117→0.077) but still clears the gate.

## Medium train

| run | wa_nr | wa_rp | global_nr | rmse | underpred |
|---|---:|---:|---:|---:|---:|
| v45a control | **0.219** | 0.278 | 0.351 | 0.744 | 2.43 |
| v45b `--desc-labels` | 0.208 | 0.278 | **0.357** | **0.739** | **2.41** |

Δ: wa_nr **−0.011** / wa_rp frozen (NR-only + hybrid) / global_nr +0.006 / rmse −0.005.

Top gains include `desc_narrative` / `desc_controversy` / `desc_max` — gain without locked-primary lift (between-author / magnitude nudge; within-author NR order hurts).

## Verdict

- Harness unchanged: `--desc-labels` / Modal `label_desc.py` — **leave opt-in**
- Denser calibrated 8B **does not promote** on medium scoreboard (wa_nr regress)
- v43 sparse 1.5B +wa_nr smoke does not survive denser 8B join
- Do not full-spend labeling the ~110k scoreboard set without a new angle (e.g. within-author-only labels, or humor/narrative subset features)
