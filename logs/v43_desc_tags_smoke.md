# v43 richer descriptive LM tags smoke (2026-07-25)

## Setup

- Dataset: ~5.5k rows / 102 accounts (smoke scale; not full scoreboard)
- Labeler: `uv run python label_desc.py --local --nonreply-only --limit 500`
  Model: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) on CPU (~0.22 tweets/s)
  Cache: `desc_cache_Qwen__Qwen2.5-1.5B-Instruct.parquet`
- Tags: humor / controversy / narrative / cta / news / dunk_bait (0–10)
- Train: `--desc-labels --desc-cache …` vs control

## Coverage / calibration

| | |
|---|---:|
| ok labels | 500 / 500 |
| dataset coverage | 9.1% (among NR 39.7%) |
| humor mean / p50 / ≥7 | 4.60 / 2.0 / **45.4%** |
| controversy mean / p50 / ≥7 | 5.55 / 6.0 / **45.2%** |
| narrative mean / p50 / ≥7 | 5.58 / 6.0 / 27.8% |
| cta mean / p50 / ≥7 | 3.01 / 4.0 / 0.6% |
| news mean / p50 / ≥7 | 2.82 / 3.0 / 0.2% |
| dunk_bait mean / p50 / ≥7 | 2.39 / 1.0 / 3.4% |

1.5B is **overconfident on humor/controversy** (prompt asked for most posts in 0–3) — same noisy-labeler pattern as v17/v42 1.5B. `cta` / `news` / `dunk_bait` look better calibrated.

## Correlation vs exposure rate (labeled non-replies)

| feature | spearman(rate) | within-account wmean |
|---|---:|---:|
| `humor` | **+0.143** | +0.061 |
| `controversy` | +0.042 | +0.041 |
| `narrative` | **+0.142** | **+0.125** |
| `cta` | +0.036 | +0.101 |
| `news` | +0.036 | +0.047 |
| `dunk_bait` | +0.018 | −0.005 |
| `desc_max` | **+0.128** | +0.087 |

`humor` / `narrative` / `desc_max` clear the ~0.06 content gate; controversy / CTA / news / dunk fail globally (CTA has modest within-author pierce).

## Small-scale train (sparse join; 9.1% `has_desc`)

| run | wa_nr_wmean | wa_rp_wmean | global_nr | rmse_log |
|---|---:|---:|---:|---:|
| v43a control | 0.148 | 0.247 | 0.394 | 0.988 |
| v43b + `--desc-labels` | **0.165** | 0.247 | **0.408** | **0.981** |

Δwa_nr **+0.016** / wa_rp frozen (NR-only labels + hybrid) / rmse −0.007 / global_nr +0.014. Desc cols absent from top-15 gains — treat the wa_nr bump as **smoke-scale only**, not a promote.

## Verdict

- Harness: `label_desc.py` + `train.py --desc-labels` — **leave opt-in**
- Smoke **passes** coverage×corr gate for humor/narrative (unlike stance v42 dense fail)
- Do **not** promote on 1.5B + 9% coverage; revisit with calibrated denser labels (Modal Qwen3-8B, all NR / full-scale) before a scoreboard bet
