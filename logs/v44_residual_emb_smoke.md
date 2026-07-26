# v44 residual emb CAV/PLS smoke (2026-07-26)

## Setup

- Dataset: ~5.5k rows / 102 accounts (smoke scale; not full scoreboard)
- Control: default hybrid stack
- Ablations (one knob each):
  - `--residual-emb-pls 8`
  - `--residual-emb-cav` (shares `--emb-cav-quantile` / `--emb-cav-nonreply-only`)
- Residualization: train-fold OLS of each MiniLM dim ~ `account_mean_rate` + `is_reply` + `inbound_replies_per_tweet` (+ intercept); CAV/PLS on residuals before emb PCA

## Residualization diagnostics

| | |
|---|---:|
| mean R²(emb \| Z) | **0.015** |
| residual RMS | 0.047 |

Account TE / reply baselines barely explain MiniLM linearly — residual space ≈ raw emb. The “new angle” is weak in practice.

## Train-fold content pierce

| feature | train corr(*, y) | notes |
|---|---:|---|
| `res_pls_0` | **−0.141** | opposite sign vs historical raw `pls_0` (~+0.17) |
| residual PLS in-sample R² | 0.090 | |
| `res_emb_cav_proj` | **+0.139** | ≈ raw emb CAV (~0.14) |

## Small-scale train

| run | wa_nr_wmean | wa_rp_wmean | global_nr | rmse_log | underpred_nr_top10 |
|---|---:|---:|---:|---:|---:|
| v44a control | 0.148 | 0.247 | 0.394 | 0.988 | 2.79 |
| v44b `--residual-emb-pls 8` | 0.146 | **0.261** | **0.309** | 0.996 | 2.85 |
| v44c `--residual-emb-cav` | **0.172** | 0.234 | **0.409** | 0.988 | 2.76 |

- **PLS:** high gain on `res_pls_*` but **flat/worse** locked primary + worse global_nr/RMSE — do not promote.
- **CAV:** smoke +wa_nr +0.024 / −wa_rp / flat RMSE; mid-gain `res_emb_cav_proj`. Treat as **smoke-scale only** (41 NR val accounts); mechanism weak given R²(emb|Z)≈0.015.

## Verdict

- Harness: `train.py --residual-emb-pls K` / `--residual-emb-cav` — **leave opt-in**
- Residual PLS: fails gate (global_nr collapse on smoke)
- Residual CAV: interesting smoke NR bump but nearly raw-CAV redux; needs larger-scale confirm before scoreboard claim
- Note: denser Modal Qwen3-8B desc-tag follow-up (v43) was blocked in this environment (no Modal token)
