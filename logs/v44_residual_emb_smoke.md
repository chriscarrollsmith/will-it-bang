# v44 residual emb CAV/PLS (2026-07-26)

## Setup

OLS-residualize MiniLM vs `account_mean_rate` + `is_reply` + `inbound_replies_per_tweet` (train fold), then CAV/PLS on residuals before emb PCA.

| scale | rows | accounts | prepare |
|---|---:|---:|---|
| smoke | ~5.5k | 102 | prior cache |
| medium | 23,456 | 315 | `--max-tweets 25000 --per-account-cap 100 --skip-media` |

## Residualization diagnostics

| scale | mean R²(emb \| Z) | residual RMS |
|---|---:|---:|
| smoke | 0.015 | 0.047 |
| medium | 0.011 | 0.047 |

Baselines barely explain MiniLM linearly — residual ≈ raw in L2, but the CAV *direction* still shifts enough to matter (see twin below).

## Train-fold pierce

| feature | scale | corr(*, y) |
|---|---|---:|
| `res_pls_0` | smoke | −0.141 |
| `res_pls_0` | medium | −0.103 |
| `res_emb_cav_proj` | smoke | +0.139 |
| `res_emb_cav_proj` | medium | +0.092 |
| raw `emb_cav_proj` | medium twin | +0.144 |

## Smoke train (~5.5k)

| run | wa_nr | wa_rp | global_nr | rmse |
|---|---:|---:|---:|---:|
| v44a control | 0.148 | 0.247 | 0.394 | 0.988 |
| v44b `--residual-emb-pls 8` | 0.146 | 0.261 | **0.309** | 0.996 |
| v44c `--residual-emb-cav` | **0.172** | 0.234 | 0.409 | 0.988 |

## Medium train (23k / 315; fair twins)

| run | wa_nr | wa_rp | global_nr | rmse | underpred |
|---|---:|---:|---:|---:|---:|
| v44d control | 0.219 | 0.278 | 0.351 | 0.744 | 2.43 |
| v44e `--residual-emb-pls 8` | **0.192** | 0.285 | 0.376 | 0.744 | 2.42 |
| v44f `--residual-emb-cav` | **0.244** | **0.285** | 0.382 | 0.740 | 2.41 |
| v44g raw `--emb-cav` twin | 0.209 | 0.291 | **0.413** | **0.734** | 2.40 |

Δ vs control (medium):

- residual PLS: wa_nr **−0.027** (regress) — high gain, wrong direction for locked primary
- residual CAV: wa_nr **+0.025**, wa_rp +0.007, rmse −0.004 — both strata up
- raw CAV twin: wa_nr **−0.010** (matches v23b/v32 “flat/worse”); residualization is not a no-op

## Verdict

- Harness: `--residual-emb-pls K` / `--residual-emb-cav` — **leave opt-in**
- Residual PLS: **do not promote** (wa_nr regress on smoke + medium)
- Residual CAV: consistent +wa_nr on smoke and medium vs control, and beats raw CAV twin on wa_nr — **promising**, but not full ~110k scoreboard; full-scale confirm before default
- Side fix: mention-cache batch flush now normalizes API ISO `created_at` before parquet (was ArrowTypeError on resume)
- Blocked: denser Modal Qwen3-8B desc tags (v43 follow-up) — no Modal token in this environment
