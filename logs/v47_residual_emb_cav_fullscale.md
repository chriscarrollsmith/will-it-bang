# v47 residual emb CAV full-scale confirm (2026-07-26)

## Setup

Full-scale same-rebuild after `uv run prepare.py --skip-media` (defaults).

| | |
|---|---|
| rows / accounts | 110,637 / 934 |
| media flags | 0% (skip-media reuse) |
| X parents | **dump-only** (HTTP 402 on first batch; 26,773 still pending) |
| reply / quote parent coverage | 59.0% / 10.2% |
| inbound mentions | 99.7% accounts |
| val stream | 100% |

Harness (unchanged from v44): OLS-residualize MiniLM vs `account_mean_rate` + `is_reply` + `inbound_replies_per_tweet`, then CAV (`q=0.1`, non-reply fit) → `res_emb_cav_proj` / `_sq` before emb PCA.

## Residualization / pierce

| | |
|---|---|
| mean R²(emb \| Z) | 0.011 |
| residual RMS | 0.047 |
| train corr(`res_emb_cav_proj`, y) | 0.084 |
| hot / cold (non-reply) | 1,486 / 2,907 |
| `\|\|μh−μc\|\|` | 0.110 |

## Train (default hybrid stack)

| Run | knob | wa_nr | wa_rp | global_nr | rmse | underpred |
|---|---|---:|---:|---:|---:|---:|
| v47a | control (`--no-residual-emb-cav`) | 0.238 | 0.266 | 0.570 | **0.602** | 2.145 |
| **v47b** | `--residual-emb-cav` | **0.259** | **0.275** | **0.580** | 0.603 | 2.152 |

Δ vs control: wa_nr **+0.021**, wa_rp **+0.009**, global_nr **+0.010**, rmse +0.001 (flat).

Notes:

- Mean within-account NR Spearman fell 0.228→0.209 while **wmean** rose — lift concentrated in larger accounts (locked primary is wmean).
- `res_emb_cav_proj` mid-pack gain (rank ~49/340); still moves locked wa_* under hybrid.
- Matches medium v44f direction (+0.025 wa_nr) at ~110k.

## Verdict

- **Promote / default on** (`BooleanOptionalAction`; `--no-residual-emb-cav` to ablate).
- Residual PLS stays opt-in / do not promote (v44 regress).
- Absolute scoreboard not cross-comparable to v39d (complete X parents); this run is dump-parent same-rebuild.
- Artifacts: `logs/v47a_metrics.json`, `logs/v47b_metrics.json`, train/prepare logs.
