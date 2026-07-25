# v40 interlocutor MiniLM affinity smoke (2026-07-25)

## Setup

- Dataset: existing `~/.cache/will-it-bang/dataset.parquet` (~5.5k rows / 102 accounts; not full-scale scoreboard)
- Cache: `interlocutor_cache.parquet` via `uv run prepare.py --interlocutor-only`
- Train: default hybrid stack; ablation `--interlocutor-emb-affinity`

## Event mix (first scan, mentions included)

| kind | n |
|---|---:|
| mention | 120,716 |
| reply | 15,041 |
| quote | 2,007 |

Mentions dominate and largely duplicate tabular `inbound_mentions*`. Default pool is **reply+quote only**.

## Coverage × correlation (train fold, exposure rate)

Reply+quote pool: ~98% row coverage, mean cos≈0.29.

| feature | spearman vs rate (all) | spearman vs rate (NR) |
|---|---:|---:|
| `interlocutor_emb_cos` | −0.11 | **0.02** |
| `interlocutor_sweet_dist` | 0.10 | −0.02 |
| `interlocutor_log_n` | 0.15 | 0.24 |

Within-sample partner-corpus mean (distinct backlog reading): ~6% coverage; NR cos≈−0.12.

## Small-scale train

| run | wa_nr_wmean | wa_rp_wmean | global_nr | rmse_log |
|---|---:|---:|---:|---:|
| v40a control | 0.148 | 0.247 | 0.394 | 0.988 |
| v40b + interlocutor | 0.139 | 0.226 | 0.369 | 0.983 |

Interlocutor columns absent from top gain list. **Verdict:** fails content gate for viral NR; leave `--interlocutor-emb-affinity` opt-in; do not promote without a new angle.
