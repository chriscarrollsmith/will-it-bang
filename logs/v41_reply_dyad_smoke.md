# v41 reply dyad + parent-author size smoke (2026-07-25)

## Setup

- Dataset: existing `~/.cache/will-it-bang/dataset.parquet` (~5.5k rows / 102 accounts; not full-scale scoreboard)
- Patch: `uv run prepare.py --dyad-only` → `dyad_cache.parquet` + parent-author size maps
- Train: default hybrid stack; ablation `--reply-dyad`

## Coverage

| slice | known parent author | prior dyad | bidir |
|---|---:|---:|---:|
| all rows (reply) | 62.7% | — | — |
| replies | **81.3%** | 74.0% of known | 55.8% of known |
| quotes | 14.6% of rows | 66.0% of known | 40.1% of known |

Parent authors resolved from dump (4,021/4,995 parents); X cache unused on this dataset. Dyad edges = reply/quote only (mentions omitted).

## Correlation vs exposure rate (replies with known parent author)

| feature | spearman(rate) | within-account wmean |
|---|---:|---:|
| `reply_dyad_outbound` | 0.017 | **0.052** |
| `reply_dyad_inbound` | 0.009 | 0.001 |
| `reply_dyad_has_prior` | −0.001 | 0.031 |
| `reply_parent_author_hist_n` | −0.001 | **0.062** |
| `reply_parent_author_inbound_per` | −0.053 | 0.045 |
| `reply_parent_replies` (existing) | 0.064 | — |
| `inbound_replies` (existing) | 0.199 | — |

Global corr fails the ~0.06 content gate; within-account signal is modest and partly level/popularity (parent size).

## Small-scale train

| run | wa_nr_wmean | wa_rp_wmean | global_nr | rmse_log |
|---|---:|---:|---:|---:|
| v41a control | 0.148 | **0.247** | 0.394 | 0.988 |
| v41b + `--reply-dyad` | **0.160** | 0.241 | **0.406** | **0.981** |

`quote_parent_author_inbound_per` appears mid-gain; reply dyad cols do not dominate. **Verdict:** dense coverage but weak target corr / flat-to-mixed wa_* on smoke — leave `--reply-dyad` **opt-in**; do not promote without a full-scale scoreboard (or a new within-author angle). Remaining backlog: agree/disagree LM, parent-author MiniLM affinity.
