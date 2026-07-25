# v41c/d parent-author MiniLM corpus affinity smoke (2026-07-25)

## Setup

- Dataset: ~5.5k rows / 102 accounts (not full-scale scoreboard)
- Patch: `uv run prepare.py --dyad-only` (even-spaced `--parent-author-emb-cap 200`)
- Train: default hybrid; ablation `--reply-dyad` (includes dyad counts + parent size + corpus MiniLM)

## Coverage

| slice | `reply_parent_author_emb_has` |
|---|---:|
| all rows | 58.0% |
| replies | **75.1%** |
| quotes | 13.4% of rows |

Note: first recency-only cap only hit ~6% (older children saw empty pre-T windows). Even spacing across parent history fixed coverage.

## Correlation vs exposure rate (replies with emb_has)

| feature | spearman(rate) | within-account wmean |
|---|---:|---:|
| `reply_parent_author_emb_cos` | −0.006 | **−0.073** |
| `reply_parent_author_emb_l2` | +0.006 | +0.070 |
| `reply_parent_author_emb_log_n` | +0.106 | +0.052 |
| `reply_parent_emb_cos` (single parent) | +0.004 | +0.026 |

`log_n` pierces but overlaps parent-author size / hist volume. Content cosine fails the gate and is **anti**-aligned within author.

## Small-scale train

| run | wa_nr_wmean | wa_rp_wmean | global_nr | rmse_log |
|---|---:|---:|---:|---:|
| v41c control | **0.148** | **0.247** | **0.394** | 0.988 |
| v41d + `--reply-dyad` (counts+size+corpus emb) | 0.140 | 0.214 | 0.382 | **0.981** |

**Verdict:** dense coverage after even-space fix, but corpus MiniLM fails content gate and regresses wa_* on smoke. Keep `--reply-dyad` **opt-in**; do not promote. Agree/disagree LM stance remains open backlog (embedding stance via parent/corpus cos is exhausted here).
