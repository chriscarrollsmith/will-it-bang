# Will it bang? — hybrid engagement prediction

Predict a tweet's engagement from Community Archive data available at post time.

## Approach

1. **`prepare.py`** — download the CA `enriched_tweets` parquet dump, sample accounts, build leakage-safe author/structure features, embed `full_text` with frozen MiniLM, write `~/.cache/will-it-bang/dataset.parquet`.
2. **`train.py`** — time-split the dataset, train LightGBM on tabular features + embeddings to regress `log1p(favorite_count + retweet_count)`.

Retweets (`full_text` starting with `RT @`) are excluded from labels/training rows. Author history uses only strictly earlier tweets; dump-time fav/RT are **labels only**, never history inputs (use pre-T reply arrivals instead). Current follower counts are **not** used.

## Setup

```bash
uv sync
uv run prepare.py          # ~1GB download + embed; cached thereafter
uv run train.py
```

Useful knobs:

```bash
uv run prepare.py --skip-media          # reuse media flags; skip slow CA API
uv run prepare.py --parent-only         # re-join reply/quote parents only
uv run prepare.py --urls-only           # CA expanded URL domains + text-url structure
uv run prepare.py --mentions-only       # timestamped inbound @-mentions (user_mentions ⋊ tweet time)
uv run modal run label_provoke.py --nonreply-only   # default: Qwen3-8B → provoke_cache_Qwen__*
uv run train.py --provoke-labels --provoke-cache ~/.cache/will-it-bang/provoke_cache_Qwen__Qwen3-8B.parquet
uv run train.py --skip-embeddings       # ablation: tabular only
```

## Metric

Primary (default): **RMSE / Spearman on exposure-adjusted log rate**
`log1p(eng / (min(age_days, 30) + 1))` — intensity controlling for dump maturity.  
Diagnostic: `rmse_log_raw_dump` decodes rate predictions back to dump-total `log1p(eng)`.  
For the older dump-total objective: `uv run train.py --no-exposure-correct --residual`.

## Experiment log

Curated history, current best, and the **unexplored backlog** live in [`experiments.md`](experiments.md).  
Raw stdout goes under `logs/`; `results.tsv` is the machine-append metric table (gitignored).

CA upload-vs-stream data-quality audit: [`logs/ca_data_quality_audit.md`](logs/ca_data_quality_audit.md) (`uv run audit_ca_quality.py`).
