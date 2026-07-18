# Will it bang? — hybrid engagement prediction

Predict a tweet's engagement from Community Archive data available at post time.

## Approach

1. **`prepare.py`** — download the CA `enriched_tweets` parquet dump, sample accounts, build leakage-safe author/structure features, embed `full_text` with frozen MiniLM, write `~/.cache/will-it-bang/dataset.parquet`.
2. **`train.py`** — time-split the dataset, train LightGBM on tabular features + embeddings to regress `log1p(favorite_count + retweet_count)`.

Retweets (`full_text` starting with `RT @`) are excluded from labels/training rows. Author history features use only strictly earlier tweets by the same account. Current follower counts are **not** used (they're a present-day snapshot → leakage).

## Setup

```bash
uv sync
uv run prepare.py          # ~1GB download + embed; cached thereafter
uv run train.py
```

Useful knobs:

```bash
uv run prepare.py --max-tweets 20000 --min-history 5
uv run train.py --val-frac 0.2
uv run train.py --skip-embeddings   # ablation: tabular only
```

## Metric

Primary: **RMSE on log engagement** (`rmse_log`), plus Spearman rank correlation. A train-mean baseline is printed for context.

## Experiment loop (optional)

After the baseline lands, iterate on `train.py` (and feature ideas in `prepare.py` when needed). Keep changes simple; avoid future leakage. Commit training code after a successful run, or roll back after a failed experiment. Always keep an experimental log.
