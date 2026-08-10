# release-v1 inference checkpoint

HF-ready bundle from X-complete full data (`dataset.parquet` ~110.6k / 934).

## Knobs

vs research default (v48b): `--no-emb-knn-te --no-provenance-features --no-domain-mean-te` (also forced by `--export-release`). Kept account TE (caller-supplied at serve), residual emb CAV, hybrid rank, exposure-rate target.

## Metrics (time holdout)

| Run | wa_nr | wa_rp | global_nr | rmse |
|---|---:|---:|---:|---:|
| **release-v1** | 0.246 | **0.307** | **0.616** | **0.588** |
| v48b (research default) | **0.262** | 0.298 | 0.579 | 0.599 |

Δ vs v48b: wa_nr −0.016 / wa_rp +0.009 / global_nr +0.037 / rmse −0.011.

## Bundle

`~/.cache/will-it-bang/release-v1/` — `model_level.txt`, `model_rank_{nr,rp}.txt`, emb PCA, TF-IDF→SVD, residual CAV, `feature_schema.json`, `constants.json`, `predict.py`, model card.

`account_mean_rate` recipe: shrink author pre-T mean exposure rate toward `global_mean≈0.151` with `m=20` (see card).

## Command

```bash
uv run train.py --export-release ~/.cache/will-it-bang/release-v1 \
  --description "release-v1: no knn/provenance/domain-TE map"
```
