# v50 — extreme LambdaRank relevance grades (viral NR under-pred)

Same rebuild: X-complete full stack (v48b/v49a control). One knob: `--rank-relevance extreme` vs default `dense`.

## Motivation

`analyze.py` on locked scoreboard: `nonreply_eng_high` under-pred frac **0.78**; viral misses are mostly already-hot authors. Under hybrid, level losses freeze `wa_*`, so attack the NR stratum ranker labels: compress mid-tier within-account order into 0–4 quantile grades (cuts 50/75/90/95) so DCG focuses on the top tail.

## Full-scale hybrid train

| Run | knob | wa_nr | wa_rp | global_nr | rmse | underpred |
|---|---|---:|---:|---:|---:|---:|
| v50a | `dense` (control) | **0.262** | **0.298** | **0.579** | **0.599** | 2.131 |
| v50b | `extreme` | 0.195 | 0.232 | 0.564 | 0.603 | 2.141 |

Δ: wa_nr **−0.067** / wa_rp **−0.066** / global_nr −0.015 / rmse +0.003 / underpred flat.

Ranker NDCG@10 also drops (NR ~0.80→0.68; reply ~0.76→0.58): collapsing mid grades destroys the pairwise order Spearman needs. Harness stays opt-in (`--rank-relevance extreme`); **do not promote**.

## Takeaway

Viral under-pred is not fixed by reweighting/compressing rank labels alone — need features that change which NR draft ranks highest within author (content / author-relative structure / link-media), not a coarser relevance scale.

Artifacts: `logs/v50_rank_relevance/`.
