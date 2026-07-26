# v48 default-stack rebase on X-complete full data (2026-07-26)

## Setup

Pulled shared S3 cache (`dataset_xcomplete_full` → `dataset.parquet`). No prepare rebuild.

| | |
|---|---|
| rows / accounts | 110,625 / 934 |
| media flags | 0% (`has_media` present; no typed media cols) |
| X parents | **complete** (among-reply / among-quote parent emb fill **100%**) |
| inbound mentions | present (default prepare path) |
| val stream | 100% |

Fair twins on the same X-complete rebuild: control = `--no-residual-emb-cav` (= prior v39d stack); treatment = default residual emb CAV (post-v47).

## Residualization / pierce (v48b)

| | |
|---|---|
| mean R²(emb \| Z) | 0.011 |
| residual RMS | 0.047 |
| train corr(`res_emb_cav_proj`, y) | 0.084 |
| hot / cold (non-reply) | 1,486 / 2,907 |
| `\|\|μh−μc\|\|` | 0.110 |
| `res_emb_cav_proj` gain rank | ~50 / 340 |

## Train (default hybrid stack)

| Run | knob | wa_nr | wa_rp | global_nr | rmse | underpred |
|---|---|---:|---:|---:|---:|---:|
| v48a | control (`--no-residual-emb-cav`) | 0.254 | 0.288 | **0.588** | **0.599** | 2.127 |
| **v48b** | `--residual-emb-cav` (default) | **0.262** | **0.298** | 0.579 | 0.599 | 2.131 |

Δ vs X-complete control: wa_nr **+0.008**, wa_rp **+0.010**, global_nr **−0.009**, rmse flat.

Cross-rebuild context (not same-rebuild):

| Run | parents | residual CAV | wa_nr | wa_rp | global_nr | rmse |
|---|---|---|---:|---:|---:|---:|
| v39d | X-complete | off | 0.254 | 0.288 | 0.588 | 0.599 |
| v47b | dump-only | on | 0.259 | 0.275 | 0.580 | 0.603 |
| **v48b** | X-complete | on | **0.262** | **0.298** | 0.579 | **0.599** |

## Notes

- Residual CAV lift shrinks vs dump-parent v47 (+0.021 → +0.008 wa_nr) once parent context is filled, but both locked primaries still rise.
- Reply within-author recovers the X-parent lift (v39d) and adds a further +0.010 from residual CAV.
- Mean within-account NR Spearman 0.199→0.253 alongside wmean — lift not only large-account weighted.
- Val remains 100% stream; media slice empty on this skip-media artifact.

## Verdict

- **Promote v48b as locked default scoreboard** (hybrid + mentions + provenance + residual emb CAV on X-complete parents).
- Keep residual emb CAV default on; `--no-residual-emb-cav` ablates to v48a / v39d-class.
- Artifacts: `logs/v48a_metrics.json`, `logs/v48b_metrics.json`, train/analyze logs; model + preds pushed to S3 cache.
