# v49 — 2nd-hop parent chain (quote-of-quote / reply-of-reply)

Same rebuild: S3 `dataset_xcomplete_full` + `--parent-chain-only` (X hop-1 from cache; hop-2 partial — HTTP 402 left 21,164 ids pending).

## Coverage / smoke

| slice | coverage |
|---|---:|
| `has_reply_chain_parent` (all rows) | 25.0% |
| among replies | 31.1% |
| `has_quote_chain_parent` (all rows) | 2.8% |
| among quotes | 19.7% |

Among-reply rate Spearman for content dims (`reply_chain_emb_cos` / `parent_cos` / `replies`) ≈ −0.01…+0.01 on chain rows; dense among-reply |ρ| for via/flag features ≲0.05 (fails ~0.06 content gate). Quote-chain global ρ≈0.08–0.09 but only 2.8% row coverage.

## Full-scale hybrid train (locked primaries)

| Run | knob | wa_nr | wa_rp | global_nr | rmse | underpred |
|---|---|---:|---:|---:|---:|---:|
| v49a | control (no chain) | **0.262** | **0.298** | **0.579** | **0.599** | 2.131 |
| v49b | `--parent-chain` | 0.253 | 0.290 | 0.573 | 0.605 | 2.154 |

Δ: wa_nr **−0.008** / wa_rp **−0.008** / global_nr −0.006 / rmse +0.005.

Chain cols do not appear in level-model top gains. Harness stays opt-in (`prepare.py --parent-chain-only`, `train.py --parent-chain`); **do not promote**. Completing remaining hop-2 X ids is unlikely to flip the dense among-reply corr gate.

Artifacts: `logs/v49_parent_chain/`.
