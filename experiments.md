# Experiment log

Train/eval target (time holdout, honest / leakage-safe):

1. **Exposure rate (default):** `log1p(eng / (min(age_days, 30) + 1))` — intensity controlling for dump maturity.  
2. **Dump total (legacy):** `log1p(eng)` — `--no-exposure-correct --residual`.

**Primary metrics (locked):** within-account Spearman by stratum  
- `spearman_within_account_nonreply_wmean` — “is this a good standalone draft?” within author  
- `spearman_within_account_reply_wmean` — same for replies among replies  

**Guardrails / secondary:** global `spearman_nonreply` / `spearman_reply` (between-author + stratum mix), rate RMSE, `mean_underpred_nonreply_top10`. Reply-vs-NR and account selection are calibration/isolation diagnostics, not promote criteria.

Raw run stdout lives in `logs/`; machine-append metrics in `results.tsv` (gitignored).  
Process retrospective (what moved the scoreboard vs what burned cycles): [`logs/meta_process.md`](logs/meta_process.md).

**Pick next experiment (ε-greedy):** usually exploit open failure modes below (or fresh `analyze.py` residuals); with small ε explore a backlog novelty bet (new tag, loss, encoder, …). Sparse signals still need coverage + target-corr smoke before a full train. One knob → log outcome here → restore.

## Current best

| Objective | Run | wa_nr_wmean | wa_rp_wmean | global_nr | rmse_log | Notes |
|---|---|---:|---:|---:|---:|---|
| **Default stack** | **v36c** | **0.249** | **0.276** | 0.569 | **0.602** | Hybrid + mentions + provenance features (`stream_x_log_age` #21 gain) |
| Media types (opt-in) | v38b | **0.252** | 0.267 | **0.589** | 0.602 | `--media-types` after media refresh; +wa_nr/−wa_rp vs v38a — leave opt-in |
| X parents (complete) | v39d | 0.254 | **0.288** | 0.588 | **0.599** | vs v39a3: wa_nr −0.009 / wa_rp +0.006; among-reply 73%→97%, quote 70%→**99%**; **default on** |
| No-X parents control | v39a3 | **0.263** | 0.282 | **0.588** | 0.600 | Dump-only twin of v39d |
| X parents (near-complete) | v39c | 0.245 | 0.303 | 0.585 | 0.599 | 2.3k still pending 402 — superseded by v39d |
| X parents (partial) | v39b | 0.247 | 0.276 | 0.578 | 0.600 | Early 402 — superseded |
| Prior (no provenance feats) | v36a / v35b | 0.225 / 0.242 | 0.251 / 0.266 | **0.593** / **0.579** | 0.602 / 0.605 | v36a = same-rebuild control (0% media); v35b had refreshed media |
| Prior hybrid (no mentions) | v31b / v35a | 0.227 / 0.237 | 0.280 / 0.275 | 0.518 | 0.608 / 0.620 | v35a = same-rebuild control for v35b |
| Level-only (ablation) | v31a | 0.187 | 0.172 | 0.508 | 0.610 | Asymmetric without hybrid blend |
| Magnitude (opt-in SERA) | v28b / v33b | 0.227† | 0.280† | 0.483 | **0.584** | †wa_* frozen by hybrid ranker; level SERA only moves RMSE/global |
| Dump total | v13 | — | — | — | 0.967 | Legacy objective |

Rebuild: `uv run prepare.py --skip-media` (defaults are the v12 scale knobs; mentions + provenance/upload-end included).  
Train: `uv run train.py --description "…"`. Patch older datasets with `--mentions-only` / `--provenance-only`.  
Ablate provenance feats with `--no-provenance-features`. Regime holdouts: `--holdout-policy upload_era|stream_only` (opt-in; do not change locked scoreboard lightly).

## Timeline (curated)

### Leaky era (do not compare to honest scores)

Dump-time fav/RT used as author history → optimistic RMSE ~0.82 (v5). Invalid once leakage was identified.

### Honest baseline

| Run | rmse_log | spearman | Outcome |
|---|---:|---:|---|
| v6–v7 | ~1.065 | ~0.49 | Reply-received history + residual + emb PCA + TF-IDF SVD |
| v8 | ≥1.065 | — | Non-reply weight / breakout head / nonreply-only: **no lift** |
| v9 | 0.998 | 0.548 | Dump-wide inbound replies/quotes + dump parent reply stats: **big lift** |
| v10 | 0.999 | 0.545 | Pre-T author MiniLM affinity: **flat** (off by default) |
| v11 | 0.997 | 0.554 | Reply/quote parent MiniLM + pre-T parent replies: **small lift** |
| v12 | 0.978 | 0.544 | ~2× scale (929 accounts): **RMSE win**; Spearman slightly down |
| v13 | **0.967** | 0.548 | Time regimes (year / month / platform eras): **lift** |
| v14a–d | — | — | Mis-scored rate/resid against dump-total rmse; superseded by v14e framing |
| v14e | **0.642** (rate) | 0.427 | Default objective = exposure rate; primary metrics in rate space; dump decode is diagnostic only |
| v15 | 0.644 (rate) | **0.443** | VADER sentiment + first-person ratio: **flat RMSE**, small Spearman lift; not default |
| v16 | 0.643 (rate) | 0.431 | CA expanded domains + structure + train-fold domain mean: **flat** (≈7% expanded coverage) |
| v17 | 0.644 (rate) | 0.428 | Qwen2.5-1.5B wrongheaded/infuriating (non-replies): **flat**; labeler ~25–30% precision @≥7 |
| v17b | 0.643 (rate) | **0.434** | Qwen3-8B same labels: better calibrated; **flat RMSE**, small Spearman lift; still not default |
| v18 | 0.642 (rate) | **0.441** | Full CA media/url API refresh (5%→9% media): **tied RMSE**, Spearman +0.014; keep refreshed flags |
| v19 | 0.639 (rate) | **0.452** | Train-fold account target encoding (LOO + m=20 shrink): **lift**; `account_mean_rate` top gain |
| v20a | **0.607** (rate) | 0.444 | Asymmetric MSE (under-pred weight 3): **big RMSE lift**; Spearman −0.008; hot non-replies still mostly under-predicted |
| v20b | 0.647 (rate) | **0.460** | Quantile α=0.65: **worse RMSE**, small Spearman lift; not default |
| v21 | 0.609 (rate) | 0.437 | TE × own-corpus novelty (revisit v10 + account TE): **flat / slightly worse**; high gain but no metric lift |
| v22 | 0.604 (rate) | 0.442 | Emb kNN TE **buggy** (PCA dropped `emb_knn_*`; only crosses survived) — not comparable |
| v22b | **0.606** (rate) | 0.440 | Emb kNN TE (PCA fix): **tiny lift** vs v20a 0.607; `te_minus_knn` / `emb_knn_mean_rate` high gain; hot non-reply under-pred unchanged (~2.24); default on |
| v23a | 0.610 (rate) | 0.460 | Same-rebuild baseline (`prepare --skip-media`, 0% media flags): fair control for v23b; RMSE slightly above v22b (media gap) |
| v23b | 0.613 (rate) | 0.457 | Emb engagement CAV (`μ_hot−μ_cold`, non-reply decile, pre-PCA): train corr(proj,y)=0.143, `emb_cav_proj` mid-gain; **flat / slightly worse** vs v23a; hot NR under-pred 2.22→2.23; leave `--emb-cav` off |
| v24a | 0.611 (rate) | 0.452 | Emb PLS-8 beside PCA: train corr(pls_0,y)=0.168, `pls_0`/`pls_1` mid-gain; **flat** vs v23a 0.610; hot NR under-pred unchanged (~2.22) |
| v24b | 0.612 (rate) | 0.448 | Emb PLS-32 **replace** PCA: **flat / slightly worse**; more dims do not help; leave `--emb-pls` off |
| v25a | 0.611 (rate) | 0.459 | Error-driven fail clusters (OOF emb-free LGBM → k=4 MiniLM): corr(min_dist,y)=−0.04; **flat** vs v23a 0.610; hot NR under-pred unchanged (~2.21); `fail_*` not in top gains; leave `--fail-clusters` off |
| v25b | 0.612 (rate) | 0.455 | Fail clusters on TE residual: **flat / slightly worse** (0.612); same conclusion |
| v26a | 0.624 (rate) | 0.437 | Emb cluster TE k=32: train corr 0.16, `cluster_mean_rate` #3 gain; **regress** vs v23a 0.610; hot NR under-pred worse (2.29); leave `--emb-cluster-te` off |
| v26b | 0.616 (rate) | 0.444 | Emb cluster TE k=64: **still worse** (0.616); coarser k hurts less but no lift |
| v27b | 0.613 (rate) | 0.457 | Same-rebuild LGBM+account-TE+asymmetric control (`prepare --skip-media`, 0% media); hot NR under-pred 2.23 |
| v27a | 0.613 (rate) | 0.431 | LGBM `--no-account-te` (asymmetric): **flat RMSE** vs v27b; Spearman −0.026 — TE mainly helps rank |
| v27a2 | 0.642 (rate) | 0.434 | LGBM `--no-account-te --loss mse`: loss-matched floor for GPBoost; asymmetric→mse costs ~0.029 RMSE |
| v27c | 0.653 (rate) | 0.376 | GPBoost account RE (`--model gpboost`, no TE, mse): **regress** vs v27b (−0.040) and vs v27a2 (−0.011); RE σ²≈0.010; early-stop @26; hot NR under-pred worse (2.38); leave `--model gpboost` opt-in, LGBM+TE default |
| v27d | 0.640 (rate) | 0.344 | GPBoost RE **plus** account TE: ≈ loss-matched floor on RMSE, Spearman collapses; double-counting does not help |
| v28a | 0.610 (rate) | 0.460 | Same-rebuild asymmetric control (`prepare --skip-media`); hot NR under-pred 2.22 |
| v28b | **0.590** (rate) | 0.409 | SERA (`--loss sera`, floor=0.05, high-only): **RMSE lift** vs v28a (−0.020); hot NR under-pred 2.22→2.04; Spearman 0.460→0.409 / NR 0.508→0.474 — keep **opt-in** (rank primary) |
| v28d | — | — | Policy: lock primary = `spearman_nonreply`; revert default loss to asymmetric; SERA stays `--loss sera` |
| v28c | 0.642 (rate) | 0.462 | MSE floor: worse RMSE + worse under-pred (2.36) — SERA lift is not “just mse” |
| v29a | 0.610 (rate) | 0.460 | RMSE early-stop control (= v28a); spearman_nr 0.508; hot NR under-pred 2.22; best_iter 263 |
| v29b | 0.611 (rate) | 0.462 | Spearman_nr early-stop: spearman_nr **0.509** (+0.0016); under-pred 2.22 (flat); best_iter 342; **tiny / leave opt-in** |
| v30a | 0.610 (rate) | 0.460 | Asymmetric control for LambdaRank (= v29a); spearman_nr 0.508; within-account NR Spearman mean 0.161 / wmean 0.187 |
| v30b | 1.016 (rate) | 0.097 | LambdaRank account groups (non-reply): **global spearman_nr collapses** 0.508→0.051 (scores not cross-account calibrated); within-account NR Spearman mean 0.168 / wmean **0.211** (+0.024); NDCG@10≈0.77; leave **opt-in** / fail on locked global primary |
| v31a | 0.610 (rate) | 0.460 | Asymmetric control with new primaries: wa_nr_wmean **0.187**, wa_rp_wmean **0.172**; global_nr 0.508; under-pred 2.22 |
| v31b | 0.608 (rate) | 0.490 | Hybrid `--hybrid-rank`: wa_nr_wmean **0.227** (+0.040), wa_rp_wmean **0.280** (+0.108); global_nr 0.518; under-pred 2.21; **promote / new default** |
| v31c | — | — | Policy: lock primary = within-account Spearman by stratum (NR + reply); global Spearman secondary |
| v32 | — | — | Rescore shortlist on level-only stack (`--no-hybrid-rank`) vs control wa_nr **0.187** / wa_rp **0.172** / rmse 0.610 / global_nr 0.508. Artifacts: `logs/rescore_within_account/`. **SERA:** wa_nr **0.204** (+0.017), wa_rp **0.140** (−0.032), rmse **0.590**, global_nr 0.474 — NR within-author + magnitude help, reply within-author and global rank hurt; stay opt-in. **PLS-8:** wa_nr +0.008 / wa_rp −0.012 / rmse flat — marginal, leave opt-in. **fail-clusters:** noise. **CAV / author affinity:** flat-to-worse. **cluster TE:** still regresses (wa_nr −0.037, wa_rp −0.057). Do not revive CAV/affinity/cluster-TE without a new angle |
| v33a | 0.608 (rate) | 0.490 | Hybrid asymmetric control (= v31b): wa_nr **0.227**, wa_rp **0.280**, global_nr 0.518 |
| v33b | **0.584** (rate) | 0.460 | Hybrid × SERA: wa_* **unchanged** (ranker sets within-author order); RMSE −0.024 / under-pred 2.21→2.04; global_nr 0.518→0.483 — magnitude-only; **do not promote** |
| v33c | 0.601 (rate) | 0.474 | Hybrid × SERA `--sera-nonreply-only`: same frozen wa_*; milder RMSE/global trade vs v33b |
| v33d | 0.604 (rate) | 0.453 | Level SERA NR-only: wa_nr **0.138** (−0.049 vs level control), wa_rp 0.170 (flat); **regress** on locked primary — NR-stratum SERA fails |
| v35a | 0.620 (rate) | 0.503 | Same-rebuild hybrid control **without** mention cols (0% media): wa_nr **0.237**, wa_rp **0.275**, global_nr 0.518; under-pred 2.22 |
| v35b | **0.605** (rate) | **0.521** | Timestamped inbound @-mentions (`user_mentions` ⋊ tweet time; 99.7% accounts / 98.7% rows): wa_nr **0.242** (+0.006), wa_rp **0.266** (−0.009), global_nr **0.579** (+0.061), under-pred 2.15; smoke NR spearman(mentions,rate)≈0.55; `reply_x_inbound_mentions` / `inbound_mentions_per_tweet` top gains; **keep default** (NR + magnitude/global win; mild reply within-author cost) |
| v36a | **0.602** (rate) | **0.521** | Same-rebuild control after reply-uid backfill + upload-end cols (0% media via `--skip-media`): wa_nr **0.225**, wa_rp **0.251**, global_nr **0.593**; val 100% stream |
| v36b | 0.482† (rate) | 0.459† | `--holdout-policy upload_era` (cap ≤ last upload-linked `created_at` 2026-03-22): wa_nr **0.143** (−0.082), wa_rp **0.299** (+0.049); †different val — **do not promote** / keep `time` default |
| v36c | **0.602** (rate) | **0.521** | `--provenance-features` on time holdout: wa_nr **0.249** (+0.024), wa_rp **0.276** (+0.025), global_nr 0.569 (−0.024); `stream_x_log_age` #21 gain; raw `is_stream` zero gain on ~pure-stream val; **promote / default on** |
| v36d | 0.615† (rate) | 0.516† | `--holdout-policy stream_only`: wa_nr 0.239 / wa_rp 0.276; †near-identical corpus (dataset already 96.7% stream) — diagnostic only |
| v37a2 | **0.602** (rate) | **0.521** | Control after asymmetric sample-weight fix (= v36c; uniform weights unchanged) |
| v37b2 | 0.603 (rate) | 0.521 | `--stream-fresh-days 90`: wa_* **frozen** (hybrid ranker); global_nr 0.569→**0.577**; rmse 0.602→0.603; under-pred flat; **leave opt-in** |
| v37d | 0.604 (rate) | 0.521 | `--stream-fresh-days 180` (35% train affected): wa_* frozen; global_nr 0.575; rmse worse — no promote |
| v37c | — | — | `--stream-fresh-days 30`: **no-op** under time holdout (train ages ≥~37d); harness now warns |
| v38a | **0.600** (rate) | 0.521 | Media refresh 0%→9% (binary flags only): wa_nr **0.234** (−0.014 vs v36c), wa_rp 0.279, global_nr **0.587** |
| v38b | 0.602 (rate) | 0.521 | `--media-types` (photo 7.7% / video 0.8% / gif 0.6%): wa_nr **0.252** (+0.018 vs v38a), wa_rp **0.267** (−0.013); `photo_x_inbound_replies` #33; **leave opt-in** |

### v37 stream-fresh weights (same rebuild; after asymmetric weight fix)

| Run | fresh_days | wa_nr | wa_rp | global_nr | rmse | Notes |
|---|---:|---:|---:|---:|---:|---|
| v37a2 control | 0 | 0.249 | 0.276 | 0.569 | 0.602 | Uniform weights; matches v36c |
| v37b2 | 90 | 0.249 | 0.276 | **0.577** | 0.603 | 9.5k train rows affected (mean_mult 0.97); wa_* frozen |
| v37d | 180 | 0.249 | 0.276 | 0.575 | 0.604 | 31k affected (mean_mult 0.90); still no wa_* move |
| v37c | 30 | 0.249 | 0.276 | 0.569 | 0.602 | 0 affected — time-holdout no-op |

Bugfix: custom asymmetric MSE ignored `Dataset` sample weights (LightGBM does not apply them to Python objectives). Pre-fix v37b matched control bit-for-bit despite n_affected>0. Fix is neutral when weights are uniform (default stack). `--nonreply-weight` under asymmetric was similarly inert before the fix.

### v39 X parent fallback (same rebuild)

Full prepare `--skip-media` (~110k / 934 accounts). Resumable X lookup; sanitize `?s=` junk so bad ids cannot 400 a batch.

| Run | fill | wa_nr | wa_rp | global_nr | rmse | Notes |
|---|---|---:|---:|---:|---:|---|
| v39a3 no-X | dump | **0.263** | 0.282 | 0.588 | 0.600 | Control |
| v39b partial-X | 2.9k/26.7k | 0.247 | 0.276 | 0.578 | 0.600 | Early 402; superseded |
| v39c near-X | 22.8k ok; 2.3k pending | 0.245 | 0.303 | 0.585 | 0.599 | Mid-credit stop |
| **v39d complete-X** | **24,988 ok** / 969 forb / 816 nf | 0.254 | **0.288** | **0.588** | **0.599** | among-reply 73%→97%, quote 70%→**99%** |

**Verdict:** **default `--x-parents` on.** Complete fill softens the NR dip vs partial/near (−0.009 vs −0.018) with a small reply lift and flat/better magnitude. Prefer richer context over CA-gap proxies.

### v38 media types (same rebuild after `--media-only`)

| Run | wa_nr | wa_rp | global_nr | rmse | Notes |
|---|---:|---:|---:|---:|---|
| v36c (0% media) | 0.249 | **0.276** | 0.569 | 0.602 | Prior default (skip-media) |
| v38a binary media | 0.234 | 0.279 | 0.587 | **0.600** | Refresh only |
| v38b + types | **0.252** | 0.267 | **0.589** | 0.602 | Opt-in `--media-types` |

### v36 regime holdout / provenance features (same rebuild)

| Run | wa_nr | wa_rp | global_nr | rmse | Notes |
|---|---:|---:|---:|---:|---|
| v36a time control | 0.225 | 0.251 | **0.593** | 0.602 | Reply-uid densification + upload-end metadata |
| v36b upload_era | **0.143** | **0.299** | 0.590 | 0.482† | Caps timeline; val still ~100% stream; hurts wa_nr |
| v36c + provenance feats | **0.249** | **0.276** | 0.569 | 0.602 | Promote; `--no-provenance-features` to ablate |
| v36d stream_only | 0.239 | 0.276 | 0.567 | 0.615† | Diagnostic (dataset already stream-dominated) |

### v35 inbound mentions (same rebuild)

| Run | wa_nr | wa_rp | global_nr | rmse | Notes |
|---|---:|---:|---:|---:|---|
| v35a no mentions | 0.237 | **0.275** | 0.518 | 0.620 | Control |
| v35b + mentions | **0.242** | 0.266 | **0.579** | **0.605** | Promote / default on |

### Rescore shortlist (level-only, locked primaries)

Fair revisit of prior ablations after the metric lock. Control = asymmetric level model (`--no-hybrid-rank`), not v31b hybrid.

| Run | Δwa_nr | Δwa_rp | Δrmse | Δglobal_nr | Verdict |
|---|---:|---:|---:|---:|---|
| sera | **+0.017** | **−0.032** | **−0.021** | −0.034 | Helps NR within-author + magnitude; hurts reply within-author — opt-in only |
| emb_pls8 | +0.008 | −0.012 | ~0 | −0.003 | Tiny NR bump; leave opt-in |
| fail_clusters | +0.003 | −0.003 | ~0 | −0.003 | Noise / flat |
| emb_cav | −0.004 | −0.006 | +0.002 | −0.008 | Flat/worse — do not revive |
| author_emb_affinity | −0.009 | −0.012 | +0.002 | −0.007 | Flat/worse — do not revive |
| emb_cluster_te | **−0.037** | **−0.057** | +0.014 | −0.025 | Still regresses |

### v33 SERA × hybrid / NR-stratum

| Run | wa_nr | wa_rp | global_nr | rmse | Notes |
|---|---:|---:|---:|---:|---|
| v33a hybrid asym | 0.227 | 0.280 | 0.518 | 0.608 | Control |
| v33b hybrid sera | 0.227 | 0.280 | 0.483 | **0.584** | wa_* frozen; better magnitude |
| v33c hybrid sera NR | 0.227 | 0.280 | 0.494 | 0.601 | Same; milder tradeoff |
| v33d level sera NR | **0.138** | 0.170 | 0.469 | 0.604 | Primary regress vs level control 0.187 |

## Failure modes (still open)

- High-engagement **non-replies** (viral under-prediction); often already-hot authors. Improved by SERA (v28b: mean under-pred 2.22→2.04) but still open.
- **Media** / short+image posts — API refresh restores ~9% coverage (v18/v38a); binary-only refresh can regress wa_nr vs 0% media; type breakdown (v38b) recovers NR at reply cost — `--media-types` opt-in. Vision smoke: Gemini tags fail CV gate; SigLIP PCA-8 marginal — no full Gemini spend.
- **Quote** / reply parents missing from CA — X fallback **complete** (**v39d**): among-reply 73%→97%, quote 70%→99%; wa_rp +0.006 / wa_nr −0.009 / rmse flat; keep default on.
- Reply **dyad** / parent-author size — **v41 smoke** dense coverage but weak rate corr; wa_rp flat on small set; `--reply-dyad` stays opt-in (`logs/v41_reply_dyad_smoke.md`).
- Parent-author **corpus MiniLM** — **v41d** 75% reply coverage but cos anti-aligned within-author; wa_* regress; stays under opt-in dyad (`logs/v41_parent_author_emb_smoke.md`).
- Reply **LM stance** (agree/disagree vs parent) — **v42** 1.5B smoke poorly calibrated; wa_rp regress under sparse join; `--stance-labels` opt-in (`logs/v42_stance_smoke.md`).
- Richer **descriptive LM tags** (humor/controversy/narrative/CTA/news/dunk) — **v43** 1.5B NR smoke: humor/narrative ρ≈0.14 (gate pass); overconfident ≥7 on humor/controversy; sparse train +wa_nr +0.016 / wa_rp frozen; `--desc-labels` opt-in (`logs/v43_desc_tags_smoke.md`). Revisit denser/8B before promote.

---

## Backlog — not yet explored

Check off when tried; note outcome in one line.

### Time / regimes
- [x] Calendar / era features (year, month seasonality, platform eras) — **v13: 0.978 → 0.967**
- [x] Exposure-rate target as default objective — **v14e: rate rmse 0.642** (vs train-mean 0.701); dump-total decode intentionally weaker
- [ ] Train-fit regime-specific baselines or separate heads by era
- [x] Explicit “CA corpus quality over time” proxies (upload density, archive coverage) — **audit 2026-07-24:** stream (`archive_upload_id` null) is 62% of 2025 / 98% of 2026 dump rows; latest time-quintile on eligible accounts is ~100% stream with ~73% replies missing `reply_to_user_id` (breaks inbound `reply_to_user` hist). Likes/followers only from archive uploads (extension does not capture). Report: `logs/ca_data_quality_audit.md`; re-run `uv run audit_ca_quality.py`.
- [x] Provenance-aware eval: `analyze.py` slices wa_* by upload-linked vs stream (`is_stream` from null `archive_upload_id`; carried in prepare / joined from dump if missing).
- [x] Backfill `reply_to_user_id` from parent `account_id` when parent ∈ dump — in `load_author_graph` (inbound hist) + sampled-row coalesce for `is_self_reply`; rebuild densifies `inbound_replies*` (v36a).
- [x] Holdout / train policy by ingestion regime — **v36b `upload_era`:** wa_nr regresses on new val (still ~stream); **v36d `stream_only`:** ≈ time on today’s corpus; keep `--holdout-policy time` default; knobs remain for regime-specific scoreboards. `--min-upload-hist` still unused.
- [x] Optional train feature `is_stream` / hours since pre-T `archive_upload.end_date` — **v36c: promote / default on** (`stream_x_log_age` #21; wa_nr +0.024 / wa_rp +0.025 / global_nr −0.024 vs v36a); `--no-provenance-features` to ablate
- [x] Label honesty for stream eng via `--stream-fresh-days` — **v37b2/d:** under hybrid, wa_* frozen (ranker dominates); mild global_nr lift / flat-to-worse RMSE; leave **opt-in**. Note: d must exceed val-window span (~37d) or train coverage is 0; asymmetric custom obj now honors sample weights
- [ ] Fixed-horizon labels if CA/X ever provide time-series engagement (cleaner maturity control)
- [ ] Do **not** build liker / likes-graph features until CA streams likes with usable timestamps (extension omits likes; upload likes undated) — revisit only if #427 lands or a dated source appears

### Language-model labels (post-time text only)
- Prefer descriptive tags of tweet text (sentiment, humor, controversy, narrative, CTA, gender presentation). Avoid fame/quality/influence scores on text or handles — cutoff/reputation leakage risk.
- [x] Lexical VADER sentiment + first-person ratio — **v15: flat rate RMSE** (0.644 vs 0.642); Spearman +0.016
- [x] Richer descriptive tags: humor, controversy, personal narrative, CTA, news, dunk/quote-bait — **v43 smoke** (`logs/v43_desc_tags_smoke.md`): Qwen2.5-1.5B local 500 NR; humor/narrative spearman(rate)≈0.14 (gate pass); humor/controversy overconfident (≥7 ~45%); train wa_nr 0.148→0.165 / wa_rp frozen / rmse −0.007. Harness: `label_desc.py --local` / Modal Qwen3-8B; train `--desc-labels` — **leave opt-in**; denser+8B before scoreboard
- [x] **Wrongheaded / mildly infuriating** LM labels — **v17** (Qwen2.5-1.5B): flat + noisy labeler; **v17b** (Qwen3-8B): better calibrated (inf≥7 ≈3.7%), still **flat rate RMSE** (0.643 vs 0.642), Spearman 0.434; Qwen3.6-27B smoke OK but ~20× slower one-at-a-time
- [ ] Breakout-likelihood score from a small LM head (text-only; no user fame priors; treat as higher leakage risk)
- [ ] Thread / reply-intent classification
- [ ] Do **not** use display name; avoid username as a feature or LM input

### Links & external context
- [x] Domain priors from CA `tweet_urls` + text-url structure — **v16: flat rate RMSE** (0.643 vs 0.642); only ~7% rows have expanded domains
- [ ] Fetch OG title/description for expanded URLs (cache aggressively; respect rate limits)
- [x] Link-vs-commentary structure (`url_only`, `url_at_end`, `n_text_urls`) — included in v16; low gain

### Vision / media
- [x] Full CA media/url API pass on the scaled set — **v18: tied rate RMSE** (0.642), Spearman 0.427→0.441; coverage 5%→9% media / 4%→7% ca_url (`prepare.py --media-only`)
- [x] Image caption / vision emb smoke — **`uv run vision_smoke.py --limit 500`** (`logs/vision_smoke.md`): download 88%; Gemini+SigLIP coverage 92%; cost ~$0.83. Gemini tag CV ridge spearman(rate) **0.028** (best singles controversy/realworld ~0.09); SigLIP CV ridge 0.067 / PCA-8 **0.101**. **Do not full-spend Gemini** (fails coverage×corr gate); SigLIP PCA emb is a marginal opt-in backlog item, not a promote.
- [x] Media-type breakdown (photo / gif / video) — **v38b:** vs binary refresh wa_nr +0.018 / wa_rp −0.013; `photo_x_inbound_replies` mid-gain; leave `--media-types` **opt-in** (cache `media_cache.parquet`)

### Graph / archive denser still
- [x] Timestamped mention joins (via dump tweet join; `user_mentions` alone has no time) — **v35b: keep default**; vs same-rebuild wa_nr +0.006 / wa_rp −0.009 / rmse −0.015 / global_nr +0.061; `prepare.py --mentions-only`
- [x] X API fallback for missing reply/quote parents — resumable `x_parent_cache.parquet` (batch flush; snowflake sanitize; no refetch of terminal statuses). **v39d complete:** cache 26,773 (24,988 ok / 969 forbidden / 816 not_found; **0 pending**); among-reply **73.4%→97.4%**, quote **70.4%→98.9%**; vs dump-only wa_nr 0.263→0.254, wa_rp 0.282→0.288, rmse 0.600→0.599 — **default on** (`--no-x-parents` to ablate).
- [ ] Quote-of-quote / deeper parent chain (1–2 hops, still pre-T)
- Note: CA `followers`/`following` are archive **snapshots** (no follow/unfollow event times) — not usable as pre-T graph history.

### Reply dyad / parent context
Mostly helps replies (not the main under-prediction bucket); still cheap if parent/dyad coverage exists.
- [x] Prior reciprocal interaction (reply/quote edges; mentions omitted) + leakage-safe parent-author size — **v41 smoke** (`logs/v41_reply_dyad_smoke.md`): among-reply parent-author **81%**, prior_dyad **74%** / bidir **56%**; global spearman(rate)≲0.02 (fails ~0.06 content gate); within-account outbound/hist_n ~0.05–0.06; small-scale train wa_nr +0.012 / wa_rp −0.006 / rmse −0.007. Harness: `prepare.py --dyad-only` + `train.py --reply-dyad` — **leave opt-in**
- [x] Agree vs disagree with the parent tweet (LM text-only) — **v42 smoke** (`logs/v42_stance_smoke.md`): Qwen2.5-1.5B local 508 pairs; agree overconfident (p50=8, ≥7 57%); agree spearman(rate)=0.057 / conflict 0.17; train wa_rp 0.247→0.221 with sparse join. Harness: `label_stance.py --local` / Modal Qwen3-8B; train `--stance-labels` — **leave opt-in**; revisit only with calibrated denser labels
- [x] Leakage-safe parent-tweet heat — already in stack as `reply_parent_replies` / `quote_parent_replies` (v11+)
- [x] Reply earliness — already in stack as `reply_parent_age_hours` / `quote_parent_age_hours` (log1p hours)
- [x] Semantic similarity to **parent author’s** pre-T MiniLM corpus mean — **v41d smoke** (`logs/v41_parent_author_emb_smoke.md`): among-reply emb_has **75%** (even-spaced cap=200); corpus cos spearman≈0 / wa **−0.07**; train wa_nr 0.148→0.140 / wa_rp 0.247→0.214 — **regress**; leave under `--reply-dyad` opt-in / do not promote

### Standalone / author–audience corpus affinity
Limited by CA completeness; affinity to own history was flat as coarse v10 mean.
- [x] Richer own-corpus similarity × account TE (novelty, te_x_novelty, 30d, nonreply) — **v21: flat** (0.609 vs 0.607); **v32 rescore:** still worse on wa_* — leave `--author-emb-affinity` off
- [x] Similarity to corpus of users who previously interacted with you (pre-T reply/quote interlocutor MiniLM): **v40 smoke** — reply/quote pool row coverage ~98%; NR spearman(cos, rate)≈0.02 / sweet_dist≈0 (fails content gate); `log_n` / partner counts pierce but overlap inbound graph volume. Mentions drown the pool (120k/138k) and duplicate `inbound_mentions` — default off. Within-sample partner-corpus mean only ~6% coverage on small set. Harness: `prepare.py --interlocutor-only` + `train.py --interlocutor-emb-affinity` (opt-in; do not promote without a new angle)

### Embedding discovery (vs arbitrary LM tags)
Discover engagement-related structure in MiniLM (or a larger frozen encoder) from the *data*, not hand-picked labels. Fit only on the train time fold; apply to val. Contrast with v15–v17b descriptive tags and with unsupervised emb PCA (already in the stack).

Cheap first bets (smoke: train-fold corr of 1–8 derived dims vs rate; then full train):
- [x] **Engagement direction / difference-of-means CAV:** — **v23b: flat / slightly worse** (0.613 vs v23a 0.610); **v32 rescore:** flat/worse on wa_*; `--emb-cav` stays opt-in / do not revive
- [x] **PLS supervised linear projection:** — **v24a/b** flat on RMSE; **v32 rescore:** PLS-8 wa_nr +0.008 / wa_rp −0.012 — marginal; `--emb-pls` opt-in. Linear emb→rate axis family exhausted for now
- [x] **Cluster TE:** — **v26a/b: regress**; **v32 rescore:** still regresses on wa_* (−0.037 / −0.057); `--emb-cluster-te` stays off. Cheap emb-discovery lane exhausted

Heavier / researchy:
- [ ] Residual emb after regressing out account TE / reply baseline, then redo direction or PLS (isolates *content* signal from account heat)
- [x] Error-driven clusters: — **v25a/b: flat**; **v32 rescore:** noise on wa_*; `--fail-clusters` stays opt-in
- [ ] Supervised manifold (UMAP on RF-leaf similarity, etc.) — usually for analysis, not tabular features; only promote if a cheap projection beats PLS

### Modeling
- [x] Quantile / pinball or underprediction-asymmetric loss — **v20a asymmetric: 0.639→0.607 rate RMSE** (default); **v20b quantile α=0.65: worse RMSE** (0.647), slight Spearman lift
- [x] SERA / relevance-weighted extreme loss — **v28b/v32** level: +wa_nr/−wa_rp; **v33b** hybrid×SERA: wa_* frozen, RMSE↑/global↓; stay **opt-in** for magnitude only
- [x] Early-stop on `spearman_nonreply` — **v29b: tiny lift** (0.508→0.509); guardrails flat; `--early-stop-metric spearman_nonreply` opt-in (default stays rmse)
- [x] Pairwise / listwise LambdaRank on non-replies by `account_id` — **v30b: fail on global spearman_nr** (0.051); slight **within-account** rank lift (wmean 0.187→0.211); `--loss lambdarank` opt-in only
- [x] Hybrid level + stratum within-author ranker — **v31b: lift** wa_nr 0.187→0.227, wa_rp 0.172→0.280; guardrails flat/up; **default on** (`--no-hybrid-rank` to ablate)
- [x] SERA on NR stratum / hybrid×SERA — **v33:** hybrid freezes wa_*; level `--sera-nonreply-only` **regress** wa_nr 0.187→0.138; leave opt-in, no promote
- [ ] Larger frozen encoder (Modal credits) vs MiniLM
- [ ] Further scale (>120k / more accounts with lower `min-account-tweets`)
- [x] Account-level **target encoding** (train-fold LOO, leakage-safe) — **v19: 0.642 → 0.639 rate RMSE**, Spearman 0.441→0.452; default on (`--no-account-te` to ablate)
- [x] True account **random effects** via GPBoost — **v27c: regress** (0.653 vs v27b 0.613; also worse than mse floor 0.642); custom asymmetric loss unsupported with `gp_model`; RE variance tiny on full stack; keep `--model gpboost` opt-in only
- [x] Emb kNN train-fold TE (content prior for cold-start accounts) — **v22b: 0.607 → 0.606**; default on (`--no-emb-knn-te` to ablate); PCA must only transform `emb_\d+`

### Tried — do not re-litigate without a new angle
- [x] Non-reply sample weight / breakout OOF head / nonreply-only train (v8: no lift)
- [x] Author emb affinity cosine/L2 to pre-T mean (v10: flat) — see richer revisit under Standalone
- [x] Dump inbound graph + parent context + scale (v9–v12: current stack)
