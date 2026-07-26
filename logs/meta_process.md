# Meta: experiment process / harness (refreshed 2026-07-24)

Snapshot of process lessons from `results.tsv`, `logs/`, `experiments.md`, and the Cursor experiment loop through **v35b** (hybrid + inbound mentions). Original write-up was 2026-07-19 (~through v17b/v20); this refresh folds in TE/asymmetric, emb-discovery flats, SERA, LambdaRank→hybrid, the within-account primary lock, mentions, and the CA upload/stream audit.

## Headline

Almost all honest scoreboard movement came from **problem formulation** (leakage bans, exposure-rate objective, then within-author ranking as primary) and **denser leakage-safe history / scale** — not from factorial training tricks, emb-discovery projections, or low-coverage text/LM labels. After v14e, the next real lifts were account TE + asymmetric loss (magnitude), hybrid rank blend (locked wa_*), and timestamped inbound mentions (graph density again). Autopilot “next experiment” velocity still stacked flats when candidates failed a coverage × correlation smoke test or when the primary metric was not frozen for the ablation family.

## Scoreboard eras (do not cross-compare)

| Era | Metric | Best honest mark | Notes |
|---|---|---|---|
| Dump-total (post fav/RT ban) | rmse_log | v13 **0.967** | v6–v13 graph/scale/eras |
| Exposure rate (global) | rmse_log / spearman | v20a **0.607** RMSE; Spearman ~0.44–0.46 | v14e locks rate space |
| Within-account (locked) | wa_nr / wa_rp wmean | **v35b 0.242 / 0.266** | v31b hybrid then v35b mentions; guardrails global_nr **0.579**, rmse **0.605** |

## Honest dump-total path (post fav/RT ban)

| Run | rmse_log | Lesson |
|---|---:|---|
| v6–v7 | ~1.065 | Validity reset; SVD tiny |
| v8 | ≥1.065 | Weight / breakout / nonreply-only: **no lift** |
| v9 | 0.998 | Dump-wide inbound graph: **largest model lift** |
| v10 | 0.999 | Author emb affinity: flat |
| v11 | 0.997 | Parent context: small lift |
| v12 | 0.978 | Diverse scale: lift |
| v13 | 0.967 | Time regimes: lift |
| v14a–d | — | Mis-scored rate/resid vs dump-total RMSE |
| v14e | 0.642 (rate) | New primary scoreboard (not comparable to dump-total) |

## Rate → within-account path (v14e–v35b)

| Block | Runs | Outcome | Lesson |
|---|---|---|---|
| Text/LM flats | v15–v17b | Rate RMSE stuck ~0.642–0.644 | Sparse / weak corr → doomed global RMSE |
| Media refresh | v18 | Tied RMSE; Spearman +0.014 | Keep denser flags; binary media still low gain |
| Account TE | v19 | Lift; `account_mean_rate` top gain | LOO shrink TE is the right account prior |
| Asymmetric loss | v20a | RMSE 0.639→**0.607** | Hit under-pred mass; default level loss |
| Emb kNN TE | v22b | Tiny RMSE lift; default on | PCA must only transform `emb_\d+` |
| Emb discovery grind | v23–v26 | CAV / PLS / fail-clusters flat; cluster-TE **regress** | Train corr ≠ promote; v32 rescore confirms |
| GPBoost RE | v27c | Regress vs LGBM+TE | Custom asymmetric unsupported with `gp_model`; RE σ² tiny |
| SERA | v28b / v33 | Level: +RMSE/−global rank; hybrid: wa_* frozen | Opt-in magnitude only |
| Metric locks | v28d → v31c | spearman_nr then wa_nr/wa_rp | Freeze primary before ablation family |
| LambdaRank alone | v30b | Within-account lift; global Spearman collapses | Uncalibrated across accounts |
| Hybrid rank | **v31b** | wa_nr 0.187→**0.227**, wa_rp 0.172→**0.280** | Level for cross-account; ranker for within-author order |
| Level rescore | v32 | SERA +wa_nr/−wa_rp; emb toys stay dead | Fair revisit under locked primaries |
| Inbound mentions | **v35b** | wa_nr **0.242**, global_nr **0.579**, rmse **0.605** | Dense timestamped graph again; mild wa_rp cost |

## Phase productivity

| Phase | Runs (approx) | Outcome | Lesson |
|---|---|---|---|
| Leaky climb (v2–v5) | 14 | Invalid once fav/RT history banned | Scoreboards lie if inputs leak |
| Honest rebuild (v6–v7) | 9 | RMSE ~1.065 | Correctness first |
| Ablation grind (v8) | 8 | Flat / regress | No new signal → no knobs |
| Graph + scale (v9–v13) | ~7 | 1.065 → 0.967 | First sustained honest-lift block |
| Objective reframe (v14) | 7 | a–d wasted; e locks rate | Freeze metric before ablation family |
| Text/LM flats (v15–v17b) | 4 + Modal smoke | Rate RMSE stuck | Sparse / weak corr → doomed |
| TE + asymmetric (v19–v20) | ~4 | Rate RMSE → 0.607 | Residual failure mass / under-pred |
| Emb discovery (v23–v26) | ~8 | Flat / regress | Mechanism research ≠ scoreboard |
| Ranking objective (v28–v33) | ~12 | Hybrid promotes; SERA opt-in | Match loss to locked primary |
| Mentions + CA audit (v35 + dq) | 2 + audit | Mentions default; stream provenance | Data regime > another emb toy |

## Highest-leverage moves (by kind)

1. **Validity** — Ban dump-time fav/RT history (user); cliff vs leaky v5.
2. **Signal density** — v9 inbound replies/quotes (−0.067 dump RMSE); later v35b inbound @-mentions (global_nr +0.061 / rmse −0.015 vs same-rebuild).
3. **Data** — v12 account-diverse scale (−0.019 dump RMSE).
4. **Regime / objective** — v13 eras; v14e exposure rate; v31c within-account Spearman by stratum as locked primary.
5. **Account prior + loss** — v19 TE + v20a asymmetric (rate RMSE → 0.607).
6. **Rank calibration** — v31b hybrid: keep per-account level moments, impose within-author order (largest wa_* jump).
7. **Flats / false leads** — v15–v17b sparse text/LM; v23–v26 emb CAV/PLS/fail-clusters/cluster-TE; v27 GPBoost RE; v30b LambdaRank alone; v33 hybrid×SERA (wa_* frozen).
8. **Corpus honesty** — 2026-07-24 CA audit: val is ~pure stream; null `reply_to_user_id` darkens inbound reply hist. Harness now backfills uid from parent when possible and slices wa_* by upload vs stream in `analyze.py`.

## Harness strengths

- Machine-append `results.tsv` + curated `experiments.md` backlog / “do not re-litigate”
- Incremental prepare (`--parent-only`, `--urls-only`, `--media-only`, `--mentions-only`, `--provenance-only`, `--skip-media`)
- Restore runs after ablations so defaults stay known
- Failure-mode list tied to `analyze.py` slices (now including provenance)
- Same-rebuild controls (v23a / v27b / v28a / v35a) before claiming lifts
- v32-style rescore when the primary metric changes underfoot

## Harness failure modes (observed)

- Primary metric not frozen before v14 (and again until v28d/v31c) ablation families
- Features/labels shipped without coverage × minimum-detectable-effect gate
- Viral/media residuals underused as the experiment picker until late
- Expensive Modal / large-model path before label–target pierce on a smoke set
- Early “don’t check in” autonomy + many continue cues → autopilot flats
- Emb-discovery train corr / gain rank mistaken for promote criteria (v23–v26)
- Treating pure LambdaRank as a drop-in when the locked metric still needed cross-account calibration (v30b)
- Ignoring CA ingestion regime shift (upload → stream) until the audit — feature dropout and label process looked like model failure

## Chat steering (qualitative)

Highest EV user interventions were intermittent and short: leakage policy, exposure-rate intent, display-name / fame-LM bans, harness consolidation (logs + backlog), primary-metric locks (spearman_nr → within-account), and the CA provenance audit ask. Many turns were autonomy cues; those preserved speed but needed the gate below.

## Operating rules (crystallized)

**Do more:** coverage of known levers (media, parents, graph, mentions); amortized scale; regime/objective clarity when intent shifts; re-run `analyze.py` after each real lift and pick from residuals (incl. provenance); one knob then restore; log outcome on the backlog; same-rebuild controls when prepare changes.

**Do less (until new angle):** loss/weight/breakout stacks without new features (v8); coarse author emb affinity (v10/v21); low-coverage domain means; provoke-style LM tags without corr smoke; giant rater upgrades when a calibrated smaller model is already flat; GPBoost account RE as a TE replacement (v27c); CAV / PLS / fail-clusters / cluster-TE revive without a new angle (v32); SERA as default under hybrid (wa_* frozen).

**Experiment policy (ε-greedy + gate):**

- **Exploit (~1−ε):** next run from open failure slices (nonreply / media / missing parent / viral under-prediction / stream provenance) after a fresh `analyze.py` when needed.
- **Explore (ε):** backlog novelty (new LM tag, encoder, holdout policy, …) so the loop does not starve for new signal — but still run the gate below before full spend.

Before full prepare/Modal/train:

1. If exploit: linked to a named failure slice? If explore: named backlog item + why it could pierce residuals?
2. Material row coverage (or a clear reason sparse is enough)?
3. For labels/features: cheap smoke (≤~500 rows) for coverage + target correlation / precision@k?
4. Primary metric locked for this ablation family? (now: wa_nr_wmean + wa_rp_wmean; guardrails secondary)
5. One change; restore baseline after?
6. If touching graph/history: does the change behave under stream-era nulls (reply uid, missing likes)?

If (2)–(3) fail, treat as mechanism research — not a scoreboard bet. ε is a process dial (e.g. ~1 in 5–10 runs), not a hyperparameter to grind.

## Still open (high EV)

- Viral non-reply under-prediction (improved by SERA magnitude, still open on rank). **v50** extreme LambdaRank grades regress wa_* hard — do not compress mid-tier relevance; prefer new within-author NR signal (structure / links / media).
- Quote/reply parents — **v39d** complete X fill default on (wa_rp +0.006 / wa_nr −0.009; richer context over CA-gap proxies); media vision smoke: Gemini tags fail CV gate (~0.03); SigLIP PCA-8 marginal (~0.10) — no full Gemini spend
- Regime-specific heads — **v46 calendar era TE/heads:** regress / flat under mono-era time holdout (`logs/v46_era_regime_smoke.md`); leave opt-in. Stream/upload heads / min-upload-hist still open once stream-only scoreboard is stable
- Fixed-horizon labels if CA/X ever provide time-series engagement

### v36 note (regime holdout + provenance features)

- **`upload_era` holdout** truncates the timeline but val remains ~stream → new scoreboard, wa_nr regresses; leave opt-in.
- **`--provenance-features`** (default on): same time holdout; `stream_x_log_age` carries the lift; raw `is_stream` is useless when val is constant-stream.

### v37 note (stream-fresh weights)

- Under time holdout, `--stream-fresh-days` must exceed the val-window dump-age span or train coverage is 0.
- Custom asymmetric MSE previously ignored `Dataset` weights — fixed; default uniform weights unchanged.
- With hybrid on, stream-fresh behaves like SERA: **wa_* frozen**, small global_nr wiggle, no promote.

### v38 note (media types)

- Refresh alone (v38a): wa_nr regresses vs 0% media; global_nr up.
- Type breakdown (v38b): recovers wa_nr, costs wa_rp; `photo_x_inbound_replies` is the useful cross. Leave `--media-types` opt-in.
