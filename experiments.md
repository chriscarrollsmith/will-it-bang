# Experiment log

Two scoreboards (both time holdout, honest / leakage-safe):

1. **Exposure rate (default):** `log1p(eng / (min(age_days, 30) + 1))` — intensity controlling for dump maturity.  
2. **Dump total (legacy):** `log1p(eng)` — `--no-exposure-correct --residual`.

Raw run stdout lives in `logs/`; machine-append metrics in `results.tsv` (gitignored).  
Process retrospective (what moved the scoreboard vs what burned cycles): [`logs/meta_process.md`](logs/meta_process.md).

**Pick next experiment (ε-greedy):** usually exploit open failure modes below (or fresh `analyze.py` residuals); with small ε explore a backlog novelty bet (new tag, loss, encoder, …). Sparse signals still need coverage + target-corr smoke before a full train. One knob → log outcome here → restore.

## Current best

| Objective | Run | rmse_log | spearman | train / val | Notes |
|---|---|---:|---:|---|---|
| **Exposure rate** | **v22b** | **0.606** | 0.440 | 88k / 22k | Emb kNN TE + asymmetric + account TE; tiny RMSE lift vs v20a |
| Dump total | v13 | 0.967 | 0.548 | 88k / 22k | Graph + parent + scale + time regimes |

Rebuild: `uv run prepare.py --skip-media` (defaults are the v12 scale knobs).  
Train: `uv run train.py --description "…"`.

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

## Failure modes (still open)

- High-engagement **non-replies** (viral under-prediction); often already-hot authors.
- **Media** / short+image posts — API refresh restored ~9% coverage (v18); binary flags still low gain vs interactions/`hist_frac_media`.
- **Quote** tweets when parent missing from CA dump (~10% quote-parent coverage).

---

## Backlog — not yet explored

Check off when tried; note outcome in one line.

### Time / regimes
- [x] Calendar / era features (year, month seasonality, platform eras) — **v13: 0.978 → 0.967**
- [x] Exposure-rate target as default objective — **v14e: rate rmse 0.642** (vs train-mean 0.701); dump-total decode intentionally weaker
- [ ] Train-fit regime-specific baselines or separate heads by era
- [ ] Explicit “CA corpus quality over time” proxies (upload density, archive coverage)
- [ ] Fixed-horizon labels if CA/X ever provide time-series engagement (cleaner maturity control)

### Language-model labels (post-time text only)
- Prefer descriptive tags of tweet text (sentiment, humor, controversy, narrative, CTA, gender presentation). Avoid fame/quality/influence scores on text or handles — cutoff/reputation leakage risk.
- [x] Lexical VADER sentiment + first-person ratio — **v15: flat rate RMSE** (0.644 vs 0.642); Spearman +0.016
- [ ] Richer descriptive tags: humor, controversy, personal narrative, CTA, news, dunk/quote-bait (self-hosted LM)
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
- [ ] Image caption or vision embedding for tweets with media URLs
- [ ] Media-type breakdown (photo / gif / video) if available

### Graph / archive denser still
- [ ] Timestamped mention joins (via dump tweet join; `user_mentions` alone has no time)
- [ ] X API fallback for missing reply/quote parents
- [ ] Quote-of-quote / deeper parent chain (1–2 hops, still pre-T)
- Note: CA `followers`/`following` are archive **snapshots** (no follow/unfollow event times) — not usable as pre-T graph history.

### Reply dyad / parent context
Mostly helps replies (not the main under-prediction bucket); still cheap if parent/dyad coverage exists.
- [ ] Prior reciprocal interaction: has this author and the parent author exchanged replies/mentions before *T*? (likely re-interact)
- [ ] Agree vs disagree with the parent tweet (LM or reply−parent embedding/stance; text-only)
- [ ] Leakage-safe “size” of parent author (pre-T proxies only: prior tweet count, inbound reply rates — never dump `num_followers`)
- [ ] Leakage-safe parent-tweet heat (pre-T reply/quote arrivals on the parent — not dump fav/RT; partly in v11 parent_replies)
- [ ] Reply earliness: hours since parent `created_at` (halo if early on a tweet that later goes viral; parent maturity is post-time observable)
- [ ] Semantic similarity of reply text to the **parent author’s** pre-T MiniLM corpus mean (distinct from cosine to the single parent tweet)

### Standalone / author–audience corpus affinity
Limited by CA completeness; affinity to own history was flat as coarse v10 mean.
- [x] Richer own-corpus similarity × account TE (novelty, te_x_novelty, 30d, nonreply) — **v21: flat** (0.609 vs 0.607); leave `--author-emb-affinity` off
- [ ] Similarity to corpus of users who previously interacted with you (pre-T reply/mention partners’ mean MiniLM): expect non-linear “not too similar, not too foreign” effect (include quadratic / distance-to-sweet-spot features)

### Embedding discovery (vs arbitrary LM tags)
Discover engagement-related structure in MiniLM (or a larger frozen encoder) from the *data*, not hand-picked labels. Fit only on the train time fold; apply to val. Contrast with v15–v17b descriptive tags and with unsupervised emb PCA (already in the stack).

Cheap first bets (smoke: train-fold corr of 1–8 derived dims vs rate; then full train):
- [x] **Engagement direction / difference-of-means CAV:** — **v23b: flat / slightly worse** (0.613 vs v23a 0.610); train corr 0.14 but redundant with emb PCA + knn TE; `--emb-cav` stays opt-in
- [x] **PLS supervised linear projection:** — **v24a** PLS-8 beside PCA: flat (0.611 vs 0.610); **v24b** PLS-32 replace PCA: flat/worse (0.612); train corr 0.17 + mid-gain but no holdout lift; `--emb-pls` stays opt-in. Linear emb→rate axis family exhausted for now
- [x] **Cluster TE:** — **v26a/b: regress** (k=32 → 0.624, k=64 → 0.616 vs 0.610); high train corr/gain but fights account TE on holdout; `--emb-cluster-te` stays opt-in. Cheap emb-discovery lane exhausted

Heavier / researchy:
- [ ] Residual emb after regressing out account TE / reply baseline, then redo direction or PLS (isolates *content* signal from account heat)
- [x] Error-driven clusters: — **v25a/b: flat** (OOF 0.611 / TE 0.612 vs 0.610); failure MiniLM geometry too weak vs account TE + knn; `--fail-clusters` stays opt-in
- [ ] Supervised manifold (UMAP on RF-leaf similarity, etc.) — usually for analysis, not tabular features; only promote if a cheap projection beats PLS

### Modeling
- [x] Quantile / pinball or underprediction-asymmetric loss — **v20a asymmetric: 0.639→0.607 rate RMSE** (default); **v20b quantile α=0.65: worse RMSE** (0.647), slight Spearman lift
- [ ] Larger frozen encoder (Modal credits) vs MiniLM
- [ ] Further scale (>120k / more accounts with lower `min-account-tweets`)
- [x] Account-level **target encoding** (train-fold LOO, leakage-safe) — **v19: 0.642 → 0.639 rate RMSE**, Spearman 0.441→0.452; default on (`--no-account-te` to ablate)
- [x] True account **random effects** via GPBoost — **v27c: regress** (0.653 vs v27b 0.613; also worse than mse floor 0.642); custom asymmetric loss unsupported with `gp_model`; RE variance tiny on full stack; keep `--model gpboost` opt-in only
- [x] Emb kNN train-fold TE (content prior for cold-start accounts) — **v22b: 0.607 → 0.606**; default on (`--no-emb-knn-te` to ablate); PCA must only transform `emb_\d+`

### Tried — do not re-litigate without a new angle
- [x] Non-reply sample weight / breakout OOF head / nonreply-only train (v8: no lift)
- [x] Author emb affinity cosine/L2 to pre-T mean (v10: flat) — see richer revisit under Standalone
- [x] Dump inbound graph + parent context + scale (v9–v12: current stack)
