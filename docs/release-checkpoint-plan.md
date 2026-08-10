# Release inference checkpoint — plan

Goal: ship a **Hugging Face–ready inference bundle** plus a model card that documents the **input feature schema and derivation recipes**. No CA hydrator service in this phase.

Research scoreboard (**v48b**) stays as-is. The release checkpoint is a **separate train** with a shippable artifact surface.

---

## 1. What we ship

A versioned directory (then HF repo), e.g. `will-it-bang-release/`:

| File | Role |
|---|---|
| `model_level.txt` | LightGBM level booster (exposure-rate target) |
| `model_rank_nr.txt` / `model_rank_rp.txt` | Optional hybrid stratum rankers (multi-draft ranking only) |
| `emb_pca.joblib` | Train-fit MiniLM → PCA (`n≈96`) |
| `tfidf.joblib` + `text_svd.joblib` | Train-fit TF-IDF → TruncatedSVD (`n≈128`) |
| `res_emb_cav_direction.npy` | Unit residual-CAV direction (384-d MiniLM residual space) |
| `res_emb_baselines.json` | OLS residualizer coeffs vs `account_mean_rate` / `is_reply` / `inbound_replies_per_tweet` |
| `feature_schema.json` | Ordered booster feature names, dtypes, defaults, groups |
| `constants.json` | Global prior mean, `account_te_prior` (m=20), exposure cap days, encoder id |
| `metrics.json` | Locked primaries + guardrails for this release train |
| `README.md` / model card | Target definition, input schema, derivation recipes, leakage notes |
| `predict.py` | Thin reference: raw/schema inputs → transforms → score |

**Do not upload:** `dataset.parquet`, val preds, embed/graph caches, per-`account_id` TE maps, emb-kNN train index.

Encoder dependency (not vendored unless we choose to): `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0). Card pins the revision.

---

## 2. Input shape (two layers)

### Layer A — caller feature row (documented contract)

Everything the booster needs that is **not** produced inside the bundle. Callers fill these from their own pre-T history / tweet JSON. No `account_id` is required as a model input.

**Always required (content / structure / calendar):**

- Tweet text → MiniLM `emb_0..383` (caller or `predict.py`)
- Structural: `text_len`, `has_url`, `has_media`, `n_mentions`, `is_reply`, `is_quote`, `is_self_reply`, hour/dow encodings, …
- Calendar: `year`, `month_sin/cos`, `platform_era` (+ era one-hots) from `created_at`

**Ego / context floats (same names as research stack; missing → documented defaults, usually 0):**

- Author history (reply-arrival based, never dump fav/RT): `hist_*`, `inbound_replies*`, `inbound_quotes*`, `inbound_mentions*`, crosses
- Parent context when reply/quote: `reply_parent_*` / `quote_parent_*` (emb cos, age, pre-T parent replies — never parent fav/RT)
- **`account_mean_rate` / `log_account_n`** — caller-supplied; recipe in §4 (not an `account_id` lookup into our weights)
- Optional URL structure flags (`url_only`, `domain_is_*`, …). **`domain_mean_rate` omitted** in v1 release (see §3)

**Out of the v1 caller contract:** CA provenance (`is_stream`, upload-end, …), media-type breakdown, LM tag caches, dyad / parent-chain / interlocutor opt-ins.

### Layer B — bundle-produced features

From Layer A + text, `predict.py` applies shipped transforms only:

1. Residual emb CAV proj / sq (using shipped direction + baseline OLS)
2. Emb PCA → `emb_0..n-1` replacing raw MiniLM in the booster matrix
3. TF-IDF → SVD → `svd_*`
4. LightGBM predict

---

## 3. Release train knobs (vs research default)

Reuse X-complete full `dataset.parquet` (S3 pull). One knob family: **what must not require private train residue at serve time.**

| Knob | Research default | Release train | Why |
|---|---|---|---|
| `--account-te` | on | **on** (fills `account_mean_rate` in-train) | Feature stays; card documents consumer recipe matching train (`m=20`, shrink to shipped `global_mean`) |
| `--emb-knn-te` | on | **off** | Needs train MiniLM index; consumers cannot rebuild without our corpus |
| Domain mean TE | on when URL cols present | **drop `domain_mean_rate` / `log_domain_n` from feature matrix** | Same class of train-only lookup; keep structural URL flags |
| `--provenance-features` | on | **off** | CA ingestion artifact; not meaningful outside CA |
| `--media-types` / LM labels / dyad / chain / interlocutor | opt-in | **off** | Keep v1 schema small |
| `--residual-emb-cav` | on | **on** | Direction is a fixed vector; baselines are caller floats |
| `--hybrid-rank` | on | **on for eval; ship level + optional rankers** | Single-draft API uses level score; rankers only if ranking ≥2 drafts/account |
| Target | exposure rate | **same** | `log1p(eng / (min(age_days,30)+1))` |
| Holdout | time | **same** | Report release metrics on the locked primaries |

Train command (sketch):

```bash
bash scripts/s3_cache_sync.sh pull   # or pull-incoming
uv run train.py \
  --no-emb-knn-te \
  --no-provenance-features \
  --description "release-v1: no knn/provenance/domain-TE map" \
  --export-release ~/.cache/will-it-bang/release-v1
```

(`--export-release` is new harness work — see §5.)

Expect: global Spearman / RMSE may move vs v48b (no knn / provenance / domain map). Within-author `wa_*` is the release quality bar; log a same-rebuild twin with research defaults for Δ.

---

## 4. Model-card recipes (the important part)

### Target

\[
y = \log(1 + \mathrm{eng} / (\min(\mathrm{age\_days}, 30) + 1)),\quad
\mathrm{eng} = \mathrm{favorite\_count} + \mathrm{retweet\_count}
\]

Exclude `RT @…` rows. Labels may use final fav+RT on the **target** tweet; history features must not.

### `account_mean_rate` / `log_account_n`

For a candidate tweet at time \(T\), let \(\mathcal{H}\) be the author’s earlier posts with known engagement (same eng definition), each converted to the same exposure-rate \(y_i\) using age-at-dump or a fixed horizon when available.

\[
\bar{y} = \mathrm{mean}_{i \in \mathcal{H}}(y_i),\quad
n = |\mathcal{H}|,\quad
m = 20,\quad
\mu = \texttt{constants.global\_mean}
\]

\[
\texttt{account\_mean\_rate} = \frac{n\,\bar{y} + m\,\mu}{n + m},\quad
\texttt{log\_account\_n} = \log(1 + n)
\]

If \(n=0\): `account_mean_rate = μ`, `log_account_n = 0`.

Training uses the existing LOO / train-fold TE implementation as the in-fold estimator of this quantity (leakage-safe). The card states the consumer formula above — **not** “load our account_id map.”

### Ego graph features

Document the prepare definitions already in code: counts of replies/quotes/mentions to the author with `created_at < T`, rates per prior tweet, 7d/30d windows, parent MiniLM cosine + pre-T reply arrivals on the parent. Ban: follower snapshots, dump-time fav/RT history, undated likes.

### Text

- MiniLM: mean-pool `all-MiniLM-L6-v2` on `full_text` (same as `prepare.py`).
- Bundle applies residual CAV → PCA; TF-IDF/SVD on raw text.

### Defaults

`feature_schema.json` lists per-column `default` (0 for most counts/flags; `global_mean` for `account_mean_rate`). Cold-start = empty ego history + defaults.

---

## 5. Harness work (implementation checklist)

1. **`train.py --export-release DIR`**
   - Persist level booster (and rankers if hybrid).
   - Persist PCA, TF-IDF, SVD, residual-CAV direction, residualizer coeffs, `global_mean`, feature name order.
   - Write `feature_schema.json` + `constants.json` + copy of run `metrics.json`.
   - Fail export if emb-kNN / domain-TE map / account-id map would be required at serve (release train flags enforce this).

2. **`predict.py` (reference)**
   - CLI: JSON/CSV of Layer A (+ optional raw text) → `y_hat` in training-target space.
   - No CA/X network I/O.

3. **Release ablation log**
   - Same rebuild: research default vs release knobs → `logs/release_v1.md` + `experiments.md` one-liner (not a promote of research defaults; a ship baseline).

4. **HF publish (manual last step)**
   - Upload bundle + model card; dataset stays private.
   - Card links this plan and leakage rules from `AGENTS.md`.

5. **Optional follow-ups (not v1)**
   - Multi-draft ranking demo using shipped rankers.
   - CA/archive hydrator that fills Layer A (the earlier “product” path).

---

## 6. Eval bar for “good enough to upload”

On the release train’s time holdout:

- Report `wa_nr_wmean`, `wa_rp_wmean`, global Spearman NR/reply, rate RMSE, `mean_underpred_nonreply_top10`.
- Smoke `predict.py` on a handful of val rows reconstructed from schema (round-trip vs in-process preds).
- Confirm artifact has **no** `account_id`→score table and **no** train embedding index.

---

## 7. Sequencing

1. ~~Land export + `predict.py` + schema writer against current code.~~
2. ~~Run release train on X-complete full data; write `logs/release_v1.md`.~~
3. ~~Freeze card text from `feature_schema.json` + §4 recipes.~~
4. ~~Upload to HF when metrics + round-trip look sane.~~ → [chriscarrollsmith/will-it-bang](https://huggingface.co/chriscarrollsmith/will-it-bang)

No change to the locked research default stack until we deliberately choose to align defaults with release knobs.
