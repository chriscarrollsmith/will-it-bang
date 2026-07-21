# Linear issue draft — paste into Linear if MCP auth unavailable

**Title:** v27: GPBoost account random effects (replace LOO account TE)

**Labels (suggested):** experiment, modeling, exploit-failure-mode

**Priority:** High (targets open hot non-reply under-prediction; literature-backed vs exhausted emb-discovery lane)

---

## Role for executing agent

Implement and run the experiment below end-to-end on `will-it-bang`. Follow `AGENTS.md` leakage rules, log outcomes in `experiments.md`, one knob then restore. Primary metric: **exposure-rate RMSE** (`rmse_log` in rate space). Compare against same-rebuild baseline (v23a ≈ 0.610) or a refreshed media baseline if you re-prepare.

Repo: https://github.com/chriscarrollsmith/will-it-bang

---

## Why this experiment (research brief)

### What the experiment log says
- Best rate model remains **v22b (0.606)** with account TE + asymmetric loss + emb kNN TE.
- Open failure mode: **hot non-reply under-prediction** (~2.22 mean under-pred); often already-hot authors.
- Recent emb-discovery flats/regresses (**v23–v26**: CAV, PLS, fail-clusters, cluster TE) show MiniLM content geometry is mostly **redundant with or fights account TE** (v26 cluster TE *regressed* to 0.624 while `cluster_mean_rate` was #3 gain).
- Meta lesson (`logs/meta_process.md`): honest lifts came from problem formulation + denser history + account modeling — not descriptive LM tags or more emb projections.

### What literature adds (new angle vs what we already tried)
Already tried: LOO shrunk **account target encoding** (v19), **asymmetric MSE** (v20a), quantile (v20b worse), breakout OOF head as a *feature* (v8 flat).

**Not tried:** a proper **grouped random-effects + tree-boosting** joint model for `account_id`.

Key references:
1. **Sigrist (2022), GPBoost / “Gaussian Process Boosting”** (JMLR) — tree-boosting + grouped random effects. Strong on high-cardinality categoricals vs TE/embeddings ([arXiv:2307.02071](https://arxiv.org/pdf/2307.02071); [mixed-effects boosting demo](https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb)). Library: Apache-2.0.
2. **Frisch–Waugh–Lovell residualization** — isolate within-author content after removing author effects (used in influencer engagement residualization, e.g. Yang et al. TikTok). Explains v26: cluster TE competed with account TE instead of conditioning on author intercepts.
3. Related but **deferred**: SERA / imbalanced-regression losses ([Ribeiro & Moniz](https://link.springer.com/article/10.1007/s10994-020-05900-9)); two-stage hurdle virality models; Hawkes intensity on reply arrivals — see “Out of scope”.

**Hypothesis:** Modeling `account_id` as a **Gaussian random intercept inside the booster** (GPBoost) will (a) improve global rate RMSE vs LOO TE, and/or (b) free trees to fit within-author content residuals and reduce hot non-reply under-prediction — without adding more MiniLM projections.

---

## Experiment ID

**v27 — GPBoost grouped random effects for `account_id`**

One knob family: replace (or ablate against) `--account-te` with GPBoost RE. Do not stack new emb features in the same run.

---

## Setup

### Dataset
- Prefer existing `~/.cache/will-it-bang/dataset.parquet` (~110k rows). Note: skip-media rebuilds have 0% media; v22b had ~9%.
- Optional: `uv run prepare.py --media-only` for closer v22b parity — document media coverage in the writeup.
- Time holdout: default `--val-frac 0.2`.

### License / deps
- GPBoost **Apache-2.0** (allowed; same class as MiniLM).
- `uv add gpboost` (pin in `pyproject.toml`). CPU is fine.
- Keep frozen MiniLM + existing tabular features; never feed handles/names into LMs for fame scoring.

### Leakage constraints (hard)
- No dump-time fav/RT as inputs; no `num_followers`; no display-name/username model features.
- Unseen val accounts: use GPBoost’s standard prediction for new groups (population-level); document cold-account behavior.
- Fit only on train time fold.

---

## Implementation sketch

1. Branch: `cursor/gpboost-account-re-e8da` from latest `main`.
2. In `train.py`:
   - `--model {lgbm,gpboost}` (default `lgbm`)
   - When `gpboost`: `--no-account-te` by default for the primary run (avoid double-counting author mean)
   - Keep emb kNN TE / emb PCA / TF-IDF SVD / asymmetric loss as in current defaults unless ablating
3. Train path (illustrative):

```python
import gpboost as gpb
gp_model = gpb.GPModel(group_data=train_account_id, likelihood="gaussian")
data_bst = gpb.Dataset(X_train, label=y_train)
bst = gpb.train(params, data_bst, gp_model=gp_model, num_boost_round=...)
pred = bst.predict(X_val, group_data_pred=val_account_id)
```

4. Mirror early stopping / val monitoring as closely as practical to the LGBM harness.
5. Preserve metrics JSON / `results.tsv` / importance logging; also log estimated RE variance if exposed by GPBoost.

### Run matrix (same dataset)

| Run | Config | Intent |
|---|---|---|
| **v27a** | LGBM stack, `--no-account-te` | Ablation floor without author mean |
| **v27b** | LGBM + account TE (current defaults) | Control |
| **v27c** | GPBoost + RE on `account_id`, `--no-account-te`, else defaults | **Primary** |
| **v27d** (optional) | GPBoost RE **plus** account TE | Sanity: expect flat/worse if double-counting |

Primary comparison: **v27c vs v27b** on rate RMSE + `mean_underpred_nonreply_top10`.

---

## Smoke / gates

1. Install + tiny fit on ≤2k rows: RE variance σ²_b > 0, predictions finite.
2. Cold-account check: val accounts absent from train → finite preds.
3. Full train only after smoke.

---

## Success criteria

- **Promote / keep as option:** v27c rate RMSE ≤ v27b − 0.003 **or** hot-NR mean under-pred ≤ control − 0.05 with RMSE not worse by >0.002.
- **Flat:** |Δ RMSE| < 0.003 and under-pred unchanged → leave LGBM+TE default; keep flag opt-in; log.
- **Fail:** clear RMSE regress (v26-like) → document and disable.

---

## Logging

- Append to `results.tsv` with clear descriptions.
- Update `experiments.md` timeline + Modeling backlog (true account RE / GPBoost).
- One-line `AGENTS.md` note if the flag is kept.
- Open/update PR; do not change defaults unless success criteria met.

---

## Out of scope (queued follow-ups)

1. **SERA / relevance-weighted extreme loss** — if v27c flats and hot-NR remains ([imbalanced regression](https://link.springer.com/article/10.1007/s10994-020-05900-9)).
2. **Proper two-stage hurdle** (P(breakout)×intensity) — distinct from v8’s breakout-*feature*.
3. **Hawkes-style intensity features** on pre-T reply arrivals — denser graph / prepare work.
4. **FWL residual-emb** then CAV/PLS — sequel if GPBoost helps but content still underused.

---

## Commands

```bash
uv sync
# optional: uv run prepare.py --media-only
uv run train.py --description "v27b lgbm account-te control"
uv run train.py --model gpboost --no-account-te --description "v27c gpboost account RE"
uv run analyze.py
```

## Context
- `experiments.md` — curated timeline through v26
- `logs/meta_process.md` — harness lessons
- `AGENTS.md` — leakage + tooling notes
