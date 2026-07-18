"""
Train a hybrid engagement model on the prepared Community Archive dataset.

Frozen MiniLM embeddings + leakage-safe tabular features → LightGBM regressor
predicting log1p(favorites + retweets). Evaluation is a global time holdout.

History features must not use dump-time fav/RT (see AGENTS.md). Author
popularity inputs are pre-T reply-received rates. Default: residual against
context reply baseline + train-fit emb PCA; optional TF-IDF SVD text features.

Usage:
    uv run train.py
    uv run train.py --text-svd 64
    uv run train.py --no-residual
    uv run train.py --skip-embeddings
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CACHE_DIR = Path.home() / ".cache" / "will-it-bang"
DATASET_PATH = CACHE_DIR / "dataset.parquet"
MODEL_PATH = CACHE_DIR / "model.txt"
METRICS_PATH = CACHE_DIR / "metrics.json"
VAL_PRED_PATH = CACHE_DIR / "val_predictions.parquet"
ANALYSIS_PATH = CACHE_DIR / "error_analysis.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.tsv"

TABULAR_FEATURES = [
    "text_len",
    "log_text_len",
    "has_url",
    "n_mentions",
    "n_hashtags",
    "has_question",
    "has_exclaim",
    "n_emoji",
    "starts_mention",
    "caps_ratio",
    "has_media",
    "n_media",
    "has_ca_url",
    "n_ca_urls",
    "is_reply",
    "is_quote",
    "is_self_reply",
    "hour",
    "dow",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "hist_n",
    "hist_tweets_7d",
    "hist_tweets_30d",
    "days_since_last",
    "hist_frac_reply",
    "hist_frac_url",
    "hist_frac_media",
    "hist_mean_replies",
    "hist_median_replies",
    "hist_p90_replies",
    "hist_last_replies",
    "hist_std_replies",
    "hist_mean_replies_7d",
    "hist_mean_replies_30d",
    "hist_decay_mean_replies",
    "hist_mean_replies_nonreply",
    "hist_mean_replies_reply",
    "reply_x_hist_replies_nr",
    "media_x_hist_replies",
]


def feature_matrix(df: pd.DataFrame, skip_embeddings: bool) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in TABULAR_FEATURES if c in df.columns]
    missing = [c for c in TABULAR_FEATURES if c not in df.columns]
    if missing:
        print(f"Warning: missing tabular features (skipped): {missing}")
    if not skip_embeddings:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            raise SystemExit("Dataset has no emb_* columns — re-run prepare.py")
        cols = cols + emb_cols
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    cols = cols + tfidf_cols
    if "p_breakout" in df.columns:
        cols = cols + ["p_breakout"]
    x = df[cols].to_numpy(dtype=np.float32)
    return x, cols


def time_split(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("created_at", kind="mergesort").reset_index(drop=True)
    cut = int(len(df) * (1.0 - val_frac))
    cut = max(1, min(cut, len(df) - 1))
    train = df.iloc[:cut]
    val = df.iloc[cut:]
    print(
        f"Time split: train={len(train):,} "
        f"[{train['created_at'].min()} → {train['created_at'].max()}], "
        f"val={len(val):,} "
        f"[{val['created_at'].min()} → {val['created_at'].max()}]"
    )
    print(
        f"Accounts: train={train['account_id'].nunique():,} "
        f"val={val['account_id'].nunique():,} "
        f"overlap={len(set(train['account_id']) & set(val['account_id'])):,}"
    )
    return train, val


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    spearman_val = spearmanr(y_true, y_pred).correlation
    spearman = float(spearman_val) if spearman_val == spearman_val else float("nan")
    pred_eng = np.expm1(np.clip(y_pred, 0, 20))
    true_eng = np.expm1(y_true)
    eng_mae = float(np.mean(np.abs(pred_eng - true_eng)))
    return {
        "rmse_log": rmse,
        "mae_log": mae,
        "r2_log": r2,
        "spearman": spearman,
        "mae_engagement": eng_mae,
    }


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    seed: int,
    weight_train: np.ndarray | None = None,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weight_train,
        feature_name=feature_names,
    )
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, feature_name=feature_names)
    params = {
        "objective": "huber",
        "alpha": 1.0,
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 47,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=1200,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )


def train_breakout_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    seed: int,
    weight_train: np.ndarray | None = None,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weight_train,
        feature_name=feature_names,
    )
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, feature_name=feature_names)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=600,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def add_breakout_probs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    baseline_kind: str,
    skip_embeddings: bool,
    seed: int,
    n_folds: int = 3,
    nonreply_weight: float = 1.0,
    mode: str = "vs_baseline",
    quantile: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """OOF P(breakout) on train; full-fit probs on val.

    mode=vs_baseline: log_eng > leakage-safe author baseline (often too frequent).
    mode=vs_quantile: log_eng >= train-fold quantile (default 0.8).
    """
    from sklearn.model_selection import KFold

    y_train_abs = train_df["log_engagement"].to_numpy(dtype=np.float64)
    y_val_abs = val_df["log_engagement"].to_numpy(dtype=np.float64)
    if mode == "vs_baseline":
        base_train = author_baseline(train_df, baseline_kind)
        base_val = author_baseline(val_df, baseline_kind)
        y_break_train = (y_train_abs > base_train).astype(np.float64)
        y_break_val = (y_val_abs > base_val).astype(np.float64)
    elif mode == "vs_quantile":
        thr = float(np.quantile(y_train_abs, quantile))
        y_break_train = (y_train_abs >= thr).astype(np.float64)
        y_break_val = (y_val_abs >= thr).astype(np.float64)
        print(f"Breakout quantile threshold (train q={quantile}): {thr:.4f}")
    else:
        raise ValueError(mode)

    x_all, feature_names = feature_matrix(train_df, skip_embeddings=skip_embeddings)
    x_val, _ = feature_matrix(val_df, skip_embeddings=skip_embeddings)
    weights = np.where(train_df["is_reply"].to_numpy() == 0, nonreply_weight, 1.0).astype(np.float64)

    oof = np.zeros(len(train_df), dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_all)):
        clf = train_breakout_classifier(
            x_all[tr_idx],
            y_break_train[tr_idx],
            x_all[va_idx],
            y_break_train[va_idx],
            feature_names,
            seed=seed + fold,
            weight_train=weights[tr_idx],
        )
        oof[va_idx] = clf.predict(x_all[va_idx], num_iteration=clf.best_iteration)

    clf_full = train_breakout_classifier(
        x_all,
        y_break_train,
        x_val,
        y_break_val,
        feature_names,
        seed=seed + 99,
        weight_train=weights,
    )
    p_val = clf_full.predict(x_val, num_iteration=clf_full.best_iteration)
    # Quick val AUC for logging.
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y_break_val, p_val)) if y_break_val.min() != y_break_val.max() else float("nan")
    print(
        f"Breakout head: rate_train={y_break_train.mean():.3f} "
        f"rate_val={y_break_val.mean():.3f} val_auc={auc:.3f}"
    )

    train_out = train_df.copy()
    val_out = val_df.copy()
    train_out["p_breakout"] = oof.astype(np.float32)
    val_out["p_breakout"] = p_val.astype(np.float32)
    return train_out, val_out, auc


def append_results(description: str, metrics: dict) -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "rmse_log\tspearman\tr2_log\ttrain_rows\tval_rows\tdescription\n"
        )
    with RESULTS_PATH.open("a") as f:
        f.write(
            f"{metrics['rmse_log']:.6f}\t{metrics['spearman']:.6f}\t"
            f"{metrics['r2_log']:.6f}\t{metrics['train_rows']}\t"
            f"{metrics['val_rows']}\t{description}\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Residualize against --baseline (default on; use --no-residual for absolute)",
    )
    p.add_argument("--skip-embeddings", action="store_true", help="Ablation: tabular features only")
    p.add_argument(
        "--split-reply",
        action="store_true",
        help="Train separate models for replies vs non-replies",
    )
    p.add_argument(
        "--emb-pca",
        type=int,
        default=96,
        help="PCA-reduce emb_* to this many dims (fit on train only); 0 disables",
    )
    p.add_argument(
        "--text-svd",
        type=int,
        default=128,
        help="Train-fit TF-IDF + TruncatedSVD dims from full_text; 0 disables",
    )
    p.add_argument(
        "--baseline",
        choices=("hist_replies", "hist_decay_replies", "context_replies"),
        default="context_replies",
        help="Leakage-safe author baseline (pre-T reply rates); used only with --residual",
    )
    p.add_argument(
        "--nonreply-weight",
        type=float,
        default=1.0,
        help="Sample weight for original (non-reply) tweets during training",
    )
    p.add_argument(
        "--breakout-head",
        action="store_true",
        help="Add OOF P(breakout) as a regressor feature",
    )
    p.add_argument(
        "--breakout-mode",
        choices=("vs_baseline", "vs_quantile"),
        default="vs_quantile",
        help="Breakout label definition for --breakout-head",
    )
    p.add_argument(
        "--breakout-quantile",
        type=float,
        default=0.8,
        help="Train quantile for --breakout-mode vs_quantile",
    )
    p.add_argument(
        "--nonreply-only",
        action="store_true",
        help="Train regressor on non-replies only; replies use baseline + mean reply residual",
    )
    p.add_argument("--description", type=str, default="leakage-safe hybrid")
    return p.parse_args()


def author_baseline(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Leakage-safe author baseline (pre-T reply-received rates on prior tweets)."""
    if kind == "hist_replies":
        return df["hist_mean_replies"].to_numpy(dtype=np.float64)
    if kind == "hist_decay_replies":
        return df["hist_decay_mean_replies"].to_numpy(dtype=np.float64)
    if kind == "context_replies":
        reply = df["is_reply"].to_numpy(dtype=bool)
        out = df["hist_mean_replies_nonreply"].to_numpy(dtype=np.float64).copy()
        out[reply] = df.loc[reply, "hist_mean_replies_reply"].to_numpy(dtype=np.float64)
        return out
    raise ValueError(kind)


def apply_emb_pca(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.decomposition import PCA

    emb_cols = [c for c in train_df.columns if c.startswith("emb_")]
    if not emb_cols:
        return train_df, val_df
    pca = PCA(n_components=min(n_components, len(emb_cols), len(train_df) - 1), random_state=0)
    train_z = pca.fit_transform(train_df[emb_cols].to_numpy(dtype=np.float32))
    val_z = pca.transform(val_df[emb_cols].to_numpy(dtype=np.float32))
    train_out = train_df.drop(columns=emb_cols).copy()
    val_out = val_df.drop(columns=emb_cols).copy()
    for i in range(train_z.shape[1]):
        train_out[f"emb_{i}"] = train_z[:, i]
        val_out[f"emb_{i}"] = val_z[:, i]
    print(f"PCA embeddings: {len(emb_cols)} → {train_z.shape[1]} (var={pca.explained_variance_ratio_.sum():.3f})")
    return train_out, val_out


def apply_text_svd(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bag-time text only; vectorizer + SVD fit on train fold."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_text = train_df["full_text"].fillna("").astype(str)
    val_text = val_df["full_text"].fillna("").astype(str)
    n_comp = min(n_components, max(2, len(train_df) - 1))
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_x = vectorizer.fit_transform(train_text)
    val_x = vectorizer.transform(val_text)
    n_comp = min(n_comp, train_x.shape[1] - 1) if train_x.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    train_z = svd.fit_transform(train_x).astype(np.float32)
    val_z = svd.transform(val_x).astype(np.float32)
    tfidf_cols = [f"tfidf_{i}" for i in range(train_z.shape[1])]
    train_out = pd.concat(
        [train_df.reset_index(drop=True), pd.DataFrame(train_z, columns=tfidf_cols)],
        axis=1,
    )
    val_out = pd.concat(
        [val_df.reset_index(drop=True), pd.DataFrame(val_z, columns=tfidf_cols)],
        axis=1,
    )
    print(
        f"TF-IDF SVD: vocab={len(vectorizer.vocabulary_):,} → {train_z.shape[1]} "
        f"(var={svd.explained_variance_ratio_.sum():.3f})"
    )
    return train_out, val_out


def fit_predict_group(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    residual: bool,
    baseline_kind: str,
    skip_embeddings: bool,
    seed: int,
    nonreply_weight: float = 1.0,
) -> tuple[np.ndarray, lgb.Booster, list[str]]:
    x_train, feature_names = feature_matrix(train_df, skip_embeddings=skip_embeddings)
    x_val, _ = feature_matrix(val_df, skip_embeddings=skip_embeddings)
    y_train_abs = train_df["log_engagement"].to_numpy(dtype=np.float64)
    y_val_abs = val_df["log_engagement"].to_numpy(dtype=np.float64)
    base_train = author_baseline(train_df, baseline_kind)
    base_val = author_baseline(val_df, baseline_kind)
    if residual:
        y_train = y_train_abs - base_train
        y_val_model = y_val_abs - base_val
    else:
        y_train = y_train_abs
        y_val_model = y_val_abs
    weight_train = np.where(
        train_df["is_reply"].to_numpy() == 0,
        nonreply_weight,
        1.0,
    ).astype(np.float64)
    model = train_model(
        x_train,
        y_train,
        x_val,
        y_val_model,
        feature_names,
        seed=seed,
        weight_train=weight_train,
    )
    raw = model.predict(x_val, num_iteration=model.best_iteration)
    pred = raw + base_val if residual else raw
    return pred, model, feature_names


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Missing dataset at {args.dataset}. Run: uv run prepare.py")

    t0 = time.time()
    df = pd.read_parquet(args.dataset)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if "hist_mean_replies" not in df.columns:
        raise SystemExit(
            "Dataset missing hist_mean_replies — re-run prepare.py "
            "(fav/RT history features are banned; need reply-received rebuild)."
        )
    if any(c.startswith("hist_mean_eng") for c in df.columns):
        print("Warning: dataset still has hist_*_eng columns; they are ignored by TABULAR_FEATURES")

    train_df, val_df = time_split(df, args.val_frac)
    if args.emb_pca > 0 and not args.skip_embeddings:
        train_df, val_df = apply_emb_pca(train_df, val_df, args.emb_pca)
    if args.text_svd > 0:
        train_df, val_df = apply_text_svd(train_df, val_df, args.text_svd)

    residual = args.residual
    y_train_abs = train_df["log_engagement"].to_numpy(dtype=np.float64)
    y_val_abs = val_df["log_engagement"].to_numpy(dtype=np.float64)
    reply_base_val = author_baseline(val_df, "hist_replies")
    chosen_base_val = author_baseline(val_df, args.baseline)

    mean_metrics = evaluate(y_val_abs, np.full(len(y_val_abs), float(y_train_abs.mean())))
    reply_base_metrics = evaluate(y_val_abs, reply_base_val)
    chosen_base_metrics = evaluate(y_val_abs, chosen_base_val)
    print("Baseline (train-mean):", {k: round(v, 4) for k, v in mean_metrics.items()})
    print("Baseline (hist_replies):", {k: round(v, 4) for k, v in reply_base_metrics.items()})
    print(f"Baseline ({args.baseline}):", {k: round(v, 4) for k, v in chosen_base_metrics.items()})
    print(
        "Target:",
        "residual" if residual else "absolute",
        f"| baseline={args.baseline} | split_reply={args.split_reply} "
        f"| nonreply_weight={args.nonreply_weight} | breakout_head={args.breakout_head}",
    )

    breakout_auc = float("nan")
    if args.breakout_head:
        train_df, val_df, breakout_auc = add_breakout_probs(
            train_df,
            val_df,
            baseline_kind=args.baseline,
            skip_embeddings=args.skip_embeddings,
            seed=args.seed,
            nonreply_weight=args.nonreply_weight,
            mode=args.breakout_mode,
            quantile=args.breakout_quantile,
        )

    if args.nonreply_only:
        tr_nr = train_df[train_df["is_reply"] == 0]
        va_nr = val_df[val_df["is_reply"] == 0]
        print(f"Non-reply-only train: {len(tr_nr):,} / {len(train_df):,} (val NR {len(va_nr):,})")
        _, model, feature_names = fit_predict_group(
            tr_nr,
            va_nr,
            residual=residual,
            baseline_kind=args.baseline,
            skip_embeddings=args.skip_embeddings,
            seed=args.seed,
            nonreply_weight=1.0,
        )
        # Score full val with the non-reply model, then overwrite replies.
        x_val_full, _ = feature_matrix(val_df, skip_embeddings=args.skip_embeddings)
        raw_full = model.predict(x_val_full, num_iteration=model.best_iteration)
        base_val = author_baseline(val_df, args.baseline)
        pred = raw_full + base_val if residual else raw_full
        tr_r = train_df[train_df["is_reply"] == 1]
        if len(tr_r):
            mean_r_resid = float(
                (
                    tr_r["log_engagement"].to_numpy(dtype=np.float64)
                    - author_baseline(tr_r, args.baseline)
                ).mean()
            )
        else:
            mean_r_resid = 0.0
        reply_m = val_df["is_reply"].to_numpy() == 1
        pred[reply_m] = base_val[reply_m] + mean_r_resid
        print(f"Reply fallback residual mean: {mean_r_resid:.4f}")
    elif args.split_reply:
        pred = np.zeros(len(val_df), dtype=np.float64)
        models = []
        feature_names: list[str] = []
        for label, mask_val in (("nonreply", 0), ("reply", 1)):
            tr = train_df[train_df["is_reply"] == mask_val]
            va = val_df[val_df["is_reply"] == mask_val]
            if len(tr) < 100 or len(va) == 0:
                print(f"  skip {label}: train={len(tr)} val={len(va)}")
                continue
            print(f"  fitting {label}: train={len(tr):,} val={len(va):,}")
            p, model, feature_names = fit_predict_group(
                tr,
                va,
                residual=residual,
                baseline_kind=args.baseline,
                skip_embeddings=args.skip_embeddings,
                seed=args.seed + mask_val,
                nonreply_weight=1.0 if mask_val == 0 else args.nonreply_weight,
            )
            pred[val_df["is_reply"].to_numpy() == mask_val] = p
            models.append(model)
        model = models[-1] if models else None
    else:
        pred, model, feature_names = fit_predict_group(
            train_df,
            val_df,
            residual=residual,
            baseline_kind=args.baseline,
            skip_embeddings=args.skip_embeddings,
            seed=args.seed,
            nonreply_weight=args.nonreply_weight,
        )

    metrics = evaluate(y_val_abs, pred)
    nonreply_mask = val_df["is_reply"].to_numpy() == 0
    reply_mask = ~nonreply_mask
    metrics["rmse_log_nonreply"] = evaluate(y_val_abs[nonreply_mask], pred[nonreply_mask])["rmse_log"]
    metrics["rmse_log_reply"] = evaluate(y_val_abs[reply_mask], pred[reply_mask])["rmse_log"]
    metrics["spearman_nonreply"] = evaluate(y_val_abs[nonreply_mask], pred[nonreply_mask])["spearman"]
    metrics["best_iteration"] = int(getattr(model, "best_iteration", 0) or 0)
    metrics["num_features"] = len(feature_names)
    metrics["train_rows"] = int(len(train_df))
    metrics["val_rows"] = int(len(val_df))
    metrics["n_accounts_train"] = int(train_df["account_id"].nunique())
    metrics["n_accounts_val"] = int(val_df["account_id"].nunique())
    metrics["baseline_rmse_log"] = mean_metrics["rmse_log"]
    metrics["reply_baseline_rmse_log"] = reply_base_metrics["rmse_log"]
    metrics["chosen_baseline_rmse_log"] = chosen_base_metrics["rmse_log"]
    metrics["residual"] = residual
    metrics["baseline_kind"] = args.baseline
    metrics["nonreply_weight"] = args.nonreply_weight
    metrics["breakout_head"] = args.breakout_head
    metrics["breakout_auc"] = breakout_auc
    metrics["seconds"] = round(time.time() - t0, 2)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if model is not None:
        # For split-reply, persist the reply model as a convenience artifact.
        model.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    append_results(args.description, metrics)

    # Persist val rows + predictions for error analysis.
    pred_df = val_df[
        [
            c
            for c in (
                "tweet_id",
                "account_id",
                "username",
                "created_at",
                "full_text",
                "engagement",
                "log_engagement",
                "is_reply",
                "is_quote",
                "has_media",
                "has_url",
                "text_len",
                "hist_mean_replies",
                "hist_mean_replies_reply",
                "hist_mean_replies_nonreply",
                "days_since_last",
            )
            if c in val_df.columns
        ]
    ].copy()
    pred_df["y_true_log"] = y_val_abs
    pred_df["y_pred_log"] = pred
    pred_df["y_base_log"] = chosen_base_val if residual else np.full(len(val_df), float(y_train_abs.mean()))
    pred_df["abs_err_log"] = np.abs(pred - y_val_abs)
    pred_df["err_log"] = pred - y_val_abs
    pred_df["y_pred_eng"] = np.expm1(np.clip(pred, 0, 20))
    pred_df.to_parquet(VAL_PRED_PATH, index=False)
    print(f"Wrote val predictions: {VAL_PRED_PATH}")

    if model is not None and feature_names:
        importance = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "gain": model.feature_importance(importance_type="gain"),
                }
            )
            .sort_values("gain", ascending=False)
            .head(15)
        )
        print("\nTop features by gain:")
        print(importance.to_string(index=False))

    print("\n---")
    for key in (
        "rmse_log",
        "rmse_log_nonreply",
        "rmse_log_reply",
        "mae_log",
        "r2_log",
        "spearman",
        "spearman_nonreply",
        "mae_engagement",
        "baseline_rmse_log",
        "reply_baseline_rmse_log",
        "chosen_baseline_rmse_log",
        "breakout_auc",
        "best_iteration",
        "num_features",
        "train_rows",
        "val_rows",
        "n_accounts_train",
        "n_accounts_val",
        "seconds",
    ):
        print(f"{key}: {metrics[key]}")
    print(f"model_path: {MODEL_PATH}")


if __name__ == "__main__":
    main()
