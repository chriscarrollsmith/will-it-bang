"""
Train a hybrid engagement model on the prepared Community Archive dataset.

Frozen MiniLM embeddings + leakage-safe tabular features → LightGBM regressor
predicting log1p(favorites + retweets). Evaluation is a global time holdout.

By default trains a residual model: predict (log_engagement - hist_mean_eng),
then add hist_mean_eng back at inference. This focuses capacity on
above/below-baseline content effects.

Usage:
    uv run train.py
    uv run train.py --absolute          # predict log engagement directly
    uv run train.py --skip-embeddings   # tabular ablation
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
    "hist_mean_eng",
    "hist_median_eng",
    "hist_p90_eng",
    "hist_last_eng",
    "hist_std_eng",
    "hist_mean_eng_7d",
    "hist_mean_eng_30d",
    "hist_decay_mean_eng",
    "hist_mean_nonreply",
    "hist_mean_reply",
    "hist_tweets_7d",
    "hist_tweets_30d",
    "days_since_last",
    "hist_frac_reply",
    "hist_frac_url",
    "hist_frac_media",
    "reply_x_hist_nonreply",
    "media_x_hist_mean",
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
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
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
    p.add_argument("--absolute", action="store_true", help="Predict log engagement directly")
    p.add_argument("--skip-embeddings", action="store_true", help="Ablation: tabular features only")
    p.add_argument("--description", type=str, default="hybrid residual")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Missing dataset at {args.dataset}. Run: uv run prepare.py")

    t0 = time.time()
    df = pd.read_parquet(args.dataset)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if "hist_mean_eng" not in df.columns:
        raise SystemExit("Dataset missing hist_mean_eng — re-run prepare.py")

    train_df, val_df = time_split(df, args.val_frac)
    residual = not args.absolute

    x_train, feature_names = feature_matrix(train_df, skip_embeddings=args.skip_embeddings)
    x_val, _ = feature_matrix(val_df, skip_embeddings=args.skip_embeddings)
    y_train_abs = train_df["log_engagement"].to_numpy(dtype=np.float64)
    y_val_abs = val_df["log_engagement"].to_numpy(dtype=np.float64)
    hist_train = train_df["hist_mean_eng"].to_numpy(dtype=np.float64)
    hist_val = val_df["hist_mean_eng"].to_numpy(dtype=np.float64)

    mean_metrics = evaluate(y_val_abs, np.full(len(y_val_abs), float(y_train_abs.mean())))
    hist_metrics = evaluate(y_val_abs, hist_val)
    print("Baseline (train-mean):", {k: round(v, 4) for k, v in mean_metrics.items()})
    print("Baseline (hist-mean): ", {k: round(v, 4) for k, v in hist_metrics.items()})

    if residual:
        y_train = y_train_abs - hist_train
        y_val_model = y_val_abs - hist_val
        print("Target: residual log_engagement - hist_mean_eng")
    else:
        y_train = y_train_abs
        y_val_model = y_val_abs
        print("Target: absolute log_engagement")

    model = train_model(x_train, y_train, x_val, y_val_model, feature_names, seed=args.seed)
    raw_pred = model.predict(x_val, num_iteration=model.best_iteration)
    pred = raw_pred + hist_val if residual else raw_pred

    metrics = evaluate(y_val_abs, pred)
    metrics["best_iteration"] = int(model.best_iteration or 0)
    metrics["num_features"] = len(feature_names)
    metrics["train_rows"] = int(len(train_df))
    metrics["val_rows"] = int(len(val_df))
    metrics["n_accounts_train"] = int(train_df["account_id"].nunique())
    metrics["n_accounts_val"] = int(val_df["account_id"].nunique())
    metrics["baseline_rmse_log"] = mean_metrics["rmse_log"]
    metrics["hist_baseline_rmse_log"] = hist_metrics["rmse_log"]
    metrics["residual"] = residual
    metrics["seconds"] = round(time.time() - t0, 2)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    append_results(args.description, metrics)

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
        "mae_log",
        "r2_log",
        "spearman",
        "mae_engagement",
        "baseline_rmse_log",
        "hist_baseline_rmse_log",
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
