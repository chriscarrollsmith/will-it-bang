"""
Summarize validation errors from train.py's val_predictions.parquet.

Usage:
    uv run analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

CACHE_DIR = Path.home() / ".cache" / "will-it-bang"
VAL_PRED_PATH = CACHE_DIR / "val_predictions.parquet"
ANALYSIS_PATH = CACHE_DIR / "error_analysis.json"


def _rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def slice_metrics(df: pd.DataFrame, name: str) -> dict:
    y = df["y_true_log"].to_numpy()
    p = df["y_pred_log"].to_numpy()
    b = df["y_base_log"].to_numpy()
    sp = spearmanr(y, p).correlation
    return {
        "slice": name,
        "n": int(len(df)),
        "rmse_log": _rmse(y, p),
        "baseline_rmse_log": _rmse(y, b),
        "r2_log": float(r2_score(y, p)) if len(df) > 1 else float("nan"),
        "spearman": float(sp) if sp == sp else float("nan"),
        "mean_true_eng": float(np.expm1(y).mean()),
        "mean_abs_err_log": float(df["abs_err_log"].mean()),
    }


def main() -> None:
    if not VAL_PRED_PATH.exists():
        raise SystemExit(f"Missing {VAL_PRED_PATH}. Run: uv run train.py")

    df = pd.read_parquet(VAL_PRED_PATH)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["eng_tercile"] = pd.qcut(df["y_true_log"], 3, labels=["low", "mid", "high"], duplicates="drop")

    slices = [
        slice_metrics(df, "all"),
        slice_metrics(df[df["is_reply"] == 1], "reply"),
        slice_metrics(df[df["is_reply"] == 0], "nonreply"),
    ]
    if "has_media" in df.columns:
        slices.append(slice_metrics(df[df["has_media"] == 1], "has_media"))
        slices.append(slice_metrics(df[df["has_media"] == 0], "no_media"))
    for label in df["eng_tercile"].dropna().unique():
        slices.append(slice_metrics(df[df["eng_tercile"] == label], f"eng_{label}"))

    # Accounts with enough val tweets: worst/best by RMSE.
    rows = []
    for (account_id, username), g in df.groupby(["account_id", "username"], dropna=False):
        if len(g) < 20:
            continue
        rows.append(
            {
                "account_id": account_id,
                "username": username,
                "n": int(len(g)),
                "rmse_log": _rmse(g["y_true_log"], g["y_pred_log"]),
                "mean_eng": float(np.expm1(g["y_true_log"]).mean()),
            }
        )
    acct = pd.DataFrame(rows).sort_values("rmse_log")
    best_accounts = acct.head(8).to_dict(orient="records")
    worst_accounts = acct.tail(8).iloc[::-1].to_dict(orient="records")

    worst = df.nlargest(12, "abs_err_log")[
        [
            c
            for c in (
                "username",
                "engagement",
                "y_pred_eng",
                "abs_err_log",
                "is_reply",
                "has_media",
                "text_len",
                "full_text",
            )
            if c in df.columns
        ]
    ].copy()
    worst["full_text"] = worst["full_text"].astype(str).str.replace("\n", " ").str.slice(0, 140)
    worst_rows = worst.to_dict(orient="records")

    # Over/under prediction rates.
    over = float((df["err_log"] > 0.5).mean())
    under = float((df["err_log"] < -0.5).mean())

    analysis = {
        "n_val": int(len(df)),
        "overall": slices[0],
        "slices": slices,
        "frac_overpredict_log_gt_0_5": over,
        "frac_underpredict_log_lt_-0_5": under,
        "best_accounts": best_accounts,
        "worst_accounts": worst_accounts,
        "worst_misses": worst_rows,
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, default=str) + "\n")

    print("Slice metrics:")
    print(pd.DataFrame(slices).to_string(index=False))
    print(f"\nOver-predict (err_log>0.5): {over:.1%}")
    print(f"Under-predict (err_log<-0.5): {under:.1%}")
    print("\nWorst misses:")
    print(worst.to_string(index=False))
    print(f"\nWrote {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
