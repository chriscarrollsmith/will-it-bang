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
    if len(df) == 0:
        return {
            "slice": name,
            "n": 0,
            "rmse_log": float("nan"),
            "baseline_rmse_log": float("nan"),
            "r2_log": float("nan"),
            "spearman": float("nan"),
            "mean_true_eng": float("nan"),
            "mean_abs_err_log": float("nan"),
            "frac_under_0_5": float("nan"),
        }
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
        "frac_under_0_5": float((df["err_log"] < -0.5).mean()),
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
    if "is_quote" in df.columns:
        slices.append(slice_metrics(df[df["is_quote"] == 1], "quote"))
        slices.append(slice_metrics(df[df["is_quote"] == 0], "nonquote"))
    if "has_media" in df.columns:
        slices.append(slice_metrics(df[df["has_media"] == 1], "has_media"))
        slices.append(slice_metrics(df[df["has_media"] == 0], "no_media"))
    for label in df["eng_tercile"].dropna().unique():
        slices.append(slice_metrics(df[df["eng_tercile"] == label], f"eng_{label}"))

    # Non-reply × engagement tercile (breakout hunting ground).
    nr = df[df["is_reply"] == 0]
    if len(nr):
        nr = nr.copy()
        nr["nr_eng_tercile"] = pd.qcut(
            nr["y_true_log"], 3, labels=["low", "mid", "high"], duplicates="drop"
        )
        for label in nr["nr_eng_tercile"].dropna().unique():
            slices.append(slice_metrics(nr[nr["nr_eng_tercile"] == label], f"nonreply_eng_{label}"))

    if "inbound_replies" in df.columns:
        df["inbound_tercile"] = pd.qcut(
            df["inbound_replies"], 3, labels=["cold", "warm", "hot"], duplicates="drop"
        )
        for label in df["inbound_tercile"].dropna().unique():
            slices.append(slice_metrics(df[df["inbound_tercile"] == label], f"inbound_{label}"))
            sub = df[(df["inbound_tercile"] == label) & (df["is_reply"] == 0)]
            slices.append(slice_metrics(sub, f"nonreply_inbound_{label}"))

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
                "mean_inbound": float(g["inbound_replies"].mean()) if "inbound_replies" in g else float("nan"),
                "frac_nonreply": float((g["is_reply"] == 0).mean()),
            }
        )
    acct = pd.DataFrame(rows).sort_values("rmse_log")
    best_accounts = acct.head(8).to_dict(orient="records")
    worst_accounts = acct.tail(8).iloc[::-1].to_dict(orient="records")

    # Biggest under-predictions among non-replies (viral misses).
    nr_under = (
        df[df["is_reply"] == 0]
        .nsmallest(15, "err_log")[
            [
                c
                for c in (
                    "username",
                    "engagement",
                    "y_pred_eng",
                    "err_log",
                    "abs_err_log",
                    "has_media",
                    "is_quote",
                    "text_len",
                    "inbound_replies",
                    "inbound_replies_per_tweet",
                    "full_text",
                )
                if c in df.columns
            ]
        ]
        .copy()
    )
    if len(nr_under):
        nr_under["full_text"] = nr_under["full_text"].astype(str).str.replace("\n", " ").str.slice(0, 140)
    viral_misses = nr_under.to_dict(orient="records")

    worst = df.nlargest(12, "abs_err_log")[
        [
            c
            for c in (
                "username",
                "engagement",
                "y_pred_eng",
                "abs_err_log",
                "err_log",
                "is_reply",
                "is_quote",
                "has_media",
                "text_len",
                "inbound_replies",
                "full_text",
            )
            if c in df.columns
        ]
    ].copy()
    worst["full_text"] = worst["full_text"].astype(str).str.replace("\n", " ").str.slice(0, 140)
    worst_rows = worst.to_dict(orient="records")

    over = float((df["err_log"] > 0.5).mean())
    under = float((df["err_log"] < -0.5).mean())
    nr_under_frac = float((nr["err_log"] < -0.5).mean()) if len(nr) else float("nan")

    # Residual lift vs baseline on hard slices.
    hard = df[df["is_reply"] == 0]
    model_gain = (
        float(_rmse(hard["y_true_log"], hard["y_base_log"]) - _rmse(hard["y_true_log"], hard["y_pred_log"]))
        if len(hard)
        else float("nan")
    )

    analysis = {
        "n_val": int(len(df)),
        "overall": slices[0],
        "slices": slices,
        "frac_overpredict_log_gt_0_5": over,
        "frac_underpredict_log_lt_-0_5": under,
        "frac_nonreply_underpredict_lt_-0_5": nr_under_frac,
        "nonreply_rmse_gain_vs_baseline": model_gain,
        "best_accounts": best_accounts,
        "worst_accounts": worst_accounts,
        "worst_misses": worst_rows,
        "viral_underpredicts": viral_misses,
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, default=str) + "\n")

    print("Slice metrics:")
    print(pd.DataFrame(slices).to_string(index=False))
    print(f"\nOver-predict (err_log>0.5): {over:.1%}")
    print(f"Under-predict (err_log<-0.5): {under:.1%}")
    print(f"Non-reply under-predict: {nr_under_frac:.1%}")
    print(f"Non-reply RMSE gain vs baseline: {model_gain:.4f}")
    print("\nWorst absolute misses:")
    print(pd.DataFrame(worst_rows).to_string(index=False))
    print("\nBiggest non-reply underpredicts (viral misses):")
    print(pd.DataFrame(viral_misses).to_string(index=False))
    print(f"\nWrote {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
