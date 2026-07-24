"""
Summarize validation errors from train.py's val_predictions.parquet.

Usage:
    uv run analyze.py
    uv run analyze.py --predictions path/to/val_predictions.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from train import within_account_spearman

CACHE_DIR = Path.home() / ".cache" / "will-it-bang"
RAW_PARQUET = CACHE_DIR / "enriched_tweets.parquet"
VAL_PRED_PATH = CACHE_DIR / "val_predictions.parquet"
ANALYSIS_PATH = CACHE_DIR / "error_analysis.json"


def _rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def ensure_is_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure is_stream is present; join dump archive_upload_id when missing."""
    out = df.copy()
    if "is_stream" in out.columns:
        return out
    if "archive_upload_id" in out.columns:
        out["is_stream"] = out["archive_upload_id"].isna().astype(np.int8)
        return out
    if not RAW_PARQUET.exists() or "tweet_id" not in out.columns:
        return out
    ids = pd.DataFrame({"tweet_id": out["tweet_id"].astype(str).unique()})
    tmp = CACHE_DIR / "_analyze_prov_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids.to_parquet(tmp, index=False)
    path = str(RAW_PARQUET)
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    prov = con.execute(
        f"""
        SELECT
          CAST(t.tweet_id AS VARCHAR) AS tweet_id,
          CAST(t.archive_upload_id AS VARCHAR) AS archive_upload_id
        FROM read_parquet('{path}') t
        SEMI JOIN read_parquet('{tmp}') c
          ON CAST(t.tweet_id AS VARCHAR) = c.tweet_id
        """
    ).fetchdf()
    con.close()
    try:
        tmp.unlink()
    except OSError:
        pass
    prov = prov.drop_duplicates("tweet_id", keep="first")
    out["tweet_id"] = out["tweet_id"].astype(str)
    out = out.merge(prov, on="tweet_id", how="left")
    out["is_stream"] = out["archive_upload_id"].isna().astype(np.int8)
    return out


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


def within_account_primary_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Locked primaries: within-author Spearman by reply stratum."""
    if len(df) == 0:
        nan = float("nan")
        return {
            "spearman_within_account_nonreply_mean": nan,
            "spearman_within_account_nonreply_wmean": nan,
            "spearman_within_account_nonreply_n": 0.0,
            "spearman_within_account_reply_mean": nan,
            "spearman_within_account_reply_wmean": nan,
            "spearman_within_account_reply_n": 0.0,
            "spearman_nonreply": nan,
            "spearman_reply": nan,
            "rmse_log": nan,
            "n": 0.0,
        }
    y = df["y_true_log"].to_numpy(dtype=np.float64)
    p = df["y_pred_log"].to_numpy(dtype=np.float64)
    acct = df["account_id"].astype(str).to_numpy()
    nr = df["is_reply"].to_numpy() == 0
    rp = ~nr
    wa_nr = within_account_spearman(y[nr], p[nr], acct[nr])
    wa_rp = within_account_spearman(y[rp], p[rp], acct[rp])
    return {
        "spearman_within_account_nonreply_mean": wa_nr["mean"],
        "spearman_within_account_nonreply_wmean": wa_nr["weighted_mean"],
        "spearman_within_account_nonreply_n": wa_nr["n_accounts"],
        "spearman_within_account_reply_mean": wa_rp["mean"],
        "spearman_within_account_reply_wmean": wa_rp["weighted_mean"],
        "spearman_within_account_reply_n": wa_rp["n_accounts"],
        "spearman_nonreply": float(spearmanr(y[nr], p[nr]).correlation) if nr.any() else float("nan"),
        "spearman_reply": float(spearmanr(y[rp], p[rp]).correlation) if rp.any() else float("nan"),
        "rmse_log": _rmse(y, p),
        "n": float(len(df)),
    }


def _print_primary(label: str, primary: dict[str, float]) -> None:
    print(f"{label}:")
    print(
        f"  NR  wmean={primary['spearman_within_account_nonreply_wmean']:.4f} "
        f"mean={primary['spearman_within_account_nonreply_mean']:.4f} "
        f"(n={int(primary['spearman_within_account_nonreply_n'])})"
    )
    print(
        f"  reply wmean={primary['spearman_within_account_reply_wmean']:.4f} "
        f"mean={primary['spearman_within_account_reply_mean']:.4f} "
        f"(n={int(primary['spearman_within_account_reply_n'])})"
    )
    print(
        f"  guardrails: global_nr={primary['spearman_nonreply']:.4f} "
        f"global_reply={primary['spearman_reply']:.4f} rmse={primary['rmse_log']:.4f} "
        f"rows={int(primary.get('n', float('nan')))}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        type=Path,
        default=VAL_PRED_PATH,
        help="val_predictions.parquet from train.py",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ANALYSIS_PATH,
        help="Write JSON analysis here",
    )
    args = ap.parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"Missing {args.predictions}. Run: uv run train.py")

    df = pd.read_parquet(args.predictions)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = ensure_is_stream(df)
    df["eng_tercile"] = pd.qcut(df["y_true_log"], 3, labels=["low", "mid", "high"], duplicates="drop")
    primary = within_account_primary_metrics(df)

    provenance_primary: dict[str, dict[str, float]] = {}
    if "is_stream" in df.columns:
        upload = df[df["is_stream"] == 0]
        stream = df[df["is_stream"] == 1]
        provenance_primary["upload"] = within_account_primary_metrics(upload)
        provenance_primary["stream"] = within_account_primary_metrics(stream)

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
    if "is_stream" in df.columns:
        slices.append(slice_metrics(df[df["is_stream"] == 0], "upload"))
        slices.append(slice_metrics(df[df["is_stream"] == 1], "stream"))
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
                "frac_stream": float(g["is_stream"].mean()) if "is_stream" in g else float("nan"),
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
                    "is_stream",
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
                "is_stream",
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
        "primary_within_account": primary,
        "primary_within_account_by_provenance": provenance_primary,
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
    if "is_stream" in df.columns:
        analysis["frac_stream"] = float(df["is_stream"].mean())
    args.out.write_text(json.dumps(analysis, indent=2, default=str) + "\n")

    _print_primary("Primary (within-account Spearman)", primary)
    if provenance_primary:
        print(
            f"\nProvenance mix: stream={df['is_stream'].mean():.1%} "
            f"upload-linked={(1.0 - df['is_stream'].mean()):.1%}"
        )
        for name, metrics in provenance_primary.items():
            _print_primary(f"Primary within-account ({name})", metrics)
    print("\nSlice metrics:")
    print(pd.DataFrame(slices).to_string(index=False))
    print(f"\nOver-predict (err_log>0.5): {over:.1%}")
    print(f"Under-predict (err_log<-0.5): {under:.1%}")
    print(f"Non-reply under-predict: {nr_under_frac:.1%}")
    print(f"Non-reply RMSE gain vs baseline: {model_gain:.4f}")
    print("\nWorst absolute misses:")
    print(pd.DataFrame(worst_rows).to_string(index=False))
    print("\nBiggest non-reply underpredicts (viral misses):")
    print(pd.DataFrame(viral_misses).to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
