"""
Adversarial CA corpus quality audit (upload vs stream regimes).

Reads the enriched_tweets parquet (local cache or remote URL), writes monthly /
yearly DQ tables under --out-dir, and prints holdout-oriented summaries that
matter for will-it-bang (null archive_upload_id, reply_to_user_id dropout).

Usage:
    uv run audit_ca_quality.py
    uv run audit_ca_quality.py --parquet ~/.cache/will-it-bang/enriched_tweets.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd

PARQUET_URL = (
    "https://fabxmporizzqflnftavs.supabase.co/storage/v1/object/public/"
    "enriched_tweets/enriched_tweets.parquet"
)
DEFAULT_CACHE = Path(os.path.expanduser("~/.cache/will-it-bang/enriched_tweets.parquet"))


def resolve_parquet(path: Path | None) -> str:
    if path is not None and path.exists():
        return str(path)
    if DEFAULT_CACHE.exists():
        return str(DEFAULT_CACHE)
    return PARQUET_URL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/ca_dq"),
        help="Directory for CSV summaries",
    )
    ap.add_argument("--min-account-tweets", type=int, default=50)
    ap.add_argument("--max-account-tweets", type=int, default=8000)
    args = ap.parse_args()

    src = resolve_parquet(args.parquet)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source: {src}")

    con = duckdb.connect()
    if src.startswith("http"):
        con.execute("INSTALL httpfs; LOAD httpfs;")

    monthly = con.execute(
        f"""
        SELECT
          strftime(CAST(created_at AS TIMESTAMPTZ), '%Y-%m') AS month,
          COUNT(*) AS n,
          SUM(CASE WHEN archive_upload_id IS NULL THEN 1 ELSE 0 END) AS n_null_upload,
          SUM(CASE WHEN reply_to_tweet_id IS NOT NULL THEN 1 ELSE 0 END) AS n_reply,
          SUM(
            CASE WHEN reply_to_tweet_id IS NOT NULL AND reply_to_user_id IS NULL
            THEN 1 ELSE 0 END
          ) AS reply_miss_uid,
          SUM(CASE WHEN quoted_tweet_id IS NOT NULL THEN 1 ELSE 0 END) AS n_quote,
          SUM(CASE WHEN conversation_id IS NULL THEN 1 ELSE 0 END) AS n_null_conv,
          SUM(
            CASE WHEN COALESCE(favorite_count, 0) = 0
                  AND COALESCE(retweet_count, 0) = 0
            THEN 1 ELSE 0 END
          ) AS n_zero_eng
        FROM read_parquet('{src}')
        WHERE CAST(created_at AS TIMESTAMPTZ) >= TIMESTAMPTZ '2023-01-01'
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    monthly_path = args.out_dir / "monthly_dq.csv"
    monthly.to_csv(monthly_path, index=False)

    yearly = con.execute(
        f"""
        SELECT
          strftime(CAST(created_at AS TIMESTAMPTZ), '%Y') AS year,
          COUNT(*) AS n,
          ROUND(
            100.0 * SUM(CASE WHEN archive_upload_id IS NULL THEN 1 ELSE 0 END)
            / COUNT(*), 2
          ) AS pct_null_upload,
          ROUND(
            100.0 * SUM(
              CASE WHEN reply_to_tweet_id IS NOT NULL AND reply_to_user_id IS NULL
              THEN 1 ELSE 0 END
            ) / NULLIF(
              SUM(CASE WHEN reply_to_tweet_id IS NOT NULL THEN 1 ELSE 0 END), 0
            ),
            2
          ) AS pct_reply_miss_uid,
          ROUND(
            100.0 * SUM(CASE WHEN conversation_id IS NULL THEN 1 ELSE 0 END)
            / COUNT(*), 2
          ) AS pct_null_conv
        FROM read_parquet('{src}')
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    yearly_path = args.out_dir / "yearly_dq.csv"
    yearly.to_csv(yearly_path, index=False)

    lo, hi = args.min_account_tweets, args.max_account_tweets
    eligible_year = con.execute(
        f"""
        WITH eligible AS (
          SELECT account_id
          FROM read_parquet('{src}')
          WHERE full_text IS NOT NULL AND full_text NOT LIKE 'RT @%'
          GROUP BY 1
          HAVING COUNT(*) BETWEEN {lo} AND {hi}
        ),
        t AS (
          SELECT
            CAST(created_at AS TIMESTAMPTZ) AS ts,
            archive_upload_id,
            reply_to_tweet_id,
            reply_to_user_id,
            account_id
          FROM read_parquet('{src}') e
          SEMI JOIN eligible a ON e.account_id = a.account_id
          WHERE full_text IS NOT NULL AND full_text NOT LIKE 'RT @%'
        )
        SELECT
          strftime(ts, '%Y') AS year,
          COUNT(*) AS n,
          ROUND(
            100.0 * SUM(CASE WHEN archive_upload_id IS NULL THEN 1 ELSE 0 END)
            / COUNT(*), 1
          ) AS pct_stream,
          ROUND(
            100.0 * SUM(
              CASE WHEN reply_to_tweet_id IS NOT NULL AND reply_to_user_id IS NULL
              THEN 1 ELSE 0 END
            ) / NULLIF(
              SUM(CASE WHEN reply_to_tweet_id IS NOT NULL THEN 1 ELSE 0 END), 0
            ),
            1
          ) AS pct_reply_miss_uid,
          COUNT(DISTINCT account_id) AS n_accounts
        FROM t
        WHERE ts >= TIMESTAMPTZ '2020-01-01'
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    eligible_path = args.out_dir / "eligible_yearly_dq.csv"
    eligible_year.to_csv(eligible_path, index=False)

    holdout = con.execute(
        f"""
        WITH eligible AS (
          SELECT account_id
          FROM read_parquet('{src}')
          WHERE full_text IS NOT NULL AND full_text NOT LIKE 'RT @%'
          GROUP BY 1
          HAVING COUNT(*) BETWEEN {lo} AND {hi}
        ),
        t AS (
          SELECT *, CAST(created_at AS TIMESTAMPTZ) AS ts
          FROM read_parquet('{src}') e
          SEMI JOIN eligible a ON e.account_id = a.account_id
          WHERE full_text IS NOT NULL AND full_text NOT LIKE 'RT @%'
        ),
        ordered AS (
          SELECT *, ntile(5) OVER (ORDER BY ts) AS quint
          FROM t
        )
        SELECT
          quint,
          MIN(ts) AS min_ts,
          MAX(ts) AS max_ts,
          COUNT(*) AS n,
          ROUND(
            100.0 * SUM(CASE WHEN archive_upload_id IS NULL THEN 1 ELSE 0 END)
            / COUNT(*), 1
          ) AS pct_stream,
          ROUND(
            100.0 * SUM(
              CASE WHEN reply_to_tweet_id IS NOT NULL AND reply_to_user_id IS NULL
              THEN 1 ELSE 0 END
            ) / NULLIF(
              SUM(CASE WHEN reply_to_tweet_id IS NOT NULL THEN 1 ELSE 0 END), 0
            ),
            1
          ) AS pct_reply_miss_uid
        FROM ordered
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    holdout_path = args.out_dir / "time_quintile_dq.csv"
    holdout.to_csv(holdout_path, index=False)

    totals = con.execute(
        f"""
        SELECT
          COUNT(*) AS n,
          COUNT(archive_upload_id) AS n_with_upload,
          COUNT(*) - COUNT(archive_upload_id) AS n_null_upload,
          COUNT(
            CASE WHEN reply_to_tweet_id IS NOT NULL AND reply_to_user_id IS NULL
            THEN 1 END
          ) AS reply_miss_uid,
          MIN(created_at) AS min_created,
          MAX(created_at) AS max_created
        FROM read_parquet('{src}')
        """
    ).fetchdf()

    print("\n=== Dump totals ===")
    print(totals.to_string(index=False))
    print("\n=== Yearly (% null upload / % reply miss uid) ===")
    print(yearly.to_string(index=False))
    print(
        f"\n=== Eligible accounts ({lo}–{hi} non-RT tweets): yearly stream share ==="
    )
    print(eligible_year.to_string(index=False))
    print("\n=== Global time quintiles on eligible pool (last quint ≈ holdout) ===")
    print(holdout.to_string(index=False))
    print(
        f"\nWrote:\n  {monthly_path}\n  {yearly_path}\n  {eligible_path}\n  {holdout_path}"
    )
    print("Narrative: logs/ca_data_quality_audit.md")


if __name__ == "__main__":
    main()
