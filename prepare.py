"""
Build a leakage-safe tweet-engagement dataset from the Community Archive.

Downloads the enriched_tweets parquet dump, samples accounts via DuckDB (so the
full dump never has to live in pandas), engineers pre-tweet author/structure
features, embeds tweet text with a frozen MiniLM encoder, and writes a training
table under ~/.cache/will-it-bang/.

Usage:
    uv run prepare.py
    uv run prepare.py --max-tweets 20000 --per-account-cap 250 --min-history 5
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~/.cache/will-it-bang"))
RAW_PARQUET = CACHE_DIR / "enriched_tweets.parquet"
DATASET_PATH = CACHE_DIR / "dataset.parquet"
EMBED_CACHE_PATH = CACHE_DIR / "embed_cache.parquet"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Apache-2.0
PARQUET_URL = (
    "https://fabxmporizzqflnftavs.supabase.co/storage/v1/object/public/"
    "enriched_tweets/enriched_tweets.parquet"
)
CA_API = "https://fabxmporizzqflnftavs.supabase.co/rest/v1"
CA_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0."
    "UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
)

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_parquet(force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PARQUET.exists() and not force:
        print(f"Using cached dump: {RAW_PARQUET}")
        return RAW_PARQUET

    print(f"Downloading {PARQUET_URL}")
    tmp = RAW_PARQUET.with_suffix(".parquet.tmp")
    with requests.get(PARQUET_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    tmp.replace(RAW_PARQUET)
    return RAW_PARQUET


# ---------------------------------------------------------------------------
# Sampling (DuckDB — avoids loading 14M rows into pandas)
# ---------------------------------------------------------------------------

def sample_accounts(
    *,
    max_tweets: int,
    min_account_tweets: int,
    max_account_tweets: int,
    per_account_cap: int,
    seed: int,
) -> pd.DataFrame:
    """Sample many accounts with a per-account tweet cap for diversity."""
    path = str(RAW_PARQUET)
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
    optional = {
        "reply_to_tweet_id": "reply_to_tweet_id",
        "reply_to_user_id": "reply_to_user_id",
        "quoted_tweet_id": "quoted_tweet_id",
        "username": "username",
    }
    select_bits = [
        "tweet_id",
        "account_id",
        "created_at",
        "full_text",
        "favorite_count",
        "retweet_count",
    ]
    for name, expr in optional.items():
        select_bits.append(expr if name in cols else f"NULL AS {name}")
    select_sql = ", ".join(select_bits)

    print("Scanning dump for eligible accounts…")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE eligible AS
        SELECT account_id, COUNT(*) AS n
        FROM read_parquet('{path}')
        WHERE full_text IS NOT NULL
          AND full_text NOT LIKE 'RT @%'
        GROUP BY account_id
        HAVING COUNT(*) BETWEEN {min_account_tweets} AND {max_account_tweets}
        """
    )
    n_eligible = con.execute("SELECT COUNT(*) FROM eligible").fetchone()[0]
    print(f"  eligible accounts: {n_eligible:,}")

    # Pick accounts until capped contribution hits max_tweets.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE picked AS
        WITH ranked AS (
          SELECT
            account_id,
            n,
            LEAST(n, {per_account_cap}) AS contrib,
            SUM(LEAST(n, {per_account_cap})) OVER (
              ORDER BY hash(account_id || '{seed}')
              ROWS UNBOUNDED PRECEDING
            ) AS cum_n
          FROM eligible
        )
        SELECT account_id, contrib
        FROM ranked
        WHERE cum_n - contrib < {max_tweets}
        """
    )
    n_picked = con.execute("SELECT COUNT(*) FROM picked").fetchone()[0]
    print(f"  picked accounts: {n_picked:,} (per-account cap={per_account_cap})")

    # Keep the most recent `contrib` originals per account (better for time-split overlap).
    df = con.execute(
        f"""
        WITH ranked_tweets AS (
          SELECT
            {select_sql},
            p.contrib,
            ROW_NUMBER() OVER (
              PARTITION BY t.account_id
              ORDER BY t.created_at DESC
            ) AS rn
          FROM read_parquet('{path}') t
          JOIN picked p USING (account_id)
          WHERE t.full_text IS NOT NULL
            AND t.full_text NOT LIKE 'RT @%'
        )
        SELECT * EXCLUDE (contrib, rn)
        FROM ranked_tweets
        WHERE rn <= contrib
        ORDER BY account_id, created_at
        """
    ).fetchdf()
    con.close()

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    print(f"Sampled {len(df):,} original tweets from {df['account_id'].nunique():,} accounts")
    return df


def _ca_headers() -> dict[str, str]:
    return {
        "apikey": CA_ANON_KEY,
        "Authorization": f"Bearer {CA_ANON_KEY}",
    }


def _fetch_id_counts(table: str, id_col: str, tweet_ids: list[str], batch_size: int = 80) -> dict[str, int]:
    """Count rows per tweet_id in a CA table (media/urls)."""
    counts: dict[str, int] = {}
    session = requests.Session()
    for start in range(0, len(tweet_ids), batch_size):
        batch = tweet_ids[start : start + batch_size]
        # PostgREST `in` filter; quote ids for safety.
        joined = ",".join(f'"{tid}"' for tid in batch)
        url = f"{CA_API}/{table}"
        params = {"select": id_col, id_col: f"in.({joined})"}
        resp = session.get(url, headers=_ca_headers(), params=params, timeout=60)
        resp.raise_for_status()
        for row in resp.json():
            tid = str(row[id_col])
            counts[tid] = counts.get(tid, 0) + 1
        if start == 0 or (start // batch_size) % 20 == 0:
            print(f"  {table}: {min(start + batch_size, len(tweet_ids)):,}/{len(tweet_ids):,}")
    return counts


def attach_media_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Join sparse CA tweet_media / tweet_urls presence flags (known at post time)."""
    df = df.copy()
    ids = df["tweet_id"].astype(str).tolist()
    print(f"Fetching attachment flags for {len(ids):,} tweets…")
    media_counts = _fetch_id_counts("tweet_media", "tweet_id", ids)
    url_counts = _fetch_id_counts("tweet_urls", "tweet_id", ids)
    df["n_media"] = df["tweet_id"].astype(str).map(lambda t: media_counts.get(t, 0)).astype(np.int16)
    df["has_media"] = (df["n_media"] > 0).astype(np.int8)
    df["n_ca_urls"] = df["tweet_id"].astype(str).map(lambda t: url_counts.get(t, 0)).astype(np.int16)
    df["has_ca_url"] = (df["n_ca_urls"] > 0).astype(np.int8)
    print(
        f"  media coverage: {df['has_media'].mean():.1%}, "
        f"ca_url coverage: {df['has_ca_url'].mean():.1%}"
    )
    return df


# ---------------------------------------------------------------------------
# Leakage-safe features
# ---------------------------------------------------------------------------

def _text_stats(text: str) -> dict[str, float]:
    if not isinstance(text, str) or not text:
        return {
            "text_len": 0,
            "has_url": 0,
            "n_mentions": 0,
            "n_hashtags": 0,
            "has_question": 0,
            "has_exclaim": 0,
            "n_emoji": 0,
            "starts_mention": 0,
            "caps_ratio": 0.0,
        }
    letters = [c for c in text if c.isalpha()]
    caps_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
    return {
        "text_len": len(text),
        "has_url": int(bool(URL_RE.search(text))),
        "n_mentions": len(MENTION_RE.findall(text)),
        "n_hashtags": len(HASHTAG_RE.findall(text)),
        "has_question": int("?" in text),
        "has_exclaim": int("!" in text),
        "n_emoji": len(EMOJI_RE.findall(text)),
        "starts_mention": int(text.lstrip().startswith("@")),
        "caps_ratio": caps_ratio,
    }


def build_features(df: pd.DataFrame, min_history: int) -> pd.DataFrame:
    """Per-tweet features using only strictly earlier tweets by the same author."""
    df = df.copy()
    df["favorite_count"] = pd.to_numeric(df["favorite_count"], errors="coerce").fillna(0).astype(np.int64)
    df["retweet_count"] = pd.to_numeric(df["retweet_count"], errors="coerce").fillna(0).astype(np.int64)
    df["engagement"] = df["favorite_count"] + df["retweet_count"]
    df["log_engagement"] = np.log1p(df["engagement"].astype(np.float64))

    stats = df["full_text"].map(_text_stats)
    for key in next(iter(stats)).keys() if len(stats) else []:
        df[key] = stats.map(lambda d, k=key: d[k])
    # Robust if empty (shouldn't happen).
    for key, default in (
        ("text_len", 0),
        ("has_url", 0),
        ("n_mentions", 0),
        ("n_hashtags", 0),
        ("has_question", 0),
        ("has_exclaim", 0),
        ("n_emoji", 0),
        ("starts_mention", 0),
        ("caps_ratio", 0.0),
    ):
        if key not in df.columns:
            df[key] = default

    df["is_reply"] = df["reply_to_tweet_id"].notna().astype(np.int8)
    df["is_quote"] = df["quoted_tweet_id"].notna().astype(np.int8)
    df["is_self_reply"] = (
        df["reply_to_user_id"].notna()
        & (df["reply_to_user_id"].astype(str) == df["account_id"].astype(str))
    ).astype(np.int8)
    if "has_media" not in df.columns:
        df["has_media"] = 0
        df["n_media"] = 0
        df["has_ca_url"] = 0
        df["n_ca_urls"] = 0
    df["hour"] = df["created_at"].dt.hour.astype(np.int8)
    df["dow"] = df["created_at"].dt.dayofweek.astype(np.int8)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    # Cyclical time encodings (leakage-free calendar features).
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    rows: list[dict] = []
    for account_id, g in df.groupby("account_id", sort=False):
        g = g.sort_values("created_at", kind="mergesort")
        times = g["created_at"].to_numpy()
        log_eng = g["log_engagement"].to_numpy(dtype=np.float64)
        is_reply = g["is_reply"].to_numpy(dtype=np.int8)
        has_url = g["has_url"].to_numpy(dtype=np.int8)
        has_media = g["has_media"].to_numpy(dtype=np.int8)

        for i in range(len(g)):
            if i < min_history:
                continue
            prior_e = log_eng[:i]
            prior_reply = is_reply[:i]
            age_days = (times[i] - times[:i]).astype("timedelta64[s]").astype(np.float64) / 86400.0
            mask_7 = age_days <= 7.0
            mask_30 = age_days <= 30.0
            non_reply = prior_reply == 0
            reply_only = prior_reply == 1

            # Exponentially decayed mean over prior tweets (half-life 14d).
            weights = np.exp(-np.log(2.0) * age_days / 14.0)
            decay_mean = float(np.average(prior_e, weights=weights))

            row = g.iloc[i]
            hist_mean = float(prior_e.mean())
            hist_mean_nr = float(prior_e[non_reply].mean()) if non_reply.any() else hist_mean
            rows.append(
                {
                    "tweet_id": row["tweet_id"],
                    "account_id": account_id,
                    "username": row["username"],
                    "created_at": row["created_at"],
                    "full_text": row["full_text"],
                    "engagement": int(row["engagement"]),
                    "log_engagement": float(row["log_engagement"]),
                    "text_len": int(row["text_len"]),
                    "log_text_len": float(np.log1p(row["text_len"])),
                    "has_url": int(row["has_url"]),
                    "n_mentions": int(row["n_mentions"]),
                    "n_hashtags": int(row["n_hashtags"]),
                    "has_question": int(row["has_question"]),
                    "has_exclaim": int(row["has_exclaim"]),
                    "n_emoji": int(row["n_emoji"]),
                    "starts_mention": int(row["starts_mention"]),
                    "caps_ratio": float(row["caps_ratio"]),
                    "has_media": int(row["has_media"]),
                    "n_media": int(row["n_media"]),
                    "has_ca_url": int(row["has_ca_url"]),
                    "n_ca_urls": int(row["n_ca_urls"]),
                    "is_reply": int(row["is_reply"]),
                    "is_quote": int(row["is_quote"]),
                    "is_self_reply": int(row["is_self_reply"]),
                    "hour": int(row["hour"]),
                    "dow": int(row["dow"]),
                    "is_weekend": int(row["is_weekend"]),
                    "hour_sin": float(row["hour_sin"]),
                    "hour_cos": float(row["hour_cos"]),
                    "dow_sin": float(row["dow_sin"]),
                    "dow_cos": float(row["dow_cos"]),
                    "hist_n": i,
                    "hist_mean_eng": hist_mean,
                    "hist_median_eng": float(np.median(prior_e)),
                    "hist_p90_eng": float(np.quantile(prior_e, 0.9)),
                    "hist_last_eng": float(prior_e[-1]),
                    "hist_std_eng": float(prior_e.std(ddof=0)) if i > 1 else 0.0,
                    "hist_mean_eng_7d": float(prior_e[mask_7].mean()) if mask_7.any() else 0.0,
                    "hist_mean_eng_30d": float(prior_e[mask_30].mean()) if mask_30.any() else 0.0,
                    "hist_decay_mean_eng": decay_mean,
                    "hist_mean_nonreply": hist_mean_nr,
                    "hist_mean_reply": float(prior_e[reply_only].mean()) if reply_only.any() else hist_mean,
                    "hist_tweets_7d": int(mask_7.sum()),
                    "hist_tweets_30d": int(mask_30.sum()),
                    "days_since_last": float(age_days[-1]),
                    "hist_frac_reply": float(prior_reply.mean()),
                    "hist_frac_url": float(has_url[:i].mean()),
                    "hist_frac_media": float(has_media[:i].mean()),
                    # Interactions: reply status vs the author's typical non-reply baseline.
                    "reply_x_hist_nonreply": float(row["is_reply"]) * hist_mean_nr,
                    "media_x_hist_mean": float(row["has_media"]) * hist_mean,
                }
            )

    out = pd.DataFrame(rows)
    print(f"Feature rows (min_history={min_history}): {len(out):,}")
    return out


# ---------------------------------------------------------------------------
# Text embeddings (frozen encoder) with tweet_id cache
# ---------------------------------------------------------------------------

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def embed_texts(texts: list[str], batch_size: int = 128) -> np.ndarray:
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding {len(texts):,} texts with {EMBED_MODEL_ID} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    model = AutoModel.from_pretrained(EMBED_MODEL_ID).to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
        if (start // batch_size) % 10 == 0:
            done = min(start + batch_size, len(texts))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  embedded {done:,}/{len(texts):,} ({rate:.1f} tw/s)")

    return np.vstack(vectors)


def load_embed_cache() -> pd.DataFrame | None:
    if not EMBED_CACHE_PATH.exists():
        return None
    cache = pd.read_parquet(EMBED_CACHE_PATH)
    print(f"Loaded embed cache: {len(cache):,} rows")
    return cache


def save_embed_cache(cache: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(EMBED_CACHE_PATH, index=False)
    print(f"Wrote embed cache: {EMBED_CACHE_PATH} ({len(cache):,} rows)")


def attach_embeddings(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    cache = load_embed_cache()
    ids = df["tweet_id"].astype(str)

    if cache is not None:
        cache = cache.copy()
        cache["tweet_id"] = cache["tweet_id"].astype(str)
        emb_cols = [c for c in cache.columns if c.startswith("emb_")]
        hit = ids.isin(set(cache["tweet_id"]))
    else:
        emb_cols = []
        hit = pd.Series(False, index=df.index)

    miss_idx = np.flatnonzero(~hit.to_numpy())
    print(f"Embeddings: {int(hit.sum()):,} cached, {len(miss_idx):,} to compute")

    if len(miss_idx):
        texts = df.iloc[miss_idx]["full_text"].fillna("").astype(str).tolist()
        new_emb = embed_texts(texts, batch_size=batch_size)
        new_df = pd.DataFrame(new_emb, columns=[f"emb_{i}" for i in range(new_emb.shape[1])])
        new_df.insert(0, "tweet_id", ids.iloc[miss_idx].to_numpy())
        cache = new_df if cache is None else pd.concat([cache, new_df], ignore_index=True)
        cache = cache.drop_duplicates("tweet_id", keep="last")
        save_embed_cache(cache)
        emb_cols = [c for c in cache.columns if c.startswith("emb_")]

    assert cache is not None and emb_cols
    merged = pd.DataFrame({"tweet_id": ids}).merge(cache[["tweet_id", *emb_cols]], on="tweet_id", how="left")
    if merged[emb_cols[0]].isna().any():
        raise RuntimeError("Missing embeddings after cache merge")
    return pd.concat([df.reset_index(drop=True), merged[emb_cols].reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-tweets", type=int, default=25000, help="Cap on sampled original tweets")
    p.add_argument("--per-account-cap", type=int, default=200, help="Max tweets kept per account")
    p.add_argument("--min-account-tweets", type=int, default=80)
    p.add_argument("--max-account-tweets", type=int, default=5000)
    p.add_argument("--min-history", type=int, default=5, help="Require this many prior tweets")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-media", action="store_true", help="Skip CA media/url API joins")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    download_parquet(force=args.force_download)
    sampled = sample_accounts(
        max_tweets=args.max_tweets,
        min_account_tweets=args.min_account_tweets,
        max_account_tweets=args.max_account_tweets,
        per_account_cap=args.per_account_cap,
        seed=args.seed,
    )
    if not args.skip_media:
        sampled = attach_media_flags(sampled)
    featured = build_features(sampled, min_history=args.min_history)
    del sampled

    dataset = attach_embeddings(featured, batch_size=args.embed_batch_size)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)
    print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
