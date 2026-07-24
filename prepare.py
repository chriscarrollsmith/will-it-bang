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
MENTION_CACHE_PATH = CACHE_DIR / "mention_cache.parquet"
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
    # Intentionally omit account_display_name (mutable; no as-of-T history).
    # username is kept for joins/debug only — not a model feature (see AGENTS.md).
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


def _fetch_id_counts(table: str, id_col: str, tweet_ids: list[str], batch_size: int = 120) -> dict[str, int]:
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


def refresh_media_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute hist_frac_media and media×interaction cols after refreshing media flags."""
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    out = out.sort_values(["account_id", "created_at", "tweet_id"], kind="mergesort").reset_index(drop=True)
    frac = np.zeros(len(out), dtype=np.float64)
    for _, g in out.groupby("account_id", sort=False):
        ix = g.index.to_numpy()
        hm = out.loc[ix, "has_media"].to_numpy(dtype=np.float64)
        csum = np.concatenate([[0.0], np.cumsum(hm[:-1])]) if len(hm) else np.array([])
        n = np.arange(len(hm), dtype=np.float64)
        vals = np.divide(csum, n, out=np.zeros_like(csum), where=n > 0)
        frac[ix] = vals
    out["hist_frac_media"] = frac
    if "hist_mean_replies" in out.columns:
        out["media_x_hist_replies"] = out["has_media"].astype(np.float64) * out["hist_mean_replies"]
    if "inbound_replies" in out.columns:
        out["media_x_inbound_replies"] = out["has_media"].astype(np.float64) * out["inbound_replies"]
    print(
        f"Refreshed media-derived cols: hist_frac_media mean={out['hist_frac_media'].mean():.3f}, "
        f"has_media={out['has_media'].mean():.1%}",
        flush=True,
    )
    return out


def _fetch_tweet_urls(tweet_ids: list[str], batch_size: int = 80) -> pd.DataFrame:
    """Fetch url / expanded_url rows from CA tweet_urls for the given tweet ids."""
    rows: list[dict[str, str]] = []
    session = requests.Session()
    for start in range(0, len(tweet_ids), batch_size):
        batch = tweet_ids[start : start + batch_size]
        joined = ",".join(f'"{tid}"' for tid in batch)
        resp = session.get(
            f"{CA_API}/tweet_urls",
            headers=_ca_headers(),
            params={"select": "tweet_id,url,expanded_url", "tweet_id": f"in.({joined})"},
            timeout=60,
        )
        resp.raise_for_status()
        for row in resp.json():
            rows.append(
                {
                    "tweet_id": str(row["tweet_id"]),
                    "url": "" if row.get("url") is None else str(row["url"]),
                    "expanded_url": (
                        "" if row.get("expanded_url") is None else str(row["expanded_url"])
                    ),
                }
            )
        if start == 0 or (start // batch_size) % 25 == 0:
            print(f"  tweet_urls: {min(start + batch_size, len(tweet_ids)):,}/{len(tweet_ids):,}")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["tweet_id", "url", "expanded_url"])


def _domain_from_url(raw: str) -> str:
    from urllib.parse import urlparse

    if not raw:
        return ""
    try:
        host = urlparse(raw if "://" in raw else f"https://{raw}").netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def attach_url_domains(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Join CA expanded URL domains (and text-url structure). Cache by tweet_id."""
    url_cache_path = CACHE_DIR / "url_cache.parquet"
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)

    texts = out["full_text"].fillna("").astype(str)
    http_re = re.compile(r"https?://\S+", re.IGNORECASE)
    n_text_urls = texts.map(lambda t: len(http_re.findall(t))).astype(np.int16)
    out["n_text_urls"] = n_text_urls
    stripped = texts.map(lambda t: http_re.sub(" ", t))
    stripped = stripped.map(lambda t: re.sub(r"\s+", " ", t).strip())
    out["url_only"] = ((n_text_urls > 0) & (stripped.str.len() <= 8)).astype(np.int8)
    out["url_at_end"] = texts.map(
        lambda t: int(bool(re.search(r"https?://\S+\s*$", t, flags=re.IGNORECASE)))
    ).astype(np.int8)

    need_ids = out.loc[out["n_text_urls"] > 0, "tweet_id"].unique().tolist()
    cache = None
    if url_cache_path.exists() and not force:
        cache = pd.read_parquet(url_cache_path)
        cache["tweet_id"] = cache["tweet_id"].astype(str)
        print(f"Loaded URL cache: {len(cache):,} rows")
    cached_ids = set(cache["tweet_id"]) if cache is not None else set()
    miss = [tid for tid in need_ids if tid not in cached_ids]
    print(f"URL domains: {len(need_ids) - len(miss):,} cached tweets, {len(miss):,} to fetch")
    if miss:
        fetched = _fetch_tweet_urls(miss)
        if not fetched.empty:
            fetched["domain"] = fetched.apply(
                lambda r: _domain_from_url(r["expanded_url"]) or _domain_from_url(r["url"]),
                axis=1,
            )
            primary = (
                fetched.loc[fetched["domain"] != ""]
                .groupby("tweet_id", sort=False)["domain"]
                .first()
                .reset_index()
            )
            n_urls = fetched.groupby("tweet_id").size().rename("n_ca_url_rows").reset_index()
            new_cache = primary.merge(n_urls, on="tweet_id", how="outer")
            new_cache["domain"] = new_cache["domain"].fillna("")
            new_cache["n_ca_url_rows"] = new_cache["n_ca_url_rows"].fillna(0).astype(np.int16)
            cache = new_cache if cache is None else pd.concat([cache, new_cache], ignore_index=True)
            cache = cache.drop_duplicates("tweet_id", keep="last")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.to_parquet(url_cache_path, index=False)
            print(f"Wrote URL cache: {url_cache_path} ({len(cache):,} rows)")

    if cache is None:
        cache = pd.DataFrame(columns=["tweet_id", "domain", "n_ca_url_rows"])

    merged = out[["tweet_id"]].merge(cache, on="tweet_id", how="left")
    domain = merged["domain"].fillna("").astype(str)
    out["url_domain"] = domain
    out["has_expanded_domain"] = (domain != "").astype(np.int8)
    out["n_ca_url_rows"] = merged["n_ca_url_rows"].fillna(0).astype(np.int16)

    out["domain_is_twitter"] = domain.str.contains(
        r"(?:^|\.)(?:t\.co|twitter\.com|x\.com)$", regex=True, na=False
    ).astype(np.int8)
    out["domain_is_youtube"] = domain.str.contains(
        r"(?:^|\.)(?:youtube\.com|youtu\.be)$", regex=True, na=False
    ).astype(np.int8)
    out["domain_is_github"] = domain.str.contains(
        r"(?:^|\.)(?:github\.com|github\.io)$", regex=True, na=False
    ).astype(np.int8)
    out["domain_is_substack"] = domain.str.contains(
        r"(?:^|\.)(?:substack\.com)$", regex=True, na=False
    ).astype(np.int8)
    out["domain_is_wiki"] = domain.str.contains(
        r"(?:^|\.)(?:wikipedia\.org|wikimedia\.org)$", regex=True, na=False
    ).astype(np.int8)

    print(
        f"  text-url tweets: {(out['n_text_urls'] > 0).mean():.1%}, "
        f"expanded domain: {out['has_expanded_domain'].mean():.1%}, "
        f"url_only: {out['url_only'].mean():.1%}"
    )
    return out

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


def _sorted_time_map(df: pd.DataFrame, key_col: str) -> dict[str, np.ndarray]:
    if df.empty:
        return {}
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    out: dict[str, np.ndarray] = {}
    for key, g in df.groupby(key_col, sort=False):
        out[str(key)] = np.sort(g["created_at"].to_numpy(dtype="datetime64[ns]"))
    return out


def _count_before(times: np.ndarray | None, cutoff) -> int:
    if times is None or len(times) == 0:
        return 0
    cut = np.datetime64(pd.Timestamp(cutoff).to_datetime64())
    return int(np.searchsorted(times, cut, side="left"))


def _count_window(times: np.ndarray | None, cutoff, days: float) -> int:
    if times is None or len(times) == 0:
        return 0
    cut = np.datetime64(pd.Timestamp(cutoff).to_datetime64())
    start = cut - np.timedelta64(int(days * 86400), "s")
    right = int(np.searchsorted(times, cut, side="left"))
    left = int(np.searchsorted(times, start, side="left"))
    return max(0, right - left)


def _fetch_mention_events_api(account_ids: list[str], batch_size: int = 40) -> pd.DataFrame:
    """Page CA user_mentions for mentioned_user_id ∈ accounts; join tweet created_at."""
    if not account_ids:
        return pd.DataFrame(columns=["mentioned_user_id", "tweet_id", "created_at"])
    session = requests.Session()
    rows: list[dict[str, object]] = []
    for start in range(0, len(account_ids), batch_size):
        batch = account_ids[start : start + batch_size]
        joined = ",".join(f'"{aid}"' for aid in batch)
        last_id = 0
        page = 0
        while True:
            params = {
                "select": "id,tweet_id,mentioned_user_id,tweets(created_at,account_id)",
                "mentioned_user_id": f"in.({joined})",
                "id": f"gt.{last_id}",
                "order": "id.asc",
                "limit": "1000",
            }
            resp = session.get(
                f"{CA_API}/user_mentions",
                headers=_ca_headers(),
                params=params,
                timeout=120,
            )
            resp.raise_for_status()
            chunk = resp.json()
            if not chunk:
                break
            for row in chunk:
                last_id = max(last_id, int(row["id"]))
                tw = row.get("tweets") or {}
                # Drop self-mentions when the mentioning author is known.
                author = tw.get("account_id")
                mentioned = str(row["mentioned_user_id"])
                if author is not None and str(author) == mentioned:
                    continue
                created = tw.get("created_at")
                rows.append(
                    {
                        "mentioned_user_id": mentioned,
                        "tweet_id": str(row["tweet_id"]),
                        "created_at": created,
                    }
                )
            page += 1
            if len(chunk) < 1000:
                break
            if page % 20 == 0:
                print(
                    f"  user_mentions batch {min(start + batch_size, len(account_ids)):,}/"
                    f"{len(account_ids):,}: {len(rows):,} events (id>{last_id})",
                    flush=True,
                )
        if start == 0 or (start // batch_size) % 5 == 0:
            print(
                f"  user_mentions: {min(start + batch_size, len(account_ids)):,}/"
                f"{len(account_ids):,} accounts → {len(rows):,} events",
                flush=True,
            )
    return pd.DataFrame(rows)


def _backfill_mention_times_from_dump(events: pd.DataFrame) -> pd.DataFrame:
    """Fill missing mention created_at from the enriched_tweets dump when present."""
    if events.empty or not RAW_PARQUET.exists():
        return events
    out = events.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    need = out["created_at"].isna() if "created_at" in out.columns else pd.Series(True, index=out.index)
    if not need.any():
        # Normalize dtype even when complete.
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
        return out
    ids = out.loc[need, "tweet_id"].drop_duplicates()
    tmp = CACHE_DIR / "_mention_tweet_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids.to_frame("tweet_id").to_parquet(tmp, index=False)
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    times = con.execute(
        f"""
        SELECT CAST(t.tweet_id AS VARCHAR) AS tweet_id, t.created_at
        FROM read_parquet('{RAW_PARQUET}') t
        SEMI JOIN read_parquet('{tmp}') i
          ON CAST(t.tweet_id AS VARCHAR) = i.tweet_id
        """
    ).fetchdf()
    con.close()
    try:
        tmp.unlink()
    except OSError:
        pass
    if times.empty:
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
        return out
    times["tweet_id"] = times["tweet_id"].astype(str)
    times["created_at"] = pd.to_datetime(times["created_at"], utc=True, errors="coerce")
    fill = out.loc[need, ["tweet_id"]].merge(times, on="tweet_id", how="left")["created_at"]
    out.loc[need, "created_at"] = fill.to_numpy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
    n_filled = int(out["created_at"].notna().sum())
    print(
        f"  mention dump backfill: {n_filled:,}/{len(out):,} events have created_at "
        f"({out['created_at'].notna().mean():.1%})",
        flush=True,
    )
    return out


def load_mention_times(account_ids: pd.Series, *, force: bool = False) -> dict[str, np.ndarray]:
    """Pre-T mention arrivals: user_mentions ⋊ tweet created_at, cached by mentioned user."""
    accounts = sorted({str(a) for a in account_ids.astype(str).tolist()})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame(columns=["mentioned_user_id", "tweet_id", "created_at"])
    if MENTION_CACHE_PATH.exists() and not force:
        cached = pd.read_parquet(MENTION_CACHE_PATH)
        cached["mentioned_user_id"] = cached["mentioned_user_id"].astype(str)
        cached["tweet_id"] = cached["tweet_id"].astype(str)
        print(f"Mention cache: {len(cached):,} rows ({MENTION_CACHE_PATH.name})")
    have = set(cached["mentioned_user_id"].unique()) if len(cached) else set()
    missing = [a for a in accounts if a not in have]
    if missing:
        print(f"Fetching user_mentions for {len(missing):,} accounts…", flush=True)
        fresh = _fetch_mention_events_api(missing)
        pieces = [cached] if len(cached) else []
        if not fresh.empty:
            fresh = _backfill_mention_times_from_dump(fresh)
            pieces.append(fresh)
        seen = set(fresh["mentioned_user_id"].astype(str).unique()) if not fresh.empty else set()
        still_empty = [a for a in missing if a not in seen]
        if still_empty:
            # Sentinel rows so zero-mention accounts are not refetched every run.
            pieces.append(
                pd.DataFrame(
                    {
                        "mentioned_user_id": still_empty,
                        "tweet_id": [""] * len(still_empty),
                        "created_at": pd.to_datetime([pd.NaT] * len(still_empty), utc=True),
                    }
                )
            )
        cached = pd.concat(pieces, ignore_index=True) if pieces else cached
        if not fresh.empty:
            cached = cached.drop_duplicates(["mentioned_user_id", "tweet_id"], keep="last")
        cached.to_parquet(MENTION_CACHE_PATH, index=False)
        print(
            f"Wrote mention cache: {len(cached):,} rows → {MENTION_CACHE_PATH} "
            f"({len(still_empty):,} zero-mention accounts marked)",
            flush=True,
        )
    else:
        # Still try dump backfill for cached rows missing timestamps.
        need_fill = cached["created_at"].isna() & (cached["tweet_id"].astype(str) != "")
        if need_fill.any():
            cached = _backfill_mention_times_from_dump(cached)
            cached.to_parquet(MENTION_CACHE_PATH, index=False)

    usable = cached[
        cached["mentioned_user_id"].isin(accounts)
        & cached["tweet_id"].astype(str).ne("")
        & cached["created_at"].notna()
    ].copy()
    print(
        f"  usable mention events: {len(usable):,} for {usable['mentioned_user_id'].nunique():,}/"
        f"{len(accounts):,} accounts ({(usable['mentioned_user_id'].nunique() / max(len(accounts), 1)):.1%} coverage)",
        flush=True,
    )
    return _sorted_time_map(usable, "mentioned_user_id")


def load_author_graph(account_ids: pd.Series) -> dict:
    """Leakage-safe inbound graph events from the full dump for sampled authors.

    Returns dict with:
      reply_to_tweet: parent tweet_id → reply times (all author tweets in dump)
      reply_to_user: account_id → times of replies addressed to that user
      quote_to_user: account_id → times of quotes of that user's tweets
      mention_to_user: account_id → times of @-mentions of that user (CA user_mentions ⋊ tweet time)
      author_timeline: account_id → (tweet_ids, created_at ns) for all dump tweets
    """
    accounts = pd.DataFrame({"account_id": account_ids.astype(str).unique()})
    tmp_accounts = CACHE_DIR / "_target_account_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    accounts.to_parquet(tmp_accounts, index=False)
    path = str(RAW_PARQUET)
    print(f"Scanning dump for inbound graph on {len(accounts):,} accounts…")
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE target_accounts AS
        SELECT account_id FROM read_parquet('{tmp_accounts}')
        """
    )
    # Every tweet ever posted by these accounts (broader parent set than the sample).
    author_tweets_df = con.execute(
        f"""
        SELECT
          CAST(t.tweet_id AS VARCHAR) AS tweet_id,
          CAST(t.account_id AS VARCHAR) AS account_id,
          t.created_at
        FROM read_parquet('{path}') t
        SEMI JOIN target_accounts a
          ON CAST(t.account_id AS VARCHAR) = a.account_id
        WHERE t.full_text IS NOT NULL
          AND t.full_text NOT LIKE 'RT @%'
        """
    ).fetchdf()
    con.register("author_tweets_df", author_tweets_df)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE author_tweets AS
        SELECT tweet_id, account_id FROM author_tweets_df
        """
    )
    print(f"  author tweets in dump: {len(author_tweets_df):,}")

    reply_to_tweet_df = con.execute(
        f"""
        SELECT
          CAST(r.reply_to_tweet_id AS VARCHAR) AS parent_id,
          r.created_at
        FROM read_parquet('{path}') r
        SEMI JOIN author_tweets p
          ON CAST(r.reply_to_tweet_id AS VARCHAR) = p.tweet_id
        WHERE r.reply_to_tweet_id IS NOT NULL
        """
    ).fetchdf()

    reply_to_user_df = con.execute(
        f"""
        SELECT
          CAST(r.reply_to_user_id AS VARCHAR) AS account_id,
          r.created_at
        FROM read_parquet('{path}') r
        SEMI JOIN target_accounts a
          ON CAST(r.reply_to_user_id AS VARCHAR) = a.account_id
        WHERE r.reply_to_user_id IS NOT NULL
        """
    ).fetchdf()

    quote_to_user_df = con.execute(
        f"""
        SELECT
          p.account_id,
          r.created_at
        FROM read_parquet('{path}') r
        JOIN author_tweets p
          ON CAST(r.quoted_tweet_id AS VARCHAR) = p.tweet_id
        WHERE r.quoted_tweet_id IS NOT NULL
        """
    ).fetchdf()
    con.close()
    try:
        tmp_accounts.unlink()
    except OSError:
        pass

    reply_to_tweet = _sorted_time_map(reply_to_tweet_df, "parent_id")
    reply_to_user = _sorted_time_map(reply_to_user_df, "account_id")
    quote_to_user = _sorted_time_map(quote_to_user_df, "account_id")

    author_tweets_df["created_at"] = pd.to_datetime(author_tweets_df["created_at"], utc=True)
    author_timeline: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for aid, g in author_tweets_df.groupby("account_id", sort=False):
        g = g.sort_values("created_at", kind="mergesort")
        author_timeline[str(aid)] = (
            g["tweet_id"].astype(str).to_numpy(),
            g["created_at"].to_numpy(dtype="datetime64[ns]"),
        )

    print(
        f"  reply→tweet events: {len(reply_to_tweet_df):,} | "
        f"reply→user: {len(reply_to_user_df):,} | "
        f"quote→user: {len(quote_to_user_df):,}"
    )
    mention_to_user = load_mention_times(accounts["account_id"])
    return {
        "reply_to_tweet": reply_to_tweet,
        "reply_to_user": reply_to_user,
        "quote_to_user": quote_to_user,
        "mention_to_user": mention_to_user,
        "author_timeline": author_timeline,
    }


def build_features(
    df: pd.DataFrame,
    min_history: int,
    graph: dict | None = None,
) -> pd.DataFrame:
    """Per-tweet features using only information available strictly before each tweet.

    Dump-time favorite_count / retweet_count are labels only — never history inputs.
    Author "success" history uses pre-cutoff reply/quote arrivals when available.
    """
    df = df.copy()
    graph = graph or {}
    reply_times = graph.get("reply_to_tweet", {})
    reply_to_user = graph.get("reply_to_user", {})
    quote_to_user = graph.get("quote_to_user", {})
    mention_to_user = graph.get("mention_to_user", {})
    author_timeline: dict[str, tuple[np.ndarray, np.ndarray]] = graph.get("author_timeline", {})
    df["favorite_count"] = pd.to_numeric(df["favorite_count"], errors="coerce").fillna(0).astype(np.int64)
    df["retweet_count"] = pd.to_numeric(df["retweet_count"], errors="coerce").fillna(0).astype(np.int64)
    # Label only (eventual engagement). Not used as a feature input.
    df["engagement"] = df["favorite_count"] + df["retweet_count"]
    df["log_engagement"] = np.log1p(df["engagement"].astype(np.float64))

    stats = df["full_text"].map(_text_stats)
    for key in next(iter(stats)).keys() if len(stats) else []:
        df[key] = stats.map(lambda d, k=key: d[k])
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
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    rows: list[dict] = []
    for account_id, g in df.groupby("account_id", sort=False):
        g = g.sort_values("created_at", kind="mergesort")
        times = g["created_at"].to_numpy()
        tweet_ids = g["tweet_id"].astype(str).to_numpy()
        is_reply = g["is_reply"].to_numpy(dtype=np.int8)
        has_url = g["has_url"].to_numpy(dtype=np.int8)
        has_media = g["has_media"].to_numpy(dtype=np.int8)
        aid = str(account_id)
        dump_ids, dump_times = author_timeline.get(
            aid, (np.array([], dtype=object), np.array([], dtype="datetime64[ns]"))
        )
        # Active dump parents: (reply_times, reply_ptr, created_at) walked forward with cutoffs.
        active_reply_arrs: list[np.ndarray] = []
        active_reply_ptrs: list[int] = []
        active_created: list[np.datetime64] = []
        dump_ptr = 0
        user_reply_times = reply_to_user.get(aid)
        user_quote_times = quote_to_user.get(aid)
        user_mention_times = mention_to_user.get(aid)

        for i in range(len(g)):
            if i < min_history:
                continue
            cutoff = times[i]
            cut64 = np.datetime64(pd.Timestamp(cutoff).to_datetime64())
            prior_reply = is_reply[:i]
            age_days = (cutoff - times[:i]).astype("timedelta64[s]").astype(np.float64) / 86400.0
            mask_7 = age_days <= 7.0
            mask_30 = age_days <= 30.0
            non_reply = prior_reply == 0
            reply_only = prior_reply == 1

            # Sample-path reply history (prior rows kept after per-account cap).
            recv = np.array(
                [_count_before(reply_times.get(str(tid)), cutoff) for tid in tweet_ids[:i]],
                dtype=np.float64,
            )
            log_recv = np.log1p(recv)
            weights = np.exp(-np.log(2.0) * age_days / 14.0)
            mean_recv = float(log_recv.mean())
            mean_recv_nr = float(log_recv[non_reply].mean()) if non_reply.any() else mean_recv
            mean_recv_r = float(log_recv[reply_only].mean()) if reply_only.any() else mean_recv

            while dump_ptr < len(dump_times) and dump_times[dump_ptr] < cut64:
                tid = str(dump_ids[dump_ptr])
                arr = reply_times.get(tid, np.array([], dtype="datetime64[ns]"))
                active_reply_arrs.append(arr)
                active_reply_ptrs.append(0)
                active_created.append(dump_times[dump_ptr])
                dump_ptr += 1

            n_prior_dump = len(active_reply_arrs)
            if n_prior_dump:
                dump_recv = np.empty(n_prior_dump, dtype=np.float64)
                for j, arr in enumerate(active_reply_arrs):
                    ptr = active_reply_ptrs[j]
                    while ptr < len(arr) and arr[ptr] < cut64:
                        ptr += 1
                    active_reply_ptrs[j] = ptr
                    dump_recv[j] = float(ptr)
                dump_log = np.log1p(dump_recv)
                dump_age = (cut64 - np.asarray(active_created, dtype="datetime64[ns]")).astype(
                    "timedelta64[s]"
                ).astype(np.float64) / 86400.0
                dump_w = np.exp(-np.log(2.0) * dump_age / 14.0)
                hist_dump_mean = float(dump_log.mean())
                hist_dump_decay = float(np.average(dump_log, weights=dump_w))
                hist_dump_p90 = float(np.quantile(dump_log, 0.9))
            else:
                hist_dump_mean = 0.0
                hist_dump_decay = 0.0
                hist_dump_p90 = 0.0

            inbound_replies = _count_before(user_reply_times, cutoff)
            inbound_quotes = _count_before(user_quote_times, cutoff)
            inbound_mentions = _count_before(user_mention_times, cutoff)
            inbound_replies_7d = _count_window(user_reply_times, cutoff, 7.0)
            inbound_replies_30d = _count_window(user_reply_times, cutoff, 30.0)
            inbound_quotes_7d = _count_window(user_quote_times, cutoff, 7.0)
            inbound_quotes_30d = _count_window(user_quote_times, cutoff, 30.0)
            inbound_mentions_7d = _count_window(user_mention_times, cutoff, 7.0)
            inbound_mentions_30d = _count_window(user_mention_times, cutoff, 30.0)

            row = g.iloc[i]
            log_in_r = float(np.log1p(inbound_replies))
            log_in_q = float(np.log1p(inbound_quotes))
            log_in_m = float(np.log1p(inbound_mentions))
            rows.append(
                {
                    "tweet_id": row["tweet_id"],
                    "account_id": account_id,
                    "username": row["username"],
                    "created_at": row["created_at"],
                    "full_text": row["full_text"],
                    "reply_to_tweet_id": (
                        None if pd.isna(row["reply_to_tweet_id"]) else str(row["reply_to_tweet_id"])
                    ),
                    "quoted_tweet_id": (
                        None if pd.isna(row["quoted_tweet_id"]) else str(row["quoted_tweet_id"])
                    ),
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
                    "hist_tweets_7d": int(mask_7.sum()),
                    "hist_tweets_30d": int(mask_30.sum()),
                    "days_since_last": float(age_days[-1]),
                    "hist_frac_reply": float(prior_reply.mean()),
                    "hist_frac_url": float(has_url[:i].mean()),
                    "hist_frac_media": float(has_media[:i].mean()),
                    # Leakage-safe popularity: pre-T reply arrivals on prior tweets.
                    "hist_mean_replies": mean_recv,
                    "hist_median_replies": float(np.median(log_recv)),
                    "hist_p90_replies": float(np.quantile(log_recv, 0.9)),
                    "hist_last_replies": float(log_recv[-1]),
                    "hist_std_replies": float(log_recv.std(ddof=0)) if i > 1 else 0.0,
                    "hist_mean_replies_7d": float(log_recv[mask_7].mean()) if mask_7.any() else 0.0,
                    "hist_mean_replies_30d": float(log_recv[mask_30].mean()) if mask_30.any() else 0.0,
                    "hist_decay_mean_replies": float(np.average(log_recv, weights=weights)),
                    "hist_mean_replies_nonreply": mean_recv_nr,
                    "hist_mean_replies_reply": mean_recv_r,
                    "reply_x_hist_replies_nr": float(row["is_reply"]) * mean_recv_nr,
                    "media_x_hist_replies": float(row["has_media"]) * mean_recv,
                    # Dump-wide parent coverage (all author tweets before T).
                    "hist_dump_n": int(n_prior_dump),
                    "hist_dump_mean_replies": hist_dump_mean,
                    "hist_dump_decay_replies": hist_dump_decay,
                    "hist_dump_p90_replies": hist_dump_p90,
                    # Account-level inbound graph (full dump, created_at < T).
                    "inbound_replies": log_in_r,
                    "inbound_replies_7d": float(np.log1p(inbound_replies_7d)),
                    "inbound_replies_30d": float(np.log1p(inbound_replies_30d)),
                    "inbound_replies_per_tweet": float(np.log1p(inbound_replies / max(n_prior_dump, 1))),
                    "inbound_quotes": log_in_q,
                    "inbound_quotes_7d": float(np.log1p(inbound_quotes_7d)),
                    "inbound_quotes_30d": float(np.log1p(inbound_quotes_30d)),
                    "inbound_quotes_per_tweet": float(np.log1p(inbound_quotes / max(n_prior_dump, 1))),
                    "inbound_mentions": log_in_m,
                    "inbound_mentions_7d": float(np.log1p(inbound_mentions_7d)),
                    "inbound_mentions_30d": float(np.log1p(inbound_mentions_30d)),
                    "inbound_mentions_per_tweet": float(
                        np.log1p(inbound_mentions / max(n_prior_dump, 1))
                    ),
                    "media_x_inbound_replies": float(row["has_media"]) * log_in_r,
                    "reply_x_inbound_replies": float(row["is_reply"]) * log_in_r,
                    "media_x_inbound_mentions": float(row["has_media"]) * log_in_m,
                    "reply_x_inbound_mentions": float(row["is_reply"]) * log_in_m,
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


def ensure_embed_cache(tweet_ids: pd.Series, texts: pd.Series, batch_size: int) -> pd.DataFrame:
    """Return embed cache covering the given tweet_ids (embeds misses from texts)."""
    cache = load_embed_cache()
    ids = tweet_ids.astype(str).reset_index(drop=True)
    texts = texts.fillna("").astype(str).reset_index(drop=True)
    if cache is not None:
        cache = cache.copy()
        cache["tweet_id"] = cache["tweet_id"].astype(str)
        known = set(cache["tweet_id"])
        miss = ~ids.isin(known)
    else:
        known = set()
        miss = pd.Series(True, index=ids.index)

    miss_idx = np.flatnonzero(miss.to_numpy())
    print(f"Embed cache fill: {len(ids) - len(miss_idx):,}/{len(ids):,} hit, {len(miss_idx):,} new")
    if len(miss_idx):
        new_emb = embed_texts(texts.iloc[miss_idx].tolist(), batch_size=batch_size)
        new_df = pd.DataFrame(new_emb, columns=[f"emb_{i}" for i in range(new_emb.shape[1])])
        new_df.insert(0, "tweet_id", ids.iloc[miss_idx].to_numpy())
        cache = new_df if cache is None else pd.concat([cache, new_df], ignore_index=True)
        cache = cache.drop_duplicates("tweet_id", keep="last")
        save_embed_cache(cache)
    assert cache is not None
    return cache


def attach_embeddings(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    cache = ensure_embed_cache(df["tweet_id"], df["full_text"], batch_size=batch_size)
    emb_cols = [c for c in cache.columns if c.startswith("emb_")]
    ids = df["tweet_id"].astype(str)
    merged = pd.DataFrame({"tweet_id": ids}).merge(cache[["tweet_id", *emb_cols]], on="tweet_id", how="left")
    if merged[emb_cols[0]].isna().any():
        raise RuntimeError("Missing embeddings after cache merge")
    return pd.concat([df.reset_index(drop=True), merged[emb_cols].reset_index(drop=True)], axis=1)


def load_tweets_by_id(tweet_ids: pd.Series) -> pd.DataFrame:
    """Fetch parent tweet rows from the dump (text + structure only; no fav/RT inputs)."""
    ids = pd.DataFrame({"tweet_id": tweet_ids.astype(str).dropna().unique()})
    if ids.empty:
        return pd.DataFrame(
            columns=[
                "tweet_id",
                "account_id",
                "full_text",
                "created_at",
                "reply_to_tweet_id",
                "quoted_tweet_id",
            ]
        )
    tmp = CACHE_DIR / "_parent_tweet_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids.to_parquet(tmp, index=False)
    path = str(RAW_PARQUET)
    print(f"Loading {len(ids):,} referenced tweets from dump…")
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    df = con.execute(
        f"""
        SELECT
          CAST(t.tweet_id AS VARCHAR) AS tweet_id,
          CAST(t.account_id AS VARCHAR) AS account_id,
          t.full_text,
          t.created_at,
          CAST(t.reply_to_tweet_id AS VARCHAR) AS reply_to_tweet_id,
          CAST(t.quoted_tweet_id AS VARCHAR) AS quoted_tweet_id
        FROM read_parquet('{path}') t
        SEMI JOIN read_parquet('{tmp}') p
          ON CAST(t.tweet_id AS VARCHAR) = p.tweet_id
        """
    ).fetchdf()
    con.close()
    try:
        tmp.unlink()
    except OSError:
        pass
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    print(f"  found {len(df):,}/{len(ids):,} parents in dump")
    return df


def load_reply_arrival_times(tweet_ids: pd.Series) -> dict[str, np.ndarray]:
    """Map parent tweet_id → sorted reply created_at (UTC ns) from the full dump."""
    ids = pd.DataFrame({"tweet_id": tweet_ids.astype(str).dropna().unique()})
    if ids.empty:
        return {}
    tmp_ids = CACHE_DIR / "_reply_parent_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids.to_parquet(tmp_ids, index=False)
    path = str(RAW_PARQUET)
    print(f"Scanning dump for replies to {len(ids):,} referenced tweets…")
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    reply_df = con.execute(
        f"""
        SELECT
          CAST(r.reply_to_tweet_id AS VARCHAR) AS parent_id,
          r.created_at
        FROM read_parquet('{path}') r
        SEMI JOIN read_parquet('{tmp_ids}') t
          ON CAST(r.reply_to_tweet_id AS VARCHAR) = t.tweet_id
        WHERE r.reply_to_tweet_id IS NOT NULL
        """
    ).fetchdf()
    con.close()
    try:
        tmp_ids.unlink()
    except OSError:
        pass
    out = _sorted_time_map(reply_df, "parent_id")
    print(f"  reply events on parents: {len(reply_df):,} across {len(out):,} parents")
    return out


def _count_replies_before_cutoffs(
    parent_ids: np.ndarray,
    cutoffs: np.ndarray,
    reply_times: dict[str, np.ndarray],
) -> np.ndarray:
    """Vectorized-ish: group by parent, walk sorted cutoffs with a reply pointer."""
    n = len(parent_ids)
    out = np.zeros(n, dtype=np.float64)
    order = np.argsort(parent_ids, kind="mergesort")
    sorted_pids = parent_ids[order]
    sorted_cuts = cutoffs[order]
    i = 0
    while i < n:
        pid = sorted_pids[i]
        j = i + 1
        while j < n and sorted_pids[j] == pid:
            j += 1
        times = reply_times.get(str(pid))
        if times is None or len(times) == 0:
            i = j
            continue
        # Stable by cutoff within this parent group.
        sub = np.arange(i, j)
        sub_order = sub[np.argsort(sorted_cuts[i:j], kind="mergesort")]
        ptr = 0
        for idx in sub_order:
            cut = np.datetime64(pd.Timestamp(sorted_cuts[idx]).to_datetime64())
            while ptr < len(times) and times[ptr] < cut:
                ptr += 1
            out[order[idx]] = float(ptr)
        i = j
    return out


def _attach_one_parent_prefix(
    out: pd.DataFrame,
    id_col: str,
    prefix: str,
    parents: pd.DataFrame,
    parent_emb: pd.DataFrame,
    reply_times: dict[str, np.ndarray],
    child_emb: np.ndarray,
) -> pd.DataFrame:
    """Merge parent rows for id_col and write has_* / cosine / pre-T reply features."""
    n = len(out)
    has = np.zeros(n, dtype=np.int8)
    cos = np.zeros(n, dtype=np.float64)
    log_len = np.zeros(n, dtype=np.float64)
    is_reply = np.zeros(n, dtype=np.int8)
    is_quote = np.zeros(n, dtype=np.int8)
    same_author = np.zeros(n, dtype=np.int8)
    age_h = np.zeros(n, dtype=np.float64)
    replies = np.zeros(n, dtype=np.float64)

    raw_ids = out[id_col]
    valid = raw_ids.notna()
    if not valid.any():
        out[f"has_{prefix}_parent"] = has
        out[f"{prefix}_parent_emb_cos"] = cos
        out[f"{prefix}_parent_log_text_len"] = log_len
        out[f"{prefix}_parent_is_reply"] = is_reply
        out[f"{prefix}_parent_is_quote"] = is_quote
        out[f"{prefix}_parent_same_author"] = same_author
        out[f"{prefix}_parent_age_hours"] = age_h
        out[f"{prefix}_parent_replies"] = replies
        return out

    keys = raw_ids.map(lambda x: str(x) if pd.notna(x) else None)
    key_arr = keys.to_numpy(dtype=object)
    meta = parents.reindex(key_arr)
    found = meta["created_at"].notna().to_numpy()
    child_t = pd.to_datetime(out["created_at"], utc=True)
    parent_t = pd.to_datetime(meta["created_at"], utc=True)
    # Parent must strictly predate child.
    ok = found & (parent_t.to_numpy() < child_t.to_numpy())
    has[ok] = 1

    text_arr = meta["full_text"].fillna("").astype(str).to_numpy()
    if ok.any():
        log_len[ok] = np.log1p(np.fromiter((len(t) for t in text_arr[ok]), dtype=np.float64, count=int(ok.sum())))
        is_reply[ok] = meta["reply_to_tweet_id"].notna().to_numpy(dtype=np.int8)[ok]
        is_quote[ok] = meta["quoted_tweet_id"].notna().to_numpy(dtype=np.int8)[ok]
        same_author[ok] = (
            meta["account_id"].astype(str).to_numpy()[ok] == out["account_id"].astype(str).to_numpy()[ok]
        ).astype(np.int8)
        delta_h = (
            (child_t.to_numpy()[ok] - parent_t.to_numpy()[ok]).astype("timedelta64[s]").astype(np.float64)
            / 3600.0
        )
        age_h[ok] = np.log1p(np.clip(delta_h, 0, None))

    # Cosine with parent MiniLM (rows already L2-normalized).
    pemb = parent_emb.reindex(key_arr)
    pemb_mat = pemb.to_numpy(dtype=np.float32)
    row_ok = ok & np.isfinite(pemb_mat[:, 0])
    if row_ok.any():
        cos[row_ok] = np.sum(child_emb[row_ok] * pemb_mat[row_ok], axis=1)

    if ok.any():
        pids = key_arr[ok].astype(str)
        cuts = child_t.to_numpy()[ok]
        replies[ok] = np.log1p(_count_replies_before_cutoffs(pids, cuts, reply_times))

    out[f"has_{prefix}_parent"] = has
    out[f"{prefix}_parent_emb_cos"] = cos
    out[f"{prefix}_parent_log_text_len"] = log_len
    out[f"{prefix}_parent_is_reply"] = is_reply
    out[f"{prefix}_parent_is_quote"] = is_quote
    out[f"{prefix}_parent_same_author"] = same_author
    out[f"{prefix}_parent_age_hours"] = age_h
    out[f"{prefix}_parent_replies"] = replies
    return out


def attach_parent_context(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Join reply/quote parents from the dump; add leakage-safe context features."""
    out = df.reset_index(drop=True).copy()
    if "reply_to_tweet_id" not in out.columns or "quoted_tweet_id" not in out.columns:
        raise RuntimeError("Dataset missing reply_to_tweet_id / quoted_tweet_id for parent join")

    reply_ids = out["reply_to_tweet_id"].dropna().astype(str)
    quote_ids = out["quoted_tweet_id"].dropna().astype(str)
    all_ids = pd.Series(pd.unique(pd.concat([reply_ids, quote_ids], ignore_index=True)))
    print(
        f"Parent context: {reply_ids.nunique():,} reply parents, "
        f"{quote_ids.nunique():,} quote parents, {len(all_ids):,} unique",
        flush=True,
    )
    if all_ids.empty:
        for prefix in ("reply", "quote"):
            out[f"has_{prefix}_parent"] = 0
            out[f"{prefix}_parent_emb_cos"] = 0.0
            out[f"{prefix}_parent_log_text_len"] = 0.0
            out[f"{prefix}_parent_is_reply"] = 0
            out[f"{prefix}_parent_is_quote"] = 0
            out[f"{prefix}_parent_same_author"] = 0
            out[f"{prefix}_parent_age_hours"] = 0.0
            out[f"{prefix}_parent_replies"] = 0.0
        return out

    parents = load_tweets_by_id(all_ids)
    if parents.empty:
        for prefix in ("reply", "quote"):
            out[f"has_{prefix}_parent"] = 0
            out[f"{prefix}_parent_emb_cos"] = 0.0
            out[f"{prefix}_parent_log_text_len"] = 0.0
            out[f"{prefix}_parent_is_reply"] = 0
            out[f"{prefix}_parent_is_quote"] = 0
            out[f"{prefix}_parent_same_author"] = 0
            out[f"{prefix}_parent_age_hours"] = 0.0
            out[f"{prefix}_parent_replies"] = 0.0
        return out

    parents = parents.drop_duplicates("tweet_id", keep="first").set_index("tweet_id")
    reply_times = load_reply_arrival_times(pd.Series(parents.index.astype(str)))

    p_ids = pd.Series(parents.index.astype(str))
    p_texts = parents["full_text"].fillna("").astype(str)
    cache = ensure_embed_cache(p_ids, p_texts, batch_size=batch_size)
    emb_cols = [c for c in cache.columns if c.startswith("emb_")]
    parent_emb = cache.set_index("tweet_id")[emb_cols]

    child_emb_cols = [c for c in out.columns if c.startswith("emb_")]
    child_emb = out[child_emb_cols].to_numpy(dtype=np.float32)

    out = _attach_one_parent_prefix(
        out, "reply_to_tweet_id", "reply", parents, parent_emb, reply_times, child_emb
    )
    out = _attach_one_parent_prefix(
        out, "quoted_tweet_id", "quote", parents, parent_emb, reply_times, child_emb
    )
    print(
        f"  reply parent coverage: {out['has_reply_parent'].mean():.1%}, "
        f"quote parent coverage: {out['has_quote_parent'].mean():.1%}",
        flush=True,
    )
    return out


def backfill_ref_ids_from_dump(df: pd.DataFrame) -> pd.DataFrame:
    """Attach reply_to_tweet_id / quoted_tweet_id from the dump when missing."""
    if "reply_to_tweet_id" in df.columns and "quoted_tweet_id" in df.columns:
        return df
    ids = pd.DataFrame({"tweet_id": df["tweet_id"].astype(str).unique()})
    tmp = CACHE_DIR / "_child_tweet_ids.parquet"
    ids.to_parquet(tmp, index=False)
    path = str(RAW_PARQUET)
    print(f"Backfilling ref ids for {len(ids):,} tweets from dump…", flush=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    refs = con.execute(
        f"""
        SELECT
          CAST(t.tweet_id AS VARCHAR) AS tweet_id,
          CAST(t.reply_to_tweet_id AS VARCHAR) AS reply_to_tweet_id,
          CAST(t.quoted_tweet_id AS VARCHAR) AS quoted_tweet_id
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
    refs = refs.drop_duplicates("tweet_id", keep="first")
    out = df.drop(columns=[c for c in ("reply_to_tweet_id", "quoted_tweet_id") if c in df.columns], errors="ignore")
    out["tweet_id"] = out["tweet_id"].astype(str)
    out = out.drop_duplicates("tweet_id", keep="first")
    out = out.merge(refs, on="tweet_id", how="left")
    print(
        f"  reply rate: {out['reply_to_tweet_id'].notna().mean():.1%}, "
        f"quote rate: {out['quoted_tweet_id'].notna().mean():.1%}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def attach_inbound_mentions(df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """Add pre-T inbound @-mention features onto an existing feature table."""
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    out["account_id"] = out["account_id"].astype(str)
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    mention_to_user = load_mention_times(out["account_id"], force=force)

    drop_cols = [
        c
        for c in (
            "inbound_mentions",
            "inbound_mentions_7d",
            "inbound_mentions_30d",
            "inbound_mentions_per_tweet",
            "media_x_inbound_mentions",
            "reply_x_inbound_mentions",
        )
        if c in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    n = len(out)
    inbound = np.zeros(n, dtype=np.float64)
    inbound_7d = np.zeros(n, dtype=np.float64)
    inbound_30d = np.zeros(n, dtype=np.float64)
    inbound_per = np.zeros(n, dtype=np.float64)
    has_media = (
        out["has_media"].to_numpy(dtype=np.float64)
        if "has_media" in out.columns
        else np.zeros(n, dtype=np.float64)
    )
    is_reply = (
        out["is_reply"].to_numpy(dtype=np.float64)
        if "is_reply" in out.columns
        else np.zeros(n, dtype=np.float64)
    )
    denom = (
        out["hist_dump_n"].to_numpy(dtype=np.float64)
        if "hist_dump_n" in out.columns
        else (
            out["hist_n"].to_numpy(dtype=np.float64)
            if "hist_n" in out.columns
            else np.ones(n, dtype=np.float64)
        )
    )

    for aid, g in out.groupby("account_id", sort=False):
        ix = g.index.to_numpy()
        times = mention_to_user.get(str(aid))
        cutoffs = g["created_at"].to_numpy()
        for j, cutoff in enumerate(cutoffs):
            row_i = ix[j]
            c = _count_before(times, cutoff)
            inbound[row_i] = float(np.log1p(c))
            inbound_7d[row_i] = float(np.log1p(_count_window(times, cutoff, 7.0)))
            inbound_30d[row_i] = float(np.log1p(_count_window(times, cutoff, 30.0)))
            inbound_per[row_i] = float(np.log1p(c / max(float(denom[row_i]), 1.0)))

    out["inbound_mentions"] = inbound
    out["inbound_mentions_7d"] = inbound_7d
    out["inbound_mentions_30d"] = inbound_30d
    out["inbound_mentions_per_tweet"] = inbound_per
    out["media_x_inbound_mentions"] = has_media * inbound
    out["reply_x_inbound_mentions"] = is_reply * inbound
    nonzero = float((inbound > 0).mean())
    print(
        f"Inbound mentions: {(inbound > 0).mean():.1%} rows with pre-T mentions; "
        f"mean log1p={inbound.mean():.3f}; account coverage proxy={nonzero:.1%}",
        flush=True,
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-tweets", type=int, default=120000, help="Cap on sampled original tweets")
    p.add_argument("--per-account-cap", type=int, default=180, help="Max tweets kept per account")
    p.add_argument("--min-account-tweets", type=int, default=50)
    p.add_argument("--max-account-tweets", type=int, default=8000)
    p.add_argument("--min-history", type=int, default=5, help="Require this many prior tweets")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-media", action="store_true", help="Skip CA media/url API joins")
    p.add_argument(
        "--parent-only",
        action="store_true",
        help="Only add reply/quote parent features onto an existing dataset.parquet",
    )
    p.add_argument(
        "--urls-only",
        action="store_true",
        help="Only add link/domain features onto an existing dataset.parquet",
    )
    p.add_argument(
        "--media-only",
        action="store_true",
        help="Refetch CA tweet_media/tweet_urls flags onto an existing dataset.parquet",
    )
    p.add_argument(
        "--mentions-only",
        action="store_true",
        help="Add timestamped inbound @-mention features onto an existing dataset.parquet",
    )
    p.add_argument(
        "--force-urls",
        action="store_true",
        help="Refetch tweet_urls even when url_cache.parquet has the tweet_id",
    )
    p.add_argument(
        "--force-mentions",
        action="store_true",
        help="Refetch user_mentions even when mention_cache.parquet has the account_id",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    # Line-buffer logs when piped (e.g. tee).
    try:
        import sys

        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    download_parquet(force=args.force_download)

    if args.mentions_only:
        if not DATASET_PATH.exists():
            raise SystemExit(f"Missing {DATASET_PATH} for --mentions-only")
        dataset = pd.read_parquet(DATASET_PATH)
        dataset = attach_inbound_mentions(dataset, force=args.force_mentions)
        dataset.to_parquet(DATASET_PATH, index=False)
        print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
        print(f"Done in {time.time() - t0:.1f}s")
        return

    if args.media_only:
        if not DATASET_PATH.exists():
            raise SystemExit(f"Missing {DATASET_PATH} for --media-only")
        dataset = pd.read_parquet(DATASET_PATH)
        dataset["tweet_id"] = dataset["tweet_id"].astype(str)
        before = {
            "has_media": float(dataset["has_media"].mean()) if "has_media" in dataset.columns else 0.0,
            "has_ca_url": float(dataset["has_ca_url"].mean()) if "has_ca_url" in dataset.columns else 0.0,
        }
        dataset = attach_media_flags(dataset)
        dataset = refresh_media_derived(dataset)
        dataset.to_parquet(DATASET_PATH, index=False)
        print(
            f"Media refresh: has_media {before['has_media']:.1%} → {dataset['has_media'].mean():.1%}, "
            f"has_ca_url {before['has_ca_url']:.1%} → {dataset['has_ca_url'].mean():.1%}"
        )
        print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
        print(f"Done in {time.time() - t0:.1f}s")
        return

    if args.urls_only:
        if not DATASET_PATH.exists():
            raise SystemExit(f"Missing {DATASET_PATH} for --urls-only")
        dataset = pd.read_parquet(DATASET_PATH)
        dataset["tweet_id"] = dataset["tweet_id"].astype(str)
        drop_cols = [
            c
            for c in (
                "n_text_urls",
                "url_only",
                "url_at_end",
                "url_domain",
                "has_expanded_domain",
                "n_ca_url_rows",
                "domain_is_twitter",
                "domain_is_youtube",
                "domain_is_github",
                "domain_is_substack",
                "domain_is_wiki",
            )
            if c in dataset.columns
        ]
        if drop_cols:
            dataset = dataset.drop(columns=drop_cols)
        dataset = attach_url_domains(dataset, force=args.force_urls)
        dataset.to_parquet(DATASET_PATH, index=False)
        print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
        print(f"Done in {time.time() - t0:.1f}s")
        return

    if args.parent_only:
        if not DATASET_PATH.exists():
            raise SystemExit(f"Missing {DATASET_PATH} for --parent-only")
        dataset = pd.read_parquet(DATASET_PATH)
        dataset["tweet_id"] = dataset["tweet_id"].astype(str)
        dataset["created_at"] = pd.to_datetime(dataset["created_at"], utc=True)
        before = len(dataset)
        dataset = dataset.drop_duplicates("tweet_id", keep="first")
        if len(dataset) != before:
            print(f"Dropped {before - len(dataset):,} duplicate tweet_id rows", flush=True)
        drop_cols = [
            c
            for c in dataset.columns
            if c.startswith(("has_reply_parent", "has_quote_parent", "reply_parent_", "quote_parent_"))
        ]
        if drop_cols:
            dataset = dataset.drop(columns=drop_cols)
        dataset = backfill_ref_ids_from_dump(dataset)
        if not any(c.startswith("emb_") for c in dataset.columns):
            raise SystemExit("Dataset has no emb_* columns; run full prepare first")
        dataset = attach_parent_context(dataset, batch_size=args.embed_batch_size)
        dataset.to_parquet(DATASET_PATH, index=False)
        print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
        print(f"Done in {time.time() - t0:.1f}s")
        return

    sampled = sample_accounts(
        max_tweets=args.max_tweets,
        min_account_tweets=args.min_account_tweets,
        max_account_tweets=args.max_account_tweets,
        per_account_cap=args.per_account_cap,
        seed=args.seed,
    )
    if args.skip_media:
        if DATASET_PATH.exists():
            prev = pd.read_parquet(
                DATASET_PATH,
                columns=["tweet_id", "has_media", "n_media", "has_ca_url", "n_ca_urls"],
            )
            prev["tweet_id"] = prev["tweet_id"].astype(str)
            sampled = sampled.merge(prev, on="tweet_id", how="left")
            for c, fill in (("has_media", 0), ("n_media", 0), ("has_ca_url", 0), ("n_ca_urls", 0)):
                sampled[c] = sampled[c].fillna(fill).astype(np.int32)
            print(
                f"Reused media flags from {DATASET_PATH.name}: "
                f"{sampled['has_media'].mean():.1%} media, {sampled['has_ca_url'].mean():.1%} ca_url"
            )
        else:
            print("No prior dataset for media reuse; media flags default to 0")
    else:
        sampled = attach_media_flags(sampled)
    graph = load_author_graph(sampled["account_id"])
    featured = build_features(sampled, min_history=args.min_history, graph=graph)
    del sampled, graph

    dataset = attach_embeddings(featured, batch_size=args.embed_batch_size)
    del featured
    dataset = attach_parent_context(dataset, batch_size=args.embed_batch_size)
    dataset = attach_url_domains(dataset, force=args.force_urls)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)
    print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
