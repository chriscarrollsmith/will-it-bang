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
from collections.abc import Callable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from requests_oauthlib import OAuth1Session
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~/.cache/will-it-bang"))
RAW_PARQUET = CACHE_DIR / "enriched_tweets.parquet"
DATASET_PATH = CACHE_DIR / "dataset.parquet"
EMBED_CACHE_PATH = CACHE_DIR / "embed_cache.parquet"
MENTION_CACHE_PATH = CACHE_DIR / "mention_cache.parquet"
INTERLOCUTOR_CACHE_PATH = CACHE_DIR / "interlocutor_cache.parquet"
UPLOAD_CACHE_PATH = CACHE_DIR / "archive_upload_cache.parquet"
MEDIA_CACHE_PATH = CACHE_DIR / "media_cache.parquet"
X_PARENT_CACHE_PATH = CACHE_DIR / "x_parent_cache.parquet"
DYAD_CACHE_PATH = CACHE_DIR / "dyad_cache.parquet"
# CA tweet_media.media_type values observed in the public API.
MEDIA_TYPE_KEYS = ("photo", "video", "animated_gif")
# Terminal X lookup statuses — do not re-request on resume.
X_PARENT_DONE_STATUSES = frozenset({"ok", "not_found", "forbidden"})
X_TWEETS_URL = "https://api.x.com/2/tweets"
X_TWEETS_BATCH = 100
# CA / paste junk sometimes leaves "?s=20" (or status URLs) in ref ids.
TWEET_ID_RE = re.compile(r"(\d{5,25})")
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
        # Provenance: null ⇒ browser-extension stream (see audit_ca_quality.py).
        "archive_upload_id": "archive_upload_id",
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


def _fetch_tweet_media(tweet_ids: list[str], batch_size: int = 80) -> pd.DataFrame:
    """Fetch tweet_id / media_type rows from CA tweet_media."""
    rows: list[dict[str, str]] = []
    session = requests.Session()
    for start in range(0, len(tweet_ids), batch_size):
        batch = tweet_ids[start : start + batch_size]
        joined = ",".join(f'"{tid}"' for tid in batch)
        resp = session.get(
            f"{CA_API}/tweet_media",
            headers=_ca_headers(),
            params={"select": "tweet_id,media_type", "tweet_id": f"in.({joined})"},
            timeout=60,
        )
        resp.raise_for_status()
        for row in resp.json():
            rows.append(
                {
                    "tweet_id": str(row["tweet_id"]),
                    "media_type": str(row.get("media_type") or ""),
                }
            )
        if start == 0 or (start // batch_size) % 20 == 0:
            print(f"  tweet_media: {min(start + batch_size, len(tweet_ids)):,}/{len(tweet_ids):,}")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["tweet_id", "media_type"])


def _aggregate_media_types(raw: pd.DataFrame, tweet_ids: list[str]) -> pd.DataFrame:
    """Per-tweet media counts by type (zeros for ids with no media rows)."""
    base = pd.DataFrame({"tweet_id": [str(t) for t in tweet_ids]})
    if raw.empty:
        out = base.copy()
        out["n_media"] = np.int16(0)
        for key in MEDIA_TYPE_KEYS:
            out[f"n_{key}"] = np.int16(0)
        out["n_media_other"] = np.int16(0)
        return out
    raw = raw.copy()
    raw["tweet_id"] = raw["tweet_id"].astype(str)
    raw["media_type"] = raw["media_type"].fillna("").astype(str).str.lower()
    known = set(MEDIA_TYPE_KEYS)
    raw["type_key"] = raw["media_type"].map(lambda t: t if t in known else "other")
    pivot = (
        raw.groupby(["tweet_id", "type_key"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for key in (*MEDIA_TYPE_KEYS, "other"):
        if key not in pivot.columns:
            pivot[key] = 0
    pivot = pivot.rename(columns={k: f"n_{k}" for k in (*MEDIA_TYPE_KEYS, "other")})
    pivot["n_media_other"] = pivot["n_other"].astype(np.int16)
    pivot = pivot.drop(columns=["n_other"])
    pivot["n_media"] = (
        pivot[[f"n_{k}" for k in MEDIA_TYPE_KEYS]].sum(axis=1) + pivot["n_media_other"]
    ).astype(np.int16)
    for key in MEDIA_TYPE_KEYS:
        pivot[f"n_{key}"] = pivot[f"n_{key}"].astype(np.int16)
    out = base.merge(pivot, on="tweet_id", how="left")
    for c in ["n_media", "n_media_other", *[f"n_{k}" for k in MEDIA_TYPE_KEYS]]:
        out[c] = out[c].fillna(0).astype(np.int16)
    return out


def attach_media_flags(df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """Join sparse CA tweet_media (with type breakdown) / tweet_urls flags."""
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    ids = out["tweet_id"].tolist()
    uniq = list(dict.fromkeys(ids))
    print(f"Fetching attachment flags for {len(uniq):,} tweets…")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = pd.DataFrame()
    if MEDIA_CACHE_PATH.exists() and not force:
        cache = pd.read_parquet(MEDIA_CACHE_PATH)
        cache["tweet_id"] = cache["tweet_id"].astype(str)
        print(f"Loaded media cache: {len(cache):,} tweets")
    have = set(cache["tweet_id"]) if len(cache) else set()
    miss = [tid for tid in uniq if tid not in have]
    print(f"  media cache: {len(uniq) - len(miss):,} hit, {len(miss):,} to fetch")
    if miss:
        raw = _fetch_tweet_media(miss)
        fresh = _aggregate_media_types(raw, miss)
        cache = fresh if cache.empty else pd.concat([cache, fresh], ignore_index=True)
        cache = cache.drop_duplicates("tweet_id", keep="last")
        cache.to_parquet(MEDIA_CACHE_PATH, index=False)
        print(f"Wrote media cache: {MEDIA_CACHE_PATH} ({len(cache):,} tweets)", flush=True)

    if cache.empty:
        cache = _aggregate_media_types(pd.DataFrame(), uniq)
    merged = out[["tweet_id"]].merge(cache, on="tweet_id", how="left")
    out["n_media"] = merged["n_media"].fillna(0).astype(np.int16)
    out["has_media"] = (out["n_media"] > 0).astype(np.int8)
    for key in MEDIA_TYPE_KEYS:
        ncol = f"n_{key}"
        hcol = f"has_{key}"
        out[ncol] = merged[ncol].fillna(0).astype(np.int16)
        out[hcol] = (out[ncol] > 0).astype(np.int8)
    out["n_media_other"] = merged["n_media_other"].fillna(0).astype(np.int16)
    out["has_media_other"] = (out["n_media_other"] > 0).astype(np.int8)

    url_counts = _fetch_id_counts("tweet_urls", "tweet_id", uniq)
    out["n_ca_urls"] = out["tweet_id"].map(lambda t: url_counts.get(t, 0)).astype(np.int16)
    out["has_ca_url"] = (out["n_ca_urls"] > 0).astype(np.int8)
    print(
        f"  media coverage: {out['has_media'].mean():.1%} "
        f"(photo={out['has_photo'].mean():.1%}, video={out['has_video'].mean():.1%}, "
        f"gif={out['has_animated_gif'].mean():.1%}), "
        f"ca_url coverage: {out['has_ca_url'].mean():.1%}",
        flush=True,
    )
    return out


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
    if "inbound_mentions" in out.columns:
        out["media_x_inbound_mentions"] = out["has_media"].astype(np.float64) * out["inbound_mentions"]
    # Type × popularity crosses (sparse but leakage-safe).
    if "inbound_replies" in out.columns:
        for key in MEDIA_TYPE_KEYS:
            hcol = f"has_{key}"
            if hcol in out.columns:
                out[f"{key}_x_inbound_replies"] = (
                    out[hcol].astype(np.float64) * out["inbound_replies"].astype(np.float64)
                )
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


def _fetch_mention_events_api(
    account_ids: list[str],
    batch_size: int = 40,
    *,
    on_batch: Callable[[pd.DataFrame, list[str]], None] | None = None,
) -> pd.DataFrame:
    """Page CA user_mentions for mentioned_user_id ∈ accounts; join tweet created_at.

    If on_batch is set, it is called after each account batch with the cumulative
    events DataFrame so callers can flush a durable cache for resume.
    """
    if not account_ids:
        return pd.DataFrame(columns=["mentioned_user_id", "tweet_id", "created_at"])
    session = requests.Session()
    rows: list[dict[str, object]] = []
    for start in range(0, len(account_ids), batch_size):
        batch = account_ids[start : start + batch_size]
        joined = ",".join(f'"{aid}"' for aid in batch)
        last_id = 0
        page = 0
        batch_rows_before = len(rows)
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
        # Zero-mention accounts in this batch: sentinel so they are not refetched.
        seen_in_batch = {
            str(r["mentioned_user_id"]) for r in rows[batch_rows_before:]
        }
        for aid in batch:
            if aid not in seen_in_batch:
                rows.append(
                    {
                        "mentioned_user_id": aid,
                        "tweet_id": "",
                        "created_at": None,
                    }
                )
        if start == 0 or (start // batch_size) % 5 == 0 or start + batch_size >= len(account_ids):
            print(
                f"  user_mentions: {min(start + batch_size, len(account_ids)):,}/"
                f"{len(account_ids):,} accounts → {len(rows):,} events",
                flush=True,
            )
        if on_batch is not None:
            on_batch(pd.DataFrame(rows), batch)
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

        def _flush_mention_progress(fresh_raw: pd.DataFrame, batch: list[str]) -> None:
            """Persist each account batch so CA mention paging can resume after kills."""
            nonlocal cached
            if fresh_raw.empty:
                return
            # Only merge the just-finished batch accounts (raw times; dump backfill later).
            piece = fresh_raw.loc[
                fresh_raw["mentioned_user_id"].astype(str).isin(batch)
            ].copy()
            if piece.empty:
                return
            piece["mentioned_user_id"] = piece["mentioned_user_id"].astype(str)
            piece["tweet_id"] = piece["tweet_id"].astype(str)
            # API returns ISO strings; cache is datetime64 — unify before concat/parquet.
            piece["created_at"] = pd.to_datetime(
                piece["created_at"], utc=True, errors="coerce"
            )
            if len(cached):
                cached = cached.copy()
                cached["created_at"] = pd.to_datetime(
                    cached["created_at"], utc=True, errors="coerce"
                )
            pieces = [cached] if len(cached) else []
            pieces.append(piece)
            cached = pd.concat(pieces, ignore_index=True)
            cached = cached.drop_duplicates(["mentioned_user_id", "tweet_id"], keep="last")
            cached["created_at"] = pd.to_datetime(
                cached["created_at"], utc=True, errors="coerce"
            )
            cached.to_parquet(MENTION_CACHE_PATH, index=False)

        fresh = _fetch_mention_events_api(missing, on_batch=_flush_mention_progress)
        # Final merge + dump backfill for timestamps (idempotent with batch flushes).
        if len(cached):
            cached = cached.copy()
            cached["created_at"] = pd.to_datetime(
                cached["created_at"], utc=True, errors="coerce"
            )
        pieces = [cached] if len(cached) else []
        if not fresh.empty:
            fresh = fresh.copy()
            fresh["mentioned_user_id"] = fresh["mentioned_user_id"].astype(str)
            fresh["tweet_id"] = fresh["tweet_id"].astype(str)
            fresh["created_at"] = pd.to_datetime(
                fresh["created_at"], utc=True, errors="coerce"
            )
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
        cached["created_at"] = pd.to_datetime(
            cached["created_at"], utc=True, errors="coerce"
        )
        # Backfill created_at on any sentinel/partial rows still missing times.
        cached = _backfill_mention_times_from_dump(cached)
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


def _empty_interlocutor_cache() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ego_account_id",
            "partner_account_id",
            "tweet_id",
            "created_at",
            "kind",
            "full_text",
        ]
    )


def load_interlocutor_events(
    account_ids: pd.Series,
    *,
    force: bool = False,
    include_mentions: bool = False,
) -> pd.DataFrame:
    """Inbound reply/quote/(optional mention) tweets directed at sampled authors.

    Cached by ego account_id. Used for interlocutor MiniLM affinity: mean of
    pre-T interaction-tweet embeddings vs the author's current tweet.

    Mentions are off by default: they dominate event volume and largely
    duplicate inbound_mentions counts already in the tabular stack.
    """
    accounts = sorted({str(a) for a in account_ids.astype(str).tolist()})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _empty_interlocutor_cache()
    if INTERLOCUTOR_CACHE_PATH.exists() and not force:
        cached = pd.read_parquet(INTERLOCUTOR_CACHE_PATH)
        for col in ("ego_account_id", "partner_account_id", "tweet_id", "kind"):
            if col in cached.columns:
                cached[col] = cached[col].astype(str)
        if "full_text" not in cached.columns:
            cached["full_text"] = ""
        print(f"Interlocutor cache: {len(cached):,} rows ({INTERLOCUTOR_CACHE_PATH.name})")
    have = set(cached["ego_account_id"].unique()) if len(cached) else set()
    # Sentinel rows mark egos with zero inbound events so we do not rescan.
    missing = [a for a in accounts if a not in have]
    if missing:
        print(f"Scanning dump for interlocutor events on {len(missing):,} accounts…", flush=True)
        tmp = CACHE_DIR / "_interlocutor_account_ids.parquet"
        pd.DataFrame({"account_id": missing}).to_parquet(tmp, index=False)
        path = str(RAW_PARQUET)
        con = duckdb.connect()
        con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
        reply_df = con.execute(
            f"""
            WITH acc AS (SELECT account_id FROM read_parquet('{tmp}')),
            auth AS (
              SELECT CAST(t.tweet_id AS VARCHAR) AS tid,
                     CAST(t.account_id AS VARCHAR) AS aid
              FROM read_parquet('{path}') t
              SEMI JOIN acc ON CAST(t.account_id AS VARCHAR) = acc.account_id
              WHERE t.full_text IS NOT NULL AND t.full_text NOT LIKE 'RT @%'
            )
            SELECT
              p.aid AS ego_account_id,
              CAST(r.account_id AS VARCHAR) AS partner_account_id,
              CAST(r.tweet_id AS VARCHAR) AS tweet_id,
              r.created_at,
              'reply' AS kind,
              r.full_text
            FROM read_parquet('{path}') r
            JOIN auth p ON CAST(r.reply_to_tweet_id AS VARCHAR) = p.tid
            WHERE r.reply_to_tweet_id IS NOT NULL
              AND CAST(r.account_id AS VARCHAR) != p.aid
              AND r.full_text IS NOT NULL
              AND r.full_text NOT LIKE 'RT @%'
            """
        ).fetchdf()
        quote_df = con.execute(
            f"""
            WITH acc AS (SELECT account_id FROM read_parquet('{tmp}')),
            auth AS (
              SELECT CAST(t.tweet_id AS VARCHAR) AS tid,
                     CAST(t.account_id AS VARCHAR) AS aid
              FROM read_parquet('{path}') t
              SEMI JOIN acc ON CAST(t.account_id AS VARCHAR) = acc.account_id
              WHERE t.full_text IS NOT NULL AND t.full_text NOT LIKE 'RT @%'
            )
            SELECT
              p.aid AS ego_account_id,
              CAST(r.account_id AS VARCHAR) AS partner_account_id,
              CAST(r.tweet_id AS VARCHAR) AS tweet_id,
              r.created_at,
              'quote' AS kind,
              r.full_text
            FROM read_parquet('{path}') r
            JOIN auth p ON CAST(r.quoted_tweet_id AS VARCHAR) = p.tid
            WHERE r.quoted_tweet_id IS NOT NULL
              AND CAST(r.account_id AS VARCHAR) != p.aid
              AND r.full_text IS NOT NULL
              AND r.full_text NOT LIKE 'RT @%'
            """
        ).fetchdf()
        mention_df = _empty_interlocutor_cache()
        if include_mentions and MENTION_CACHE_PATH.exists():
            # Mention cache stores inbound @ events; join dump for mentioner + text.
            mc = pd.read_parquet(MENTION_CACHE_PATH)
            mc["mentioned_user_id"] = mc["mentioned_user_id"].astype(str)
            mc["tweet_id"] = mc["tweet_id"].astype(str)
            mc = mc[
                mc["mentioned_user_id"].isin(missing)
                & mc["tweet_id"].ne("")
            ][["mentioned_user_id", "tweet_id"]].drop_duplicates()
            if len(mc):
                mt = CACHE_DIR / "_interlocutor_mention_ids.parquet"
                mc.to_parquet(mt, index=False)
                mention_df = con.execute(
                    f"""
                    SELECT
                      m.mentioned_user_id AS ego_account_id,
                      CAST(t.account_id AS VARCHAR) AS partner_account_id,
                      CAST(t.tweet_id AS VARCHAR) AS tweet_id,
                      t.created_at,
                      'mention' AS kind,
                      t.full_text
                    FROM read_parquet('{mt}') m
                    JOIN read_parquet('{path}') t
                      ON CAST(t.tweet_id AS VARCHAR) = m.tweet_id
                    WHERE t.full_text IS NOT NULL
                      AND t.full_text NOT LIKE 'RT @%'
                      AND CAST(t.account_id AS VARCHAR) != m.mentioned_user_id
                    """
                ).fetchdf()
                try:
                    mt.unlink()
                except OSError:
                    pass
        con.close()
        try:
            tmp.unlink()
        except OSError:
            pass

        new_parts = [p for p in (reply_df, quote_df, mention_df) if p is not None and len(p)]
        fresh = pd.concat(new_parts, ignore_index=True) if new_parts else _empty_interlocutor_cache()
        if len(fresh):
            fresh["ego_account_id"] = fresh["ego_account_id"].astype(str)
            fresh["partner_account_id"] = fresh["partner_account_id"].astype(str)
            fresh["tweet_id"] = fresh["tweet_id"].astype(str)
            fresh["kind"] = fresh["kind"].astype(str)
            fresh["full_text"] = fresh["full_text"].fillna("").astype(str)
            fresh["created_at"] = pd.to_datetime(fresh["created_at"], utc=True, errors="coerce")
            fresh = fresh[fresh["created_at"].notna()]
            fresh = fresh.drop_duplicates(["ego_account_id", "tweet_id", "kind"], keep="last")

        seen_ego = set(fresh["ego_account_id"].unique()) if len(fresh) else set()
        zero = [a for a in missing if a not in seen_ego]
        pieces = [cached, fresh]
        if zero:
            pieces.append(
                pd.DataFrame(
                    {
                        "ego_account_id": zero,
                        "partner_account_id": [""] * len(zero),
                        "tweet_id": [""] * len(zero),
                        "created_at": pd.to_datetime([pd.NaT] * len(zero), utc=True),
                        "kind": ["none"] * len(zero),
                        "full_text": [""] * len(zero),
                    }
                )
            )
        cached = pd.concat([p for p in pieces if len(p)], ignore_index=True)
        cached = cached.drop_duplicates(["ego_account_id", "tweet_id", "kind"], keep="last")
        cached.to_parquet(INTERLOCUTOR_CACHE_PATH, index=False)
        n_evt = int((cached["tweet_id"].astype(str) != "").sum())
        print(
            f"Wrote interlocutor cache: {n_evt:,} events / {len(cached):,} rows → "
            f"{INTERLOCUTOR_CACHE_PATH} ({len(zero):,} zero-inbound accounts marked)",
            flush=True,
        )

    usable = cached[
        cached["ego_account_id"].isin(accounts)
        & cached["tweet_id"].astype(str).ne("")
        & cached["created_at"].notna()
    ].copy()
    kinds = {"reply", "quote"}
    if include_mentions:
        kinds.add("mention")
    usable = usable[usable["kind"].astype(str).isin(kinds)]
    usable["created_at"] = pd.to_datetime(usable["created_at"], utc=True, errors="coerce")
    usable = usable[usable["created_at"].notna()]
    print(
        f"  usable interlocutor events: {len(usable):,} for "
        f"{usable['ego_account_id'].nunique():,}/{len(accounts):,} accounts "
        f"({(usable['ego_account_id'].nunique() / max(len(accounts), 1)):.1%} coverage; "
        f"kinds={sorted(kinds)})",
        flush=True,
    )
    return usable.reset_index(drop=True)


def ensure_interlocutor_embeddings(
    events: pd.DataFrame,
    *,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Ensure MiniLM vectors for interlocutor interaction tweets; return emb cols merged."""
    if events.empty:
        return events
    cache = ensure_embed_cache(events["tweet_id"], events["full_text"], batch_size=batch_size)
    emb_cols = [c for c in cache.columns if c.startswith("emb_")]
    merged = events.drop(columns=[c for c in events.columns if c.startswith("emb_")], errors="ignore")
    merged = merged.merge(cache[["tweet_id", *emb_cols]], on="tweet_id", how="left")
    if merged[emb_cols[0]].isna().any():
        n_miss = int(merged[emb_cols[0]].isna().sum())
        raise RuntimeError(f"Missing interlocutor embeddings after cache merge ({n_miss})")
    return merged


def load_author_graph(
    account_ids: pd.Series,
    *,
    include_mentions: bool = True,
    include_quotes: bool = True,
    include_reply_to_tweet: bool = True,
) -> dict:
    """Leakage-safe inbound graph events from the full dump for sampled authors.

    Returns dict with:
      reply_to_tweet: parent tweet_id → reply times (all author tweets in dump)
      reply_to_user: account_id → times of replies addressed to that user
      quote_to_user: account_id → times of quotes of that user's tweets
      mention_to_user: account_id → times of @-mentions of that user (CA user_mentions ⋊ tweet time)
      author_timeline: account_id → (tweet_ids, created_at ns) for all dump tweets

    Parent-author size (dyad) only needs timeline + reply_to_user — pass
    include_mentions=False / include_quotes=False / include_reply_to_tweet=False
    to skip the expensive mention API and unused edges.
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

    if include_reply_to_tweet:
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
    else:
        reply_to_tweet_df = pd.DataFrame(columns=["parent_id", "created_at"])

    # Prefer dump reply_to_user_id; when null (common on stream-era rows), recover
    # from parent tweet's account_id if that parent is by a sampled author.
    reply_to_user_df = con.execute(
        f"""
        WITH resolved AS (
          SELECT
            COALESCE(
              CAST(r.reply_to_user_id AS VARCHAR),
              p.account_id
            ) AS account_id,
            r.created_at,
            (r.reply_to_user_id IS NULL AND p.account_id IS NOT NULL) AS from_parent
          FROM read_parquet('{path}') r
          LEFT JOIN author_tweets p
            ON CAST(r.reply_to_tweet_id AS VARCHAR) = p.tweet_id
          WHERE COALESCE(
            CAST(r.reply_to_user_id AS VARCHAR),
            p.account_id
          ) IS NOT NULL
        )
        SELECT account_id, created_at, from_parent
        FROM resolved
        SEMI JOIN target_accounts a
          ON resolved.account_id = a.account_id
        """
    ).fetchdf()

    if include_quotes:
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
    else:
        quote_to_user_df = pd.DataFrame(columns=["account_id", "created_at"])
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

    n_reply_user = len(reply_to_user_df)
    n_from_parent = (
        int(reply_to_user_df["from_parent"].sum()) if n_reply_user and "from_parent" in reply_to_user_df else 0
    )
    print(
        f"  reply→tweet events: {len(reply_to_tweet_df):,} | "
        f"reply→user: {n_reply_user:,} ({n_from_parent:,} via parent account_id) | "
        f"quote→user: {len(quote_to_user_df):,}"
    )
    mention_to_user = (
        load_mention_times(accounts["account_id"]) if include_mentions else {}
    )
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
    if "archive_upload_id" in df.columns:
        df["is_stream"] = df["archive_upload_id"].isna().astype(np.int8)
    else:
        df["is_stream"] = np.int8(0)
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
                    "archive_upload_id": (
                        None
                        if "archive_upload_id" not in row.index or pd.isna(row["archive_upload_id"])
                        else str(row["archive_upload_id"])
                    ),
                    "is_stream": int(row["is_stream"]),
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


class XParentFetchInterrupted(Exception):
    """Raised when X API rate/credits stop a fetch; cache is already flushed."""

    def __init__(self, message: str, *, fetched: int, remaining: int):
        super().__init__(message)
        self.fetched = fetched
        self.remaining = remaining


def _x_oauth_session() -> OAuth1Session | None:
    ck = os.environ.get("X_CONSUMER_KEY", "").strip()
    cs = os.environ.get("X_CONSUMER_KEY_SECRET", "").strip()
    at = os.environ.get("X_ACCESS_TOKEN", "").strip()
    ats = os.environ.get("X_ACCESS_TOKEN_SECRET", "").strip()
    if not (ck and cs and at and ats):
        return None
    return OAuth1Session(ck, client_secret=cs, resource_owner_key=at, resource_owner_secret=ats)


def _empty_x_parent_cache() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "tweet_id",
            "account_id",
            "full_text",
            "created_at",
            "reply_to_tweet_id",
            "quoted_tweet_id",
            "fetch_status",
            "fetched_at",
        ]
    )


def load_x_parent_cache() -> pd.DataFrame:
    if not X_PARENT_CACHE_PATH.exists():
        return _empty_x_parent_cache()
    cache = pd.read_parquet(X_PARENT_CACHE_PATH)
    cache["tweet_id"] = cache["tweet_id"].astype(str)
    if "fetch_status" not in cache.columns:
        cache["fetch_status"] = "ok"
    return cache


def save_x_parent_cache(cache: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = cache.drop_duplicates("tweet_id", keep="last").reset_index(drop=True)
    out.to_parquet(X_PARENT_CACHE_PATH, index=False)


def _parse_x_tweet_row(tweet: dict) -> dict:
    reply_to = None
    quoted = None
    for ref in tweet.get("referenced_tweets") or []:
        typ = ref.get("type")
        rid = ref.get("id")
        if typ == "replied_to" and rid:
            reply_to = str(rid)
        elif typ == "quoted" and rid:
            quoted = str(rid)
    return {
        "tweet_id": str(tweet["id"]),
        "account_id": str(tweet["author_id"]) if tweet.get("author_id") is not None else pd.NA,
        "full_text": tweet.get("text") or "",
        "created_at": tweet.get("created_at"),
        "reply_to_tweet_id": reply_to,
        "quoted_tweet_id": quoted,
        "fetch_status": "ok",
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def _x_error_status(err: dict) -> str:
    title = str(err.get("title") or "").lower()
    detail = str(err.get("detail") or "").lower()
    typ = str(err.get("type") or "").lower()
    blob = f"{title} {detail} {typ}"
    if "not found" in blob or "could not find" in blob:
        return "not_found"
    if "forbidden" in blob or "not authorized" in blob or "protected" in blob:
        return "forbidden"
    return "not_found"


def _is_quota_http_status(status: int) -> bool:
    return status in {402, 429}


def normalize_tweet_id(value: object) -> str | None:
    """Return a bare snowflake id, or None if the value is not a usable tweet id."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s or s.lower() in {"none", "nan", "<na>"}:
        return None
    if "status/" in s:
        s = s.split("status/")[-1]
    s = s.split("?")[0].split("#")[0].strip().rstrip("/")
    if s.isdigit() and 5 <= len(s) <= 25:
        return s
    m = TWEET_ID_RE.search(str(value))
    return m.group(1) if m else None


def _normalize_ref_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip URL/query junk from reply/quote parent ids so dump + X joins match."""
    out = df.copy()
    for col in ("reply_to_tweet_id", "quoted_tweet_id"):
        if col not in out.columns:
            continue
        out[col] = out[col].map(normalize_tweet_id)
    return out


def _invalid_id_cache_rows(tweet_ids: list[str], *, status: str = "not_found") -> pd.DataFrame:
    ts = pd.Timestamp.now(tz="UTC").isoformat()
    return pd.DataFrame(
        [
            {
                "tweet_id": tid,
                "account_id": pd.NA,
                "full_text": "",
                "created_at": pd.NaT,
                "reply_to_tweet_id": None,
                "quoted_tweet_id": None,
                "fetch_status": status,
                "fetched_at": ts,
            }
            for tid in tweet_ids
        ]
    )


def _x_parent_rows_from_cache(tweet_ids: list[str], cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dump-shaped ok rows for ids already in the X parent cache (no API calls)."""
    cols = [
        "tweet_id",
        "account_id",
        "full_text",
        "created_at",
        "reply_to_tweet_id",
        "quoted_tweet_id",
    ]
    empty = pd.DataFrame(columns=cols)
    if not tweet_ids:
        return empty
    if cache is None:
        cache = load_x_parent_cache()
    if cache.empty:
        return empty
    want = set(tweet_ids)
    ok = cache.loc[
        (cache["tweet_id"].isin(want)) & (cache["fetch_status"] == "ok"),
        cols,
    ].copy()
    if ok.empty:
        return empty
    ok["created_at"] = pd.to_datetime(ok["created_at"], utc=True, errors="coerce")
    ok = ok.loc[ok["created_at"].notna()].copy()
    return ok.reset_index(drop=True)


def fetch_x_parent_tweets(
    tweet_ids: pd.Series | list[str] | np.ndarray,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Lookup missing parents via X API; cache after every batch for resume.

    Returns dump-shaped parent rows with fetch_status == 'ok' only.
    On rate-limit / credit exhaustion, flushes the cache and raises
    XParentFetchInterrupted so callers can keep going with partial coverage.
    """
    raw_uniq = list(dict.fromkeys(str(t) for t in tweet_ids if pd.notna(t) and str(t)))
    # Normalize before any API call so "?s=20" / status URLs never burn a batch.
    normalized: list[str] = []
    invalid_raw: list[str] = []
    for raw in raw_uniq:
        nid = normalize_tweet_id(raw)
        if nid is None:
            invalid_raw.append(raw)
        else:
            normalized.append(nid)
    uniq = list(dict.fromkeys(normalized))
    if invalid_raw:
        print(
            f"X parent ids: dropped {len(invalid_raw):,} non-snowflake ref ids "
            f"(cached as not_found, no API)",
            flush=True,
        )

    if not uniq and not invalid_raw:
        return _x_parent_rows_from_cache([])

    cache = _empty_x_parent_cache() if force else load_x_parent_cache()
    if force and X_PARENT_CACHE_PATH.exists():
        print(f"Force X parent refetch: ignoring {X_PARENT_CACHE_PATH.name}", flush=True)

    if invalid_raw:
        inv = _invalid_id_cache_rows(invalid_raw)
        cache = inv if cache.empty else pd.concat([cache, inv], ignore_index=True)
        cache = cache.drop_duplicates("tweet_id", keep="last")
        save_x_parent_cache(cache)

    have_done: set[str] = set()
    if not cache.empty:
        done = cache.loc[cache["fetch_status"].isin(X_PARENT_DONE_STATUSES), "tweet_id"]
        have_done = set(done.astype(str))
    miss = [tid for tid in uniq if tid not in have_done]
    n_hit = len(uniq) - len(miss)
    print(f"X parent cache: {n_hit:,}/{len(uniq):,} done, {len(miss):,} to fetch", flush=True)

    if miss:
        sess = _x_oauth_session()
        if sess is None:
            print(
                "X parent fetch skipped: set X_CONSUMER_KEY / X_CONSUMER_KEY_SECRET / "
                "X_ACCESS_TOKEN / X_ACCESS_TOKEN_SECRET to enable fallback",
                flush=True,
            )
        else:
            params_base = {
                "tweet.fields": "created_at,author_id,text,referenced_tweets",
            }
            fetched = 0
            now_iso = lambda: pd.Timestamp.now(tz="UTC").isoformat()
            pending = list(miss)
            while pending:
                batch = pending[:X_TWEETS_BATCH]
                pending = pending[X_TWEETS_BATCH:]
                params = {**params_base, "ids": ",".join(batch)}
                try:
                    resp = sess.get(X_TWEETS_URL, params=params, timeout=60)
                except requests.RequestException as exc:
                    save_x_parent_cache(cache)
                    raise XParentFetchInterrupted(
                        f"X parent fetch network error after {fetched:,} new rows: {exc}",
                        fetched=fetched,
                        remaining=len(pending) + len(batch),
                    ) from exc

                if _is_quota_http_status(resp.status_code) or resp.status_code in {401, 403}:
                    save_x_parent_cache(cache)
                    retry_after = resp.headers.get("retry-after") or resp.headers.get(
                        "x-rate-limit-reset"
                    )
                    raise XParentFetchInterrupted(
                        f"X parent fetch stopped on HTTP {resp.status_code}"
                        f"{f' (retry-after/reset={retry_after})' if retry_after else ''}; "
                        f"cache has {len(cache):,} rows. "
                        f"Re-run: uv run prepare.py --parent-only --x-parents",
                        fetched=fetched,
                        remaining=len(pending) + len(batch),
                    )

                if resp.status_code == 400:
                    # One bad id can fail a whole batch; quarantine and retry the rest.
                    bad: list[str] = []
                    try:
                        err_body = resp.json()
                        for e in err_body.get("errors") or []:
                            for v in (e.get("parameters") or {}).get("ids") or []:
                                nid = normalize_tweet_id(v) or str(v)
                                bad.append(nid)
                            msg = str(e.get("message") or "")
                            m = TWEET_ID_RE.search(msg)
                            if m:
                                bad.append(m.group(1))
                    except Exception:
                        bad = []
                    bad = [b for b in dict.fromkeys(bad) if b in batch]
                    if not bad and len(batch) == 1:
                        bad = list(batch)
                    if not bad:
                        # Fall back: split batch so a single poison id cannot block the rest.
                        if len(batch) > 1:
                            mid = len(batch) // 2
                            pending = batch[:mid] + batch[mid:] + pending
                            print(
                                f"  X parents: HTTP 400 on {len(batch)}-id batch; "
                                f"bisecting to isolate bad id",
                                flush=True,
                            )
                            continue
                        bad = list(batch)
                    inv = _invalid_id_cache_rows(bad)
                    cache = inv if cache.empty else pd.concat([cache, inv], ignore_index=True)
                    cache = cache.drop_duplicates("tweet_id", keep="last")
                    save_x_parent_cache(cache)
                    rest = [t for t in batch if t not in set(bad)]
                    pending = rest + pending
                    print(
                        f"  X parents: quarantined {len(bad):,} invalid id(s) from HTTP 400 "
                        f"(no credit wasted on retry of those)",
                        flush=True,
                    )
                    continue

                if resp.status_code != 200:
                    save_x_parent_cache(cache)
                    raise XParentFetchInterrupted(
                        f"X parent fetch HTTP {resp.status_code}: {resp.text[:300]}",
                        fetched=fetched,
                        remaining=len(pending) + len(batch),
                    )

                body = resp.json()
                rows: list[dict] = []
                seen: set[str] = set()
                for tweet in body.get("data") or []:
                    row = _parse_x_tweet_row(tweet)
                    rows.append(row)
                    seen.add(row["tweet_id"])
                ts = now_iso()
                for err in body.get("errors") or []:
                    rid = err.get("resource_id") or err.get("value")
                    if rid is None:
                        continue
                    tid = str(rid)
                    if tid in seen:
                        continue
                    rows.append(
                        {
                            "tweet_id": tid,
                            "account_id": pd.NA,
                            "full_text": "",
                            "created_at": pd.NaT,
                            "reply_to_tweet_id": None,
                            "quoted_tweet_id": None,
                            "fetch_status": _x_error_status(err),
                            "fetched_at": ts,
                        }
                    )
                    seen.add(tid)
                for tid in batch:
                    if tid not in seen:
                        rows.append(
                            {
                                "tweet_id": tid,
                                "account_id": pd.NA,
                                "full_text": "",
                                "created_at": pd.NaT,
                                "reply_to_tweet_id": None,
                                "quoted_tweet_id": None,
                                "fetch_status": "not_found",
                                "fetched_at": ts,
                            }
                        )

                batch_df = pd.DataFrame(rows)
                cache = batch_df if cache.empty else pd.concat([cache, batch_df], ignore_index=True)
                cache = cache.drop_duplicates("tweet_id", keep="last")
                save_x_parent_cache(cache)
                fetched += len(batch)
                done_n = fetched
                total_n = len(miss)
                if done_n == len(batch) or done_n % 500 < len(batch) or not pending:
                    print(
                        f"  X parents: {min(done_n, total_n):,}/{total_n:,} "
                        f"(cache={len(cache):,})",
                        flush=True,
                    )
                if pending:
                    time.sleep(0.5)

    ok = _x_parent_rows_from_cache(uniq, cache)
    print(f"  X parents usable: {len(ok):,}/{len(uniq):,} requested", flush=True)
    return ok


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
    parent_aid = np.full(n, None, dtype=object)

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
        out[f"{prefix}_parent_account_id"] = parent_aid
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
        aids = meta["account_id"].astype(str).to_numpy()[ok]
        parent_aid[ok] = aids
        same_author[ok] = (
            aids == out["account_id"].astype(str).to_numpy()[ok]
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
    out[f"{prefix}_parent_account_id"] = parent_aid
    return out


def _empty_parent_feature_frame(out: pd.DataFrame) -> pd.DataFrame:
    for prefix in ("reply", "quote"):
        out[f"has_{prefix}_parent"] = 0
        out[f"{prefix}_parent_emb_cos"] = 0.0
        out[f"{prefix}_parent_log_text_len"] = 0.0
        out[f"{prefix}_parent_is_reply"] = 0
        out[f"{prefix}_parent_is_quote"] = 0
        out[f"{prefix}_parent_same_author"] = 0
        out[f"{prefix}_parent_age_hours"] = 0.0
        out[f"{prefix}_parent_replies"] = 0.0
        out[f"{prefix}_parent_account_id"] = None
    return out


def attach_parent_context(
    df: pd.DataFrame,
    batch_size: int,
    *,
    use_x: bool = True,
    force_x: bool = False,
) -> pd.DataFrame:
    """Join reply/quote parents from the dump (+ optional X API); leakage-safe context."""
    out = df.reset_index(drop=True).copy()
    if "reply_to_tweet_id" not in out.columns or "quoted_tweet_id" not in out.columns:
        raise RuntimeError("Dataset missing reply_to_tweet_id / quoted_tweet_id for parent join")
    out = _normalize_ref_id_columns(out)

    reply_ids = out["reply_to_tweet_id"].dropna().astype(str)
    quote_ids = out["quoted_tweet_id"].dropna().astype(str)
    all_ids = pd.Series(pd.unique(pd.concat([reply_ids, quote_ids], ignore_index=True)))
    print(
        f"Parent context: {reply_ids.nunique():,} reply parents, "
        f"{quote_ids.nunique():,} quote parents, {len(all_ids):,} unique",
        flush=True,
    )
    if all_ids.empty:
        return _empty_parent_feature_frame(out)

    parents = load_tweets_by_id(all_ids)
    dump_ids = set(parents["tweet_id"].astype(str)) if not parents.empty else set()
    missing = [tid for tid in all_ids.astype(str).tolist() if tid not in dump_ids]
    print(f"  dump parents: {len(dump_ids):,}; missing for X fallback: {len(missing):,}", flush=True)

    x_interrupted = False
    if use_x and missing:
        try:
            x_parents = fetch_x_parent_tweets(missing, force=force_x)
        except XParentFetchInterrupted as exc:
            x_interrupted = True
            print(f"WARNING: {exc}", flush=True)
            # Cache was flushed before the raise; do not re-hit the API.
            x_parents = _x_parent_rows_from_cache(missing)
            print(
                f"  continuing with {len(x_parents):,} cached X parents "
                f"({exc.remaining:,} ids still pending)",
                flush=True,
            )
        if not x_parents.empty:
            parents = (
                x_parents
                if parents.empty
                else pd.concat([parents, x_parents], ignore_index=True)
            )
            # Prefer dump row when both exist.
            parents = parents.drop_duplicates("tweet_id", keep="first")
            print(
                f"  parents after X merge: {len(parents):,} "
                f"(+{len(x_parents):,} from X/cache)",
                flush=True,
            )
    elif missing and not use_x:
        print("  X parent fallback off (--no-x-parents)", flush=True)

    if parents.empty:
        out = _empty_parent_feature_frame(out)
        if x_interrupted:
            print(
                "Parent features empty after interrupt; re-run: uv run prepare.py --parent-only",
                flush=True,
            )
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
    if x_interrupted:
        print(
            "X parent fetch incomplete — cache is durable; "
            "re-run: uv run prepare.py --parent-only",
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


def backfill_reply_user_from_dump(df: pd.DataFrame) -> pd.DataFrame:
    """Fill null reply_to_user_id from parent account_id when the parent is in the dump."""
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    if "reply_to_tweet_id" not in out.columns:
        return out
    if "reply_to_user_id" not in out.columns:
        out["reply_to_user_id"] = pd.NA
    need = out["reply_to_tweet_id"].notna() & out["reply_to_user_id"].isna()
    if not need.any():
        return out
    parent_ids = out.loc[need, "reply_to_tweet_id"].astype(str).unique()
    tmp = CACHE_DIR / "_reply_parent_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tweet_id": parent_ids}).to_parquet(tmp, index=False)
    path = str(RAW_PARQUET)
    print(f"Backfilling reply_to_user_id for {int(need.sum()):,} replies via parent…", flush=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    parents = con.execute(
        f"""
        SELECT
          CAST(t.tweet_id AS VARCHAR) AS reply_to_tweet_id,
          CAST(t.account_id AS VARCHAR) AS parent_account_id
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
    parents = parents.drop_duplicates("reply_to_tweet_id", keep="first")
    merged = out.loc[need, ["tweet_id", "reply_to_tweet_id"]].merge(
        parents, on="reply_to_tweet_id", how="left"
    )
    fill = merged.set_index("tweet_id")["parent_account_id"]
    filled = int(fill.notna().sum())
    out.loc[need, "reply_to_user_id"] = out.loc[need, "tweet_id"].map(fill).fillna(
        out.loc[need, "reply_to_user_id"]
    )
    print(f"  recovered reply_to_user_id on {filled:,}/{int(need.sum()):,} null-uid replies", flush=True)
    return out


def fetch_archive_uploads(*, force: bool = False) -> pd.DataFrame:
    """Page CA archive_upload rows (id, account_id, end_date, phase, created_at)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if UPLOAD_CACHE_PATH.exists() and not force:
        cached = pd.read_parquet(UPLOAD_CACHE_PATH)
        print(f"Using cached archive_upload rows: {len(cached):,}", flush=True)
        return cached

    print("Fetching archive_upload rows from CA…", flush=True)
    session = requests.Session()
    rows: list[dict[str, object]] = []
    last_id = 0
    while True:
        resp = session.get(
            f"{CA_API}/archive_upload",
            headers=_ca_headers(),
            params={
                "select": "id,account_id,end_date,upload_phase,created_at",
                "id": f"gt.{last_id}",
                "order": "id.asc",
                "limit": "1000",
            },
            timeout=60,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for row in batch:
            rows.append(
                {
                    "id": int(row["id"]),
                    "account_id": str(row["account_id"]),
                    "end_date": row.get("end_date"),
                    "upload_phase": row.get("upload_phase"),
                    "created_at": row.get("created_at"),
                }
            )
            last_id = max(last_id, int(row["id"]))
        if len(batch) < 1000:
            break
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=["id", "account_id", "end_date", "upload_phase", "created_at"]
        )
    else:
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
        # end_date is a calendar date from CA; treat as UTC midnight.
        out["end_date"] = pd.to_datetime(out["end_date"], utc=True, errors="coerce")
    out.to_parquet(UPLOAD_CACHE_PATH, index=False)
    print(f"  cached {len(out):,} archive_upload rows → {UPLOAD_CACHE_PATH}", flush=True)
    return out


def attach_upload_end_features(df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """Leakage-safe hours since account's latest archive upload end_date before T.

    Only uploads with created_at < tweet.created_at are eligible (upload event
    observed before the tweet). Prefer completed phases when present.
    """
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    out["account_id"] = out["account_id"].astype(str)
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    uploads = fetch_archive_uploads(force=force)
    drop = [c for c in ("hours_since_upload_end", "has_upload_end", "upload_end_date") if c in out.columns]
    if drop:
        out = out.drop(columns=drop)

    hours = np.full(len(out), np.nan, dtype=np.float64)
    has = np.zeros(len(out), dtype=np.int8)
    end_dates = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    if not uploads.empty:
        up = uploads.copy()
        up["account_id"] = up["account_id"].astype(str)
        up["created_at"] = pd.to_datetime(up["created_at"], utc=True, errors="coerce")
        up["end_date"] = pd.to_datetime(up["end_date"], utc=True, errors="coerce")
        up = up.dropna(subset=["created_at", "end_date"])
        # Prefer completed; keep failed/other as fallback when no completed exists.
        up["_pref"] = (up["upload_phase"].astype(str) == "completed").astype(np.int8)
        up = up.sort_values(["account_id", "created_at", "end_date"], kind="mergesort")
        # Parallel arrays of upload-created / end / pref, keyed by account.
        by_acct: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for aid, g in up.groupby("account_id", sort=False):
            by_acct[str(aid)] = (
                g["created_at"].astype("int64").to_numpy(),
                g["end_date"].astype("int64").to_numpy(),
                g["_pref"].to_numpy(dtype=np.int8),
            )
        tweet_ns = pd.to_datetime(out["created_at"], utc=True).astype("int64").to_numpy()
        for i, (aid, t_ns) in enumerate(
            zip(out["account_id"].to_numpy(), tweet_ns, strict=True)
        ):
            pack = by_acct.get(str(aid))
            if pack is None:
                continue
            created_u, ends_u, pref_u = pack
            n_before = int(np.searchsorted(created_u, t_ns, side="left"))
            if n_before <= 0:
                continue
            ends = ends_u[:n_before]
            prefs = pref_u[:n_before]
            if prefs.max() > 0:
                ends = ends[prefs == 1]
            end_ns = int(ends.max())
            hours[i] = float(t_ns - end_ns) / 3.6e12
            has[i] = 1
            end_dates[i] = np.datetime64(end_ns, "ns")

    out["hours_since_upload_end"] = np.where(has == 1, hours, 0.0).astype(np.float32)
    out["has_upload_end"] = has
    out["upload_end_date"] = pd.to_datetime(end_dates, utc=True)
    if has.any():
        print(
            f"Upload-end features: has_upload_end={has.mean():.1%} "
            f"hours p50={np.nanmedian(hours):.1f}",
            flush=True,
        )
    else:
        print("Upload-end features: has_upload_end=0.0%", flush=True)
    return out


def attach_provenance(df: pd.DataFrame, *, force_uploads: bool = False) -> pd.DataFrame:
    """Attach archive_upload_id / is_stream from the dump (stream ⇒ null upload id)."""
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    ids = pd.DataFrame({"tweet_id": out["tweet_id"].unique()})
    tmp = CACHE_DIR / "_provenance_tweet_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids.to_parquet(tmp, index=False)
    path = str(RAW_PARQUET)
    print(f"Attaching provenance for {len(ids):,} tweets…", flush=True)
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
    out = out.drop(columns=[c for c in ("archive_upload_id", "is_stream") if c in out.columns], errors="ignore")
    out = out.merge(prov, on="tweet_id", how="left")
    out["is_stream"] = out["archive_upload_id"].isna().astype(np.int8)
    print(
        f"  is_stream={out['is_stream'].mean():.1%} "
        f"(upload-linked={(1.0 - out['is_stream'].mean()):.1%})",
        flush=True,
    )
    out = attach_upload_end_features(out, force=force_uploads)
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


# ---------------------------------------------------------------------------
# Reply / quote dyad + parent-author size (leakage-safe, pre-T only)
# ---------------------------------------------------------------------------

DYAD_FEATURE_SUFFIXES = (
    "parent_author_known",
    "parent_author_hist_n",
    "parent_author_inbound",
    "parent_author_inbound_per",
    "dyad_inbound",
    "dyad_outbound",
    "dyad_has_prior",
    "dyad_bidir",
    # Cosine / L2 vs mean of parent author's pre-T MiniLM corpus (capped).
    "parent_author_emb_cos",
    "parent_author_emb_l2",
    "parent_author_emb_log_n",
    "parent_author_emb_has",
)


def dyad_feature_columns() -> list[str]:
    cols: list[str] = []
    for prefix in ("reply", "quote"):
        for suf in DYAD_FEATURE_SUFFIXES:
            cols.append(f"{prefix}_{suf}")
    return cols


def load_parent_author_size_maps(account_ids: pd.Series) -> dict:
    """Pre-T size proxies for parent authors: tweet times + inbound reply times.

    Lighter than load_author_graph: no tweet_id arrays, quotes, or mention API.
    """
    accounts = pd.DataFrame({"account_id": account_ids.astype(str).unique()})
    if accounts.empty:
        return {"author_times": {}, "reply_to_user": {}}
    tmp_accounts = CACHE_DIR / "_parent_author_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    accounts.to_parquet(tmp_accounts, index=False)
    path = str(RAW_PARQUET)
    print(f"Scanning dump for parent-author size on {len(accounts):,} accounts…", flush=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    author_times_df = con.execute(
        f"""
        SELECT
          CAST(t.account_id AS VARCHAR) AS account_id,
          t.created_at
        FROM read_parquet('{path}') t
        SEMI JOIN read_parquet('{tmp_accounts}') a
          ON CAST(t.account_id AS VARCHAR) = a.account_id
        WHERE t.full_text IS NOT NULL
          AND t.full_text NOT LIKE 'RT @%'
        """
    ).fetchdf()
    # reply→user with parent-account coalesce (same repair as load_author_graph).
    reply_to_user_df = con.execute(
        f"""
        WITH target AS (SELECT account_id FROM read_parquet('{tmp_accounts}')),
        author_tweets AS (
          SELECT CAST(t.tweet_id AS VARCHAR) AS tweet_id,
                 CAST(t.account_id AS VARCHAR) AS account_id
          FROM read_parquet('{path}') t
          SEMI JOIN target a ON CAST(t.account_id AS VARCHAR) = a.account_id
        ),
        resolved AS (
          SELECT
            COALESCE(CAST(r.reply_to_user_id AS VARCHAR), p.account_id) AS account_id,
            r.created_at
          FROM read_parquet('{path}') r
          LEFT JOIN author_tweets p
            ON CAST(r.reply_to_tweet_id AS VARCHAR) = p.tweet_id
          WHERE COALESCE(CAST(r.reply_to_user_id AS VARCHAR), p.account_id) IS NOT NULL
        )
        SELECT account_id, created_at
        FROM resolved
        SEMI JOIN target a ON resolved.account_id = a.account_id
        """
    ).fetchdf()
    con.close()
    try:
        tmp_accounts.unlink()
    except OSError:
        pass
    author_times = _sorted_time_map(author_times_df, "account_id")
    reply_to_user = _sorted_time_map(reply_to_user_df, "account_id")
    print(
        f"  parent-author tweets: {len(author_times_df):,} | "
        f"reply→user events: {len(reply_to_user_df):,}",
        flush=True,
    )
    return {"author_times": author_times, "reply_to_user": reply_to_user}


def resolve_parent_account_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure reply/quote_parent_account_id from dump (+ X parent cache when present)."""
    out = df.reset_index(drop=True).copy()
    out = _normalize_ref_id_columns(out)
    for prefix, id_col in (("reply", "reply_to_tweet_id"), ("quote", "quoted_tweet_id")):
        aid_col = f"{prefix}_parent_account_id"
        if aid_col not in out.columns:
            out[aid_col] = pd.NA
        else:
            # Normalize empty strings / "None" from prior object columns.
            out[aid_col] = out[aid_col].map(
                lambda x: str(x) if pd.notna(x) and str(x) not in {"", "None", "nan", "<NA>"} else pd.NA
            )

    need_ids: list[str] = []
    for prefix, id_col in (("reply", "reply_to_tweet_id"), ("quote", "quoted_tweet_id")):
        aid_col = f"{prefix}_parent_account_id"
        if id_col not in out.columns:
            continue
        need = out[id_col].notna() & out[aid_col].isna()
        if need.any():
            need_ids.extend(out.loc[need, id_col].astype(str).tolist())

    if not need_ids:
        return out

    uniq = pd.Series(pd.unique(pd.Series(need_ids)))
    parents = load_tweets_by_id(uniq)
    by_id: dict[str, str] = {}
    if not parents.empty:
        for _, row in parents.iterrows():
            aid = row["account_id"]
            if pd.notna(aid):
                by_id[str(row["tweet_id"])] = str(aid)

    missing = [tid for tid in uniq.astype(str).tolist() if tid not in by_id]
    if missing and X_PARENT_CACHE_PATH.exists():
        xcache = load_x_parent_cache()
        ok = xcache["fetch_status"].astype(str).eq("ok") if "fetch_status" in xcache.columns else True
        xok = xcache.loc[ok].copy()
        xok = xok[xok["tweet_id"].astype(str).isin(missing)]
        for _, row in xok.iterrows():
            aid = row.get("account_id")
            if pd.notna(aid) and str(aid) not in {"", "None", "nan"}:
                by_id[str(row["tweet_id"])] = str(aid)

    filled = 0
    for prefix, id_col in (("reply", "reply_to_tweet_id"), ("quote", "quoted_tweet_id")):
        aid_col = f"{prefix}_parent_account_id"
        if id_col not in out.columns:
            continue
        need = out[id_col].notna() & out[aid_col].isna()
        if not need.any():
            continue
        mapped = out.loc[need, id_col].astype(str).map(by_id)
        out.loc[need, aid_col] = mapped
        filled += int(mapped.notna().sum())
    print(f"Parent author ids: filled {filled:,} missing reply/quote parent accounts", flush=True)
    return out


def load_dyad_events(
    ego_ids: pd.Series,
    partner_ids: pd.Series,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Reply/quote interactions between ego authors and partner (parent) authors.

    Returns ego_account_id, partner_account_id, created_at, direction ∈ {inbound,outbound}.
    Cached by ego so --dyad-only / rebuilds can resume.
    """
    egos = sorted({str(a) for a in ego_ids.astype(str).tolist() if a and a != "nan"})
    partners = sorted({str(a) for a in partner_ids.astype(str).tolist() if a and a != "nan"})
    if not egos or not partners:
        return pd.DataFrame(
            columns=["ego_account_id", "partner_account_id", "created_at", "direction"]
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame(
        columns=["ego_account_id", "partner_account_id", "created_at", "direction"]
    )
    if DYAD_CACHE_PATH.exists() and not force:
        cached = pd.read_parquet(DYAD_CACHE_PATH)
        for col in ("ego_account_id", "partner_account_id", "direction"):
            if col in cached.columns:
                cached[col] = cached[col].astype(str)
        print(f"Dyad cache: {len(cached):,} rows ({DYAD_CACHE_PATH.name})", flush=True)

    have = set(cached["ego_account_id"].unique()) if len(cached) else set()
    missing = [a for a in egos if a not in have]
    if missing:
        # Full ego history (any partner): cache is complete per ego, then filter
        # to requested parent authors at read time.
        print(f"Scanning dump for dyad events on {len(missing):,} ego accounts…", flush=True)
        tmp_ego = CACHE_DIR / "_dyad_ego_ids.parquet"
        pd.DataFrame({"account_id": missing}).to_parquet(tmp_ego, index=False)
        path = str(RAW_PARQUET)
        con = duckdb.connect()
        con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
        # Materialize ego tweet ids + outbound parent refs so partner lookup
        # is a small SEMI JOIN rather than a full dump self-join.
        ego_rows = con.execute(
            f"""
            SELECT
              CAST(t.tweet_id AS VARCHAR) AS tweet_id,
              CAST(t.account_id AS VARCHAR) AS account_id,
              CAST(t.reply_to_tweet_id AS VARCHAR) AS reply_to_tweet_id,
              CAST(t.quoted_tweet_id AS VARCHAR) AS quoted_tweet_id,
              t.created_at
            FROM read_parquet('{path}') t
            SEMI JOIN read_parquet('{tmp_ego}') e
              ON CAST(t.account_id AS VARCHAR) = e.account_id
            WHERE t.full_text IS NOT NULL AND t.full_text NOT LIKE 'RT @%'
            """
        ).fetchdf()
        tmp_ego_tweets = CACHE_DIR / "_dyad_ego_tweets.parquet"
        ego_rows.to_parquet(tmp_ego_tweets, index=False)
        parent_ids = sorted(
            {
                str(x)
                for x in pd.concat(
                    [ego_rows["reply_to_tweet_id"], ego_rows["quoted_tweet_id"]],
                    ignore_index=True,
                )
                .dropna()
                .astype(str)
                .tolist()
                if x and x not in {"None", "nan", "<NA>"}
            }
        )
        tmp_parents = CACHE_DIR / "_dyad_outbound_parent_ids.parquet"
        pd.DataFrame({"tweet_id": parent_ids}).to_parquet(tmp_parents, index=False)
        parent_authors = (
            con.execute(
                f"""
                SELECT
                  CAST(t.tweet_id AS VARCHAR) AS tweet_id,
                  CAST(t.account_id AS VARCHAR) AS account_id
                FROM read_parquet('{path}') t
                SEMI JOIN read_parquet('{tmp_parents}') p
                  ON CAST(t.tweet_id AS VARCHAR) = p.tweet_id
                """
            ).fetchdf()
            if parent_ids
            else pd.DataFrame(columns=["tweet_id", "account_id"])
        )
        tmp_parent_authors = CACHE_DIR / "_dyad_parent_authors.parquet"
        parent_authors.to_parquet(tmp_parent_authors, index=False)

        fresh = con.execute(
            f"""
            WITH ego_tweets AS (
              SELECT tweet_id, account_id, reply_to_tweet_id, quoted_tweet_id, created_at
              FROM read_parquet('{tmp_ego_tweets}')
            ),
            parent_authors AS (
              SELECT tweet_id, account_id FROM read_parquet('{tmp_parent_authors}')
            ),
            inbound_reply AS (
              SELECT e.account_id AS ego_account_id,
                     CAST(r.account_id AS VARCHAR) AS partner_account_id,
                     r.created_at,
                     'inbound' AS direction
              FROM read_parquet('{path}') r
              JOIN ego_tweets e ON CAST(r.reply_to_tweet_id AS VARCHAR) = e.tweet_id
              WHERE r.reply_to_tweet_id IS NOT NULL
                AND CAST(r.account_id AS VARCHAR) != e.account_id
            ),
            inbound_quote AS (
              SELECT e.account_id AS ego_account_id,
                     CAST(r.account_id AS VARCHAR) AS partner_account_id,
                     r.created_at,
                     'inbound' AS direction
              FROM read_parquet('{path}') r
              JOIN ego_tweets e ON CAST(r.quoted_tweet_id AS VARCHAR) = e.tweet_id
              WHERE r.quoted_tweet_id IS NOT NULL
                AND CAST(r.account_id AS VARCHAR) != e.account_id
            ),
            outbound_reply AS (
              SELECT e.account_id AS ego_account_id,
                     p.account_id AS partner_account_id,
                     e.created_at,
                     'outbound' AS direction
              FROM ego_tweets e
              JOIN parent_authors p ON e.reply_to_tweet_id = p.tweet_id
              WHERE e.reply_to_tweet_id IS NOT NULL
                AND e.account_id != p.account_id
            ),
            outbound_quote AS (
              SELECT e.account_id AS ego_account_id,
                     p.account_id AS partner_account_id,
                     e.created_at,
                     'outbound' AS direction
              FROM ego_tweets e
              JOIN parent_authors p ON e.quoted_tweet_id = p.tweet_id
              WHERE e.quoted_tweet_id IS NOT NULL
                AND e.account_id != p.account_id
            )
            SELECT * FROM inbound_reply
            UNION ALL SELECT * FROM inbound_quote
            UNION ALL SELECT * FROM outbound_reply
            UNION ALL SELECT * FROM outbound_quote
            """
        ).fetchdf()
        con.close()
        for tmp in (tmp_ego, tmp_ego_tweets, tmp_parents, tmp_parent_authors):
            try:
                tmp.unlink()
            except OSError:
                pass

        if len(fresh):
            fresh["ego_account_id"] = fresh["ego_account_id"].astype(str)
            fresh["partner_account_id"] = fresh["partner_account_id"].astype(str)
            fresh["direction"] = fresh["direction"].astype(str)
            fresh["created_at"] = pd.to_datetime(fresh["created_at"], utc=True, errors="coerce")
            fresh = fresh[fresh["created_at"].notna()]
        else:
            fresh = pd.DataFrame(
                columns=["ego_account_id", "partner_account_id", "created_at", "direction"]
            )

        seen_ego = set(fresh["ego_account_id"].unique()) if len(fresh) else set()
        zero = [a for a in missing if a not in seen_ego]
        pieces = [cached, fresh]
        if zero:
            # Sentinel so we do not rescan egos with no dyad events vs current partners.
            pieces.append(
                pd.DataFrame(
                    {
                        "ego_account_id": zero,
                        "partner_account_id": [""] * len(zero),
                        "created_at": pd.to_datetime([pd.NaT] * len(zero), utc=True),
                        "direction": ["none"] * len(zero),
                    }
                )
            )
        cached = pd.concat([p for p in pieces if len(p)], ignore_index=True)
        cached.to_parquet(DYAD_CACHE_PATH, index=False)
        n_evt = int(
            (cached["partner_account_id"].astype(str) != "").sum()
            if len(cached)
            else 0
        )
        print(
            f"Wrote dyad cache: {n_evt:,} events / {len(cached):,} rows → {DYAD_CACHE_PATH} "
            f"({len(zero):,} zero-dyad egos marked)",
            flush=True,
        )

    partner_set = set(partners)
    usable = cached[
        cached["ego_account_id"].isin(egos)
        & cached["partner_account_id"].astype(str).isin(partner_set)
        & cached["direction"].astype(str).isin({"inbound", "outbound"})
        & cached["created_at"].notna()
    ].copy()
    if len(usable):
        usable["created_at"] = pd.to_datetime(usable["created_at"], utc=True, errors="coerce")
        usable = usable[usable["created_at"].notna()]
    print(
        f"  usable dyad events: {len(usable):,} "
        f"({usable['ego_account_id'].nunique() if len(usable) else 0:,} egos × "
        f"{usable['partner_account_id'].nunique() if len(usable) else 0:,} partners)",
        flush=True,
    )
    return usable.reset_index(drop=True)


def _dyad_time_maps(
    events: pd.DataFrame,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """(ego, partner) → sorted inbound / outbound times."""
    inbound: dict[tuple[str, str], np.ndarray] = {}
    outbound: dict[tuple[str, str], np.ndarray] = {}
    if events.empty:
        return inbound, outbound
    for (ego, partner, direction), g in events.groupby(
        ["ego_account_id", "partner_account_id", "direction"], sort=False
    ):
        key = (str(ego), str(partner))
        times = np.sort(g["created_at"].to_numpy(dtype="datetime64[ns]"))
        if direction == "inbound":
            inbound[key] = times
        elif direction == "outbound":
            outbound[key] = times
    return inbound, outbound


def load_parent_author_corpus_tweets(
    parent_max_t: pd.DataFrame,
    *,
    per_account_cap: int = 200,
) -> pd.DataFrame:
    """Capped parent-author tweets for MiniLM corpus affinity.

    For each parent account, keep up to ``per_account_cap`` dump tweets with
    ``created_at < max_child_created_at``, spaced evenly across that history
    (not just the most recent — recency-only caps leave older children with
    empty pre-T means). Leakage-safe at feature time: each child still uses
    only tweets strictly before its own *T*.
    """
    empty = pd.DataFrame(columns=["tweet_id", "account_id", "full_text", "created_at"])
    if parent_max_t is None or parent_max_t.empty or per_account_cap <= 0:
        return empty
    need = parent_max_t[["account_id", "max_created_at"]].copy()
    need["account_id"] = need["account_id"].astype(str)
    need["max_created_at"] = pd.to_datetime(need["max_created_at"], utc=True)
    need = need.dropna(subset=["account_id", "max_created_at"])
    need = need[need["account_id"].ne("")]
    if need.empty:
        return empty
    tmp = CACHE_DIR / "_parent_author_max_t.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    need.to_parquet(tmp, index=False)
    path = str(RAW_PARQUET)
    cap = int(per_account_cap)
    print(
        f"Loading parent-author corpus texts: {len(need):,} authors, "
        f"cap={cap} even-spaced pre-max-child…",
        flush=True,
    )
    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    df = con.execute(
        f"""
        WITH caps AS (
          SELECT
            account_id,
            CAST(max_created_at AS TIMESTAMPTZ) AS max_created_at
          FROM read_parquet('{tmp}')
        ),
        ranked AS (
          SELECT
            CAST(t.tweet_id AS VARCHAR) AS tweet_id,
            CAST(t.account_id AS VARCHAR) AS account_id,
            t.full_text,
            t.created_at,
            ROW_NUMBER() OVER (
              PARTITION BY CAST(t.account_id AS VARCHAR)
              ORDER BY t.created_at ASC
            ) AS rn,
            COUNT(*) OVER (
              PARTITION BY CAST(t.account_id AS VARCHAR)
            ) AS n
          FROM read_parquet('{path}') t
          JOIN caps c
            ON CAST(t.account_id AS VARCHAR) = c.account_id
          WHERE t.full_text IS NOT NULL
            AND t.full_text NOT LIKE 'RT @%'
            AND CAST(t.created_at AS TIMESTAMPTZ) < c.max_created_at
        )
        SELECT tweet_id, account_id, full_text, created_at
        FROM ranked
        WHERE n <= {cap}
           OR rn = 1
           OR rn = n
           OR (
             {cap} > 2
             AND ((rn - 1) % GREATEST(
               1,
               CAST(CEIL((n - 1) * 1.0 / ({cap} - 1)) AS BIGINT)
             )) = 0
           )
        """
    ).fetchdf()
    con.close()
    try:
        tmp.unlink()
    except OSError:
        pass
    if df.empty:
        print("  parent-author corpus: 0 tweets", flush=True)
        return empty
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["account_id"] = df["account_id"].astype(str)
    df["tweet_id"] = df["tweet_id"].astype(str)
    print(
        f"  parent-author corpus: {len(df):,} tweets / "
        f"{df['account_id'].nunique():,} authors "
        f"(mean {len(df) / max(df['account_id'].nunique(), 1):.1f}/author)",
        flush=True,
    )
    return df


def _attach_parent_author_emb_affinity(
    out: pd.DataFrame,
    *,
    per_account_cap: int,
    batch_size: int,
) -> pd.DataFrame:
    """Cosine / L2 of child MiniLM vs mean of parent author's pre-T corpus."""
    emb_cols = [c for c in out.columns if c.startswith("emb_")]
    for prefix in ("reply", "quote"):
        out[f"{prefix}_parent_author_emb_cos"] = 0.0
        out[f"{prefix}_parent_author_emb_l2"] = 0.0
        out[f"{prefix}_parent_author_emb_log_n"] = 0.0
        out[f"{prefix}_parent_author_emb_has"] = np.int8(0)

    if not emb_cols:
        print("Parent-author emb affinity: no emb_* on dataset; left at 0", flush=True)
        return out
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))

    max_rows: list[dict] = []
    for prefix, aid_col in (
        ("reply", "reply_parent_account_id"),
        ("quote", "quote_parent_account_id"),
    ):
        if aid_col not in out.columns:
            continue
        sub = out.loc[out[aid_col].notna(), ["created_at", aid_col]].copy()
        if sub.empty:
            continue
        sub["account_id"] = sub[aid_col].astype(str)
        sub = sub[sub["account_id"].ne("") & ~sub["account_id"].isin({"None", "nan", "<NA>"})]
        # Exclude self-replies: corpus mean would equal own-history affinity.
        ego = out.loc[sub.index, "account_id"].astype(str)
        sub = sub.loc[sub["account_id"].to_numpy() != ego.to_numpy()]
        if sub.empty:
            continue
        g = sub.groupby("account_id", sort=False)["created_at"].max().reset_index()
        g = g.rename(columns={"created_at": "max_created_at"})
        max_rows.append(g)
    if not max_rows:
        print("Parent-author emb affinity: no external parent authors", flush=True)
        return out

    parent_max = (
        pd.concat(max_rows, ignore_index=True)
        .groupby("account_id", sort=False)["max_created_at"]
        .max()
        .reset_index()
    )
    corpus = load_parent_author_corpus_tweets(parent_max, per_account_cap=per_account_cap)
    if corpus.empty:
        return out

    cache = ensure_embed_cache(corpus["tweet_id"], corpus["full_text"], batch_size=batch_size)
    cache_emb = cache.set_index("tweet_id")[emb_cols]
    corpus = corpus.merge(
        cache_emb.reset_index(),
        on="tweet_id",
        how="left",
    )
    if corpus[emb_cols[0]].isna().any():
        n_miss = int(corpus[emb_cols[0]].isna().sum())
        raise RuntimeError(f"Missing parent-author corpus embeddings ({n_miss})")

    # Per parent: sorted times + prefix-sum embeddings for O(log n) pre-T means.
    packs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for aid, g in corpus.groupby("account_id", sort=False):
        g = g.sort_values("created_at", kind="mergesort")
        times = g["created_at"].to_numpy(dtype="datetime64[ns]")
        vecs = g[emb_cols].to_numpy(dtype=np.float64)
        packs[str(aid)] = (times, np.cumsum(vecs, axis=0))

    child_emb = out[emb_cols].to_numpy(dtype=np.float64)
    child_t = pd.to_datetime(out["created_at"], utc=True).to_numpy(dtype="datetime64[ns]")
    n = len(out)

    for prefix, aid_col in (
        ("reply", "reply_parent_account_id"),
        ("quote", "quote_parent_account_id"),
    ):
        if aid_col not in out.columns:
            continue
        cos = np.zeros(n, dtype=np.float64)
        l2 = np.zeros(n, dtype=np.float64)
        log_n = np.zeros(n, dtype=np.float64)
        has = np.zeros(n, dtype=np.int8)
        aids = out[aid_col]
        ego = out["account_id"].astype(str).to_numpy()
        for i in range(n):
            raw = aids.iloc[i]
            if pd.isna(raw):
                continue
            pid = str(raw)
            if not pid or pid in {"None", "nan", "<NA>"} or pid == ego[i]:
                continue
            pack = packs.get(pid)
            if pack is None:
                continue
            times, cum = pack
            n_prior = int(np.searchsorted(times, child_t[i], side="left"))
            if n_prior <= 0:
                continue
            mean = cum[n_prior - 1] / n_prior
            v = child_emb[i]
            mean_n = float(np.linalg.norm(mean))
            v_n = float(np.linalg.norm(v))
            if mean_n > 1e-12 and v_n > 1e-12:
                cos[i] = float(np.dot(mean, v) / (mean_n * v_n))
            l2[i] = float(np.linalg.norm(v - mean))
            log_n[i] = float(np.log1p(n_prior))
            has[i] = 1
        out[f"{prefix}_parent_author_emb_cos"] = cos
        out[f"{prefix}_parent_author_emb_l2"] = l2
        out[f"{prefix}_parent_author_emb_log_n"] = log_n
        out[f"{prefix}_parent_author_emb_has"] = has
        among = has.astype(bool)
        print(
            f"  {prefix} parent-author emb: has={has.mean():.1%} rows; "
            f"among-has mean_cos={(cos[among].mean() if among.any() else 0):.3f} "
            f"mean_log_n={(log_n[among].mean() if among.any() else 0):.3f}",
            flush=True,
        )
    return out


def attach_reply_dyad_features(
    df: pd.DataFrame,
    *,
    force: bool = False,
    parent_author_emb_cap: int = 200,
    skip_parent_author_emb: bool = False,
    embed_batch_size: int = 256,
) -> pd.DataFrame:
    """Parent-author size + reciprocal reply/quote dyad + optional corpus MiniLM.

    Does not use dump fav/RT or current follower counts. Parent author ids come
    from dump / X parent cache; dyad events are reply/quote edges only (mentions
    omitted — overlap inbound_mentions and drown the pair signal).
    """
    out = resolve_parent_account_ids(df)
    drop_cols = [c for c in dyad_feature_columns() if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    partner_series = pd.concat(
        [
            out["reply_parent_account_id"]
            if "reply_parent_account_id" in out.columns
            else pd.Series(dtype=object),
            out["quote_parent_account_id"]
            if "quote_parent_account_id" in out.columns
            else pd.Series(dtype=object),
        ],
        ignore_index=True,
    ).dropna()
    partner_series = partner_series.astype(str)
    partner_series = partner_series[partner_series.ne("")]

    size_maps = (
        load_parent_author_size_maps(partner_series)
        if len(partner_series)
        else {"author_times": {}, "reply_to_user": {}}
    )
    reply_to_user = size_maps.get("reply_to_user", {})
    author_times = size_maps.get("author_times", {})

    dyad_events = load_dyad_events(out["account_id"], partner_series, force=force)
    inbound_map, outbound_map = _dyad_time_maps(dyad_events)

    n = len(out)
    child_t = pd.to_datetime(out["created_at"], utc=True)
    ego_ids = out["account_id"].astype(str).to_numpy()

    for prefix in ("reply", "quote"):
        aid_col = f"{prefix}_parent_account_id"
        known = np.zeros(n, dtype=np.int8)
        hist_n = np.zeros(n, dtype=np.float64)
        inbound = np.zeros(n, dtype=np.float64)
        inbound_per = np.zeros(n, dtype=np.float64)
        dyad_in = np.zeros(n, dtype=np.float64)
        dyad_out = np.zeros(n, dtype=np.float64)
        has_prior = np.zeros(n, dtype=np.int8)
        bidir = np.zeros(n, dtype=np.int8)

        if aid_col not in out.columns:
            out[f"{prefix}_parent_author_known"] = known
            out[f"{prefix}_parent_author_hist_n"] = hist_n
            out[f"{prefix}_parent_author_inbound"] = inbound
            out[f"{prefix}_parent_author_inbound_per"] = inbound_per
            out[f"{prefix}_dyad_inbound"] = dyad_in
            out[f"{prefix}_dyad_outbound"] = dyad_out
            out[f"{prefix}_dyad_has_prior"] = has_prior
            out[f"{prefix}_dyad_bidir"] = bidir
            continue

        aids = out[aid_col]
        for i in range(n):
            raw = aids.iloc[i]
            if pd.isna(raw):
                continue
            pid = str(raw)
            if not pid or pid in {"None", "nan", "<NA>"}:
                continue
            ego = ego_ids[i]
            if pid == ego:
                # Self-reply / self-quote: size is own hist (redundant); dyad undefined.
                known[i] = 1
                continue
            known[i] = 1
            cutoff = child_t.iloc[i]
            dump_times = author_times.get(pid)
            n_prior = _count_before(dump_times, cutoff)
            n_in = _count_before(reply_to_user.get(pid), cutoff)
            hist_n[i] = float(np.log1p(n_prior))
            inbound[i] = float(np.log1p(n_in))
            inbound_per[i] = float(np.log1p(n_in / max(n_prior, 1)))

            key = (ego, pid)
            c_in = _count_before(inbound_map.get(key), cutoff)
            c_out = _count_before(outbound_map.get(key), cutoff)
            dyad_in[i] = float(np.log1p(c_in))
            dyad_out[i] = float(np.log1p(c_out))
            has_prior[i] = int(c_in > 0 or c_out > 0)
            bidir[i] = int(c_in > 0 and c_out > 0)

        out[f"{prefix}_parent_author_known"] = known
        out[f"{prefix}_parent_author_hist_n"] = hist_n
        out[f"{prefix}_parent_author_inbound"] = inbound
        out[f"{prefix}_parent_author_inbound_per"] = inbound_per
        out[f"{prefix}_dyad_inbound"] = dyad_in
        out[f"{prefix}_dyad_outbound"] = dyad_out
        out[f"{prefix}_dyad_has_prior"] = has_prior
        out[f"{prefix}_dyad_bidir"] = bidir

        among = known.astype(bool)
        n_among = int(among.sum())
        print(
            f"  {prefix} dyad: parent_author known {known.mean():.1%} rows; "
            f"among known n={n_among:,} prior_dyad="
            f"{(has_prior[among].mean() if n_among else 0):.1%} "
            f"bidir={(bidir[among].mean() if n_among else 0):.1%} "
            f"mean_hist_n={(hist_n[among].mean() if n_among else 0):.3f}",
            flush=True,
        )

    if skip_parent_author_emb:
        for prefix in ("reply", "quote"):
            out[f"{prefix}_parent_author_emb_cos"] = 0.0
            out[f"{prefix}_parent_author_emb_l2"] = 0.0
            out[f"{prefix}_parent_author_emb_log_n"] = 0.0
            out[f"{prefix}_parent_author_emb_has"] = np.int8(0)
        print("Parent-author emb affinity skipped (--skip-parent-author-emb)", flush=True)
    else:
        out = _attach_parent_author_emb_affinity(
            out,
            per_account_cap=parent_author_emb_cap,
            batch_size=embed_batch_size,
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
        help="Only add reply/quote parent features onto an existing dataset.parquet "
        "(resumes x_parent_cache after rate/credits stops; default --x-parents)",
    )
    p.add_argument(
        "--x-parents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch reply/quote parents missing from the CA dump via X API "
        "(resumable x_parent_cache.parquet; default on after v39c). "
        "Use --no-x-parents for dump-only ablation",
    )
    p.add_argument(
        "--force-x-parents",
        action="store_true",
        help="With --x-parents, refetch even when x_parent_cache.parquet has the tweet_id",
    )
    p.add_argument(
        "--urls-only",
        action="store_true",
        help="Only add link/domain features onto an existing dataset.parquet",
    )
    p.add_argument(
        "--media-only",
        action="store_true",
        help="Refetch CA tweet_media (with type breakdown) / tweet_urls onto dataset.parquet",
    )
    p.add_argument(
        "--mentions-only",
        action="store_true",
        help="Add timestamped inbound @-mention features onto an existing dataset.parquet",
    )
    p.add_argument(
        "--interlocutor-only",
        action="store_true",
        help="Build/refresh interlocutor_cache.parquet (inbound reply/quote/mention texts) "
        "and embed those tweets for train --interlocutor-emb-affinity",
    )
    p.add_argument(
        "--force-interlocutor",
        action="store_true",
        help="Rescan dump interlocutor events even when interlocutor_cache.parquet has the account_id",
    )
    p.add_argument(
        "--interlocutor-include-mentions",
        action="store_true",
        help="Include inbound @-mention tweets in interlocutor_cache / affinity pool "
        "(off by default; mentions drown reply/quote signal and overlap inbound_mentions)",
    )
    p.add_argument(
        "--provenance-only",
        action="store_true",
        help="Attach archive_upload_id / is_stream / upload-end features onto an existing dataset.parquet",
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
    p.add_argument(
        "--force-uploads",
        action="store_true",
        help="Refetch archive_upload even when archive_upload_cache.parquet exists",
    )
    p.add_argument(
        "--force-media",
        action="store_true",
        help="Refetch tweet_media even when media_cache.parquet has the tweet_id",
    )
    p.add_argument(
        "--dyad-only",
        action="store_true",
        help="Add reply/quote dyad + parent-author size features onto an existing "
        "dataset.parquet (train with --reply-dyad)",
    )
    p.add_argument(
        "--force-dyad",
        action="store_true",
        help="With --dyad-only (or full prepare), rescan dump dyad events even when "
        "dyad_cache.parquet has the ego account_id",
    )
    p.add_argument(
        "--parent-author-emb-cap",
        type=int,
        default=200,
        help="Max pre-max-child tweets per parent author for MiniLM corpus affinity "
        "(default 200; used by --dyad-only / full prepare)",
    )
    p.add_argument(
        "--skip-parent-author-emb",
        action="store_true",
        help="Skip parent-author MiniLM corpus affinity when building dyad features",
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

    if args.dyad_only:
        if not DATASET_PATH.exists():
            raise SystemExit(f"Missing {DATASET_PATH} for --dyad-only")
        dataset = pd.read_parquet(DATASET_PATH)
        dataset["tweet_id"] = dataset["tweet_id"].astype(str)
        dataset["created_at"] = pd.to_datetime(dataset["created_at"], utc=True)
        dataset = backfill_ref_ids_from_dump(dataset)
        dataset = attach_reply_dyad_features(
            dataset,
            force=args.force_dyad,
            parent_author_emb_cap=args.parent_author_emb_cap,
            skip_parent_author_emb=args.skip_parent_author_emb,
            embed_batch_size=args.embed_batch_size,
        )
        dataset.to_parquet(DATASET_PATH, index=False)
        print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
        print(f"Done in {time.time() - t0:.1f}s")
        return

    if args.mentions_only:
        if not DATASET_PATH.exists():
            raise SystemExit(f"Missing {DATASET_PATH} for --mentions-only")
        dataset = pd.read_parquet(DATASET_PATH)
        dataset = attach_inbound_mentions(dataset, force=args.force_mentions)
        dataset.to_parquet(DATASET_PATH, index=False)
        print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
        print(f"Done in {time.time() - t0:.1f}s")
        return

    if args.interlocutor_only:
        if not DATASET_PATH.exists():
            raise SystemExit(f"Missing {DATASET_PATH} for --interlocutor-only")
        dataset = pd.read_parquet(DATASET_PATH)
        events = load_interlocutor_events(
            dataset["account_id"],
            force=args.force_interlocutor,
            include_mentions=args.interlocutor_include_mentions,
        )
        if len(events):
            ensure_interlocutor_embeddings(events, batch_size=args.embed_batch_size)
        print(f"Done in {time.time() - t0:.1f}s")
        return

    if args.provenance_only:
        if not DATASET_PATH.exists():
            raise SystemExit(f"Missing {DATASET_PATH} for --provenance-only")
        dataset = pd.read_parquet(DATASET_PATH)
        dataset = attach_provenance(dataset, force_uploads=args.force_uploads)
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
        dataset = attach_media_flags(dataset, force=args.force_media)
        dataset = refresh_media_derived(dataset)
        dataset.to_parquet(DATASET_PATH, index=False)
        print(
            f"Media refresh: has_media {before['has_media']:.1%} → {dataset['has_media'].mean():.1%}, "
            f"has_ca_url {before['has_ca_url']:.1%} → {dataset['has_ca_url'].mean():.1%}; "
            f"photo={dataset['has_photo'].mean():.1%} video={dataset['has_video'].mean():.1%} "
            f"gif={dataset['has_animated_gif'].mean():.1%}"
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
        before_cov = {
            "reply": float(dataset["has_reply_parent"].mean())
            if "has_reply_parent" in dataset.columns
            else None,
            "quote": float(dataset["has_quote_parent"].mean())
            if "has_quote_parent" in dataset.columns
            else None,
        }
        drop_cols = [
            c
            for c in dataset.columns
            if (
                c.startswith(("has_reply_parent", "has_quote_parent", "reply_parent_", "quote_parent_"))
                and not c.startswith(("reply_parent_author_", "quote_parent_author_"))
            )
        ]
        if drop_cols:
            dataset = dataset.drop(columns=drop_cols)
        dataset = backfill_ref_ids_from_dump(dataset)
        if not any(c.startswith("emb_") for c in dataset.columns):
            raise SystemExit("Dataset has no emb_* columns; run full prepare first")
        dataset = attach_parent_context(
            dataset,
            batch_size=args.embed_batch_size,
            use_x=args.x_parents,
            force_x=args.force_x_parents,
        )
        if before_cov["reply"] is not None:
            print(
                f"Parent coverage: reply {before_cov['reply']:.1%} → "
                f"{dataset['has_reply_parent'].mean():.1%}, "
                f"quote {before_cov['quote']:.1%} → {dataset['has_quote_parent'].mean():.1%}",
                flush=True,
            )
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
    media_fill_cols = (
        ("has_media", 0),
        ("n_media", 0),
        ("has_ca_url", 0),
        ("n_ca_urls", 0),
        ("has_photo", 0),
        ("n_photo", 0),
        ("has_video", 0),
        ("n_video", 0),
        ("has_animated_gif", 0),
        ("n_animated_gif", 0),
        ("has_media_other", 0),
        ("n_media_other", 0),
    )
    if args.skip_media:
        reuse_cols = ["tweet_id", *[c for c, _ in media_fill_cols]]
        if DATASET_PATH.exists():
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(DATASET_PATH).schema_arrow.names)
            cols = [c for c in reuse_cols if c in available]
            prev = pd.read_parquet(DATASET_PATH, columns=cols)
            prev["tweet_id"] = prev["tweet_id"].astype(str)
            sampled = sampled.merge(prev, on="tweet_id", how="left")
            for c, fill in media_fill_cols:
                if c not in sampled.columns:
                    sampled[c] = fill
                sampled[c] = sampled[c].fillna(fill).astype(np.int32)
            print(
                f"Reused media flags from {DATASET_PATH.name}: "
                f"{sampled['has_media'].mean():.1%} media, {sampled['has_ca_url'].mean():.1%} ca_url"
            )
        else:
            print("No prior dataset for media reuse; media flags default to 0")
            for c, fill in media_fill_cols:
                sampled[c] = fill
    else:
        sampled = attach_media_flags(sampled, force=args.force_media)
    sampled = backfill_reply_user_from_dump(sampled)
    graph = load_author_graph(sampled["account_id"])
    featured = build_features(sampled, min_history=args.min_history, graph=graph)
    del sampled, graph

    dataset = attach_embeddings(featured, batch_size=args.embed_batch_size)
    del featured
    dataset = attach_parent_context(
        dataset,
        batch_size=args.embed_batch_size,
        use_x=args.x_parents,
        force_x=args.force_x_parents,
    )
    dataset = attach_reply_dyad_features(
        dataset,
        force=args.force_dyad,
        parent_author_emb_cap=args.parent_author_emb_cap,
        skip_parent_author_emb=args.skip_parent_author_emb,
        embed_batch_size=args.embed_batch_size,
    )
    dataset = attach_url_domains(dataset, force=args.force_urls)
    dataset = attach_upload_end_features(dataset, force=args.force_uploads)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)
    print(f"Wrote {DATASET_PATH} ({len(dataset):,} rows, {dataset.shape[1]} cols)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
