"""Comparative vision smoke: Gemini structured tags vs SigLIP embeddings.

Samples ≤N media tweets that appear in the CA dump, downloads first photo,
labels with gemini-3.6-flash (opt-in paid API), embeds with google/siglip-base
(Apache-2.0, local CPU/GPU), and reports train-fold-style correlations vs
exposure rate. Caches under ~/.cache/will-it-bang/.

    uv run vision_smoke.py --limit 500
    uv run vision_smoke.py --limit 500 --skip-gemini   # SigLIP only
    uv run vision_smoke.py --limit 500 --skip-siglip   # Gemini only

Requires GEMINI_API_KEY in the environment (or .env) unless --skip-gemini.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from scipy.stats import spearmanr

CACHE_DIR = Path(os.path.expanduser("~/.cache/will-it-bang"))
RAW_PARQUET = CACHE_DIR / "enriched_tweets.parquet"
MEDIA_URL_CACHE = CACHE_DIR / "media_url_cache.parquet"
GEMINI_CACHE = CACHE_DIR / "vision_gemini_cache.parquet"
SIGLIP_CACHE = CACHE_DIR / "vision_siglip_cache.parquet"
SMOKE_SAMPLE = CACHE_DIR / "vision_smoke_sample.parquet"
SMOKE_REPORT = Path(__file__).resolve().parent / "logs" / "vision_smoke.md"

CA_API = "https://fabxmporizzqflnftavs.supabase.co/rest/v1"
CA_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0."
    "UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
)

GEMINI_MODEL = "gemini-3.6-flash"
SIGLIP_MODEL = "google/siglip-base-patch16-224"  # Apache-2.0
RT_RE = re.compile(r"^RT @", re.IGNORECASE)

GEMINI_BOOL_COLS = [
    "has_text_in_image",
    "is_screenshot",
    "is_meme",
    "is_chart",
    "has_face",
    "is_photo_realworld",
]
GEMINI_INT_COLS = ["humor_0_10", "controversy_0_10"]
GEMINI_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "has_text_in_image": {"type": "boolean"},
        "is_screenshot": {"type": "boolean"},
        "is_meme": {"type": "boolean"},
        "is_chart": {"type": "boolean"},
        "has_face": {"type": "boolean"},
        "is_photo_realworld": {"type": "boolean"},
        "humor_0_10": {"type": "integer"},
        "controversy_0_10": {"type": "integer"},
    },
    "required": [
        "caption",
        "has_text_in_image",
        "is_screenshot",
        "is_meme",
        "is_chart",
        "has_face",
        "is_photo_realworld",
        "humor_0_10",
        "controversy_0_10",
    ],
}


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _ca_headers() -> dict[str, str]:
    return {
        "apikey": CA_ANON_KEY,
        "Authorization": f"Bearer {CA_ANON_KEY}",
    }


def _read_cache(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or ["tweet_id"])
    df = pd.read_parquet(path)
    if "tweet_id" in df.columns:
        df["tweet_id"] = df["tweet_id"].astype(str)
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = df.drop_duplicates("tweet_id", keep="last")
    out.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(out):,} rows)", flush=True)


def fetch_photo_media_urls(
    n: int,
    *,
    page_size: int = 100,
    start_offset: int = 0,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Page CA tweet_media photos until n new unique tweet_ids (first URL each)."""
    session = requests.Session()
    rows: list[dict] = []
    seen: set[str] = set(exclude or ())
    start_seen = len(seen)
    start = int(start_offset)
    empty_pages = 0
    while len(seen) - start_seen < n:
        end = start + page_size - 1
        headers = {**_ca_headers(), "Range": f"{start}-{end}"}
        resp = session.get(
            f"{CA_API}/tweet_media",
            headers=headers,
            params={
                "select": "tweet_id,media_url,media_type",
                "media_type": "eq.photo",
                "order": "tweet_id.asc",
            },
            timeout=60,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            empty_pages += 1
            if empty_pages >= 3:
                break
            start += page_size
            continue
        empty_pages = 0
        for row in batch:
            tid = str(row["tweet_id"])
            url = (row.get("media_url") or "").strip()
            if not tid or not url or tid in seen:
                continue
            seen.add(tid)
            rows.append(
                {
                    "tweet_id": tid,
                    "media_url": url,
                    "media_type": str(row.get("media_type") or "photo"),
                }
            )
            if len(seen) - start_seen >= n:
                break
        start += page_size
        if start % 2000 == 0:
            print(
                f"  tweet_media offset={start:,} new={len(seen) - start_seen:,}/{n:,}",
                flush=True,
            )
    return pd.DataFrame(rows)


def join_dump_labels(media: pd.DataFrame) -> pd.DataFrame:
    """Attach dump engagement / text / time; drop RTs and missing dump rows."""
    if not RAW_PARQUET.exists():
        raise SystemExit(f"Missing {RAW_PARQUET}; run prepare.py once to download the dump")
    ids = media["tweet_id"].astype(str).tolist()
    con = duckdb.connect()
    con.register("media_ids", media[["tweet_id", "media_url", "media_type"]])
    dump = con.execute(
        f"""
        SELECT
          CAST(m.tweet_id AS VARCHAR) AS tweet_id,
          m.media_url,
          m.media_type,
          CAST(t.account_id AS VARCHAR) AS account_id,
          t.created_at,
          t.full_text,
          COALESCE(t.favorite_count, 0) AS favorite_count,
          COALESCE(t.retweet_count, 0) AS retweet_count,
          t.reply_to_tweet_id
        FROM media_ids m
        INNER JOIN read_parquet('{RAW_PARQUET}') t
          ON CAST(t.tweet_id AS VARCHAR) = CAST(m.tweet_id AS VARCHAR)
        """
    ).fetchdf()
    if dump.empty:
        print(f"Dump join: {len(ids):,} media ids → 0 labeled", flush=True)
        return dump
    dump["tweet_id"] = dump["tweet_id"].astype(str)
    dump["created_at"] = pd.to_datetime(dump["created_at"], utc=True, errors="coerce")
    dump = dump.loc[dump["created_at"].notna()].copy()
    text = dump["full_text"].fillna("").astype(str)
    dump = dump.loc[~text.str.match(RT_RE)].copy()
    dump["engagement"] = (
        dump["favorite_count"].fillna(0).astype(np.int64)
        + dump["retweet_count"].fillna(0).astype(np.int64)
    )
    dump["is_reply"] = dump["reply_to_tweet_id"].notna().astype(np.int8)
    dump_time = dump["created_at"].max()
    age = (dump_time - dump["created_at"]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    age = np.clip(age, 0.0, None)
    eff = np.minimum(age, 30.0)
    dump["exposure_age_days"] = age
    dump["rate"] = np.log1p(dump["engagement"].to_numpy(dtype=np.float64) / (eff + 1.0))
    dump["log_engagement"] = np.log1p(dump["engagement"].to_numpy(dtype=np.float64))
    print(
        f"Dump join: {len(ids):,} media ids → {len(dump):,} labeled "
        f"(dump_time={dump_time.date()}, hit={len(dump)/max(len(ids),1):.1%})",
        flush=True,
    )
    return dump.reset_index(drop=True)


def fetch_media_for_ids(tweet_ids: list[str], *, batch_size: int = 80) -> pd.DataFrame:
    """Fetch preferred media URL per tweet_id from CA tweet_media (photos first)."""
    session = requests.Session()
    best: dict[str, dict] = {}
    for start in range(0, len(tweet_ids), batch_size):
        batch = tweet_ids[start : start + batch_size]
        joined = ",".join(f'"{tid}"' for tid in batch)
        resp = session.get(
            f"{CA_API}/tweet_media",
            headers=_ca_headers(),
            params={
                "select": "tweet_id,media_url,media_type",
                "tweet_id": f"in.({joined})",
            },
            timeout=60,
        )
        resp.raise_for_status()
        for row in resp.json():
            tid = str(row["tweet_id"])
            url = (row.get("media_url") or "").strip()
            mtype = str(row.get("media_type") or "unknown").lower()
            if not tid or not url:
                continue
            cand = {"tweet_id": tid, "media_url": url, "media_type": mtype}
            prev = best.get(tid)
            if prev is None or (prev["media_type"] != "photo" and mtype == "photo"):
                best[tid] = cand
        if start == 0 or (start // batch_size) % 20 == 0:
            print(
                f"  media-by-id {min(start + batch_size, len(tweet_ids)):,}/"
                f"{len(tweet_ids):,} hits={len(best):,}",
                flush=True,
            )
    return pd.DataFrame(list(best.values())) if best else pd.DataFrame(
        columns=["tweet_id", "media_url", "media_type"]
    )


def sample_dump_candidates(n: int, *, seed: int = 0) -> pd.DataFrame:
    """Sample non-RT dump rows as media-lookup candidates."""
    if not RAW_PARQUET.exists():
        raise SystemExit(f"Missing {RAW_PARQUET}; run prepare.py once to download the dump")
    con = duckdb.connect()
    # Hash-based sample is stable-ish and avoids ORDER BY RANDOM over 8M rows.
    df = con.execute(
        f"""
        SELECT
          CAST(tweet_id AS VARCHAR) AS tweet_id,
          CAST(account_id AS VARCHAR) AS account_id,
          created_at,
          full_text,
          COALESCE(favorite_count, 0) AS favorite_count,
          COALESCE(retweet_count, 0) AS retweet_count,
          reply_to_tweet_id
        FROM read_parquet('{RAW_PARQUET}')
        WHERE full_text IS NOT NULL
          AND NOT starts_with(full_text, 'RT @')
          AND (hash(tweet_id) % 1000) < 40
        LIMIT {int(n)}
        """
    ).fetchdf()
    if len(df) < n:
        df = con.execute(
            f"""
            SELECT
              CAST(tweet_id AS VARCHAR) AS tweet_id,
              CAST(account_id AS VARCHAR) AS account_id,
              created_at,
              full_text,
              COALESCE(favorite_count, 0) AS favorite_count,
              COALESCE(retweet_count, 0) AS retweet_count,
              reply_to_tweet_id
            FROM read_parquet('{RAW_PARQUET}')
            WHERE full_text IS NOT NULL
              AND NOT starts_with(full_text, 'RT @')
            USING SAMPLE {int(n)}
            """
        ).fetchdf()
    df["tweet_id"] = df["tweet_id"].astype(str)
    if seed and len(df):
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"Dump candidates: {len(df):,}", flush=True)
    return df


def build_sample(limit: int, *, seed: int = 0) -> pd.DataFrame:
    """Build a leakage-safe media smoke sample with dump labels."""
    cache = _read_cache(
        MEDIA_URL_CACHE,
        columns=["tweet_id", "media_url", "media_type"],
    )
    labeled = join_dump_labels(cache) if len(cache) else pd.DataFrame()

    cand_n = max(limit * 40, 8000)
    while len(labeled) < limit:
        print(
            f"Need {limit:,} labeled media; have {len(labeled):,}. "
            f"Sampling dump candidates + CA media lookup…",
            flush=True,
        )
        cands = sample_dump_candidates(cand_n, seed=seed + len(labeled))
        have = set(cache["tweet_id"].astype(str)) if len(cache) else set()
        miss = [tid for tid in cands["tweet_id"].astype(str).tolist() if tid not in have]
        if not miss:
            cand_n = int(cand_n * 1.5)
            if cand_n > 200_000:
                break
            continue
        fresh = fetch_media_for_ids(miss)
        if fresh.empty:
            cand_n = int(cand_n * 1.5)
            if cand_n > 200_000:
                break
            continue
        cache = (
            fresh
            if cache.empty
            else pd.concat([cache, fresh], ignore_index=True).drop_duplicates(
                "tweet_id", keep="last"
            )
        )
        _write_cache(cache, MEDIA_URL_CACHE)
        labeled = join_dump_labels(cache)
        if len(labeled) < limit:
            cand_n = int(cand_n * 1.5)
        if cand_n > 200_000 and len(labeled) < limit:
            raise SystemExit(
                f"Could not find {limit} dump-joined media tweets "
                f"(labeled={len(labeled)}, url_cache={len(cache)})"
            )

    # Prefer non-replies (main under-prediction bucket) but keep some replies.
    nr = labeled.loc[labeled["is_reply"] == 0]
    rp = labeled.loc[labeled["is_reply"] == 1]
    rng = np.random.default_rng(seed)
    n_nr = min(len(nr), int(round(limit * 0.7)))
    n_rp = min(len(rp), limit - n_nr)
    if n_nr + n_rp < limit:
        n_nr = min(len(nr), limit - n_rp)
    pick_nr = nr.iloc[rng.choice(len(nr), size=n_nr, replace=False)] if n_nr else nr.iloc[:0]
    pick_rp = rp.iloc[rng.choice(len(rp), size=n_rp, replace=False)] if n_rp else rp.iloc[:0]
    sample = pd.concat([pick_nr, pick_rp], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    _write_cache(sample, SMOKE_SAMPLE)
    print(
        f"Smoke sample: {len(sample):,} "
        f"(NR={int((sample['is_reply']==0).sum())}, RP={int((sample['is_reply']==1).sum())})",
        flush=True,
    )
    return sample


def download_image(url: str, *, timeout: int = 30) -> tuple[bytes | None, str]:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "will-it-bang-vision-smoke/0.1"},
        )
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"
        ctype = (resp.headers.get("content-type") or "image/jpeg").split(";")[0].strip()
        if not ctype.startswith("image/"):
            ctype = "image/jpeg"
        if len(resp.content) < 32:
            return None, "empty"
        return resp.content, ctype
    except requests.RequestException as exc:
        return None, f"err:{type(exc).__name__}"


def label_gemini(sample: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    from google import genai
    from google.genai import types

    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing (set in .env or environment)")

    cache = _read_cache(GEMINI_CACHE)
    have = set(cache.loc[cache.get("ok", 0) == 1, "tweet_id"]) if len(cache) else set()
    todo = sample if force else sample.loc[~sample["tweet_id"].isin(have)]
    print(f"Gemini: {len(sample) - len(todo):,} cached, {len(todo):,} to label", flush=True)

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=GEMINI_SCHEMA,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    )
    prompt = (
        "Describe this social-media image with the JSON schema. "
        "Use descriptive, time-invariant tags only. "
        "Do not judge popularity, fame, influence, or whether it will go viral."
    )

    rows: list[dict] = []
    prompt_tok = 0
    out_tok = 0
    t0 = time.time()
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        blob, ctype = download_image(str(row.media_url))
        if blob is None:
            rows.append(
                {
                    "tweet_id": str(row.tweet_id),
                    "ok": 0,
                    "error": ctype,
                    "model": GEMINI_MODEL,
                    "caption": None,
                    **{c: None for c in GEMINI_BOOL_COLS + GEMINI_INT_COLS},
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                }
            )
        else:
            try:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=blob, mime_type=ctype),
                        prompt,
                    ],
                    config=cfg,
                )
                obj = json.loads(resp.text or "{}")
                um = resp.usage_metadata
                pt = int(getattr(um, "prompt_token_count", 0) or 0)
                ot = int(getattr(um, "candidates_token_count", 0) or 0) + int(
                    getattr(um, "thoughts_token_count", 0) or 0
                )
                prompt_tok += pt
                out_tok += ot
                rows.append(
                    {
                        "tweet_id": str(row.tweet_id),
                        "ok": 1,
                        "error": None,
                        "model": GEMINI_MODEL,
                        "caption": str(obj.get("caption") or "")[:500],
                        **{c: bool(obj.get(c)) for c in GEMINI_BOOL_COLS},
                        **{
                            c: int(max(0, min(10, int(obj.get(c, 0) or 0))))
                            for c in GEMINI_INT_COLS
                        },
                        "prompt_tokens": pt,
                        "output_tokens": ot,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — cache failure, continue smoke
                rows.append(
                    {
                        "tweet_id": str(row.tweet_id),
                        "ok": 0,
                        "error": f"{type(exc).__name__}:{exc}"[:200],
                        "model": GEMINI_MODEL,
                        "caption": None,
                        **{c: None for c in GEMINI_BOOL_COLS + GEMINI_INT_COLS},
                        "prompt_tokens": 0,
                        "output_tokens": 0,
                    }
                )
        if i == 1 or i % 25 == 0 or i == len(todo):
            elapsed = max(time.time() - t0, 1e-6)
            est_cost = prompt_tok / 1e6 * 1.5 + out_tok / 1e6 * 7.5
            print(
                f"  gemini {i:,}/{len(todo):,}  "
                f"{i / elapsed:.2f}/s  tokens in={prompt_tok:,} out={out_tok:,}  "
                f"est_cost=${est_cost:.3f}",
                flush=True,
            )

    if rows:
        fresh = pd.DataFrame(rows)
        cache = fresh if cache.empty else pd.concat([cache, fresh], ignore_index=True)
        _write_cache(cache, GEMINI_CACHE)
    return cache


def embed_siglip(sample: pd.DataFrame, *, force: bool = False, batch_size: int = 16) -> pd.DataFrame:
    from transformers import AutoModel, AutoProcessor

    cache = _read_cache(SIGLIP_CACHE)
    emb_cols = [c for c in cache.columns if re.fullmatch(r"sig_\d+", c)]
    have = set(cache.loc[cache.get("ok", 0) == 1, "tweet_id"]) if len(cache) and emb_cols else set()
    todo = sample if force else sample.loc[~sample["tweet_id"].isin(have)]
    print(f"SigLIP: {len(sample) - len(todo):,} cached, {len(todo):,} to embed", flush=True)
    if todo.empty:
        return cache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {SIGLIP_MODEL} on {device}…", flush=True)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).to(device)
    model.eval()

    rows: list[dict] = []
    t0 = time.time()
    urls = todo["media_url"].astype(str).tolist()
    ids = todo["tweet_id"].astype(str).tolist()
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        batch_urls = urls[start : start + batch_size]
        images: list[Image.Image] = []
        keep_ids: list[str] = []
        fail_rows: list[dict] = []
        for tid, url in zip(batch_ids, batch_urls, strict=True):
            blob, err = download_image(url)
            if blob is None:
                fail_rows.append({"tweet_id": tid, "ok": 0, "error": err, "model": SIGLIP_MODEL})
                continue
            try:
                images.append(Image.open(BytesIO(blob)).convert("RGB"))
                keep_ids.append(tid)
            except Exception as exc:  # noqa: BLE001
                fail_rows.append(
                    {
                        "tweet_id": tid,
                        "ok": 0,
                        "error": f"pil:{type(exc).__name__}",
                        "model": SIGLIP_MODEL,
                    }
                )
        rows.extend(fail_rows)
        if images:
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                feats = model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            arr = feats.detach().cpu().numpy().astype(np.float32)
            for tid, vec in zip(keep_ids, arr, strict=True):
                row = {"tweet_id": tid, "ok": 1, "error": None, "model": SIGLIP_MODEL}
                for j, val in enumerate(vec):
                    row[f"sig_{j}"] = float(val)
                rows.append(row)
        done = min(start + batch_size, len(ids))
        if done == batch_size or done % 64 == 0 or done == len(ids):
            elapsed = max(time.time() - t0, 1e-6)
            print(f"  siglip {done:,}/{len(ids):,}  {done / elapsed:.2f}/s", flush=True)

    fresh = pd.DataFrame(rows)
    cache = fresh if cache.empty else pd.concat([cache, fresh], ignore_index=True)
    _write_cache(cache, SIGLIP_CACHE)
    return cache


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10 or np.nanstd(x[mask]) < 1e-12:
        return float("nan")
    corr, _ = spearmanr(x[mask], y[mask])
    return float(corr)


def _cv_ridge_spearman(
    X: np.ndarray,
    y: np.ndarray,
    *,
    folds: int = 5,
    ridge: float = 1.0,
    seed: int = 0,
) -> float:
    """Out-of-fold ridge probe Spearman (smoke diagnostic; not train TE)."""
    n = len(y)
    if n < 40 or X.ndim != 2 or X.shape[0] != n:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = int(min(folds, max(2, n // 20)))
    oof = np.full(n, np.nan, dtype=np.float64)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te, assume_unique=False)
        if len(tr) < 10 or len(te) < 5:
            continue
        Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-8
        Ztr = (Xtr - mu) / sd
        Zte = (X[te] - mu) / sd
        xtx = Ztr.T @ Ztr + float(ridge) * np.eye(Ztr.shape[1])
        w = np.linalg.solve(xtx, Ztr.T @ ytr)
        oof[te] = Zte @ w
    return _spearman(oof, y)


def evaluate(
    sample: pd.DataFrame,
    gemini: pd.DataFrame | None,
    siglip: pd.DataFrame | None,
) -> dict:
    report: dict = {
        "n_sample": int(len(sample)),
        "n_nonreply": int((sample["is_reply"] == 0).sum()),
        "download_probe": {},
        "gemini": {},
        "siglip": {},
    }

    # Download success on sample URLs.
    ok_dl = 0
    for url in sample["media_url"].astype(str).head(min(50, len(sample))):
        blob, _ = download_image(url)
        ok_dl += int(blob is not None)
    report["download_probe"] = {
        "n": int(min(50, len(sample))),
        "ok": int(ok_dl),
        "rate": float(ok_dl / max(min(50, len(sample)), 1)),
    }

    if gemini is not None and len(gemini):
        g = sample[["tweet_id", "rate", "log_engagement", "is_reply"]].merge(
            gemini, on="tweet_id", how="left"
        )
        ok = g["ok"].fillna(0).astype(int) == 1
        y = g.loc[ok, "rate"].to_numpy(dtype=np.float64)
        y_log = g.loc[ok, "log_engagement"].to_numpy(dtype=np.float64)
        report["gemini"]["n_ok"] = int(ok.sum())
        report["gemini"]["coverage"] = float(ok.mean())
        if "prompt_tokens" in g.columns:
            pt = float(g.loc[ok, "prompt_tokens"].fillna(0).sum())
            ot = float(g.loc[ok, "output_tokens"].fillna(0).sum())
            report["gemini"]["prompt_tokens"] = pt
            report["gemini"]["output_tokens"] = ot
            report["gemini"]["est_cost_usd"] = pt / 1e6 * 1.5 + ot / 1e6 * 7.5
        corrs = {}
        for col in GEMINI_BOOL_COLS + GEMINI_INT_COLS:
            if col not in g.columns:
                continue
            x = pd.to_numeric(g.loc[ok, col], errors="coerce").to_numpy(dtype=np.float64)
            corrs[col] = {
                "rate": _spearman(x, y),
                "log_eng": _spearman(x, y_log),
                "mean": float(np.nanmean(x)) if len(x) else float("nan"),
            }
        report["gemini"]["corr"] = corrs
        # Linear combo of tags: report OOF CV (fair) and in-sample (upper bound).
        if ok.sum() >= 20:
            mats = []
            names = []
            for col in GEMINI_BOOL_COLS + GEMINI_INT_COLS:
                x = pd.to_numeric(g.loc[ok, col], errors="coerce").to_numpy(dtype=np.float64)
                if np.nanstd(x) < 1e-12:
                    continue
                mats.append(x)
                names.append(col)
            if mats:
                X = np.column_stack(mats)
                report["gemini"]["linear_combo_features"] = names
                report["gemini"]["cv_ridge_spearman_rate"] = _cv_ridge_spearman(X, y, ridge=1.0)
                mu = X.mean(axis=0, keepdims=True)
                sd = X.std(axis=0, keepdims=True) + 1e-8
                Z = (X - mu) / sd
                xtx = Z.T @ Z + 1.0 * np.eye(Z.shape[1])
                w = np.linalg.solve(xtx, Z.T @ y)
                report["gemini"]["insample_ridge_spearman_rate"] = _spearman(Z @ w, y)

    if siglip is not None and len(siglip):
        emb_cols = sorted(
            [c for c in siglip.columns if re.fullmatch(r"sig_\d+", c)],
            key=lambda c: int(c.split("_")[1]),
        )
        s = sample[["tweet_id", "rate", "log_engagement", "is_reply"]].merge(
            siglip, on="tweet_id", how="left"
        )
        ok = s["ok"].fillna(0).astype(int) == 1
        y = s.loc[ok, "rate"].to_numpy(dtype=np.float64)
        report["siglip"]["n_ok"] = int(ok.sum())
        report["siglip"]["coverage"] = float(ok.mean())
        report["siglip"]["dims"] = len(emb_cols)
        if ok.sum() >= 20 and emb_cols:
            E = s.loc[ok, emb_cols].to_numpy(dtype=np.float64)
            # First PC of embeddings vs rate (unsupervised content axis).
            E0 = E - E.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(E0, full_matrices=False)
            pc1 = E0 @ vt[0]
            report["siglip"]["pc1_spearman_rate"] = _spearman(pc1, y)
            report["siglip"]["cv_ridge_spearman_rate"] = _cv_ridge_spearman(E, y, ridge=10.0)
            # PCA-8 then CV ridge — closer to how we'd actually train-fit.
            k = int(min(8, E.shape[0] - 1, E.shape[1]))
            _, _, vt8 = np.linalg.svd(E0, full_matrices=False)
            Z8 = E0 @ vt8[:k].T
            report["siglip"]["cv_pca8_ridge_spearman_rate"] = _cv_ridge_spearman(
                Z8, y, ridge=1.0
            )
            dim_corrs = [_spearman(E[:, j], y) for j in range(E.shape[1])]
            report["siglip"]["best_dim_spearman_rate"] = float(
                dim_corrs[int(np.nanargmax(np.abs(dim_corrs)))]
            )
    return report


def write_report(report: dict, path: Path = SMOKE_REPORT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g = report.get("gemini") or {}
    s = report.get("siglip") or {}
    lines = [
        "# Vision smoke (Gemini vs SigLIP)",
        "",
        f"- Sample n={report['n_sample']} (NR={report['n_nonreply']})",
        f"- Download probe (first {report['download_probe']['n']}): "
        f"{report['download_probe']['ok']}/{report['download_probe']['n']} "
        f"({report['download_probe']['rate']:.0%})",
        "",
        "## Gemini (`gemini-3.6-flash`, medium res, thinking=MINIMAL)",
        "",
    ]
    if g:
        lines += [
            f"- ok/coverage: {g.get('n_ok')} / {g.get('coverage', float('nan')):.1%}",
            f"- est_cost_usd (this run tokens): {g.get('est_cost_usd', float('nan')):.3f}",
            f"- **cv ridge spearman(rate): {g.get('cv_ridge_spearman_rate', float('nan')):.4f}**",
            f"- in-sample ridge spearman(rate): {g.get('insample_ridge_spearman_rate', float('nan')):.4f}",
            "",
            "| feature | spearman(rate) | spearman(log_eng) | mean |",
            "|---|---:|---:|---:|",
        ]
        for col, stats in (g.get("corr") or {}).items():
            lines.append(
                f"| {col} | {stats['rate']:.4f} | {stats['log_eng']:.4f} | {stats['mean']:.3f} |"
            )
    else:
        lines.append("- skipped")
    lines += [
        "",
        f"## SigLIP (`{SIGLIP_MODEL}`)",
        "",
    ]
    if s:
        lines += [
            f"- ok/coverage: {s.get('n_ok')} / {s.get('coverage', float('nan')):.1%}",
            f"- dims: {s.get('dims')}",
            f"- pc1 spearman(rate): {s.get('pc1_spearman_rate', float('nan')):.4f}",
            f"- best dim spearman(rate): {s.get('best_dim_spearman_rate', float('nan')):.4f}",
            f"- **cv ridge spearman(rate): {s.get('cv_ridge_spearman_rate', float('nan')):.4f}**",
            f"- cv PCA-8 ridge spearman(rate): {s.get('cv_pca8_ridge_spearman_rate', float('nan')):.4f}",
        ]
    else:
        lines.append("- skipped")
    lines += [
        "",
        "## Read",
        "",
        "- Primary smoke metric is **5-fold CV ridge Spearman vs exposure rate** (not val `wa_*`).",
        "- In-sample probes with dim ≫ n are ignored (overfit).",
        "- Prefer SigLIP if its CV probe ≥ Gemini CV combo; pay for Gemini only if "
        "OCR/structure tags clearly add complementary signal.",
        "- Full-corpus spend (~10k media) only after CV corr clears ~0.10+ with decent coverage.",
        "",
        "```json",
        json.dumps(report, indent=2, default=float),
        "```",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-gemini", action="store_true")
    p.add_argument("--skip-siglip", action="store_true")
    p.add_argument("--force-gemini", action="store_true")
    p.add_argument("--force-siglip", action="store_true")
    p.add_argument("--reuse-sample", action="store_true")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Re-score from existing sample + caches (no Gemini/SigLIP calls)",
    )
    p.add_argument("--siglip-batch-size", type=int, default=16)
    args = p.parse_args()

    _load_dotenv()
    if (args.reuse_sample or args.eval_only) and SMOKE_SAMPLE.exists():
        sample = pd.read_parquet(SMOKE_SAMPLE)
        sample["tweet_id"] = sample["tweet_id"].astype(str)
        print(f"Reusing sample {SMOKE_SAMPLE} ({len(sample):,})", flush=True)
        if len(sample) > args.limit:
            sample = sample.head(args.limit).copy()
    else:
        sample = build_sample(args.limit, seed=args.seed)
        if len(sample) > args.limit:
            sample = sample.head(args.limit).copy()

    gemini = None
    siglip = None
    if args.eval_only:
        gemini = None if args.skip_gemini else _read_cache(GEMINI_CACHE)
        siglip = None if args.skip_siglip else _read_cache(SIGLIP_CACHE)
    else:
        if not args.skip_gemini:
            gemini = label_gemini(sample, force=args.force_gemini)
        if not args.skip_siglip:
            siglip = embed_siglip(
                sample, force=args.force_siglip, batch_size=args.siglip_batch_size
            )

    report = evaluate(sample, gemini, siglip)
    write_report(report)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
