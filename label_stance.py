"""LM agree/disagree stance of a reply (or quote) vs its parent tweet.

Text-only descriptive labels — no fame/quality/influence judgments.
Default rater: Qwen2.5-1.5B-Instruct (Apache-2.0) for local CPU smoke;
Modal path defaults to Qwen3-8B (same family as label_provoke.py).

Writes `~/.cache/will-it-bang/stance_cache_<model_slug>.parquet`.

    # Local CPU smoke (no Modal token required):
    uv run python label_stance.py --local --limit 500

    # Modal (needs `modal token`):
    uv run modal run label_stance.py --limit 500
    uv run modal run label_stance.py --model Qwen/Qwen3-8B --gpu A10G
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

CACHE_DIR = Path(os.path.expanduser("~/.cache/will-it-bang"))
DATASET_PATH = CACHE_DIR / "dataset.parquet"
RAW_PARQUET = CACHE_DIR / "enriched_tweets.parquet"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MODAL_MODEL = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You rate how a reply relates to a parent social media post, from text alone. "
    "Do not judge popularity, fame, correctness of facts, or whether either author "
    "is important. Reply with a single JSON object only — no markdown, no commentary."
)
USER_TEMPLATE = """Rate how the REPLY relates to the PARENT on two integer scales from 0 to 10:

- agree: endorses, affirms, thanks, elaborates in support, or piles on with the parent (0=none, 10=strong agreement)
- disagree: contradicts, challenges, corrects, dunks on, or argues against the parent (0=none, 10=strong disagreement)

Neutral replies (clarifying questions, tangents, jokes that neither support nor oppose, \
simple acknowledgements) should score 0–3 on both. Reserve 7+ for clear stance. \
A reply can score mid on both if it partly agrees and partly disagrees.

PARENT:
\"\"\"{parent}\"\"\"

REPLY:
\"\"\"{reply}\"\"\"

JSON: {{"agree": <int>, "disagree": <int>}}"""

_JSON_RE = re.compile(r'\{\s*"agree"\s*:\s*(\d+)\s*,\s*"disagree"\s*:\s*(\d+)\s*\}')
_LOOSE_RE = re.compile(
    r"agree[^0-9]*(\d+).*?disagree[^0-9]*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def cache_path_for(model_id: str) -> Path:
    slug = model_id.replace("/", "__")
    return CACHE_DIR / f"stance_cache_{slug}.parquet"


def _clip01_10(n: int) -> float:
    return float(max(0, min(10, n)))


def parse_stance(raw: str) -> tuple[float, float] | None:
    text = _THINK_RE.sub("", raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = _JSON_RE.search(text)
    if not m:
        try:
            obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
            return _clip01_10(int(obj["agree"])), _clip01_10(int(obj["disagree"]))
        except Exception:
            m = _LOOSE_RE.search(text)
            if not m:
                return None
    return _clip01_10(int(m.group(1))), _clip01_10(int(m.group(2)))


def load_parent_texts(parent_ids) -> "pd.DataFrame":
    """Dump (+ optional X parent cache) texts for reply/quote parent tweet ids."""
    import pandas as pd

    ids = pd.DataFrame({"tweet_id": pd.Series(parent_ids).astype(str).dropna().unique()})
    empty = pd.DataFrame(columns=["tweet_id", "full_text"])
    if ids.empty:
        return empty
    if not RAW_PARQUET.exists():
        raise SystemExit(f"Missing {RAW_PARQUET}; run prepare.py first")
    tmp = CACHE_DIR / "_stance_parent_ids.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids.to_parquet(tmp, index=False)
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads TO {max(1, os.cpu_count() or 4)}")
    dump = con.execute(
        f"""
        SELECT
          CAST(t.tweet_id AS VARCHAR) AS tweet_id,
          t.full_text
        FROM read_parquet('{RAW_PARQUET}') t
        SEMI JOIN read_parquet('{tmp}') p
          ON CAST(t.tweet_id AS VARCHAR) = p.tweet_id
        """
    ).fetchdf()
    con.close()
    try:
        tmp.unlink()
    except OSError:
        pass
    by_id = {
        str(r.tweet_id): (r.full_text or "")
        for r in dump.itertuples(index=False)
        if r.full_text
    }
    x_path = CACHE_DIR / "x_parent_cache.parquet"
    if x_path.exists():
        x = pd.read_parquet(x_path)
        if "fetch_status" in x.columns:
            x = x.loc[x["fetch_status"].astype(str).eq("ok")]
        for r in x.itertuples(index=False):
            tid = str(r.tweet_id)
            if tid in by_id:
                continue
            text = getattr(r, "full_text", None) or ""
            if text:
                by_id[tid] = str(text)
    out = pd.DataFrame(
        {"tweet_id": list(by_id.keys()), "full_text": list(by_id.values())}
    )
    print(f"Parent texts: {len(out):,}/{len(ids):,} resolved", flush=True)
    return out


def build_pair_frame(
    *,
    reply_only: bool = True,
    quote_only: bool = False,
    limit: int = 0,
    seed: int = 42,
):
    """Rows with child text + parent text for stance labeling."""
    import pandas as pd

    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing {DATASET_PATH}; run prepare.py first")
    cols = [
        "tweet_id",
        "full_text",
        "created_at",
        "is_reply",
        "is_quote",
        "reply_to_tweet_id",
        "quoted_tweet_id",
        "has_reply_parent",
        "has_quote_parent",
        "engagement",
    ]
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(DATASET_PATH).schema_arrow.names)
    cols = [c for c in cols if c in have]
    if "tweet_id" not in cols or "full_text" not in cols:
        raise SystemExit(f"Dataset missing tweet_id/full_text; have={sorted(have)[:20]}…")
    df = pd.read_parquet(DATASET_PATH, columns=cols)
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    rows: list[pd.DataFrame] = []
    if not quote_only:
        need = df["reply_to_tweet_id"].notna()
        if "has_reply_parent" in df.columns:
            need = need & (df["has_reply_parent"].fillna(0).astype(int) == 1)
        if "is_reply" in df.columns:
            need = need & (df["is_reply"].fillna(0).astype(int) == 1)
        sub = df.loc[need].copy()
        sub["parent_tweet_id"] = sub["reply_to_tweet_id"].map(
            lambda x: str(x) if pd.notna(x) else None
        )
        sub["kind"] = "reply"
        rows.append(sub)
    if quote_only or not reply_only:
        need = df["quoted_tweet_id"].notna() if "quoted_tweet_id" in df.columns else None
        if need is not None:
            if "has_quote_parent" in df.columns:
                need = need & (df["has_quote_parent"].fillna(0).astype(int) == 1)
            sub = df.loc[need].copy()
            sub["parent_tweet_id"] = sub["quoted_tweet_id"].map(
                lambda x: str(x) if pd.notna(x) else None
            )
            sub["kind"] = "quote"
            rows.append(sub)
    if not rows:
        raise SystemExit("No reply/quote rows with parent ids in dataset")
    pairs = pd.concat(rows, ignore_index=True)
    pairs = pairs.dropna(subset=["parent_tweet_id"])
    pairs = pairs.drop_duplicates("tweet_id", keep="first")

    parents = load_parent_texts(pairs["parent_tweet_id"])
    parents = parents.rename(columns={"full_text": "parent_text", "tweet_id": "parent_tweet_id"})
    pairs = pairs.merge(parents, on="parent_tweet_id", how="inner")
    pairs = pairs.loc[
        pairs["full_text"].fillna("").astype(str).str.len().gt(0)
        & pairs["parent_text"].fillna("").astype(str).str.len().gt(0)
    ].copy()
    pairs = pairs.rename(columns={"full_text": "reply_text"})
    if limit > 0 and len(pairs) > limit:
        pairs = pairs.sample(n=limit, random_state=seed).sort_values(
            "created_at", kind="mergesort"
        )
    print(
        f"Stance pairs: {len(pairs):,} "
        f"(reply={(pairs['kind']=='reply').sum():,} quote={(pairs['kind']=='quote').sum():,})",
        flush=True,
    )
    return pairs.reset_index(drop=True)


def _load_cache(path: Path):
    import pandas as pd

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "tweet_id",
                "parent_tweet_id",
                "kind",
                "agree",
                "disagree",
                "ok",
                "model",
                "raw",
            ]
        )
    cache = pd.read_parquet(path)
    cache["tweet_id"] = cache["tweet_id"].astype(str)
    return cache


def _save_cache(cache, path: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = cache.drop_duplicates("tweet_id", keep="last")
    out.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(out):,} rows)", flush=True)


def score_batch_local(
    model,
    tokenizer,
    tweet_ids: list[str],
    parent_ids: list[str],
    kinds: list[str],
    parents: list[str],
    replies: list[str],
    model_id: str,
) -> list[dict]:
    import torch

    prompts: list[str] = []
    for parent, reply in zip(parents, replies, strict=True):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    parent=(parent or "")[:900],
                    reply=(reply or "")[:900],
                ),
            },
        ]
        try:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1536,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    rows: list[dict] = []
    for tid, pid, kind, raw in zip(tweet_ids, parent_ids, kinds, decoded, strict=True):
        parsed = parse_stance(raw)
        if parsed is None:
            rows.append(
                {
                    "tweet_id": str(tid),
                    "parent_tweet_id": str(pid),
                    "kind": kind,
                    "agree": None,
                    "disagree": None,
                    "raw": raw[:300],
                    "ok": 0,
                    "model": model_id,
                }
            )
        else:
            ag, dg = parsed
            rows.append(
                {
                    "tweet_id": str(tid),
                    "parent_tweet_id": str(pid),
                    "kind": kind,
                    "agree": ag,
                    "disagree": dg,
                    "raw": raw[:300],
                    "ok": 1,
                    "model": model_id,
                }
            )
    return rows


def run_local(
    *,
    model_id: str,
    limit: int,
    batch_size: int,
    force: bool,
    reply_only: bool,
    include_quotes: bool,
    seed: int,
) -> None:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = cache_path_for(model_id)
    pairs = build_pair_frame(
        reply_only=not include_quotes,
        quote_only=False,
        limit=0,
        seed=seed,
    )
    cache = _load_cache(out_path)
    done = set() if force else set(cache.loc[cache["ok"] == 1, "tweet_id"].astype(str))
    todo = pairs.loc[~pairs["tweet_id"].isin(done)].copy()
    if limit > 0:
        todo = todo.head(limit)
    print(
        f"Local model={model_id} cache={out_path.name}\n"
        f"To score: {len(todo):,} (cached ok={len(done):,} force={force} batch={batch_size})",
        flush=True,
    )
    if todo.empty:
        print("Nothing to do.", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    t0 = time.time()
    new_rows: list[dict] = []
    n_ok = 0
    # CPU: keep batches small.
    bs = max(1, batch_size if device == "cuda" else min(batch_size, 2))
    for start in range(0, len(todo), bs):
        chunk = todo.iloc[start : start + bs]
        rows = score_batch_local(
            model,
            tok,
            chunk["tweet_id"].tolist(),
            chunk["parent_tweet_id"].tolist(),
            chunk["kind"].tolist(),
            chunk["parent_text"].fillna("").astype(str).tolist(),
            chunk["reply_text"].fillna("").astype(str).tolist(),
            model_id,
        )
        new_rows.extend(rows)
        n_ok += sum(int(r["ok"]) for r in rows)
        done_n = start + len(chunk)
        elapsed = time.time() - t0
        rate = done_n / max(elapsed, 1e-6)
        print(
            f"  {done_n:,}/{len(todo):,}  ok={n_ok:,}  "
            f"{rate:.2f} pairs/s  elapsed={elapsed:.0f}s",
            flush=True,
        )
        if done_n % max(bs * 10, 1) == 0 or done_n >= len(todo):
            merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            _save_cache(merged, out_path)

    merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    _save_cache(merged, out_path)
    ok = merged.loc[merged["ok"] == 1]
    print(
        f"Done. ok={len(ok):,} / {len(merged):,} | "
        f"agree mean={ok['agree'].mean():.2f} disagree mean={ok['disagree'].mean():.2f} | "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Modal path (optional; requires modal token)
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App("will-it-bang-stance")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.7.1",
            "accelerate==1.7.0",
            "sentencepiece==0.2.0",
            "protobuf==5.29.5",
            "transformers>=4.51.0",
        )
    )
    hf_cache = modal.Volume.from_name("will-it-bang-hf-cache", create_if_missing=True)

    @app.cls(
        image=image,
        gpu="A10G",
        timeout=60 * 60 * 4,
        volumes={"/root/.cache/huggingface": hf_cache},
        scaledown_window=120,
    )
    class StanceScorer:
        @modal.enter()
        def load(self) -> None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = os.environ.get("STANCE_MODEL", DEFAULT_MODAL_MODEL)
            self.model_id = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            hf_cache.commit()

        @modal.method()
        def score_batch(
            self,
            tweet_ids: list[str],
            parent_ids: list[str],
            kinds: list[str],
            parents: list[str],
            replies: list[str],
        ) -> list[dict]:
            return score_batch_local(
                self.model,
                self.tokenizer,
                tweet_ids,
                parent_ids,
                kinds,
                parents,
                replies,
                self.model_id,
            )

    @app.local_entrypoint()
    def modal_main(
        model: str = DEFAULT_MODAL_MODEL,
        limit: int = 0,
        batch_size: int = 8,
        force: bool = False,
        gpu: str = "A10G",
        include_quotes: bool = False,
        seed: int = 42,
    ) -> None:
        import pandas as pd

        out_path = cache_path_for(model)
        pairs = build_pair_frame(
            reply_only=not include_quotes,
            quote_only=False,
            limit=0,
            seed=seed,
        )
        if include_quotes:
            pairs = build_pair_frame(reply_only=False, limit=0, seed=seed)
        cache = _load_cache(out_path)
        done = set() if force else set(cache.loc[cache["ok"] == 1, "tweet_id"].astype(str))
        todo = pairs.loc[~pairs["tweet_id"].isin(done)].copy()
        if limit > 0:
            todo = todo.head(limit)
        print(
            f"Model={model} gpu={gpu} cache={out_path.name}\n"
            f"To score: {len(todo):,} (cached ok={len(done):,} force={force} batch={batch_size})",
            flush=True,
        )
        if todo.empty:
            print("Nothing to do.", flush=True)
            return

        scorer = StanceScorer.with_options(gpu=gpu, env={"STANCE_MODEL": model})()
        t0 = time.time()
        new_rows: list[dict] = []
        n_ok = 0
        for start in range(0, len(todo), batch_size):
            chunk = todo.iloc[start : start + batch_size]
            rows = scorer.score_batch.remote(
                chunk["tweet_id"].tolist(),
                chunk["parent_tweet_id"].tolist(),
                chunk["kind"].tolist(),
                chunk["parent_text"].fillna("").astype(str).tolist(),
                chunk["reply_text"].fillna("").astype(str).tolist(),
            )
            new_rows.extend(rows)
            n_ok += sum(int(r["ok"]) for r in rows)
            done_n = start + len(chunk)
            elapsed = time.time() - t0
            rate = done_n / max(elapsed, 1e-6)
            print(
                f"  {done_n:,}/{len(todo):,}  ok={n_ok:,}  "
                f"{rate:.2f} pairs/s  elapsed={elapsed:.0f}s",
                flush=True,
            )
            if done_n % max(batch_size * 20, 1) == 0 or done_n >= len(todo):
                merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                _save_cache(merged, out_path)

        merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        _save_cache(merged, out_path)
        ok = merged.loc[merged["ok"] == 1]
        print(
            f"Done. ok={len(ok):,} / {len(merged):,} | "
            f"agree mean={ok['agree'].mean():.2f} disagree mean={ok['disagree'].mean():.2f} | "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )

except ImportError:
    app = None  # type: ignore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--local",
        action="store_true",
        help="Run on local CPU/GPU with transformers (no Modal token)",
    )
    p.add_argument("--model", type=str, default=None, help="HF model id")
    p.add_argument("--limit", type=int, default=0, help="Max new pairs to score (0=all)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--include-quotes",
        action="store_true",
        help="Also label quote→parent pairs (default: replies only)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.local:
        raise SystemExit(
            "Use --local for CPU smoke, or: uv run modal run label_stance.py …"
        )
    model = args.model or DEFAULT_LOCAL_MODEL
    run_local(
        model_id=model,
        limit=args.limit,
        batch_size=args.batch_size,
        force=args.force,
        reply_only=True,
        include_quotes=args.include_quotes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
