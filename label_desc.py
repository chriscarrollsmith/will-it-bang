"""Richer descriptive LM tags on tweet text (post-time text only).

Scores humor / controversy / personal narrative / CTA / news / dunk-bait.
No fame, quality, or influence judgments — descriptive tags of the text.

Default local rater: Qwen2.5-1.5B-Instruct (Apache-2.0) for CPU smoke.
Modal path defaults to Qwen3-8B (same family as label_provoke.py).

Writes `~/.cache/will-it-bang/desc_cache_<model_slug>.parquet`.

    # Local CPU smoke (no Modal token required):
    uv run python label_desc.py --local --nonreply-only --limit 500

    # Modal (needs `modal token`):
    uv run modal run label_desc.py --nonreply-only --limit 500
    uv run modal run label_desc.py --model Qwen/Qwen3-8B --gpu A10G
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
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MODAL_MODEL = "Qwen/Qwen3-8B"

TAG_KEYS = (
    "humor",
    "controversy",
    "narrative",
    "cta",
    "news",
    "dunk_bait",
)

SYSTEM_PROMPT = (
    "You rate social media posts from text alone. "
    "Do not judge popularity, fame, quality, correctness, or whether the author "
    "is important. Reply with a single JSON object only — no markdown, no commentary."
)
USER_TEMPLATE = """Rate this post on six integer scales from 0 to 10:

- humor: jokes, wit, irony, playful sarcasm (0=none, 10=very funny/witty)
- controversy: polarizing, hot-take, tribal, likely to spark argument (0=none, 10=extreme)
- narrative: personal story, anecdote, lived experience, first-person recounting (0=none, 10=strong)
- cta: asks reader to do something (like, RT, follow, click, reply, vote, buy) (0=none, 10=hard sell)
- news: reporting, headlines, current-events framing, link-dump of news (0=none, 10=pure news)
- dunk_bait: quote-dunk setup, ratio bait, clapback bait, deliberately provocative framing meant to invite pile-ons (0=none, 10=extreme)

Most ordinary posts should score 0–3 on each. Reserve 7+ for clear cases. Scales are independent.

Post:
\"\"\"{text}\"\"\"

JSON: {{"humor": <int>, "controversy": <int>, "narrative": <int>, "cta": <int>, "news": <int>, "dunk_bait": <int>}}"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_KEY_RES = {k: re.compile(rf"{k}\D{{0,12}}(\d+)", re.IGNORECASE) for k in TAG_KEYS}


def cache_path_for(model_id: str) -> Path:
    slug = model_id.replace("/", "__")
    return CACHE_DIR / f"desc_cache_{slug}.parquet"


def _clip01_10(n: int) -> float:
    return float(max(0, min(10, n)))


def parse_desc(raw: str) -> dict[str, float] | None:
    text = _THINK_RE.sub("", raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = None
    try:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
    except Exception:
        obj = None
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k in TAG_KEYS:
            if k not in obj:
                # Accept dunkBait / dunk-bait variants.
                alt = obj.get(k.replace("_", "")) or obj.get(k.replace("_", "-"))
                if alt is None:
                    break
                obj[k] = alt
            try:
                out[k] = _clip01_10(int(obj[k]))
            except (TypeError, ValueError):
                break
        if len(out) == len(TAG_KEYS):
            return out
    # Loose fallback: pull each key's first nearby integer.
    out = {}
    for k, rx in _KEY_RES.items():
        m = rx.search(text)
        if not m:
            return None
        out[k] = _clip01_10(int(m.group(1)))
    return out if len(out) == len(TAG_KEYS) else None


def _load_cache(path: Path):
    import pandas as pd

    cols = ["tweet_id", *TAG_KEYS, "ok", "model", "raw"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    cache = pd.read_parquet(path)
    cache["tweet_id"] = cache["tweet_id"].astype(str)
    return cache


def _save_cache(cache, path: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = cache.drop_duplicates("tweet_id", keep="last")
    out.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(out):,} rows)", flush=True)


def _empty_row(tid: str, model_id: str, raw: str, ok: int) -> dict:
    row = {
        "tweet_id": str(tid),
        "raw": (raw or "")[:400],
        "ok": ok,
        "model": model_id,
    }
    for k in TAG_KEYS:
        row[k] = None
    return row


def score_batch_local(
    model,
    tokenizer,
    tweet_ids: list[str],
    texts: list[str],
    model_id: str,
) -> list[dict]:
    import torch

    prompts: list[str] = []
    for text in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(text=(text or "")[:1200]),
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
        max_length=1024,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    rows: list[dict] = []
    for tid, raw in zip(tweet_ids, decoded, strict=True):
        parsed = parse_desc(raw)
        if parsed is None:
            rows.append(_empty_row(tid, model_id, raw, 0))
        else:
            row = _empty_row(tid, model_id, raw, 1)
            row.update(parsed)
            rows.append(row)
    return rows


def _select_tweets(*, nonreply_only: bool, limit: int, seed: int):
    import pandas as pd

    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing {DATASET_PATH}; run prepare.py first")
    cols = ["tweet_id", "full_text", "is_reply"]
    df = pd.read_parquet(DATASET_PATH, columns=cols)
    df["tweet_id"] = df["tweet_id"].astype(str)
    if nonreply_only:
        df = df.loc[df["is_reply"].fillna(0).astype(int) == 0].copy()
        print(f"Non-reply subset: {len(df):,}", flush=True)
    # Stable shuffle so --limit smokes are not just earliest-created tweets.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit > 0:
        df = df.head(limit)
    return df


def run_local(
    *,
    model_id: str,
    limit: int,
    batch_size: int,
    force: bool,
    nonreply_only: bool,
    seed: int,
) -> None:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = cache_path_for(model_id)
    # Select full candidate set first, then subtract cache (limit applies to new work).
    pool = _select_tweets(nonreply_only=nonreply_only, limit=0, seed=seed)
    cache = _load_cache(out_path)
    done = set() if force else set(cache.loc[cache["ok"] == 1, "tweet_id"].astype(str))
    todo = pool.loc[~pool["tweet_id"].isin(done)].copy()
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
    bs = max(1, batch_size if device == "cuda" else min(batch_size, 2))
    for start in range(0, len(todo), bs):
        chunk = todo.iloc[start : start + bs]
        rows = score_batch_local(
            model,
            tok,
            chunk["tweet_id"].tolist(),
            chunk["full_text"].fillna("").astype(str).tolist(),
            model_id,
        )
        new_rows.extend(rows)
        n_ok += sum(int(r["ok"]) for r in rows)
        done_n = start + len(chunk)
        elapsed = time.time() - t0
        rate = done_n / max(elapsed, 1e-6)
        print(
            f"  {done_n:,}/{len(todo):,}  ok={n_ok:,}  "
            f"{rate:.2f} tweets/s  elapsed={elapsed:.0f}s",
            flush=True,
        )
        if done_n % max(bs * 10, 1) == 0 or done_n >= len(todo):
            merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            _save_cache(merged, out_path)

    merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    _save_cache(merged, out_path)
    ok = merged.loc[merged["ok"] == 1]
    means = " ".join(f"{k}={ok[k].mean():.2f}" for k in TAG_KEYS if k in ok.columns and len(ok))
    print(f"Done. ok={len(ok):,} / {len(merged):,} | {means} | {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Modal path (optional; requires modal token)
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App("will-it-bang-desc")
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
    class DescScorer:
        @modal.enter()
        def load(self) -> None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = os.environ.get("DESC_MODEL", DEFAULT_MODAL_MODEL)
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
        def score_batch(self, tweet_ids: list[str], texts: list[str]) -> list[dict]:
            return score_batch_local(
                self.model, self.tokenizer, tweet_ids, texts, self.model_id
            )

    @app.local_entrypoint()
    def modal_main(
        model: str = DEFAULT_MODAL_MODEL,
        nonreply_only: bool = False,
        limit: int = 0,
        batch_size: int = 8,
        force: bool = False,
        gpu: str = "A10G",
        seed: int = 42,
    ) -> None:
        import pandas as pd

        out_path = cache_path_for(model)
        pool = _select_tweets(nonreply_only=nonreply_only, limit=0, seed=seed)
        cache = _load_cache(out_path)
        done = set() if force else set(cache.loc[cache["ok"] == 1, "tweet_id"].astype(str))
        todo = pool.loc[~pool["tweet_id"].isin(done)].copy()
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

        scorer = DescScorer.with_options(gpu=gpu, env={"DESC_MODEL": model})()
        t0 = time.time()
        new_rows: list[dict] = []
        n_ok = 0
        for start in range(0, len(todo), batch_size):
            chunk = todo.iloc[start : start + batch_size]
            rows = scorer.score_batch.remote(
                chunk["tweet_id"].tolist(),
                chunk["full_text"].fillna("").astype(str).tolist(),
            )
            new_rows.extend(rows)
            n_ok += sum(int(r["ok"]) for r in rows)
            done_n = start + len(chunk)
            elapsed = time.time() - t0
            rate = done_n / max(elapsed, 1e-6)
            print(
                f"  {done_n:,}/{len(todo):,}  ok={n_ok:,}  "
                f"{rate:.2f} tweets/s  elapsed={elapsed:.0f}s",
                flush=True,
            )
            if done_n % max(batch_size * 20, 1) == 0 or done_n >= len(todo):
                merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                _save_cache(merged, out_path)

        merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        _save_cache(merged, out_path)
        ok = merged.loc[merged["ok"] == 1]
        means = " ".join(
            f"{k}={ok[k].mean():.2f}" for k in TAG_KEYS if k in ok.columns and len(ok)
        )
        print(
            f"Done. ok={len(ok):,} / {len(merged):,} | {means} | {time.time() - t0:.0f}s",
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
    p.add_argument("--limit", type=int, default=0, help="Max new tweets to score (0=all)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--nonreply-only",
        action="store_true",
        help="Label non-replies only (primary viral-underpred slice)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.local:
        raise SystemExit(
            "Use --local for CPU smoke, or: uv run modal run label_desc.py …"
        )
    model = args.model or DEFAULT_LOCAL_MODEL
    run_local(
        model_id=model,
        limit=args.limit,
        batch_size=args.batch_size,
        force=args.force,
        nonreply_only=args.nonreply_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
