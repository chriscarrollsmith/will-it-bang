"""Score tweets for vaguely-wrongheaded / mildly-infuriating tone (post-time text only).

Default rater: Qwen3-8B (Apache-2.0) on Modal A10G, non-thinking mode.
Qwen3.6-27B also works (`--model Qwen/Qwen3.6-27B --gpu A100-80GB`) but is ~20×
slower with the current one-at-a-time multimodal generate path.

Writes `~/.cache/will-it-bang/provoke_cache_<model_slug>.parquet`.

    uv run modal run label_provoke.py --nonreply-only
    uv run modal run label_provoke.py --model Qwen/Qwen3.6-27B --gpu A100-80GB --limit 64
    uv run modal run label_provoke.py --limit 64   # smoke test
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import modal

CACHE_DIR = Path(os.path.expanduser("~/.cache/will-it-bang"))
DATASET_PATH = CACHE_DIR / "dataset.parquet"
DEFAULT_MODEL = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You rate social media posts from text alone. "
    "Do not judge popularity, fame, or whether the author is important. "
    "Reply with a single JSON object only — no markdown, no commentary."
)
USER_TEMPLATE = """Rate this post on two integer scales from 0 to 10:

- wrongheaded: overconfident, vaguely incorrect, confused, or shaky reasoning (0=none, 10=extreme)
- infuriating: mildly smug, baiting, rude, or irritating tone that might provoke argument (0=none, 10=extreme)

Most ordinary posts should score 0–3 on both. Reserve 7+ for clear cases.

Post:
\"\"\"{text}\"\"\"

JSON: {{"wrongheaded": <int>, "infuriating": <int>}}"""

_JSON_RE = re.compile(
    r'\{\s*"wrongheaded"\s*:\s*(\d+)\s*,\s*"infuriating"\s*:\s*(\d+)\s*\}'
)
_LOOSE_RE = re.compile(
    r"wrongheaded[^0-9]*(\d+).*?infuriating[^0-9]*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

app = modal.App("will-it-bang-provoke")

# Qwen3.5/3.6 need transformers from source (qwen3_5 arch); Qwen3-8B works from ~4.51+.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        "accelerate==1.7.0",
        "sentencepiece==0.2.0",
        "protobuf==5.29.5",
        "pillow==11.2.1",
        "torchvision==0.22.1",
        "einops==0.8.1",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
)
hf_cache = modal.Volume.from_name("will-it-bang-hf-cache", create_if_missing=True)


def cache_path_for(model_id: str) -> Path:
    slug = model_id.replace("/", "__")
    return CACHE_DIR / f"provoke_cache_{slug}.parquet"


def _clip01_10(n: int) -> float:
    return float(max(0, min(10, n)))


def parse_scores(raw: str) -> tuple[float, float] | None:
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
            return _clip01_10(int(obj["wrongheaded"])), _clip01_10(int(obj["infuriating"]))
        except Exception:
            m = _LOOSE_RE.search(text)
            if not m:
                return None
    return _clip01_10(int(m.group(1))), _clip01_10(int(m.group(2)))


def _is_qwen35_family(model_id: str) -> bool:
    mid = model_id.lower()
    return "qwen3.5" in mid or "qwen3.6" in mid


@app.cls(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=120,
)
class ProvokeScorer:
    @modal.enter()
    def load(self) -> None:
        import torch

        model_id = os.environ.get("PROVOKE_MODEL", DEFAULT_MODEL)
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if _is_qwen35_family(model_id):
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.kind = "qwen35"
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.kind = "causal"
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
        import torch

        assert len(tweet_ids) == len(texts)
        rows: list[dict] = []

        if self.kind == "qwen35":
            # Multimodal stack: score one-at-a-time (batching chat+processor is brittle).
            for tid, text in zip(tweet_ids, texts, strict=True):
                clipped = (text or "")[:1200]
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": USER_TEMPLATE.format(text=clipped)}],
                    },
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                raw = self.processor.batch_decode(
                    out[:, prompt_len:], skip_special_tokens=True
                )[0]
                parsed = parse_scores(raw)
                if parsed is None:
                    rows.append(
                        {
                            "tweet_id": str(tid),
                            "wrongheaded": None,
                            "infuriating": None,
                            "raw": raw[:300],
                            "ok": 0,
                            "model": self.model_id,
                        }
                    )
                else:
                    wh, inf = parsed
                    rows.append(
                        {
                            "tweet_id": str(tid),
                            "wrongheaded": wh,
                            "infuriating": inf,
                            "raw": raw[:300],
                            "ok": 1,
                            "model": self.model_id,
                        }
                    )
            return rows

        # CausalLM batch path (e.g. Qwen3-8B).
        prompts: list[str] = []
        for text in texts:
            clipped = (text or "")[:1200]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=clipped)},
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        for tid, raw in zip(tweet_ids, decoded, strict=True):
            parsed = parse_scores(raw)
            if parsed is None:
                rows.append(
                    {
                        "tweet_id": str(tid),
                        "wrongheaded": None,
                        "infuriating": None,
                        "raw": raw[:300],
                        "ok": 0,
                        "model": self.model_id,
                    }
                )
            else:
                wh, inf = parsed
                rows.append(
                    {
                        "tweet_id": str(tid),
                        "wrongheaded": wh,
                        "infuriating": inf,
                        "raw": raw[:300],
                        "ok": 1,
                        "model": self.model_id,
                    }
                )
        return rows


def _load_cache(path: Path):
    import pandas as pd

    if not path.exists():
        return pd.DataFrame(
            columns=["tweet_id", "wrongheaded", "infuriating", "ok", "model", "raw"]
        )
    cache = pd.read_parquet(path)
    cache["tweet_id"] = cache["tweet_id"].astype(str)
    return cache


def _save_cache(cache, path: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = cache.drop_duplicates("tweet_id", keep="last")
    out.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(out):,} rows)", flush=True)


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    nonreply_only: bool = False,
    limit: int = 0,
    batch_size: int = 8,
    force: bool = False,
    gpu: str = "A10G",
) -> None:
    import pandas as pd

    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing {DATASET_PATH}; run prepare.py first")

    out_path = cache_path_for(model)
    cols = ["tweet_id", "full_text"] + (["is_reply"] if nonreply_only else [])
    df = pd.read_parquet(DATASET_PATH, columns=cols)
    df["tweet_id"] = df["tweet_id"].astype(str)
    if nonreply_only:
        df = df.loc[df["is_reply"].fillna(0).astype(int) == 0].copy()
        print(f"Non-reply subset: {len(df):,}", flush=True)

    cache = _load_cache(out_path)
    done = set() if force else set(cache.loc[cache["ok"] == 1, "tweet_id"])
    todo = df.loc[~df["tweet_id"].isin(done)].copy()
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

    scorer = ProvokeScorer.with_options(gpu=gpu, env={"PROVOKE_MODEL": model})()
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
        if done_n == len(chunk) or done_n % max(batch_size * 5, 1) == 0 or done_n >= len(todo):
            elapsed = time.time() - t0
            rate = done_n / max(elapsed, 1e-6)
            print(
                f"  {done_n:,}/{len(todo):,}  ok={n_ok:,}  "
                f"{rate:.2f} tweets/s  elapsed={elapsed:.0f}s",
                flush=True,
            )
        if done_n % max(batch_size * 20, 1) == 0 or done_n >= len(todo):
            partial = pd.DataFrame(new_rows)
            merged = pd.concat([cache, partial], ignore_index=True)
            _save_cache(merged, out_path)

    merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    _save_cache(merged, out_path)
    ok = merged.loc[merged["ok"] == 1]
    print(
        f"Done. ok={len(ok):,} / {len(merged):,} | "
        f"wrongheaded mean={ok['wrongheaded'].mean():.2f} "
        f"infuriating mean={ok['infuriating'].mean():.2f} | "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )
    print(f"Symlink tip: ln -sfn {out_path.name} {CACHE_DIR / 'provoke_cache.parquet'}", flush=True)
