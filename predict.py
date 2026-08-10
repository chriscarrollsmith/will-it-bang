"""Reference scorer for a will-it-bang release inference bundle.

Usage:
    uv run predict.py --bundle ~/.cache/will-it-bang/release-v1 --input row.json
    uv run predict.py --bundle DIR --input rows.jsonl

Input JSON object(s) may include `full_text` plus any Layer-A fields from
`feature_schema.json`. Missing fields use schema defaults. MiniLM is computed
from `full_text` when raw `emb_0..383` are absent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise SystemExit(f"Unsupported JSON in {path}")


def _minilm_embed(texts: list[str], encoder_id: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_id)
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=False)
    return np.asarray(emb, dtype=np.float64)


def _raw_emb_matrix(rows: list[dict[str, Any]], n_dim: int) -> np.ndarray | None:
    cols = [f"emb_{i}" for i in range(n_dim)]
    if not all(c in rows[0] for c in cols):
        return None
    return np.asarray([[float(r.get(c, 0.0)) for c in cols] for r in rows], dtype=np.float64)


def build_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    bundle: Path,
) -> tuple[np.ndarray, list[str]]:
    schema = json.loads((bundle / "feature_schema.json").read_text())
    constants = json.loads((bundle / "constants.json").read_text())
    feature_names: list[str] = list(schema["feature_names"])
    defaults = {f["name"]: float(f["default"]) for f in schema["features"]}

    pca = joblib.load(bundle / "emb_pca.joblib")
    tfidf = joblib.load(bundle / "tfidf.joblib")
    svd = joblib.load(bundle / "text_svd.joblib")
    direction = np.load(bundle / "res_emb_cav_direction.npy")
    coef = np.load(bundle / "res_emb_baseline_coef.npy")
    baselines = json.loads((bundle / "res_emb_baselines.json").read_text())
    baseline_names: list[str] = list(baselines["names"])

    n_raw = int(constants["encoder_dims"])
    emb = _raw_emb_matrix(rows, n_raw)
    if emb is None:
        texts = [str(r.get("full_text", "")) for r in rows]
        if not any(texts):
            raise SystemExit("Need full_text or emb_0..emb_{n-1} on each row")
        emb = _minilm_embed(texts, constants["encoder_id"])
        if emb.shape[1] != n_raw:
            raise SystemExit(f"Encoder dim {emb.shape[1]} != {n_raw}")

    # Layer-A frame for residual baselines + booster columns.
    frame = pd.DataFrame(rows)
    for name, default in defaults.items():
        if name not in frame.columns and not re.fullmatch(r"emb_\d+", name) and not name.startswith(
            "tfidf_"
        ):
            frame[name] = default
        elif name in frame.columns:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(default)

    # Residualize raw MiniLM with shipped OLS coeffs, then CAV projection.
    z_cols: list[np.ndarray] = []
    for name in baseline_names:
        if name == "intercept":
            z_cols.append(np.ones(len(rows), dtype=np.float64))
            continue
        col = frame[name].to_numpy(dtype=np.float64) if name in frame.columns else np.zeros(len(rows))
        z_cols.append(np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0))
    z = np.column_stack(z_cols)
    if z.shape[1] != coef.shape[0]:
        raise SystemExit(
            f"Residual baseline width {z.shape[1]} != coef rows {coef.shape[0]} "
            f"(names={baseline_names})"
        )
    resid = emb - z @ coef
    proj = resid @ direction.reshape(-1)
    frame["res_emb_cav_proj"] = proj.astype(np.float32)
    frame["res_emb_cav_proj_sq"] = (proj * proj).astype(np.float32)

    emb_pca = pca.transform(emb.astype(np.float32))
    emb_pca_df = pd.DataFrame(
        emb_pca, columns=[f"emb_{i}" for i in range(emb_pca.shape[1])]
    )
    texts = [str(r.get("full_text", "")) for r in rows]
    tfidf_z = svd.transform(tfidf.transform(texts)).astype(np.float32)
    tfidf_df = pd.DataFrame(
        tfidf_z, columns=[f"tfidf_{i}" for i in range(tfidf_z.shape[1])]
    )
    drop_emb = [c for c in frame.columns if re.fullmatch(r"emb_\d+", c)]
    if drop_emb:
        frame = frame.drop(columns=drop_emb)
    frame = pd.concat(
        [frame.reset_index(drop=True), emb_pca_df, tfidf_df],
        axis=1,
    )

    x = np.column_stack(
        [
            pd.to_numeric(frame[c], errors="coerce").fillna(defaults.get(c, 0.0)).to_numpy(
                dtype=np.float32
            )
            if c in frame.columns
            else np.full(len(rows), defaults.get(c, 0.0), dtype=np.float32)
            for c in feature_names
        ]
    )
    return x, feature_names


def predict_rows(bundle: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x, _names = build_feature_matrix(rows, bundle=bundle)
    model = lgb.Booster(model_file=str(bundle / "model_level.txt"))
    scores = np.asarray(model.predict(x), dtype=np.float64)
    out: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        item = {
            "y_pred": float(score),
            "y_pred_eng_rate": float(np.expm1(np.clip(score, -1.0, 30.0))),
        }
        if "tweet_id" in row:
            item["tweet_id"] = row["tweet_id"]
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="Release bundle directory")
    p.add_argument("--input", type=Path, required=True, help="JSON or JSONL feature rows")
    p.add_argument("--output", type=Path, default=None, help="Write predictions JSON")
    args = p.parse_args()
    rows = _load_json_rows(args.input)
    preds = predict_rows(args.bundle, rows)
    text = json.dumps(preds, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"Wrote {len(preds)} predictions → {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
