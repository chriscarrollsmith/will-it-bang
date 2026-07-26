"""
Train a hybrid engagement model on the prepared Community Archive dataset.

Frozen MiniLM embeddings + leakage-safe tabular features → LightGBM regressor.
Default target is an exposure-adjusted log rate (maturity-controlled intensity),
not raw dump-time log1p(favorites + retweets). Evaluation is a global time holdout.

History features must not use dump-time fav/RT (see AGENTS.md). Author
popularity inputs are pre-T reply-received rates. Default stack also uses
train-fit emb PCA + TF-IDF SVD. Context residual is off under rate targets
(mixed units); use --no-exposure-correct for dump-total prediction + residual.

Usage:
    uv run train.py
    uv run train.py --no-exposure-correct --residual   # dump-total / v13-style
    uv run train.py --skip-embeddings
    uv run train.py --model gpboost --no-account-te    # account RE instead of LOO TE
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CACHE_DIR = Path.home() / ".cache" / "will-it-bang"
DATASET_PATH = CACHE_DIR / "dataset.parquet"
MODEL_PATH = CACHE_DIR / "model.txt"
METRICS_PATH = CACHE_DIR / "metrics.json"
VAL_PRED_PATH = CACHE_DIR / "val_predictions.parquet"
ANALYSIS_PATH = CACHE_DIR / "error_analysis.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.tsv"

TABULAR_FEATURES = [
    "text_len",
    "log_text_len",
    "has_url",
    "n_mentions",
    "n_hashtags",
    "has_question",
    "has_exclaim",
    "n_emoji",
    "starts_mention",
    "caps_ratio",
    "has_media",
    "n_media",
    "has_ca_url",
    "n_ca_urls",
    "is_reply",
    "is_quote",
    "is_self_reply",
    "hour",
    "dow",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "hist_n",
    "hist_tweets_7d",
    "hist_tweets_30d",
    "days_since_last",
    "hist_frac_reply",
    "hist_frac_url",
    "hist_frac_media",
    "hist_mean_replies",
    "hist_median_replies",
    "hist_p90_replies",
    "hist_last_replies",
    "hist_std_replies",
    "hist_mean_replies_7d",
    "hist_mean_replies_30d",
    "hist_decay_mean_replies",
    "hist_mean_replies_nonreply",
    "hist_mean_replies_reply",
    "reply_x_hist_replies_nr",
    "media_x_hist_replies",
    "hist_dump_n",
    "hist_dump_mean_replies",
    "hist_dump_decay_replies",
    "hist_dump_p90_replies",
    "inbound_replies",
    "inbound_replies_7d",
    "inbound_replies_30d",
    "inbound_replies_per_tweet",
    "inbound_quotes",
    "inbound_quotes_7d",
    "inbound_quotes_30d",
    "inbound_quotes_per_tweet",
    "inbound_mentions",
    "inbound_mentions_7d",
    "inbound_mentions_30d",
    "inbound_mentions_per_tweet",
    "media_x_inbound_replies",
    "reply_x_inbound_replies",
    "media_x_inbound_mentions",
    "reply_x_inbound_mentions",
    "has_reply_parent",
    "reply_parent_emb_cos",
    "reply_parent_log_text_len",
    "reply_parent_is_reply",
    "reply_parent_is_quote",
    "reply_parent_same_author",
    "reply_parent_age_hours",
    "reply_parent_replies",
    "has_quote_parent",
    "quote_parent_emb_cos",
    "quote_parent_log_text_len",
    "quote_parent_is_reply",
    "quote_parent_is_quote",
    "quote_parent_same_author",
    "quote_parent_age_hours",
    "quote_parent_replies",
    "year",
    "month_sin",
    "month_cos",
    "platform_era",
    "era_pre_timeline",
    "era_algo_timeline",
    "era_pre_musk",
    "era_musk_early",
    "era_musk_late",
]

# Opt-in CA provenance / upload-regime features (--provenance-features).
PROVENANCE_FEATURES = [
    "is_stream",
    "has_upload_end",
    "hours_since_upload_end",
    "log_hours_since_upload_end",
    "stream_x_log_age",
    "stream_x_inbound_replies",
]

# CA tweet_media type breakdown (--media-types); requires prepare media refresh.
MEDIA_TYPE_FEATURES = [
    "has_photo",
    "n_photo",
    "has_video",
    "n_video",
    "has_animated_gif",
    "n_animated_gif",
    "has_media_other",
    "n_media_other",
    "photo_x_inbound_replies",
    "video_x_inbound_replies",
    "animated_gif_x_inbound_replies",
]

# Opt-in descriptive text tags (--descriptive-text); flat on rate RMSE in v15.
DESCRIPTIVE_TEXT_FEATURES = [
    "sent_neg",
    "sent_neu",
    "sent_pos",
    "sent_compound",
    "sent_abs_compound",
    "first_person_ratio",
    "n_sentences",
]

# From prepare.attach_url_domains (+ train-fold domain_mean_rate).
URL_DOMAIN_FEATURES = [
    "n_text_urls",
    "url_only",
    "url_at_end",
    "has_expanded_domain",
    "n_ca_url_rows",
    "domain_is_twitter",
    "domain_is_youtube",
    "domain_is_github",
    "domain_is_substack",
    "domain_is_wiki",
    "domain_mean_rate",
    "log_domain_n",
]

# Cached LM descriptive scores (label_provoke.py); used when columns present.
PROVOKE_FEATURES = [
    "wrongheaded",
    "infuriating",
    "provoke",
    "has_provoke",
]

# Cached LM agree/disagree vs parent (label_stance.py); opt-in --stance-labels.
STANCE_FEATURES = [
    "stance_agree",
    "stance_disagree",
    "stance_net",
    "stance_conflict",
    "has_stance",
]

# Cached richer descriptive LM tags (label_desc.py); opt-in --desc-labels.
DESC_FEATURES = [
    "desc_humor",
    "desc_controversy",
    "desc_narrative",
    "desc_cta",
    "desc_news",
    "desc_dunk_bait",
    "desc_max",
    "has_desc",
]

# Train-fold account target encoding (add_account_target_encoding).
ACCOUNT_TE_FEATURES = [
    "account_mean_rate",
    "log_account_n",
]

# Train-fold platform-era target encoding (--era-te); leakage-safe calendar prior.
ERA_TE_FEATURES = [
    "era_mean_rate",
    "log_era_n",
    "era_stratum_mean_rate",
    "log_era_stratum_n",
    "te_minus_era",
]

# Train-fold MiniLM neighborhood TE (cold-start content prior).
EMB_KNN_TE_FEATURES = [
    "emb_knn_mean_rate",
    "emb_knn_std_rate",
    "emb_knn_mean_sim",
    "account_is_cold",
    "knn_x_cold",
    "te_minus_knn",
]

# Train-fold engagement CAV: projection onto μ_hot − μ_cold (raw MiniLM).
EMB_CAV_FEATURES = [
    "emb_cav_proj",
    "emb_cav_proj_sq",
]

# CAV on MiniLM residualized against account TE / reply baselines (opt-in).
RESIDUAL_EMB_CAV_FEATURES = [
    "res_emb_cav_proj",
    "res_emb_cav_proj_sq",
]

# Default covariates for residual-emb (leakage-safe; account TE when present).
RESIDUAL_EMB_BASELINES = (
    "account_mean_rate",
    "is_reply",
    "inbound_replies_per_tweet",
)

# Train-fold MiniLM k-means cluster target encoding (content neighborhoods).
EMB_CLUSTER_TE_FEATURES = [
    "cluster_mean_rate",
    "log_cluster_n",
    "cluster_min_dist",
    "te_minus_cluster",
    "knn_minus_cluster",
]

# Added when --author-emb-affinity is set (flat alone in v10).
AFFINITY_FEATURES = [
    "author_emb_cos",
    "author_emb_l2",
    "author_emb_cos_30d",
    "author_emb_l2_30d",
]

# Pre-T inbound interlocutor MiniLM affinity (--interlocutor-emb-affinity).
INTERLOCUTOR_FEATURES = [
    "interlocutor_emb_cos",
    "interlocutor_emb_l2",
    "interlocutor_emb_cos_sq",
    "interlocutor_sweet_dist",
    "interlocutor_log_n",
    "interlocutor_has",
    "interlocutor_emb_cos_30d",
    "interlocutor_log_n_30d",
    "interlocutor_n_partners",
]

# Reply/quote dyad + parent-author size/corpus MiniLM (--reply-dyad); prepare --dyad-only.
DYAD_FEATURES = [
    "reply_parent_author_known",
    "reply_parent_author_hist_n",
    "reply_parent_author_inbound",
    "reply_parent_author_inbound_per",
    "reply_dyad_inbound",
    "reply_dyad_outbound",
    "reply_dyad_has_prior",
    "reply_dyad_bidir",
    "reply_parent_author_emb_cos",
    "reply_parent_author_emb_l2",
    "reply_parent_author_emb_log_n",
    "reply_parent_author_emb_has",
    "quote_parent_author_known",
    "quote_parent_author_hist_n",
    "quote_parent_author_inbound",
    "quote_parent_author_inbound_per",
    "quote_dyad_inbound",
    "quote_dyad_outbound",
    "quote_dyad_has_prior",
    "quote_dyad_bidir",
    "quote_parent_author_emb_cos",
    "quote_parent_author_emb_l2",
    "quote_parent_author_emb_log_n",
    "quote_parent_author_emb_has",
]

# 2nd-hop reply/quote parents (--parent-chain); prepare --parent-chain-only / --parent-only.
PARENT_CHAIN_FEATURES = [
    "has_reply_chain_parent",
    "reply_chain_emb_cos",
    "reply_chain_parent_cos",
    "reply_chain_log_text_len",
    "reply_chain_same_author",
    "reply_chain_age_hours",
    "reply_chain_replies",
    "reply_chain_via_reply",
    "reply_chain_via_quote",
    "has_quote_chain_parent",
    "quote_chain_emb_cos",
    "quote_chain_parent_cos",
    "quote_chain_log_text_len",
    "quote_chain_same_author",
    "quote_chain_age_hours",
    "quote_chain_replies",
    "quote_chain_via_reply",
    "quote_chain_via_quote",
]

# TE × own-corpus novelty (needs affinity + account TE columns).
BREAKOUT_FEATURES = [
    "author_emb_novelty",
    "author_emb_novelty_sq",
    "te_x_novelty",
    "te_x_author_l2",
    "te_x_novelty_30d",
    "novelty_x_nonreply",
]


def feature_matrix(df: pd.DataFrame, skip_embeddings: bool) -> tuple[np.ndarray, list[str]]:
    # Provenance raw cols (is_stream, …) live on the dataset for holdout/analyze;
    # only include them when --provenance-features has built the derived marker col.
    provenance_cols = (
        PROVENANCE_FEATURES if "log_hours_since_upload_end" in df.columns else []
    )
    # Media-type cols are prepared always after media refresh; include when the
    # train-time marker `has_photo` is present *and* caller kept those columns
    # (see --media-types / drop in main).
    media_type_cols = MEDIA_TYPE_FEATURES if "has_photo" in df.columns else []
    extra = [
        c
        for c in (
            AFFINITY_FEATURES
            + INTERLOCUTOR_FEATURES
            + DYAD_FEATURES
            + PARENT_CHAIN_FEATURES
            + BREAKOUT_FEATURES
            + DESCRIPTIVE_TEXT_FEATURES
            + URL_DOMAIN_FEATURES
            + PROVOKE_FEATURES
            + STANCE_FEATURES
            + DESC_FEATURES
            + ACCOUNT_TE_FEATURES
            + ERA_TE_FEATURES
            + EMB_KNN_TE_FEATURES
            + EMB_CAV_FEATURES
            + RESIDUAL_EMB_CAV_FEATURES
            + EMB_CLUSTER_TE_FEATURES
            + provenance_cols
            + media_type_cols
        )
        if c in df.columns
    ]
    tabular = TABULAR_FEATURES + extra
    cols = [c for c in tabular if c in df.columns]
    missing = [c for c in TABULAR_FEATURES if c not in df.columns]
    if missing:
        print(f"Warning: missing tabular features (skipped): {missing}")
    if not skip_embeddings:
        emb_cols = [c for c in df.columns if re.fullmatch(r"emb_\d+", c)]
        if not emb_cols:
            raise SystemExit("Dataset has no emb_* columns — re-run prepare.py")
        emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))
        cols = cols + emb_cols
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    cols = cols + tfidf_cols
    pls_cols = [c for c in df.columns if re.fullmatch(r"pls_\d+", c)]
    if pls_cols:
        pls_cols = sorted(pls_cols, key=lambda c: int(c.split("_", 1)[1]))
        cols = cols + pls_cols
    res_pls_cols = [c for c in df.columns if re.fullmatch(r"res_pls_\d+", c)]
    if res_pls_cols:
        res_pls_cols = sorted(res_pls_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
        cols = cols + res_pls_cols
    fail_cols = [c for c in df.columns if c == "fail_min_dist" or re.fullmatch(r"fail_dist_\d+", c)]
    if fail_cols:
        fail_cols = sorted(
            fail_cols,
            key=lambda c: (-1 if c == "fail_min_dist" else int(c.rsplit("_", 1)[1])),
        )
        cols = cols + fail_cols
    if "p_breakout" in df.columns:
        cols = cols + ["p_breakout"]
    x = df[cols].to_numpy(dtype=np.float32)
    return x, cols


def time_split(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("created_at", kind="mergesort").reset_index(drop=True)
    cut = int(len(df) * (1.0 - val_frac))
    cut = max(1, min(cut, len(df) - 1))
    train = df.iloc[:cut]
    val = df.iloc[cut:]
    print(
        f"Time split: train={len(train):,} "
        f"[{train['created_at'].min()} → {train['created_at'].max()}], "
        f"val={len(val):,} "
        f"[{val['created_at'].min()} → {val['created_at'].max()}]"
    )
    print(
        f"Accounts: train={train['account_id'].nunique():,} "
        f"val={val['account_id'].nunique():,} "
        f"overlap={len(set(train['account_id']) & set(val['account_id'])):,}"
    )
    if "is_stream" in train.columns:
        print(
            f"Provenance: train stream={train['is_stream'].mean():.1%} "
            f"val stream={val['is_stream'].mean():.1%}",
            flush=True,
        )
    return train, val


def prior_upload_hist_counts(df: pd.DataFrame) -> np.ndarray:
    """Per-row count of earlier upload-linked tweets by the same account (pre-T)."""
    if "is_stream" not in df.columns:
        return np.zeros(len(df), dtype=np.int32)
    ordered = df.sort_values(["account_id", "created_at"], kind="mergesort")
    is_upload = (ordered["is_stream"].to_numpy() == 0).astype(np.int32)
    # Cumsum within account, then subtract current row ⇒ strictly earlier uploads.
    prior = (
        pd.Series(is_upload, index=ordered.index)
        .groupby(ordered["account_id"], sort=False)
        .cumsum()
        .to_numpy(dtype=np.int32)
        - is_upload
    )
    out = np.zeros(len(df), dtype=np.int32)
    out[ordered.index.to_numpy()] = prior
    return out


def apply_holdout_policy(
    df: pd.DataFrame,
    *,
    policy: str,
    min_upload_hist: int = 0,
) -> pd.DataFrame:
    """Filter rows before the global time split by CA ingestion regime.

    Policies:
      time         — no regime filter (default global time holdout)
      upload_era   — cap at the latest upload-linked created_at (drop pure-stream tail)
      stream_only  — keep only browser-extension stream rows (null archive_upload_id)
    """
    out = df.copy()
    if policy not in {"time", "upload_era", "stream_only"}:
        raise SystemExit(f"Unknown holdout policy: {policy}")
    if policy != "time" or min_upload_hist > 0:
        if "is_stream" not in out.columns:
            raise SystemExit(
                "Holdout policy requires is_stream; run: uv run prepare.py --provenance-only"
            )

    if min_upload_hist > 0:
        prior = prior_upload_hist_counts(out)
        keep = prior >= int(min_upload_hist)
        print(
            f"min_upload_hist={min_upload_hist}: kept {int(keep.sum()):,}/{len(out):,} "
            f"(mean prior upload-linked={float(prior.mean()):.2f})",
            flush=True,
        )
        out = out.loc[keep].reset_index(drop=True)

    if policy == "upload_era":
        upload_times = out.loc[out["is_stream"].to_numpy() == 0, "created_at"]
        if upload_times.empty:
            raise SystemExit("upload_era holdout: no upload-linked rows in dataset")
        cap = upload_times.max()
        before = len(out)
        out = out.loc[out["created_at"] <= cap].reset_index(drop=True)
        print(
            f"Holdout policy=upload_era: cap created_at≤{cap} "
            f"kept {len(out):,}/{before:,} "
            f"(stream after filter={out['is_stream'].mean():.1%})",
            flush=True,
        )
    elif policy == "stream_only":
        before = len(out)
        out = out.loc[out["is_stream"].to_numpy() == 1].reset_index(drop=True)
        print(
            f"Holdout policy=stream_only: kept {len(out):,}/{before:,} stream rows",
            flush=True,
        )
    else:
        print("Holdout policy=time (global time split, no regime filter)", flush=True)

    if len(out) < 50:
        raise SystemExit(f"Holdout policy left only {len(out)} rows; need a denser dataset")
    return out


def add_provenance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build opt-in is_stream / upload-end features + light interactions."""
    if "is_stream" not in df.columns:
        raise SystemExit(
            "Missing is_stream; run: uv run prepare.py --provenance-only"
        )
    out = df.copy()
    out["is_stream"] = out["is_stream"].fillna(0).astype(np.int8)
    if "has_upload_end" not in out.columns:
        out["has_upload_end"] = np.int8(0)
    if "hours_since_upload_end" not in out.columns:
        out["hours_since_upload_end"] = np.float32(0.0)
    hours = out["hours_since_upload_end"].to_numpy(dtype=np.float64)
    # log1p of non-negative hours; negative ⇒ still inside prior archive coverage window.
    out["log_hours_since_upload_end"] = np.log1p(np.clip(hours, 0.0, None)).astype(np.float32)
    if "exposure_age_days" in out.columns:
        age = out["exposure_age_days"].to_numpy(dtype=np.float64)
    else:
        # Fallback: age vs dataset max (same dump-time proxy as exposure rate).
        dump_time = pd.Timestamp(out["created_at"].max())
        age = (dump_time - pd.to_datetime(out["created_at"], utc=True)).dt.total_seconds().to_numpy()
        age = np.clip(age / 86400.0, 0.0, None)
    out["stream_x_log_age"] = (
        out["is_stream"].to_numpy(dtype=np.float64) * np.log1p(age)
    ).astype(np.float32)
    if "inbound_replies" in out.columns:
        out["stream_x_inbound_replies"] = (
            out["is_stream"].to_numpy(dtype=np.float64)
            * out["inbound_replies"].to_numpy(dtype=np.float64)
        ).astype(np.float32)
    else:
        out["stream_x_inbound_replies"] = np.float32(0.0)
    print(
        f"Provenance features: stream={out['is_stream'].mean():.1%} "
        f"has_upload_end={out['has_upload_end'].mean():.1%}",
        flush=True,
    )
    return out


def stream_fresh_sample_weights(
    train_df: pd.DataFrame,
    base_weights: np.ndarray,
    *,
    fresh_days: float,
    floor: float = 0.25,
) -> np.ndarray:
    """Down-weight very fresh stream rows (as-observed eng; maturity still uncertain)."""
    w = np.asarray(base_weights, dtype=np.float64).copy()
    if fresh_days <= 0 or "is_stream" not in train_df.columns:
        return w
    if "exposure_age_days" in train_df.columns:
        age = train_df["exposure_age_days"].to_numpy(dtype=np.float64)
    else:
        dump_time = pd.Timestamp(train_df["created_at"].max())
        age = (
            dump_time - pd.to_datetime(train_df["created_at"], utc=True)
        ).dt.total_seconds().to_numpy() / 86400.0
        age = np.clip(age, 0.0, None)
    stream = train_df["is_stream"].to_numpy() == 1
    # Linear ramp: age=0 → floor, age≥fresh_days → 1.
    ramp = np.clip(age / float(fresh_days), 0.0, 1.0)
    mult = floor + (1.0 - floor) * ramp
    w[stream] *= mult[stream]
    n_affected = int((stream & (mult < 1.0 - 1e-12)).sum())
    print(
        f"Stream-fresh weights: days={fresh_days:g} floor={floor:g} "
        f"n_stream={int(stream.sum()):,} n_affected={n_affected:,} "
        f"mean_mult={float(mult[stream].mean()) if stream.any() else 1.0:.3f} "
        f"train_age_min={float(age.min()):.1f}d",
        flush=True,
    )
    if stream.any() and n_affected == 0:
        print(
            "Warning: stream-fresh affected 0 train rows — under a time holdout, "
            "train ages are ≥ the val-window span; raise --stream-fresh-days "
            f"(need >~{float(age.min()):.0f}d here) or the knob is a no-op",
            flush=True,
        )
    return w


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    spearman_val = spearmanr(y_true, y_pred).correlation
    spearman = float(spearman_val) if spearman_val == spearman_val else float("nan")
    pred_eng = np.expm1(np.clip(y_pred, 0, 20))
    true_eng = np.expm1(y_true)
    eng_mae = float(np.mean(np.abs(pred_eng - true_eng)))
    return {
        "rmse_log": rmse,
        "mae_log": mae,
        "r2_log": r2,
        "spearman": spearman,
        "mae_engagement": eng_mae,
    }


def within_account_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    account_ids: np.ndarray,
    *,
    min_n: int = 2,
) -> dict[str, float]:
    """Mean/median/size-weighted Spearman within each account (draft-quality metric)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    account_ids = np.asarray(account_ids)
    corrs: list[float] = []
    weights: list[float] = []
    # Group by sorting accounts (faster than pandas groupby for numpy arrays).
    order = np.argsort(account_ids, kind="mergesort")
    ids_sorted = account_ids[order]
    y_s = y_true[order]
    p_s = y_pred[order]
    _, starts, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    for start, n in zip(starts, counts, strict=True):
        if int(n) < min_n:
            continue
        sl = slice(start, start + int(n))
        c = spearmanr(y_s[sl], p_s[sl]).correlation
        if c == c:
            corrs.append(float(c))
            weights.append(float(n))
    if not corrs:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "weighted_mean": float("nan"),
            "n_accounts": 0.0,
        }
    c_arr = np.asarray(corrs, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    return {
        "mean": float(c_arr.mean()),
        "median": float(np.median(c_arr)),
        "weighted_mean": float(np.average(c_arr, weights=w_arr)),
        "n_accounts": float(c_arr.size),
    }


def blend_level_with_rank_scores(
    level: np.ndarray,
    rank_score: np.ndarray,
    account_ids: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Keep per-account level mean/std; impose ranker's within-account order.

    For each account (optionally masked to a stratum): replace within-account
    variation with the ranker's z-scored order, scaled to the level model's
    within-account std. Singleton accounts keep the level prediction.
    """
    out = np.asarray(level, dtype=np.float64).copy()
    rank_score = np.asarray(rank_score, dtype=np.float64)
    account_ids = np.asarray(account_ids)
    if mask is None:
        idx = np.arange(out.shape[0])
    else:
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return out
    ids = account_ids[idx]
    order = np.argsort(ids, kind="mergesort")
    idx_s = idx[order]
    ids_s = ids[order]
    _, starts, counts = np.unique(ids_s, return_index=True, return_counts=True)
    for start, n in zip(starts, counts, strict=True):
        if int(n) < 2:
            continue
        rows = idx_s[start : start + int(n)]
        lev = out[rows]
        r = rank_score[rows]
        r_std = float(r.std())
        if r_std < 1e-12:
            continue
        r_z = (r - float(r.mean())) / r_std
        lev_std = float(lev.std())
        if lev_std < 1e-12:
            lev_std = 1.0
        out[rows] = float(lev.mean()) + r_z * lev_std
    return out


def _medcouple(x: np.ndarray, *, max_n: int = 2000, seed: int = 0) -> float:
    """Medcouple skewness (Hubert & Vandervieren) for adjusted boxplot fences."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    if x.size > max_n:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=max_n, replace=False)
    x = np.sort(x)
    med = float(np.median(x))
    left = x[x <= med]
    right = x[x >= med]
    if left.size == 0 or right.size == 0:
        return 0.0
    a = (med - left)[:, None]
    b = (right - med)[None, :]
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(denom > 0.0, (b - a) / denom, np.nan)
    mc = np.nanmedian(h)
    return float(mc) if mc == mc else 0.0


def relevance_phi_sera(
    y: np.ndarray,
    *,
    high_only: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Train-fold SERA relevance φ(y)∈[0,1] via adjusted-boxplot control points + PCHIP.

    High-only (default): extremes above the upper fence are relevant; below the
    median φ=0. Matches Ribeiro & Moniz (2021) when only high extremes matter.
    Per-sample contribution to SERA is φ(y)·(ŷ−y)², so φ is the MSE weight.
    """
    y = np.asarray(y, dtype=np.float64)
    finite = y[np.isfinite(y)]
    if finite.size < 10:
        phi = np.ones(y.shape[0], dtype=np.float64)
        return phi, {"sera_med": 0.0, "sera_upper": 1.0, "sera_mc": 0.0, "sera_frac_pos": 1.0}

    q1, med, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    iqr = float(q3 - q1)
    mc = _medcouple(finite, seed=seed)
    if iqr <= 1e-12:
        upper = float(np.quantile(finite, 0.95))
        lower = float(np.quantile(finite, 0.05))
    elif mc >= 0.0:
        upper = float(q3 + 1.5 * np.exp(3.0 * mc) * iqr)
        lower = float(q1 - 1.5 * np.exp(4.0 * mc) * iqr)
    else:
        upper = float(q3 + 1.5 * np.exp(4.0 * mc) * iqr)
        lower = float(q1 - 1.5 * np.exp(-3.0 * mc) * iqr)

    y_min = float(finite.min())
    y_max = float(finite.max())
    # Ensure fences are interior enough for a non-degenerate interpolant.
    upper = min(max(upper, float(med) + 1e-6), y_max)
    lower = max(min(lower, float(med) - 1e-6), y_min)

    if high_only:
        xs = np.array([y_min, float(med), upper, y_max], dtype=np.float64)
        ys = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    else:
        xs = np.array([y_min, lower, float(med), upper, y_max], dtype=np.float64)
        ys = np.array([1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    # Drop duplicate x (can happen on discrete / clipped targets).
    uniq_x, uniq_idx = np.unique(xs, return_index=True)
    uniq_y = ys[uniq_idx]
    if uniq_x.size < 2:
        phi = np.ones(y.shape[0], dtype=np.float64)
    else:
        interp = PchipInterpolator(uniq_x, uniq_y, extrapolate=True)
        phi = np.clip(interp(y), 0.0, 1.0).astype(np.float64)
    meta = {
        "sera_med": float(med),
        "sera_upper": float(upper),
        "sera_lower": float(lower),
        "sera_mc": float(mc),
        "sera_frac_pos": float(np.mean(phi > 1e-6)),
        "sera_mean_phi": float(np.mean(phi)),
    }
    return phi, meta


def apply_sera_sample_weights(
    y_train: np.ndarray,
    weight_train: np.ndarray | None,
    *,
    sera_floor: float = 0.05,
    high_only: bool = True,
    seed: int = 0,
    apply_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Multiply sample weights by floor+(1−floor)·φ(y) for SERA-weighted MSE.

    If ``apply_mask`` is set, fit φ on the masked rows only and leave unmasked
    rows at their base weight (used for non-reply-stratum SERA).
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    floor = float(np.clip(sera_floor, 0.0, 1.0))
    base = np.ones(y_train.shape[0], dtype=np.float64) if weight_train is None else np.asarray(
        weight_train, dtype=np.float64
    )
    if apply_mask is None:
        phi, meta = relevance_phi_sera(y_train, high_only=high_only, seed=seed)
        sera_w = floor + (1.0 - floor) * phi
        out = base * sera_w
    else:
        mask = np.asarray(apply_mask, dtype=bool)
        if mask.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"SERA apply_mask length {mask.shape[0]} != y_train {y_train.shape[0]}"
            )
        if not mask.any():
            phi, meta = relevance_phi_sera(y_train, high_only=high_only, seed=seed)
            out = base.copy()
        else:
            phi_sub, meta = relevance_phi_sera(y_train[mask], high_only=high_only, seed=seed)
            sera_w = np.ones(y_train.shape[0], dtype=np.float64)
            sera_w[mask] = floor + (1.0 - floor) * phi_sub
            out = base * sera_w
            meta["sera_frac_pos"] = float(np.mean(phi_sub > 1e-6))
            meta["sera_mean_phi"] = float(np.mean(phi_sub))
            meta["sera_n_apply"] = float(mask.sum())
    meta["sera_floor"] = floor
    meta["sera_high_only"] = float(high_only)
    meta["sera_nonreply_only"] = float(apply_mask is not None)
    return out, meta


def _spearman_nonreply_feval(is_nonreply: np.ndarray):
    """LightGBM feval: Spearman on val non-replies (higher is better)."""
    mask = np.asarray(is_nonreply, dtype=bool)

    def feval(y_pred: np.ndarray, dataset: lgb.Dataset):
        y_true = np.asarray(dataset.get_label(), dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        if mask.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"spearman_nonreply feval mask length {mask.shape[0]} != val {y_true.shape[0]}"
            )
        if not mask.any():
            return "spearman_nonreply", 0.0, True
        corr = spearmanr(y_true[mask], pred[mask]).correlation
        if corr != corr:  # NaN
            corr = 0.0
        return "spearman_nonreply", float(corr), True

    return feval


def _spearman_feval_all():
    """Spearman on the full validation set (e.g. rank subset that is already non-reply)."""

    def feval(y_pred: np.ndarray, dataset: lgb.Dataset):
        y_true = np.asarray(dataset.get_label(), dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.size < 2:
            return "spearman_nonreply", 0.0, True
        corr = spearmanr(y_true, pred).correlation
        if corr != corr:
            corr = 0.0
        return "spearman_nonreply", float(corr), True

    return feval


def prepare_account_rank_order(
    account_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_group: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Sort by account, build group sizes, and integer within-group relevance ranks.

    LightGBM LambdaRank requires int labels; higher = more relevant. Ranks are
    computed from continuous ``y`` within each account (draft-ranking groups).
    """
    account_ids = np.asarray(account_ids)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(account_ids, kind="mergesort")
    ids_sorted = account_ids[order]
    _, idx_start, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    keep_acct = counts >= int(min_group)
    if not keep_acct.any():
        raise ValueError(f"No accounts with >={min_group} rows for LambdaRank")
    keep_row = np.zeros(ids_sorted.shape[0], dtype=bool)
    for start, n, ok in zip(idx_start, counts, keep_acct, strict=True):
        if ok:
            keep_row[start : start + n] = True
    order_keep = order[keep_row]
    ids_keep = account_ids[order_keep]
    y_keep = y[order_keep]
    _, _, group = np.unique(ids_keep, return_index=True, return_counts=True)
    group = group.astype(np.int32)
    # Dense within-group ranks: higher y → higher relevance label.
    rel = np.empty(y_keep.shape[0], dtype=np.int32)
    start = 0
    for n in group:
        sl = slice(start, start + int(n))
        order_y = np.argsort(y_keep[sl], kind="mergesort")
        ranks = np.empty(int(n), dtype=np.int32)
        ranks[order_y] = np.arange(int(n), dtype=np.int32)
        rel[sl] = ranks
        start += int(n)
    meta = {
        "rank_n_rows": float(order_keep.size),
        "rank_n_groups": float(group.size),
        "rank_min_group": float(min_group),
        "rank_mean_group": float(np.mean(group)),
        "rank_frac_rows_kept": float(order_keep.size / max(account_ids.shape[0], 1)),
        "rank_label_max": float(rel.max()) if rel.size else 0.0,
    }
    return x[order_keep], y_keep, rel, group, meta


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    seed: int,
    weight_train: np.ndarray | None = None,
    *,
    loss: str = "asymmetric",
    quantile_alpha: float = 0.65,
    asymmetric_under_weight: float = 3.0,
    sera_floor: float = 0.05,
    sera_high_only: bool = True,
    sera_nonreply_only: bool = False,
    early_stop_metric: str = "rmse",
    val_is_nonreply: np.ndarray | None = None,
    train_is_nonreply: np.ndarray | None = None,
    group_train: np.ndarray | None = None,
    group_val: np.ndarray | None = None,
) -> tuple[lgb.Booster, dict[str, float]]:
    sera_meta: dict[str, float] = {}
    effective_weight = weight_train
    effective_loss = loss
    if loss == "sera":
        # SERA ∝ Σ φ(y)·(ŷ−y)² — optimize via MSE + relevance sample weights.
        apply_mask = None
        if sera_nonreply_only:
            if train_is_nonreply is None:
                raise ValueError("sera_nonreply_only requires train_is_nonreply")
            apply_mask = train_is_nonreply
        effective_weight, sera_meta = apply_sera_sample_weights(
            y_train,
            weight_train,
            sera_floor=sera_floor,
            high_only=sera_high_only,
            seed=seed,
            apply_mask=apply_mask,
        )
        effective_loss = "mse"
        scope = "nonreply-only" if sera_nonreply_only else "all"
        print(
            f"SERA relevance ({scope}): med={sera_meta['sera_med']:.4f} "
            f"upper={sera_meta['sera_upper']:.4f} "
            f"mc={sera_meta['sera_mc']:.3f} mean_phi={sera_meta['sera_mean_phi']:.3f} "
            f"frac_pos={sera_meta['sera_frac_pos']:.3f} floor={sera_floor:g}",
            flush=True,
        )

    if loss == "lambdarank":
        if group_train is None or group_val is None:
            raise ValueError("loss=lambdarank requires group_train and group_val")
        if int(np.sum(group_train)) != x_train.shape[0]:
            raise ValueError("sum(group_train) must equal n_train")
        if int(np.sum(group_val)) != x_val.shape[0]:
            raise ValueError("sum(group_val) must equal n_val")
        # Ranking objectives do not use regression sample weights.
        effective_weight = None

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=effective_weight,
        group=group_train if loss == "lambdarank" else None,
        feature_name=feature_names,
    )
    val_set = lgb.Dataset(
        x_val,
        label=y_val,
        group=group_val if loss == "lambdarank" else None,
        reference=train_set,
        feature_name=feature_names,
    )
    params: dict = {
        "learning_rate": 0.04,
        "num_leaves": 47,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
    }
    feval = None
    stop_metric = early_stop_metric
    if loss == "lambdarank" and early_stop_metric == "rmse":
        # Ranking scores are not in label units; use NDCG for selection by default.
        stop_metric = "ndcg"
        print("Note: lambdarank uses early_stop=ndcg (rmse is not meaningful for rank scores)", flush=True)

    if stop_metric == "spearman_nonreply":
        params["metric"] = "None"
        if val_is_nonreply is None:
            # Rank subset already filtered to non-replies → Spearman on all val rows.
            feval = _spearman_feval_all()
        else:
            feval = _spearman_nonreply_feval(val_is_nonreply)
    elif stop_metric == "ndcg":
        params["metric"] = "ndcg"
        params["eval_at"] = [10, 20]
    elif stop_metric == "rmse":
        params["metric"] = "rmse"
    else:
        raise ValueError(f"Unknown early_stop_metric={early_stop_metric!r}")

    if effective_loss == "huber":
        params["objective"] = "huber"
        params["alpha"] = 1.0
    elif effective_loss == "mse":
        params["objective"] = "regression"
    elif effective_loss == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = float(quantile_alpha)
    elif effective_loss == "asymmetric":
        under_w = float(asymmetric_under_weight)

        def asymmetric_mse(y_pred: np.ndarray, dataset: lgb.Dataset):
            # Custom objectives must apply Dataset weights themselves (LightGBM
            # does not multiply them into grad/hess for Python objectives).
            y_true = dataset.get_label()
            sw = dataset.get_weight()
            if sw is None:
                sw = np.ones_like(y_true, dtype=np.float64)
            else:
                sw = np.asarray(sw, dtype=np.float64)
            resid = y_true - y_pred
            w = np.where(resid > 0.0, under_w, 1.0) * sw
            grad = -2.0 * w * resid
            hess = 2.0 * w
            return grad, hess

        params["objective"] = asymmetric_mse
    elif effective_loss == "lambdarank":
        params["objective"] = "lambdarank"
        params["lambdarank_truncation_level"] = 30
        # Within-group ranks can exceed LightGBM's default label_gain length.
        max_lab = int(max(np.max(y_train), np.max(y_val))) if y_val.size else int(np.max(y_train))
        params["label_gain"] = list(range(max_lab + 1))
    else:
        raise ValueError(f"Unknown loss={loss!r}")

    print(
        f"Loss: {loss}"
        + (f" alpha={quantile_alpha:g}" if loss == "quantile" else "")
        + (f" under_weight={asymmetric_under_weight:g}" if loss == "asymmetric" else "")
        + (
            f" floor={sera_floor:g}"
            + (" nonreply_only" if sera_nonreply_only else "")
            if loss == "sera"
            else ""
        )
        + f" | early_stop={stop_metric}",
        flush=True,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1200,
        valid_sets=[val_set],
        valid_names=["val"],
        feval=feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return booster, sera_meta


def _gpboost_response_mean(pred: Any) -> np.ndarray:
    if isinstance(pred, dict):
        if "response_mean" in pred:
            return np.asarray(pred["response_mean"], dtype=np.float64)
        if "fixed_effect" in pred:
            fixed = np.asarray(pred["fixed_effect"], dtype=np.float64)
            re_mean = pred.get("random_effect_mean")
            if re_mean is not None:
                return fixed + np.asarray(re_mean, dtype=np.float64)
            return fixed
    return np.asarray(pred, dtype=np.float64)


def train_gpboost_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    group_train: np.ndarray,
    group_val: np.ndarray,
    seed: int,
    weight_train: np.ndarray | None = None,
    *,
    loss: str = "mse",
    quantile_alpha: float = 0.65,
    sera_floor: float = 0.05,
    sera_high_only: bool = True,
    sera_nonreply_only: bool = False,
    train_is_nonreply: np.ndarray | None = None,
) -> tuple[Any, Any, dict[str, float]]:
    """Train GPBoost with a grouped Gaussian random intercept on account_id.

    Custom objectives (e.g. asymmetric MSE) are not supported with gp_model;
    callers should pass a built-in loss (mse / huber / quantile) or sera
    (relevance sample weights + mse).
    """
    import gpboost as gpb

    if loss == "asymmetric":
        raise ValueError(
            "GPBoost+RE does not support custom objectives; use --loss mse|huber|quantile|sera"
        )

    sera_meta: dict[str, float] = {}
    effective_weight = weight_train
    effective_loss = loss
    if loss == "sera":
        apply_mask = None
        if sera_nonreply_only:
            if train_is_nonreply is None:
                raise ValueError("sera_nonreply_only requires train_is_nonreply")
            apply_mask = train_is_nonreply
        effective_weight, sera_meta = apply_sera_sample_weights(
            y_train,
            weight_train,
            sera_floor=sera_floor,
            high_only=sera_high_only,
            seed=seed,
            apply_mask=apply_mask,
        )
        effective_loss = "mse"
        scope = "nonreply-only" if sera_nonreply_only else "all"
        print(
            f"SERA relevance ({scope}): med={sera_meta['sera_med']:.4f} "
            f"upper={sera_meta['sera_upper']:.4f} "
            f"mc={sera_meta['sera_mc']:.3f} mean_phi={sera_meta['sera_mean_phi']:.3f} "
            f"frac_pos={sera_meta['sera_frac_pos']:.3f} floor={sera_floor:g}",
            flush=True,
        )

    gp_model = gpb.GPModel(group_data=group_train, likelihood="gaussian")
    gp_model.set_prediction_data(group_data_pred=group_val)

    train_set = gpb.Dataset(
        data=x_train,
        label=y_train,
        weight=effective_weight,
        feature_name=feature_names,
    )
    val_set = gpb.Dataset(
        data=x_val,
        label=y_val,
        reference=train_set,
        feature_name=feature_names,
    )
    params: dict = {
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 47,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        # GPBoost algorithm rejects bagging; keep feature_fraction subsample only.
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "verbose": -1,
        "seed": seed,
    }
    if effective_loss == "huber":
        params["objective"] = "huber"
        params["alpha"] = 1.0
    elif effective_loss == "mse":
        params["objective"] = "regression"
    elif effective_loss == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = float(quantile_alpha)
    else:
        raise ValueError(f"Unknown loss={loss!r}")

    print(
        f"Model: gpboost+RE | Loss: {loss}"
        + (f" alpha={quantile_alpha:g}" if loss == "quantile" else "")
        + (f" floor={sera_floor:g}" if loss == "sera" else ""),
        flush=True,
    )
    evals_result: dict = {}
    bst = gpb.train(
        params=params,
        train_set=train_set,
        gp_model=gp_model,
        num_boost_round=1200,
        valid_sets=val_set,
        valid_names=["val"],
        early_stopping_rounds=60,
        evals_result=evals_result,
        verbose_eval=100,
        use_gp_model_for_validation=True,
    )

    re_meta: dict[str, float] = {}
    try:
        cov = gp_model.get_cov_pars(format_pandas=True)
        # Typical columns: Error_var, Group_1 (RE variance).
        if hasattr(cov, "columns"):
            if "Error_var" in cov.columns:
                re_meta["gpboost_error_var"] = float(cov["Error_var"].iloc[0])
            group_cols = [c for c in cov.columns if c != "Error_var"]
            if group_cols:
                re_meta["gpboost_re_var"] = float(cov[group_cols[0]].iloc[0])
        print(f"GPBoost cov_pars: {re_meta}", flush=True)
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        print(f"Warning: could not read GPBoost cov_pars ({exc})", flush=True)

    n_train_groups = int(pd.unique(group_train).size)
    n_val_groups = int(pd.unique(group_val).size)
    n_cold = int(len(set(map(str, group_val)) - set(map(str, group_train))))
    re_meta["gpboost_n_train_groups"] = float(n_train_groups)
    re_meta["gpboost_n_val_groups"] = float(n_val_groups)
    re_meta["gpboost_n_cold_val_groups"] = float(n_cold)
    re_meta.update(sera_meta)
    print(
        f"GPBoost groups: train={n_train_groups} val={n_val_groups} "
        f"cold_val={n_cold} (population-level RE for unseen)",
        flush=True,
    )
    return bst, gp_model, re_meta


def train_breakout_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    seed: int,
    weight_train: np.ndarray | None = None,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weight_train,
        feature_name=feature_names,
    )
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, feature_name=feature_names)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=600,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def add_breakout_probs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    baseline_kind: str,
    skip_embeddings: bool,
    seed: int,
    n_folds: int = 3,
    nonreply_weight: float = 1.0,
    mode: str = "vs_baseline",
    quantile: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """OOF P(breakout) on train; full-fit probs on val.

    mode=vs_baseline: log_eng > leakage-safe author baseline (often too frequent).
    mode=vs_quantile: log_eng >= train-fold quantile (default 0.8).
    """
    from sklearn.model_selection import KFold

    y_train_abs = train_df["log_engagement"].to_numpy(dtype=np.float64)
    y_val_abs = val_df["log_engagement"].to_numpy(dtype=np.float64)
    if mode == "vs_baseline":
        base_train = author_baseline(train_df, baseline_kind)
        base_val = author_baseline(val_df, baseline_kind)
        y_break_train = (y_train_abs > base_train).astype(np.float64)
        y_break_val = (y_val_abs > base_val).astype(np.float64)
    elif mode == "vs_quantile":
        thr = float(np.quantile(y_train_abs, quantile))
        y_break_train = (y_train_abs >= thr).astype(np.float64)
        y_break_val = (y_val_abs >= thr).astype(np.float64)
        print(f"Breakout quantile threshold (train q={quantile}): {thr:.4f}")
    else:
        raise ValueError(mode)

    x_all, feature_names = feature_matrix(train_df, skip_embeddings=skip_embeddings)
    x_val, _ = feature_matrix(val_df, skip_embeddings=skip_embeddings)
    weights = np.where(train_df["is_reply"].to_numpy() == 0, nonreply_weight, 1.0).astype(np.float64)

    oof = np.zeros(len(train_df), dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_all)):
        clf = train_breakout_classifier(
            x_all[tr_idx],
            y_break_train[tr_idx],
            x_all[va_idx],
            y_break_train[va_idx],
            feature_names,
            seed=seed + fold,
            weight_train=weights[tr_idx],
        )
        oof[va_idx] = clf.predict(x_all[va_idx], num_iteration=clf.best_iteration)

    clf_full = train_breakout_classifier(
        x_all,
        y_break_train,
        x_val,
        y_break_val,
        feature_names,
        seed=seed + 99,
        weight_train=weights,
    )
    p_val = clf_full.predict(x_val, num_iteration=clf_full.best_iteration)
    # Quick val AUC for logging.
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y_break_val, p_val)) if y_break_val.min() != y_break_val.max() else float("nan")
    print(
        f"Breakout head: rate_train={y_break_train.mean():.3f} "
        f"rate_val={y_break_val.mean():.3f} val_auc={auc:.3f}"
    )

    train_out = train_df.copy()
    val_out = val_df.copy()
    train_out["p_breakout"] = oof.astype(np.float32)
    val_out["p_breakout"] = p_val.astype(np.float32)
    return train_out, val_out, auc


def append_results(description: str, metrics: dict) -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "rmse_log\tspearman\tr2_log\ttrain_rows\tval_rows\tdescription\n"
        )
    with RESULTS_PATH.open("a") as f:
        f.write(
            f"{metrics['rmse_log']:.6f}\t{metrics['spearman']:.6f}\t"
            f"{metrics['r2_log']:.6f}\t{metrics['train_rows']}\t"
            f"{metrics['val_rows']}\t{description}\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        choices=("lgbm", "gpboost"),
        default="lgbm",
        help="Booster backend. gpboost adds a Gaussian random intercept on account_id",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap after time-sort (smoke tests); 0 = full dataset",
    )
    p.add_argument(
        "--holdout-policy",
        choices=("time", "upload_era", "stream_only"),
        default="time",
        help="Regime filter before the global time split: upload_era caps at the "
        "latest upload-linked tweet; stream_only trains/evals on extension rows only",
    )
    p.add_argument(
        "--min-upload-hist",
        type=int,
        default=0,
        help="Drop rows whose author has fewer than N prior upload-linked tweets (0=off)",
    )
    p.add_argument(
        "--provenance-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add is_stream / upload-end / provenance×age features (default on; "
        "v36c: +wa_nr/+wa_rp vs same-rebuild; --no-provenance-features to ablate)",
    )
    p.add_argument(
        "--stream-fresh-days",
        type=float,
        default=0.0,
        help="Down-weight stream rows with dump-age < this many days (0=off); "
        "linear ramp from --stream-fresh-floor to 1.0. Under a time holdout, "
        "must exceed the val-window age span or no train rows are affected",
    )
    p.add_argument(
        "--stream-fresh-floor",
        type=float,
        default=0.25,
        help="Minimum sample-weight multiplier for brand-new stream rows",
    )
    p.add_argument(
        "--media-types",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CA tweet_media type breakdown (photo/video/gif); "
        "requires prepare.py --media-only (off by default until promoted)",
    )
    p.add_argument(
        "--residual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Residualize against --baseline. Default: on for dump-total targets, "
        "off for --exposure-mode rate (mixed units)",
    )
    p.add_argument("--skip-embeddings", action="store_true", help="Ablation: tabular features only")
    p.add_argument(
        "--split-reply",
        action="store_true",
        help="Train separate models for replies vs non-replies",
    )
    p.add_argument(
        "--emb-pca",
        type=int,
        default=96,
        help="PCA-reduce emb_* to this many dims (fit on train only); 0 disables",
    )
    p.add_argument(
        "--text-svd",
        type=int,
        default=128,
        help="Train-fit TF-IDF + TruncatedSVD dims from full_text; 0 disables",
    )
    p.add_argument(
        "--baseline",
        choices=("hist_replies", "hist_decay_replies", "context_replies"),
        default="context_replies",
        help="Leakage-safe author baseline (pre-T reply rates); used only with --residual",
    )
    p.add_argument(
        "--nonreply-weight",
        type=float,
        default=1.0,
        help="Sample weight for original (non-reply) tweets during training",
    )
    p.add_argument(
        "--breakout-head",
        action="store_true",
        help="Add OOF P(breakout) as a regressor feature",
    )
    p.add_argument(
        "--breakout-mode",
        choices=("vs_baseline", "vs_quantile"),
        default="vs_quantile",
        help="Breakout label definition for --breakout-head",
    )
    p.add_argument(
        "--breakout-quantile",
        type=float,
        default=0.8,
        help="Train quantile for --breakout-mode vs_quantile",
    )
    p.add_argument(
        "--nonreply-only",
        action="store_true",
        help="Train regressor on non-replies only; replies use baseline + mean reply residual",
    )
    p.add_argument(
        "--author-emb-affinity",
        action="store_true",
        help="Add pre-T mean-author MiniLM cosine/L2 features (flat in v10; off by default)",
    )
    p.add_argument(
        "--interlocutor-emb-affinity",
        action="store_true",
        help="Add pre-T inbound interlocutor MiniLM affinity (reply/quote texts to "
        "the author by default; sweet-spot + 30d). Builds interlocutor_cache via "
        "prepare helpers; off by default until promoted",
    )
    p.add_argument(
        "--reply-dyad",
        action="store_true",
        help="Include reply/quote dyad + parent-author size/corpus MiniLM features "
        "(prepare.py --dyad-only or full prepare). Opt-in until smoke clears gate",
    )
    p.add_argument(
        "--parent-chain",
        action="store_true",
        help="Include 2nd-hop reply/quote chain features (quote-of-quote / "
        "reply-of-reply; prepare.py --parent-chain-only). Opt-in until smoke clears gate",
    )
    p.add_argument(
        "--force-interlocutor",
        action="store_true",
        help="With --interlocutor-emb-affinity, rescan dump interlocutor events",
    )
    p.add_argument(
        "--interlocutor-include-mentions",
        action="store_true",
        help="Pool inbound @-mention tweets into interlocutor affinity (off by default)",
    )
    p.add_argument(
        "--interlocutor-sweet-cos",
        type=float,
        default=0.5,
        help="Target cosine for interlocutor_sweet_dist = (cos - sweet)^2",
    )
    p.add_argument(
        "--descriptive-text",
        action="store_true",
        help="Add VADER sentiment + first-person ratio (v15: flat rate RMSE; off by default)",
    )
    p.add_argument(
        "--provoke-labels",
        action="store_true",
        help="Join LM wrongheaded/infuriating scores from provoke cache",
    )
    p.add_argument(
        "--provoke-cache",
        type=Path,
        default=None,
        help="Path to provoke cache parquet (default: ~/.cache/will-it-bang/provoke_cache.parquet)",
    )
    p.add_argument(
        "--stance-labels",
        action="store_true",
        help="Join LM agree/disagree-vs-parent scores from stance cache "
        "(label_stance.py; opt-in until smoke clears gate)",
    )
    p.add_argument(
        "--stance-cache",
        type=Path,
        default=None,
        help="Path to stance cache parquet "
        "(default: ~/.cache/will-it-bang/stance_cache.parquet)",
    )
    p.add_argument(
        "--desc-labels",
        action="store_true",
        help="Join richer LM descriptive tags (humor/controversy/narrative/cta/news/"
        "dunk_bait) from desc cache (label_desc.py; opt-in until smoke clears gate)",
    )
    p.add_argument(
        "--desc-cache",
        type=Path,
        default=None,
        help="Path to desc cache parquet "
        "(default: ~/.cache/will-it-bang/desc_cache.parquet)",
    )
    p.add_argument(
        "--account-te",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train-fold smoothed account mean of the model target (LOO on train)",
    )
    p.add_argument(
        "--account-te-prior",
        type=float,
        default=20.0,
        help="Shrinkage strength toward global mean for --account-te",
    )
    p.add_argument(
        "--era-te",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train-fold smoothed mean target by --era-key (+ key×reply); "
        "opt-in calendar prior (v46)",
    )
    p.add_argument(
        "--era-te-prior",
        type=float,
        default=50.0,
        help="Shrinkage strength toward global mean for --era-te",
    )
    p.add_argument(
        "--era-heads",
        action="store_true",
        help="Fit a specialist model per --era-key bucket (fallback to global when "
        "train n < --era-head-min-train); opt-in (v46)",
    )
    p.add_argument(
        "--era-key",
        choices=("platform_era", "year"),
        default="platform_era",
        help="Regime column for --era-te / --era-heads (year is denser under "
        "time-holdout musk_late dominance)",
    )
    p.add_argument(
        "--era-head-min-train",
        type=int,
        default=1500,
        help="Min train rows required to fit an era specialist under --era-heads",
    )
    p.add_argument(
        "--loss",
        choices=("huber", "mse", "quantile", "asymmetric", "sera", "lambdarank"),
        default="asymmetric",
        help="Objective (default asymmetric; sera = extreme relevance weights; "
        "lambdarank = account-grouped listwise rank on non-replies)",
    )
    p.add_argument(
        "--rank-nonreply-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --loss lambdarank, train on non-replies only (default on)",
    )
    p.add_argument(
        "--rank-min-group",
        type=int,
        default=2,
        help="Min tweets per account_id group for --loss lambdarank / --hybrid-rank",
    )
    p.add_argument(
        "--hybrid-rank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit level model + stratum LambdaRank (NR + reply); blend keeps "
        "per-account level mean/std and imposes ranker order (default on; v31b)",
    )
    p.add_argument(
        "--quantile-alpha",
        type=float,
        default=0.65,
        help="Quantile level when --loss quantile",
    )
    p.add_argument(
        "--asymmetric-under-weight",
        type=float,
        default=3.0,
        help="Under-prediction weight when --loss asymmetric",
    )
    p.add_argument(
        "--sera-floor",
        type=float,
        default=0.05,
        help="Minimum relevance weight when --loss sera (φ_eff = floor+(1-floor)·φ)",
    )
    p.add_argument(
        "--sera-high-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SERA relevance on high extremes only (default); --no-sera-high-only uses both tails",
    )
    p.add_argument(
        "--sera-nonreply-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --loss sera, fit/apply φ on non-replies only; replies keep base weight",
    )
    p.add_argument(
        "--early-stop-metric",
        choices=("rmse", "spearman_nonreply", "ndcg"),
        default="rmse",
        help="Val metric for LGBM early stopping / best iteration (default rmse; "
        "spearman_nonreply aligns with draft-ranking primary; ndcg for lambdarank)",
    )
    p.add_argument(
        "--emb-knn-te",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train-fold MiniLM kNN target encoding (cold-start content prior; default on)",
    )
    p.add_argument("--emb-knn-k", type=int, default=40, help="Neighbors for --emb-knn-te")
    p.add_argument(
        "--emb-knn-cold-max-n",
        type=float,
        default=5.0,
        help="Train account count ≤ this counts as cold for knn_x_cold",
    )
    p.add_argument(
        "--emb-cav",
        action="store_true",
        help="Add train-fold MiniLM engagement CAV projection (μ_hot−μ_cold; fit before emb PCA)",
    )
    p.add_argument(
        "--emb-cav-quantile",
        type=float,
        default=0.1,
        help="Tail mass for hot/cold means when --emb-cav (0.1 → top/bottom decile)",
    )
    p.add_argument(
        "--emb-cav-nonreply-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit CAV direction on non-replies only (default on; still project all rows)",
    )
    p.add_argument(
        "--emb-pls",
        type=int,
        default=0,
        help="Train-fold PLS of raw MiniLM→target; add this many pls_* dims "
        "(fit before emb PCA; 0 disables). Use with --emb-pca 0 to replace variance PCA",
    )
    p.add_argument(
        "--residual-emb-pls",
        type=int,
        default=0,
        help="OLS-residualize MiniLM vs account TE / reply baselines, then train-fold "
        "PLS→target; add this many res_pls_* dims (fit before emb PCA; 0 disables)",
    )
    p.add_argument(
        "--residual-emb-cav",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CAV on MiniLM residualized vs account TE / reply baselines "
        "(res_emb_cav_proj; uses --emb-cav-quantile / --emb-cav-nonreply-only; "
        "default on after v47; --no-residual-emb-cav to ablate)",
    )
    p.add_argument(
        "--fail-clusters",
        action="store_true",
        help="Cluster train OOF under-prediction MiniLM vectors; add fail_min_dist / fail_dist_*",
    )
    p.add_argument(
        "--fail-clusters-k",
        type=int,
        default=4,
        help="K-means centroids for --fail-clusters",
    )
    p.add_argument(
        "--fail-clusters-quantile",
        type=float,
        default=0.1,
        help="Top under-pred mass used as failure set (0.1 → worst 10%%)",
    )
    p.add_argument(
        "--fail-clusters-mode",
        choices=("oof", "te"),
        default="oof",
        help="Under-pred definition: 3-fold LGBM OOF (no emb) or vs account_mean_rate",
    )
    p.add_argument(
        "--fail-clusters-nonreply-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build failure set from non-replies only (default on; still score all rows)",
    )
    p.add_argument(
        "--emb-cluster-te",
        action="store_true",
        help="Train-fold MiniLM k-means + smoothed cluster mean rate (fit before emb PCA)",
    )
    p.add_argument(
        "--emb-cluster-k",
        type=int,
        default=32,
        help="K-means clusters for --emb-cluster-te",
    )
    p.add_argument(
        "--emb-cluster-te-prior",
        type=float,
        default=20.0,
        help="Shrinkage strength toward global mean for --emb-cluster-te",
    )
    p.add_argument(
        "--exposure-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train/eval maturity-adjusted intensity (default on). "
        "Use --no-exposure-correct for dump-total log1p(eng)",
    )
    p.add_argument(
        "--exposure-mode",
        choices=("rate", "resid"),
        default="rate",
        help="rate: log1p(eng/(min(age,cap)+eps)^beta) [default]; "
        "resid: subtract train-fit a+b*log1p(eff_age)",
    )
    p.add_argument(
        "--exposure-cap-days",
        type=float,
        default=30.0,
        help="Saturating exposure cap in days (with --exposure-correct)",
    )
    p.add_argument(
        "--exposure-beta",
        type=float,
        default=1.0,
        help="Exponent on effective age when --exposure-mode rate",
    )
    p.add_argument(
        "--exposure-eps",
        type=float,
        default=1.0,
        help="Smoothing days added to effective age",
    )
    p.add_argument("--description", type=str, default="exposure-rate hybrid")
    args = p.parse_args()
    # GPBoost already models account_id as a RE — default off LOO TE to avoid
    # double-counting author mean unless the user explicitly passes --account-te.
    if (
        args.model == "gpboost"
        and "--account-te" not in sys.argv
        and "--no-account-te" not in sys.argv
    ):
        args.account_te = False
    # Custom objectives are unsupported with gp_model; fall back to MSE.
    if args.model == "gpboost" and args.loss == "asymmetric":
        print(
            "Note: GPBoost+RE does not support custom objectives; "
            "using --loss mse instead of asymmetric",
            flush=True,
        )
        args.loss = "mse"
    if args.model == "gpboost" and args.loss == "lambdarank":
        raise SystemExit("GPBoost does not support --loss lambdarank; use --model lgbm")
    if args.model == "gpboost" and args.hybrid_rank:
        raise SystemExit("GPBoost does not support --hybrid-rank; use --model lgbm")
    if args.hybrid_rank and args.loss == "lambdarank":
        raise SystemExit("Use --hybrid-rank with a level loss (e.g. asymmetric), not --loss lambdarank")
    if args.era_heads and args.split_reply:
        raise SystemExit("--era-heads is incompatible with --split-reply")
    if args.era_heads and args.nonreply_only:
        raise SystemExit("--era-heads is incompatible with --nonreply-only")
    if args.era_heads and args.model == "gpboost":
        raise SystemExit("--era-heads requires --model lgbm")
    return args


def _exposure_eff_days(
    created_at: pd.Series,
    dump_time: pd.Timestamp,
    cap_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(created_at, utc=True)
    age_days = (dump_time - ts).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    age_days = np.clip(age_days, 0.0, None)
    eff = np.minimum(age_days, float(cap_days))
    return age_days, eff


def fit_exposure_resid(
    log_eng: np.ndarray,
    eff_days: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    """OLS log_eng ~ a + b * log1p(eff) on the train fold."""
    x = np.log1p(eff_days + float(eps))
    x_c = x - x.mean()
    y_c = log_eng - log_eng.mean()
    var = float(np.dot(x_c, x_c))
    b = float(np.dot(x_c, y_c) / var) if var > 1e-12 else 0.0
    a = float(log_eng.mean() - b * x.mean())
    return a, b


def exposure_resid_baseline(eff_days: np.ndarray, a: float, b: float, eps: float) -> np.ndarray:
    return a + b * np.log1p(eff_days + float(eps))


def apply_exposure_rate_targets(
    df: pd.DataFrame,
    *,
    dump_time: pd.Timestamp,
    cap_days: float,
    beta: float,
    eps: float,
) -> pd.DataFrame:
    """Replace log_engagement with saturating exposure-adjusted log rate."""
    out = df.copy()
    age_days, eff = _exposure_eff_days(out["created_at"], dump_time, cap_days)
    log_raw = out["log_engagement"].to_numpy(dtype=np.float64)
    eng = np.expm1(np.clip(log_raw, 0.0, 30.0))
    log_rate = np.log1p(eng / np.power(eff + float(eps), float(beta)))
    out["log_engagement_raw"] = log_raw
    out["exposure_age_days"] = age_days
    out["exposure_eff_days"] = eff
    out["log_engagement"] = log_rate
    print(
        f"Exposure rate: dump={dump_time.date()} cap={cap_days:g}d "
        f"beta={beta:g} eps={eps:g} | "
        f"age p50={np.median(age_days):.1f} eff p50={np.median(eff):.1f} | "
        f"log_raw mean={log_raw.mean():.3f} → log_rate mean={log_rate.mean():.3f}",
        flush=True,
    )
    return out


def log_rate_to_log_eng(
    log_rate: np.ndarray,
    eff_days: np.ndarray,
    *,
    beta: float,
    eps: float,
) -> np.ndarray:
    rate_eng = np.expm1(np.clip(log_rate, -1.0, 30.0))
    eng = np.clip(rate_eng, 0.0, None) * np.power(eff_days + float(eps), float(beta))
    return np.log1p(eng)


def author_baseline(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Leakage-safe author baseline (pre-T reply-received rates on prior tweets)."""
    if kind == "hist_replies":
        return df["hist_mean_replies"].to_numpy(dtype=np.float64)
    if kind == "hist_decay_replies":
        return df["hist_decay_mean_replies"].to_numpy(dtype=np.float64)
    if kind == "context_replies":
        reply = df["is_reply"].to_numpy(dtype=bool)
        out = df["hist_mean_replies_nonreply"].to_numpy(dtype=np.float64).copy()
        out[reply] = df.loc[reply, "hist_mean_replies_reply"].to_numpy(dtype=np.float64)
        return out
    raise ValueError(kind)


_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|i’ve|i've|me|my|mine|we|we're|we’ve|we've|us|our|ours)\b",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def add_provoke_features(df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    """Join cached LM tone scores (text-only; no engagement in the labeler)."""
    path = cache_path or (CACHE_DIR / "provoke_cache.parquet")
    if not path.exists():
        raise SystemExit(f"Missing {path}; run: uv run modal run label_provoke.py")
    cache = pd.read_parquet(path)
    cache["tweet_id"] = cache["tweet_id"].astype(str)
    cache = cache.drop_duplicates("tweet_id", keep="last")
    if "ok" in cache.columns:
        cache = cache.loc[cache["ok"].fillna(0).astype(int) == 1]
    ok = cache[["tweet_id", "wrongheaded", "infuriating"]]
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    merged = out[["tweet_id"]].merge(ok, on="tweet_id", how="left")
    wh = merged["wrongheaded"].to_numpy(dtype=np.float64)
    inf = merged["infuriating"].to_numpy(dtype=np.float64)
    has = np.isfinite(wh) & np.isfinite(inf)
    # Scale 0-10 → 0-1; missing → 0 with has_provoke=0.
    wh_s = np.where(has, wh / 10.0, 0.0)
    inf_s = np.where(has, inf / 10.0, 0.0)
    out["wrongheaded"] = wh_s.astype(np.float32)
    out["infuriating"] = inf_s.astype(np.float32)
    out["provoke"] = np.maximum(wh_s, inf_s).astype(np.float32)
    out["has_provoke"] = has.astype(np.int8)
    print(
        f"Provoke labels: coverage={has.mean():.1%} "
        f"mean wrongheaded={wh_s[has].mean() if has.any() else float('nan'):.3f} "
        f"infuriating={inf_s[has].mean() if has.any() else float('nan'):.3f}",
        flush=True,
    )
    return out


def add_stance_features(df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    """Join cached LM agree/disagree-vs-parent scores (text-only)."""
    path = cache_path or (CACHE_DIR / "stance_cache.parquet")
    if not path.exists():
        # Fall back to any stance_cache_*.parquet (model-slugged default writes).
        cands = sorted(CACHE_DIR.glob("stance_cache_*.parquet"))
        if not cands:
            raise SystemExit(
                f"Missing {path}; run: uv run python label_stance.py --local --limit 500"
            )
        path = cands[-1]
        print(f"Stance cache: using {path.name}", flush=True)
    cache = pd.read_parquet(path)
    cache["tweet_id"] = cache["tweet_id"].astype(str)
    cache = cache.drop_duplicates("tweet_id", keep="last")
    if "ok" in cache.columns:
        cache = cache.loc[cache["ok"].fillna(0).astype(int) == 1]
    ok = cache[["tweet_id", "agree", "disagree"]]
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    merged = out[["tweet_id"]].merge(ok, on="tweet_id", how="left")
    ag = merged["agree"].to_numpy(dtype=np.float64)
    dg = merged["disagree"].to_numpy(dtype=np.float64)
    has = np.isfinite(ag) & np.isfinite(dg)
    ag_s = np.where(has, ag / 10.0, 0.0)
    dg_s = np.where(has, dg / 10.0, 0.0)
    out["stance_agree"] = ag_s.astype(np.float32)
    out["stance_disagree"] = dg_s.astype(np.float32)
    out["stance_net"] = (ag_s - dg_s).astype(np.float32)
    out["stance_conflict"] = np.minimum(ag_s, dg_s).astype(np.float32)
    out["has_stance"] = has.astype(np.int8)
    print(
        f"Stance labels: coverage={has.mean():.1%} "
        f"mean agree={ag_s[has].mean() if has.any() else float('nan'):.3f} "
        f"disagree={dg_s[has].mean() if has.any() else float('nan'):.3f} "
        f"net={(ag_s - dg_s)[has].mean() if has.any() else float('nan'):.3f}",
        flush=True,
    )
    return out


_DESC_TAG_COLS = (
    "humor",
    "controversy",
    "narrative",
    "cta",
    "news",
    "dunk_bait",
)


def add_desc_features(df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    """Join cached richer LM descriptive tags (text-only; no fame/quality)."""
    path = cache_path or (CACHE_DIR / "desc_cache.parquet")
    if not path.exists():
        cands = sorted(CACHE_DIR.glob("desc_cache_*.parquet"))
        if not cands:
            raise SystemExit(
                f"Missing {path}; run: uv run python label_desc.py --local --limit 500"
            )
        path = cands[-1]
        print(f"Desc cache: using {path.name}", flush=True)
    cache = pd.read_parquet(path)
    cache["tweet_id"] = cache["tweet_id"].astype(str)
    cache = cache.drop_duplicates("tweet_id", keep="last")
    if "ok" in cache.columns:
        cache = cache.loc[cache["ok"].fillna(0).astype(int) == 1]
    missing = [c for c in _DESC_TAG_COLS if c not in cache.columns]
    if missing:
        raise SystemExit(f"Desc cache missing columns {missing}: {path}")
    ok = cache[["tweet_id", *_DESC_TAG_COLS]]
    out = df.copy()
    out["tweet_id"] = out["tweet_id"].astype(str)
    merged = out[["tweet_id"]].merge(ok, on="tweet_id", how="left")
    mats = [merged[c].to_numpy(dtype=np.float64) for c in _DESC_TAG_COLS]
    has = np.ones(len(merged), dtype=bool)
    for m in mats:
        has &= np.isfinite(m)
    scaled = [np.where(has, m / 10.0, 0.0).astype(np.float32) for m in mats]
    for name, arr in zip(_DESC_TAG_COLS, scaled, strict=True):
        out[f"desc_{name}"] = arr
    stacked = np.vstack(scaled)
    out["desc_max"] = stacked.max(axis=0).astype(np.float32)
    out["has_desc"] = has.astype(np.int8)
    means = " ".join(
        f"{n}={scaled[i][has].mean():.3f}" if has.any() else f"{n}=nan"
        for i, n in enumerate(_DESC_TAG_COLS)
    )
    print(f"Desc labels: coverage={has.mean():.1%} mean {means}", flush=True)
    return out


def add_descriptive_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lexical sentiment + light descriptive text stats (no user/handle LM scoring)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    out = df.copy()
    texts = out["full_text"].fillna("").astype(str)
    analyzer = SentimentIntensityAnalyzer()
    neg = np.zeros(len(out), dtype=np.float64)
    neu = np.zeros(len(out), dtype=np.float64)
    pos = np.zeros(len(out), dtype=np.float64)
    compound = np.zeros(len(out), dtype=np.float64)
    first_person = np.zeros(len(out), dtype=np.float64)
    n_sent = np.zeros(len(out), dtype=np.float64)

    for i, text in enumerate(texts):
        s = analyzer.polarity_scores(text)
        neg[i] = s["neg"]
        neu[i] = s["neu"]
        pos[i] = s["pos"]
        compound[i] = s["compound"]
        toks = re.findall(r"[A-Za-z']+", text)
        if toks:
            first_person[i] = sum(1 for t in toks if _FIRST_PERSON_RE.fullmatch(t)) / len(toks)
        parts = [p for p in _SENT_SPLIT_RE.split(text) if p.strip()]
        n_sent[i] = float(max(len(parts), 1 if text.strip() else 0))

    out["sent_neg"] = neg
    out["sent_neu"] = neu
    out["sent_pos"] = pos
    out["sent_compound"] = compound
    out["sent_abs_compound"] = np.abs(compound)
    out["first_person_ratio"] = first_person
    out["n_sentences"] = n_sent
    print(
        f"Descriptive text features: mean compound={compound.mean():.3f}, "
        f"mean |compound|={np.abs(compound).mean():.3f}, "
        f"mean first_person={first_person.mean():.3f}",
        flush=True,
    )
    return out


def add_domain_priors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    min_count: int = 30,
    target_col: str = "log_engagement",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-fold mean target by expanded domain; rare/unknown → global mean."""
    if "url_domain" not in train_df.columns:
        return train_df, val_df
    train_out = train_df.copy()
    val_out = val_df.copy()
    domain_tr = train_out["url_domain"].fillna("").astype(str)
    y = train_out[target_col].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y))
    stats = (
        pd.DataFrame({"domain": domain_tr.to_numpy(), "y": y})
        .loc[lambda d: d["domain"] != ""]
        .groupby("domain", sort=False)["y"]
        .agg(["mean", "count"])
    )
    stats = stats.loc[stats["count"] >= min_count]
    mean_map = stats["mean"].to_dict()
    count_map = stats["count"].to_dict()

    def _encode(domains: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        means = np.empty(len(domains), dtype=np.float32)
        log_n = np.empty(len(domains), dtype=np.float32)
        for i, d in enumerate(domains.fillna("").astype(str)):
            if d in mean_map:
                means[i] = mean_map[d]
                log_n[i] = np.log1p(count_map[d])
            else:
                means[i] = global_mean
                log_n[i] = 0.0
        return means, log_n

    train_out["domain_mean_rate"], train_out["log_domain_n"] = _encode(domain_tr)
    val_out["domain_mean_rate"], val_out["log_domain_n"] = _encode(val_out["url_domain"])
    print(
        f"Domain priors: {len(mean_map):,} domains with n>={min_count} "
        f"(coverage train={(domain_tr != '').mean():.1%}, "
        f"mapped={(domain_tr.map(lambda d: d in mean_map)).mean():.1%})",
        flush=True,
    )
    return train_out, val_out


def add_account_target_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    prior_strength: float = 20.0,
    target_col: str = "log_engagement",
    id_col: str = "account_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-fold smoothed mean target by account; LOO on train to avoid row leak."""
    if id_col not in train_df.columns:
        return train_df, val_df
    train_out = train_df.copy()
    val_out = val_df.copy()
    ids_tr = train_out[id_col].astype(str)
    y = train_out[target_col].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y))
    m = float(prior_strength)

    grp = pd.DataFrame({"aid": ids_tr.to_numpy(), "y": y}).groupby("aid", sort=False)["y"]
    sums = grp.sum()
    counts = grp.count().astype(np.float64)
    means = grp.mean()

    # Train: leave-one-out mean, then shrink toward global.
    n = counts.reindex(ids_tr).to_numpy(dtype=np.float64)
    s = sums.reindex(ids_tr).to_numpy(dtype=np.float64)
    loo = np.where(n > 1.0, (s - y) / (n - 1.0), global_mean)
    n_loo = np.maximum(n - 1.0, 0.0)
    train_te = (n_loo * loo + m * global_mean) / (n_loo + m)
    train_out["account_mean_rate"] = train_te.astype(np.float32)
    train_out["log_account_n"] = np.log1p(n).astype(np.float32)

    # Val: full train-fold mean + same shrinkage; unseen accounts → global.
    ids_va = val_out[id_col].astype(str)
    n_va = counts.reindex(ids_va).fillna(0.0).to_numpy(dtype=np.float64)
    mean_va = means.reindex(ids_va).fillna(global_mean).to_numpy(dtype=np.float64)
    val_te = (n_va * mean_va + m * global_mean) / (n_va + m)
    val_out["account_mean_rate"] = val_te.astype(np.float32)
    val_out["log_account_n"] = np.log1p(n_va).astype(np.float32)

    overlap = float(ids_va.isin(set(ids_tr)).mean())
    print(
        f"Account TE: prior_m={m:g} global={global_mean:.4f} "
        f"train accounts={len(counts):,} val overlap={overlap:.1%} "
        f"val te p50={np.median(val_te):.4f}",
        flush=True,
    )
    return train_out, val_out


def add_era_target_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    prior_strength: float = 50.0,
    target_col: str = "log_engagement",
    era_col: str = "platform_era",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-fold smoothed mean target by platform era (+ era×reply stratum).

    LOO on train avoids row leak. On a mono-era val fold the era mean is nearly
    constant, so this mainly calibrates level / RMSE; within-account rank needs
    --era-heads. Cross with account TE via te_minus_era when both are present.
    """
    if era_col not in train_df.columns:
        print(f"Skipping era TE (missing {era_col})", flush=True)
        return train_df, val_df
    train_out = train_df.copy()
    val_out = val_df.copy()
    y = train_out[target_col].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y))
    m = float(prior_strength)
    era_tr = train_out[era_col].astype(np.int16)
    reply_tr = train_out["is_reply"].astype(np.int8) if "is_reply" in train_out.columns else 0

    def _loo_shrink(keys: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        grp = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k", sort=False)["y"]
        sums = grp.sum()
        counts = grp.count().astype(np.float64)
        means = grp.mean()
        n = counts.reindex(keys).to_numpy(dtype=np.float64)
        s = sums.reindex(keys).to_numpy(dtype=np.float64)
        loo = np.where(n > 1.0, (s - y) / (n - 1.0), global_mean)
        n_loo = np.maximum(n - 1.0, 0.0)
        te = (n_loo * loo + m * global_mean) / (n_loo + m)
        return te.astype(np.float64), n, counts, means

    era_te, era_n, era_counts, era_means = _loo_shrink(era_tr)
    train_out["era_mean_rate"] = era_te.astype(np.float32)
    train_out["log_era_n"] = np.log1p(era_n).astype(np.float32)

    stratum_tr = era_tr.astype(str) + "_" + pd.Series(reply_tr, index=train_out.index).astype(str)
    st_te, st_n, st_counts, st_means = _loo_shrink(stratum_tr)
    train_out["era_stratum_mean_rate"] = st_te.astype(np.float32)
    train_out["log_era_stratum_n"] = np.log1p(st_n).astype(np.float32)

    era_va = val_out[era_col].astype(np.int16)
    n_va = era_counts.reindex(era_va).fillna(0.0).to_numpy(dtype=np.float64)
    mean_va = era_means.reindex(era_va).fillna(global_mean).to_numpy(dtype=np.float64)
    val_era_te = (n_va * mean_va + m * global_mean) / (n_va + m)
    val_out["era_mean_rate"] = val_era_te.astype(np.float32)
    val_out["log_era_n"] = np.log1p(n_va).astype(np.float32)

    reply_va = val_out["is_reply"].astype(np.int8) if "is_reply" in val_out.columns else 0
    stratum_va = era_va.astype(str) + "_" + pd.Series(reply_va, index=val_out.index).astype(str)
    n_st = st_counts.reindex(stratum_va).fillna(0.0).to_numpy(dtype=np.float64)
    mean_st = st_means.reindex(stratum_va).fillna(global_mean).to_numpy(dtype=np.float64)
    val_st_te = (n_st * mean_st + m * global_mean) / (n_st + m)
    val_out["era_stratum_mean_rate"] = val_st_te.astype(np.float32)
    val_out["log_era_stratum_n"] = np.log1p(n_st).astype(np.float32)

    if "account_mean_rate" in train_out.columns:
        train_out["te_minus_era"] = (
            train_out["account_mean_rate"].to_numpy(dtype=np.float64) - era_te
        ).astype(np.float32)
        val_out["te_minus_era"] = (
            val_out["account_mean_rate"].to_numpy(dtype=np.float64) - val_era_te
        ).astype(np.float32)

    era_vc = era_tr.value_counts().sort_index()
    print(
        f"Era TE: prior_m={m:g} global={global_mean:.4f} "
        f"train eras={ {int(k): int(v) for k, v in era_vc.items()} } "
        f"val era_mean p50={np.median(val_era_te):.4f} "
        f"val stratum_mean p50={np.median(val_st_te):.4f}",
        flush=True,
    )
    return train_out, val_out


def add_te_affinity_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cross account TE with own-corpus novelty (breakout / off-brand signal)."""
    if "author_emb_cos" not in df.columns or "account_mean_rate" not in df.columns:
        return df
    out = df.copy()
    cos = out["author_emb_cos"].to_numpy(dtype=np.float64)
    nov = np.clip(1.0 - cos, 0.0, 2.0)
    te = out["account_mean_rate"].to_numpy(dtype=np.float64)
    l2 = out["author_emb_l2"].to_numpy(dtype=np.float64)
    out["author_emb_novelty"] = nov.astype(np.float32)
    out["author_emb_novelty_sq"] = (nov * nov).astype(np.float32)
    out["te_x_novelty"] = (te * nov).astype(np.float32)
    out["te_x_author_l2"] = (te * l2).astype(np.float32)
    if "author_emb_cos_30d" in out.columns:
        nov30 = np.clip(1.0 - out["author_emb_cos_30d"].to_numpy(dtype=np.float64), 0.0, 2.0)
        out["te_x_novelty_30d"] = (te * nov30).astype(np.float32)
    if "is_reply" in out.columns:
        nr = 1.0 - out["is_reply"].to_numpy(dtype=np.float64)
        out["novelty_x_nonreply"] = (nov * nr).astype(np.float32)
    print(
        f"TE×affinity: mean novelty={nov.mean():.3f}, "
        f"mean te_x_novelty={(te * nov).mean():.4f}",
        flush=True,
    )
    return out


def add_time_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + coarse platform-era indicators from tweet created_at (leakage-safe)."""
    out = df.copy()
    ts = pd.to_datetime(out["created_at"], utc=True)
    year = ts.dt.year.astype(np.int16)
    month = ts.dt.month.astype(np.float64)
    out["year"] = year
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Coarse product/algorithm regimes (UTC cutpoints; ordinal + one-hots).
    # 0: pre-2015  1: 2015–2017 algo timeline  2: 2018–2022-10 pre-Musk
    # 3: 2022-11–2023 Musk early  4: 2024+ Musk late / For You era
    era = np.zeros(len(out), dtype=np.int8)
    era[ts >= "2015-01-01"] = 1
    era[ts >= "2018-01-01"] = 2
    era[ts >= "2022-11-01"] = 3
    era[ts >= "2024-01-01"] = 4
    out["platform_era"] = era
    out["era_pre_timeline"] = (era == 0).astype(np.int8)
    out["era_algo_timeline"] = (era == 1).astype(np.int8)
    out["era_pre_musk"] = (era == 2).astype(np.int8)
    out["era_musk_early"] = (era == 3).astype(np.int8)
    out["era_musk_late"] = (era == 4).astype(np.int8)

    vc = pd.Series(era).value_counts().sort_index()
    print("Time regimes:", {int(k): int(v) for k, v in vc.items()}, flush=True)
    return out


def add_author_emb_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """Cosine / L2 of tweet emb vs mean of author's earlier MiniLM vectors (pre-T only)."""
    emb_cols = [c for c in df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols:
        print("Skipping author emb affinity (no emb_* columns)")
        return df
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))

    out = df.reset_index(drop=True).copy()
    emb = out[emb_cols].to_numpy(dtype=np.float64)
    n = len(out)
    cos_all = np.zeros(n, dtype=np.float64)
    l2_all = np.zeros(n, dtype=np.float64)
    cos_30 = np.zeros(n, dtype=np.float64)
    l2_30 = np.zeros(n, dtype=np.float64)

    for _, g in out.groupby("account_id", sort=False):
        g = g.sort_values("created_at", kind="mergesort")
        ix = g.index.to_numpy()
        e = emb[ix]
        times = g["created_at"].to_numpy()
        cumsum = np.cumsum(e, axis=0)
        for k in range(len(ix)):
            if k == 0:
                continue
            mean = cumsum[k - 1] / k
            v = e[k]
            mean_n = np.linalg.norm(mean)
            v_n = np.linalg.norm(v)
            if mean_n > 1e-12 and v_n > 1e-12:
                cos_all[ix[k]] = float(np.dot(mean, v) / (mean_n * v_n))
            l2_all[ix[k]] = float(np.linalg.norm(v - mean))

            cut = times[k]
            start = cut - np.timedelta64(30, "D")
            left = int(np.searchsorted(times[:k], start, side="left"))
            n_win = k - left
            if n_win <= 0:
                continue
            # Mean of embeddings in [start, cut) via prefix sums.
            mean_w = (cumsum[k - 1] - (cumsum[left - 1] if left > 0 else 0.0)) / n_win
            mean_wn = np.linalg.norm(mean_w)
            if mean_wn > 1e-12 and v_n > 1e-12:
                cos_30[ix[k]] = float(np.dot(mean_w, v) / (mean_wn * v_n))
            l2_30[ix[k]] = float(np.linalg.norm(v - mean_w))

    out["author_emb_cos"] = cos_all
    out["author_emb_l2"] = l2_all
    out["author_emb_cos_30d"] = cos_30
    out["author_emb_l2_30d"] = l2_30
    print(
        f"Author emb affinity: mean cos={cos_all.mean():.3f}, "
        f"mean cos_30d={cos_30.mean():.3f}"
    )
    return out


def add_interlocutor_emb_affinity(
    df: pd.DataFrame,
    *,
    force: bool = False,
    include_mentions: bool = False,
    sweet_cos: float = 0.5,
    embed_batch_size: int = 256,
) -> pd.DataFrame:
    """Cosine / L2 vs mean of pre-T inbound interlocutor MiniLM vectors.

    Interlocutor texts are reply/quote tweets directed at the author (dump),
    optionally mentions. Includes quadratic and distance-to-sweet-spot terms
    for a non-linear audience-fit effect.
    """
    emb_cols = [c for c in df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols:
        print("Skipping interlocutor emb affinity (no emb_* columns)")
        return df
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))

    # Local import keeps prepare's heavy deps out of the default train path.
    from prepare import ensure_interlocutor_embeddings, load_interlocutor_events

    events = load_interlocutor_events(
        df["account_id"],
        force=force,
        include_mentions=include_mentions,
    )
    if events.empty:
        print("Interlocutor emb affinity: no inbound events; features left at 0")
        out = df.copy()
        for c in INTERLOCUTOR_FEATURES:
            out[c] = 0.0
        return out
    events = ensure_interlocutor_embeddings(events, batch_size=embed_batch_size)
    evt_emb_cols = [c for c in events.columns if c.startswith("emb_")]
    evt_emb_cols = sorted(evt_emb_cols, key=lambda c: int(c.split("_", 1)[1]))
    if len(evt_emb_cols) != len(emb_cols):
        raise SystemExit(
            f"Interlocutor emb dim mismatch: events={len(evt_emb_cols)} dataset={len(emb_cols)}"
        )

    out = df.reset_index(drop=True).copy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    tweet_emb = out[emb_cols].to_numpy(dtype=np.float64)
    n = len(out)
    cos_all = np.zeros(n, dtype=np.float64)
    l2_all = np.zeros(n, dtype=np.float64)
    cos_30 = np.zeros(n, dtype=np.float64)
    log_n = np.zeros(n, dtype=np.float64)
    log_n_30 = np.zeros(n, dtype=np.float64)
    n_partners = np.zeros(n, dtype=np.float64)
    has = np.zeros(n, dtype=np.float64)

    events = events.sort_values(["ego_account_id", "created_at"], kind="mergesort")
    evt_by_ego: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ego, g in events.groupby("ego_account_id", sort=False):
        evt_by_ego[str(ego)] = (
            g["created_at"].to_numpy(dtype="datetime64[ns]"),
            g[evt_emb_cols].to_numpy(dtype=np.float64),
            g["partner_account_id"].astype(str).to_numpy(),
        )

    for aid, g in out.groupby("account_id", sort=False):
        g = g.sort_values("created_at", kind="mergesort")
        ix = g.index.to_numpy()
        times = g["created_at"].to_numpy(dtype="datetime64[ns]")
        pack = evt_by_ego.get(str(aid))
        if pack is None:
            continue
        etimes, evecs, epartners = pack
        if len(etimes) == 0:
            continue
        ecum = np.cumsum(evecs, axis=0)
        # Partner first-seen times for leakage-safe partner counts.
        first_partner: dict[str, np.datetime64] = {}
        for t, p in zip(etimes, epartners, strict=True):
            if p not in first_partner:
                first_partner[p] = t
        partner_items = sorted(first_partner.items(), key=lambda kv: kv[1])
        partner_times = np.array([t for _, t in partner_items], dtype="datetime64[ns]")

        for k, row_i in enumerate(ix):
            cut = times[k]
            n_evt = int(np.searchsorted(etimes, cut, side="left"))
            if n_evt <= 0:
                continue
            mean = ecum[n_evt - 1] / n_evt
            v = tweet_emb[row_i]
            mean_n = np.linalg.norm(mean)
            v_n = np.linalg.norm(v)
            if mean_n > 1e-12 and v_n > 1e-12:
                cos_all[row_i] = float(np.dot(mean, v) / (mean_n * v_n))
            l2_all[row_i] = float(np.linalg.norm(v - mean))
            log_n[row_i] = float(np.log1p(n_evt))
            has[row_i] = 1.0
            n_partners[row_i] = float(np.log1p(int(np.searchsorted(partner_times, cut, side="left"))))

            start = cut - np.timedelta64(30, "D")
            left = int(np.searchsorted(etimes[:n_evt], start, side="left"))
            n_win = n_evt - left
            if n_win <= 0:
                continue
            mean_w = (ecum[n_evt - 1] - (ecum[left - 1] if left > 0 else 0.0)) / n_win
            mean_wn = np.linalg.norm(mean_w)
            if mean_wn > 1e-12 and v_n > 1e-12:
                cos_30[row_i] = float(np.dot(mean_w, v) / (mean_wn * v_n))
            log_n_30[row_i] = float(np.log1p(n_win))

    sweet = float(sweet_cos)
    out["interlocutor_emb_cos"] = cos_all.astype(np.float32)
    out["interlocutor_emb_l2"] = l2_all.astype(np.float32)
    out["interlocutor_emb_cos_sq"] = (cos_all * cos_all).astype(np.float32)
    out["interlocutor_sweet_dist"] = ((cos_all - sweet) ** 2).astype(np.float32)
    out["interlocutor_log_n"] = log_n.astype(np.float32)
    out["interlocutor_has"] = has.astype(np.float32)
    out["interlocutor_emb_cos_30d"] = cos_30.astype(np.float32)
    out["interlocutor_log_n_30d"] = log_n_30.astype(np.float32)
    out["interlocutor_n_partners"] = n_partners.astype(np.float32)

    covered = has.mean()
    covered_cos = cos_all[has > 0]
    print(
        f"Interlocutor emb affinity: row coverage={covered:.1%}, "
        f"mean cos={covered_cos.mean() if len(covered_cos) else 0.0:.3f}, "
        f"sweet={sweet:g}",
        flush=True,
    )
    return out


def add_emb_knn_target_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    k: int = 40,
    cold_max_n: float = 5.0,
    target_col: str = "log_engagement",
    exclude_same_account: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-fold mean target of K nearest MiniLM neighbors (content prior for cold accounts)."""
    from sklearn.neighbors import NearestNeighbors

    emb_cols = [c for c in train_df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols:
        return train_df, val_df
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))

    train_out = train_df.reset_index(drop=True).copy()
    val_out = val_df.reset_index(drop=True).copy()
    y = train_out[target_col].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y))

    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        return x / n

    x_tr = _l2_normalize(train_out[emb_cols].to_numpy(dtype=np.float64))
    x_va = _l2_normalize(val_out[emb_cols].to_numpy(dtype=np.float64))
    aids_tr = train_out["account_id"].astype(str).to_numpy()
    aids_va = val_out["account_id"].astype(str).to_numpy()

    # Query k+1 on train so we can drop self; optionally drop same account.
    kk = min(k + 25, len(train_out) - 1)
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean", algorithm="auto")
    nn.fit(x_tr)

    def _aggregate(dist: np.ndarray, idx: np.ndarray, query_aids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_q = idx.shape[0]
        means = np.empty(n_q, dtype=np.float64)
        stds = np.empty(n_q, dtype=np.float64)
        sims = np.empty(n_q, dtype=np.float64)
        for i in range(n_q):
            neigh = idx[i]
            d = dist[i]
            # Drop self (dist ~ 0) and optionally same-account neighbors.
            keep = []
            for j, ni in enumerate(neigh):
                if d[j] < 1e-9:
                    continue
                if exclude_same_account and aids_tr[ni] == query_aids[i]:
                    continue
                keep.append(j)
                if len(keep) >= k:
                    break
            if not keep:
                means[i] = global_mean
                stds[i] = 0.0
                sims[i] = 0.0
                continue
            sel = neigh[keep]
            sd = d[keep]
            yy = y[sel]
            means[i] = float(np.mean(yy))
            stds[i] = float(np.std(yy)) if len(yy) > 1 else 0.0
            # Euclidean on L2-normalized vectors: dist^2 = 2-2cos → cos = 1 - dist^2/2.
            sims[i] = float(np.mean(1.0 - np.clip(sd, 0, 2) ** 2 / 2.0))
        return means, stds, sims

    d_tr, i_tr = nn.kneighbors(x_tr)
    m_tr, s_tr, sim_tr = _aggregate(d_tr, i_tr, aids_tr)
    d_va, i_va = nn.kneighbors(x_va)
    m_va, s_va, sim_va = _aggregate(d_va, i_va, aids_va)

    def _attach(df: pd.DataFrame, means: np.ndarray, stds: np.ndarray, sims: np.ndarray) -> pd.DataFrame:
        out = df
        out["emb_knn_mean_rate"] = means.astype(np.float32)
        out["emb_knn_std_rate"] = stds.astype(np.float32)
        out["emb_knn_mean_sim"] = sims.astype(np.float32)
        if "log_account_n" in out.columns:
            # log_account_n is log1p(train count); cold ≈ unseen or rare in train.
            n = np.expm1(out["log_account_n"].to_numpy(dtype=np.float64))
            cold = (n <= float(cold_max_n)).astype(np.float32)
        else:
            cold = np.zeros(len(out), dtype=np.float32)
        out["account_is_cold"] = cold
        out["knn_x_cold"] = (means * cold).astype(np.float32)
        if "account_mean_rate" in out.columns:
            out["te_minus_knn"] = (
                out["account_mean_rate"].to_numpy(dtype=np.float64) - means
            ).astype(np.float32)
        return out

    train_out = _attach(train_out, m_tr, s_tr, sim_tr)
    val_out = _attach(val_out, m_va, s_va, sim_va)
    cold_rate = float(val_out["account_is_cold"].mean())
    print(
        f"Emb kNN TE: k={k} exclude_same_account={exclude_same_account} "
        f"val_cold={cold_rate:.1%} knn_mean p50={np.median(m_va):.4f} "
        f"sim p50={np.median(sim_va):.3f}",
        flush=True,
    )
    return train_out, val_out


def add_emb_cav_direction(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    quantile: float = 0.1,
    nonreply_only: bool = True,
    target_col: str = "log_engagement",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project MiniLM onto the unit difference-of-means direction μ_hot − μ_cold.

    Fit on the train fold only (optionally non-replies), before emb PCA, so the
    direction lives in the raw embedding space. Dense feature — every row gets
    a projection; smoke via train-fold corr(proj, target) before promoting.
    """
    emb_cols = [c for c in train_df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols:
        print("Skipping emb CAV (no emb_* columns)", flush=True)
        return train_df, val_df
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))
    q = float(np.clip(quantile, 0.01, 0.49))

    train_out = train_df.reset_index(drop=True).copy()
    val_out = val_df.reset_index(drop=True).copy()
    y = train_out[target_col].to_numpy(dtype=np.float64)
    fit_mask = np.ones(len(train_out), dtype=bool)
    if nonreply_only and "is_reply" in train_out.columns:
        fit_mask &= train_out["is_reply"].to_numpy(dtype=np.float64) < 0.5
    if fit_mask.sum() < 50:
        print("Skipping emb CAV (too few fit rows)", flush=True)
        return train_df, val_df

    y_fit = y[fit_mask]
    lo = float(np.quantile(y_fit, q))
    hi = float(np.quantile(y_fit, 1.0 - q))
    cold_mask = fit_mask & (y <= lo)
    hot_mask = fit_mask & (y >= hi)
    if cold_mask.sum() < 10 or hot_mask.sum() < 10:
        print("Skipping emb CAV (empty hot/cold tails)", flush=True)
        return train_df, val_df

    x_tr = train_out[emb_cols].to_numpy(dtype=np.float64)
    mu_hot = x_tr[hot_mask].mean(axis=0)
    mu_cold = x_tr[cold_mask].mean(axis=0)
    direction = mu_hot - mu_cold
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        print("Skipping emb CAV (degenerate direction)", flush=True)
        return train_df, val_df
    direction = direction / norm

    proj_tr = (x_tr @ direction).astype(np.float32)
    proj_va = (val_out[emb_cols].to_numpy(dtype=np.float64) @ direction).astype(np.float32)
    train_out["emb_cav_proj"] = proj_tr
    train_out["emb_cav_proj_sq"] = (proj_tr * proj_tr).astype(np.float32)
    val_out["emb_cav_proj"] = proj_va
    val_out["emb_cav_proj_sq"] = (proj_va * proj_va).astype(np.float32)

    corr = float(np.corrcoef(proj_tr, y)[0, 1]) if len(y) > 2 else float("nan")
    print(
        f"Emb CAV: q={q:g} nonreply_only={nonreply_only} "
        f"hot={int(hot_mask.sum()):,} cold={int(cold_mask.sum()):,} "
        f"|μh−μc|={norm:.4f} train corr(proj,y)={corr:.4f} "
        f"val proj p50={float(np.median(proj_va)):.4f}",
        flush=True,
    )
    return train_out, val_out


def add_emb_pls(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_components: int,
    *,
    target_col: str = "log_engagement",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-fold PLS of MiniLM → target; dense pls_* scores (fit before emb PCA)."""
    from sklearn.cross_decomposition import PLSRegression

    emb_cols = [c for c in train_df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols or n_components <= 0:
        return train_df, val_df
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))
    k = min(int(n_components), len(emb_cols), len(train_df) - 1)
    if k < 1:
        return train_df, val_df

    train_out = train_df.reset_index(drop=True).copy()
    val_out = val_df.reset_index(drop=True).copy()
    x_tr = train_out[emb_cols].to_numpy(dtype=np.float64)
    x_va = val_out[emb_cols].to_numpy(dtype=np.float64)
    y = train_out[target_col].to_numpy(dtype=np.float64)

    pls = PLSRegression(n_components=k, scale=True)
    z_tr = np.asarray(pls.fit_transform(x_tr, y)[0], dtype=np.float64)
    z_va = np.asarray(pls.transform(x_va), dtype=np.float64)

    # Drop any prior pls_* then attach.
    drop = [c for c in train_out.columns if re.fullmatch(r"pls_\d+", c)]
    if drop:
        train_out = train_out.drop(columns=drop)
        val_out = val_out.drop(columns=[c for c in drop if c in val_out.columns])
    for i in range(z_tr.shape[1]):
        train_out[f"pls_{i}"] = z_tr[:, i].astype(np.float32)
        val_out[f"pls_{i}"] = z_va[:, i].astype(np.float32)

    corr0 = float(np.corrcoef(z_tr[:, 0], y)[0, 1]) if len(y) > 2 else float("nan")
    # How much of train target linear signal the k scores capture (in-sample R² of PLS).
    y_hat = np.asarray(pls.predict(x_tr), dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(
        f"Emb PLS: k={k} train corr(pls_0,y)={corr0:.4f} in-sample R²={r2:.4f} "
        f"val pls_0 p50={float(np.median(z_va[:, 0])):.4f}",
        flush=True,
    )
    return train_out, val_out


def _residual_emb_design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Intercept + leakage-safe account/reply baselines for emb residualization."""
    names = [c for c in RESIDUAL_EMB_BASELINES if c in df.columns]
    n = len(df)
    if not names:
        return np.ones((n, 1), dtype=np.float64), ["intercept"]
    z = df[names].to_numpy(dtype=np.float64)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.concatenate([np.ones((n, 1), dtype=np.float64), z], axis=1)
    return z, ["intercept", *names]


def residualize_embeddings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], float] | None:
    """Per-dim OLS: emb ~ account TE / reply baselines; return residual MiniLM matrices.

    Fit on the train fold only. Isolates content variation orthogonal (in OLS sense)
    to account heat and reply stratum before CAV/PLS.
    """
    emb_cols = [c for c in train_df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols:
        return None
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))
    z_tr, names = _residual_emb_design(train_df)
    z_va, _ = _residual_emb_design(val_df)
    if z_tr.shape[1] != z_va.shape[1]:
        # Val missing a baseline col that train had — rebuild val with train names.
        names_core = [n for n in names if n != "intercept"]
        z_va = np.ones((len(val_df), 1), dtype=np.float64)
        if names_core:
            missing = [c for c in names_core if c not in val_df.columns]
            if missing:
                print(f"Skipping residual emb (val missing baselines: {missing})", flush=True)
                return None
            z_rest = val_df[names_core].to_numpy(dtype=np.float64)
            z_rest = np.nan_to_num(z_rest, nan=0.0, posinf=0.0, neginf=0.0)
            z_va = np.concatenate([z_va, z_rest], axis=1)
    x_tr = train_df[emb_cols].to_numpy(dtype=np.float64)
    x_va = val_df[emb_cols].to_numpy(dtype=np.float64)
    coef, *_ = np.linalg.lstsq(z_tr, x_tr, rcond=None)
    hat_tr = z_tr @ coef
    r_tr = x_tr - hat_tr
    r_va = x_va - z_va @ coef
    # Mean R² across emb dims (how much baseline explains raw MiniLM).
    ss_res = np.sum((x_tr - hat_tr) ** 2, axis=0)
    ss_tot = np.sum((x_tr - x_tr.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_dims = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    mean_r2 = float(np.nanmean(r2_dims))
    print(
        f"Residual emb: baselines={names} mean R²(emb|Z)={mean_r2:.4f} "
        f"residual ||.||_rms train={float(np.sqrt(np.mean(r_tr ** 2))):.4f}",
        flush=True,
    )
    return r_tr, r_va, names, mean_r2


def add_residual_emb_cav(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    quantile: float = 0.1,
    nonreply_only: bool = True,
    target_col: str = "log_engagement",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CAV on residual MiniLM (after OLS vs account TE / reply baselines)."""
    res = residualize_embeddings(train_df, val_df)
    if res is None:
        print("Skipping residual emb CAV (no emb_* columns)", flush=True)
        return train_df, val_df
    r_tr, r_va, _names, _mean_r2 = res
    q = float(np.clip(quantile, 0.01, 0.49))

    train_out = train_df.reset_index(drop=True).copy()
    val_out = val_df.reset_index(drop=True).copy()
    y = train_out[target_col].to_numpy(dtype=np.float64)
    fit_mask = np.ones(len(train_out), dtype=bool)
    if nonreply_only and "is_reply" in train_out.columns:
        fit_mask &= train_out["is_reply"].to_numpy(dtype=np.float64) < 0.5
    if fit_mask.sum() < 50:
        print("Skipping residual emb CAV (too few fit rows)", flush=True)
        return train_df, val_df

    y_fit = y[fit_mask]
    lo = float(np.quantile(y_fit, q))
    hi = float(np.quantile(y_fit, 1.0 - q))
    cold_mask = fit_mask & (y <= lo)
    hot_mask = fit_mask & (y >= hi)
    if cold_mask.sum() < 10 or hot_mask.sum() < 10:
        print("Skipping residual emb CAV (empty hot/cold tails)", flush=True)
        return train_df, val_df

    mu_hot = r_tr[hot_mask].mean(axis=0)
    mu_cold = r_tr[cold_mask].mean(axis=0)
    direction = mu_hot - mu_cold
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        print("Skipping residual emb CAV (degenerate direction)", flush=True)
        return train_df, val_df
    direction = direction / norm

    proj_tr = (r_tr @ direction).astype(np.float32)
    proj_va = (r_va @ direction).astype(np.float32)
    train_out["res_emb_cav_proj"] = proj_tr
    train_out["res_emb_cav_proj_sq"] = (proj_tr * proj_tr).astype(np.float32)
    val_out["res_emb_cav_proj"] = proj_va
    val_out["res_emb_cav_proj_sq"] = (proj_va * proj_va).astype(np.float32)

    corr = float(np.corrcoef(proj_tr, y)[0, 1]) if len(y) > 2 else float("nan")
    print(
        f"Residual emb CAV: q={q:g} nonreply_only={nonreply_only} "
        f"hot={int(hot_mask.sum()):,} cold={int(cold_mask.sum()):,} "
        f"|μh−μc|={norm:.4f} train corr(proj,y)={corr:.4f}",
        flush=True,
    )
    return train_out, val_out


def add_residual_emb_pls(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_components: int,
    *,
    target_col: str = "log_engagement",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PLS of residual MiniLM → target; dense res_pls_* scores (fit before emb PCA)."""
    from sklearn.cross_decomposition import PLSRegression

    if n_components <= 0:
        return train_df, val_df
    res = residualize_embeddings(train_df, val_df)
    if res is None:
        print("Skipping residual emb PLS (no emb_* columns)", flush=True)
        return train_df, val_df
    r_tr, r_va, _names, _mean_r2 = res
    k = min(int(n_components), r_tr.shape[1], len(train_df) - 1)
    if k < 1:
        return train_df, val_df

    train_out = train_df.reset_index(drop=True).copy()
    val_out = val_df.reset_index(drop=True).copy()
    y = train_out[target_col].to_numpy(dtype=np.float64)

    pls = PLSRegression(n_components=k, scale=True)
    z_tr = np.asarray(pls.fit_transform(r_tr, y)[0], dtype=np.float64)
    z_va = np.asarray(pls.transform(r_va), dtype=np.float64)

    drop = [c for c in train_out.columns if re.fullmatch(r"res_pls_\d+", c)]
    if drop:
        train_out = train_out.drop(columns=drop)
        val_out = val_out.drop(columns=[c for c in drop if c in val_out.columns])
    for i in range(z_tr.shape[1]):
        train_out[f"res_pls_{i}"] = z_tr[:, i].astype(np.float32)
        val_out[f"res_pls_{i}"] = z_va[:, i].astype(np.float32)

    corr0 = float(np.corrcoef(z_tr[:, 0], y)[0, 1]) if len(y) > 2 else float("nan")
    y_hat = np.asarray(pls.predict(r_tr), dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(
        f"Residual emb PLS: k={k} train corr(res_pls_0,y)={corr0:.4f} "
        f"in-sample R²={r2:.4f} val res_pls_0 p50={float(np.median(z_va[:, 0])):.4f}",
        flush=True,
    )
    return train_out, val_out


def _train_oof_underpred(
    train_df: pd.DataFrame,
    *,
    target_col: str,
    seed: int,
    n_splits: int = 3,
) -> np.ndarray:
    """3-fold OOF preds from a light LGBM on leakage-safe tabular+TE features (no emb)."""
    from sklearn.model_selection import KFold

    feat_cols = [
        c
        for c in (
            TABULAR_FEATURES
            + ACCOUNT_TE_FEATURES
            + EMB_KNN_TE_FEATURES
            + URL_DOMAIN_FEATURES
        )
        if c in train_df.columns
    ]
    if not feat_cols:
        raise ValueError("No tabular/TE features available for fail-cluster OOF")
    x = train_df[feat_cols].to_numpy(dtype=np.float32)
    y = train_df[target_col].to_numpy(dtype=np.float64)
    oof = np.full(len(train_df), np.nan, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
    }
    for fold, (tr_ix, va_ix) in enumerate(kf.split(x)):
        dtr = lgb.Dataset(x[tr_ix], label=y[tr_ix], feature_name=feat_cols)
        dva = lgb.Dataset(x[va_ix], label=y[va_ix], reference=dtr, feature_name=feat_cols)
        model = lgb.train(
            params,
            dtr,
            num_boost_round=200,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_ix] = model.predict(x[va_ix], num_iteration=model.best_iteration)
    if np.isnan(oof).any():
        oof = np.where(np.isnan(oof), float(np.nanmean(oof)), oof)
    return oof


def add_fail_clusters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_clusters: int = 4,
    quantile: float = 0.1,
    mode: str = "oof",
    nonreply_only: bool = True,
    target_col: str = "log_engagement",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster MiniLM of large train under-predictions; distance-to-centroid features.

    OOF mode fits a light LGBM without embeddings so residual failures are the
    content-shaped slice the final model still misses (esp. hot non-replies).
    """
    from sklearn.cluster import KMeans

    emb_cols = [c for c in train_df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols:
        print("Skipping fail clusters (no emb_* columns)", flush=True)
        return train_df, val_df
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))
    q = float(np.clip(quantile, 0.01, 0.5))
    k = max(1, int(n_clusters))

    train_out = train_df.reset_index(drop=True).copy()
    val_out = val_df.reset_index(drop=True).copy()
    y = train_out[target_col].to_numpy(dtype=np.float64)

    if mode == "te":
        if "account_mean_rate" not in train_out.columns:
            print("Skipping fail clusters (no account_mean_rate for te mode)", flush=True)
            return train_df, val_df
        pred = train_out["account_mean_rate"].to_numpy(dtype=np.float64)
    else:
        pred = _train_oof_underpred(train_out, target_col=target_col, seed=seed)
    under = y - pred

    cand = np.ones(len(train_out), dtype=bool)
    if nonreply_only and "is_reply" in train_out.columns:
        cand &= train_out["is_reply"].to_numpy(dtype=np.float64) < 0.5
    if cand.sum() < 50:
        print("Skipping fail clusters (too few candidate rows)", flush=True)
        return train_df, val_df
    thr = float(np.quantile(under[cand], 1.0 - q))
    fail_mask = cand & (under >= thr)
    n_fail = int(fail_mask.sum())
    if n_fail < max(10, k):
        print(f"Skipping fail clusters (only {n_fail} failure rows)", flush=True)
        return train_df, val_df

    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-12)

    x_fail = _l2_normalize(train_out.loc[fail_mask, emb_cols].to_numpy(dtype=np.float64))
    kk = min(k, n_fail)
    km = KMeans(n_clusters=kk, random_state=seed, n_init=10)
    km.fit(x_fail)
    centers = km.cluster_centers_

    def _attach(df: pd.DataFrame) -> pd.DataFrame:
        x = _l2_normalize(df[emb_cols].to_numpy(dtype=np.float64))
        # Squared euclidean on unit vectors ↔ 2-2cos; report euclidean dist.
        # (n, k): ||x - c||^2 = 2 - 2 x·c for unit x,c
        sims = x @ centers.T
        dists = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))
        out = df
        out["fail_min_dist"] = dists.min(axis=1).astype(np.float32)
        for i in range(dists.shape[1]):
            out[f"fail_dist_{i}"] = dists[:, i].astype(np.float32)
        return out

    train_out = _attach(train_out)
    val_out = _attach(val_out)

    corr = (
        float(np.corrcoef(train_out["fail_min_dist"].to_numpy(), y)[0, 1])
        if len(y) > 2
        else float("nan")
    )
    corr_u = (
        float(np.corrcoef(train_out["fail_min_dist"].to_numpy(), under)[0, 1])
        if len(y) > 2
        else float("nan")
    )
    oof_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    print(
        f"Fail clusters: mode={mode} k={kk} q={q:g} nonreply_only={nonreply_only} "
        f"n_fail={n_fail:,} thr_under={thr:.4f} oof_rmse={oof_rmse:.4f} "
        f"corr(min_dist,y)={corr:.4f} corr(min_dist,under)={corr_u:.4f} "
        f"val min_dist p50={float(val_out['fail_min_dist'].median()):.4f}",
        flush=True,
    )
    return train_out, val_out


def add_emb_cluster_te(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_clusters: int = 32,
    prior_strength: float = 20.0,
    target_col: str = "log_engagement",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-means on train MiniLM; smoothed LOO mean target by cluster + dist-to-centroid."""
    from sklearn.cluster import MiniBatchKMeans

    emb_cols = [c for c in train_df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols:
        print("Skipping emb cluster TE (no emb_* columns)", flush=True)
        return train_df, val_df
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))
    k = max(2, min(int(n_clusters), len(train_df) // 20))
    m = float(prior_strength)

    train_out = train_df.reset_index(drop=True).copy()
    val_out = val_df.reset_index(drop=True).copy()
    y = train_out[target_col].to_numpy(dtype=np.float64)
    global_mean = float(np.mean(y))

    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-12)

    x_tr = _l2_normalize(train_out[emb_cols].to_numpy(dtype=np.float64))
    x_va = _l2_normalize(val_out[emb_cols].to_numpy(dtype=np.float64))

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(4096, len(train_out)),
        n_init=3,
        max_iter=100,
    )
    labels_tr = km.fit_predict(x_tr)
    labels_va = km.predict(x_va)
    centers = km.cluster_centers_

    # Per-cluster sums/counts on train.
    sums = np.bincount(labels_tr, weights=y, minlength=k).astype(np.float64)
    counts = np.bincount(labels_tr, minlength=k).astype(np.float64)
    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), global_mean)

    # Train: leave-one-out cluster mean, then shrink.
    n_i = counts[labels_tr]
    s_i = sums[labels_tr]
    loo = np.where(n_i > 1.0, (s_i - y) / (n_i - 1.0), global_mean)
    n_loo = np.maximum(n_i - 1.0, 0.0)
    te_tr = (n_loo * loo + m * global_mean) / (n_loo + m)

    # Val: full train cluster mean + same shrink; empty → global.
    n_va = counts[labels_va]
    mean_va = means[labels_va]
    te_va = (n_va * mean_va + m * global_mean) / (n_va + m)

    dist_tr = np.linalg.norm(x_tr - centers[labels_tr], axis=1)
    dist_va = np.linalg.norm(x_va - centers[labels_va], axis=1)

    train_out["cluster_mean_rate"] = te_tr.astype(np.float32)
    train_out["log_cluster_n"] = np.log1p(n_i).astype(np.float32)
    train_out["cluster_min_dist"] = dist_tr.astype(np.float32)
    val_out["cluster_mean_rate"] = te_va.astype(np.float32)
    val_out["log_cluster_n"] = np.log1p(n_va).astype(np.float32)
    val_out["cluster_min_dist"] = dist_va.astype(np.float32)

    if "account_mean_rate" in train_out.columns:
        train_out["te_minus_cluster"] = (
            train_out["account_mean_rate"].to_numpy(dtype=np.float64) - te_tr
        ).astype(np.float32)
        val_out["te_minus_cluster"] = (
            val_out["account_mean_rate"].to_numpy(dtype=np.float64) - te_va
        ).astype(np.float32)
    if "emb_knn_mean_rate" in train_out.columns:
        train_out["knn_minus_cluster"] = (
            train_out["emb_knn_mean_rate"].to_numpy(dtype=np.float64) - te_tr
        ).astype(np.float32)
        val_out["knn_minus_cluster"] = (
            val_out["emb_knn_mean_rate"].to_numpy(dtype=np.float64) - te_va
        ).astype(np.float32)

    corr = float(np.corrcoef(te_tr, y)[0, 1]) if len(y) > 2 else float("nan")
    nonempty = int((counts > 0).sum())
    print(
        f"Emb cluster TE: k={k} prior_m={m:g} nonempty={nonempty} "
        f"size p50={float(np.median(counts[counts > 0])):.0f} "
        f"train corr(cluster_te,y)={corr:.4f} "
        f"val te p50={float(np.median(te_va)):.4f} "
        f"val dist p50={float(np.median(dist_va)):.4f}",
        flush=True,
    )
    return train_out, val_out


def apply_emb_pca(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.decomposition import PCA

    emb_cols = [c for c in train_df.columns if re.fullmatch(r"emb_\d+", c)]
    if not emb_cols:
        return train_df, val_df
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))
    pca = PCA(n_components=min(n_components, len(emb_cols), len(train_df) - 1), random_state=0)
    train_z = pca.fit_transform(train_df[emb_cols].to_numpy(dtype=np.float32))
    val_z = pca.transform(val_df[emb_cols].to_numpy(dtype=np.float32))
    train_out = train_df.drop(columns=emb_cols).copy()
    val_out = val_df.drop(columns=emb_cols).copy()
    for i in range(train_z.shape[1]):
        train_out[f"emb_{i}"] = train_z[:, i]
        val_out[f"emb_{i}"] = val_z[:, i]
    print(f"PCA embeddings: {len(emb_cols)} → {train_z.shape[1]} (var={pca.explained_variance_ratio_.sum():.3f})")
    return train_out, val_out


def apply_text_svd(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bag-time text only; vectorizer + SVD fit on train fold."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_text = train_df["full_text"].fillna("").astype(str)
    val_text = val_df["full_text"].fillna("").astype(str)
    n_comp = min(n_components, max(2, len(train_df) - 1))
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_x = vectorizer.fit_transform(train_text)
    val_x = vectorizer.transform(val_text)
    n_comp = min(n_comp, train_x.shape[1] - 1) if train_x.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    train_z = svd.fit_transform(train_x).astype(np.float32)
    val_z = svd.transform(val_x).astype(np.float32)
    tfidf_cols = [f"tfidf_{i}" for i in range(train_z.shape[1])]
    train_out = pd.concat(
        [train_df.reset_index(drop=True), pd.DataFrame(train_z, columns=tfidf_cols)],
        axis=1,
    )
    val_out = pd.concat(
        [val_df.reset_index(drop=True), pd.DataFrame(val_z, columns=tfidf_cols)],
        axis=1,
    )
    print(
        f"TF-IDF SVD: vocab={len(vectorizer.vocabulary_):,} → {train_z.shape[1]} "
        f"(var={svd.explained_variance_ratio_.sum():.3f})"
    )
    return train_out, val_out


def fit_predict_group(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    residual: bool,
    baseline_kind: str,
    skip_embeddings: bool,
    seed: int,
    nonreply_weight: float = 1.0,
    loss: str = "asymmetric",
    quantile_alpha: float = 0.65,
    asymmetric_under_weight: float = 3.0,
    sera_floor: float = 0.05,
    sera_high_only: bool = True,
    sera_nonreply_only: bool = False,
    early_stop_metric: str = "rmse",
    rank_nonreply_only: bool = True,
    rank_min_group: int = 2,
    hybrid_rank: bool = True,
    model_kind: str = "lgbm",
    stream_fresh_days: float = 0.0,
    stream_fresh_floor: float = 0.25,
) -> tuple[np.ndarray, Any, list[str], dict[str, float]]:
    x_train, feature_names = feature_matrix(train_df, skip_embeddings=skip_embeddings)
    x_val, _ = feature_matrix(val_df, skip_embeddings=skip_embeddings)
    y_train_abs = train_df["log_engagement"].to_numpy(dtype=np.float64)
    y_val_abs = val_df["log_engagement"].to_numpy(dtype=np.float64)
    base_train = author_baseline(train_df, baseline_kind)
    base_val = author_baseline(val_df, baseline_kind)
    if residual:
        y_train = y_train_abs - base_train
        y_val_model = y_val_abs - base_val
    else:
        y_train = y_train_abs
        y_val_model = y_val_abs
    weight_train = np.where(
        train_df["is_reply"].to_numpy() == 0,
        nonreply_weight,
        1.0,
    ).astype(np.float64)
    if stream_fresh_days > 0:
        weight_train = stream_fresh_sample_weights(
            train_df,
            weight_train,
            fresh_days=stream_fresh_days,
            floor=stream_fresh_floor,
        )
    train_is_nonreply = train_df["is_reply"].to_numpy() == 0
    val_is_nonreply = val_df["is_reply"].to_numpy() == 0
    account_train = train_df["account_id"].astype(str).to_numpy()
    account_val = val_df["account_id"].astype(str).to_numpy()
    extra_metrics: dict[str, float] = {}

    def _fit_stratum_lambdarank(
        tr_mask: np.ndarray,
        va_mask: np.ndarray,
        *,
        label: str,
        seed_offset: int,
    ) -> tuple[Any, np.ndarray]:
        x_tr_r, _y_tr_cont, rel_tr, g_tr, meta_tr = prepare_account_rank_order(
            account_train[tr_mask],
            x_train[tr_mask],
            y_train[tr_mask],
            min_group=rank_min_group,
        )
        x_va_r, _y_va_cont, rel_va, g_va, meta_va = prepare_account_rank_order(
            account_val[va_mask],
            x_val[va_mask],
            y_val_model[va_mask],
            min_group=rank_min_group,
        )
        extra_metrics.update({f"{label}_train_{k}": v for k, v in meta_tr.items()})
        extra_metrics.update({f"{label}_val_{k}": v for k, v in meta_va.items()})
        print(
            f"LambdaRank[{label}]: train rows={int(meta_tr['rank_n_rows']):,} "
            f"accounts={int(meta_tr['rank_n_groups']):,} "
            f"(mean={meta_tr['rank_mean_group']:.1f}); "
            f"val rows={int(meta_va['rank_n_rows']):,} "
            f"accounts={int(meta_va['rank_n_groups']):,}",
            flush=True,
        )
        ranker, _meta = train_model(
            x_tr_r,
            rel_tr,
            x_va_r,
            rel_va,
            feature_names,
            seed=seed + seed_offset,
            weight_train=None,
            loss="lambdarank",
            quantile_alpha=quantile_alpha,
            asymmetric_under_weight=asymmetric_under_weight,
            sera_floor=sera_floor,
            sera_high_only=sera_high_only,
            sera_nonreply_only=False,
            early_stop_metric="ndcg" if early_stop_metric == "rmse" else early_stop_metric,
            val_is_nonreply=None,
            train_is_nonreply=None,
            group_train=g_tr,
            group_val=g_va,
        )
        scores = ranker.predict(x_val, num_iteration=ranker.best_iteration)
        return ranker, np.asarray(scores, dtype=np.float64)

    if model_kind == "gpboost":
        if loss == "lambdarank" or hybrid_rank:
            raise ValueError("GPBoost does not support lambdarank/hybrid-rank; use --model lgbm")
        if early_stop_metric != "rmse":
            print(
                f"Warning: GPBoost ignores --early-stop-metric={early_stop_metric}; using RMSE",
                flush=True,
            )
        model, _gp_model, extra_metrics = train_gpboost_model(
            x_train,
            y_train,
            x_val,
            y_val_model,
            feature_names,
            group_train=account_train,
            group_val=account_val,
            seed=seed,
            weight_train=weight_train,
            loss=loss,
            quantile_alpha=quantile_alpha,
            sera_floor=sera_floor,
            sera_high_only=sera_high_only,
            sera_nonreply_only=sera_nonreply_only,
            train_is_nonreply=train_is_nonreply,
        )
        raw = _gpboost_response_mean(
            model.predict(
                data=x_val,
                group_data_pred=account_val,
                pred_latent=False,
            )
        )
    elif hybrid_rank:
        if loss not in ("asymmetric", "mse", "huber", "sera", "quantile"):
            raise ValueError(f"--hybrid-rank incompatible with --loss {loss}")
        # 1) Level model (between-author + magnitude calibration).
        level_loss = loss
        model, sera_meta = train_model(
            x_train,
            y_train,
            x_val,
            y_val_model,
            feature_names,
            seed=seed,
            weight_train=weight_train,
            loss=level_loss,
            quantile_alpha=quantile_alpha,
            asymmetric_under_weight=asymmetric_under_weight,
            sera_floor=sera_floor,
            sera_high_only=sera_high_only,
            sera_nonreply_only=sera_nonreply_only,
            early_stop_metric=early_stop_metric if early_stop_metric != "ndcg" else "rmse",
            val_is_nonreply=val_is_nonreply,
            train_is_nonreply=train_is_nonreply,
        )
        extra_metrics.update(sera_meta)
        level_raw = np.asarray(
            model.predict(x_val, num_iteration=model.best_iteration), dtype=np.float64
        )
        level_pred = level_raw + base_val if residual else level_raw
        # 2) Stratum rankers (within-author order for NR and reply separately).
        tr_nr = train_df["is_reply"].to_numpy() == 0
        va_nr = val_df["is_reply"].to_numpy() == 0
        tr_rp = ~tr_nr
        va_rp = ~va_nr
        _nr_model, rank_nr = _fit_stratum_lambdarank(tr_nr, va_nr, label="nr", seed_offset=11)
        _rp_model, rank_rp = _fit_stratum_lambdarank(tr_rp, va_rp, label="reply", seed_offset=17)
        rank_score = np.where(va_nr, rank_nr, rank_rp)
        # 3) Blend per stratum: keep level mean/std, impose ranker order.
        pred = blend_level_with_rank_scores(level_pred, rank_score, account_val, mask=va_nr)
        pred = blend_level_with_rank_scores(pred, rank_score, account_val, mask=va_rp)
        extra_metrics["hybrid_rank"] = 1.0
        print("Hybrid-rank: blended level + stratum LambdaRank within accounts", flush=True)
        # LambdaRank scores already in pred; skip residual add below.
        return pred, model, feature_names, extra_metrics
    elif loss == "lambdarank":
        tr_mask = np.ones(len(train_df), dtype=bool)
        va_mask = np.ones(len(val_df), dtype=bool)
        if rank_nonreply_only:
            tr_mask &= train_df["is_reply"].to_numpy() == 0
            va_mask &= val_df["is_reply"].to_numpy() == 0
        model, rank_scores = _fit_stratum_lambdarank(
            tr_mask, va_mask, label="all" if not rank_nonreply_only else "nr", seed_offset=0
        )
        raw = rank_scores
    else:
        model, sera_meta = train_model(
            x_train,
            y_train,
            x_val,
            y_val_model,
            feature_names,
            seed=seed,
            weight_train=weight_train,
            loss=loss,
            quantile_alpha=quantile_alpha,
            asymmetric_under_weight=asymmetric_under_weight,
            sera_floor=sera_floor,
            sera_high_only=sera_high_only,
            sera_nonreply_only=sera_nonreply_only,
            early_stop_metric=early_stop_metric,
            val_is_nonreply=val_is_nonreply,
            train_is_nonreply=train_is_nonreply,
        )
        extra_metrics.update(sera_meta)
        raw = model.predict(x_val, num_iteration=model.best_iteration)
    # LambdaRank scores are not in label units — do not add residual baseline.
    if residual and loss != "lambdarank":
        pred = raw + base_val
    else:
        pred = raw
    return pred, model, feature_names, extra_metrics


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Missing dataset at {args.dataset}. Run: uv run prepare.py")

    t0 = time.time()
    df = pd.read_parquet(args.dataset)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if not args.media_types:
        drop_mt = [c for c in MEDIA_TYPE_FEATURES if c in df.columns]
        if drop_mt:
            df = df.drop(columns=drop_mt)
    elif "has_photo" not in df.columns:
        print(
            "Warning: --media-types on but has_photo missing; "
            "run: uv run prepare.py --media-only",
            flush=True,
        )
    if not args.reply_dyad:
        drop_dyad = [c for c in DYAD_FEATURES if c in df.columns]
        if drop_dyad:
            df = df.drop(columns=drop_dyad)
    elif "reply_dyad_has_prior" not in df.columns:
        print(
            "Warning: --reply-dyad on but dyad columns missing; "
            "run: uv run prepare.py --dyad-only",
            flush=True,
        )
    if not args.parent_chain:
        drop_chain = [c for c in PARENT_CHAIN_FEATURES if c in df.columns]
        if drop_chain:
            df = df.drop(columns=drop_chain)
    elif "has_reply_chain_parent" not in df.columns:
        print(
            "Warning: --parent-chain on but chain columns missing; "
            "run: uv run prepare.py --parent-chain-only",
            flush=True,
        )
    if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sort_values("created_at", kind="mergesort").tail(args.max_rows).reset_index(drop=True)
        print(f"Smoke subsample: keeping last {len(df):,} rows by created_at", flush=True)
    if "hist_mean_replies" not in df.columns:
        raise SystemExit(
            "Dataset missing hist_mean_replies — re-run prepare.py "
            "(fav/RT history features are banned; need reply-received rebuild)."
        )
    if any(c.startswith("hist_mean_eng") for c in df.columns):
        print("Warning: dataset still has hist_*_eng columns; they are ignored by TABULAR_FEATURES")

    if args.author_emb_affinity:
        # Affinity on raw MiniLM before train-fit PCA replaces emb_*.
        df = add_author_emb_affinity(df)
    if args.interlocutor_emb_affinity:
        # Same raw-MiniLM timing as author affinity (before emb PCA).
        df = add_interlocutor_emb_affinity(
            df,
            force=args.force_interlocutor,
            include_mentions=args.interlocutor_include_mentions,
            sweet_cos=args.interlocutor_sweet_cos,
        )
    if args.descriptive_text:
        df = add_descriptive_text_features(df)
    if args.provoke_labels:
        df = add_provoke_features(df, cache_path=args.provoke_cache)
    if args.stance_labels:
        df = add_stance_features(df, cache_path=args.stance_cache)
    if args.desc_labels:
        df = add_desc_features(df, cache_path=args.desc_cache)

    df = add_time_regime_features(df)

    df = apply_holdout_policy(
        df,
        policy=args.holdout_policy,
        min_upload_hist=args.min_upload_hist,
    )

    dump_time = pd.Timestamp(df["created_at"].max())
    y_train_raw_ref: np.ndarray | None = None
    y_val_raw_ref: np.ndarray | None = None
    val_eff_days: np.ndarray | None = None
    exposure_a = 0.0
    exposure_b = 0.0
    if args.exposure_correct and args.exposure_mode == "rate":
        df = apply_exposure_rate_targets(
            df,
            dump_time=dump_time,
            cap_days=args.exposure_cap_days,
            beta=args.exposure_beta,
            eps=args.exposure_eps,
        )

    if args.provenance_features:
        if "is_stream" not in df.columns:
            print(
                "Warning: --provenance-features on but is_stream missing; "
                "skipping (run: uv run prepare.py --provenance-only)",
                flush=True,
            )
        else:
            df = add_provenance_features(df)

    train_df, val_df = time_split(df, args.val_frac)
    if args.exposure_correct and args.exposure_mode == "resid":
        age_tr, eff_tr = _exposure_eff_days(train_df["created_at"], dump_time, args.exposure_cap_days)
        age_va, eff_va = _exposure_eff_days(val_df["created_at"], dump_time, args.exposure_cap_days)
        y_train_raw_ref = train_df["log_engagement"].to_numpy(dtype=np.float64)
        y_val_raw_ref = val_df["log_engagement"].to_numpy(dtype=np.float64)
        exposure_a, exposure_b = fit_exposure_resid(y_train_raw_ref, eff_tr, args.exposure_eps)
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df["log_engagement_raw"] = y_train_raw_ref
        val_df["log_engagement_raw"] = y_val_raw_ref
        train_df["exposure_eff_days"] = eff_tr
        val_df["exposure_eff_days"] = eff_va
        train_df["exposure_age_days"] = age_tr
        val_df["exposure_age_days"] = age_va
        base_tr = exposure_resid_baseline(eff_tr, exposure_a, exposure_b, args.exposure_eps)
        base_va = exposure_resid_baseline(eff_va, exposure_a, exposure_b, args.exposure_eps)
        train_df["log_engagement"] = y_train_raw_ref - base_tr
        val_df["log_engagement"] = y_val_raw_ref - base_va
        val_eff_days = eff_va
        print(
            f"Exposure resid: dump={dump_time.date()} cap={args.exposure_cap_days:g}d "
            f"a={exposure_a:.4f} b={exposure_b:.4f} | "
            f"train eff p50={np.median(eff_tr):.1f} val eff p50={np.median(eff_va):.1f}",
            flush=True,
        )
    elif args.exposure_correct and args.exposure_mode == "rate":
        y_train_raw_ref = train_df["log_engagement_raw"].to_numpy(dtype=np.float64)
        y_val_raw_ref = val_df["log_engagement_raw"].to_numpy(dtype=np.float64)
        val_eff_days = val_df["exposure_eff_days"].to_numpy(dtype=np.float64)
    if "url_domain" in train_df.columns:
        train_df, val_df = add_domain_priors(train_df, val_df)
    if args.account_te:
        train_df, val_df = add_account_target_encoding(
            train_df, val_df, prior_strength=args.account_te_prior
        )
        train_df = add_te_affinity_interactions(train_df)
        val_df = add_te_affinity_interactions(val_df)
    if args.era_te:
        train_df, val_df = add_era_target_encoding(
            train_df,
            val_df,
            prior_strength=args.era_te_prior,
            era_col=args.era_key,
        )
    if args.emb_knn_te and not args.skip_embeddings:
        train_df, val_df = add_emb_knn_target_encoding(
            train_df,
            val_df,
            k=args.emb_knn_k,
            cold_max_n=args.emb_knn_cold_max_n,
        )
    if args.emb_cav and not args.skip_embeddings:
        # Fit on raw MiniLM before train-fit PCA replaces emb_*.
        train_df, val_df = add_emb_cav_direction(
            train_df,
            val_df,
            quantile=args.emb_cav_quantile,
            nonreply_only=args.emb_cav_nonreply_only,
        )
    if args.emb_pls > 0 and not args.skip_embeddings:
        # Fit on raw MiniLM before train-fit PCA replaces emb_*.
        train_df, val_df = add_emb_pls(train_df, val_df, args.emb_pls)
    if args.residual_emb_pls > 0 and not args.skip_embeddings:
        # Residualize raw MiniLM vs TE/reply baselines, then PLS (before emb PCA).
        train_df, val_df = add_residual_emb_pls(
            train_df, val_df, args.residual_emb_pls
        )
    if args.residual_emb_cav and not args.skip_embeddings:
        train_df, val_df = add_residual_emb_cav(
            train_df,
            val_df,
            quantile=args.emb_cav_quantile,
            nonreply_only=args.emb_cav_nonreply_only,
        )
    if args.fail_clusters and not args.skip_embeddings:
        # Cluster raw MiniLM of OOF under-preds before emb PCA replaces emb_*.
        train_df, val_df = add_fail_clusters(
            train_df,
            val_df,
            n_clusters=args.fail_clusters_k,
            quantile=args.fail_clusters_quantile,
            mode=args.fail_clusters_mode,
            nonreply_only=args.fail_clusters_nonreply_only,
            seed=args.seed,
        )
    if args.emb_cluster_te and not args.skip_embeddings:
        # Fit on raw MiniLM before train-fit PCA replaces emb_*.
        train_df, val_df = add_emb_cluster_te(
            train_df,
            val_df,
            n_clusters=args.emb_cluster_k,
            prior_strength=args.emb_cluster_te_prior,
            seed=args.seed,
        )
    if args.emb_pca > 0 and not args.skip_embeddings:
        train_df, val_df = apply_emb_pca(train_df, val_df, args.emb_pca)
    if args.text_svd > 0:
        train_df, val_df = apply_text_svd(train_df, val_df, args.text_svd)

    rate_target = bool(args.exposure_correct and args.exposure_mode == "rate")
    if args.residual is None:
        residual = not rate_target
    else:
        residual = bool(args.residual)
        if rate_target and residual:
            print(
                "Warning: context residual + rate target mixes units; "
                "prefer --no-residual (the rate-mode default)",
                flush=True,
            )

    y_train_abs = train_df["log_engagement"].to_numpy(dtype=np.float64)
    y_val_abs = val_df["log_engagement"].to_numpy(dtype=np.float64)
    reply_base_val = author_baseline(val_df, "hist_replies")
    chosen_base_val = author_baseline(val_df, args.baseline)

    # Primary metrics are in model-target space (rate when exposure-rate is on).
    y_train_metric = y_train_abs
    y_val_metric = y_val_abs
    mean_metrics = evaluate(y_val_metric, np.full(len(y_val_metric), float(y_train_metric.mean())))
    # Reply-history baselines are in a different unit than rate targets — diagnostic only.
    reply_base_metrics = evaluate(y_val_metric, reply_base_val)
    chosen_base_metrics = evaluate(y_val_metric, chosen_base_val)
    print("Baseline (train-mean, target space):", {k: round(v, 4) for k, v in mean_metrics.items()})
    if not rate_target:
        print("Baseline (hist_replies):", {k: round(v, 4) for k, v in reply_base_metrics.items()})
        print(f"Baseline ({args.baseline}):", {k: round(v, 4) for k, v in chosen_base_metrics.items()})
    else:
        print(
            "Note: reply-history baselines skipped as primary comps under rate target "
            f"(hist_replies rmse vs rate={reply_base_metrics['rmse_log']:.3f}, not comparable)",
            flush=True,
        )
    print(
        "Target:",
        "residual" if residual else "absolute",
        f"| model={args.model} | baseline={args.baseline} | split_reply={args.split_reply} "
        f"| nonreply_weight={args.nonreply_weight} | breakout_head={args.breakout_head} "
        f"| account_te={args.account_te} "
        f"| exposure={args.exposure_correct}:{args.exposure_mode if args.exposure_correct else 'off'}",
    )

    breakout_auc = float("nan")
    if args.breakout_head:
        if args.model == "gpboost":
            print(
                "Warning: --breakout-head still uses LGBM OOF; combining with --model gpboost "
                "is experimental",
                flush=True,
            )
        train_df, val_df, breakout_auc = add_breakout_probs(
            train_df,
            val_df,
            baseline_kind=args.baseline,
            skip_embeddings=args.skip_embeddings,
            seed=args.seed,
            nonreply_weight=args.nonreply_weight,
            mode=args.breakout_mode,
            quantile=args.breakout_quantile,
        )

    gpboost_meta: dict[str, float] = {}
    fit_kwargs = dict(
        residual=residual,
        baseline_kind=args.baseline,
        skip_embeddings=args.skip_embeddings,
        seed=args.seed,
        nonreply_weight=args.nonreply_weight,
        loss=args.loss,
        quantile_alpha=args.quantile_alpha,
        asymmetric_under_weight=args.asymmetric_under_weight,
        sera_floor=args.sera_floor,
        sera_high_only=args.sera_high_only,
        sera_nonreply_only=args.sera_nonreply_only,
        early_stop_metric=args.early_stop_metric,
        rank_nonreply_only=args.rank_nonreply_only,
        rank_min_group=args.rank_min_group,
        hybrid_rank=args.hybrid_rank,
        model_kind=args.model,
        stream_fresh_days=args.stream_fresh_days,
        stream_fresh_floor=args.stream_fresh_floor,
    )
    if args.nonreply_only:
        tr_nr = train_df[train_df["is_reply"] == 0]
        va_nr = val_df[val_df["is_reply"] == 0]
        print(f"Non-reply-only train: {len(tr_nr):,} / {len(train_df):,} (val NR {len(va_nr):,})")
        nr_kwargs = {**fit_kwargs, "nonreply_weight": 1.0}
        _, model, feature_names, gpboost_meta = fit_predict_group(
            tr_nr,
            va_nr,
            **nr_kwargs,
        )
        # Score full val with the non-reply model, then overwrite replies.
        x_val_full, _ = feature_matrix(val_df, skip_embeddings=args.skip_embeddings)
        if args.model == "gpboost":
            group_val = val_df["account_id"].astype(str).to_numpy()
            raw_full = _gpboost_response_mean(
                model.predict(data=x_val_full, group_data_pred=group_val, pred_latent=False)
            )
        else:
            raw_full = model.predict(x_val_full, num_iteration=model.best_iteration)
        base_val = author_baseline(val_df, args.baseline)
        pred = raw_full + base_val if residual else raw_full
        tr_r = train_df[train_df["is_reply"] == 1]
        if len(tr_r):
            mean_r_resid = float(
                (
                    tr_r["log_engagement"].to_numpy(dtype=np.float64)
                    - author_baseline(tr_r, args.baseline)
                ).mean()
            )
        else:
            mean_r_resid = 0.0
        reply_m = val_df["is_reply"].to_numpy() == 1
        pred[reply_m] = base_val[reply_m] + mean_r_resid
        print(f"Reply fallback residual mean: {mean_r_resid:.4f}")
    elif args.era_heads:
        era_col = args.era_key
        if era_col not in train_df.columns:
            raise SystemExit(
                f"--era-heads requires {era_col} (add_time_regime_features)"
            )
        # Global fallback, then overwrite eras with enough train mass.
        pred, model, feature_names, gpboost_meta = fit_predict_group(
            train_df,
            val_df,
            **fit_kwargs,
        )
        era_tr = train_df[era_col].to_numpy()
        era_va = val_df[era_col].to_numpy()
        specialists = 0
        for era in sorted(set(int(x) for x in era_va.tolist())):
            tr_m = era_tr == era
            va_m = era_va == era
            n_tr = int(tr_m.sum())
            n_va = int(va_m.sum())
            if n_tr < args.era_head_min_train or n_va == 0:
                print(
                    f"  era head {era_col}={era}: skip (train={n_tr:,} val={n_va:,}; "
                    f"min={args.era_head_min_train})",
                    flush=True,
                )
                continue
            print(
                f"  era head {era_col}={era}: train={n_tr:,} val={n_va:,}",
                flush=True,
            )
            era_kwargs = {**fit_kwargs, "seed": args.seed + 100 + int(era)}
            p_era, model_era, feature_names, meta_era = fit_predict_group(
                train_df.loc[tr_m].reset_index(drop=True),
                val_df.loc[va_m].reset_index(drop=True),
                **era_kwargs,
            )
            pred[va_m] = p_era
            model = model_era
            gpboost_meta.update({f"era{int(era)}_{k}": v for k, v in meta_era.items()})
            specialists += 1
        gpboost_meta["era_heads"] = 1.0
        gpboost_meta["era_head_specialists"] = float(specialists)
        gpboost_meta["era_key_year"] = float(era_col == "year")
        print(
            f"Era heads ({era_col}): specialists={specialists} "
            f"(min_train={args.era_head_min_train}); rest use global fallback",
            flush=True,
        )
    elif args.split_reply:
        pred = np.zeros(len(val_df), dtype=np.float64)
        models = []
        feature_names: list[str] = []
        for label, mask_val in (("nonreply", 0), ("reply", 1)):
            tr = train_df[train_df["is_reply"] == mask_val]
            va = val_df[val_df["is_reply"] == mask_val]
            if len(tr) < 100 or len(va) == 0:
                print(f"  skip {label}: train={len(tr)} val={len(va)}")
                continue
            print(f"  fitting {label}: train={len(tr):,} val={len(va):,}")
            p, model, feature_names, meta = fit_predict_group(
                tr,
                va,
                residual=residual,
                baseline_kind=args.baseline,
                skip_embeddings=args.skip_embeddings,
                seed=args.seed + mask_val,
                nonreply_weight=1.0 if mask_val == 0 else args.nonreply_weight,
                loss=args.loss,
                quantile_alpha=args.quantile_alpha,
                asymmetric_under_weight=args.asymmetric_under_weight,
                sera_floor=args.sera_floor,
                sera_high_only=args.sera_high_only,
                sera_nonreply_only=args.sera_nonreply_only,
                early_stop_metric=args.early_stop_metric,
                rank_nonreply_only=args.rank_nonreply_only,
                rank_min_group=args.rank_min_group,
                hybrid_rank=args.hybrid_rank,
                model_kind=args.model,
                stream_fresh_days=args.stream_fresh_days,
                stream_fresh_floor=args.stream_fresh_floor,
            )
            pred[val_df["is_reply"].to_numpy() == mask_val] = p
            models.append(model)
            gpboost_meta.update(meta)
        model = models[-1] if models else None
    else:
        pred, model, feature_names, gpboost_meta = fit_predict_group(
            train_df,
            val_df,
            residual=residual,
            baseline_kind=args.baseline,
            skip_embeddings=args.skip_embeddings,
            seed=args.seed,
            nonreply_weight=args.nonreply_weight,
            loss=args.loss,
            quantile_alpha=args.quantile_alpha,
            asymmetric_under_weight=args.asymmetric_under_weight,
            sera_floor=args.sera_floor,
            sera_high_only=args.sera_high_only,
            sera_nonreply_only=args.sera_nonreply_only,
            early_stop_metric=args.early_stop_metric,
            rank_nonreply_only=args.rank_nonreply_only,
            rank_min_group=args.rank_min_group,
            hybrid_rank=args.hybrid_rank,
            model_kind=args.model,
            stream_fresh_days=args.stream_fresh_days,
            stream_fresh_floor=args.stream_fresh_floor,
        )

    # Primary prediction/label are in training-target space.
    y_eval = y_val_abs
    pred_primary = pred
    metrics = evaluate(y_eval, pred_primary)
    nonreply_mask = val_df["is_reply"].to_numpy() == 0
    reply_mask = ~nonreply_mask
    metrics["rmse_log_nonreply"] = evaluate(y_eval[nonreply_mask], pred_primary[nonreply_mask])["rmse_log"]
    metrics["rmse_log_reply"] = evaluate(y_eval[reply_mask], pred_primary[reply_mask])["rmse_log"]
    metrics["spearman_nonreply"] = evaluate(y_eval[nonreply_mask], pred_primary[nonreply_mask])["spearman"]
    metrics["spearman_reply"] = evaluate(y_eval[reply_mask], pred_primary[reply_mask])["spearman"]
    acct_val = val_df["account_id"].astype(str).to_numpy()
    # Primary: within-author ranking by stratum (draft quality, not fame).
    wa_nr = within_account_spearman(
        y_eval[nonreply_mask], pred_primary[nonreply_mask], acct_val[nonreply_mask]
    )
    wa_rp = within_account_spearman(
        y_eval[reply_mask], pred_primary[reply_mask], acct_val[reply_mask]
    )
    metrics["spearman_within_account_nonreply_mean"] = wa_nr["mean"]
    metrics["spearman_within_account_nonreply_wmean"] = wa_nr["weighted_mean"]
    metrics["spearman_within_account_nonreply_n"] = wa_nr["n_accounts"]
    metrics["spearman_within_account_reply_mean"] = wa_rp["mean"]
    metrics["spearman_within_account_reply_wmean"] = wa_rp["weighted_mean"]
    metrics["spearman_within_account_reply_n"] = wa_rp["n_accounts"]
    print(
        f"Within-account Spearman NR: mean={wa_nr['mean']:.4f} wmean={wa_nr['weighted_mean']:.4f} "
        f"(n={int(wa_nr['n_accounts'])}); reply: mean={wa_rp['mean']:.4f} "
        f"wmean={wa_rp['weighted_mean']:.4f} (n={int(wa_rp['n_accounts'])})",
        flush=True,
    )
    # Under-prediction diagnostics on hot non-replies (failure mode of interest).
    if nonreply_mask.any():
        y_nr = y_eval[nonreply_mask]
        p_nr = pred_primary[nonreply_mask]
        thr = float(np.quantile(y_nr, 0.9))
        hot = y_nr >= thr
        if hot.any():
            metrics["rmse_log_nonreply_top10"] = evaluate(y_nr[hot], p_nr[hot])["rmse_log"]
            metrics["mean_underpred_nonreply_top10"] = float(np.mean(np.maximum(y_nr[hot] - p_nr[hot], 0.0)))
            print(
                f"Hot non-reply (top10% thr={thr:.3f}): "
                f"rmse={metrics['rmse_log_nonreply_top10']:.4f} "
                f"mean_underpred={metrics['mean_underpred_nonreply_top10']:.4f}",
                flush=True,
            )
    # Diagnostic: decode rate → dump-total log eng (not the optimization target).
    if args.exposure_correct and args.exposure_mode == "rate":
        assert y_val_raw_ref is not None and val_eff_days is not None
        pred_raw = log_rate_to_log_eng(
            pred_primary,
            val_eff_days,
            beta=args.exposure_beta,
            eps=args.exposure_eps,
        )
        raw_m = evaluate(y_val_raw_ref, pred_raw)
        metrics["rmse_log_raw_dump"] = raw_m["rmse_log"]
        metrics["spearman_raw_dump"] = raw_m["spearman"]
        print(
            "Diagnostic dump-total decode:",
            {k: round(v, 4) for k, v in raw_m.items() if k in ("rmse_log", "spearman")},
            flush=True,
        )
    elif args.exposure_correct and args.exposure_mode == "resid":
        assert y_val_raw_ref is not None and val_eff_days is not None
        pred_raw = pred_primary + exposure_resid_baseline(
            val_eff_days, exposure_a, exposure_b, args.exposure_eps
        )
        raw_m = evaluate(y_val_raw_ref, pred_raw)
        metrics["rmse_log_raw_dump"] = raw_m["rmse_log"]
        metrics["spearman_raw_dump"] = raw_m["spearman"]
    else:
        metrics["rmse_log_raw_dump"] = metrics["rmse_log"]
        metrics["spearman_raw_dump"] = metrics["spearman"]
        pred_raw = pred_primary

    metrics["exposure_correct"] = bool(args.exposure_correct)
    metrics["exposure_mode"] = args.exposure_mode if args.exposure_correct else ""
    metrics["exposure_cap_days"] = float(args.exposure_cap_days) if args.exposure_correct else 0.0
    metrics["exposure_beta"] = float(args.exposure_beta) if args.exposure_correct else 0.0
    metrics["exposure_a"] = float(exposure_a) if args.exposure_correct else 0.0
    metrics["exposure_b"] = float(exposure_b) if args.exposure_correct else 0.0
    metrics["target_space"] = "exposure_rate" if rate_target else (
        "exposure_resid" if args.exposure_correct else "log_engagement"
    )
    metrics["best_iteration"] = int(getattr(model, "best_iteration", 0) or 0)
    metrics["num_features"] = len(feature_names)
    metrics["train_rows"] = int(len(train_df))
    metrics["val_rows"] = int(len(val_df))
    metrics["n_accounts_train"] = int(train_df["account_id"].nunique())
    metrics["n_accounts_val"] = int(val_df["account_id"].nunique())
    metrics["baseline_rmse_log"] = mean_metrics["rmse_log"]
    metrics["reply_baseline_rmse_log"] = reply_base_metrics["rmse_log"]
    metrics["chosen_baseline_rmse_log"] = chosen_base_metrics["rmse_log"]
    metrics["residual"] = residual
    metrics["baseline_kind"] = args.baseline
    metrics["nonreply_weight"] = args.nonreply_weight
    metrics["breakout_head"] = args.breakout_head
    metrics["breakout_auc"] = breakout_auc
    metrics["model"] = args.model
    metrics["account_te"] = bool(args.account_te)
    metrics["era_te"] = float(args.era_te)
    metrics["era_te_prior"] = float(args.era_te_prior) if args.era_te else 0.0
    metrics["era_heads"] = float(args.era_heads)
    metrics["era_key"] = args.era_key if (args.era_te or args.era_heads) else ""
    metrics["era_head_min_train"] = float(args.era_head_min_train) if args.era_heads else 0.0
    metrics["loss"] = args.loss
    metrics["sera_floor"] = float(args.sera_floor) if args.loss == "sera" else 0.0
    metrics["sera_high_only"] = float(args.sera_high_only) if args.loss == "sera" else 0.0
    metrics["sera_nonreply_only"] = (
        float(args.sera_nonreply_only) if args.loss == "sera" else 0.0
    )
    metrics["early_stop_metric"] = args.early_stop_metric
    metrics["hybrid_rank"] = float(args.hybrid_rank)
    metrics["rank_nonreply_only"] = (
        float(args.rank_nonreply_only) if args.loss == "lambdarank" or args.hybrid_rank else 0.0
    )
    metrics["rank_min_group"] = (
        float(args.rank_min_group) if args.loss == "lambdarank" or args.hybrid_rank else 0.0
    )
    metrics["holdout_policy"] = args.holdout_policy
    metrics["min_upload_hist"] = float(args.min_upload_hist)
    metrics["provenance_features"] = float(args.provenance_features)
    metrics["stream_fresh_days"] = float(args.stream_fresh_days)
    metrics["media_types"] = float(args.media_types)
    metrics.update(gpboost_meta)
    metrics["seconds"] = round(time.time() - t0, 2)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if model is not None:
        # For split-reply, persist the reply model as a convenience artifact.
        model.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    append_results(args.description, metrics)

    # Persist val rows + predictions for error analysis.
    pred_df = val_df[
        [
            c
            for c in (
                "tweet_id",
                "account_id",
                "username",
                "created_at",
                "full_text",
                "engagement",
                "log_engagement",
                "is_reply",
                "is_quote",
                "is_stream",
                "archive_upload_id",
                "has_media",
                "has_url",
                "text_len",
                "hist_mean_replies",
                "hist_mean_replies_reply",
                "hist_mean_replies_nonreply",
                "days_since_last",
                "inbound_replies",
                "inbound_replies_per_tweet",
                "inbound_quotes",
                "hist_dump_mean_replies",
                "hist_n",
            )
            if c in val_df.columns
        ]
    ].copy()
    pred_df["y_true_log"] = y_eval
    pred_df["y_pred_log"] = pred_primary
    pred_df["y_base_log"] = (
        chosen_base_val if residual else np.full(len(val_df), float(y_train_metric.mean()))
    )
    pred_df["abs_err_log"] = np.abs(pred_primary - y_eval)
    pred_df["err_log"] = pred_primary - y_eval
    if args.exposure_correct and y_val_raw_ref is not None:
        pred_df["y_true_log_raw"] = y_val_raw_ref
        pred_df["y_pred_log_raw"] = pred_raw
        pred_df["exposure_eff_days"] = val_eff_days
        pred_df["y_pred_eng"] = np.expm1(np.clip(pred_raw, 0, 20))
    else:
        pred_df["y_pred_eng"] = np.expm1(np.clip(pred_primary, 0, 20))
    pred_df.to_parquet(VAL_PRED_PATH, index=False)
    print(f"Wrote val predictions: {VAL_PRED_PATH}")

    if model is not None and feature_names:
        try:
            gain = model.feature_importance(importance_type="gain")
            importance = (
                pd.DataFrame({"feature": feature_names, "gain": gain})
                .sort_values("gain", ascending=False)
                .head(15)
            )
            print("\nTop features by gain:")
            print(importance.to_string(index=False))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: feature importance unavailable ({exc})", flush=True)

    print("\n---")
    print(f"target_space: {metrics['target_space']}")
    print(
        f"model: {metrics['model']} | account_te: {metrics['account_te']} | "
        f"era_te: {bool(metrics.get('era_te', 0))} | "
        f"era_heads: {bool(metrics.get('era_heads', 0))} | "
        f"loss: {metrics['loss']} | early_stop: {metrics.get('early_stop_metric', 'rmse')} | "
        f"hybrid_rank: {bool(metrics.get('hybrid_rank', 0))}"
    )
    print(
        "primary_metrics: spearman_within_account_{nonreply,reply}_wmean "
        "(draft quality within author×stratum); "
        "guardrails: global spearman_*, rmse_log, mean_underpred_nonreply_top10"
    )
    for key in (
        "spearman_within_account_nonreply_wmean",
        "spearman_within_account_nonreply_mean",
        "spearman_within_account_reply_wmean",
        "spearman_within_account_reply_mean",
        "spearman_nonreply",
        "spearman_reply",
        "spearman",
        "rmse_log",
        "mean_underpred_nonreply_top10",
        "rmse_log_nonreply",
        "rmse_log_reply",
        "mae_log",
        "r2_log",
        "rmse_log_raw_dump",
        "spearman_raw_dump",
        "mae_engagement",
        "baseline_rmse_log",
        "breakout_auc",
        "best_iteration",
        "num_features",
        "train_rows",
        "val_rows",
        "n_accounts_train",
        "n_accounts_val",
        "gpboost_re_var",
        "gpboost_error_var",
        "gpboost_n_cold_val_groups",
        "seconds",
    ):
        if key in metrics:
            print(f"{key}: {metrics[key]}")
    print(f"model_path: {MODEL_PATH}")


if __name__ == "__main__":
    main()
