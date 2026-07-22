"""
Re-train a shortlist of prior ablations on the level-only stack and compare
locked within-account Spearman primaries (fair vs v31a / --no-hybrid-rank).

Usage:
    uv run rescore_shortlist.py
    uv run rescore_shortlist.py --only sera,emb_cav
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "will-it-bang"
METRICS_PATH = CACHE_DIR / "metrics.json"
PRED_PATH = CACHE_DIR / "val_predictions.parquet"
OUT_DIR = Path(__file__).resolve().parent / "logs" / "rescore_within_account"

# Level-only stack (no hybrid): same defaults as historical ablations.
BASE = ["--no-hybrid-rank"]

RUNS: list[tuple[str, list[str]]] = [
    ("control_no_hybrid", []),
    ("sera", ["--loss", "sera"]),
    ("emb_cav", ["--emb-cav"]),
    ("emb_pls8", ["--emb-pls", "8"]),
    ("fail_clusters", ["--fail-clusters"]),
    ("emb_cluster_te", ["--emb-cluster-te"]),
    ("author_emb_affinity", ["--author-emb-affinity"]),
]

PRIMARY_KEYS = [
    "spearman_within_account_nonreply_wmean",
    "spearman_within_account_nonreply_mean",
    "spearman_within_account_reply_wmean",
    "spearman_within_account_reply_mean",
    "spearman_nonreply",
    "spearman_reply",
    "rmse_log",
    "mean_underpred_nonreply_top10",
    "best_iteration",
    "seconds",
]


def _run_one(name: str, extra: list[str]) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc = f"rescore {name} within-account primaries"
    cmd = [
        "uv",
        "run",
        "train.py",
        *BASE,
        *extra,
        "--description",
        desc,
    ]
    log_path = OUT_DIR / f"{name}.log"
    print(f"\n=== {name}: {' '.join(cmd)} ===", flush=True)
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed (exit {proc.returncode}); see {log_path}")
    metrics = json.loads(METRICS_PATH.read_text())
    shutil.copy2(METRICS_PATH, OUT_DIR / f"{name}_metrics.json")
    shutil.copy2(PRED_PATH, OUT_DIR / f"{name}_preds.parquet")
    row = {"run": name, **{k: metrics.get(k) for k in PRIMARY_KEYS}}
    print(
        f"{name}: wa_nr_wmean={row['spearman_within_account_nonreply_wmean']:.4f} "
        f"wa_rp_wmean={row['spearman_within_account_reply_wmean']:.4f} "
        f"global_nr={row['spearman_nonreply']:.4f} rmse={row['rmse_log']:.4f}",
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated run names to keep (default: all)",
    )
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    runs = [(n, e) for n, e in RUNS if not only or n in only]
    if not runs:
        raise SystemExit(f"No runs matched --only={args.only!r}")

    rows = [_run_one(name, extra) for name, extra in runs]
    table_path = OUT_DIR / "summary.json"
    table_path.write_text(json.dumps(rows, indent=2) + "\n")

    # TSV for quick paste into experiments.md
    cols = [
        "run",
        "spearman_within_account_nonreply_wmean",
        "spearman_within_account_reply_wmean",
        "spearman_nonreply",
        "spearman_reply",
        "rmse_log",
        "mean_underpred_nonreply_top10",
    ]
    tsv = OUT_DIR / "summary.tsv"
    with tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    ctrl = next((r for r in rows if r["run"] == "control_no_hybrid"), None)
    print("\n=== deltas vs control_no_hybrid (wa_* = primary) ===", flush=True)
    if ctrl:
        for row in rows:
            if row["run"] == "control_no_hybrid":
                continue
            d_nr = (
                row["spearman_within_account_nonreply_wmean"]
                - ctrl["spearman_within_account_nonreply_wmean"]
            )
            d_rp = (
                row["spearman_within_account_reply_wmean"]
                - ctrl["spearman_within_account_reply_wmean"]
            )
            d_rmse = row["rmse_log"] - ctrl["rmse_log"]
            print(
                f"{row['run']}: Δwa_nr={d_nr:+.4f} Δwa_rp={d_rp:+.4f} Δrmse={d_rmse:+.4f}",
                flush=True,
            )
    print(f"\nWrote {table_path} and {tsv}", flush=True)


if __name__ == "__main__":
    main()
