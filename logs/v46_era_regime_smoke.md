# v46 regime-specific baselines / heads by era (2026-07-26)

## Setup

Medium same-rebuild: `uv run prepare.py --max-tweets 25000 --per-account-cap 100 --skip-media` → 23,664 rows / 287 accounts. Default hybrid stack; one knob per run.

| fold | platform_era | year |
|---|---|---|
| train | 4 (musk_late) 18,685 / others 246 | 2026: 9,032; 2025: 9,209; older: ~690 |
| val | **100% era 4** | **100% 2026** |

Time-holdout val is mono-regime on coarse platform eras — `platform_era` specialists barely differ from global. `--era-key year` is the denser stress test.

## Harness

- `--era-te` — train-fold LOO + shrink mean by `--era-key` and key×reply (`era_mean_rate`, `era_stratum_mean_rate`, `te_minus_era`, …)
- `--era-heads` — global fallback, then specialist per val bucket with train n ≥ `--era-head-min-train` (default 1500)
- `--era-key {platform_era,year}`

## Results (medium)

| run | knob | wa_nr | wa_rp | global_nr | rmse | underpred |
|---|---|---:|---:|---:|---:|---:|
| v46a | control | **0.221** | 0.214 | 0.504 | **0.726** | 2.40 |
| v46b | `--era-te` (platform_era) | 0.082 | 0.175 | 0.309 | 0.803 | 2.73 |
| v46c | `--era-heads` (platform_era) | 0.217 | **0.237** | **0.506** | 0.728 | 2.41 |
| v46d | `--era-te --era-key year` | 0.157 | 0.137 | 0.376 | 0.770 | 2.63 |
| v46e | `--era-heads --era-key year` | 0.185 | 0.145 | 0.482 | 0.728 | 2.39 |

Δ vs control:

- era TE (platform): wa_nr **−0.140**, wa_rp −0.039, rmse +0.076 — `era_mean_rate` #1 gain (~80k); near-constant on val but poisons level + ranker capacity
- era heads (platform): wa_nr −0.005 / wa_rp +0.022 / rmse flat — one specialist on era 4 (drops ~246 older train rows); noise-scale
- year TE: wa_nr −0.064 / wa_rp −0.078 — same failure mode, slightly less extreme
- year heads: wa_nr −0.036 / wa_rp −0.070 — 2026-only specialist (9.0k train) loses useful 2025 mass for late-2026 val

## Verdict

- **Leave opt-in; do not promote.**
- Calendar era *indicators* (v13) stay; train-fit era **means** as features are harmful under mono-era val.
- Separate heads do not help when the holdout is dominated by one bucket (platform) and hurt when the specialist truncates recent-but-not-val-year history (year).
- Related open item remains stream/upload heads / `--min-upload-hist` once a stable stream-only scoreboard exists — not unlocked by this calendar split.
