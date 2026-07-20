# Meta: experiment process / harness (2026-07-19)

Snapshot of process lessons from `results.tsv` (~51 runs through v17b at review time; timeline later extended through v20), `logs/`, `experiments.md`, and the main Cursor chat. Visual companion (Cursor canvas, not in-repo): `experiment-process-meta.canvas.tsx` in the project canvases folder.

## Headline

Almost all honest scoreboard movement came from **problem formulation** (leakage bans, exposure-rate objective) and **denser leakage-safe history / scale** — not from factorial training tricks or low-coverage text/LM labels. Autopilot “next experiment” velocity stacked flats when candidates failed a coverage × correlation smoke test.

## Honest dump-total path (post fav/RT ban)

| Run | rmse_log | Lesson |
|---|---:|---|
| v6–v7 | ~1.065 | Validity reset; SVD tiny |
| v8 | ≥1.065 | Weight / breakout / nonreply-only: **no lift** |
| v9 | 0.998 | Dump-wide inbound graph: **largest model lift** |
| v10 | 0.999 | Author emb affinity: flat |
| v11 | 0.997 | Parent context: small lift |
| v12 | 0.978 | Diverse scale: lift |
| v13 | 0.967 | Time regimes: lift |
| v14a–d | — | Mis-scored rate/resid vs dump-total RMSE |
| v14e | 0.642 (rate) | New primary scoreboard (not comparable to dump-total) |

Post-v14e rate RMSE plateaus ~0.642–0.644 through v17b (VADER / domains / provoke). Later (after this review): v18 media refresh tied RMSE; v19 account TE and v20a asymmetric loss moved the rate scoreboard again — consistent with “hit residual failure mass / modeling of known under-prediction,” not another descriptive LM tag.

## Phase productivity

| Phase | Runs (approx) | Outcome | Lesson |
|---|---|---|---|
| Leaky climb (v2–v5) | 14 | Invalid once fav/RT history banned | Scoreboards lie if inputs leak |
| Honest rebuild (v6–v7) | 9 | RMSE ~1.065 | Correctness first |
| Ablation grind (v8) | 8 | Flat / regress | No new signal → no knobs |
| Graph + scale (v9–v13) | ~7 | 1.065 → 0.967 | Only sustained honest-lift block |
| Objective reframe (v14) | 7 | a–d wasted; e locks rate | Freeze metric before ablation family |
| Text/LM flats (v15–v17b) | 4 + Modal smoke | Rate RMSE stuck | Sparse / weak corr → doomed global RMSE |

## Highest-leverage moves (by kind)

1. **Validity** — Ban dump-time fav/RT history (user); cliff vs leaky v5.
2. **Signal density** — v9 inbound replies/quotes (−0.067 dump RMSE).
3. **Data** — v12 account-diverse scale (−0.019).
4. **Regime** — v13 calendar / eras (−0.011).
5. **Objective** — v14e exposure rate as primary (user clarification after misread “improve RMSE”).
6. **Flats** — v15–v17b: ~7% domain coverage; provoke corr≈0.06; Spearman-only wiggles.

## Harness strengths

- Machine-append `results.tsv` + curated `experiments.md` backlog / “do not re-litigate”
- Incremental prepare (`--parent-only`, `--urls-only`, `--media-only`, `--skip-media`)
- Restore runs after ablations so defaults stay known
- Failure-mode list tied to analyze slices

## Harness failure modes (observed)

- Primary metric not frozen before v14 ablation family
- Features/labels shipped without coverage × minimum-detectable-effect gate
- `analyze_v9` viral/media residuals underused as the experiment picker until late
- Expensive Modal / large-model path before label–target pierce on a smoke set
- Early “don’t check in” autonomy + many continue cues → autopilot flats

## Chat steering (qualitative)

Highest EV user interventions were intermittent and short: leakage policy, exposure-rate intent, display-name / fame-LM bans, harness consolidation (logs + backlog). Many turns were autonomy cues; those preserved speed but needed the gate below.

## Operating rules (crystallized)

**Do more:** coverage of known levers (media, parents, graph); amortized scale; regime/objective clarity when intent shifts; re-run `analyze.py` after each real lift and pick from residuals; one knob then restore; log outcome on the backlog.

**Do less (until new angle):** loss/weight/breakout stacks without new features (v8); coarse author emb affinity (v10); low-coverage domain means without denser context; provoke-style LM tags without corr smoke; giant rater upgrades when a calibrated smaller model is already flat.

**Experiment policy (ε-greedy + gate):**

- **Exploit (~1−ε):** next run from open failure slices (nonreply / media / missing parent / viral under-prediction) after a fresh `analyze.py` when needed.
- **Explore (ε):** backlog novelty (new LM tag, encoder, loss shape, …) so the loop does not starve for new signal — but still run the gate below before full spend.

Before full prepare/Modal/train:

1. If exploit: linked to a named failure slice? If explore: named backlog item + why it could pierce residuals?
2. Material row coverage (or a clear reason sparse is enough)?
3. For labels/features: cheap smoke (≤~500 rows) for coverage + target correlation / precision@k?
4. Primary metric locked for this ablation family?
5. One change; restore baseline after?

If (2)–(3) fail, treat as mechanism research — not a scoreboard bet. ε is a process dial (e.g. ~1 in 5–10 runs), not a hyperparameter to grind.
