## Will it bang?

The objective of this project is to predict a Tweet's engagement based only on past/current information available from the Twitter Community Archive API.

The rules:

- You may use or fine-tune any MIT-licensed model, as long as it can (safely) be self-hosted or used through a free API.
- Alternatively, you may train your own model.
- Modal is installed in this project, and you have $30 in credits.
- Optionally, you may train and run locally on this consumer laptop.
- You may experiment with different ways of constructing input features, but all input-output pairs must use the same programmatic approach, and it is critically important to avoid future information leakage.

## Technical guidance

Run Python code with `uv run`.

Any time you have to look up how to do something you might want to do again, add a single line to this document to record the information so you don't have to look it up next time.

- API docs live at https://github.com/TheExGenesis/community-archive/blob/main/docs/api-doc.md (also https://www.community-archive.org/docs); Supabase REST at https://fabxmporizzqflnftavs.supabase.co.
- Public anon key (send as both `apikey` and `Authorization: Bearer`): `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8`
- Count rows with `Prefer: count=exact` + `Range: 0-0`; total is in the `Content-Range` header (e.g. `0-0/11703`). Filter usernames with `username=ilike.handle` on `enriched_tweets` or `all_account`.
- `tweets` only store `favorite_count` and `retweet_count` (no `quote_count` / `reply_count` / `bookmark_count`). When summing engagement received, exclude rows whose `full_text` starts with `RT @` — those inherit the original tweet’s RT count.
- Reply chains: ancestors via `reply_to_tweet_id`; own descendants via `account_id` + `reply_to_tweet_id` on `tweets`. Prefer CA; X fallback is `GET https://api.x.com/2/tweets` (OAuth1) with `referenced_tweets` / `author_id` expansions.
- `likes` has `liked_tweet_id` but no like timestamp; archive `like` entries are `{tweetId, fullText}` only. Liked text is in `liked_tweets`; full structured rows are in `tweets` only if someone uploaded (~32% of liked IDs).
- Liked-tweet `created_at` without a `tweets` row: decode Twitter snowflake — `datetime_ms = (tweet_id >> 22) + 1288834974657`.
- Tweet attachments in CA: `tweet_media` (`media_url`, `media_type`) and `tweet_urls` (`url`, `expanded_url`); often sparse — fall back to X `attachments.media_keys` / `entities.urls`.
- Bulk ML dump: `enriched_tweets.parquet` at `https://fabxmporizzqflnftavs.supabase.co/storage/v1/object/public/enriched_tweets/enriched_tweets.parquet` (~900MB); prefer this over paging the REST API for corpus work.
- Agent index for CA: https://www.community-archive.org/llms.txt
- Do not use `all_account.num_followers` (etc.) as historical features — those fields are current snapshots and leak the future for older tweets.
- Ban `account_display_name` (and similar mutable profile text) as features — users change it and CA has no as-of-T history. Prefer not to use `username` as a model feature either (more stable, but renames happen); keep handle only for joins/debug. Never feed handles/names into an LM for “quality / fame / influence” scoring if the model’s training or knowledge cutoff may post-date the tweet — that can leak future reputation. Safer LM labels are descriptive and time-invariant enough in principle (e.g. sentiment, gender presentation, humor/controversy tags of the *text*).
- Lexical tweet sentiment via MIT `vaderSentiment` (`uv run train.py --descriptive-text`); flat on rate RMSE in v15 — off by default. No handle/display-name scoring.
- Ban dump-time `favorite_count` / `retweet_count` as **input** history features. CA stores eventual totals, not counts-as-of-time-T, so using prior tweets’ fav/RT for a later tweet leaks post-T engagement. Labels may still use final fav+RT on the target tweet.
- Safe engagement-style history: count replies (and similarly timestamped events) to the author’s earlier tweets with `created_at < T` via `reply_to_tweet_id`; do not use undated `likes`.
- Frozen text encoder used here: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0); dataset/model cache under `~/.cache/will-it-bang/`.
- Sample the CA parquet with DuckDB (`read_parquet`) instead of loading the full dump into pandas — 14M rows OOMs a 16GB laptop.
- Cache MiniLM vectors by `tweet_id` in `~/.cache/will-it-bang/embed_cache.parquet` so feature rebuilds don't re-embed.
- Prefer many accounts with a per-account tweet cap over a few fat accounts — otherwise the time-split barely tests generalization. Scale defaults: `--max-tweets 120000 --min-account-tweets 50 --max-account-tweets 8000 --per-account-cap 180` (~110k rows / ~900 accounts).
- Attachment flags: page `tweet_media` / `tweet_urls` with `tweet_id=in.(...)` batches (~80 ids); coverage is sparse but leakage-safe. Refresh on an existing dataset: `uv run prepare.py --media-only` (v18: ~9% media / ~7% ca_url on the scaled set).
- Train-fit emb PCA (~96) on MiniLM; residualize only against leakage-safe baselines (e.g. pre-T reply-received rates), never against dump-time fav/RT history.
- Optional train-fit TF-IDF→SVD on `full_text` (default 128 dims) adds a small honest lift once fav/RT history is banned.
- Denser leakage-safe popularity: scan full dump for replies/quotes to all tweets by sampled `account_id`s (`reply_to_tweet_id` / `quoted_tweet_id` / `reply_to_user_id` with `created_at < T`); `user_mentions` lacks timestamps unless joined.
- Rebuild features with `uv run prepare.py --skip-media` to reuse media flags from an existing `dataset.parquet` and skip the slow CA attachment API.
- Pre-T author text affinity: mean of prior MiniLM vectors per account, then cosine/L2 vs current tweet (compute before emb PCA so PCA does not wipe the raw space).
- Reply/quote parent context: join `reply_to_tweet_id` / `quoted_tweet_id` from the dump; use parent MiniLM cosine + pre-T reply arrivals on the parent (never parent fav/RT). Coverage is partial — parents outside CA are missing. Fast re-run: `uv run prepare.py --parent-only`.
- Curated experiment history + unexplored backlog: `experiments.md`. Raw run logs go in `logs/` (process retrospective: `logs/meta_process.md`); metrics append to `results.tsv` (gitignored).
- Experiment gate (ε-greedy): with high probability (~1−ε) pick the next run from open `analyze.py` failure slices (nonreply / media / missing parent / viral under-pred); with probability ε try a backlog novelty bet. Still lock the primary metric before an ablation family; change one knob, then restore. For sparse features or LM labels, smoke ≤~500 rows for coverage + target correlation first — flats at ~7% coverage or ~0.06 corr are expected; treat failed smokes as mechanism research, not scoreboard bets.
- Time regimes: `year` / month seasonality + coarse platform eras from `created_at` (pre-2015, algo timeline, pre-Musk, Musk early/late) — leakage-safe calendar features in `train.py`.
- Default train target is exposure **rate** `log1p(eng/(min(age_days,30)+1))` so the model fits intensity, not dump maturity; primary metrics are in rate space. Dump-total decode is diagnostic only (`rmse_log_raw_dump`). Use `--no-exposure-correct --residual` for the old dump-total objective.
- Train-fold account target encoding: LOO mean of the model target by `account_id`, shrunk to the train global mean (`--account-te`, default on; `--account-te-prior 20`).
- Emb kNN TE (default on): train-fold MiniLM neighbors’ mean/std rate + sim (`--emb-knn-te`, `--emb-knn-k 40`); PCA / affinity must match only `emb_\d+` so `emb_knn_*` survive. Tiny lift in v22b (0.607→0.606).
- Emb PLS (opt-in `--emb-pls K`): train-fold `PLSRegression` of raw MiniLM→target → `pls_0..K-1` before emb PCA; use `--emb-pca 0` to replace variance PCA. Dense; smoke via train `corr(pls_0,y)`.
- Default LGBM objective is asymmetric MSE (under-prediction weight 3 via `--loss asymmetric`); `--loss huber` restores the older objective. Quantile pinball: `--loss quantile --quantile-alpha 0.65` (worse RMSE in v20b).
- Link/domain features: `uv run prepare.py --urls-only` caches CA `tweet_urls` domains in `~/.cache/will-it-bang/url_cache.parquet`; train adds train-fold `domain_mean_rate` (min_count=30). Flat on rate RMSE in v16 (~7% expanded-domain coverage).
- LM tone labels (wrongheaded/infuriating): `uv run modal run label_provoke.py --nonreply-only` → `provoke_cache_Qwen__Qwen3-8B.parquet`; train with `--provoke-labels --provoke-cache …`. Default rater Qwen3-8B (Apache-2.0); Qwen3.6-27B works on A100 but ~20× slower without batched multimodal gen. Flat rate RMSE in v17b.
