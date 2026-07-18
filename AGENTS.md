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
- Frozen text encoder used here: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0); dataset/model cache under `~/.cache/will-it-bang/`.
- Sample the CA parquet with DuckDB (`read_parquet`) instead of loading the full dump into pandas — 14M rows OOMs a 16GB laptop.
- Cache MiniLM vectors by `tweet_id` in `~/.cache/will-it-bang/embed_cache.parquet` so feature rebuilds don't re-embed.
- Prefer many accounts with a per-account tweet cap over a few fat accounts — otherwise the time-split barely tests generalization.
