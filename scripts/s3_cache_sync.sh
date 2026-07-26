#!/usr/bin/env bash
# Sync ~/.cache/will-it-bang/ with the shared S3 triage bucket across Cursor cloud containers.
#
# Env (required): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_S3_TRIAGE_BUCKET
# Optional: CACHE_DIR (default ~/.cache/will-it-bang), AWS_S3_PREFIX (default cache/will-it-bang)
#
# Usage:
#   scripts/s3_cache_sync.sh pull              # S3 canonical cache → local
#   scripts/s3_cache_sync.sh push              # local → S3 canonical cache
#   scripts/s3_cache_sync.sh pull-incoming     # promote useful incoming/ drop-zone objects
#   scripts/s3_cache_sync.sh promote-incoming  # incoming/ → cache/will-it-bang/ (S3→S3)
#   scripts/s3_cache_sync.sh ls                # list bucket prefixes
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/will-it-bang}"
PREFIX="${AWS_S3_PREFIX:-cache/will-it-bang}"
INCOMING_PREFIX="${AWS_S3_INCOMING_PREFIX:-incoming}"
BUCKET="${AWS_S3_TRIAGE_BUCKET:-}"

EXCLUDE_ARGS=(
  --exclude "enriched_tweets.parquet"
  --exclude ".env"
  --exclude "*.tmp"
  --exclude "*_access_probe.json"
)

die() { echo "error: $*" >&2; exit 1; }

need_aws() {
  command -v aws >/dev/null 2>&1 || die "aws CLI not found (pip install awscli / uv tool install awscli)"
  [[ -n "${BUCKET}" ]] || die "AWS_S3_TRIAGE_BUCKET is unset"
  [[ -n "${AWS_DEFAULT_REGION:-}" ]] || die "AWS_DEFAULT_REGION is unset"
}

s3_uri() { echo "s3://${BUCKET}/${1%/}/"; }

cmd="${1:-}"
[[ -n "${cmd}" ]] || die "usage: $0 {pull|push|pull-incoming|promote-incoming|ls}"

need_aws
mkdir -p "${CACHE_DIR}"

case "${cmd}" in
  pull)
    echo "pull $(s3_uri "${PREFIX}") → ${CACHE_DIR}/"
    aws s3 sync "$(s3_uri "${PREFIX}")" "${CACHE_DIR}/" "${EXCLUDE_ARGS[@]}"
    ;;
  push)
    echo "push ${CACHE_DIR}/ → $(s3_uri "${PREFIX}")"
    aws s3 sync "${CACHE_DIR}/" "$(s3_uri "${PREFIX}")" "${EXCLUDE_ARGS[@]}"
    ;;
  pull-incoming)
    # Drop zone may use dated/variant names; map known full-scale artifacts into local cache names.
    src="$(s3_uri "${INCOMING_PREFIX}")"
    echo "pull-incoming ${src} → ${CACHE_DIR}/"
    aws s3 cp "${src}embed_cache.parquet" "${CACHE_DIR}/embed_cache.parquet" 2>/dev/null || true
    aws s3 cp "${src}mention_cache.parquet" "${CACHE_DIR}/mention_cache.parquet" 2>/dev/null || true
    aws s3 cp "${src}url_cache.parquet" "${CACHE_DIR}/url_cache.parquet" 2>/dev/null || true
    aws s3 cp "${src}x_parent_cache.parquet" "${CACHE_DIR}/x_parent_cache.parquet" 2>/dev/null || true
    aws s3 cp "${src}archive_upload_cache.parquet" "${CACHE_DIR}/archive_upload_cache.parquet" 2>/dev/null || true
    aws s3 cp "${src}model.txt" "${CACHE_DIR}/model.txt" 2>/dev/null || true
    aws s3 cp "${src}val_predictions.parquet" "${CACHE_DIR}/val_predictions.parquet" 2>/dev/null || true
    aws s3 cp "${src}metrics.json" "${CACHE_DIR}/metrics.json" 2>/dev/null || true
    # Prefer X-complete dataset when present; else nox; else plain dataset.parquet
    if aws s3 ls "${src}dataset_xcomplete_full.parquet" >/dev/null 2>&1; then
      aws s3 cp "${src}dataset_xcomplete_full.parquet" "${CACHE_DIR}/dataset.parquet"
    elif aws s3 ls "${src}dataset_nox_full.parquet" >/dev/null 2>&1; then
      aws s3 cp "${src}dataset_nox_full.parquet" "${CACHE_DIR}/dataset.parquet"
    elif aws s3 ls "${src}dataset.parquet" >/dev/null 2>&1; then
      aws s3 cp "${src}dataset.parquet" "${CACHE_DIR}/dataset.parquet"
    else
      echo "warning: no dataset*.parquet found under ${src}" >&2
    fi
    ;;
  promote-incoming)
    src="$(s3_uri "${INCOMING_PREFIX}")"
    dst="$(s3_uri "${PREFIX}")"
    echo "promote ${src} → ${dst}"
    aws s3 sync "${src}" "${dst}" "${EXCLUDE_ARGS[@]}"
    # Also publish a canonical dataset.parquet from the preferred full-scale variant.
    if aws s3 ls "${src}dataset_xcomplete_full.parquet" >/dev/null 2>&1; then
      aws s3 cp "${src}dataset_xcomplete_full.parquet" "${dst}dataset.parquet"
    elif aws s3 ls "${src}dataset_nox_full.parquet" >/dev/null 2>&1; then
      aws s3 cp "${src}dataset_nox_full.parquet" "${dst}dataset.parquet"
    fi
    ;;
  ls)
    echo "=== ${BUCKET}/ ==="
    aws s3 ls "s3://${BUCKET}/"
    echo "=== ${PREFIX}/ ==="
    aws s3 ls "$(s3_uri "${PREFIX}")" || true
    echo "=== ${INCOMING_PREFIX}/ ==="
    aws s3 ls "$(s3_uri "${INCOMING_PREFIX}")" || true
    ;;
  *)
    die "unknown command '${cmd}' (expected pull|push|pull-incoming|promote-incoming|ls)"
    ;;
esac
