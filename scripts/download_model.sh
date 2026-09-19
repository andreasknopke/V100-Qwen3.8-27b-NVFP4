#!/usr/bin/env bash
# ============================================================================
#  Download the OFFICIAL Qwen3.8-27B NVFP4 NInfer artifact (~22 GiB).
#
#  Source: neroued/Qwen3.8-27B-nvfp4-NInfer  ->  qwen3_8_27b_nvfp4.ninfer
#  SHA256 verified against the value pinned below (matches upstream SHA256SUMS).
#
#  Why aria2c: Hugging Face resets long single connections (observed: curl died
#  at 1.3 GB with "Recv failure: Connection reset by peer" at 3.4 MB/s). aria2c
#  with 8 parallel range requests reaches ~34 MB/s and resumes across resets.
#  Falls back to single-connection curl with range resume if aria2c is absent.
# ============================================================================
set -uo pipefail

REPO="neroued/Qwen3.8-27B-nvfp4-NInfer"
FILE="qwen3_8_27b_nvfp4.ninfer"
EXPECT_SHA="552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c"

NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"
DEST="$NINFER_HOME/models"
BASE="https://huggingface.co/$REPO/resolve/main"
URL="$BASE/$FILE?download=true"
TARGET="$DEST/$FILE"
MAX_ATTEMPTS=40

log() { printf '\n===== %s =====\n' "$*"; }
mkdir -p "$DEST"

log "PHASE 0/3 resolve expected size + disk check"
API_SIZE="$(curl -s --max-time 30 "https://huggingface.co/api/models/$REPO?blobs=true" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(next((s['size'] for s in d['siblings'] if s['rfilename']=='$FILE'),''))" 2>/dev/null)"
if [ -n "$API_SIZE" ]; then
    echo "size: $API_SIZE bytes ($(awk -v b="$API_SIZE" 'BEGIN{printf "%.2f", b/1073741824}') GiB)"
    avail_kb=$(df -k --output=avail "$DEST" | tail -1 | tr -d ' ')
    avail_gib=$((avail_kb / 1024 / 1024))
    need_gib=$(( API_SIZE / 1024 / 1024 / 1024 + 2 ))
    echo "free: ${avail_gib} GiB (need ~${need_gib} GiB)"
    [ "$avail_gib" -lt "$need_gib" ] && { echo "FATAL: not enough disk space"; exit 1; }
else
    echo "warning: size not resolved from API, will rely on content-length + checksum"
fi

# skip if already complete and valid
if [ -f "$TARGET" ] && [ -n "$API_SIZE" ] && [ "$(stat -c %s "$TARGET")" = "$API_SIZE" ]; then
    echo "file present at full size; verifying checksum"
    if [ "$(sha256sum "$TARGET" | awk '{print $1}')" = "$EXPECT_SHA" ]; then
        echo "CHECKSUM_OK - nothing to do"; exit 0
    fi
    echo "checksum mismatch, re-downloading"
fi

log "PHASE 1/3 download with resume (up to $MAX_ATTEMPTS attempts)"
if command -v aria2c >/dev/null 2>&1; then
    echo "using aria2c (8 connections, resumable)"
    attempt=0
    while :; do
        attempt=$((attempt + 1))
        cur=0; [ -f "$TARGET" ] && cur="$(stat -c %s "$TARGET")"
        [ -n "$API_SIZE" ] && [ "$cur" -ge "$API_SIZE" ] && { echo "size reached ($cur)"; break; }
        [ "$attempt" -gt "$MAX_ATTEMPTS" ] && { echo "FATAL: giving up after $MAX_ATTEMPTS attempts"; exit 1; }
        pct="$(awk -v c="$cur" -v t="${API_SIZE:-0}" 'BEGIN{if(t>0)printf "%.1f%%",100*c/t; else print "?"}')"
        echo "--- aria2c attempt $attempt/$MAX_ATTEMPTS : resuming from $cur ($pct) ---"
        aria2c -c -x 8 -s 8 -k 1M -j 1 --file-allocation=none \
               --max-tries=0 --retry-wait=5 --timeout=60 --connect-timeout=30 \
               --lowest-speed-limit=1024 --summary-interval=30 --console-log-level=warn \
               --auto-file-renaming=false --allow-overwrite=true \
               -d "$DEST" -o "$FILE" "$URL"
        rc=$?
        newsize=0; [ -f "$TARGET" ] && newsize="$(stat -c %s "$TARGET")"
        echo "aria2c rc=$rc, size now $newsize"
        [ $rc -eq 0 ] && break
        sleep 5
    done
else
    echo "aria2c not found, falling back to curl (single connection, range resume)"
    attempt=0
    while :; do
        attempt=$((attempt + 1))
        cur=0; [ -f "$TARGET" ] && cur="$(stat -c %s "$TARGET")"
        [ -n "$API_SIZE" ] && [ "$cur" -ge "$API_SIZE" ] && { echo "size reached ($cur)"; break; }
        [ "$attempt" -gt "$MAX_ATTEMPTS" ] && { echo "FATAL: giving up after $MAX_ATTEMPTS attempts"; exit 1; }
        echo "--- curl attempt $attempt/$MAX_ATTEMPTS : resuming from $cur ---"
        curl -L -C - --fail --retry 8 --retry-all-errors --retry-delay 5 \
             --speed-limit 1024 --speed-time 60 --connect-timeout 30 \
             --no-progress-meter -o "$TARGET" "$URL"
        rc=$?
        [ $rc -eq 0 ] && break
        # 33/36 = server does not honour range -> restart from scratch
        if [ $rc -eq 33 ] || [ $rc -eq 36 ]; then echo "range refused, restarting"; rm -f "$TARGET"; fi
        sleep 3
    done
fi

log "PHASE 2/3 size verification"
final="$(stat -c %s "$TARGET")"
echo "final size: $final bytes ($(awk -v b="$final" 'BEGIN{printf "%.2f", b/1073741824}') GiB)"
if [ -n "$API_SIZE" ] && [ "$final" != "$API_SIZE" ]; then
    echo "FATAL: size mismatch (expected $API_SIZE)"; exit 1
fi

log "PHASE 3/3 SHA256 verification"
echo "computing sha256 (a few minutes on 22 GiB)..."
actual="$(sha256sum "$TARGET" | awk '{print $1}')"
echo "expected: $EXPECT_SHA"
echo "actual  : $actual"
if [ "$actual" = "$EXPECT_SHA" ]; then echo "CHECKSUM_OK"; else echo "FATAL: CHECKSUM MISMATCH"; exit 1; fi
echo "DOWNLOAD_DONE"
