#!/usr/bin/env bash
# ============================================================================
#  Download the Cold-Fusion GAIN V1.1 bf16 checkpoint (DavidAU) for NInfer.
#
#  This is the SOURCE checkpoint that build_coldfusion_nvfp4.sh quantizes on the
#  fly into the NVFP4 .ninfer artifact. It is NOT a pre-quantized artifact.
#
#  Usage: download_coldfusion.sh [OUT_DIR]
#    OUT_DIR  default $NINFER_HOME/models/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1
#
#  Env overrides: NINFER_HOME, PY (python with huggingface_hub).
#
#  NOTE: the HuggingFace Xet storage backend crashes on these large shards
#  ("Background writer channel closed" -> bus error, can kill the WSL VM).
#  We disable it and use the classic resumable HTTP path with a single worker.
# ============================================================================
set -u

NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"
OUT="${1:-$NINFER_HOME/models/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1}"
PY="${PY:-$(command -v python3)}"

mkdir -p "$OUT"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Downloading Cold Fusion GAIN V1.1 -> $OUT (Xet disabled, 1 worker, auto-retry)"
# Single worker keeps memory/mmap pressure minimal (concurrent large-shard
# downloads were crashing the WSL VM). The retry loop resumes automatically
# after transient errors like "OSError: [Errno 5] Input/output error".
for attempt in 1 2 3 4 5 6 7 8; do
  echo "--- attempt $attempt ---"
  "$PY" - "$OUT" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
out = sys.argv[1]
p = snapshot_download(
    repo_id="DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1",
    local_dir=out,
    allow_patterns=[
        "*.safetensors", "*.safetensors.index.json",
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "preprocessor_config.json",
        "video_preprocessor_config.json", "processor_config.json",
    ],
    max_workers=1,
)
print("DONE", p)
PY
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "download completed cleanly"
    break
  fi
  echo "attempt $attempt failed (rc=$rc); retrying in 5s"
  sleep 5
done
echo "exit: $?"
echo "=== result ==="
ls -la "$OUT" | head -40
du -sh "$OUT"
df -h / | tail -1
