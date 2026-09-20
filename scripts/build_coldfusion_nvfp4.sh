#!/usr/bin/env bash
# ============================================================================
#  Build a NInfer NVFP4 artifact from the Cold-Fusion GAIN fine-tune (MTP-only).
#
#  Quantizes the raw bf16 checkpoint on the fly (weight-only NVFP4/FP8), so the
#  artifact loads like the official NVFP4 model (20 GiB weights -> single
#  concurrency, text-only) and decodes faster than the groupwise-int build.
#
#  Runs INSIDE WSL Ubuntu (or native Linux). Requires:
#    * the NInfer repo + build (scripts/build_ninfer.sh)
#    * the Cold-Fusion bf16 safetensors checkpoint (scripts/download_coldfusion.sh)
#    * a torch environment with CUDA (e.g. a venv)
#
#  Usage: build_coldfusion_nvfp4.sh [CHECKPOINT_DIR] [OUT_ARTIFACT]
#    CHECKPOINT_DIR  default $NINFER_HOME/models/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1
#    OUT_ARTIFACT    default $NINFER_HOME/models/qwen3_8_27b_coldfusion_nvfp4.ninfer
#
#  Env overrides: NINFER_HOME, PY (python interpreter), REPO_DIR (this repo).
# ============================================================================
set -uo pipefail

NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"
# Locate this repo (the directory containing scripts/).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CKPT="${1:-$NINFER_HOME/models/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1}"
OUT="${2:-$NINFER_HOME/models/qwen3_8_27b_coldfusion_nvfp4.ninfer}"
PY="${PY:-$(command -v python3)}"

# Stage the derivative converter into the package tree (idempotent).
install -m 644 "$REPO_DIR/scripts/convert_coldfusion_nvfp4.py" \
    "$NINFER_HOME/tools/convert/qwen3_8_27b/convert_coldfusion_nvfp4.py" 2>/dev/null || true
sed -i 's/\r$//' "$NINFER_HOME/tools/convert/qwen3_8_27b/convert_coldfusion_nvfp4.py" 2>/dev/null || true

if [ ! -d "$CKPT" ]; then
    echo "FATAL: checkpoint dir not found: $CKPT"
    echo "Download first:  bash $REPO_DIR/scripts/download_coldfusion.sh"
    exit 1
fi
if ! ls "$CKPT"/model-*.safetensors >/dev/null 2>&1; then
    echo "FATAL: no model-*.safetensors in $CKPT (download incomplete?)"
    exit 1
fi

# Same tokenizer fix as the groupwise build: the runtime frontend rejects a
# tokenizer config whose add_bos_token is missing (defaults absent -> true).
# Official Qwen sets add_bos_token=false; Cold-Fusion omits it. Patch the SOURCE
# before conversion so the corrected resource is embedded in the .ninfer.
"$PY" - "$CKPT" <<'PY'
import json, sys, os
p = os.path.join(sys.argv[1], 'tokenizer_config.json')
d = json.load(open(p))
if d.get('add_bos_token', 'MISSING') is not False:
    d['add_bos_token'] = False
    json.dump(d, open(p, 'w'), ensure_ascii=False, indent=2)
    print("patched tokenizer_config.json: add_bos_token -> false")
else:
    print("tokenizer_config.json already has add_bos_token=false")
PY

echo "=============================================="
echo " Cold-Fusion GAIN -> NInfer NVFP4 artifact (MTP-only)"
echo "=============================================="
echo " checkpoint : $CKPT"
echo " out        : $OUT"
echo " python     : $PY"
echo "=============================================="

cd "$NINFER_HOME" || exit 1
exec "$PY" -m tools.convert.qwen3_8_27b.convert_coldfusion_nvfp4 \
    --model "$CKPT" --out "$OUT" --device cuda
