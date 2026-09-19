#!/usr/bin/env bash
# ============================================================================
#  Patch: make NInfer's default reasoning effort "low" instead of "xhigh".
#
#  Why: ninfer-serve has NO --reasoning-effort flag (CLI-only) and accepts only
#  chat_template_kwargs.preserve_thinking, so a client that omits reasoning_effort
#  gets the template's hardcoded default of XHigh. On Qwen3.8-27B xhigh never
#  terminates: it consumes 100% of the output budget on reasoning and returns
#  ZERO content tokens (measured: 4000/4000 reasoning, empty answer, 130 s).
#  Two places hardcode it:
#    1. render() fallback  -> value_or(ReasoningEffort::XHigh)
#    2. capabilities()     -> default_effort = ReasoningEffort::XHigh
#  Both must change for the rendered prompt AND the reported/logged effective
#  effort to agree.
#
#  Idempotent: re-running makes no further changes. Saves a git patch so the
#  fork stays auditable and revertible (git checkout <file> to restore).
# ============================================================================
set -uo pipefail
NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"
cd "$NINFER_HOME" || exit 1

FILE=src/targets/qwen3_6/impl/frontend/chat_template.cpp
PATCH=ninfer-default-low-effort.patch

if [ ! -f "$FILE" ]; then echo "FATAL: $FILE not found"; exit 1; fi

echo "=== before ==="
grep -n 'ReasoningEffort::XHigh' "$FILE" || echo "(no XHigh occurrences)"

changed=0
if grep -q 'value_or(ReasoningEffort::XHigh)' "$FILE"; then
    sed -i 's/value_or(ReasoningEffort::XHigh)/value_or(ReasoningEffort::Low)/' "$FILE"
    changed=1
fi
if grep -q 'default_effort = ReasoningEffort::XHigh;' "$FILE"; then
    sed -i 's/default_effort = ReasoningEffort::XHigh;/default_effort = ReasoningEffort::Low;/' "$FILE"
    changed=1
fi

if [ "$changed" -eq 0 ]; then
    echo "=== already patched, nothing to do ==="
else
    echo "=== patch applied ==="
fi

echo "=== after ==="
grep -n 'ReasoningEffort::XHigh\|ReasoningEffort::Low' "$FILE"

echo
echo "=== recording patch for audit / revert ==="
git diff -- "$FILE" > "$PATCH"
wc -l "$PATCH"

echo
echo "=== verifying no other XHigh default remains ==="
if grep -rn 'default_effort = ReasoningEffort::XHigh' src/ 2>/dev/null; then
    echo "WARNING: another XHigh default exists outside $FILE"
else
    echo "clean: no remaining XHigh default_effort"
fi
echo "PATCH_DONE"
