#!/usr/bin/env bash
# ============================================================================
#  PATCH: stop NInfer from RESETTING the connection on client backpressure.
#
#  ROOT CAUSE (reproduced): upstream hardcodes TCP_USER_TIMEOUT to 15000 ms.
#  That option aborts a connection when transmitted data stays UNACKNOWLEDGED.
#  A client that reads slower than we generate - or pauses for a moment - fills
#  its receive window, our data goes unacked, and the kernel resets the socket
#  MID-STREAM. For a reasoning model the only thing on the wire during the long
#  thinking phase is reasoning_content, so the abort looks exactly like
#  "reasoning interrupted / no answer returned".
#
#  Change: raise the default to 600000 ms and make it overridable via
#  NINFER_TCP_USER_TIMEOUT_MS (0 = do not set the option at all). The TCP
#  keepalive probes are left as upstream set them, so a genuinely dead peer is
#  still detected in ~19 s. "Slow" is no longer treated as "dead".
#
#  Idempotent; writes a git patch file for audit/revert.
# ============================================================================
set -uo pipefail
NINFER_HOME="${NINFER_HOME:-$HOME/ninfer}"
cd "$NINFER_HOME" || exit 1

FILE=src/serve/http_transport.cpp
PATCH=ninfer-tcp-user-timeout.patch

if [ ! -f "$FILE" ]; then echo "FATAL: $FILE not found"; exit 1; fi

if grep -q 'tcp_user_timeout_milliseconds' "$FILE"; then
    echo "=== already patched, nothing to do ==="
    grep -n 'kDefaultTcpUserTimeoutMilliseconds\|NINFER_TCP_USER_TIMEOUT_MS' "$FILE"
    exit 0
fi

# --- 1. add <cstdlib> include ----------------------------------------------
python3 - "$FILE" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "#include <cstdlib>" not in s:
    s = s.replace('#include <stdexcept>', '#include <cstdlib>\n#include <stdexcept>', 1)
open(p, "w").write(s)
print("[1/4] added <cstdlib>")
PY

# --- 2. replace the constant with a tunable helper -------------------------
python3 - "$FILE" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;"
new = '''// Upstream used 15000 ms here. TCP_USER_TIMEOUT aborts a connection whose
// transmitted data stays unacknowledged, so a client that merely reads slower
// than we generate (or pauses briefly) would have its stream reset mid-answer
// with no finish_reason and no [DONE]. Slow is not dead: the TCP keepalive
// probes below still detect a genuinely dead peer. Override with
// NINFER_TCP_USER_TIMEOUT_MS (0 disables the option entirely).
constexpr unsigned int kDefaultTcpUserTimeoutMilliseconds = 600000;

unsigned int tcp_user_timeout_milliseconds() noexcept {
    if (const char* env = std::getenv("NINFER_TCP_USER_TIMEOUT_MS")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        if (end != env && end != nullptr && *end == '\\0') {
            return static_cast<unsigned int>(parsed);
        }
    }
    return kDefaultTcpUserTimeoutMilliseconds;
}'''
if old not in s:
    raise SystemExit("FATAL: constant not found - source differs from expectation")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("[2/4] replaced fixed timeout with tunable helper")
PY

# --- 3. use the helper at socket setup ------------------------------------
python3 - "$FILE" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);"
new = """    const unsigned int user_timeout_ms = tcp_user_timeout_milliseconds();
    if (user_timeout_ms != 0) {
        set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, user_timeout_ms);
    }"""
if old not in s:
    raise SystemExit("FATAL: setsockopt line not found")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("[3/4] applied tunable timeout at socket setup")
PY

echo "[4/4] diff:"
git diff -- "$FILE" > "$PATCH"
cat "$PATCH"
echo
echo "patch recorded in $NINFER_HOME/$PATCH"
echo "PATCH_DONE"
