# Qwen3.8-27B NVFP4 on a single V100 (32 GB) — NInfer

Run **Qwen3.8-27B** at **~66 tok/s decode** on **one Tesla V100-SXM2-32GB** using
[NInfer](https://github.com/geoffwatts/ninfer-v100) (a C++/CUDA inference engine)
and the official **NVFP4** `.ninfer` artifact. OpenAI-compatible HTTP server with
streaming, function/tool calling and reasoning.

This repository is the **complete, reproducible recipe**: one command installs
prerequisites, builds NInfer for Volta (`sm_70`), applies the two source patches
we needed, and downloads the SHA256-verified model.

> **V100 = Volta = `sm_70`.** NInfer is **Linux-only** and needs **CUDA 12.8**
> (CUDA 13 removed offline compilation for Volta). This works on native Ubuntu
> 24.04 **or** Ubuntu 24.04 under WSL2 with the V100 passed through.

---

## One command

```bash
git clone https://github.com/andreasknopke/V100-Qwen3.8-27b-NVFP4.git
cd V100-Qwen3.8-27b-NVFP4
sudo ./install.sh
```

`install.sh` runs, in order:

1. system + CUDA 12.8 prerequisites (`scripts/install_deps.sh`)
2. clone + build NInfer for `sm_70` (`scripts/build_ninfer.sh`)
3. the two source patches (see below)
4. rebuild the binaries with the patches
5. download the official NVFP4 artifact, SHA256-verified (`scripts/download_model.sh`)

It is **idempotent** — every phase skips work already done, so you can re-run it.

## Start the server

```bash
./serve.sh            # foreground, prints the effective config
# or, with single-instance guards + logging:
./start_single.sh     # kills strays, waits for VRAM release, verifies 1 instance
```

Test it:

```bash
curl http://127.0.0.1:8084/v1/models
curl http://127.0.0.1:8084/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

The `model` field is **required** by NInfer (unlike llama-server it cannot be
defaulted). The id is `qwen3.8-27b` — it must match `/v1/models` exactly.

---

## What you get (measured on V100-SXM2-32GB)

| | NVFP4 (this recipe) |
|---|---|
| decode | **~66 tok/s** (single request) |
| prefill | ~1,030–1,260 tok/s steady-state |
| GPU weights | ~20.0 GiB |
| max context | **212,992** (or **262,144** text-only via `serve_262k.sh`) |
| KV dtype on Volta | int8 only (NVFP4/K8V4 KV unavailable on Volta) |
| MTP acceptance | ~70–79% (draft window 4, `--lm-head-draft`) |
| load time | ~15–18 s |

Two launchers are provided:

- `./serve.sh` — the standard NVFP4 server at the safe **212,992** ceiling.
- `./serve_262k.sh` — the **full 262,144 context** via five memory-fit levers
  (text-only, single concurrency). See [Full 262,144 context](#full-262144-context-serve_262ksh).

Endpoints verified: `GET /v1/models`, `POST /v1/chat/completions` non-stream and
SSE stream, with `reasoning_content`, `timings` and `usage`. Function calling
returns real `tool_calls` (NInfer parses/returns them, does not execute them).

---

## Why NVFP4 caps at 212,992 context

NVFP4 weights occupy ~20.0 GiB even though NVFP4 is "4-bit", because the
attention projections, GDN Q/K/V/Z, output head, embedding and the last MLP
layers are row-scaled **FP8**, not FP4. At `--max-context 262144` the engine
needs ~12.0 GiB of runtime reservation but only ~10.55 GiB is free after
weights, so it refuses to load:

```
minimum Engine runtime reservation requires ... 11.18 GiB ... but only
10.55 GiB are available after weights
```

`212992` is the measured safe ceiling (runtime ~8.9 GiB, free ~1.8 GiB).
Runtime scales ~40 KiB/token.

There are **two** ways to get the full 262,144 context:

- **`./serve_262k.sh`** — keep NVFP4 (fast) and squeeze the full 262,144 in
  text-only via memory-fit levers. See below.
- **groupwise-int artifact** — 16.9 GiB weights, ~44 tok/s, a different
  download. Slower but leaves more VRAM headroom for concurrency.

---

## Full 262,144 context (`serve_262k.sh`)

`./serve_262k.sh` runs NVFP4 at the **full 262,144 context** by applying five
memory-fit levers that reclaim the ~1.5 GiB the engine would otherwise reserve:

| lever | flag | effect |
|---|---|---|
| 1 | `--kv-capacity $CTX` | explicit KV drops the 1 GiB auto headroom |
| 2 | `--device-state-slots 0` | 1 state slot instead of 2 (prefix reuse still ok) |
| 3 | `--no-cuda-graph` | drops the CUDA-graph allowance |
| 4 | `--prefill-chunk 1024` | shrinks the prefill workspace |
| 5 | `VISION=0` | frees the ~0.3 GiB vision tower (text-only) |

Measured 2026-09-20: runtime **10.6 GiB**, ~190 MiB free, MTP intact.

```bash
./serve_262k.sh                          # official NVFP4 @ 262144, text-only
./serve_262k.sh 262144 mtp 4 0 8084 1 cf-nvfp4   # Cold-Fusion NVFP4 @ 262144
```

Concurrency is forced to **1** (NVFP4's fixed minimum reservation only fits one
device-state slot at this context). Vision defaults OFF at 262,144; pass
`VISION=1` and keep `ctx <= 212992` for a comfortable fit.

---

## Cold-Fusion GAIN → NVFP4 (on-the-fly quantization recipe)

[Cold-Fusion GAIN V1.1](https://huggingface.co/DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1)
is a bf16 fine-tune of Qwen3.8-27B (MTP embedded, reduced overthinking). The
official NVFP4 converter is *closed* — it only repacks a pre-quantized
compressed-tensors checkpoint. This repo ships a **derivative converter** that
quantizes the raw bf16 Cold-Fusion checkpoint **on the fly**, weight-only, so
Cold-Fusion loads exactly like the official NVFP4 artifact (single concurrency,
text-only) and decodes faster than the groupwise-int build.

**Quantization scheme (weight-only, no calibration needed on Volta):**

- `mlp.(gate|up|down)_proj` layers 0..55 → **NVFP4**: E2M1 codes + E4M3 K16
  block scale + per-tensor FP32 global divisor. Runtime dequant:
  `w = e2m1 * e4m3_scale * (1/divisor)`, with `divisor = 2688 / tensor_amax`.
- attention / GDN projections, `lm_head`, layers 56..63 MLP → **FP8 row-scaled**
  (E4M3 codes + BF16 row multipliers).
- norms, GDN small tensors, embedding, draft head, MTP, vision → the same
  official routes as `convert_nvfp4.py`, materialized straight from bf16.
- On Volta (`sm_70`) `kNvfp4InternalPolicy` is `A16Only`: the W4A4 activation
  path (which would need a calibrated `input_scale_divisor`) is never reached,
  so `*/input_scale_divisor` objects are emitted as a constant `1.0`.

**Steps:**

```bash
# 1. download the bf16 Cold-Fusion checkpoint (resumable, Xet disabled)
bash scripts/download_coldfusion.sh

# 2. quantize on the fly -> qwen3_8_27b_coldfusion_nvfp4.ninfer (~21.5 GiB)
bash scripts/build_coldfusion_nvfp4.sh

# 3. serve at the full 262,144 context (text-only, MTP)
./serve_262k.sh 262144 mtp 4 0 8084 1 cf-nvfp4
```

**Measured A/B (2026-09-20, V100, 600-token prompt, `reasoning_effort=none`):**

| | CF-NVFP4 | CF-groupwise-int |
|---|---|---|
| decode | **54.2 tok/s** | 43.5 tok/s (+24.7% for NVFP4) |
| runtime VRAM | 10.6 GiB | 12.9 GiB |
| quality (4-prompt A/B) | identical correct | identical correct |

The NVFP4 build is larger (21.5 vs 18.2 GB) and leaves almost no VRAM headroom
at 262,144 (hence `cc=1`), but it is faster and quality is preserved.

---

## The two source patches (applied automatically)

These are the only deviations from upstream NInfer. Both are idempotent and
record a `.patch` file in the build tree for audit/revert.

### 1. TCP backpressure fix — `scripts/patch_tcp_user_timeout.sh` (critical)

Upstream hardcodes `TCP_USER_TIMEOUT = 15000 ms`. That option aborts a
connection whose transmitted data stays **unacknowledged**. A client that merely
reads slower than we generate (or pauses briefly) fills its receive window, our
data goes unacked, and the kernel **RSTs the socket mid-answer**. For a
reasoning model the only thing on the wire during the long thinking phase is
`reasoning_content`, so the abort looks exactly like *"reasoning got cut off /
no answer returned"*.

The patch raises the default to **600,000 ms** and makes it overridable via
`NINFER_TCP_USER_TIMEOUT_MS` (`0` disables the option). TCP keepalive probes are
left as upstream set them, so a genuinely dead peer is still detected in ~19 s.
**Slow is no longer treated as dead.** This is why llama.cpp never showed the
bug: it does not set `TCP_USER_TIMEOUT` at all.

### 2. Low reasoning-effort default — `scripts/patch_default_low_effort.sh`

`ninfer-serve` has **no** `--reasoning-effort` flag (CLI-only) and accepts only
`chat_template_kwargs.preserve_thinking`, so a client that omits
`reasoning_effort` got the template's hardcoded default of **XHigh**. On
Qwen3.8-27B **xhigh never terminates**: it consumes 100% of the output budget on
reasoning and returns **zero** content tokens (measured: 4000/4000 reasoning,
empty answer, 130 s). The patch changes the default to **Low** in both places
that hardcode it (render fallback + reported capability).

Effort tiers exposed are **low / medium / xhigh** only — there is **no "minimal"**
tier, and `high`/`max` are rejected with HTTP 400. `reasoning_effort:"none"`
disables thinking per request.

---

## Thinking & output budget (read this if answers come back empty)

Reasoning and content **share one budget** (both draw from `max_tokens`). With
thinking on, a small `max_tokens` can be consumed entirely by reasoning, leaving
an empty completion. Two consequences:

- **VS Code Copilot sends no `max_tokens` at all**, so `--default-max-tokens`
  governs. Upstream default is 8192, which truncated agentic requests to no
  answer. We default it to **32768** (`DEFAULT_MAX_TOKENS` env to override).
- Prefer **low** or **medium** effort (both terminate). Use `none` when a fast
  answer matters more than reasoning. `xhigh` is pathological — do not expose it.

No thinking **budget cap** is applied on purpose: a cap can truncate reasoning
and leave an empty completion, which clients report as "no response returned".

---

## Concurrency

`--max-concurrency` (default **1** for NVFP4) is how many requests run at once;
the rest wait in a queue (`--max-pending-requests 16`, admitted within
`--pending-timeout-ms 600000` = 10 min before HTTP 503). A single request stays
at ~66 tok/s regardless of this setting, so a higher value costs no latency — it
only adds parallelism when requests overlap.

**Why 1 and not 3:** concurrency `N` reserves `N` active + `N` cached
device-state slots (`2N` slots). Their *minimum* reservation is a **fixed** cost
independent of `--max-context` (decoder state + GDN state image + CUDA-graph
allowance). With only ~9.55 GiB of runtime budget left after the 20.0 GiB
weights, 2 slots (~9.0 GiB) fit, but 6 slots (~10.0 GiB) **do not** — the engine
aborts startup with *"minimum Engine runtime reservation requires …"* **even at
a reduced context**, because the binding constraint is the fixed minimum, not the
per-token KV growth. Lowering the context cannot help. If you need parallelism,
either drop the context substantially or switch to the groupwise-int artifact
(16.9 GiB weights).

---

## VS Code Copilot / OpenAI-compatible clients

In `chatLanguageModels.json` the model entry needs:

- `"id": "qwen3.8-27b"` — **not empty**. VS Code sends the entry's `id` as the
  request `model`; an empty id makes Copilot omit `model` and NInfer returns
  `400 missing required field: model`.
- `supportsReasoningEffort`: `["none","low","medium"]` — **do not include
  `high` or `xhigh`** (400 / never-terminating respectively).
- Token budgets must fit the context: e.g. `maxInputTokens` 180000 /
  `maxOutputTokens` 32000 (input + output ≤ `--max-context`).

Reload the VS Code window after editing so the model picker refreshes.

---

## Layout & environment

Everything installs under `$NINFER_HOME` (default `~/ninfer`):

```
~/ninfer/                 NInfer repo (cloned by build_ninfer.sh)
~/ninfer/build-v100/apps/ ninfer, ninfer-serve   (no install target upstream)
~/ninfer/models/          qwen3_8_27b_nvfp4.ninfer
~/ninfer/models/          qwen3_8_27b_coldfusion_nvfp4.ninfer   (optional, CF recipe)
~/ninfer/models/          Qwen3.8-27B-Cold-Fusion-GAIN-V1.1/    (optional, CF bf16 source)
~/ninfer/*.patch          the two recorded source patches
```

Env overrides: `NINFER_HOME`, `JOBS`, `SKIP_DOWNLOAD`, `NINFER_MODEL`,
`DEFAULT_MAX_TOKENS`, `CONCURRENCY`, `MAX_PENDING_REQUESTS`,
`PENDING_TIMEOUT_MS`, `EXTRA_ARGS`, `NINFER_TCP_USER_TIMEOUT_MS`.

## WSL2 notes

- The V100 is visible inside WSL2 via `/usr/lib/wsl`. **Do not install a Linux
  NVIDIA driver in the distro** — WSL uses the Windows host driver. The pin file
  written by `install_deps.sh` keeps driver packages out; the script warns if any
  are present.
- Default WSL memory is 50% of host RAM. A 22 GiB model + build wants a generous
  `.wslconfig` (e.g. `memory=48GB`). If the model file is near your RAM size,
  load-time measurements are confounded by page cache.
- `pkill -f 'apps/ninfer-serve'` kills its own shell (the pattern matches the
  shell's command line). Use the bracket trick `'[a]pps/ninfer-serve'` — already
  baked into `start_single.sh`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `400 missing required field: model` | Client omitted `model`. Set the entry `id` to `qwen3.8-27b`. |
| Empty completion, `finish_reason=length` | Reasoning ate the whole budget. Raise `max_tokens` / `DEFAULT_MAX_TOKENS`, or use `reasoning_effort:"none"`. |
| Stream cut mid-reasoning | You rebuilt without the TCP patch. Re-run `scripts/patch_tcp_user_timeout.sh` + rebuild. |
| Refuses to load at 262144 | NVFP4 ceiling is ~212992. Lower `--max-context` or use groupwise-int. |
| Port LISTENing but nothing answered | Two instances loaded onto one card. Use `./start_single.sh`. |
| Download dies at ~1.3 GB | HF resets long connections. Use aria2c (installed by deps) — already the default. |

## Third-party & licenses

- **NInfer** (`geoffwatts/ninfer-v100`, a fork of `Neroued/ninfer`): Apache-2.0.
- **Model weights** (`neroued/Qwen3.8-27B-nvfp4-NInfer`, derived from
  `Qwen/Qwen3.8-27B`): subject to the Qwen license — check the Hugging Face repo.
- The scripts in this repository are MIT (see `LICENSE`). They download and build
  third-party components under their own licenses; you are responsible for
  complying with the model license.

## Disclaimer

Provided as-is, no warranty. The measured numbers are from a single
Tesla V100-SXM2-32GB and will differ on other V100 variants and hosts.
