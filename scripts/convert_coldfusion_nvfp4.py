"""Build a Qwen3.8-27B Cold-Fusion (GAIN) NVFP4 artifact for NInfer, MTP-only.

This is a *derivative* converter, analogous to ``convert_coldfusion.py`` but for
the NVFP4 weight layout instead of groupwise-int. It lets Cold-Fusion load the
same way the official NVFP4 artifact does (20 GiB weights -> single concurrency,
text-only), which is markedly faster to decode than the groupwise-int path.

Unlike the closed ``convert_nvfp4.py`` -- which reads a *pre-quantized*
compressed-tensors checkpoint and only repacks it -- this converter quantizes
the raw bf16 Cold-Fusion checkpoint **on the fly**, weight-only:

  * ``mlp.(gate|up|down)_proj`` for layers 0..55  -> NVFP4 (E2M1 + E4M3 K16
    block scale + a per-tensor FP32 global divisor), matching the runtime
    dequant ``w = e2m1 * e4m3_scale * (1/divisor)``.
  * attention / GDN projections, ``lm_head`` and layers 56..63 MLP -> FP8
    row-scaled (E4M3 codes + BF16 row multipliers).
  * norms, GDN small tensors, embedding, draft head, MTP, vision -> the same
    official routes as ``convert_nvfp4.py`` (direct / fp8-embedding / W8),
    materialized straight from the bf16 checkpoint.

On Volta (sm_70) ``kNvfp4InternalPolicy`` is ``A16Only``: the W4A4 activation
path (which would need a calibrated ``input_scale_divisor``) is never reached,
so the ``*/input_scale_divisor`` objects are emitted as a constant 1.0 and are
never read numerically. Weight-only quantization is therefore sound here.

The chat template is byte-identical to official Qwen3.8-27B, so the
low/medium/xhigh reasoning-effort surface is unchanged.

Canonical invocation (run from the NInfer repo root, after this file has been
staged into the package tree by build_coldfusion_nvfp4.sh)::

    python3 -m tools.convert.qwen3_8_27b.convert_coldfusion_nvfp4 \
      --model /path/to/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1 \
      --out out/qwen3_8_27b_coldfusion_nvfp4.ninfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import (
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import draft_head

from . import convert as base_convert
from . import fp8_embedding
from . import inventory_nvfp4 as inventory
from . import recipe_nvfp4 as recipe

RECIPE_ID = "qwen3_8_27b_coldfusion-nvfp4-mtp-v1"

# NVFP4 two-level scaling constant: e2m1 max magnitude (6) times e4m3 max finite
# value (448). The per-tensor global divisor D = NVFP4_GLOBAL / tensor_amax keeps
# every K16 block scale representable as a nonnegative finite E4M3 word.
NVFP4_GLOBAL = 2688.0
FP8_ROW_MAX = 448.0

# e2m1 (FP4) magnitude table, code index 0..7 (bit 3 carries the sign).
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_E2M1_BOUNDS = (_E2M1_MAG[:-1] + _E2M1_MAG[1:]) / 2.0  # 7 rounding boundaries

# MTP-only object set: the frontend resources plus every non-DFlash2 tensor.
# ``BASE_TENSOR_SPECS`` already excludes the DFlash2 bundle (it lives in
# ``TENSOR_SPECS``), so resources + base specs is exactly the MTP-only plan.
MTP_ONLY_OBJECT_SPECS: tuple = tuple(inventory.RESOURCE_SPECS) + tuple(
    inventory.BASE_TENSOR_SPECS
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_resources_unpinned(model_dir: Path) -> dict[str, bytes]:
    """Load the frontend resources without the official-SHA gate.

    The runtime only requires internal consistency (tokenizer_config.json's
    embedded chat template must equal chat_template.jinja); it does not pin any
    frontend hash. Cold-Fusion ships the identical template, so this is safe.
    """
    resources = family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)
    return {resource.name: resource.data for resource in resources}


# ---------------------------------------------------------------------------
# bf16 matrix fusion: concatenate the bf16 source rows a recipe selects into one
# [N, K] matrix, exactly mirroring recipe_nvfp4's RowRange/MatrixPart geometry.
# ---------------------------------------------------------------------------
def _select_rows(tensor: torch.Tensor, part: recipe.MatrixPart) -> torch.Tensor:
    pieces = [
        tensor.narrow(0, row_range.begin, row_range.rows)
        for row_range in part.rows
    ]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)


def _materialize_bf16_matrix(
    parts: tuple[recipe.MatrixPart, ...],
    reader: ShardReader,
    device: torch.device,
) -> torch.Tensor:
    pieces = []
    for part in parts:
        source = reader.get(part.source.field("weight")).to(
            device=device, dtype=torch.float32
        )
        pieces.append(_select_rows(source, part))
    matrix = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    return matrix


# ---------------------------------------------------------------------------
# Weight-only quantizers (validated by cf_nvfp4_roundtrip.py).
# ---------------------------------------------------------------------------
def _quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    tensor_amax = weight.abs().amax().clamp_min(1e-12)
    divisor = NVFP4_GLOBAL / tensor_amax
    groups = weight.reshape(n, k // 16, 16)
    amax_g = groups.abs().amax(dim=2)  # [N, K/16]
    scale_unenc = divisor * amax_g / 6.0
    scale_e4m3 = scale_unenc.to(torch.float8_e4m3fn)
    scale_words = (scale_e4m3.view(torch.uint8)) & 0x7F
    decoded_scale = scale_e4m3.to(torch.float32)
    scaled = groups * divisor / decoded_scale.unsqueeze(2).clamp_min(1e-30)
    scaled = scaled.reshape(n, k).clamp(-6.0, 6.0)
    mag = scaled.abs()
    idx = torch.bucketize(mag, _E2M1_BOUNDS.to(mag.device))  # 0..7
    codes = idx.to(torch.uint8)
    codes = codes | (((scaled < 0) & (idx != 0)).to(torch.uint8) << 3)
    lo = codes[:, 0::2] & 0x0F
    hi = codes[:, 1::2] & 0x0F
    packed = (lo | (hi << 4)).to(torch.uint8)  # [N, K/2]
    div_t = divisor.detach().to(dtype=torch.float32).reshape(())
    return packed, scale_words, div_t


def _quantize_fp8_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_amax = weight.abs().amax(dim=1).clamp_min(1e-12)  # [N]
    scale = (row_amax / FP8_ROW_MAX).to(torch.bfloat16)
    codes = (weight / scale.float().unsqueeze(1)).to(torch.float8_e4m3fn)
    return codes.view(torch.uint8), scale


def _encode_nvfp4_from_bf16(
    spec, parts, reader: ShardReader, device: torch.device
) -> bytes:
    matrix = _materialize_bf16_matrix(parts, reader, device)
    if tuple(matrix.shape) != tuple(spec.shape):
        raise ValueError(
            f"{spec.name}: fused shape {tuple(matrix.shape)} != {tuple(spec.shape)}"
        )
    packed, scales, divisor = _quantize_nvfp4(matrix)
    del matrix
    return encode_nvfp4(
        packed.cpu(), scales.cpu(), divisor.cpu(), tuple(spec.shape)
    )


def _encode_fp8_from_bf16(
    spec, parts, reader: ShardReader, device: torch.device
) -> bytes:
    matrix = _materialize_bf16_matrix(parts, reader, device)
    if tuple(matrix.shape) != tuple(spec.shape):
        raise ValueError(
            f"{spec.name}: fused shape {tuple(matrix.shape)} != {tuple(spec.shape)}"
        )
    codes, scales = _quantize_fp8_row(matrix)
    del matrix
    return encode_fp8_row_scaled(
        codes.cpu(), scales.cpu(), tuple(spec.shape)
    )


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | object = "cuda",
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    resolved_device = pick_device(device)

    config_summary = base_convert.qwen3_6_convert.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    resources = load_resources_unpinned(model)
    object_plan = family_conversion.build_object_plan(
        MTP_ONLY_OBJECT_SPECS, resources
    )
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)

    print(
        f"preflight complete: {len(object_plan.objects)} objects (MTP-only NVFP4), "
        f"device={resolved_device}",
        flush=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    total = len(MTP_ONLY_OBJECT_SPECS)
    index = 0
    one = torch.ones((), dtype=torch.float32)
    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        object_plan.specs,
    ) as writer:
        for spec in inventory.RESOURCE_SPECS:
            index += 1
            writer.write(spec.name, resources[spec.name])
            print(f"[{index}/{total}] {spec.name}", flush=True)

        draft_ids = draft_head.materialize_draft_head_token_ids(draft)
        derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}

        with ShardReader(model) as reader:
            for spec in inventory.BASE_TENSOR_SPECS:
                index += 1
                name = spec.name
                payload: bytes
                if name == "text/token_embedding":
                    payload = fp8_embedding.iter_reader_payload(
                        reader,
                        recipe.OFFICIAL_EMBEDDING_SOURCE.name,
                        spec.shape,
                    )
                elif name in recipe.FP8_WEIGHTS_BY_NAME:
                    fp8_recipe = recipe.FP8_WEIGHTS_BY_NAME[name]
                    payload = _encode_fp8_from_bf16(
                        spec, fp8_recipe.parts, reader, resolved_device
                    )
                elif name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    nvfp4_recipe = recipe.NVFP4_WEIGHTS_BY_NAME[name]
                    payload = _encode_nvfp4_from_bf16(
                        spec, nvfp4_recipe.parts, reader, resolved_device
                    )
                elif name in recipe.INPUT_DIVISORS_BY_NAME:
                    # Volta A16Only never reads this numerically; emit 1.0.
                    payload = encode_direct(one, inventory.FP32)
                elif name in recipe.QUANTIZED_DIRECT_BY_NAME:
                    tensor = recipe.materialize_quantized_direct(name, reader)
                    payload = encode_direct(tensor, spec.format)
                    del tensor
                else:
                    # draft head, MTP, vision: official routes off the bf16 source.
                    tensor = recipe.materialize_official(name, reader, dict(derived))
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(name, payload)
                del payload
                print(f"[{index}/{total}] {name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report = {
        "recipe_id": RECIPE_ID,
        "identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
        "target_key": inventory.TARGET_KEY,
        "speculative_backend": "mtp",
        "quantization": "weight-only bf16 -> NVFP4/FP8 (on the fly)",
        "dflash2": None,
        "source": {
            "model_path": str(model.resolve()),
            "note": "Cold-Fusion GAIN fine-tune bf16; frontend template identical to official Qwen3.8-27B",
        },
        "config_summary": config_summary,
        "objects": len(object_plan.objects),
        "final_bytes": final_bytes,
        "elapsed_seconds": round(elapsed, 1),
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
