"""Fold Sofia's LoRA (training/train.py) into the uncensored model she runs on, as a new GGUF.

    ./scripts/finetune.sh merge      (runs this in the training venv)

HauhauCS's Qwen3.5-4B is only published as GGUF, so the LoRA, trained on Qwen/Qwen3.5-4B, is added
to its BF16 GGUF directly: delta = B @ A * alpha / r on each tensor the LoRA covers. Before
writing anything, every covered tensor is checked against the same tensor of Qwen/Qwen3.5-4B:
they must be near copies (HauhauCS edits the weights only lightly), which proves the name
mapping and the orientation. Ollama then quantizes the result (scripts/finetune.sh).
"""
import json
import re
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

TRAIN = Path.home() / ".local/share/sofia/train"
SOURCE = TRAIN / "hauhau" / "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
OFFICIAL = TRAIN / "Qwen3.5-4B"
LORA = TRAIN / "lora"
OUT = TRAIN / "sofia-tuned-BF16.gguf"
MIN_SIMILARITY = 0.95  # cosine between HauhauCS's tensor and the official one

# LoRA module (Hugging Face name) -> GGUF tensor, per layer.
GGUF_NAMES = {
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "self_attn.o_proj": "attn_output",
}  # not linear_attn.out_proj: its GGUF tensor, ssm_out, has its heads reordered
_MODULE = re.compile(r"layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")


def deltas() -> dict[str, torch.Tensor]:
    """GGUF tensor name -> the change the LoRA makes to it, [out, in] in float32."""
    config = json.loads((LORA / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    weights = load_file(str(LORA / "adapter_model.safetensors"))
    parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key, value in weights.items():
        match = _MODULE.search(key)
        if not match:
            raise SystemExit(f"unexpected LoRA tensor {key}")
        layer, module, half = int(match.group(1)), match.group(2), match.group(3)
        parts.setdefault((layer, module), {})[half] = value.float()
    found = {}
    for (layer, module), halves in parts.items():
        if module not in GGUF_NAMES:
            raise SystemExit(f"the LoRA covers {module}, which can't be merged into the GGUF")
        found[f"blk.{layer}.{GGUF_NAMES[module]}.weight"] = (halves["B"] @ halves["A"]) * scale
    return found


def official(name: str) -> np.ndarray:
    """The official tensor for GGUF tensor `name`, from Qwen/Qwen3.5-4B's safetensors."""
    layer, short = re.match(r"blk\.(\d+)\.(\w+)\.weight", name).groups()
    module = next(m for m, g in GGUF_NAMES.items() if g == short)
    key = f"model.language_model.layers.{layer}.{module}.weight"
    index = json.loads((OFFICIAL / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(OFFICIAL / index[key]), framework="pt") as file:
        return file.get_tensor(key).float().numpy()


def as_float(tensor) -> np.ndarray:
    data = tensor.data
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        data = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif tensor.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        raise SystemExit(f"{tensor.name} is {tensor.tensor_type.name}; merge into the BF16 GGUF")
    return data.astype(np.float32).reshape([int(n) for n in reversed(tensor.shape)])


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)  # round to nearest even
    return (rounded >> 16).astype(np.uint16)


def main() -> None:
    changes = deltas()
    reader = gguf.GGUFReader(str(SOURCE))
    tensors = {t.name: t for t in reader.tensors}
    missing = [name for name in changes if name not in tensors]
    if missing:
        raise SystemExit(f"not in the GGUF: {missing[:5]}")
    # Prove the mapping before writing: HauhauCS's tensor must be a near copy of the official one.
    worst = 1.0
    for name in changes:
        mine, theirs = as_float(tensors[name]), official(name)
        if mine.shape != theirs.shape:
            raise SystemExit(f"{name}: GGUF {mine.shape} vs official {theirs.shape}")
        similarity = float((mine * theirs).sum() / (np.linalg.norm(mine) * np.linalg.norm(theirs)))
        worst = min(worst, similarity)
        if similarity < MIN_SIMILARITY:
            raise SystemExit(f"{name} is not the tensor the LoRA was trained on (cosine {similarity:.3f})")
    print(f"{len(changes)} tensors map onto the GGUF (worst cosine with the official weights {worst:.4f})", flush=True)

    architecture = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(str(OUT), architecture)
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=sub)
    # As gguf's own gguf_new_metadata.py copies a file: every tensor's info first, then the data.
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        data = tensor.data
        if tensor.name in changes:
            merged = to_bf16(as_float(tensor) + changes[tensor.name].numpy())
            data = merged.view(data.dtype).reshape(data.shape)
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
