"""Minimal GGUF reader and exact HF-to-GGUF tensor policy audit.

The reader intentionally supports metadata and tensor directories only.  It
never loads model payloads, so a multi-gigabyte artifact can be audited with a
small and predictable memory footprint.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from experiments.deployment_gguf.protocol import (
    PROTOCOL_VERSION,
    artifact_manifest_path,
    artifact_path,
    atomic_write_json,
    build_registration_path,
    imatrix_path,
    load_frozen_selection,
    protocol_lock,
    quantization_policy_sha256,
    selection_path,
    sha256_file,
    source_fp16_path,
)


GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

# Enum numbers and block sizes are part of the GGML on-disk ABI.  Only types
# needed by this protocol receive byte-size formulas; unfamiliar non-target
# types remain visible in the manifest and cannot silently satisfy a gate.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}

HF_MODULE_RE = re.compile(
    r"^(?:model\.)?layers\.(?P<layer>\d+)\."
    r"(?P<family>self_attn|mlp)\."
    r"(?P<short>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
SHORT_TO_GGUF = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_output",
    "gate_proj": "ffn_gate",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type_id: int
    offset: int

    @property
    def n_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type_id, (f"UNKNOWN_{self.ggml_type_id}", 0, 0))[0]

    @property
    def logical_bytes(self) -> int | None:
        record = GGML_TYPES.get(self.ggml_type_id)
        if record is None:
            return None
        _, block, type_bytes = record
        if self.n_elements % block:
            raise RuntimeError(
                f"Tensor {self.name} has {self.n_elements} elements, not divisible "
                f"by {self.type_name} block size {block}"
            )
        return self.n_elements // block * type_bytes


class _Reader:
    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("rb")

    def close(self) -> None:
        self.stream.close()

    def unpack(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        raw = self.stream.read(size)
        if len(raw) != size:
            raise RuntimeError(f"Unexpected EOF in {self.path}")
        values = struct.unpack("<" + fmt, raw)
        return values[0] if len(values) == 1 else values

    def string(self) -> str:
        length = self.unpack("Q")
        raw = self.stream.read(length)
        if len(raw) != length:
            raise RuntimeError(f"Unexpected EOF in GGUF string: {self.path}")
        return raw.decode("utf-8")

    def value(self, value_type: int):
        scalar = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }
        if value_type in scalar:
            return self.unpack(scalar[value_type])
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type = self.unpack("I")
            length = self.unpack("Q")
            return [self.value(element_type) for _ in range(length)]
        raise RuntimeError(
            f"Unsupported GGUF metadata value type {value_type} "
            f"({GGUF_VALUE_TYPES.get(value_type, 'unknown')})"
        )


def read_gguf(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = _Reader(path)
    try:
        if reader.stream.read(4) != b"GGUF":
            raise RuntimeError(f"Not a GGUF file: {path}")
        version = reader.unpack("I")
        if version not in (2, 3):
            raise RuntimeError(f"Unsupported GGUF version {version}: {path}")
        tensor_count = reader.unpack("Q")
        metadata_count = reader.unpack("Q")
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            value_type = reader.unpack("I")
            metadata[key] = reader.value(value_type)
        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.unpack("I")
            if not 1 <= n_dims <= 4:
                raise RuntimeError(f"Invalid GGUF tensor rank {n_dims} for {name}: {path}")
            shape = tuple(reader.unpack("Q") for _ in range(n_dims))
            if any(dimension <= 0 for dimension in shape):
                raise RuntimeError(f"Invalid GGUF tensor shape {shape} for {name}: {path}")
            ggml_type_id = reader.unpack("I")
            offset = reader.unpack("Q")
            tensors.append(TensorInfo(name, shape, ggml_type_id, offset))
        alignment = int(metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise RuntimeError(f"Invalid GGUF alignment {alignment}: {path}")
        directory_end = reader.stream.tell()
        data_start = (directory_end + alignment - 1) // alignment * alignment
    finally:
        reader.close()
    if len({tensor.name for tensor in tensors}) != len(tensors):
        raise RuntimeError(f"GGUF contains duplicate tensor names: {path}")
    offsets = [tensor.offset for tensor in tensors]
    if offsets != sorted(offsets):
        raise RuntimeError(f"GGUF tensor offsets are not monotonic: {path}")
    file_bytes = path.stat().st_size
    if data_start > file_bytes:
        raise RuntimeError(f"GGUF ends before its tensor payload: {path}")
    previous_end = 0
    for tensor in tensors:
        if tensor.offset % alignment:
            raise RuntimeError(
                f"Unaligned GGUF tensor payload at {tensor.name}: {path}"
            )
        logical_bytes = tensor.logical_bytes
        if logical_bytes is None:
            continue
        if tensor.offset < previous_end:
            raise RuntimeError(f"Overlapping GGUF tensor payload at {tensor.name}: {path}")
        previous_end = tensor.offset + logical_bytes
        if data_start + previous_end > file_bytes:
            raise RuntimeError(f"Truncated GGUF tensor payload at {tensor.name}: {path}")
    return {
        "path": path,
        "version": version,
        "metadata": metadata,
        "alignment": alignment,
        "data_start": data_start,
        "tensors": tensors,
    }


def hf_to_gguf_tensor(module_name: str) -> str:
    match = HF_MODULE_RE.fullmatch(module_name)
    if match is None:
        raise ValueError(f"Unsupported eligible HF module name: {module_name}")
    short = match.group("short")
    return f"blk.{int(match.group('layer'))}.{SHORT_TO_GGUF[short]}.weight"


def tensor_override(module_name: str, quant_type: str = "q8_0") -> str:
    exact = re.escape(hf_to_gguf_tensor(module_name))
    return f"^{exact}$={quant_type}"


def expected_type(method: str, selected: bool) -> str:
    method_policy = protocol_lock()["methods"].get(method)
    if method_policy is None:
        raise ValueError(method)
    if method == "sg" and selected:
        return str(method_policy["selected_type"])
    return str(method_policy["base_type"])


def audit_artifact(model_key: str, method: str) -> dict:
    lock = protocol_lock()
    selection = load_frozen_selection(model_key)
    path = artifact_path(model_key, method)
    registration_path = build_registration_path(model_key, method)
    if not registration_path.is_file():
        raise RuntimeError(f"Artifact lacks immutable build registration: {path}")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if (
        registration.get("protocol_version") != PROTOCOL_VERSION
        or registration.get("model_key") != model_key
        or registration.get("method") != method
        or registration.get("artifact_sha256") != sha256_file(path)
    ):
        raise RuntimeError(f"Artifact build registration mismatch: {path}")
    if method != "fp16":
        policy_sha256 = quantization_policy_sha256(method)
        if registration.get("quantization_policy_sha256") != policy_sha256:
            raise RuntimeError(
                "Packed artifact was built under a different quantization policy; "
                f"preserve it as diagnostic evidence and rebuild: {path}"
            )
    gguf = read_gguf(path)
    by_name = {tensor.name: tensor for tensor in gguf["tensors"]}
    selected_hf = set(selection["w8_module_names"])
    selected_gguf = {hf_to_gguf_tensor(name) for name in selected_hf}
    eligible_gguf = {
        hf_to_gguf_tensor(str(row["name"])): row for row in selection["module_rows"]
    }
    if len(eligible_gguf) != len(selection["module_rows"]):
        raise RuntimeError("HF-to-GGUF mapping is not one-to-one")

    missing = sorted(set(eligible_gguf) - set(by_name))
    if missing:
        raise RuntimeError(f"Eligible tensors missing from {path}: {missing[:8]}")

    tensor_records = []
    eligible_payload_bytes = 0
    eligible_parameters = 0
    total_payload_bytes = 0
    total_elements = 0
    for tensor in gguf["tensors"]:
        logical_bytes = tensor.logical_bytes
        if logical_bytes is None:
            raise RuntimeError(
                f"Cannot account type {tensor.type_name} for tensor {tensor.name}"
            )
        total_payload_bytes += logical_bytes
        total_elements += tensor.n_elements
        row = eligible_gguf.get(tensor.name)
        eligible = row is not None
        selected = tensor.name in selected_gguf
        expected = expected_type(method, selected) if eligible else None
        if eligible and tensor.n_elements != int(row["n_params"]):
            raise RuntimeError(
                f"Parameter count mismatch for {tensor.name}: GGUF={tensor.n_elements}, "
                f"selection={row['n_params']}"
            )
        if eligible and tensor.type_name != expected:
            raise RuntimeError(
                f"Wrong type for {tensor.name}: {tensor.type_name}, expected {expected}"
            )
        if eligible:
            eligible_payload_bytes += logical_bytes
            eligible_parameters += tensor.n_elements
        tensor_records.append(
            {
                "gguf_name": tensor.name,
                "hf_module": None if row is None else row["name"],
                "shape": list(tensor.shape),
                "parameters": tensor.n_elements,
                "ggml_type": tensor.type_name,
                "payload_bytes": logical_bytes,
                "eligible": eligible,
                "frozen_sg_selected": selected,
                "selection_basis": (
                    "revision-full-v4 train-only frozen W8 set" if selected else None
                ),
            }
        )

    embedding = by_name.get("token_embd.weight")
    output = by_name.get("output.weight")
    if method != "fp16":
        if embedding is None or embedding.type_name != "F16":
            raise RuntimeError("Quantized artifact must store token_embd.weight as F16")
        if output is not None and output.type_name != "F16":
            raise RuntimeError("Stored output.weight must be F16")

    artifact_bytes = path.stat().st_size
    metadata_bytes = artifact_bytes - total_payload_bytes
    if metadata_bytes < 0:
        raise RuntimeError(f"GGUF payload accounting exceeds file size: {path}")
    record = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "method": method,
        "method_label": lock["methods"][method]["label"],
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "build_registration": str(registration_path),
        "build_registration_sha256": sha256_file(registration_path),
        "quantization_policy_sha256": (
            None if method == "fp16" else quantization_policy_sha256(method)
        ),
        "artifact_bytes": artifact_bytes,
        "tensor_payload_bytes": total_payload_bytes,
        "container_metadata_and_alignment_bytes": metadata_bytes,
        "stored_tensor_elements": total_elements,
        "whole_stored_payload_bits_per_element": total_payload_bytes * 8 / total_elements,
        "eligible_parameters": eligible_parameters,
        "eligible_payload_bytes": eligible_payload_bytes,
        "packed_eligible_bits_per_weight": eligible_payload_bytes * 8 / eligible_parameters,
        "original_gptq_logical_bits_per_weight": float(selection["actual_avg_bits"]),
        "selection": str(selection_path(model_key)),
        "selection_sha256": sha256_file(selection_path(model_key)),
        "source_fp16": str(source_fp16_path(model_key)),
        "source_fp16_sha256": sha256_file(source_fp16_path(model_key)),
        "imatrix": None if method == "fp16" else str(imatrix_path(model_key)),
        "imatrix_sha256": (
            None if method == "fp16" else sha256_file(imatrix_path(model_key))
        ),
        "tied_or_shared_output": output is None,
        "token_embedding_type": None if embedding is None else embedding.type_name,
        "output_tensor_type": None if output is None else output.type_name,
        "gguf_version": gguf["version"],
        "gguf_alignment": gguf["alignment"],
        "tensor_count": len(tensor_records),
        "eligible_tensor_count": len(eligible_gguf),
        "selected_tensor_count": len(selected_gguf),
        "tensors": tensor_records,
        "gate_passed": True,
    }
    atomic_write_json(artifact_manifest_path(model_key, method), record)
    return record


def audit_model_set(model_key: str) -> dict:
    records = {method: audit_artifact(model_key, method) for method in ("fp16", "q4", "q5", "sg")}
    signatures = {}
    for method in ("q4", "q5", "sg"):
        signatures[method] = {
            row["gguf_name"]: (tuple(row["shape"]), row["ggml_type"])
            for row in records[method]["tensors"]
            if not row["eligible"]
        }
    if set(signatures["q4"]) != set(signatures["q5"]) or set(signatures["q4"]) != set(signatures["sg"]):
        raise RuntimeError("Quantized artifacts do not contain the same non-eligible tensors")
    for name in signatures["q4"]:
        if len({signatures[method][name] for method in signatures}) != 1:
            raise RuntimeError(f"Non-eligible tensor policy differs for {name}")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "methods": {
            method: {
                "artifact_sha256": record["artifact_sha256"],
                "artifact_bytes": record["artifact_bytes"],
                "packed_eligible_bits_per_weight": record["packed_eligible_bits_per_weight"],
                "whole_stored_payload_bits_per_element": record["whole_stored_payload_bits_per_element"],
            }
            for method, record in records.items()
        },
        "noneligible_policy_identical_across_quantized_methods": True,
        "gate_passed": True,
    }
    atomic_write_json(manifest_set_path(model_key), summary)
    return summary


def manifest_set_path(model_key: str) -> Path:
    return artifact_manifest_path(model_key, "fp16").parents[1] / f"{model_key}__set.json"
