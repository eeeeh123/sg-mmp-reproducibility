"""Constants and fail-closed helpers for deployment-gguf-v1."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from experiments.revision_full.protocol import (
    CALIB_LENGTH,
    CALIB_SAMPLES,
    CALIB_SEEDS,
    GSM8K_TEST_SIZE,
    MAX_NEW_TOKENS,
    MODEL_SPECS,
    PROTOCOL_VERSION as SOURCE_PROTOCOL_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / "protocol_lock.json"
PROTOCOL_VERSION = "deployment-gguf-v1"
LLAMA_CPP_COMMIT = "050dde50c9d70cf207db84f7224eedc491d817b2"
REQUIRED_CMAKE_TOOLCHAIN = {
    "CMAKE_CUDA_COMPILER": "/usr/local/cuda-12.4/bin/nvcc",
    "CMAKE_CXX_COMPILER": "/usr/bin/g++-12",
    "CUDAToolkit_NVCC_EXECUTABLE": "/usr/local/cuda-12.4/bin/nvcc",
}
BACKEND_TEST_TIMEOUT_SECONDS = 900
METHODS = ("fp16", "q4", "q5", "sg")
QUANTIZED_METHODS = ("q4", "q5", "sg")
PHASE_MODELS = {
    "engineering": ("qwen05",),
    "value": ("qwen15", "smollm"),
    "formal": ("qwen05", "gemma2"),
}
PILOT_BLOCKS = 5
PILOT_REPETITIONS = 5
FORMAL_MIN_BLOCKS = 10
FORMAL_MAX_BLOCKS = 30
PERFORMANCE_GENERATED_TOKENS = 128
SERVICE_CONCURRENCY = (1, 4)
MICRO_PROMPT_TOKENS = (128, 512, 1024)
CPU_THREADS = 8
CONTEXT_TOKENS_PER_SLOT = 4096
IMATRIX_OUTPUT_FREQUENCY = 10
IMATRIX_SAVE_FREQUENCY = 0


def _configured_output_dir() -> Path:
    raw = os.environ.get("DEPLOYMENT_GGUF_OUTPUT_DIR")
    path = Path(raw).expanduser() if raw else HERE / "outputs"
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


OUT = _configured_output_dir()
ARTIFACT_DIR = OUT / "artifacts"
MANIFEST_DIR = OUT / "manifests"
CALIBRATION_DIR = OUT / "calibration"
QUALITY_DIR = OUT / "quality"
BENCH_DIR = OUT / "benchmarks"
STATUS_DIR = OUT / "status"


def protocol_lock() -> dict:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Protocol lock mismatch: {LOCK_PATH}")
    if value.get("llama_cpp", {}).get("commit") != LLAMA_CPP_COMMIT:
        raise RuntimeError("Pinned llama.cpp commit disagrees with code")
    if value.get("llama_cpp", {}).get("cmake_toolchain") != REQUIRED_CMAKE_TOOLCHAIN:
        raise RuntimeError("Pinned llama.cpp toolchain disagrees with code")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def quantization_policy_sha256(method: str) -> str:
    """Bind a packed artifact to the exact method and tensor policy."""
    if method not in QUANTIZED_METHODS:
        raise ValueError(f"No packed quantization policy for method {method!r}")
    lock = protocol_lock()
    return json_sha256(
        {
            "method": lock["methods"][method],
            "tensor_policy": lock["tensor_policy"],
        }
    )


def conversion_gate_policy_sha256() -> str:
    """Bind conversion evidence to the exact currently locked gate semantics."""
    return json_sha256(protocol_lock()["gates"]["conversion"])


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_path(model_key: str, method: str) -> Path:
    require_model_method(model_key, method)
    return ARTIFACT_DIR / model_key / f"{model_key}__{method}.gguf"


def artifact_manifest_path(model_key: str, method: str) -> Path:
    require_model_method(model_key, method)
    return MANIFEST_DIR / "artifacts" / model_key / f"{method}.json"


def build_registration_path(model_key: str, method: str) -> Path:
    require_model_method(model_key, method)
    return MANIFEST_DIR / "build_commands" / model_key / f"{method}.json"


def source_fp16_path(model_key: str) -> Path:
    return artifact_path(model_key, "fp16")


def selection_path(model_key: str) -> Path:
    return (
        ROOT
        / "experiments"
        / "revision_full"
        / "outputs"
        / "selections"
        / f"{model_key}.json"
    )


def imatrix_path(model_key: str) -> Path:
    return CALIBRATION_DIR / model_key / "imatrix.gguf"


def calibration_corpus_path(model_key: str) -> Path:
    return CALIBRATION_DIR / model_key / "wikitext_union_c41_c97_c193.txt"


def require_model_method(model_key: str, method: str) -> None:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model_key!r}")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")


def load_frozen_selection(model_key: str) -> dict:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model_key!r}")
    path = selection_path(model_key)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen revision selection: {path}. Copy/preserve the completed "
            "revision-full-v4 outputs before running this extension."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError(f"Selection does not use {SOURCE_PROTOCOL_VERSION}: {path}")
    if value.get("model_key") != model_key or value.get("test_data_used") is not False:
        raise RuntimeError(f"Selection is not the test-clean {model_key} record: {path}")
    rows = value.get("module_rows")
    selected = value.get("w8_module_names")
    if not isinstance(rows, list) or not rows or not isinstance(selected, list):
        raise RuntimeError(f"Selection lacks module rows or W8 names: {path}")
    row_names = [str(row.get("name")) for row in rows]
    if len(row_names) != len(set(row_names)):
        raise RuntimeError(f"Selection contains duplicate modules: {path}")
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"Selection contains duplicate W8 modules: {path}")
    if not set(selected).issubset(row_names):
        raise RuntimeError(f"Selection references unknown modules: {path}")
    return value


def binary_paths(llama_cpp_dir: Path) -> dict[str, Path]:
    root = llama_cpp_dir.resolve()
    names = {
        "quantize": "llama-quantize",
        "imatrix": "llama-imatrix",
        "bench": "llama-bench",
        "server": "llama-server",
        "cli": "llama-cli",
    }
    return {key: root / "build" / "bin" / name for key, name in names.items()}
