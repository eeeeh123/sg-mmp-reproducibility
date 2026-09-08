"""Build commands and immutable registrations for GGUF artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.deployment_gguf.gguf_manifest import (
    audit_artifact,
    tensor_override,
)
from experiments.deployment_gguf.protocol import (
    CALIB_LENGTH,
    CALIB_SAMPLES,
    CALIB_SEEDS,
    CPU_THREADS,
    LLAMA_CPP_COMMIT,
    MANIFEST_DIR,
    MODEL_SPECS,
    OUT,
    PROTOCOL_VERSION,
    artifact_path,
    atomic_write_json,
    binary_paths,
    build_registration_path,
    calibration_corpus_path,
    imatrix_path,
    load_frozen_selection,
    selection_path,
    sha256_file,
    source_fp16_path,
)


def _run_logged(command: list[str], log_path: Path, *, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            + json.dumps(command, ensure_ascii=False)
            + "\n"
        )
        stream.flush()
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log_path}")


def require_llama_cpp(
    llama_cpp_dir: Path, *, minimum_free_disk_gib: float | None = None
) -> dict:
    root = llama_cpp_dir.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Not a llama.cpp Git checkout: {root}")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LLAMA_CPP_COMMIT:
        raise RuntimeError(
            f"llama.cpp is {commit}, expected frozen {LLAMA_CPP_COMMIT}. "
            "Do not continue with a moving backend."
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Pinned llama.cpp checkout is modified: {root}")
    binaries = binary_paths(root)
    missing = [str(path) for path in binaries.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pinned llama.cpp binaries: {missing}")
    binary_versions = {}
    for name, binary in binaries.items():
        version_result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        version_text = version_result.stdout + version_result.stderr
        reported_commits = re.findall(
            r"\(`?([0-9a-f]{7,40})`?\)", version_text, flags=re.IGNORECASE
        )
        if (
            version_result.returncode
            or not reported_commits
            or not any(LLAMA_CPP_COMMIT.startswith(item) for item in reported_commits)
        ):
            raise RuntimeError(
                f"{name} binary is stale or does not identify pinned commit "
                f"{LLAMA_CPP_COMMIT}: {version_text.strip()}"
            )
        binary_versions[name] = {
            "sha256": sha256_file(binary),
            "version_output": version_text.strip(),
        }
    converter = root / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(converter)
    conversion_python = Path(
        os.environ.get(
            "DEPLOYMENT_LLAMA_CPP_PYTHON",
            Path(f"{root}-convert-venv") / "bin" / "python",
        )
    )
    if not conversion_python.is_file():
        raise FileNotFoundError(
            f"Missing isolated llama.cpp conversion Python: {conversion_python}"
        )
    conversion_check = subprocess.run(
        [str(conversion_python), "-c", "import gguf, numpy"],
        capture_output=True,
        text=True,
    )
    if conversion_check.returncode:
        raise RuntimeError(
            f"llama.cpp conversion environment is incomplete: {conversion_check.stderr}"
        )
    freeze_result = subprocess.run(
        [str(conversion_python), "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if freeze_result.returncode:
        raise RuntimeError(
            f"Cannot freeze llama.cpp conversion environment: {freeze_result.stderr}"
        )
    device_result = subprocess.run(
        [str(binaries["bench"]), "--list-devices"],
        capture_output=True,
        text=True,
    )
    if device_result.returncode or "CUDA" not in (
        device_result.stdout + device_result.stderr
    ).upper():
        raise RuntimeError("Pinned llama.cpp build does not expose a CUDA device")
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    gpu_rows = [line.strip() for line in gpu_result.stdout.splitlines() if line.strip()]
    if gpu_result.returncode or len(gpu_rows) < 2:
        raise RuntimeError("deployment-gguf-v1 requires the declared two-GPU server")
    memory_totals = []
    for row in gpu_rows:
        try:
            memory_totals.append(float(row.rsplit(",", 1)[1].strip()))
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"Cannot parse nvidia-smi row: {row}") from exc
    if min(memory_totals[:2]) < 23000:
        raise RuntimeError(f"Expected two 24-GiB-class GPUs, found: {gpu_rows[:2]}")
    disk = shutil.disk_usage(OUT.parent)
    if (
        minimum_free_disk_gib is not None
        and disk.free < minimum_free_disk_gib * 1024**3
    ):
        raise RuntimeError(
            f"Only {disk.free / 1024**3:.1f} GiB is free; this stage requires at "
            f"least {minimum_free_disk_gib:.1f} GiB before artifact construction"
        )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "llama_cpp_dir": str(root),
        "llama_cpp_commit": commit,
        "dirty": False,
        "binaries": {key: str(path) for key, path in binaries.items()},
        "binary_versions": binary_versions,
        "conversion_python": str(conversion_python),
        "conversion_packages": sorted(
            line.strip() for line in freeze_result.stdout.splitlines() if line.strip()
        ),
        "device_listing": device_result.stdout + device_result.stderr,
        "nvidia_smi_gpus": gpu_rows,
        "workspace_disk_free_gib": disk.free / 1024**3,
        "minimum_free_disk_gib_enforced": minimum_free_disk_gib,
        "gate_passed": True,
    }
    record_name = (
        "llama_cpp_preflight.json"
        if minimum_free_disk_gib is not None
        else "llama_cpp_verification_latest.json"
    )
    atomic_write_json(MANIFEST_DIR / record_name, record)
    return record


def _token_hash(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(struct.pack("<i", int(token)))
    return digest.hexdigest()


def prepare_calibration_corpus(model_key: str) -> dict:
    """Decode the exact frozen token streams into one shared imatrix corpus."""
    from transformers import AutoTokenizer

    from experiments.revision_full.run import (
        dataset_provenance,
        frozen_wikitext_calibration,
        model_provenance,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SPECS[model_key]["path"], local_files_only=True
    )
    sequences = []
    texts = []
    if not tokenizer.eos_token:
        raise RuntimeError(f"{model_key} tokenizer has no EOS token for corpus boundaries")
    for seed in CALIB_SEEDS:
        calibration = frozen_wikitext_calibration(tokenizer, seed)
        if tuple(calibration.shape) != (CALIB_SAMPLES, CALIB_LENGTH):
            raise RuntimeError(
                f"Frozen calibration shape changed for seed {seed}: "
                f"{tuple(calibration.shape)}"
            )
        for sample_index, tensor in enumerate(calibration):
            token_ids = [int(token) for token in tensor.tolist()]
            text = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            roundtrip_ids = [
                int(token)
                for token in tokenizer.encode(text, add_special_tokens=False)
            ]
            if roundtrip_ids != token_ids:
                raise RuntimeError(
                    f"Frozen calibration sequence is not decode/encode exact for "
                    f"{model_key}, seed {seed}, sample {sample_index}; refusing an "
                    "unregistered imatrix input change"
                )
            sequences.append(
                {
                    "calibration_seed": int(seed),
                    "sample_index": sample_index,
                    "tokens": len(token_ids),
                    "token_ids_sha256": _token_hash(token_ids),
                    "decoded_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            texts.append(text)

    corpus = calibration_corpus_path(model_key)
    corpus.parent.mkdir(parents=True, exist_ok=True)
    record_path = MANIFEST_DIR / "calibration" / f"{model_key}__corpus.json"
    temporary = corpus.with_name(f".{corpus.name}.{os.getpid()}.tmp")
    temporary.write_text(tokenizer.eos_token.join(texts), encoding="utf-8")
    candidate_sha256 = sha256_file(temporary)
    dataset_snapshot = dataset_provenance()
    model_snapshot = model_provenance(model_key)
    if corpus.exists():
        if not record_path.is_file():
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Calibration corpus exists without registration: {corpus}")
        registered = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            registered.get("protocol_version") != PROTOCOL_VERSION
            or registered.get("model_key") != model_key
            or registered.get("corpus_sha256") != sha256_file(corpus)
            or registered.get("corpus_sha256") != candidate_sha256
            or registered.get("dataset_snapshot") != dataset_snapshot
            or registered.get("model_snapshot") != model_snapshot
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Frozen calibration corpus/provenance changed for {model_key}; "
                "refusing to overwrite the registered imatrix input"
            )
        temporary.unlink(missing_ok=True)
        return registered
    if record_path.is_file():
        registered = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            registered.get("protocol_version") != PROTOCOL_VERSION
            or registered.get("model_key") != model_key
            or registered.get("corpus_sha256") != candidate_sha256
            or registered.get("dataset_snapshot") != dataset_snapshot
            or registered.get("model_snapshot") != model_snapshot
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Reconstructed calibration corpus disagrees with registration: {record_path}"
            )
    os.replace(temporary, corpus)
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "source": "frozen WikiText train packed streams from revision-full-v4",
        "test_data_used": False,
        "calibration_seeds": list(CALIB_SEEDS),
        "samples_per_seed": CALIB_SAMPLES,
        "tokens_per_sample": CALIB_LENGTH,
        "total_sequences": len(sequences),
        "source_token_count": len(sequences) * CALIB_LENGTH,
        "sequence_boundary_token": tokenizer.eos_token,
        "sequence_boundary_token_id": tokenizer.eos_token_id,
        "sequence_boundary_count": len(sequences) - 1,
        "all_sequences_decode_encode_roundtrip_exact": True,
        "sequences": sequences,
        "corpus": str(corpus),
        "corpus_bytes": corpus.stat().st_size,
        "corpus_sha256": sha256_file(corpus),
        "dataset_snapshot": dataset_snapshot,
        "model_snapshot": model_snapshot,
    }
    atomic_write_json(record_path, record)
    return record


def convert_fp16(model_key: str, llama_cpp_dir: Path, *, force: bool = False) -> dict:
    require_llama_cpp(llama_cpp_dir)
    output = source_fp16_path(model_key)
    registration_path = build_registration_path(model_key, "fp16")
    if output.exists() and not force:
        if not registration_path.is_file():
            raise RuntimeError(
                f"GGUF-FP16 exists without immutable build registration: {output}"
            )
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        if (
            registration.get("artifact_sha256") != sha256_file(output)
            or registration.get("llama_cpp_commit") != LLAMA_CPP_COMMIT
            or registration.get("converter_sha256")
            != sha256_file(llama_cpp_dir.resolve() / "convert_hf_to_gguf.py")
        ):
            raise RuntimeError(f"GGUF-FP16 build registration mismatch: {output}")
        return audit_artifact(model_key, "fp16")
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    conversion_python = Path(
        os.environ.get(
            "DEPLOYMENT_LLAMA_CPP_PYTHON",
            Path(f"{llama_cpp_dir.resolve()}-convert-venv") / "bin" / "python",
        )
    )
    command = [
        str(conversion_python),
        str(llama_cpp_dir.resolve() / "convert_hf_to_gguf.py"),
        str(Path(MODEL_SPECS[model_key]["path"]).resolve()),
        "--outfile",
        str(output),
        "--outtype",
        "f16",
    ]
    _run_logged(
        command,
        MANIFEST_DIR / "build_logs" / f"{model_key}__fp16.log",
    )
    corpus_manifest = MANIFEST_DIR / "calibration" / f"{model_key}__corpus.json"
    if not corpus_manifest.is_file():
        raise RuntimeError(
            f"Missing source-model provenance registration: {corpus_manifest}"
        )
    registration = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "method": "fp16",
        "source_model": json.loads(corpus_manifest.read_text(encoding="utf-8"))[
            "model_snapshot"
        ],
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "command": command,
        "converter_sha256": sha256_file(
            llama_cpp_dir.resolve() / "convert_hf_to_gguf.py"
        ),
        "artifact_sha256": sha256_file(output),
        "requantization": False,
    }
    atomic_write_json(registration_path, registration)
    return audit_artifact(model_key, "fp16")


def build_imatrix(
    model_key: str, llama_cpp_dir: Path, *, gpu: int, force: bool = False
) -> dict:
    require_llama_cpp(llama_cpp_dir)
    corpus = calibration_corpus_path(model_key)
    corpus_manifest = MANIFEST_DIR / "calibration" / f"{model_key}__corpus.json"
    if not corpus.is_file() or not corpus_manifest.is_file():
        raise RuntimeError(f"Run prepare-calibration for {model_key} first")
    output = imatrix_path(model_key)
    record_path = MANIFEST_DIR / "calibration" / f"{model_key}__imatrix.json"
    if output.exists() and not force:
        if not record_path.is_file():
            raise RuntimeError(f"Imatrix exists without registration: {output}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        corpus_record = json.loads(corpus_manifest.read_text(encoding="utf-8"))
        if (
            record.get("imatrix_sha256") != sha256_file(output)
            or record.get("protocol_version") != PROTOCOL_VERSION
            or record.get("model_key") != model_key
            or record.get("llama_cpp_commit") != LLAMA_CPP_COMMIT
            or record.get("corpus_sha256") != sha256_file(corpus)
            or corpus_record.get("corpus_sha256") != sha256_file(corpus)
            or record.get("source_fp16_sha256")
            != sha256_file(source_fp16_path(model_key))
            or record.get("imatrix_binary_sha256")
            != sha256_file(binary_paths(llama_cpp_dir)["imatrix"])
        ):
            raise RuntimeError(f"Imatrix hash changed: {output}")
        return record
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    binaries = binary_paths(llama_cpp_dir)
    command = [
        str(binaries["imatrix"]),
        "--model",
        str(source_fp16_path(model_key)),
        "--file",
        str(corpus),
        "--output",
        str(output),
        "--no-ppl",
        "--ctx-size",
        str(CALIB_LENGTH),
        "--threads",
        str(CPU_THREADS),
        "--n-gpu-layers",
        "all",
        "--output-frequency",
        "0",
        "--save-frequency",
        "0",
        "--parse-special",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _run_logged(
        command,
        MANIFEST_DIR / "build_logs" / f"{model_key}__imatrix.log",
        env=env,
    )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "test_data_used": False,
        "shared_by_methods": ["q4", "q5", "sg"],
        "corpus_sha256": sha256_file(corpus),
        "source_fp16_sha256": sha256_file(source_fp16_path(model_key)),
        "imatrix": str(output),
        "imatrix_bytes": output.stat().st_size,
        "imatrix_sha256": sha256_file(output),
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "imatrix_binary_sha256": sha256_file(binaries["imatrix"]),
        "command": command,
    }
    atomic_write_json(record_path, record)
    return record


def quantize_artifact(
    model_key: str,
    method: str,
    llama_cpp_dir: Path,
    *,
    force: bool = False,
) -> dict:
    if method not in {"q4", "q5", "sg"}:
        raise ValueError("quantize supports q4, q5, or sg")
    require_llama_cpp(llama_cpp_dir)
    selection = load_frozen_selection(model_key)
    output = artifact_path(model_key, method)
    registration_path = build_registration_path(model_key, method)
    if output.exists() and not force:
        if not registration_path.is_file():
            raise RuntimeError(
                f"Quantized artifact exists without immutable build registration: {output}"
            )
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        if (
            registration.get("artifact_sha256") != sha256_file(output)
            or registration.get("source_fp16_sha256")
            != sha256_file(source_fp16_path(model_key))
            or registration.get("imatrix_sha256") != sha256_file(imatrix_path(model_key))
            or registration.get("selection_sha256")
            != sha256_file(selection_path(model_key))
            or registration.get("llama_cpp_commit") != LLAMA_CPP_COMMIT
            or registration.get("quantize_binary_sha256")
            != sha256_file(binary_paths(llama_cpp_dir)["quantize"])
        ):
            raise RuntimeError(f"Quantized artifact build registration mismatch: {output}")
        return audit_artifact(model_key, method)
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    base = "Q5_K_M" if method == "q5" else "Q4_K_M"
    binaries = binary_paths(llama_cpp_dir)
    command = [
        str(binaries["quantize"]),
        "--pure",
        "--imatrix",
        str(imatrix_path(model_key)),
        "--output-tensor-type",
        "f16",
        "--token-embedding-type",
        "f16",
    ]
    if method == "sg":
        for module_name in sorted(selection["w8_module_names"]):
            command.extend(["--tensor-type", tensor_override(module_name, "q8_0")])
    command.extend([str(source_fp16_path(model_key)), str(output), base])
    _run_logged(
        command,
        MANIFEST_DIR / "build_logs" / f"{model_key}__{method}.log",
    )
    registration = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "method": method,
        "source_fp16_sha256": sha256_file(source_fp16_path(model_key)),
        "imatrix_sha256": sha256_file(imatrix_path(model_key)),
        "selection_sha256": sha256_file(selection_path(model_key)),
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "quantize_binary_sha256": sha256_file(binaries["quantize"]),
        "command": command,
        "requantization": False,
        "artifact_sha256": sha256_file(output),
    }
    atomic_write_json(registration_path, registration)
    return audit_artifact(model_key, method)
