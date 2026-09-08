from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.deployment_gguf import artifacts, gates, quality
from experiments.deployment_gguf.analyze import _exact_mcnemar, paired_ratio
from experiments.deployment_gguf.gguf_manifest import (
    hf_to_gguf_tensor,
    read_gguf,
    tensor_override,
)
from experiments.deployment_gguf.make_server_plan import (
    balanced_methods,
    build_formal_extension_plan,
    build_plan,
)
from experiments.deployment_gguf.protocol import (
    BACKEND_ADD_STABILITY_REPETITIONS,
    BACKEND_TEST_GPUS,
    LLAMA_CPP_COMMIT,
    REQUIRED_CMAKE_TOOLCHAIN,
    protocol_lock,
)
from experiments.deployment_gguf.quality import strict_prediction


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _fake_gguf(path: Path) -> None:
    header = b"GGUF" + struct.pack("<IQQ", 3, 2, 1)
    metadata = _string("general.alignment") + struct.pack("<II", 4, 32)
    first = (
        _string("blk.0.attn_q.weight")
        + struct.pack("<I", 2)
        + struct.pack("<QQ", 256, 1)
        + struct.pack("<IQ", 12, 0)
    )
    second = (
        _string("token_embd.weight")
        + struct.pack("<I", 2)
        + struct.pack("<QQ", 8, 2)
        + struct.pack("<IQ", 1, 160)
    )
    directory = header + metadata + first + second
    data_start = (len(directory) + 31) // 32 * 32
    # Q4_K payload is 144 bytes; second tensor begins at aligned offset 160.
    payload = b"\0" * (data_start - len(directory)) + b"\0" * 192
    path.write_bytes(directory + payload)


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_pinned_and_claim_limited(self):
        lock = protocol_lock()
        self.assertEqual(lock["llama_cpp"]["commit"], LLAMA_CPP_COMMIT)
        self.assertIn("not deployment validation", lock["inference_scope"])
        self.assertFalse(lock["importance_matrix"]["test_data_used"])
        self.assertFalse(lock["generation"]["quality"]["online_stop"])

    def test_hf_to_gguf_mapping_is_exact(self):
        self.assertEqual(
            hf_to_gguf_tensor("model.layers.12.self_attn.o_proj"),
            "blk.12.attn_output.weight",
        )
        self.assertEqual(
            hf_to_gguf_tensor("model.layers.3.mlp.up_proj"),
            "blk.3.ffn_up.weight",
        )
        self.assertEqual(
            tensor_override("model.layers.3.mlp.up_proj"),
            r"^blk\.3\.ffn_up\.weight$=q8_0",
        )
        with self.assertRaises(ValueError):
            hf_to_gguf_tensor("model.embed_tokens")

    def test_minimal_gguf_parser_accounts_payloads(self):
        path = Path(__file__).parent / f".test-{os.getpid()}-tiny.gguf"
        try:
            _fake_gguf(path)
            record = read_gguf(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(record["version"], 3)
        self.assertEqual(record["alignment"], 32)
        self.assertEqual(record["tensors"][0].type_name, "Q4_K")
        self.assertEqual(record["tensors"][0].logical_bytes, 144)
        self.assertEqual(record["tensors"][1].logical_bytes, 32)

    def test_gguf_parser_rejects_truncated_payload(self):
        path = Path(__file__).parent / f".test-{os.getpid()}-truncated.gguf"
        try:
            _fake_gguf(path)
            path.write_bytes(path.read_bytes()[:-64])
            with self.assertRaises(RuntimeError):
                read_gguf(path)
        finally:
            path.unlink(missing_ok=True)

    def test_server_plans_separate_engineering_from_test_quality(self):
        root = "/data/experiment/LQ/llama.cpp-deployment-gguf-v1"
        engineering = build_plan("engineering", root, 1, 0)
        value = build_plan("value", root, 1, 0)
        self.assertIn(f"LLAMA_CPP_DIR={root}", engineering)
        self.assertNotIn("deployment_gguf.run quality", engineering)
        self.assertIn("deployment_gguf.run quality", value)
        self.assertIn("--run-phase engineering", engineering)
        self.assertIn("--run-phase value", value)
        self.assertIn("--gpu 1 --block", value)
        self.assertIn("quality --model qwen15", value)
        self.assertIn("--gpu 0", value)
        self.assertIn("unset CUDA_VISIBLE_DEVICES GGML_CUDA_DISABLE_GRAPHS", value)
        self.assertIn("export CUDA_DEVICE_ORDER=PCI_BUS_ID", value)
        self.assertLess(
            value.index("packed-gate --model smollm"),
            value.index("quality --model qwen15"),
        )
        self.assertEqual(set(balanced_methods(0)), {"fp16", "q4", "q5", "sg"})

    def test_server_plan_rejects_shared_timing_and_quality_gpu(self):
        with self.assertRaises(ValueError):
            build_plan("engineering", "/tmp/llama.cpp", 0, 0)

    def test_formal_extension_contains_only_new_performance_blocks(self):
        plan = build_formal_extension_plan("/tmp/llama.cpp", 1, 0, 10, 20)
        self.assertIn("--block 10 --run-phase formal", plan)
        self.assertIn("--block 19 --run-phase formal", plan)
        self.assertNotIn("deployment_gguf.run quality", plan)
        self.assertNotIn("deployment_gguf.run quantize", plan)
        with self.assertRaises(ValueError):
            build_formal_extension_plan("/tmp/llama.cpp", 1, 0, 9, 20)

    def test_strict_extraction_never_uses_invented_next_question(self):
        text = "reasoning\n#### 64\n\nQuestion: invented\nAnswer: #### 99"
        self.assertEqual(strict_prediction(text), "64")

    def test_train_prompt_loading_never_materializes_test(self):
        calls = []

        def frozen_rows(dataset_key, filename):
            calls.append((dataset_key, filename))
            return [
                {"question": f"train question {index}", "answer": "work #### 1"}
                for index in range(16)
            ]

        class DummyTokenizer:
            chat_template = None

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = DummyTokenizer
        revision_run = types.ModuleType("experiments.revision_full.run")
        revision_run.frozen_arrow_rows = frozen_rows
        with mock.patch.dict(
            sys.modules,
            {
                "transformers": transformers,
                "experiments.revision_full.run": revision_run,
            },
        ):
            rows, prompts, _ = quality.load_prompts("qwen05", "train")
        self.assertEqual(len(rows), 16)
        self.assertEqual(len(prompts), 16)
        self.assertEqual(
            calls,
            [("openai/gsm8k/main/train", "gsm8k-train.arrow")],
        )

    def test_cmake_build_provenance_requires_frozen_toolchain(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-cmake-provenance"
        try:
            build = root / "build"
            build.mkdir(parents=True)
            cache = [f"CMAKE_HOME_DIRECTORY:INTERNAL={root.resolve()}"]
            cache.extend(
                f"{key}:FILEPATH={value}"
                for key, value in REQUIRED_CMAKE_TOOLCHAIN.items()
            )
            (build / "CMakeCache.txt").write_text("\n".join(cache), encoding="utf-8")
            compiler = SimpleNamespace(
                returncode=0,
                stdout="Cuda compilation tools, release 12.4\n",
                stderr="",
            )
            with mock.patch.object(artifacts.subprocess, "run", return_value=compiler):
                record = artifacts.cmake_build_provenance(root)
            self.assertTrue(record["gate_passed"])
            self.assertEqual(record["actual_entries"], REQUIRED_CMAKE_TOOLCHAIN)

            wrong = cache.copy()
            wrong[1] = "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda-11.6/bin/nvcc"
            (build / "CMakeCache.txt").write_text("\n".join(wrong), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                artifacts.cmake_build_provenance(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_official_backend_gate_runs_frozen_matrix(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-backend-matrix"
        status = root / "status"
        build_bin = root / "build" / "bin"
        try:
            build_bin.mkdir(parents=True)
            backend = build_bin / "test-backend-ops"
            quantize = build_bin / "test-quantize-fns"
            backend.write_bytes(b"backend")
            quantize.write_bytes(b"quantize")
            runtime_binary = root / "runtime-binary"
            runtime_binary.write_bytes(b"runtime")
            calls = []

            def fake_case(command, log_path, *, gpu):
                calls.append((command, gpu))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("PASS\n", encoding="utf-8")
                return {
                    "command": command,
                    "physical_gpu": gpu,
                    "returncode": 0,
                    "output_bytes": 5,
                    "passed": True,
                }

            with mock.patch.object(gates, "STATUS_DIR", status), mock.patch.object(
                gates, "cmake_build_provenance", return_value={"gate_passed": True}
            ), mock.patch.object(
                gates, "_run_logged_backend_case", side_effect=fake_case
            ), mock.patch.object(
                gates.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=f"{LLAMA_CPP_COMMIT}\n",
                    stderr="",
                ),
            ), mock.patch.object(
                gates,
                "binary_paths",
                return_value={
                    name: runtime_binary
                    for name in ("server", "bench", "quantize", "imatrix", "cli")
                },
            ):
                record = gates.run_official_backend_tests(root)
            self.assertTrue(record["gate_passed"])
            self.assertEqual(record["executions_passed"], record["executions_required"])
            self.assertEqual(
                record["executions_required"],
                len(BACKEND_TEST_GPUS) * (BACKEND_ADD_STABILITY_REPETITIONS + 1) + 1,
            )
            for gpu in BACKEND_TEST_GPUS:
                self.assertEqual(
                    sum(
                        command[-2:] == ["-o", "ADD"] and seen_gpu == gpu
                        for command, seen_gpu in calls
                    ),
                    BACKEND_ADD_STABILITY_REPETITIONS,
                )
                self.assertEqual(
                    sum(
                        command == [str(backend)] and seen_gpu == gpu
                        for command, seen_gpu in calls
                    ),
                    1,
                )
            self.assertEqual(sum(seen_gpu is None for _, seen_gpu in calls), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_backend_case_sanitizes_environment_and_fails_on_timeout(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-backend-timeout"
        log = root / "timeout.log"
        try:
            def time_out(command, **kwargs):
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
                self.assertEqual(kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
                self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", kwargs["env"])
                raise gates.subprocess.TimeoutExpired(
                    command, kwargs["timeout"], output="partial output\n"
                )

            with mock.patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "7",
                    "GGML_CUDA_DISABLE_GRAPHS": "1",
                },
            ), mock.patch.object(gates.subprocess, "run", side_effect=time_out):
                record = gates._run_logged_backend_case(
                    ["test-backend-ops"], log, gpu=1
                )
            self.assertTrue(record["timed_out"])
            self.assertFalse(record["passed"])
            self.assertIn("TIMEOUT", log.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_backend_gate_archives_legacy_latest_before_replacement(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-backend-archive"
        latest = root / "gates" / "official_backend_tests.json"
        previous = {"gate_passed": False, "returncode": 1, "legacy": True}
        try:
            latest.parent.mkdir(parents=True)
            latest.write_text(json.dumps(previous), encoding="utf-8")
            gates._archive_previous_backend_gate(latest)
            archives = list(
                (latest.parent / "official_backend_test_attempts").glob("*.json")
            )
            self.assertEqual(len(archives), 1)
            self.assertEqual(
                json.loads(archives[0].read_text(encoding="utf-8")), previous
            )
            gates._archive_previous_backend_gate(latest)
            self.assertEqual(
                len(
                    list(
                        (latest.parent / "official_backend_test_attempts").glob(
                            "*.json"
                        )
                    )
                ),
                1,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_official_backend_gate_rejects_any_failed_execution(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-backend-failure"
        status = root / "status"
        build_bin = root / "build" / "bin"
        try:
            build_bin.mkdir(parents=True)
            for name in ("test-backend-ops", "test-quantize-fns"):
                (build_bin / name).write_bytes(name.encode())
            runtime_binary = root / "runtime-binary"
            runtime_binary.write_bytes(b"runtime")
            invocation = 0

            def fake_case(command, log_path, *, gpu):
                nonlocal invocation
                invocation += 1
                return {
                    "command": command,
                    "physical_gpu": gpu,
                    "returncode": 1 if invocation == 3 else 0,
                    "output_bytes": 5,
                    "passed": invocation != 3,
                }

            with mock.patch.object(gates, "STATUS_DIR", status), mock.patch.object(
                gates, "cmake_build_provenance", return_value={"gate_passed": True}
            ), mock.patch.object(
                gates, "_run_logged_backend_case", side_effect=fake_case
            ), mock.patch.object(
                gates.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout=f"{LLAMA_CPP_COMMIT}\n", stderr=""
                ),
            ), mock.patch.object(
                gates,
                "binary_paths",
                return_value={
                    name: runtime_binary
                    for name in ("server", "bench", "quantize", "imatrix", "cli")
                },
            ):
                with self.assertRaises(RuntimeError):
                    gates.run_official_backend_tests(root)
            record = json.loads(
                (status / "gates" / "official_backend_tests.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(record["gate_passed"])
            self.assertEqual(
                record["executions_passed"], record["executions_required"] - 1
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_binary_provenance_does_not_require_quantize_version_flag(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-binary-provenance"
        try:
            root.mkdir(parents=True)
            cli = root / "llama-cli"
            quantize = root / "llama-quantize"
            cli.write_bytes(b"cli")
            quantize.write_bytes(b"quantize")

            def fake_run(command, **_kwargs):
                self.assertEqual(command, [str(cli), "--version"])
                return SimpleNamespace(
                    returncode=0,
                    stdout=f"version: 1 ({LLAMA_CPP_COMMIT[:7]})\n",
                    stderr="",
                )

            with mock.patch.object(artifacts.subprocess, "run", side_effect=fake_run):
                record = artifacts._binary_provenance(
                    {"cli": cli, "quantize": quantize}
                )
            self.assertTrue(record["cli"]["identifies_frozen_commit"])
            self.assertIsNone(record["quantize"]["version_output"])
            self.assertEqual(record["quantize"]["commit_attested_by"], ["cli"])
            self.assertEqual(record["quantize"]["bytes"], len(b"quantize"))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_quality_gate_rejects_artifact_hash_disagreement(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-quality-status"
        manifest = root / "manifest.json"
        gate_dir = root / "gates" / "qwen05"
        try:
            gate_dir.mkdir(parents=True)
            (gate_dir / "conversion.json").write_text(
                json.dumps({"gate_passed": True, "model_key": "qwen05"}),
                encoding="utf-8",
            )
            (gate_dir / "packed__q4.json").write_text(
                json.dumps(
                    {
                        "gate_passed": True,
                        "model_key": "qwen05",
                        "method": "q4",
                        "artifact_sha256": "a",
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "gate_passed": True,
                        "model_key": "qwen05",
                        "method": "q4",
                        "artifact_sha256": "b",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(quality, "STATUS_DIR", root), mock.patch.object(
                quality, "artifact_manifest_path", return_value=manifest
            ):
                with self.assertRaises(RuntimeError):
                    quality.require_quality_gates("qwen05", "q4")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_process_block_ratio_and_exact_mcnemar(self):
        result = paired_ratio({0: 110.0, 1: 90.0}, {0: 100.0, 1: 100.0})
        self.assertEqual(result["n_process_blocks"], 2)
        self.assertAlmostEqual(result["geometric_mean_ratio"], (0.99) ** 0.5)
        self.assertEqual(_exact_mcnemar(0, 2), 0.5)
        self.assertEqual(_exact_mcnemar(1, 1), 1.0)


if __name__ == "__main__":
    unittest.main()
