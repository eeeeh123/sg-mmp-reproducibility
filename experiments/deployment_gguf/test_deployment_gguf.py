from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.deployment_gguf import artifacts, benchmark, gates, quality
from experiments.deployment_gguf import diagnostics
from experiments.deployment_gguf.analyze import _exact_mcnemar, paired_ratio
from experiments.deployment_gguf.gguf_manifest import (
    expected_type,
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
    IMATRIX_OUTPUT_FREQUENCY,
    IMATRIX_SAVE_FREQUENCY,
    LLAMA_CPP_COMMIT,
    REQUIRED_CMAKE_TOOLCHAIN,
    conversion_gate_policy_sha256,
    packed_gate_policy_sha256,
    deployment_check_policy_sha256,
    protocol_lock,
    quantization_policy_sha256,
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
        self.assertEqual(len(conversion_gate_policy_sha256()), 64)

    def test_packed_formats_are_exact_32_element_block_controls(self):
        lock = protocol_lock()
        self.assertEqual(lock["methods"]["q4"]["cli_type"], "Q4_0")
        self.assertEqual(lock["methods"]["q5"]["cli_type"], "Q5_0")
        self.assertEqual(expected_type("q4", False), "Q4_0")
        self.assertEqual(expected_type("q5", False), "Q5_0")
        self.assertEqual(expected_type("sg", False), "Q4_0")
        self.assertEqual(expected_type("sg", True), "Q8_0")
        self.assertEqual(len(quantization_policy_sha256("q4")), 64)
        self.assertNotEqual(
            quantization_policy_sha256("q4"), quantization_policy_sha256("sg")
        )

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

    @staticmethod
    def _top8(first: tuple[int, float], second: tuple[int, float], tail_shift=0.0):
        rows = [first, second]
        rows.extend((token, -3.0 - token / 1000 + tail_shift) for token in range(6))
        return dict(rows)

    def test_reference_logits_accept_explained_near_tie(self):
        hf = self._top8((10, -1.00), (11, -1.04))
        gguf = self._top8((11, -1.01), (10, -1.03))
        row = gates._reference_logprob_agreement(hf, gguf)
        self.assertFalse(row["same_top1"])
        self.assertTrue(row["explained_near_tie"])
        self.assertTrue(row["passed"])

    def test_packed_same_history_is_independent_of_free_running_cascade(self):
        cpu = self._top8((10, -1.00), (11, -1.04))
        cuda = self._top8((11, -1.01), (10, -1.03))
        identities = [{"train_index": i} for i in gates.TRAIN_GATE_INDICES]
        left = [[cpu] * 16 for _ in identities]
        right = [[cuda] * 16 for _ in identities]
        check = gates._packed_teacher_agreement(left, right, identities)
        self.assertTrue(check["passed"])
        self.assertEqual(check["positions"], 128)
        self.assertIn("cpu_top1_id", check["rows"][0])
        self.assertFalse(gates._continuation_agreement([[10]*16]*8, [[11]*16]*8)["passed"])
        right[0] = [self._top8((11, -1.0), (10, -1.8))] + [cuda]*15
        self.assertFalse(gates._packed_teacher_agreement(left, right, identities)["passed"])
        with self.assertRaises(RuntimeError):
            gates._packed_teacher_agreement(left[:-1], right, identities)

    def test_packed_teacher_queries_use_reference_token_ids(self):
        server = mock.Mock()
        with mock.patch.object(gates, "_server_next_token_logprobs", return_value={}) as query:
            gates._teacher_forced_rows(server, [[100, 101]], [list(range(16))])
        self.assertEqual(query.call_count, 16)
        for step, call in enumerate(query.call_args_list):
            self.assertEqual(call.args[1], [100, 101] + list(range(step)))

    def test_reference_logits_reject_nonfinite_values(self):
        row = self._top8((10, -1.0), (11, -1.3))
        for value in (float("nan"), float("inf"), float("-inf")):
            bad = {**row, 0: value}
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                gates._reference_logprob_agreement(row, bad)

    def test_packed_gate_wires_teacher_result_and_archives_failure(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_dir = root / "gates" / "qwen05"
            gate_dir.mkdir(parents=True)
            old_path = gate_dir / "packed__fp16.json"
            old_path.write_text(json.dumps({"gate_passed": False}))
            (root / "gates" / "official_backend_tests.json").write_text(json.dumps({
                "gate_passed": True, "llama_cpp_commit": LLAMA_CPP_COMMIT,
                "binary_sha256": {"server": "hash"},
            }))
            (gate_dir / "conversion.json").write_text(json.dumps({
                "gate_passed": True, "model_key": "qwen05",
                "conversion_gate_policy_sha256": conversion_gate_policy_sha256(),
                "server_binary_sha256": "hash",
                "fp16_artifact_sha256": "hash",
            }))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"gate_passed": True, "model_key": "qwen05",
                                            "method": "fp16", "artifact_sha256": "hash"}))
            cpu, cuda = mock.MagicMock(), mock.MagicMock()
            for server in (cpu, cuda):
                server.__enter__.return_value = server
                server.tokenize.return_value = [100, 101]
                server.resource_record.return_value = {}
            left = self._top8((10, -1.00), (11, -1.04))
            right = self._top8((11, -1.01), (10, -1.03))
            identities = [{"train_index": i} for i in gates.TRAIN_GATE_INDICES]
            with ExitStack() as stack:
                patches = {
                    "STATUS_DIR": root,
                    "LlamaServer": mock.Mock(side_effect=[cpu, cuda]),
                    "sha256_file": mock.Mock(return_value="hash"),
                    "artifact_manifest_path": mock.Mock(return_value=manifest),
                    "artifact_path": mock.Mock(return_value=root / "model.gguf"),
                    "binary_paths": mock.Mock(return_value={"server": root / "server"}),
                    "_gate_prompts": mock.Mock(return_value=(["prompt"]*8, identities, None)),
                    "_server_tokens": mock.Mock(side_effect=[[[10]*16]*8, [[11]*16]*8]),
                    "_server_next_token_logprobs": mock.Mock(
                        side_effect=lambda server, *args, **kwargs: left if server is cpu else right
                    ),
                }
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(gates, name, value))
                record = gates.packed_backend_gate("qwen05", "fp16", root, gpu=0)
                old_bytes = old_path.read_bytes()
                with mock.patch.object(gates, "LlamaServer", return_value=cuda), mock.patch.object(
                    gates, "_server_tokens", return_value=[[11]*16]*8
                ):
                    operational = gates.deployment_check("qwen05", "fp16", root, gpu=0)
                    self.assertTrue(operational["gate_passed"])
                    self.assertEqual(old_path.read_bytes(), old_bytes)
                    with mock.patch.object(gates, "_server_next_token_logprobs", return_value={
                        **right, 0: float("nan")
                    }):
                        with self.assertRaisesRegex(RuntimeError, "non-finite"):
                            gates.deployment_check("qwen05", "fp16", root, gpu=0)
                    failed = json.loads((gate_dir / "deployment__fp16.json").read_text())
                    self.assertFalse(failed["gate_passed"])
                    self.assertEqual(old_path.read_bytes(), old_bytes)
            self.assertTrue(record["gate_passed"])
            self.assertFalse(record["cpu_vs_cuda_continuation"]["passed"])
            self.assertEqual(record["packed_gate_policy_sha256"], packed_gate_policy_sha256())
            self.assertEqual(record["train_prompts"][0]["first_divergence_step"], 0)
            archives = list((gate_dir / "packed__fp16_attempts").glob("*.json"))
            self.assertEqual(len(archives), 1)
            self.assertFalse(json.loads(archives[0].read_text())["gate_passed"])

    def test_diagnostic_preserves_gate_and_runs_three_fa_controls(self):
        import tempfile
        from experiments.deployment_gguf.llama_server import LlamaServer

        for cuda, override, expected in ((True, None, "on"), (False, None, "off"),
                                         (True, "off", "off")):
            server = LlamaServer(Path("server"), Path("model"), Path("log"),
                                 cuda=cuda, flash_attn=override)
            self.assertEqual(server.command[server.command.index("--flash-attn") + 1], expected)
        identities = [{"train_index": i, "effective_input_token_ids": [100],
                       "cpu_token_ids": [10]*16} for i in gates.TRAIN_GATE_INDICES]
        top = self._top8((10, -1.0), (11, -1.3))
        rows = [[top]*16 for _ in identities]
        teacher = gates._packed_teacher_agreement(rows, rows, identities)
        record = {"model_key": "qwen05", "method": "q4", "tokenizer_passed": True,
                  "gate_passed": False, "artifact_sha256": "hash",
                  "server_binary_sha256": "hash",
                  "packed_gate_policy_sha256": packed_gate_policy_sha256(),
                  "train_prompts": identities, "teacher_forced_cpu_vs_cuda": teacher}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "gates" / "qwen05" / "packed__q4.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(record))
            original = source.read_bytes()
            server = mock.MagicMock()
            server.__enter__.return_value = server
            server.resource_record.return_value = {"command": ["fake"]}
            with ExitStack() as stack:
                factory = stack.enter_context(mock.patch.object(diagnostics, "LlamaServer", return_value=server))
                stack.enter_context(mock.patch.object(diagnostics, "STATUS_DIR", root))
                stack.enter_context(mock.patch.object(diagnostics, "sha256_file", return_value="hash"))
                query = stack.enter_context(mock.patch.object(diagnostics, "_teacher_forced_rows", return_value=rows))
                result = diagnostics.packed_diagnostic("qwen05", "q4", root, gpu=0)
            self.assertTrue(result["complete"])
            self.assertNotIn("gate_passed", result)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual([c.kwargs["flash_attn"] for c in factory.call_args_list], ["on", "on", "off"])
            for call in query.call_args_list:
                self.assertEqual(call.args[1], [[100]]*8)
                self.assertEqual(call.args[2], [[10]*16]*8)
            self.assertEqual(len(result["comparisons"]), 5)
            self.assertEqual(result["comparisons"]["cuda_on_1__vs__cuda_on_2"]["exact_top8_rows"], 128)
            self.assertTrue((Path(result["output_directory"]) / "summary.json").is_file())
        record["teacher_forced_cpu_vs_cuda"]["rows"] = teacher["rows"][:-1]
        with self.assertRaisesRegex(RuntimeError, "incomplete or reordered"):
            diagnostics._saved_reference(record)

    def test_reference_logits_ignore_noncritical_tail_boundary(self):
        hf = self._top8((10, -1.00), (11, -1.30))
        gguf = self._top8((10, -0.98), (11, -1.31), tail_shift=-0.0501)
        row = gates._reference_logprob_agreement(hf, gguf)
        self.assertGreater(row["max_common_logprob_abs_error"], 0.05)
        self.assertLessEqual(row["max_critical_logprob_abs_error"], 0.05)
        self.assertTrue(row["passed"])

    def test_reference_logits_use_decision_relative_gap(self):
        hf = self._top8((10, -1.00), (11, -1.50))
        gguf = self._top8((10, -1.08), (11, -1.66), tail_shift=-0.08)
        row = gates._reference_logprob_agreement(hf, gguf)
        self.assertGreater(row["max_critical_logprob_abs_error"], 0.05)
        self.assertAlmostEqual(row["max_critical_logprob_gap_abs_error"], 0.08)
        self.assertTrue(row["same_top1"])
        self.assertTrue(row["passed"])

    def test_reference_logits_reject_distorted_decision_margin(self):
        hf = self._top8((10, -1.00), (11, -1.50))
        gguf = self._top8((10, -1.08), (11, -1.69), tail_shift=-0.08)
        row = gates._reference_logprob_agreement(hf, gguf)
        self.assertGreater(row["max_critical_logprob_gap_abs_error"], 0.10)
        self.assertTrue(row["same_top1"])
        self.assertFalse(row["passed"])

    def test_reference_logits_reject_decisive_top1_flip(self):
        hf = self._top8((10, -1.00), (11, -1.30))
        gguf = self._top8((11, -1.00), (10, -1.30))
        row = gates._reference_logprob_agreement(hf, gguf)
        self.assertFalse(row["explained_near_tie"])
        self.assertFalse(row["passed"])

    def test_reference_logits_require_both_runner_up_candidates(self):
        hf = self._top8((10, -1.00), (11, -1.30))
        gguf = self._top8((10, -0.99), (12, -1.29))
        row = gates._reference_logprob_agreement(hf, gguf)
        self.assertEqual(row["top_k_overlap"], 7)
        self.assertIsNone(row["max_critical_logprob_abs_error"])
        self.assertFalse(row["passed"])

    def test_reference_logprob_query_does_not_suppress_eos(self):
        class DummyServer:
            def __init__(self):
                self.prompt = None
                self.kwargs = None

            def complete(self, prompt, **kwargs):
                self.prompt = prompt
                self.kwargs = kwargs
                return {
                    "probs": [
                        {
                            "top_logprobs": [
                                {"id": token, "logprob": -float(token)}
                                for token in range(8)
                            ]
                        }
                    ]
                }

        server = DummyServer()
        row = gates._server_next_token_logprobs(server, [1, 2, 3], seed=7)
        self.assertEqual(server.prompt, [1, 2, 3])
        self.assertFalse(server.kwargs["ignore_eos"])
        self.assertEqual(server.kwargs["n_predict"], 1)
        self.assertEqual(len(row), 8)

    def test_imatrix_uses_nonzero_output_frequency_and_atomic_publication(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-imatrix"
        manifest_dir = root / "manifests"
        corpus = root / "calibration.txt"
        output = root / "calibration" / "qwen05" / "imatrix.gguf"
        fp16 = root / "qwen05-fp16.gguf"
        binary = root / "llama-imatrix"
        corpus_manifest = manifest_dir / "calibration" / "qwen05__corpus.json"
        try:
            corpus_manifest.parent.mkdir(parents=True)
            corpus_manifest.write_text("{}", encoding="utf-8")
            corpus.write_text("training text", encoding="utf-8")
            fp16.write_bytes(b"fp16")
            binary.write_bytes(b"imatrix-binary")

            def fake_run(command, _log_path, *, env):
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
                output_frequency = command[command.index("--output-frequency") + 1]
                save_frequency = command[command.index("--save-frequency") + 1]
                self.assertEqual(output_frequency, str(IMATRIX_OUTPUT_FREQUENCY))
                self.assertNotEqual(output_frequency, "0")
                self.assertEqual(save_frequency, str(IMATRIX_SAVE_FREQUENCY))
                staging = Path(command[command.index("--output") + 1])
                self.assertEqual(staging.name, "imatrix.incomplete.gguf")
                self.assertNotEqual(staging, output)
                staging.parent.mkdir(parents=True, exist_ok=True)
                staging.write_bytes(b"complete-imatrix")

            patches = (
                mock.patch.object(artifacts, "MANIFEST_DIR", manifest_dir),
                mock.patch.object(artifacts, "require_llama_cpp"),
                mock.patch.object(
                    artifacts, "calibration_corpus_path", return_value=corpus
                ),
                mock.patch.object(artifacts, "imatrix_path", return_value=output),
                mock.patch.object(artifacts, "source_fp16_path", return_value=fp16),
                mock.patch.object(
                    artifacts, "binary_paths", return_value={"imatrix": binary}
                ),
                mock.patch.object(artifacts, "_run_logged", side_effect=fake_run),
            )
            with ExitStack() as stack:
                for patch in patches[:-1]:
                    stack.enter_context(patch)
                run_mock = stack.enter_context(patches[-1])
                record = artifacts.build_imatrix("qwen05", root, gpu=0)
                self.assertEqual(output.read_bytes(), b"complete-imatrix")
                staging = output.with_name("imatrix.incomplete.gguf")
                self.assertFalse(staging.exists())
                self.assertEqual(
                    record["imatrix_sha256"], artifacts.sha256_file(output)
                )

                output.unlink()
                (manifest_dir / "calibration" / "qwen05__imatrix.json").unlink()

                def fail_after_partial(command, _log_path, *, env):
                    del env
                    partial = Path(command[command.index("--output") + 1])
                    partial.write_bytes(b"partial")
                    raise RuntimeError("simulated failure")

                run_mock.side_effect = fail_after_partial
                with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                    artifacts.build_imatrix("qwen05", root, gpu=0)
                self.assertFalse(staging.exists())
                self.assertFalse(output.exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

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
            value.index("deployment-check --model smollm"),
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

    def test_backend_gate_runs_only_strict_quantize_self_test(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-backend-quantize"
        status = root / "status"
        build_bin = root / "build" / "bin"
        try:
            build_bin.mkdir(parents=True)
            quantize = build_bin / "test-quantize-fns"
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
            self.assertEqual(record["strict_checks_required"], 1)
            self.assertEqual(record["strict_checks_passed"], 1)
            self.assertFalse(record["generic_backend_ops"]["executed"])
            self.assertEqual(calls, [([str(quantize)], None)])
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

    def test_conversion_gate_archives_legacy_latest_before_replacement(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-conversion-archive"
        latest = root / "gates" / "qwen05" / "conversion.json"
        previous = {"gate_passed": False, "legacy": True}
        try:
            latest.parent.mkdir(parents=True)
            latest.write_text(json.dumps(previous), encoding="utf-8")
            gates._archive_previous_gate(latest, "conversion_attempts")
            archives = list(
                (latest.parent / "conversion_attempts").glob("*.json")
            )
            self.assertEqual(len(archives), 1)
            self.assertEqual(
                json.loads(archives[0].read_text(encoding="utf-8")), previous
            )
            gates._archive_previous_gate(latest, "conversion_attempts")
            self.assertEqual(
                len(list((latest.parent / "conversion_attempts").glob("*.json"))),
                1,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_backend_gate_rejects_failed_quantize_self_test(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-backend-failure"
        status = root / "status"
        build_bin = root / "build" / "bin"
        try:
            build_bin.mkdir(parents=True)
            (build_bin / "test-quantize-fns").write_bytes(b"quantize")
            runtime_binary = root / "runtime-binary"
            runtime_binary.write_bytes(b"runtime")
            def fake_case(command, log_path, *, gpu):
                return {
                    "command": command,
                    "physical_gpu": gpu,
                    "returncode": 1,
                    "output_bytes": 5,
                    "passed": False,
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
            self.assertEqual(record["strict_checks_required"], 1)
            self.assertEqual(record["strict_checks_passed"], 0)
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
                json.dumps(
                    {
                        "gate_passed": True,
                        "model_key": "qwen05",
                        "conversion_gate_policy_sha256": (
                            conversion_gate_policy_sha256()
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (gate_dir / "deployment__q4.json").write_text(
                json.dumps(
                    {
                        "gate_passed": True,
                        "model_key": "qwen05",
                        "method": "q4",
                        "artifact_sha256": "a",
                        "deployment_check_policy_sha256": deployment_check_policy_sha256(),
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
                with self.assertRaisesRegex(RuntimeError, "hashes disagree"):
                    quality.require_quality_gates("qwen05", "q4")
                for check in (quality.require_quality_gates, benchmark._require_benchmark_gate):
                    packed_path = gate_dir / "deployment__q4.json"
                    packed_record = json.loads(packed_path.read_text())
                    packed_record["deployment_check_policy_sha256"] = "obsolete"
                    packed_path.write_text(json.dumps(packed_record))
                    with mock.patch.object(benchmark, "STATUS_DIR", root), mock.patch.object(
                        benchmark, "artifact_manifest_path", return_value=manifest
                    ):
                        with self.assertRaisesRegex(RuntimeError, "obsolete gate policy"):
                            check("qwen05", "q4")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_benchmark_gate_cannot_bypass_failed_conversion(self):
        root = Path(__file__).parent / f".test-{os.getpid()}-benchmark-status"
        manifest = root / "manifest.json"
        gate_dir = root / "gates" / "qwen05"
        common = {
            "gate_passed": True,
            "model_key": "qwen05",
            "method": "q4",
            "artifact_sha256": "same",
        }
        try:
            gate_dir.mkdir(parents=True)
            (gate_dir / "conversion.json").write_text(
                json.dumps({"gate_passed": False, "model_key": "qwen05"}),
                encoding="utf-8",
            )
            (gate_dir / "deployment__q4.json").write_text(
                json.dumps(common), encoding="utf-8"
            )
            manifest.write_text(json.dumps(common), encoding="utf-8")
            with mock.patch.object(benchmark, "STATUS_DIR", root), mock.patch.object(
                benchmark, "artifact_manifest_path", return_value=manifest
            ):
                with self.assertRaisesRegex(RuntimeError, "Benchmark locked"):
                    benchmark._require_benchmark_gate("qwen05", "q4")
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
