import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.revision_full import lifecycle, protocol, run as revision_run
from experiments.revision_full.make_server_plan import commands
from experiments.revision_full.make_server_shard import shard_commands


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        protocol.OUT.mkdir(parents=True, exist_ok=True)
        self.root = protocol.OUT / f".test_lifecycle_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.results = self.root / "results"
        self.states = self.root / "states"
        self.metadata = self.root / "state_metadata"
        self.receipts = self.root / "receipts"
        self.screens = self.root / "screens"
        self.data_hash = "d" * 64
        self.model_revision = "a" * 40
        self.dataset_snapshot = {
            "manifest": "revision_full/outputs/dataset_snapshot_manifest.json",
            "manifest_sha256": self.data_hash,
        }
        self.model_snapshot = {
            "repo_id": "test/qwen05",
            "resolved_revision": self.model_revision,
            "weight_file_records": [{"path": "model.safetensors", "sha256": "b" * 64}],
        }
        self.patches = [
            patch.object(lifecycle, "OUT", self.root),
            patch.object(lifecycle, "RESULTS_DIR", self.results),
            patch.object(lifecycle, "RECEIPT_DIR", self.receipts),
            patch.object(lifecycle, "SCREEN_DIR", self.screens),
            patch.object(protocol, "STATE_DIR", self.states),
            patch.object(protocol, "STATE_METADATA_DIR", self.metadata),
        ]
        for item in self.patches:
            item.start()
        (self.root / "protocol_lock.json").write_text(
            json.dumps(
                {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "dataset_snapshot": self.dataset_snapshot,
                    "model_snapshots": {"qwen05": self.model_snapshot},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def write_sample(self, variant: str, ids) -> Path:
        path = lifecycle.gsm8k_sample_path("qwen05", variant, 97)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for doc_id in ids:
                stream.write(
                    json.dumps(
                        {
                            "protocol_version": protocol.PROTOCOL_VERSION,
                            "eval_batch_size_per_gpu": protocol.DEFAULT_EVAL_BATCH_SIZE,
                            "max_new_tokens": protocol.MAX_NEW_TOKENS,
                            "dataset_manifest_sha256": self.data_hash,
                            "model_revision": self.model_revision,
                            "canonical_test_set": "openai/gsm8k/main:test:all-1319",
                            "doc_id": doc_id,
                            "correct": 1,
                        }
                    )
                    + "\n"
                )
        return path

    def write_state(self, variant: str) -> tuple[Path, Path]:
        state = protocol.state_path("qwen05", 97, variant)
        metadata = protocol.state_metadata_path("qwen05", 97, variant)
        state.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"reconstructible-state")
        metadata.write_text(
            json.dumps({"protocol_version": protocol.PROTOCOL_VERSION}),
            encoding="utf-8",
        )
        return state, metadata

    def test_incomplete_evidence_never_deletes_state(self):
        state, _ = self.write_state("gptq_w5")
        self.write_sample("gptq_w5", range(protocol.GSM8K_TEST_SIZE - 1))

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            lifecycle.cleanup_state_artifact("qwen05", 97, "gptq_w5")

        self.assertTrue(state.exists())
        self.assertFalse(lifecycle.receipt_path("qwen05", 97, "gptq_w5").exists())

    def test_complete_evidence_deletes_only_state_and_writes_receipt(self):
        state, metadata = self.write_state("gptq_w5")
        sample = self.write_sample("gptq_w5", range(protocol.GSM8K_TEST_SIZE))

        receipt = lifecycle.cleanup_state_artifact("qwen05", 97, "gptq_w5")

        self.assertFalse(state.exists())
        self.assertTrue(metadata.exists())
        self.assertEqual(receipt["bytes_deleted"], len(b"reconstructible-state"))
        self.assertEqual(receipt["evidence"][0]["sha256"], lifecycle.sha256(sample))
        self.assertTrue(lifecycle.receipt_path("qwen05", 97, "gptq_w5").exists())

    def test_stale_result_provenance_never_deletes_state(self):
        with patch.object(lifecycle, "GSM8K_TEST_SIZE", 2):
            state, _ = self.write_state("gptq_w5")
            sample = self.write_sample("gptq_w5", [0, 1])
            sample.write_text(
                sample.read_text(encoding="utf-8").replace(
                    self.model_revision, "c" * 40
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                lifecycle.cleanup_state_artifact("qwen05", 97, "gptq_w5")

            self.assertTrue(state.exists())

    def test_seed_41_w4_requires_all_downstream_consumers(self):
        path = lifecycle.gsm8k_sample_path(
            "qwen15", "gptq_w4", protocol.RANDOM_CALIB_SEED
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for doc_id in range(protocol.GSM8K_TEST_SIZE):
                stream.write(json.dumps({"doc_id": doc_id}) + "\n")

        self.assertFalse(
            lifecycle.state_consumers_complete(
                "qwen15", protocol.RANDOM_CALIB_SEED, "gptq_w4"
            )
        )

    def test_bank_cleanup_requires_every_seed_consumer(self):
        state, metadata = self.write_state("precision_bank")
        for variant in lifecycle.CORE_VARIANTS:
            self.write_sample(variant, range(protocol.GSM8K_TEST_SIZE))
            self.write_state(variant)
            lifecycle.cleanup_state_artifact("qwen05", 97, variant)

        lifecycle.cleanup_state_artifact("qwen05", 97, "precision_bank")

        self.assertFalse(state.exists())
        self.assertTrue(metadata.exists())

    def test_screen_state_cleanup_requires_exact_gptq_screen(self):
        variant = "screen_gptq_w4_split0"
        state = protocol.state_path("qwen05", 41, variant)
        metadata = protocol.state_metadata_path("qwen05", 41, variant)
        state.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"screen-state")
        metadata.write_text(
            json.dumps(
                {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "split_id": 0,
                    "calibration_seed": 41,
                    "screened_layers": [0, 1],
                    "dataset_snapshot": self.dataset_snapshot,
                    "model_snapshot": self.model_snapshot,
                }
            ),
            encoding="utf-8",
        )
        screen = self.screens / "qwen05" / "split_0.jsonl"
        screen.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "protocol_version": protocol.PROTOCOL_VERSION,
                "type": "baseline",
                "model_key": "qwen05",
                "split_id": 0,
                "quantizer": "GPTQ-W4",
                "calibration_seed": 41,
                "dataset_manifest_sha256": self.data_hash,
                "model_revision": self.model_revision,
                "eval_batch_size_per_gpu": protocol.DEFAULT_EVAL_BATCH_SIZE,
                "max_new_tokens": protocol.MAX_NEW_TOKENS,
            },
            *[
                {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "type": "layer",
                    "model_key": "qwen05",
                    "split_id": 0,
                    "layer": layer,
                    "quantizer": "GPTQ-W4",
                    "calibration_seed": 41,
                    "dataset_manifest_sha256": self.data_hash,
                    "model_revision": self.model_revision,
                    "eval_batch_size_per_gpu": protocol.DEFAULT_EVAL_BATCH_SIZE,
                    "max_new_tokens": protocol.MAX_NEW_TOKENS,
                }
                for layer in [0, 1]
            ],
        ]
        screen.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

        receipt = lifecycle.cleanup_screen_state_artifact("qwen05", 0, 41)

        self.assertFalse(state.exists())
        self.assertEqual(receipt["action"], "reconstructible_screen_state_deleted")

    def test_causal_completion_requires_all_three_interventions(self):
        with (
            patch.object(lifecycle, "CAUSAL_PATCH_N", 2),
            patch.object(lifecycle, "fixed_causal_patch_indices", return_value=[0, 1]),
        ):
            result, summary = lifecycle.causal_result_paths("qwen05", 41)
            result.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for doc_id in [0, 1]:
                rows.append(
                    {
                        "protocol_version": protocol.PROTOCOL_VERSION,
                        "dataset_manifest_sha256": self.data_hash,
                        "model_revision": self.model_revision,
                        "calibration_seed": 41,
                        "doc_id": doc_id,
                        "w4_correct": doc_id,
                        "final_answer_tokens": 1,
                        "patches": [
                            {"intervention": intervention, "layer": layer}
                            for intervention in ["block", "attention", "mlp"]
                            for layer in [0, 1]
                        ],
                    }
                )
            result.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary.write_text(
                json.dumps(
                    {
                        "protocol_version": protocol.PROTOCOL_VERSION,
                        "dataset_manifest_sha256": self.data_hash,
                        "model_revision": self.model_revision,
                        "calibration_seed": 41,
                        "n": 2,
                        "interventions": ["block", "attention", "mlp"],
                        "layers": [
                            {"intervention": intervention, "layer": layer}
                            for intervention in ["block", "attention", "mlp"]
                            for layer in [0, 1]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(lifecycle.causal_complete("qwen05", 41))

    def test_logged_samples_allow_complete_multiple_filter_views(self):
        samples = [
            {"doc_id": doc_id, "filter": filter_name}
            for filter_name in ["strict-match", "flexible-extract"]
            for doc_id in range(3)
        ]
        item = {"n_samples": len(samples), "samples": samples}

        self.assertTrue(lifecycle.logged_samples_complete(item, 3))
        self.assertFalse(lifecycle.extra_task_complete(item, 3))
        item["metrics"] = {"exact_match,flexible-extract": 1.0}
        self.assertTrue(lifecycle.extra_task_complete(item, 3))

    def test_logged_samples_reject_missing_or_duplicate_filter_rows(self):
        missing = {
            "n_samples": 5,
            "samples": [
                {"doc_id": doc_id, "filter": filter_name}
                for filter_name, doc_ids in [
                    ("strict-match", range(3)),
                    ("flexible-extract", range(2)),
                ]
                for doc_id in doc_ids
            ],
        }
        duplicate = {
            "n_samples": 4,
            "samples": [
                {"doc_id": 0, "filter": "none"},
                {"doc_id": 1, "filter": "none"},
                {"doc_id": 2, "filter": "none"},
                {"doc_id": 2, "filter": "none"},
            ],
        }

        self.assertFalse(lifecycle.logged_samples_complete(missing, 3))
        self.assertFalse(lifecycle.logged_samples_complete(duplicate, 3))


class ServerPlanLifecycleTests(unittest.TestCase):
    def test_run_json_writes_use_atomic_replace(self):
        temporary = protocol.OUT / f".test_atomic_json_{uuid.uuid4().hex}"
        path = temporary / "record.json"
        real_replace = revision_run.os.replace
        try:
            with patch.object(
                revision_run.os, "replace", wraps=real_replace
            ) as replace:
                revision_run.write_json(path, {"ready": True})
            replace.assert_called_once()
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ready": True})
            self.assertEqual(list(temporary.glob("*.tmp")), [])
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def test_failed_torch_state_write_removes_large_temporary_file(self):
        temporary = protocol.OUT / f".test_atomic_torch_{uuid.uuid4().hex}"
        path = temporary / "state.pt"

        def interrupted_save(_value, target):
            Path(target).write_bytes(b"incomplete")
            raise RuntimeError("simulated interrupted state write")

        fake_torch = SimpleNamespace(save=interrupted_save)
        try:
            with patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                    revision_run.save_torch_atomic({}, path)
            self.assertFalse(path.exists())
            self.assertEqual(list(temporary.glob("*.tmp")), [])
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def test_ram_builder_waits_for_memory_then_runs(self):
        calls = []
        fake_fcntl = SimpleNamespace(
            LOCK_EX=1,
            LOCK_UN=2,
            flock=lambda _descriptor, mode: calls.append(mode),
        )
        temporary = protocol.OUT / f".test_ram_lock_{uuid.uuid4().hex}"
        temporary.mkdir(parents=True)
        try:
            with (
                patch.object(revision_run, "MAX_CONCURRENT_RAM_BUILDERS", 1),
                patch.object(revision_run, "MIN_AVAILABLE_RAM_GIB", 24),
                patch.object(revision_run, "RAM_BUILDER_WAIT_POLL_SECONDS", 30),
                patch.object(revision_run, "RAM_BUILDER_WAIT_TIMEOUT_SECONDS", 0),
                patch.object(revision_run, "OUT", temporary),
                patch.object(
                    revision_run,
                    "system_available_ram_gib",
                    side_effect=[18.8, 26.9],
                ),
                patch.object(revision_run, "supports_posix_file_lock", return_value=True),
                patch.object(revision_run, "status") as status,
                patch.object(revision_run.time, "monotonic", side_effect=[0, 0]),
                patch.object(revision_run.time, "sleep") as sleep,
                patch.dict(sys.modules, {"fcntl": fake_fcntl}),
            ):
                with revision_run.ram_builder_slot("build-bank", "gemma2", 41):
                    calls.append("builder")
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        self.assertEqual(
            calls,
            [fake_fcntl.LOCK_EX, "builder", fake_fcntl.LOCK_UN],
        )
        sleep.assert_called_once_with(30)
        self.assertEqual(
            [item.args[0] for item in status.call_args_list],
            ["ram_builder_wait", "ram_builder_memory_wait", "ram_builder_acquired"],
        )

    def test_ram_builder_lock_releases_when_memory_wait_times_out(self):
        calls = []
        fake_fcntl = SimpleNamespace(
            LOCK_EX=1,
            LOCK_UN=2,
            flock=lambda _descriptor, mode: calls.append(mode),
        )
        temporary = protocol.OUT / f".test_ram_lock_{uuid.uuid4().hex}"
        temporary.mkdir(parents=True)
        try:
            with (
                patch.object(revision_run, "MAX_CONCURRENT_RAM_BUILDERS", 1),
                patch.object(revision_run, "MIN_AVAILABLE_RAM_GIB", 24),
                patch.object(revision_run, "RAM_BUILDER_WAIT_POLL_SECONDS", 30),
                patch.object(revision_run, "RAM_BUILDER_WAIT_TIMEOUT_SECONDS", 60),
                patch.object(revision_run, "OUT", temporary),
                patch.object(revision_run, "system_available_ram_gib", return_value=23),
                patch.object(revision_run, "supports_posix_file_lock", return_value=True),
                patch.object(revision_run, "status"),
                patch.object(
                    revision_run.time,
                    "monotonic",
                    side_effect=[0, 0, 30, 60],
                ),
                patch.object(revision_run.time, "sleep"),
                patch.dict(sys.modules, {"fcntl": fake_fcntl}),
            ):
                with self.assertRaisesRegex(RuntimeError, "after waiting 60 seconds"):
                    with revision_run.ram_builder_slot("build-bank", "gemma2", 41):
                        self.fail("the low-RAM guard must fail before the builder runs")
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        self.assertEqual(calls, [fake_fcntl.LOCK_EX, fake_fcntl.LOCK_UN])

    def test_complete_full_result_skips_before_state_lookup(self):
        with (
            patch.object(revision_run, "require_locked_batch"),
            patch.object(revision_run, "gsm8k_complete", return_value=True),
            patch.object(revision_run, "dataset_provenance", return_value={"manifest_sha256": "data"}),
            patch.object(
                revision_run,
                "model_provenance",
                return_value={"resolved_revision": "a" * 40},
            ),
            patch.object(
                revision_run,
                "read_jsonl",
                return_value=[
                    {
                        "protocol_version": protocol.PROTOCOL_VERSION,
                        "dataset_manifest_sha256": "data",
                        "model_revision": "a" * 40,
                        "canonical_test_set": "openai/gsm8k/main:test:all-1319",
                        "eval_batch_size_per_gpu": protocol.DEFAULT_EVAL_BATCH_SIZE,
                        "max_new_tokens": protocol.MAX_NEW_TOKENS,
                    }
                ],
            ),
            patch.object(revision_run, "configure_direct_eval") as configure,
            patch.object(revision_run, "status"),
        ):
            revision_run.evaluate_full("qwen05", "gptq_w5", 97, 4, 256, False)
        configure.assert_not_called()

    def test_plan_never_accumulates_materialized_states(self):
        outstanding = 0
        peak = 0
        for command in commands():
            if "run.py materialize " in command or "run.py quantize-uniform " in command:
                outstanding += 1
                peak = max(peak, outstanding)
            elif "run.py cleanup-state " in command:
                outstanding -= 1
                self.assertGreaterEqual(outstanding, 0)
        self.assertEqual(outstanding, 0)
        self.assertEqual(peak, 1)

    def test_causal_patch_precedes_qwen_w4_cleanup(self):
        plan = list(commands())
        causal = plan.index(
            "python experiments/revision_full/causal_patch.py run "
            "--model qwen05 --calib-seed 41"
        )
        cleanup = plan.index(
            "python experiments/revision_full/run.py cleanup-state "
            "--model qwen05 --calib-seed 41 --variant gptq_w4"
        )
        self.assertLess(causal, cleanup)

    def test_plan_runs_hardware_preflight_before_any_model_command(self):
        plan = list(commands())
        hardware = plan.index(
            "python experiments/revision_full/server_preflight.py "
            "--expected-gpus 2 --concurrent-models 2"
        )
        first_model = next(
            index for index, command in enumerate(plan) if "--model " in command
        )
        self.assertLess(hardware, first_model)

    def test_two_gpu_shards_partition_every_model_command(self):
        gpu0 = shard_commands(["gemma2", "qwen15"])
        gpu1 = shard_commands(["smollm", "qwen05"])
        self.assertEqual(len(gpu0), len(set(gpu0)))
        self.assertEqual(len(gpu1), len(set(gpu1)))
        self.assertFalse(set(gpu0) & set(gpu1))
        model_commands = {
            command for command in commands() if "--model " in command
        }
        self.assertEqual(set(gpu0) | set(gpu1), model_commands)
        self.assertFalse(any("analyze.py" in command for command in gpu0 + gpu1))

    def test_plan_uses_model_specific_locked_random_allocation_ids(self):
        plan = list(commands())
        dynamic = [command for command in plan if "allocation-ids" in command]
        self.assertEqual(len(dynamic), 6)
        self.assertTrue(
            all("allocation_ids=$(python " in command for command in dynamic)
        )
        self.assertTrue(all(
            "for allocation_id in $allocation_ids;" in command for command in dynamic
        ))
        self.assertFalse(any("--variant random_29" in command for command in plan))

    def test_shard_setup_includes_both_preflight_gates(self):
        shard = shard_commands(["qwen05"], include_setup=True)
        self.assertIn(
            "python experiments/revision_full/server_preflight.py "
            "--expected-gpus 2 --concurrent-models 2",
            shard,
        )
        self.assertIn(
            "python experiments/revision_full/readiness.py --stage preflight",
            shard,
        )


if __name__ == "__main__":
    unittest.main()
