import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from experiments.revision_full import lifecycle, protocol, run as revision_run
from experiments.revision_full.make_server_plan import commands


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        protocol.OUT.mkdir(parents=True, exist_ok=True)
        self.root = protocol.OUT / f".test_lifecycle_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.results = self.root / "results"
        self.states = self.root / "states"
        self.metadata = self.root / "state_metadata"
        self.receipts = self.root / "receipts"
        self.patches = [
            patch.object(lifecycle, "RESULTS_DIR", self.results),
            patch.object(lifecycle, "RECEIPT_DIR", self.receipts),
            patch.object(protocol, "STATE_DIR", self.states),
            patch.object(protocol, "STATE_METADATA_DIR", self.metadata),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def write_sample(self, variant: str, ids) -> Path:
        path = lifecycle.gsm8k_sample_path("qwen05", variant, 97)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for doc_id in ids:
                stream.write(json.dumps({"doc_id": doc_id, "correct": 1}) + "\n")
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


class ServerPlanLifecycleTests(unittest.TestCase):
    def test_complete_full_result_skips_before_state_lookup(self):
        with (
            patch.object(revision_run, "require_protocol"),
            patch.object(revision_run, "gsm8k_complete", return_value=True),
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


if __name__ == "__main__":
    unittest.main()
