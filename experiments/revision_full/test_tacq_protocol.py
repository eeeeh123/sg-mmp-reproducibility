"""CPU-only tests for the rejected Shadow candidate and TaCQ extension."""

import json
import math
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch

from experiments.revision_full import analyze as revision_analysis
from experiments.revision_full import make_tacq_plan
from experiments.revision_full import readiness as revision_readiness
from experiments.revision_full import shadow_gate
from experiments.revision_full import tacq as tacq_protocol
from experiments.revision_full.question_stop import (
    BASE_GENERATION_KWARGS_SHA256,
    GeneratedQuestionStopLogitsProcessor,
    canonical_answer_prefix,
    generation_diagnostics,
)
from experiments.revision_full.external_baselines import validate_tacq_samples
from experiments.revision_full.tacq import (
    _float32_threshold,
    fp16_count_for_budget,
    official_importance_score,
)
from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu


class _AsciiTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(value)) for value in ids)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 1, bias=False)


class QuestionStopTests(unittest.TestCase):
    def test_only_generated_blank_line_question_marker_is_canonical_stop(self):
        ordinary = "The Question: wording is not a delimiter. #### 7"
        repeated = "reasoning\n#### 7\n\nQuestion: invented follow-up"
        self.assertFalse(canonical_answer_prefix(ordinary).marker_found)
        parsed = canonical_answer_prefix(repeated)
        self.assertTrue(parsed.marker_found)
        self.assertEqual(parsed.text, "reasoning\n#### 7")

    def test_logits_processor_stops_rows_independently_and_never_scans_prompt(self):
        tokenizer = _AsciiTokenizer()
        prompt = [ord(char) for char in "Question: prompt\nAnswer:"]
        first = [ord(char) for char in "#### 1\n\nQuestion:"]
        second = [ord(char) for char in "still solving"]
        width = len(prompt)
        max_generated = max(len(first), len(second))
        first += [32] * (max_generated - len(first))
        second += [32] * (max_generated - len(second))
        ids = torch.tensor([prompt + first, prompt + second])
        scores = torch.zeros((2, 256), dtype=torch.float32)
        processor = GeneratedQuestionStopLogitsProcessor(tokenizer, width, 2)
        processed = processor(ids, scores)
        self.assertEqual(processor.finished, [True, False])
        self.assertEqual(float(processed[0, 2]), 0.0)
        self.assertTrue(torch.all(processed[0, :2] < -1e20))
        self.assertTrue(torch.all(processed[1] == 0))

    def test_diagnostics_distinguish_marker_stop_from_length_truncation(self):
        row = generation_diagnostics(
            "#### 4\n\nQuestion:", [1, 2, 9], eos_token_id=9, max_new_tokens=3
        )
        self.assertEqual(row["stop_reason"], "generated_question_marker")
        self.assertFalse(row["truncated"])
        self.assertEqual(row["generation"], "#### 4")


class TacqMathTests(unittest.TestCase):
    @staticmethod
    def _sample_config():
        return {
            "state_identity": {
                "state_sha256": "state",
                "mask_sha256": "mask",
            },
            "validity_receipts": {"train_only_smoke_sha256": "smoke"},
        }

    def test_tacq_sample_contract_requires_original_protocol_and_seed(self):
        row = {
            "doc_id": 0,
            "online_question_stop": False,
            "generation_protocol": "original-max-new-tokens-256-v1",
            "calibration_seed": 41,
            "tacq_manifest_sha256": "manifest",
            "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
            "external_extension_role": "external_baseline",
            "external_extension_method_id": "external_tacq__c41",
            "tacq_state_sha256": "state",
            "tacq_mask_sha256": "mask",
            "tacq_smoke_receipt_sha256": "smoke",
            "generation": "work\n#### 4",
            "generated_token_count": 9,
            "prediction": "4",
            "gold": "4",
            "correct": 1,
            "truncated": False,
            "ended_with_eos": True,
        }
        config = self._sample_config()
        validate_tacq_samples([row], 41, "manifest", config)
        with self.assertRaisesRegex(RuntimeError, "original evaluator contract"):
            validate_tacq_samples(
                [{**row, "calibration_seed": 97}], 41, "manifest", config
            )
        with self.assertRaisesRegex(RuntimeError, "original evaluator contract"):
            validate_tacq_samples(
                [{**row, "online_question_stop": True}], 41, "manifest", config
            )

    def test_tacq_sample_contract_recomputes_canonical_prediction_and_correctness(self):
        row = {
            "doc_id": 0,
            "online_question_stop": False,
            "generation_protocol": "original-max-new-tokens-256-v1",
            "calibration_seed": 41,
            "tacq_manifest_sha256": "manifest",
            "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
            "external_extension_role": "external_baseline",
            "external_extension_method_id": "external_tacq__c41",
            "tacq_state_sha256": "state",
            "tacq_mask_sha256": "mask",
            "tacq_smoke_receipt_sha256": "smoke",
            "generation": "work\n#### 4\n\nQuestion: invented",
            "generated_token_count": 17,
            "prediction": "4",
            "gold": "4",
            "correct": 1,
            "truncated": False,
            "ended_with_eos": True,
        }
        config = self._sample_config()
        validate_tacq_samples([row], 41, "manifest", config)
        with self.assertRaisesRegex(RuntimeError, "original evaluator contract"):
            validate_tacq_samples(
                [{**row, "prediction": "5"}], 41, "manifest", config
            )
        with self.assertRaisesRegex(RuntimeError, "original evaluator contract"):
            validate_tacq_samples(
                [{**row, "tacq_mask_sha256": "another-mask"}],
                41,
                "manifest",
                config,
            )

    def test_tacq_freeze_rejects_unverified_source_commit_before_data_access(self):
        with self.assertRaisesRegex(ValueError, "pinned commit"):
            tacq_protocol.freeze_manifest("a" * 40)

    def test_tacq_freeze_cannot_replace_a_missing_manifest_after_test_access(self):
        with patch.object(tacq_protocol, "MANIFEST_PATH", Path("missing.json")), patch.object(
            tacq_protocol, "_existing_test_outputs", return_value=[Path("test.jsonl")]
        ), patch.object(
            tacq_protocol, "_existing_registrations", return_value=[]
        ):
            with self.assertRaisesRegex(RuntimeError, "after test output"):
                tacq_protocol.freeze_manifest(tacq_protocol.OFFICIAL_SOURCE_COMMIT)

    def test_tacq_requires_an_exact_untampered_shadow_pass(self):
        root = Path(__file__).resolve().parent / f".test_shadow_pass_{uuid.uuid4().hex}"
        rows = root / "rows"
        rows.mkdir(parents=True)
        paths = {}
        for model in shadow_gate.SHADOW_MODELS:
            for variant in shadow_gate.SHADOW_VARIANTS:
                path = rows / f"{model}__{variant}.jsonl"
                path.write_text('{"ok": true}\n', encoding="utf-8")
                paths[(model, variant)] = path
        manifest = {"manifest_sha256": "manifest", "total_formal_generations": 200}
        receipt_path = root / "PASS.json"
        receipt = {
            "pass": True,
            "manifest_sha256": "manifest",
            "checks": {
                "rows": 200,
                "canonical_prefix_matches": 200,
                "prediction_matches": 200,
                "correctness_matches": 200,
            },
            "errors": [],
            "row_file_sha256": {
                f"{model}/{variant}": tacq_protocol.sha256(path)
                for (model, variant), path in paths.items()
            },
        }
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        try:
            with patch.object(
                shadow_gate, "RECEIPT_PATH", receipt_path
            ), patch.object(
                shadow_gate, "require_manifest", return_value=manifest
            ), patch.object(
                shadow_gate,
                "output_path",
                side_effect=lambda model, variant: paths[(model, variant)],
            ):
                tacq_protocol.require_shadow_pass()
                paths[("qwen05", "gptq_w4")].write_text(
                    '{"ok": false}\n', encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "changed after PASS"):
                    tacq_protocol.require_shadow_pass()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_smoke_receipt_requires_exact_state_and_all_32_train_generations(self):
        manifest = {"manifest_sha256": "manifest"}
        metadata = {
            "model_key": "qwen05",
            "calibration_seed": 41,
            "state_sha256": "state",
        }
        receipt = {
            "schema": "tacq-train-smoke-v2",
            "pass": True,
            "train_only": True,
            "save_reload_validated": True,
            "manifest_sha256": "manifest",
            "model_key": "qwen05",
            "calibration_seed": 41,
            "generated": 32,
            "state_sha256": "state",
            "generation_protocol": "original-max-new-tokens-256-v1",
            "online_question_stop": False,
            "max_new_tokens": 256,
        }
        self.assertTrue(
            tacq_protocol._valid_smoke_receipt(receipt, manifest, metadata)
        )
        self.assertFalse(
            tacq_protocol._valid_smoke_receipt(
                {**receipt, "generated": 31}, manifest, metadata
            )
        )

    def test_official_contrastive_weight_product_formula(self):
        gradient = torch.tensor([2.0, 3.0])
        clean = torch.tensor([-4.0, 5.0])
        corrupt = torch.tensor([-3.0, 7.0])
        expected = torch.tensor([8.0, 30.0])
        self.assertTrue(
            torch.equal(official_importance_score(gradient, clean, corrupt), expected)
        )

    def test_budget_rounding_never_exceeds_sg_and_is_within_one_weight(self):
        count, logical = fp16_count_for_budget(101, 4.9)
        self.assertLessEqual(logical, 4.9)
        self.assertLess(4.9 - logical, 12 / 101)
        self.assertEqual(count, math.floor((4.9 - 4) * 101 / 12))

    def test_streaming_threshold_is_exact_with_ties(self):
        from experiments.revision_full import protocol

        directory = protocol.OUT / f".test_tacq_threshold_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            paths = []
            for index, values in enumerate(([1.0, 4.0, 4.0], [2.0, 3.0])):
                path = directory / f"{index}.pt"
                torch.save(torch.tensor(values, dtype=torch.float32), path)
                paths.append(path)
            self.assertEqual(_float32_threshold(paths, 1), 4.0)
            self.assertEqual(_float32_threshold(paths, 2), 4.0)
            self.assertEqual(_float32_threshold(paths, 3), 3.0)
            self.assertTrue(math.isinf(_float32_threshold(paths, 0)))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_model_level_inference_clusters_repeated_seed_items(self):
        pairs = []
        for seed_shift in (0, 1, 0):
            external = {index: 0 for index in range(4)}
            sg = {
                index: int(index in ({0, 1} if seed_shift == 0 else {0, 2}))
                for index in range(4)
            }
            pairs.append((external, sg))
        with patch.object(revision_analysis, "GSM8K_TEST_SIZE", 4):
            result = revision_analysis.hierarchical_seed_example_bootstrap(
                pairs, iters=200, seed=7
            )
        self.assertEqual(result["n_seeds"], 3)
        self.assertEqual(result["n_examples_per_seed"], 4)
        self.assertIn("paired_item_cluster_sign_flip_p", result)
        self.assertNotIn("mcnemar_p_exact", result)

    def test_tacq_state_applies_only_bitpacked_fp16_exceptions(self):
        model = _TinyModel()
        mask = np.asarray([1, 0, 0, 1], dtype=np.uint8)
        state = {
            "proj": {
                "method": "tacq_w4_fp16",
                "w_q": torch.zeros((1, 4), dtype=torch.int8),
                "scale": torch.ones((1, 2), dtype=torch.float32),
                "zero": torch.zeros((1, 2), dtype=torch.float32),
                "group_size": 2,
                "fp16_mask_packbits": torch.from_numpy(
                    np.packbits(mask, bitorder="big")
                ),
                "mask_numel": 4,
                "fp16_values": torch.tensor([7.0, 9.0], dtype=torch.float16),
            }
        }
        apply_mixed_precision_to_model_gpu(model, state)
        self.assertTrue(
            torch.equal(
                model.proj.weight.detach().cpu(),
                torch.tensor([[7.0, 0.0, 0.0, 9.0]], dtype=torch.float16),
            )
        )

    def test_contemporary_sg_control_is_bound_to_manifest_state_and_bank(self):
        root = Path(__file__).resolve().parent / f".test_control_{uuid.uuid4().hex}"
        root.mkdir()
        record_path = root / "control.json"
        sample_path = root / "samples.jsonl"
        sample_path.write_text("{}\n", encoding="utf-8")
        manifest = {
            "manifest_sha256": "manifest",
            "dataset_provenance": {"manifest_sha256": "dataset"},
            "model_provenance": {"qwen05": {"resolved_revision": "model"}},
        }
        record = {
            "schema": "tacq-contemporary-sg-control-v1",
            "protocol_version": "revision-full-v4",
            "manifest_sha256": "manifest",
            "model_key": "qwen05",
            "calibration_seed": 41,
            "method_id": "external_sg_contemporary__c41",
            "generation_protocol": "original-max-new-tokens-256-v1",
            "online_question_stop": False,
            "max_new_tokens": 256,
            "old_core_outputs_replaced_or_pooled": False,
            "samples": str(sample_path),
            "samples_sha256": "samplehash",
            "sg_state_sha256": "statehash",
            "source_precision_bank_sha256": "bankhash",
        }
        row = {
            "doc_id": 0,
            "protocol_version": "revision-full-v4",
            "dataset_manifest_sha256": "dataset",
            "model_revision": "model",
            "canonical_test_set": "openai/gsm8k/main:test:all-1319",
            "tacq_manifest_sha256": "manifest",
            "calibration_seed": 41,
            "generation_protocol": "original-max-new-tokens-256-v1",
            "online_question_stop": False,
            "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
            "external_extension_role": "contemporaneous_control",
            "external_extension_method_id": "external_sg_contemporary__c41",
            "sg_state_sha256": "statehash",
            "source_precision_bank_sha256": "bankhash",
            "eval_batch_size_per_gpu": tacq_protocol.DEFAULT_EVAL_BATCH_SIZE,
            "max_new_tokens": 256,
            "generation": "work\n#### 4",
            "prediction": "4",
            "gold": "4",
            "correct": 1,
        }
        record_path.write_text(json.dumps(record), encoding="utf-8")
        try:
            with patch.object(
                tacq_protocol, "control_record_path", return_value=record_path
            ), patch.object(
                tacq_protocol, "sha256", return_value="samplehash"
            ), patch(
                "experiments.revision_full.external_baselines.resolve_record_path",
                return_value=sample_path,
            ), patch(
                "experiments.revision_full.external_baselines.read_samples",
                return_value=[row],
            ):
                self.assertEqual(
                    tacq_protocol.require_contemporary_sg_control(
                        "qwen05", 41, manifest
                    )["source_precision_bank_sha256"],
                    "bankhash",
                )
                row["online_question_stop"] = True
                with self.assertRaisesRegex(RuntimeError, "frozen contract"):
                    tacq_protocol.require_contemporary_sg_control(
                        "qwen05", 41, manifest
                    )
        finally:
            shutil.rmtree(root, ignore_errors=True)


class ServerPlanTests(unittest.TestCase):
    def test_shadow_readiness_uses_the_untampered_pass_validator(self):
        with patch.object(revision_readiness, "core_errors", return_value=[]), patch.object(
            tacq_protocol,
            "require_shadow_pass",
            side_effect=RuntimeError("changed after PASS"),
        ):
            self.assertEqual(
                revision_readiness.shadow_errors(),
                ["invalid or missing question-stop shadow gate: changed after PASS"],
            )

    def test_existing_valid_shadow_manifest_is_a_restart_safe_noop(self):
        root = Path(__file__).resolve().parent / f".test_shadow_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            expected = {"manifest_sha256": "valid"}
            with patch.object(shadow_gate, "MANIFEST_PATH", manifest_path), patch.object(
                shadow_gate, "require_manifest", return_value=expected
            ):
                self.assertEqual(shadow_gate.prepare(), expected)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_stale_shadow_manifest_is_immutable_after_formal_output(self):
        root = Path(__file__).resolve().parent / f".test_shadow_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            with patch.object(shadow_gate, "MANIFEST_PATH", manifest_path), patch.object(
                shadow_gate,
                "require_manifest",
                side_effect=RuntimeError("commit changed"),
            ), patch.object(
                shadow_gate, "_shadow_has_started", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "refusing to change"):
                    shadow_gate.prepare()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_missing_shadow_manifest_cannot_replace_formal_output(self):
        root = Path(__file__).resolve().parent / f".test_shadow_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            with patch.object(
                shadow_gate, "MANIFEST_PATH", root / "missing.json"
            ), patch.object(
                shadow_gate, "_shadow_has_started", return_value=True
            ), patch.object(
                shadow_gate, "_tacq_has_started", return_value=False
            ):
                with self.assertRaisesRegex(RuntimeError, "without its frozen"):
                    shadow_gate.prepare()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_stale_empty_shadow_manifest_refreshes_after_code_update(self):
        root = Path(__file__).resolve().parent / f".test_shadow_{uuid.uuid4().hex}"
        root.mkdir()
        sources = {}
        for model in shadow_gate.SHADOW_MODELS:
            for variant in shadow_gate.SHADOW_VARIANTS:
                path = root / f"{model}__{variant}.jsonl"
                path.write_text("source\n", encoding="utf-8")
                sources[(model, variant)] = path
        manifest_path = root / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        try:
            with patch.object(shadow_gate, "MANIFEST_PATH", manifest_path), patch.object(
                shadow_gate,
                "require_manifest",
                side_effect=RuntimeError("commit changed"),
            ), patch.object(
                shadow_gate, "_shadow_has_started", return_value=False
            ), patch.object(
                shadow_gate, "_tacq_has_started", return_value=False
            ), patch.object(
                shadow_gate, "_tracked_worktree_is_clean", return_value=True
            ), patch.object(
                shadow_gate,
                "source_path",
                side_effect=lambda model, variant: sources[(model, variant)],
            ), patch.object(
                shadow_gate, "_complete_rows", return_value={}
            ), patch.object(
                shadow_gate, "_coverage_select", return_value=list(range(50))
            ), patch.object(
                shadow_gate.subprocess,
                "run",
                return_value=type("Result", (), {"stdout": "new-commit\n"})(),
            ), patch(
                "experiments.revision_full.run.dataset_provenance",
                return_value={"manifest_sha256": "dataset"},
            ), patch(
                "experiments.revision_full.run.model_provenance",
                return_value={"resolved_revision": "model"},
            ):
                refreshed = shadow_gate.prepare()
            self.assertEqual(refreshed["implementation_commit"], "new-commit")
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))[
                    "manifest_sha256"
                ],
                refreshed["manifest_sha256"],
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_rejected_shadow_is_not_rerun_and_extension_is_fail_closed(self):
        shadow = make_tacq_plan.commands(1, "shadow")
        tacq = make_tacq_plan.commands(1, "tacq")
        self.assertIn("exit 2", shadow[-1])
        self.assertFalse(any("capture-importance" in command for command in shadow))
        self.assertTrue(any("server_preflight.py" in command for command in tacq))
        self.assertFalse(any("shadow_gate.py" in command for command in tacq))
        self.assertTrue(any("readiness.py --stage core" in command for command in tacq))
        required_artifacts = [
            command
            for command in tacq
            if "build-bank" in command or " materialize " in command
        ]
        self.assertTrue(required_artifacts)
        self.assertTrue(all("--require-output" in command for command in required_artifacts))
        self.assertFalse(any("--force" in command for command in required_artifacts))
        self.assertTrue(any("capture-importance" in command for command in tacq))
        self.assertEqual(make_tacq_plan.commands(1, "all"), tacq)
        self.assertEqual(
            sum("evaluate-control" in command for command in tacq), 6
        )
        self.assertEqual(
            sum("tacq.py evaluate --model" in command for command in tacq), 6
        )

    def test_tacq_plan_skips_registered_seed_but_resumes_partial_seed(self):
        plan = "\n".join(make_tacq_plan.commands(0, "tacq"))
        record = (
            "experiments/revision_full/outputs/external_baselines/"
            "qwen05__tacq__c41.json"
        )
        control = (
            "experiments/revision_full/outputs/tacq/controls/"
            "qwen05__sg_contemporary__c41.json"
        )
        self.assertIn(f"if [[ ! -f {record} || ! -f {control} ]]; then", plan)
        self.assertIn("build-bank --model qwen05 --calib-seed 41", plan)
        self.assertIn("materialize --model qwen05 --calib-seed 41 --variant sg_mmp", plan)
        self.assertIn("evaluate-control --model qwen05 --calib-seed 41", plan)
        self.assertIn("cleanup --model qwen05 --calib-seed 41", plan)


if __name__ == "__main__":
    unittest.main()
