"""CPU-only tests for the Shadow/TaCQ gates and deterministic accounting."""

import shutil
import math
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch

from experiments.revision_full import analyze as revision_analysis
from experiments.revision_full import make_tacq_plan
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
    def test_tacq_sample_contract_rejects_unstopped_or_cross_seed_rows(self):
        row = {
            "doc_id": 0,
            "online_question_stop": True,
            "stop_protocol": "generated-question-marker-v1",
            "calibration_seed": 41,
            "tacq_manifest_sha256": "manifest",
            "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
            "stop_reason": "model_eos",
            "raw_generation": "work\n#### 4",
            "generation": "work\n#### 4",
            "prediction": "4",
            "gold": "4",
            "correct": 1,
            "truncated": False,
            "generated_question_marker_found": False,
            "ended_with_eos": True,
        }
        validate_tacq_samples([row], 41, "manifest")
        with self.assertRaisesRegex(RuntimeError, "stopped evaluator contract"):
            validate_tacq_samples([{**row, "calibration_seed": 97}], 41, "manifest")

    def test_tacq_sample_contract_recomputes_canonical_prediction_and_correctness(self):
        row = {
            "doc_id": 0,
            "online_question_stop": True,
            "stop_protocol": "generated-question-marker-v1",
            "calibration_seed": 41,
            "tacq_manifest_sha256": "manifest",
            "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
            "stop_reason": "generated_question_marker",
            "raw_generation": "work\n#### 4\n\nQuestion: invented",
            "generation": "work\n#### 4",
            "prediction": "4",
            "gold": "4",
            "correct": 1,
            "truncated": False,
            "generated_question_marker_found": True,
            "ended_with_eos": True,
        }
        validate_tacq_samples([row], 41, "manifest")
        with self.assertRaisesRegex(RuntimeError, "stopped evaluator contract"):
            validate_tacq_samples([{**row, "prediction": "5"}], 41, "manifest")

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


class ServerPlanTests(unittest.TestCase):
    def test_shadow_and_tacq_are_separate_fail_closed_phases(self):
        shadow = make_tacq_plan.commands(1, "shadow")
        tacq = make_tacq_plan.commands(1, "tacq")
        self.assertIn("readiness.py --stage shadow", shadow[-1])
        self.assertFalse(any("capture-importance" in command for command in shadow))
        self.assertIn("readiness.py --stage shadow", tacq[0])
        self.assertTrue(any("capture-importance" in command for command in tacq))
        self.assertEqual(make_tacq_plan.commands(1, "all"), shadow + tacq)

    def test_tacq_plan_skips_registered_seed_but_resumes_partial_seed(self):
        plan = "\n".join(make_tacq_plan.commands(0, "tacq"))
        record = (
            "experiments/revision_full/outputs/external_baselines/"
            "qwen05__tacq__c41.json"
        )
        self.assertIn(f"if [[ ! -f {record} ]]; then", plan)
        self.assertIn("build-bank --model qwen05 --calib-seed 41", plan)
        self.assertIn("cleanup --model qwen05 --calib-seed 41", plan)


if __name__ == "__main__":
    unittest.main()
