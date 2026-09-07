"""CPU-only checks for the revised diagnostic result contracts."""

import csv
import json
import shutil
import unittest
import uuid
from unittest.mock import patch

import torch

from experiments.revision_full import analyze as revision_analysis
from experiments.revision_full import error_analysis, protocol
from experiments.revision_full.causal_patch import target_metrics
from experiments.revision_full.external_baselines import validate_external_config


class DiagnosticTests(unittest.TestCase):
    def test_module_control_rows_cover_each_model_variant_once(self):
        root = protocol.OUT / f".test_analysis_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        variants = [
            "qkv_only",
            "o_only",
            "ffn_only",
            "qkv_priority_matched",
            "o_priority_matched",
            "ffn_priority_matched",
            "hessian_diag_matched",
        ]
        models = {
            "model_a": {"display_name": "Model A"},
            "model_b": {"display_name": "Model B"},
        }
        try:
            for model in models:
                directory = root / model / f"calib_{protocol.RANDOM_CALIB_SEED}"
                directory.mkdir(parents=True)
                for variant in variants:
                    (directory / f"{variant}.json").write_text(
                        json.dumps({"parameter_weighted_average_bits": 4.9}),
                        encoding="utf-8",
                    )
            with (
                patch.object(revision_analysis, "MODEL_SPECS", models),
                patch.object(revision_analysis, "STATE_METADATA_DIR", root),
                patch.object(revision_analysis, "correctness", return_value={0: 1}),
                patch.object(
                    revision_analysis,
                    "paired",
                    return_value={"mcnemar_p_exact": 1.0},
                ),
            ):
                rows = revision_analysis.module_placement_control_rows()
        finally:
            shutil.rmtree(root, ignore_errors=True)

        self.assertEqual(len(rows), len(models) * len(variants))
        self.assertEqual(
            {(row["model_key"], row["variant"]) for row in rows},
            {(model, variant) for model in models for variant in variants},
        )

    def test_target_metrics_include_trace_and_final_answer_spans(self):
        logits = torch.zeros((1, 4, 5), dtype=torch.float32)
        input_ids = torch.tensor([[0, 1, 2, 3]])
        metrics = target_metrics(
            logits, logits, input_ids, prompt_length=2, final_answer_start=3
        )
        self.assertEqual(metrics["target_tokens"], 2)
        self.assertEqual(metrics["final_answer_tokens"], 1)
        self.assertAlmostEqual(metrics["target_nll"], metrics["final_answer_nll"])

    def test_annotation_summary_unblinds_two_output_labels(self):
        root = protocol.OUT / f".test_diagnostics_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            annotations = root / "qwen05__c41__blinded_annotation.csv"
            fields = [
                "annotation_id",
                "double_code_required",
                "rater1_output_a_label",
                "rater1_output_b_label",
                "rater2_output_a_label",
                "rater2_output_b_label",
                "consensus_output_a_label",
                "consensus_output_b_label",
            ]
            with annotations.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "annotation_id": "0",
                        "double_code_required": "1",
                        "rater1_output_a_label": "arithmetic",
                        "rater1_output_b_label": "correct",
                        "rater2_output_a_label": "arithmetic",
                        "rater2_output_b_label": "correct",
                        "consensus_output_a_label": "arithmetic",
                        "consensus_output_b_label": "correct",
                    }
                )
            key = root / "qwen05__c41__blinding_key.json"
            key.write_text(
                json.dumps(
                    {
                        "0": {
                            "doc_id": 0,
                            "output_order": ["gptq_w4__c41", "sg_mmp__c41"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            error_analysis.summarize_annotations(annotations)

            summary = json.loads(
                annotations.with_name(
                    annotations.stem + "__summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["consensus_labeled_cases"], 1)
            self.assertEqual(summary["double_coded_outputs"], 2)
            self.assertEqual(
                summary["per_method_consensus_counts"]["sg_mmp__c41"]["correct"],
                1,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_external_config_recomputes_exact_bit_budget(self):
        root = protocol.OUT / f".test_external_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            path = root / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "method": "tacq",
                        "source_commit": "a" * 40,
                        "model_revision": "b" * 40,
                        "test_data_used_for_selection": False,
                        "calibration_and_selection_data": {"split": "train"},
                        "command": "python official.py",
                        "environment_lock": {"python": "3.12"},
                        "state_identity": {
                            "state_sha256": "c" * 64,
                            "mask_sha256": "d" * 64,
                            "source_precision_bank_sha256": "e" * 64,
                            "gradient_chunk_sha256": ["f" * 64],
                            "selected_fp16_parameters": 10,
                            "eligible_parameters": 100,
                        },
                        "validity_receipts": {
                            "train_only_smoke_sha256": "1" * 64,
                            "generated": 32,
                            "state_sha256": "c" * 64,
                        },
                        "adaptations_from_official_source": [],
                        "budget_search": {"uses_test_accuracy": False},
                        "bit_width_parameter_counts": {"4": 90, "16": 10},
                        "bit_accounting_scope": "locked eligible projections",
                        "parameter_weighted_average_bits": 5.2,
                        "canonical_evaluator": {
                            "dataset": "openai/gsm8k/main:test",
                            "n": 1319,
                            "decoding": "greedy",
                            "max_new_tokens": 256,
                            "online_stop": "generated-question-marker-v1",
                        },
                        "calibration_seed": 41,
                        "adaptation_freeze": {
                            "importance_train_samples": 128,
                            "importance_sample_selection_seed": 20260906,
                            "importance_target_doc_ids_exclude_fewshot_ids": [0, 1, 2, 3, 4],
                            "importance_batch_size": 1,
                            "gradient_accumulation": "sum abs",
                            "gradient_compute_dtype": "float16",
                            "gradient_accumulator_dtype": "float32",
                            "importance_loss": "full causal loss",
                            "importance_max_length": 2048,
                            "importance_normalization": "none",
                            "mask_granularity": "element",
                            "tie_breaking": "index",
                            "budget_rounding": "floor",
                            "importance_recomputed_per_calibration_seed": False,
                            "hyperparameters_frozen_before_test": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = validate_external_config(
                path,
                method="tacq",
                model_revision="b" * 40,
                source_commit="a" * 40,
                average_bits=5.2,
                expected_parameter_count=100,
                calibration_seed=41,
            )
            self.assertEqual(config["parameter_weighted_average_bits"], 5.2)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
