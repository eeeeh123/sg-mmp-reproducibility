"""CPU-only checks for the revised diagnostic result contracts."""

import csv
import json
import shutil
import unittest
import uuid

import torch

from experiments.revision_full import error_analysis, protocol
from experiments.revision_full.causal_patch import target_metrics
from experiments.revision_full.external_baselines import validate_external_config


class DiagnosticTests(unittest.TestCase):
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
                        "adaptations_from_official_source": [],
                        "budget_search": {"uses_test_accuracy": False},
                        "bit_width_parameter_counts": {"4": 75, "8": 25},
                        "bit_accounting_scope": "locked eligible projections",
                        "parameter_weighted_average_bits": 5.0,
                        "canonical_evaluator": {
                            "dataset": "openai/gsm8k/main:test",
                            "n": 1319,
                            "decoding": "greedy",
                            "max_new_tokens": 256,
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
                average_bits=5.0,
                expected_parameter_count=100,
            )
            self.assertEqual(config["parameter_weighted_average_bits"], 5.0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
