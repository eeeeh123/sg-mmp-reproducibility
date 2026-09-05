"""CPU smoke tests for the resource-aware revision quantization path."""

import json
import shutil
import unittest
import uuid
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from types import SimpleNamespace

from experiments.revision_full.format_control import score_choice_batch, score_choices
from ptq.data import _packed_random_segments
from ptq.quant.gptq import (
    _hessian_from_calib,
    collect_linear_inputs,
    gptq_quantize_linear_multi,
)
from ptq.quant.mixed_precision import compose_precision_state
from ptq.eval import run_eval_on_model
from experiments.revision_full import protocol


class _TinyTokenizer:
    eos_token_id = 99

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) % 31 + 1 for char in text]}


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 5, bias=False)
        self.l2 = nn.Linear(5, 3, bias=False)

    def forward(self, input_ids, use_cache=False):
        del use_cache
        values = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        return self.l2(torch.relu(self.l1(values)))


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyBackbone()


class _ChoiceTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, **_kwargs):
        ids = [ord(char) % 15 + 1 for char in text]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor if return_tensors == "pt" else ids}


class _ChoiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        vocabulary = torch.arange(16, device=input_ids.device).float()
        logits = input_ids.unsqueeze(-1).float() * 0.01 + vocabulary
        return SimpleNamespace(logits=logits)


class _GenerationInputs(dict):
    def to(self, device):
        return self


class SampleResumeTests(unittest.TestCase):
    def setUp(self):
        from experiments.fix_gsm8k_500 import direct_eval

        self.direct = direct_eval
        self.root = protocol.OUT / f".test_sample_resume_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.path = self.root / "samples.jsonl"
        self.row = {
            "doc_id": 0, "correct": 1, "protocol_version": protocol.PROTOCOL_VERSION,
            "dataset_manifest_sha256": "data", "model_revision": "model",
            "eval_batch_size_per_gpu": 2, "max_new_tokens": 256,
        }
        self.examples = [{"question": f"question {i}", "answer": "#### 2"} for i in range(3)]
        for item in [
            patch.object(direct_eval, "sample_path", return_value=self.path),
            patch.object(direct_eval, "fixed_indices", return_value=[0, 1, 2]),
            patch.object(direct_eval, "get_dataset", return_value=([], self.examples)),
            patch.object(direct_eval, "ROW_METADATA", {
                key: self.row[key] for key in
                ["protocol_version", "dataset_manifest_sha256", "model_revision"]
            }),
            patch.object(direct_eval, "status"),
            patch.object(direct_eval, "cleanup_gpu"),
        ]:
            item.start()
            self.addCleanup(item.stop)

    def save(self, rows):
        self.path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return self.path.read_bytes()

    def test_partial_samples_append_only_missing_ids_and_keep_existing_bytes(self):
        before = self.save([self.row])
        tokenizer = Mock(eos_token_id=99, pad_token_id=0)
        tokenizer.return_value = _GenerationInputs(input_ids=torch.ones((2, 2), dtype=torch.long))
        tokenizer.batch_decode.return_value = ["#### 2", "#### 2"]
        model = SimpleNamespace(
            device="cpu", generate=Mock(return_value=torch.tensor([[1, 1, 2, 99]] * 2))
        )
        with (
            patch.object(self.direct, "load_model", return_value=(model, tokenizer)),
            patch.object(self.direct, "build_fewshot", return_value=""),
            patch.object(self.direct, "build_model_prompts", return_value=["q1", "q2"]) as prompts,
            patch.object(self.direct, "summarize_one"),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            self.direct.evaluate("smollm", "random_0__c41", 3, 2, 256)
        self.assertEqual(prompts.call_args.args[-1], ["question 1", "question 2"])
        self.assertTrue(self.path.read_bytes().startswith(before))
        rows = self.direct.read_jsonl(self.path)
        self.assertEqual([row["doc_id"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[0], self.row)
        model.generate.assert_called_once()

    def test_duplicate_or_incompatible_samples_are_rejected_without_append(self):
        for rows in [
            [self.row, self.row],
            [{**self.row, "dataset_manifest_sha256": "changed"}],
            [{**self.row, "model_revision": "changed"}],
            [{**self.row, "eval_batch_size_per_gpu": 4}],
            [{**self.row, "max_new_tokens": 128}],
        ]:
            with self.subTest(rows=rows), patch.object(self.direct, "load_model") as load:
                before = self.save(rows)
                with self.assertRaisesRegex(RuntimeError, "do not append mixed evidence"):
                    self.direct.evaluate("smollm", "random_0__c41", 3, 2, 256)
                load.assert_not_called()
                self.assertEqual(self.path.read_bytes(), before)


class QuantizationPipelineTests(unittest.TestCase):
    def test_packed_segments_are_exact_and_deterministic(self):
        tokenizer = _TinyTokenizer()
        first = _packed_random_segments(
            tokenizer, ["calibration text " * 20], n_samples=4, max_length=16, seed=7
        )
        second = _packed_random_segments(
            tokenizer, ["calibration text " * 20], n_samples=4, max_length=16, seed=7
        )
        self.assertEqual(tuple(first.shape), (4, 16))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.any(first == 0))

    def test_activation_capture_balances_all_samples(self):
        model = _TinyModel()
        calibration = torch.arange(32).reshape(4, 8)
        layers, activations = collect_linear_inputs(
            model, calibration, max_tokens=8
        )
        self.assertEqual([name for name, _ in layers], ["model.l1", "model.l2"])
        self.assertEqual(tuple(activations["model.l1"].shape), (8, 4))
        self.assertEqual(tuple(activations["model.l2"].shape), (8, 5))

    def test_multi_bit_bank_reuses_one_hessian_result_shape(self):
        torch.manual_seed(3)
        layer = nn.Linear(4, 3, bias=False)
        inputs = torch.randn(32, 4)
        with patch(
            "ptq.quant.gptq._hessian_from_calib", wraps=_hessian_from_calib
        ) as hessian_spy:
            choices = gptq_quantize_linear_multi(
                layer, inputs, bits_values=(4, 5, 6), group_size=2
            )
        self.assertEqual(hessian_spy.call_count, 1)
        self.assertEqual(set(choices), {4, 5, 6})
        for bits, state in choices.items():
            self.assertEqual(state["bits"], bits)
            self.assertEqual(state["w_q"].dtype, torch.int8)
            self.assertEqual(tuple(state["w_q"].shape), (3, 4))

        bank = {
            "model.l1": {
                "w4": choices[4],
                "w5": choices[5],
                "w6": choices[6],
                "scores": {},
            }
        }
        state = compose_precision_state(bank, lambda *_: "w5")
        self.assertEqual(state["model.l1"]["bits"], 5)

    def test_batched_choice_scoring_matches_single_item_scoring(self):
        model = _ChoiceModel()
        tokenizer = _ChoiceTokenizer()
        prompts = ["short prompt", "a substantially longer prompt"]
        batched = score_choice_batch(model, tokenizer, prompts)
        separate = [score_choices(model, tokenizer, prompt) for prompt in prompts]
        self.assertEqual(len(batched), 2)
        self.assertTrue(
            torch.allclose(torch.tensor(batched), torch.tensor(separate), atol=1e-6)
        )

    def test_broad_evaluator_extracts_group_metric(self):
        with (
            patch("ptq.eval.HFLM", return_value=object()),
            patch(
                "ptq.eval.simple_evaluate",
                return_value={
                    "results": {},
                    "groups": {"mmlu": {"acc,none": 0.25}},
                },
            ),
        ):
            scores = run_eval_on_model(
                object(), object(), ["mmlu"], batch_size=4, limit=None
            )
        self.assertEqual(scores, {"mmlu": 25.0})


if __name__ == "__main__":
    unittest.main()
