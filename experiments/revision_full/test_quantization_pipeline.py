"""CPU smoke tests for the resource-aware revision quantization path."""

import unittest
from unittest.mock import patch

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
