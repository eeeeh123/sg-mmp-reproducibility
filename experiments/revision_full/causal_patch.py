"""Teacher-forced activation patching for a causal error-propagation test."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from experiments.revision_full.protocol import (
    CAUSAL_PATCH_N,
    GSM8K_TEST_SIZE,
    MODEL_SPECS,
    RANDOM_CALIB_SEED,
    RESULTS_DIR,
    fixed_causal_patch_indices,
    state_path,
)
from experiments.revision_full.run import (
    configure_determinism,
    get_dataset,
    load_model_tokenizer,
    require_protocol,
)


PATCH_DIR = RESULTS_DIR / "causal_patch"


def result_path(model_key: str, calib_seed: int) -> Path:
    return PATCH_DIR / f"{model_key}__gptq_w4__c{calib_seed}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def target_metrics(logits, reference_logits, input_ids, prompt_length: int) -> dict:
    import torch
    import torch.nn.functional as functional

    start = max(0, prompt_length - 1)
    candidate = logits[:, start:-1, :].float()
    reference = reference_logits[:, start:-1, :].float()
    targets = input_ids[:, prompt_length:]
    if candidate.shape[1] != targets.shape[1] or targets.numel() == 0:
        raise RuntimeError("Teacher-forced target span is empty or misaligned")
    nll = functional.cross_entropy(
        candidate.reshape(-1, candidate.shape[-1]), targets.reshape(-1)
    )
    cosine = functional.cosine_similarity(candidate, reference, dim=-1).mean()
    reference_probs = functional.softmax(reference, dim=-1)
    kl = functional.kl_div(
        functional.log_softmax(candidate, dim=-1),
        reference_probs,
        reduction="batchmean",
    ) / candidate.shape[1]
    return {
        "target_nll": float(nll.item()),
        "target_logit_cosine": float(cosine.item()),
        "target_kl_from_fp16": float(kl.item()),
    }


def patch_hook(reference_hidden):
    def hook(module, args, output):
        patched = reference_hidden.to(
            output[0].device if isinstance(output, tuple) else output.device,
            dtype=output[0].dtype if isinstance(output, tuple) else output.dtype,
        )
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    return hook


def run(model_key: str, calib_seed: int, force: bool) -> None:
    import torch
    from experiments.fix_gsm8k_500.direct_eval import build_fewshot, build_model_prompts
    from ptq.eval import cleanup_gpu
    from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu

    lock = require_protocol()
    locked = lock.get("causal_patch_diagnostic", {})
    expected = fixed_causal_patch_indices()
    if locked.get("indices") != expected or locked.get("n") != CAUSAL_PATCH_N:
        raise RuntimeError("Protocol lock does not contain the v2 causal patch subset")
    path = result_path(model_key, calib_seed)
    if force and path.exists():
        path.unlink()
    done = {int(row["doc_id"]) for row in read_jsonl(path)}
    state = state_path(model_key, calib_seed, "gptq_w4")
    if not state.exists():
        raise FileNotFoundError(f"Run gptq_w4 materialization first: {state}")

    configure_determinism(calib_seed)
    fp16_model, tokenizer = load_model_tokenizer(model_key)
    w4_model, second_tokenizer = load_model_tokenizer(model_key)
    del second_tokenizer
    quant_state = torch.load(state, map_location="cpu", weights_only=False)
    apply_mixed_precision_to_model_gpu(w4_model, quant_state)
    del quant_state
    layers = list(w4_model.model.layers)
    train, test = get_dataset()
    prefix = build_fewshot(train, k=5)

    for ordinal, doc_id in enumerate(expected, start=1):
        if doc_id in done:
            continue
        example = test[doc_id]
        prompt = build_model_prompts(
            model_key, tokenizer, train, prefix, [example["question"]]
        )[0]
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(fp16_model.device)
        answer_ids = tokenizer(
            example["answer"], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(fp16_model.device)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        prompt_length = int(prompt_ids.shape[1])

        with torch.inference_mode():
            fp16_output = fp16_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            w4_output = w4_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        baseline = target_metrics(
            w4_output.logits, fp16_output.logits, input_ids, prompt_length
        )
        fp16_metrics = target_metrics(
            fp16_output.logits, fp16_output.logits, input_ids, prompt_length
        )
        patches = []
        for layer_index, layer in enumerate(layers):
            reference_hidden = fp16_output.hidden_states[layer_index + 1].detach()
            handle = layer.register_forward_hook(patch_hook(reference_hidden))
            try:
                with torch.inference_mode():
                    patched_output = w4_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                metrics = target_metrics(
                    patched_output.logits,
                    fp16_output.logits,
                    input_ids,
                    prompt_length,
                )
            finally:
                handle.remove()
            patches.append({"layer": layer_index, **metrics})
            del patched_output, reference_hidden
            torch.cuda.empty_cache()

        append_jsonl(
            path,
            {
                "doc_id": doc_id,
                "question": example["question"],
                "prompt_tokens": prompt_length,
                "target_tokens": int(answer_ids.shape[1]),
                "fp16": fp16_metrics,
                "gptq_w4": baseline,
                "patches": patches,
            },
        )
        print(f"{model_key}: causal patch {ordinal}/{CAUSAL_PATCH_N}", flush=True)
        del fp16_output, w4_output, input_ids, attention_mask, prompt_ids, answer_ids
        torch.cuda.empty_cache()

    summarize(model_key, calib_seed)
    del fp16_model, w4_model, tokenizer
    cleanup_gpu()


def bootstrap_ci(values: list[float], seed: int, iters: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, iters, 250):
        batch = min(250, iters - start)
        indices = rng.integers(0, len(array), size=(batch, len(array)))
        draws.append(array[indices].mean(axis=1))
    draws = np.sort(np.concatenate(draws))
    return [
        float(draws[int(0.025 * iters)]),
        float(draws[int(0.975 * iters)]),
    ]


def sign_flip_p(values: list[float], seed: int, iters: int = 10000) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, iters, 500):
        batch = min(500, iters - start)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(array)))
        extreme += int((np.abs((signs * array).mean(axis=1)) >= observed).sum())
    return (extreme + 1) / (iters + 1)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def summarize(model_key: str, calib_seed: int) -> dict:
    rows = read_jsonl(result_path(model_key, calib_seed))
    if len(rows) != CAUSAL_PATCH_N:
        raise RuntimeError(f"Causal patch result incomplete: {len(rows)}/{CAUSAL_PATCH_N}")
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        baseline = row["gptq_w4"]
        for patch in row["patches"]:
            by_layer[int(patch["layer"])].append(
                {
                    "nll_reduction": baseline["target_nll"] - patch["target_nll"],
                    "cosine_gain": patch["target_logit_cosine"]
                    - baseline["target_logit_cosine"],
                    "kl_reduction": baseline["target_kl_from_fp16"]
                    - patch["target_kl_from_fp16"],
                }
            )
    summaries = []
    p_values = []
    for layer, values in sorted(by_layer.items()):
        nll = [row["nll_reduction"] for row in values]
        p_value = sign_flip_p(nll, seed=20268100 + layer)
        p_values.append(p_value)
        summaries.append(
            {
                "layer": layer,
                "n": len(values),
                "mean_nll_reduction": statistics.mean(nll),
                "nll_reduction_ci95": bootstrap_ci(nll, seed=20268200 + layer),
                "nll_reduction_sign_flip_p": p_value,
                "fraction_nll_improved": sum(value > 0 for value in nll) / len(nll),
                "mean_logit_cosine_gain": statistics.mean(
                    row["cosine_gain"] for row in values
                ),
                "mean_kl_reduction": statistics.mean(
                    row["kl_reduction"] for row in values
                ),
            }
        )
    for row, corrected in zip(summaries, holm(p_values)):
        row["nll_reduction_sign_flip_p_holm"] = corrected
    result = {
        "model_key": model_key,
        "model": MODEL_SPECS[model_key]["display_name"],
        "calibration_seed": calib_seed,
        "n": CAUSAL_PATCH_N,
        "test_size": GSM8K_TEST_SIZE,
        "design": "teacher-forced layer-output replacement: GPTQ-W4 activation replaced by aligned FP16 activation",
        "outcome": "gold reasoning-trace token NLL, logit cosine, and KL from FP16",
        "layers": summaries,
    }
    output = PATCH_DIR / f"{model_key}__gptq_w4__c{calib_seed}__summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    run_parser.add_argument("--calib-seed", type=int, default=RANDOM_CALIB_SEED)
    run_parser.add_argument("--force", action="store_true")
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    summary_parser.add_argument("--calib-seed", type=int, default=RANDOM_CALIB_SEED)
    args = parser.parse_args()
    if args.command == "run":
        run(args.model, args.calib_seed, args.force)
    else:
        summarize(args.model, args.calib_seed)


if __name__ == "__main__":
    main()
