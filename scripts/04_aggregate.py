"""汇总所有评测结果为 CSV/表格。

用法: python scripts/04_aggregate.py
"""

import sys
import json
import csv

sys.path.insert(0, ".")

from ptq.config import MODELS, QUANT_CONFIGS


def load_jsonl(path: str) -> list:
    results = []
    try:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except FileNotFoundError:
        pass
    return results


def main():
    ppl_results = load_jsonl("results/perplexity.jsonl")
    task_results = load_jsonl("results/task_results.jsonl")

    # 建立索引: (model, method) -> ppl
    ppl_map = {}
    for r in ppl_results:
        ppl_map[(r["model"], r["method"])] = r["perplexity"]

    # 建立索引: (model, method) -> {task: score}
    task_map = {}
    for r in task_results:
        task_map[(r["model"], r["method"])] = r["scores"]

    # ---- 主结果表 (Table 1) ----
    print("\n" + "=" * 100)
    print("Table 1: WikiText-2 Perplexity (lower is better)")
    print("=" * 100)

    print(f"{'Model':<20}", end="")
    for q in QUANT_CONFIGS:
        print(f"  {q['method']:>8}  ", end="")
    print()

    for m in MODELS:
        print(f"{m['name']:<20}", end="")
        for q in QUANT_CONFIGS:
            ppl = ppl_map.get((m["name"], q["method"]))
            if ppl is not None:
                print(f"  {ppl:>8.2f}  ", end="")
            else:
                print(f"  {'--':>8}  ", end="")
        print()

    # ---- 下游任务表 (Table 2) ----
    tasks_order = ["mmlu", "hellaswag", "arc_challenge", "gsm8k"]
    print("\n" + "=" * 100)
    print("Table 2: Downstream Task Accuracy (higher is better)")
    print("=" * 100)

    for task in tasks_order:
        print(f"\n--- {task.upper()} ---")
        print(f"{'Model':<20}", end="")
        for q in QUANT_CONFIGS:
            print(f"  {q['method']:>8}  ", end="")
        print()

        for m in MODELS:
            print(f"{m['name']:<20}", end="")
            for q in QUANT_CONFIGS:
                scores = task_map.get((m["name"], q["method"]), {})
                score = scores.get(task)
                if score is not None:
                    print(f"  {score:>7.2f}% ", end="")
                else:
                    print(f"  {'--':>8}  ", end="")
            print()

    # ---- 退化分析 (Table 3) ----
    print("\n" + "=" * 100)
    print("Table 3: Quantization Degradation (Δ from FP16)")
    print("=" * 100)

    for m in MODELS:
        print(f"\n{m['name']}:")
        fp16_ppl = ppl_map.get((m["name"], "fp16"))
        if fp16_ppl is None:
            continue
        print(f"  FP16 baseline PPL: {fp16_ppl:.2f}")
        for q in QUANT_CONFIGS:
            if q["method"] == "fp16":
                continue
            ppl = ppl_map.get((m["name"], q["method"]))
            if ppl is not None:
                delta = ppl - fp16_ppl
                ratio = ppl / fp16_ppl
                print(f"  {q['method']:>12}: PPL={ppl:.2f}  Δ=+{delta:.2f}  ratio={ratio:.3f}")

    # ---- 导出 CSV ----
    with open("results/summary_perplexity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Model"] + [q["method"] for q in QUANT_CONFIGS]
        writer.writerow(header)
        for m in MODELS:
            row = [m["name"]]
            for q in QUANT_CONFIGS:
                ppl = ppl_map.get((m["name"], q["method"]))
                row.append(f"{ppl:.4f}" if ppl is not None else "")
            writer.writerow(row)

    with open("results/summary_tasks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Model", "Method"] + tasks_order
        writer.writerow(header)
        for m in MODELS:
            for q in QUANT_CONFIGS:
                row = [m["name"], q["method"]]
                scores = task_map.get((m["name"], q["method"]), {})
                for task in tasks_order:
                    s = scores.get(task)
                    row.append(f"{s:.2f}" if s is not None else "")
                writer.writerow(row)

    print("\nCSV files saved to results/summary_perplexity.csv and results/summary_tasks.csv")


if __name__ == "__main__":
    main()
