# Artifact manifest

| Reported analysis or figure | Released source data or code |
|---|---|
| Study overview | Direct GSM8K-500 summaries, paired statistics, bit-budget summary, layer diagnostics, `scripts/generate_concept_figures.py` |
| SG-MMP precision policy and component evidence | `data/processed/bit_budget_summary.json`, `qwen05_ablation_analysis_gsm8k500.json`, single-layer screen, `scripts/generate_concept_figures.py` |
| Broad benchmark degradation | `data/processed/source_artifacts/results/main_results.csv`, `scripts/generate_figures.py` |
| Direct GSM8K-500 repair comparison | `data/processed/gsm8k500/`, direct-summary artifacts, `scripts/analyze_released_gsm8k500.py`, `scripts/generate_figures.py` |
| Direct-result reference values | `data/processed/expected_results.json` |
| Qwen2.5-0.5B non-overlap robustness check | `selection_eval_split_gsm8k500.json` |
| Bit-budget table | `data/processed/bit_budget_summary.json`, `experiments/analysis/bit_budget.py` |
| Module-allocation ablation | `qwen05_ablation_analysis_gsm8k500.json`, `experiments/fix_gsm8k_500/run.py` |
| Same-budget allocation comparison | `results/task_results_full.jsonl`, `experiments/exp17_same_budget/run.py`, `scripts/generate_figures.py` |
| Error-propagation interpretation | `exp02_per_layer`, `exp07_layer_replacement`, `exp14_first_divergent_step`, and `scripts/generate_concept_figures.py` |
| Layer sensitivity and divergence diagnostics | `exp02_per_layer`, `exp14_first_divergent_step`, and `scripts/generate_figures.py` |
| Calibration, LoRA, and OOD diagnostics | `results/task_results_full.jsonl` and corresponding experiment scripts |

All paths are relative to the repository root. Historical raw filenames may
use `SmolLM-1.7B`; the actual checkpoint identity is SmolLM2-1.7B, as
documented in `docs/model_provenance.md`.
