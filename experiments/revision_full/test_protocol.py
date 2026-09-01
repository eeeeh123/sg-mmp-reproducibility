"""CPU-only checks for the locked revision protocol."""

import unittest

from experiments.revision_full import protocol
from experiments.revision_full.protocol import (
    CALIB_SEEDS,
    CAUSAL_PATCH_N,
    QKV_SHORT_NAMES,
    SCREEN_N,
    average_bits,
    fixed_causal_patch_indices,
    make_disjoint_screen_splits,
    make_unique_random_layer_allocations,
    make_unique_random_module_allocations,
    role_priority_budget_match,
    scored_budget_match,
    select_layers_under_budget,
)
from experiments.revision_full.server_preflight import (
    EXPECTED_VERSIONS,
    ram_thresholds,
    storage_thresholds,
)
from experiments.revision_full.download_core_datasets import snapshot_sha256
from experiments.revision_full.download_models import stable_model_record


class ProtocolTests(unittest.TestCase):
    def test_three_calibration_seeds_are_frozen(self):
        self.assertEqual(len(CALIB_SEEDS), 3)

    def test_causal_patch_subset_is_fixed_and_unique(self):
        indices = fixed_causal_patch_indices()
        self.assertEqual(len(indices), CAUSAL_PATCH_N)
        self.assertEqual(len(set(indices)), CAUSAL_PATCH_N)
        self.assertTrue(all(0 <= index < 1319 for index in indices))

    def test_screen_splits_are_disjoint_and_reserve_fewshot(self):
        splits = make_disjoint_screen_splits(train_size=7473, n=SCREEN_N)
        seen = set()
        for split in splits:
            current = set(split["indices"])
            self.assertEqual(len(current), SCREEN_N)
            self.assertFalse(current & set(range(5)))
            self.assertFalse(current & seen)
            seen |= current

    def test_budgeted_selection_never_exceeds_target(self):
        modules = []
        for layer in range(4):
            for short in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]:
                modules.append(
                    {
                        "name": f"model.layers.{layer}.{short}",
                        "layer": layer,
                        "short": short,
                        "n_params": 100,
                    }
                )
        ranking = [{"layer": layer, "mean_drop": 4 - layer} for layer in range(4)]
        selected = select_layers_under_budget(ranking, modules, target_avg_bits=7.0)
        self.assertLessEqual(selected["actual_avg_bits"], 7.0)
        qkv = {row["name"] for row in modules if row["short"] in QKV_SHORT_NAMES}
        self.assertGreaterEqual(selected["actual_avg_bits"], average_bits(modules, qkv))

    def test_role_priority_control_matches_target_budget(self):
        modules = [
            {
                "name": f"layer.{index}.{short}",
                "layer": index,
                "short": short,
                "n_params": 100,
            }
            for index in range(3)
            for short in ["q_proj", "o_proj", "gate_proj"]
        ]
        target = {row["name"] for row in modules[:3]}
        matched = role_priority_budget_match(
            modules, target, {"o_proj"}, seed=7
        )
        self.assertEqual(matched["actual_w8_params"], matched["target_w8_params"])
        self.assertTrue(set(matched["preferred_module_names"]))

    def test_scored_control_respects_target_budget(self):
        modules = [
            {
                "name": f"layer.{index}.q_proj",
                "layer": index,
                "short": "q_proj",
                "n_params": 100,
            }
            for index in range(6)
        ]
        target = {row["name"] for row in modules[:3]}
        scores = {row["name"]: float(index + 1) for index, row in enumerate(modules)}
        matched = scored_budget_match(modules, target, scores)
        self.assertEqual(matched["actual_w8_params"], matched["target_w8_params"])

    def test_random_controls_are_unique_deterministic_and_budget_matched(self):
        modules = [
            {
                "name": f"model.layers.{layer}.{short}",
                "layer": layer,
                "short": short,
                "n_params": 100,
            }
            for layer in range(6)
            for short in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        ]
        target = {
            row["name"]
            for row in modules
            if row["short"] in QKV_SHORT_NAMES or row["layer"] in {0, 1}
        }
        layers = make_unique_random_layer_allocations(
            modules, target, [0, 1], count=5, seed=9
        )
        module_sets = make_unique_random_module_allocations(
            modules, target, count=5, seed=10
        )
        self.assertEqual(len({tuple(row) for row in layers}), 5)
        self.assertEqual(len({tuple(row) for row in module_sets}), 5)
        target_bits = average_bits(modules, target)
        for selected in module_sets:
            self.assertAlmostEqual(
                average_bits(modules, set(selected)), target_bits, places=8
            )

    def test_two_gpu_storage_threshold_sums_concurrent_states(self):
        thresholds = storage_thresholds([9.90, 7.88, 6.41, 1.75], 2)
        self.assertAlmostEqual(thresholds["estimated_state_peak_gib"], 17.78)
        self.assertEqual(thresholds["minimum_shared_free_gib"], 55)
        self.assertEqual(thresholds["recommended_shared_free_gib"], 92)

    def test_two_gpu_low_ram_mode_serializes_only_builders(self):
        low_ram = ram_thresholds(2, 1)
        self.assertEqual(
            low_ram["mode"],
            "serialized_ram_builders_with_parallel_gpu_evaluation",
        )
        self.assertEqual(low_ram["minimum_total_gib"], 30)
        self.assertEqual(low_ram["minimum_available_gib"], 24)
        stricter = ram_thresholds(2, 1, configured_min_available_gib=28)
        self.assertEqual(stricter["minimum_available_gib"], 28)
        full_concurrency = ram_thresholds(2, 2)
        self.assertEqual(full_concurrency["minimum_total_gib"], 64)
        self.assertEqual(full_concurrency["minimum_available_gib"], 48)

    def test_preflight_version_lock_matches_server_requirements(self):
        requirements = {}
        for line in (protocol.ROOT / "requirements-server.txt").read_text(
            encoding="utf-8"
        ).splitlines():
            if "==" in line:
                package, version = line.split("==", 1)
                requirements[package] = version
        distributions = {"lm_eval": "lm-eval"}
        for module, version in EXPECTED_VERSIONS.items():
            package = distributions.get(module, module)
            self.assertEqual(requirements.get(package), version)

    def test_dataset_identity_ignores_cache_path_and_creation_time(self):
        def manifest(path, created):
            return {
                "schema_version": 2,
                "created_at_utc": created,
                "core": {
                    "openai/gsm8k/main/test": {
                        "splits": {"data": {"rows": 1319, "fingerprint": "abc"}},
                        "cache_files": [
                            {"path": path, "bytes": 7, "sha256": "f" * 64}
                        ],
                    }
                },
                "panels": {"tasks": {}, "datasets": {}, "cache_files": []},
            }

        self.assertEqual(
            snapshot_sha256(manifest("/server/a.arrow", "first")),
            snapshot_sha256(manifest("/other/a.arrow", "second")),
        )

    def test_model_identity_ignores_download_timestamp(self):
        base = {
            "repo_id": "org/model",
            "resolved_revision": "a" * 40,
            "local_directory": "models/model",
            "weight_bytes": 10,
            "weight_file_records": [
                {"name": "model.safetensors", "bytes": 10, "sha256": "b" * 64}
            ],
        }
        first = {**base, "downloaded_at_utc": "first"}
        second = {**base, "downloaded_at_utc": "second"}
        self.assertEqual(stable_model_record(first), stable_model_record(second))


if __name__ == "__main__":
    unittest.main()
