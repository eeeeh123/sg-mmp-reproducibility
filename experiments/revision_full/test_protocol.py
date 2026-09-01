"""CPU-only checks for the locked revision protocol."""

import unittest

from experiments.revision_full.protocol import (
    CALIB_SEEDS,
    CAUSAL_PATCH_N,
    QKV_SHORT_NAMES,
    SCREEN_N,
    average_bits,
    fixed_causal_patch_indices,
    make_disjoint_screen_splits,
    role_priority_budget_match,
    scored_budget_match,
    select_layers_under_budget,
)


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


if __name__ == "__main__":
    unittest.main()
