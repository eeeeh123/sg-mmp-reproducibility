"""CPU-only regressions for frozen selections and fail-closed shard execution."""

import json
import shutil
import subprocess
import unittest
import uuid
from unittest.mock import patch

from experiments.revision_full import lifecycle, protocol, run
from experiments.revision_full.make_server_shard import shard_commands


class SelectionResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = protocol.OUT / f".test_resume_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.path = self.root / "selections" / "smollm.json"
        self.path.parent.mkdir()
        self.dataset = {"manifest_sha256": "frozen-data"}
        self.model = {"resolved_revision": "frozen-model"}
        self.lock = {
            "screen_splits": [
                {"split_id": index, "calibration_seed": seed,
                 "seed": index, "indices_sha256": f"split-{index}"}
                for index, seed in enumerate([41, 97, 193])
            ],
            "screen_quantizer": "GPTQ-W4",
        }
        self.screens = {}
        for split in self.lock["screen_splits"]:
            index = split["split_id"]
            common = {
                "protocol_version": protocol.PROTOCOL_VERSION,
                "model_key": "smollm",
                "split_id": index,
                "split_seed": split["seed"],
                "split_indices_sha256": split["indices_sha256"],
                "calibration_seed": split["calibration_seed"],
                "dataset_manifest_sha256": self.dataset["manifest_sha256"],
                "model_revision": self.model["resolved_revision"],
                "eval_batch_size_per_gpu": protocol.DEFAULT_EVAL_BATCH_SIZE,
                "max_new_tokens": protocol.MAX_NEW_TOKENS,
                "quantizer": "GPTQ-W4",
                "group_size": protocol.GROUP_SIZE,
            }
            rows = [
                {**common, "type": "baseline"},
                {**common, "type": "layer", "layer": 10, "drop_vs_fp16": 1.0},
            ]
            self.screens[str(index)] = rows
            (self.root / f"screen_{index}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (self.root / f"metadata_{split['calibration_seed']}.json").write_text(
                json.dumps({
                    **common, "dataset_snapshot": self.dataset,
                    "model_snapshot": self.model, "screened_layers": [10],
                }), encoding="utf-8",
            )
        for item in [
            patch.object(run, "OUT", self.root),
            patch.object(run, "require_protocol", return_value=self.lock),
            patch.object(run, "dataset_provenance", return_value=self.dataset),
            patch.object(run, "model_provenance", return_value=self.model),
            patch.object(
                run, "screen_file",
                side_effect=lambda model, index: self.root / f"screen_{index}.jsonl",
            ),
            patch.object(
                run, "state_metadata_path",
                side_effect=lambda model, seed, variant: self.root / f"metadata_{seed}.json",
            ),
            patch.object(run, "status"),
        ]:
            item.start()
            self.addCleanup(item.stop)
        model_patch = patch.object(
            run, "load_model_tokenizer",
            side_effect=AssertionError("resume must not load a model or redraw allocations"),
        )
        self.load_model = model_patch.start()
        self.addCleanup(model_patch.stop)

    def selection(self, count=24):
        layers = [[index] for index in reversed(range(count))]
        modules = [[f"module.{index}"] for index in reversed(range(30))]
        manifest = {
            "count_per_family": 30,
            "layer_sets": layers,
            "module_sets": modules,
            "layer_sets_sha256": protocol.json_sha256(layers),
            "module_sets_sha256": protocol.json_sha256(modules),
        }
        if count < 30:
            manifest["layer_feasibility"] = {
                "requested_count": 30,
                "actual_count": count,
                "feasible_candidate_sets": count,
                "exhaustive": True,
            }
        return {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "model_key": "smollm",
            "test_data_used": False,
            "dataset_snapshot": self.dataset,
            "model_snapshot": self.model,
            "screen_calibration_seeds": [41, 97, 193],
            "screen_quantizer": "GPTQ-W4",
            "screen_file_sha256": {
                index: protocol.json_sha256(rows)
                for index, rows in self.screens.items()
            },
            "selected_layers": [10],
            "actual_avg_bits": 4.885416666666667,
            "random_allocation_manifest": manifest,
        }

    def save(self, selection):
        self.path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        return self.path.read_bytes()

    def test_resume_preserves_legacy_and_feasibility_aware_selections_byte_for_byte(self):
        for count in [30, 24]:
            with self.subTest(count=count):
                selection = self.selection(count)
                before = self.save(selection)
                self.assertEqual(run.select_model("smollm"), selection)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(run.allocation_ids("smollm", "layer"), list(range(count)))
        self.load_model.assert_not_called()

    def test_first_selection_writes_only_selection_and_can_then_resume(self):
        before = {
            path: path.read_bytes() for path in self.root.glob("screen_*.jsonl")
        }
        modules = [
            {"name": f"model.layers.{layer}.{short}", "layer": layer,
             "short": short, "n_params": size}
            for layer in range(24)
            for short, size in [("q_proj", 1), ("o_proj", 9)]
        ]
        with (
            patch.object(run, "load_model_tokenizer", return_value=(object(), object())),
            patch.object(run, "module_rows", return_value=modules),
            patch.object(run, "SELECTION_BOOTSTRAP_REPLICATES", 2),
        ):
            selection = run.select_model("smollm")
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), selection)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(run.select_model("smollm"), selection)
        self.load_model.assert_not_called()

    def test_resume_rejects_stale_identity_manifest_and_screen_hash_without_overwrite(self):
        for field, value in [
            ("protocol_version", "old"),
            ("model_key", "gemma2"),
            ("test_data_used", True),
            ("dataset_snapshot", {}),
            ("model_snapshot", {}),
            ("random_allocation_manifest", {}),
            ("screen_calibration_seeds", [41, 41, 41]),
            ("screen_quantizer", "changed"),
            ("screen_file_sha256", {}),
        ]:
            with self.subTest(field=field):
                selection = self.selection()
                selection[field] = value
                before = self.save(selection)
                with self.assertRaises(RuntimeError):
                    run.select_model("smollm")
                self.assertEqual(self.path.read_bytes(), before)
        self.load_model.assert_not_called()

    def test_resume_rejects_changed_or_missing_screen_without_redrawing(self):
        before = self.save(self.selection())
        screen = self.root / "screen_0.jsonl"
        screen.write_text('{"type": "changed"}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "screen"):
            run.select_model("smollm")
        screen.unlink()
        with self.assertRaisesRegex(RuntimeError, "screen"):
            run.select_model("smollm")
        self.assertEqual(self.path.read_bytes(), before)
        self.load_model.assert_not_called()

    def test_bank_cleanup_rejects_missing_or_invalid_allocation_manifest(self):
        with patch.object(lifecycle, "OUT", self.root):
            for value in [None, {}]:
                with self.subTest(value=value):
                    if value is not None:
                        self.save(value)
                    with self.assertRaises(RuntimeError):
                        lifecycle.bank_consumer_variants("smollm", 41)
                    self.assertFalse(lifecycle.bank_consumers_complete("smollm", 41))

    def test_bank_consumers_use_the_saved_feasible_count(self):
        self.save(self.selection())
        with patch.object(lifecycle, "OUT", self.root):
            variants = lifecycle.bank_consumer_variants("smollm", 41)
        self.assertIn("random_23", variants)
        self.assertNotIn("random_24", variants)
        self.assertIn("random_modules_29", variants)

    def test_readiness_reports_missing_manifest_instead_of_crashing(self):
        from experiments.revision_full import readiness

        with (
            patch.object(lifecycle, "OUT", self.root),
            patch.object(readiness, "OUT", self.root),
            patch.object(readiness, "RESULTS_DIR", self.root / "results"),
            patch.object(readiness, "MODEL_SPECS", {"smollm": protocol.MODEL_SPECS["smollm"]}),
            patch.object(readiness, "preflight_errors", return_value=[]),
            patch.object(readiness, "require_complete_sample"),
            patch.object(readiness, "require_cleanup_receipt"),
            patch.object(readiness, "require_format_control"),
            patch.object(readiness, "require_task_panels"),
        ):
            errors = readiness.core_errors()
        self.assertTrue(any("cannot verify bank consumers for smollm/c41" in error for error in errors))
        self.assertIn("missing native train-only selection for smollm", errors)


@unittest.skipUnless(shutil.which("bash"), "Bash is required for executable shard tests")
class ShardFailureTests(unittest.TestCase):
    def test_both_complete_generated_shards_pass_bash_syntax_check(self):
        for models in [["gemma2", "qwen15"], ["smollm", "qwen05"]]:
            with self.subTest(models=models):
                script = "set -euo pipefail\n" + "\n".join(shard_commands(models)) + "\n"
                result = subprocess.run(
                    [shutil.which("bash"), "--noprofile", "--norc", "-n", "-s"],
                    input=script, capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def execute_loop(self, lookup, evaluation='printf "%s\\n" "$6"'):
        command = next(
            command for command in shard_commands(["smollm"])
            if "allocation-ids" in command and "--family layer" in command
        )
        script = (
            "set -euo pipefail\n"
            "python() {\n"
            "case \"$2\" in\n"
            f"allocation-ids) {lookup} ;;\n"
            f"evaluate-allocation) {evaluation} ;;\n"
            "*) return 99 ;;\n"
            "esac\n}\n"
            f"{command}\n"
            "printf 'NEXT_STAGE\\n'\n"
        )
        return subprocess.run(
            [shutil.which("bash"), "--noprofile", "--norc", "-s"],
            input=script, capture_output=True, text=True, timeout=15,
        )

    def test_failed_lookup_stops_even_when_it_printed_partial_ids(self):
        result = self.execute_loop("printf '0 1\\n'; return 17")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_empty_lookup_cannot_silently_skip_random_family(self):
        result = self.execute_loop("return 0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_each_locked_id_executes_once_for_24_and_30_allocations(self):
        for count in [24, 30]:
            with self.subTest(count=count):
                result = self.execute_loop(f"printf '%s\\n' {{0..{count - 1}}}")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout.splitlines(),
                    [f"random_{index}" for index in range(count)] + ["NEXT_STAGE"],
                )

    def test_evaluation_failure_stops_before_remaining_ids_and_next_stage(self):
        result = self.execute_loop(
            "printf '0 1 2\\n'",
            'printf "%s\\n" "$6"; if [[ "$6" == random_1 ]]; then return 23; fi',
        )
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["random_0", "random_1"])


if __name__ == "__main__":
    unittest.main()
