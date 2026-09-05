"""Tests for aggregation of parallel LIBERO suite result sidecars."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/collect_libero_eval_results.py"
)
SPEC = importlib.util.spec_from_file_location("collect_libero_eval_results", MODULE_PATH)
COLLECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COLLECTOR)


def suite_payload(checkpoint, task_suite, successes, episodes):
    return {
        "status": "completed",
        "checkpoint": str(checkpoint),
        "task_suite": task_suite,
        "evaluation": {"seed": 7, "num_trials_per_task": 50, "include_state": True},
        "total_successes": successes,
        "total_episodes": episodes,
        "success_rate": successes / episodes,
    }


class LiberoEvalResultCollectionTests(unittest.TestCase):
    def _checkpoint(self, root, name="steps_10000_pytorch_model.pt"):
        checkpoint = root / "run" / "checkpoints" / name
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        return checkpoint

    def test_collects_complete_checkpoint_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = self._checkpoint(Path(temp_dir))
            for suite, successes, episodes in (("libero_spatial", 8, 10), ("libero_goal", 14, 20)):
                COLLECTOR.write_json_atomic(
                    COLLECTOR.suite_result_path(checkpoint, suite),
                    suite_payload(checkpoint, suite, successes, episodes),
                )

            output, summary = COLLECTOR.collect_results(
                checkpoint,
                ["libero_spatial", "libero_goal"],
                result_timestamp="20260905T074512123456Z",
            )

            self.assertEqual(output.name, "eval_result_10000_20260905T074512123456Z.json")
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["result_timestamp"], "20260905T074512123456Z")
            self.assertEqual(summary["overall"], {"successes": 22, "episodes": 30, "success_rate": 22 / 30})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["suites"]["libero_goal"]["successes"], 14)
            self.assertNotIn("per_task", summary["suites"]["libero_goal"])

    def test_marks_missing_suite_as_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = self._checkpoint(Path(temp_dir))
            COLLECTOR.write_json_atomic(
                COLLECTOR.suite_result_path(checkpoint, "libero_spatial"),
                suite_payload(checkpoint, "libero_spatial", 5, 10),
            )

            output, summary = COLLECTOR.collect_results(
                checkpoint,
                ["libero_spatial", "libero_goal"],
                result_timestamp="20260905T074512123456Z",
            )

            self.assertTrue(output.is_file())
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["missing_suites"], ["libero_goal"])
            self.assertEqual(summary["overall"]["success_rate"], 0.5)

    def test_final_model_uses_final_model_result_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            checkpoint = root / "final_model" / "pytorch_model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            self.assertEqual(COLLECTOR.checkpoint_id(checkpoint), "final_model")
            self.assertEqual(COLLECTOR.suite_result_path(checkpoint, "libero_10").name, "final_model_libero_10.json")

    def test_step_checkpoint_uses_numeric_timestep_identifier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = self._checkpoint(Path(temp_dir), "steps_250000_pytorch_model.pt")

            self.assertEqual(COLLECTOR.checkpoint_id(checkpoint), "250000")
