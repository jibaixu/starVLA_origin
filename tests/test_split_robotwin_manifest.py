import json
import tempfile
import unittest
from pathlib import Path

from tools.split_robotwin_manifest import _read_rows, split_rows, write_shards


def _rows(count: int) -> list[dict]:
    return [
        {
            "manifest_id": index,
            "dataset_root": f"/dataset/task_{index % 3}",
            "episode_index": index,
            "length": 10 + index,
        }
        for index in range(count)
    ]


class RobotwinManifestSplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_balanced(self):
        rows = _rows(16)
        first = split_rows(rows, 8, seed=42)
        second = split_rows(rows, 8, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(sum(len(shard) for shard in first), 16)
        self.assertEqual(
            len({(row["dataset_root"], row["episode_index"]) for shard in first for row in shard}),
            16,
        )
        totals = [sum(row["length"] for row in shard) for shard in first]
        self.assertLessEqual(max(totals) - min(totals), 10)

    def test_write_shards_preserves_json_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "manifest.jsonl"
            rows = _rows(8)
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "parts"
            paths = write_shards(_read_rows(source), output, num_shards=8, seed=42, prefix="full")
            self.assertEqual(len(paths), 8)
            self.assertTrue(paths[0].name.startswith("full_part"))
            loaded = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
            self.assertEqual(
                {(row["dataset_root"], row["episode_index"]) for row in loaded},
                {(row["dataset_root"], row["episode_index"]) for row in rows},
            )


if __name__ == "__main__":
    unittest.main()
