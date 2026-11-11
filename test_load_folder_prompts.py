"""Synthetic tests for LoboForge Load Folder Prompts.

Uses custom_nodes/LoboForge/test_prompts (or /tmp .txt files).
Does not read user dataset caption folders.

MANDATORY MACHINE-WIDE PRIVACY RULE:
NEVER fetch, read, download, or submit images from the user's machine, servers, CDN, or sites to any cloud or non-local vision/multimodal model.
ONLY analyze an image if the user explicitly pasted or uploaded it in the current chat.
Do NOT auto-score, describe, or pass/fail private image outputs. Work from text, metadata, filenames, and human labels only.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_PROMPTS = Path(__file__).resolve().parent / "test_prompts"


def _ensure_txts(directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "a_red_balloon.txt": "a red balloon floating in a blue sky",
        "b_green_hill.txt": "a green hill under soft sunlight",
        "c_empty.txt": "",
    }
    for name, body in files.items():
        path = directory / name
        if not path.is_file():
            path.write_text(body, encoding="utf-8")
    return sorted(files)


class LoadFolderPromptsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_PROMPTS.is_dir() and any(TEST_PROMPTS.glob("*.txt")):
            cls.folder = TEST_PROMPTS
            cls.tmpdir = None
        else:
            cls.tmpdir = tempfile.TemporaryDirectory(prefix="loboforge_prompts_")
            cls.folder = Path(cls.tmpdir.name)
            _ensure_txts(cls.folder)

        from custom_nodes.LoboForge import load_folder_prompts as load_mod

        cls.load_mod = load_mod
        cls.ORDER_FOLDER = load_mod.ORDER_FOLDER
        cls.ORDER_RANDOM = load_mod.ORDER_RANDOM
        cls.prompt_paths = load_mod.list_batch_prompts(str(cls.folder), "*.txt")
        assert cls.prompt_paths, f"no prompts in {cls.folder}"

    @classmethod
    def tearDownClass(cls):
        if cls.tmpdir is not None:
            cls.tmpdir.cleanup()

    def test_folder_order_walks_distinct_files(self):
        n = len(self.prompt_paths)
        names = [
            os.path.basename(
                self.prompt_paths[
                    self.load_mod.select_prompt_id(
                        self.ORDER_FOLDER, 0, 0, i, n
                    )
                ]
            )
            for i in range(n)
        ]
        self.assertEqual(len(set(names)), n, names)

    def test_random_seed_zero_varies_with_item_index(self):
        n = len(self.prompt_paths)
        names = [
            os.path.basename(
                self.prompt_paths[
                    self.load_mod.select_prompt_id(
                        self.ORDER_RANDOM, 0, 0, i, n
                    )
                ]
            )
            for i in range(n)
        ]
        self.assertGreater(len(set(names)), 1, names)

    def test_node_how_many_zero_includes_empty(self):
        node = self.load_mod.LoboForgeLoadFolderPrompts()
        prompts, filenames, file_count = node.load_folder_prompts(
            folder=str(self.folder),
            pattern="*.txt",
            how_many=0,
            order=self.ORDER_FOLDER,
            seed=0,
            start_at=0,
        )
        self.assertEqual(file_count, len(self.prompt_paths))
        self.assertEqual(len(prompts), file_count)
        self.assertEqual(len(filenames), file_count)
        self.assertTrue(self.load_mod.LoboForgeLoadFolderPrompts.OUTPUT_IS_LIST[0])
        self.assertTrue(self.load_mod.LoboForgeLoadFolderPrompts.OUTPUT_IS_LIST[1])
        # Default strips .txt so SaveImage filename_prefix matches the prompt stem.
        self.assertIn("c_empty", filenames)
        self.assertEqual(prompts[filenames.index("c_empty")], "")
        for name in filenames:
            self.assertFalse(name.endswith(".txt"), name)

    def test_filename_keeps_extension_when_requested(self):
        node = self.load_mod.LoboForgeLoadFolderPrompts()
        _prompts, filenames, _file_count = node.load_folder_prompts(
            folder=str(self.folder),
            pattern="*.txt",
            how_many=0,
            order=self.ORDER_FOLDER,
            seed=0,
            start_at=0,
            filename_text_extension="true",
        )
        self.assertTrue(any(name.endswith(".txt") for name in filenames), filenames)
        self.assertIn("c_empty.txt", filenames)

    def test_missing_folder_raises(self):
        node = self.load_mod.LoboForgeLoadFolderPrompts()
        with self.assertRaises(RuntimeError) as ctx:
            node.load_folder_prompts(
                folder="/tmp/loboforge_does_not_exist_prompts_xyz",
                pattern="*.txt",
                how_many=0,
                order=self.ORDER_FOLDER,
                seed=0,
                start_at=0,
            )
        self.assertIn("folder not found", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
