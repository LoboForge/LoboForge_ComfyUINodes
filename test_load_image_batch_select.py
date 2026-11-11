"""Synthetic filename-only tests for LoboForge Load Folder Images.

Compares basenames only — does not vision/score image pixels.
Uses custom_nodes/LoboForge/test_images (or /tmp fallback PNGs).

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
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_IMAGES = Path(__file__).resolve().parent / "test_images"


def _ensure_pngs(directory: Path, count: int) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    names = []
    for i in range(count):
        name = f"synth_{i:02d}.png"
        path = directory / name
        if not path.is_file():
            Image.new("RGB", (8, 8), (i * 40 % 256, 80, 160)).save(path)
        names.append(name)
    return names


class LoadFolderImagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_IMAGES.is_dir() and any(TEST_IMAGES.glob("*.png")):
            cls.folder = TEST_IMAGES
            cls.tmpdir = None
        else:
            cls.tmpdir = tempfile.TemporaryDirectory(prefix="loboforge_batch_")
            cls.folder = Path(cls.tmpdir.name)
            _ensure_pngs(cls.folder, 4)

        with mock.patch.dict(sys.modules, {"node_helpers": mock.MagicMock()}):
            from custom_nodes.LoboForge import load_image_batch as load_mod  # noqa: WPS433
        cls.load_mod = load_mod
        cls.ORDER_FOLDER = load_mod.ORDER_FOLDER
        cls.ORDER_RANDOM = load_mod.ORDER_RANDOM
        cls.image_paths = load_mod.list_batch_images(str(cls.folder), "*")
        assert cls.image_paths, f"no images in {cls.folder}"

    @classmethod
    def tearDownClass(cls):
        if cls.tmpdir is not None:
            cls.tmpdir.cleanup()

    def _filename_for(self, order: str, seed: int, start_at: int, item_i: int) -> str:
        image_id = self.load_mod.select_image_id(
            order, seed, start_at, item_i, len(self.image_paths)
        )
        return os.path.basename(self.image_paths[image_id])

    def test_folder_order_walks_distinct_files(self):
        n = len(self.image_paths)
        names = [
            self._filename_for(self.ORDER_FOLDER, seed=0, start_at=0, item_i=i)
            for i in range(n)
        ]
        self.assertEqual(len(set(names)), n, f"expected {n} distinct filenames, got {names}")

    def test_legacy_order_names_map_to_folder_order(self):
        from custom_nodes.LoboForge.load_image_batch import normalize_order

        self.assertEqual(normalize_order("single_image"), self.ORDER_FOLDER)
        self.assertEqual(normalize_order("incremental_image"), self.ORDER_FOLDER)
        self.assertEqual(normalize_order("in order"), self.ORDER_FOLDER)

    def test_random_seed_zero_varies_with_item_index(self):
        names = [
            self._filename_for(self.ORDER_RANDOM, seed=0, start_at=0, item_i=i)
            for i in range(len(self.image_paths))
        ]
        self.assertGreater(
            len(set(names)),
            1,
            f"random order with seed=0 must not collapse all items to one file: {names}",
        )

    def test_random_same_seed_and_index_is_stable(self):
        a = self._filename_for(self.ORDER_RANDOM, seed=42, start_at=3, item_i=0)
        b = self._filename_for(self.ORDER_RANDOM, seed=42, start_at=3, item_i=0)
        self.assertEqual(a, b)

    def test_node_how_many_zero_returns_all_filenames_via_list(self):
        import importlib
        import types

        node_helpers = types.SimpleNamespace(
            pillow=lambda fn, *a, **k: fn(*a, **k) if callable(fn) else fn,
        )
        with mock.patch.dict(sys.modules, {"node_helpers": node_helpers}):
            import custom_nodes.LoboForge.images as images_mod
            import custom_nodes.LoboForge.load_image_batch as load_mod

            importlib.reload(images_mod)
            importlib.reload(load_mod)

            node = load_mod.LoboForgeLoadImageBatch()
            n = len(self.image_paths)
            images, filenames, file_count = node.load_folder_images(
                folder=str(self.folder),
                pattern="*",
                how_many=0,
                order=self.ORDER_FOLDER,
                seed=0,
                start_at=0,
                allow_RGBA_output="false",
                filename_text_extension="true",
            )
            self.assertEqual(file_count, n)
            self.assertIsInstance(images, list)
            self.assertIsInstance(filenames, list)
            self.assertEqual(len(images), n)
            self.assertEqual(len(filenames), n)
            self.assertEqual(len(set(filenames)), n, f"non-distinct: {filenames}")
            self.assertTrue(load_mod.LoboForgeLoadImageBatch.OUTPUT_IS_LIST[0])
            self.assertTrue(load_mod.LoboForgeLoadImageBatch.OUTPUT_IS_LIST[1])

    def test_node_how_many_four_returns_four_distinct(self):
        import importlib
        import types

        node_helpers = types.SimpleNamespace(
            pillow=lambda fn, *a, **k: fn(*a, **k) if callable(fn) else fn,
        )
        with mock.patch.dict(sys.modules, {"node_helpers": node_helpers}):
            import custom_nodes.LoboForge.images as images_mod
            import custom_nodes.LoboForge.load_image_batch as load_mod

            importlib.reload(images_mod)
            importlib.reload(load_mod)

            node = load_mod.LoboForgeLoadImageBatch()
            n = len(self.image_paths)
            want = min(4, n)
            _images, filenames, file_count = node.load_folder_images(
                folder=str(self.folder),
                pattern="*",
                how_many=want,
                order=self.ORDER_FOLDER,
                seed=0,
                start_at=0,
                allow_RGBA_output="false",
                filename_text_extension="true",
            )
            self.assertEqual(file_count, n)
            self.assertEqual(len(filenames), want)
            self.assertEqual(len(set(filenames)), want, f"non-distinct: {filenames}")

    def test_missing_folder_raises(self):
        import importlib
        import types

        node_helpers = types.SimpleNamespace(
            pillow=lambda fn, *a, **k: fn(*a, **k) if callable(fn) else fn,
        )
        with mock.patch.dict(sys.modules, {"node_helpers": node_helpers}):
            import custom_nodes.LoboForge.images as images_mod
            import custom_nodes.LoboForge.load_image_batch as load_mod

            importlib.reload(images_mod)
            importlib.reload(load_mod)
            node = load_mod.LoboForgeLoadImageBatch()
            with self.assertRaises(RuntimeError) as ctx:
                node.load_folder_images(
                    folder="/tmp/loboforge_does_not_exist_xyz",
                    pattern="*",
                    how_many=0,
                    order=self.ORDER_FOLDER,
                    seed=0,
                    start_at=0,
                    allow_RGBA_output="false",
                )
            self.assertIn("folder not found", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
