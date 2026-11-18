"""Tests for per-image SaveVideo filename prefixes (no vision / no real media)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class VideoFilenamePrefixTests(unittest.TestCase):
    def test_strips_extension_and_joins_folder(self):
        from custom_nodes.LoboForge.video_io import video_filename_prefix

        self.assertEqual(
            video_filename_prefix("hero shot.png", "video/H3_batch"),
            "video/H3_batch/hero_shot",
        )
        self.assertEqual(
            video_filename_prefix("a/b/c.JPEG", "video/H3_batch/"),
            "video/H3_batch/c",
        )

    def test_empty_falls_back(self):
        from custom_nodes.LoboForge.video_io import video_filename_prefix

        self.assertEqual(video_filename_prefix("", ""), "video/H3_batch/video")

    def test_save_named_writes_unique_stems(self):
        from custom_nodes.LoboForge.video_io import LoboForgeSaveVideoNamed, video_filename_prefix

        class FakeVideo:
            def get_dimensions(self):
                return (8, 8)

            def save_to(self, path, **kwargs):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"mp4")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "custom_nodes.LoboForge.video_io.folder_paths.get_output_directory",
                return_value=tmp,
            ):
                node = LoboForgeSaveVideoNamed()
                for name in ("alpha.png", "beta.jpg"):
                    out = node.save(
                        FakeVideo(),
                        filename_text=name,
                        folder="video/H3_batch",
                        format="mp4",
                        codec="h264",
                    )
                    self.assertIn("result", out)
                saved = sorted(p.name for p in Path(tmp, "video", "H3_batch").glob("*.mp4"))
                self.assertEqual(len(saved), 2)
                self.assertTrue(any(n.startswith("alpha_") for n in saved), saved)
                self.assertTrue(any(n.startswith("beta_") for n in saved), saved)
                self.assertEqual(
                    video_filename_prefix("alpha.png", "video/H3_batch"),
                    "video/H3_batch/alpha",
                )


if __name__ == "__main__":
    unittest.main()
