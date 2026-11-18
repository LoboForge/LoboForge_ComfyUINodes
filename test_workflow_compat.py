"""Unit tests for workflow_compat pure helpers (no media / no GPU models)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class MegapixelsSizeTests(unittest.TestCase):
    def test_aligns_to_downsample(self):
        from custom_nodes.LoboForge.workflow_compat import megapixels_to_pixel_size

        # latent 48x80 -> pixels 768x1280 before megapixel scale
        w, h = megapixels_to_pixel_size(48, 80, megapixels=1.0, align=32, downsample=16)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)
        # ~1 MP with 80/48 aspect
        self.assertAlmostEqual((w * h) / (1024 * 1024), 1.0, delta=0.15)

    def test_preserves_aspect_roughly(self):
        from custom_nodes.LoboForge.workflow_compat import megapixels_to_pixel_size

        w, h = megapixels_to_pixel_size(40, 40, megapixels=2.0, align=32, downsample=16)
        self.assertEqual(w, h)


class FilmGrainTests(unittest.TestCase):
    def test_zero_intensity_is_identity(self):
        from custom_nodes.LoboForge.workflow_compat import apply_film_grain

        images = torch.rand(2, 8, 8, 3)
        out = apply_film_grain(images, intensity=0.0)
        self.assertTrue(torch.equal(out, images))

    def test_positive_intensity_stays_in_range(self):
        from custom_nodes.LoboForge.workflow_compat import apply_film_grain

        images = torch.full((1, 4, 4, 3), 0.5)
        out = apply_film_grain(images, intensity=0.05, saturation_mix=0.25)
        self.assertEqual(out.shape, images.shape)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class ExecutionBarrierTests(unittest.TestCase):
    def test_passthrough(self):
        from custom_nodes.LoboForge.workflow_compat import LoboForgeExecutionBarrier

        node = LoboForgeExecutionBarrier()
        self.assertEqual(node.run("payload", run_after="ignored"), ("payload",))


class AnySwitchTests(unittest.TestCase):
    def test_first_non_null(self):
        from custom_nodes.LoboForge.util_nodes import LoboForgeAnySwitch

        node = LoboForgeAnySwitch()
        self.assertEqual(node.switch(any_01=None, any_02="b"), ("b",))


class SeedTests(unittest.TestCase):
    def test_negative_draws_random(self):
        from custom_nodes.LoboForge.workflow_compat import LoboForgeSeed

        node = LoboForgeSeed()
        out = node.get_seed(-1)[0]
        self.assertIsInstance(out, int)
        self.assertGreaterEqual(out, 0)

    def test_fixed_seed(self):
        from custom_nodes.LoboForge.workflow_compat import LoboForgeSeed

        self.assertEqual(LoboForgeSeed().get_seed(42), (42,))


if __name__ == "__main__":
    unittest.main()
