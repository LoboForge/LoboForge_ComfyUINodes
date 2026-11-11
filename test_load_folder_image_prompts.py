"""Synthetic tests for LoboForge Load Folder Image Prompts (from metadata).

Creates temporary PNGs with embedded text chunks. Does not read user datasets
or dump prompt bodies from real CivitAI files.

MANDATORY MACHINE-WIDE PRIVACY RULE:
NEVER fetch, read, download, or submit images from the user's machine, servers, CDN, or sites to any cloud or non-local vision/multimodal model.
ONLY analyze an image if the user explicitly pasted or uploaded it in the current chat.
Do NOT auto-score, describe, or pass/fail private image outputs. Work from text, metadata, filenames, and human labels only.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_png(path: Path, **text_chunks: str) -> None:
    img = Image.new("RGB", (8, 8), color=(12, 34, 56))
    meta = PngInfo()
    for key, value in text_chunks.items():
        meta.add_text(key, value)
    img.save(path, pnginfo=meta)


class LoadFolderImagePromptsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        import types

        package_dir = Path(__file__).resolve().parent
        # Load without executing LoboForge/__init__.py (pulls CUDA-heavy deps).
        pkg_name = "custom_nodes.LoboForge"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(package_dir)]
            sys.modules.setdefault("custom_nodes", types.ModuleType("custom_nodes"))
            sys.modules[pkg_name] = pkg

        def _load(mod_name, file_name):
            full = f"{pkg_name}.{mod_name}"
            if full in sys.modules:
                return sys.modules[full]
            spec = importlib.util.spec_from_file_location(full, package_dir / file_name)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
            return mod

        _load("dialog", "dialog.py")
        load_mod = _load("load_folder_image_prompts", "load_folder_image_prompts.py")
        cls.load_mod = load_mod
        cls.ORDER_FOLDER = load_mod.ORDER_FOLDER
        cls.ORDER_RANDOM = load_mod.ORDER_RANDOM

    def test_parse_a1111_before_negative(self):
        blob = (
            "a red balloon, blue sky\n"
            "Negative prompt: blurry, low quality\n"
            "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512"
        )
        positive = self.load_mod.parse_a1111_positive(blob)
        self.assertEqual(positive, "a red balloon, blue sky")

    def test_parse_a1111_before_steps_without_negative(self):
        blob = "green hill under soft light\nSteps: 28, Sampler: DPM++ 2M, CFG scale: 5, Seed: 42"
        positive = self.load_mod.parse_a1111_positive(blob)
        self.assertEqual(positive, "green hill under soft light")

    def test_extract_from_parameters_chunk(self):
        with tempfile.TemporaryDirectory(prefix="loboforge_img_prompts_") as tmp:
            path = Path(tmp) / "a_params.png"
            _write_png(
                path,
                parameters=(
                    "synth cat on a windowsill\n"
                    "Negative prompt: dog\n"
                    "Steps: 10, Sampler: Euler, CFG scale: 7, Seed: 0, Size: 64x64"
                ),
            )
            got = self.load_mod.extract_prompt_from_image_path(str(path))
            self.assertEqual(got, "synth cat on a windowsill")

    def test_extract_from_json_prompt_field(self):
        with tempfile.TemporaryDirectory(prefix="loboforge_img_prompts_") as tmp:
            path = Path(tmp) / "b_json.png"
            payload = json.dumps({"meta": {"prompt": "synth lantern in fog"}})
            _write_png(path, Comment=payload)
            got = self.load_mod.extract_prompt_from_image_path(str(path))
            self.assertEqual(got, "synth lantern in fog")

    def test_extract_comfy_clip_text_encode(self):
        with tempfile.TemporaryDirectory(prefix="loboforge_img_prompts_") as tmp:
            path = Path(tmp) / "c_comfy.png"
            prompt_graph = {
                "3": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "synth comfy positive moon", "clip": ["4", 0]},
                    "_meta": {"title": "CLIP Text Encode (Prompt)"},
                },
                "5": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "blurry", "clip": ["4", 0]},
                    "_meta": {"title": "CLIP Text Encode (Negative)"},
                },
            }
            _write_png(path, prompt=json.dumps(prompt_graph))
            got = self.load_mod.extract_prompt_from_image_path(str(path))
            self.assertEqual(got, "synth comfy positive moon")

    def test_node_skip_missing_and_sidecar(self):
        with tempfile.TemporaryDirectory(prefix="loboforge_img_prompts_") as tmp:
            folder = Path(tmp)
            _write_png(
                folder / "a_ok.png",
                parameters="alpha subject\nSteps: 1, Sampler: Euler, CFG scale: 1, Seed: 0, Size: 8x8",
            )
            _write_png(folder / "b_empty.png")  # no metadata
            _write_png(
                folder / "c_ok.png",
                parameters="gamma subject\nSteps: 1, Sampler: Euler, CFG scale: 1, Seed: 0, Size: 8x8",
            )

            node = self.load_mod.LoboForgeLoadFolderImagePrompts()
            prompts, filenames, file_count, missing_count = node.load_folder_image_prompts(
                folder=str(folder),
                pattern="*",
                how_many=0,
                order=self.ORDER_FOLDER,
                seed=0,
                start_at=0,
                skip_missing=True,
                write_sidecar_txt=True,
            )
            self.assertEqual(file_count, 3)
            self.assertEqual(missing_count, 1)
            self.assertEqual(filenames, ["a_ok.png", "c_ok.png"])
            self.assertEqual(prompts, ["alpha subject", "gamma subject"])
            self.assertTrue((folder / "a_ok.txt").is_file())
            self.assertTrue((folder / "c_ok.txt").is_file())
            self.assertFalse((folder / "b_empty.txt").is_file())
            self.assertEqual((folder / "a_ok.txt").read_text(encoding="utf-8"), "alpha subject")

            prompts2, filenames2, _, missing2 = node.load_folder_image_prompts(
                folder=str(folder),
                pattern="*",
                how_many=0,
                order=self.ORDER_FOLDER,
                seed=0,
                start_at=0,
                skip_missing=False,
                write_sidecar_txt=False,
            )
            self.assertEqual(missing2, 1)
            self.assertEqual(len(prompts2), 3)
            self.assertEqual(filenames2[1], "b_empty.png")
            self.assertEqual(prompts2[1], "")

    def test_missing_folder_raises(self):
        node = self.load_mod.LoboForgeLoadFolderImagePrompts()
        with self.assertRaises(RuntimeError) as ctx:
            node.load_folder_image_prompts(
                folder="/tmp/loboforge_does_not_exist_img_prompts_xyz",
                pattern="*",
                how_many=0,
                order=self.ORDER_FOLDER,
                seed=0,
                start_at=0,
                skip_missing=True,
                write_sidecar_txt=False,
            )
        self.assertIn("folder not found", str(ctx.exception).lower())

    def test_registration(self):
        self.assertIn(
            "LoboForgeLoadFolderImagePrompts",
            self.load_mod.NODE_CLASS_MAPPINGS,
        )
        self.assertEqual(
            self.load_mod.NODE_DISPLAY_NAME_MAPPINGS["LoboForgeLoadFolderImagePrompts"],
            "LoboForge Load Folder Image Prompts (from metadata)",
        )
        init_path = Path(__file__).resolve().parent / "__init__.py"
        init_text = init_path.read_text(encoding="utf-8")
        self.assertIn("LOAD_FOLDER_IMAGE_PROMPTS_CLASSES", init_text)


if __name__ == "__main__":
    unittest.main()
