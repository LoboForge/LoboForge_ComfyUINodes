"""Minimal replacements for third-party nodes used by H3_Turbo_batchLoop_fixed.

Covers only the sockets/behavior that workflow actually wires — not full pack clones.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from fractions import Fraction

import torch

import comfy.model_management
import folder_paths
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
from comfy_extras.nodes_minimax_h3 import MiniMaxH3LatentUpscaleWithModel


ENCODER_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)


def megapixels_to_pixel_size(latent_h, latent_w, megapixels, align=32, downsample=16):
    """Aspect-preserving pixel W×H for an H3 video latent at target megapixels."""
    h_in = max(1, int(latent_h))
    w_in = max(1, int(latent_w))
    mp = max(0.01, float(megapixels))
    align = max(1, int(align))
    downsample = max(1, int(downsample))
    target_pixels = mp * 1024 * 1024
    aspect = w_in / h_in
    h_pixel = math.sqrt(target_pixels / aspect)
    w_pixel = h_pixel * aspect
    w_aligned = round(w_pixel / align) * align
    h_aligned = round(h_pixel / align) * align
    w_final = max(downsample, round(w_aligned / downsample) * downsample)
    h_final = max(downsample, round(h_aligned / downsample) * downsample)
    return int(w_final), int(h_final)


def apply_film_grain(images, intensity, saturation_mix=0.25):
    """Fast film-grain on float NHWC images in [0, 1]."""
    if images is None or images.numel() == 0:
        return images
    intensity = float(intensity)
    if intensity <= 0:
        return images
    saturation_mix = float(saturation_mix)
    device = images.device
    grain = torch.randn(images.shape, device=device, dtype=images.dtype)
    grain[..., 0] *= 2.0
    grain[..., 2] *= 3.0
    gray = grain[..., 1:2]
    noise = grain * saturation_mix + gray * (1.0 - saturation_mix)
    return (images + noise * intensity).clamp(0.0, 1.0)


class LoboForgeSeed:
    """INT seed with -1/-2/-3 meaning “new random seed” (rgthree Seed stand-in)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": -0xFFFFFFFFFFFFFFF, "max": 0xFFFFFFFFFFFFFFF}),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "get_seed"
    DESCRIPTION = "Output seed. Values -1, -2, or -3 draw a new random seed each run."

    @classmethod
    def IS_CHANGED(cls, seed):
        if int(seed) in (-1, -2, -3):
            return random.random()
        return int(seed)

    def get_seed(self, seed):
        seed = int(seed)
        if seed in (-1, -2, -3):
            seed = random.randint(0, 1125899906842624)
        return (seed,)


class LoboForgeExecutionBarrier:
    """Passthrough with optional run_after dependency (ControlFlowUtils HaltExecution stand-in)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
            },
            "optional": {
                "run_after": ("*",),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("*",)
    FUNCTION = "run"
    DESCRIPTION = (
        "Pass input through unchanged. Wire run_after so this node executes after another branch."
    )

    def run(self, input, run_after=None):
        return (input,)


class LoboForgeModelPassthrough:
    """MODEL identity — replaces preview-override / empty LoRA-stack helpers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "preview_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "passthrough"
    DESCRIPTION = "Return the model unchanged. Optional sockets exist only for workflow wiring."

    def passthrough(self, model, vae=None, audio_vae=None, preview_frames=0):
        return (model,)


class LoboForgeMiniMaxH3LatentUpscaleMegapixels:
    """Megapixels-mode H3 latent upscale via core LatentUpscaleModelLoader + MiniMaxH3LatentUpscaleWithModel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model_name": (folder_paths.get_filename_list("latent_upscale_models"),),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "align": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale"
    DESCRIPTION = (
        "Upscale MiniMax-H3 video latent to a target megapixel area using the core latent upscale model."
    )

    def upscale(self, latent, model_name, megapixels, align=32):
        samples = latent["samples"]
        video = samples.tensors[0] if getattr(samples, "is_nested", False) else samples
        # Expected [B, C, T, H, W] or [B, C, H, W]
        if video.ndim == 5:
            h_in, w_in = int(video.shape[-2]), int(video.shape[-1])
        elif video.ndim == 4:
            h_in, w_in = int(video.shape[-2]), int(video.shape[-1])
        else:
            raise RuntimeError(
                f"LoboForge MiniMax H3 Latent Upscale: unexpected latent rank {video.ndim}"
            )

        width, height = megapixels_to_pixel_size(h_in, w_in, megapixels, align=align, downsample=16)
        if width // 16 == w_in and height // 16 == h_in:
            return (latent,)

        model = LatentUpscaleModelLoader.execute(model_name).args[0]
        out = MiniMaxH3LatentUpscaleWithModel.execute(model, latent, width, height)
        return (out.args[0],)


class LoboForgeEncodeVideo:
    """Encode images(+audio) to VIDEO with optional film grain — Lokken post-process subset."""

    @classmethod
    def INPUT_TYPES(cls):
        # Slot order (required then optional) must stay stable for saved workflows:
        # 0 images, 1 fps, 2 crf, 3 grain, 4 grain_intensity, 5 saturation_mix,
        # 6 encoder_preset, 7 audio_pass, 8 video
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "grain": ("BOOLEAN", {"default": False}),
                "grain_intensity": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 1.0, "step": 0.001}),
                "saturation_mix": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "encoder_preset": (list(ENCODER_PRESETS), {"default": "medium"}),
            },
            "optional": {
                "audio_pass": ("AUDIO",),
                "video": ("VIDEO",),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("VIDEO", "FLOAT", "STRING")
    RETURN_NAMES = ("VIDEO", "FPS", "stats")
    FUNCTION = "encode"
    DESCRIPTION = (
        "Build an H.264 VIDEO from images (or an input video), optional audio, and optional film grain."
    )

    def encode(
        self,
        images,
        fps,
        crf,
        grain=False,
        grain_intensity=0.025,
        saturation_mix=0.25,
        encoder_preset="medium",
        audio_pass=None,
        video=None,
    ):
        if video is not None:
            components = video.get_components()
            images = components.images
            fps = float(components.frame_rate)
            if audio_pass is None:
                audio_pass = components.audio
        if images is None or images.shape[0] == 0:
            raise RuntimeError("LoboForge Encode Video: needs images or video frames")

        frames = images
        if grain and float(grain_intensity) > 0:
            device = comfy.model_management.get_torch_device()
            frames = apply_film_grain(
                frames.to(device),
                grain_intensity,
                saturation_mix=saturation_mix,
            ).to(device="cpu")

        video_out = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=frames,
                audio=audio_pass,
                frame_rate=Fraction(round(float(fps) * 1000), 1000),
            )
        )

        temp_dir = tempfile.TemporaryDirectory(prefix="loboforge_video_")
        path = os.path.join(temp_dir.name, "video.mp4")
        video_out.save_to(
            path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            crf=float(crf),
            preset=str(encoder_preset),
        )
        file_video = InputImpl.VideoFromFile(path)
        # Keep the temp dir alive for as long as the VIDEO object is used downstream.
        file_video._loboforge_temp_dir = temp_dir  # noqa: SLF001
        stats = f"frames={int(frames.shape[0])} fps={float(fps):.3f} crf={int(crf)} preset={encoder_preset} grain={bool(grain)}"
        return (file_video, float(fps), stats)


NODE_CLASS_MAPPINGS = {
    "LoboForgeSeed": LoboForgeSeed,
    "LoboForgeExecutionBarrier": LoboForgeExecutionBarrier,
    "LoboForgeModelPassthrough": LoboForgeModelPassthrough,
    "LoboForgeMiniMaxH3LatentUpscaleMegapixels": LoboForgeMiniMaxH3LatentUpscaleMegapixels,
    "LoboForgeEncodeVideo": LoboForgeEncodeVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoboForgeSeed": "LoboForge Seed",
    "LoboForgeExecutionBarrier": "LoboForge Execution Barrier",
    "LoboForgeModelPassthrough": "LoboForge Model Passthrough",
    "LoboForgeMiniMaxH3LatentUpscaleMegapixels": "LoboForge MiniMax H3 Latent Upscale (Megapixels)",
    "LoboForgeEncodeVideo": "LoboForge Encode Video",
}
