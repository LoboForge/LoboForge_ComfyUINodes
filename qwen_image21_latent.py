"""Empty latent canvas for Qwen Image 2.1 with independent width/height."""

import torch

import comfy.model_management
from nodes import MAX_RESOLUTION


class LoboForgeEmptyQwenImage21Latent:
    """Empty canvas for Qwen Image 2.1 (64 channels, 16x spatial downscale).

    Use instead of TextEncodeQwenImage21's latent when the output size/aspect
    should differ from the reference image (e.g. wide sprite sheets).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 2560,
                        "min": 16,
                        "max": MAX_RESOLUTION,
                        "step": 16,
                        "tooltip": "Output width in pixels (multiple of 16).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 16,
                        "max": MAX_RESOLUTION,
                        "step": 16,
                        "tooltip": "Output height in pixels (multiple of 16).",
                    },
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Empty Qwen Image 2.1 latent (64ch / 16x). Set width/height independently "
        "of the TextEncode reference image aspect."
    )

    def generate(self, width, height, batch_size=1):
        # Match TextEncodeQwenImage21: [B, 64, H/16, W/16]
        width = max(16, (int(width) // 16) * 16)
        height = max(16, (int(height) // 16) * 16)
        latent = torch.zeros(
            [batch_size, 64, height // 16, width // 16],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "LoboForgeEmptyQwenImage21Latent": LoboForgeEmptyQwenImage21Latent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoboForgeEmptyQwenImage21Latent": "Empty Qwen Image 2.1 Latent (LoboForge)",
}
