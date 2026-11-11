"""Drop-in KSampler that grows a stock Preview Image gallery mid list-map."""

import comfy.samplers
import folder_paths
import nodes

from .live_preview_common import (
    execution_ids,
    new_temp_prefix,
    push_live_tiles,
    save_image_tiles,
)

NODE_CLASS = "LoboForgeKSamplerLive"


class LoboForgeKSamplerLive:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.prefix_append = new_temp_prefix("loboksamp")
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "The random seed used for creating the noise.",
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "The number of steps used in the denoising process.",
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "The algorithm used when sampling.",
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "The scheduler controls how noise is gradually removed to form the image.",
                }),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "The amount of denoising applied; lower values keep more of the initial image structure.",
                }),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "Connect VAE to decode a live gallery image after each fan-out sample.",
                }),
                "filename_text": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional filename stored in image metadata (e.g. Load Folder Images filename_text).",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    OUTPUT_NODE = True
    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Same as KSampler, plus a live Preview Image gallery. "
        "Wire VAE (and optional filename_text). After each list fan-out sample finishes, "
        "decodes into the growing stock gallery (click to zoom, arrows to browse) — "
        "without waiting for VAEDecode / a post-decode sink."
    )

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        vae=None,
        filename_text="",
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        (out,) = nodes.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, denoise=denoise,
        )

        ui = {}
        if vae is not None:
            prompt_id, list_index, unique_id = execution_ids(unique_id)
            preview_latent = out["samples"]
            if getattr(preview_latent, "is_nested", False):
                preview_latent = preview_latent.unbind()[0]
            images = vae.decode(preview_latent)
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            results = save_image_tiles(
                images,
                self.output_dir,
                self.prefix_append,
                filename_text=filename_text,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
                compress_level=self.compress_level,
            )
            ui = push_live_tiles(unique_id, prompt_id, list_index, results)

        if ui:
            return {"ui": ui, "result": (out,)}
        return (out,)


NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgeKSamplerLive,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge KSampler (Live Preview)",
}
