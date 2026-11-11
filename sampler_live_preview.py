"""Drop-in SamplerCustomAdvanced that grows a stock Preview Image gallery mid list-map."""

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import folder_paths
import latent_preview

from .live_preview_common import (
    execution_ids,
    new_temp_prefix,
    push_live_tiles,
    save_image_tiles,
)

NODE_CLASS = "LoboForgeSamplerCustomAdvancedLive"


class LoboForgeSamplerCustomAdvancedLive:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.prefix_append = new_temp_prefix("lobosamp")
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
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

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    OUTPUT_NODE = True
    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Same as SamplerCustomAdvanced, plus a live Preview Image gallery. "
        "Wire VAE (and optional filename_text). After each list fan-out sample finishes, "
        "decodes into the growing stock gallery (click to zoom, arrows to browse) — "
        "without waiting for VAEDecode / the sink."
    )

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        vae=None,
        filename_text="",
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        latent = latent_image
        samples_in = latent["samples"]
        latent = latent.copy()
        samples_in = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            samples_in,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = samples_in

        noise_mask = latent.get("noise_mask")

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent),
            samples_in,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples

        if "x0" in x0_output:
            x0 = x0_output["x0"]
            if samples.is_nested and not x0.is_nested:
                latent_shapes = [x.shape for x in samples.unbind()]
                x0 = comfy.nested_tensor.NestedTensor(
                    comfy.utils.unpack_latents(x0, latent_shapes)
                )
            x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        ui = {}
        if vae is not None:
            prompt_id, list_index, unique_id = execution_ids(unique_id)
            preview_latent = out_denoised["samples"]
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
            return {"ui": ui, "result": (out, out_denoised)}
        return (out, out_denoised)


NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgeSamplerCustomAdvancedLive,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge Sampler Custom Advanced (Live Preview)",
}
