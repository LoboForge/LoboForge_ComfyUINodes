import folder_paths

from .live_preview_common import (
    execution_ids,
    new_temp_prefix,
    push_live_tiles,
    save_image_tiles,
)

NODE_CLASS = "LoboForgeLiveImagePreview"


class LoboForgeLiveImagePreview:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.prefix_append = new_temp_prefix()
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Drop on the image line after generate. Each fan-out item appears in the gallery as it finishes.",
                }),
            },
            "optional": {
                "filename_text": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional filename caption stored in image metadata (e.g. from Load Folder Images).",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_images"
    OUTPUT_NODE = True
    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Live Preview Image gallery for list fan-out sinks (e.g. Load Folder → Preview). "
        "Same click-to-zoom / arrow-key browse as stock Preview Image; tiles appear as "
        "each item finishes. For Flux/SamplerCustomAdvanced batches, use LoboForge "
        "Sampler Custom Advanced (Live Preview) instead — a post-decode sink only runs "
        "after every sample finishes."
    )

    def preview_images(self, images, filename_text="", prompt=None, extra_pnginfo=None, unique_id=None):
        prompt_id, list_index, unique_id = execution_ids(unique_id)
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
        return {"ui": ui}


NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgeLiveImagePreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge Preview Images (Live)",
}
