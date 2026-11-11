from .load_image_batch import list_batch_images

NODE_CLASS = "LoboForgeBatchIncrement"


def build_index_list(start, count, step, wrap):
    """Build count indices starting at start, stepping by step; wrap>0 applies modulo."""
    start = int(start)
    count = max(1, int(count))
    step = int(step)
    wrap = int(wrap)
    indices = []
    for i in range(count):
        value = start + (i * step)
        if wrap > 0:
            value %= wrap
        indices.append(value)
    return indices


def resolve_count(count, path, pattern):
    """count>=1 uses the widget; count==0 sizes from folder via path+pattern."""
    count = int(count)
    if count > 0:
        return count
    path = (path or "").strip()
    if not path:
        raise RuntimeError(
            "LoboForge Batch Increment (legacy): count=0 needs a folder path."
        )
    image_paths = list_batch_images(path, pattern or "*")
    if not image_paths:
        raise RuntimeError(
            f"LoboForge Batch Increment (legacy): count=0 but no images in `{path}` "
            f"matching `{pattern or '*'}`"
        )
    return len(image_paths)


class LoboForgeBatchIncrement:
    """Legacy index driver. Use LoboForge Load Folder Images instead."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {
                    "default": 0,
                    "min": -0x7fffffffffffffff,
                    "max": 0x7fffffffffffffff,
                    "step": 1,
                    "control_after_generate": "increment",
                }),
                "count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 150000,
                    "step": 1,
                }),
                "step": ("INT", {
                    "default": 1,
                    "min": -100000,
                    "max": 100000,
                    "step": 1,
                }),
                "wrap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 150000,
                    "step": 1,
                }),
            },
            "optional": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*", "multiline": False}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "emit_indices"
    CATEGORY = "LoboForge/legacy"
    DESCRIPTION = (
        "LEGACY — use LoboForge Load Folder Images (how_many + order in one node)."
    )

    def emit_indices(self, start, count, step, wrap, path="", pattern="*"):
        resolved = resolve_count(count, path, pattern)
        indices = build_index_list(start, resolved, step, wrap)
        print(
            f"[LoboForge Batch Increment (legacy)] start={start} count={len(indices)} "
            f"(requested={count}) step={step} wrap={wrap} → {indices}"
        )
        return (indices,)


NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgeBatchIncrement,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge Batch Increment (legacy)",
}
