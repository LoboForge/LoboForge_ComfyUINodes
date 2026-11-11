"""Small utility nodes for core+LoboForge-only VideoDirector workflows."""

import gc


class LoboForgeAnySwitch:
    """Pick the first non-null optional input (rgthree Any Switch replacement)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_01": ("*",),
                "any_02": ("*",),
                "any_03": ("*",),
                "any_04": ("*",),
                "any_05": ("*",),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("*",)
    FUNCTION = "switch"
    DESCRIPTION = "Return the first connected non-null input among any_01..any_05."

    def switch(self, any_01=None, any_02=None, any_03=None, any_04=None, any_05=None):
        for value in (any_01, any_02, any_03, any_04, any_05):
            if value is not None:
                return (value,)
        raise RuntimeError("LoboForge Any Switch: all inputs were empty")


class LoboForgeImpactSwitch:
    """Index-select among optional inputs (ImpactSwitch replacement)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "input1": ("*",),
                "input2": ("*",),
                "input3": ("*",),
                "input4": ("*",),
                "input5": ("*",),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("*", "STRING", "INT")
    RETURN_NAMES = ("selected_value", "selected_label", "selected_index")
    FUNCTION = "switch"
    DESCRIPTION = "Select inputN by 1-based index."

    def switch(self, select, input1=None, input2=None, input3=None, input4=None, input5=None):
        inputs = {1: input1, 2: input2, 3: input3, 4: input4, 5: input5}
        idx = int(select)
        if idx not in inputs:
            raise RuntimeError(f"LoboForge Impact Switch: select={idx} out of range")
        value = inputs[idx]
        if value is None:
            raise RuntimeError(f"LoboForge Impact Switch: input{idx} is empty")
        return (value, f"input{idx}", idx)


class LoboForgeMemoryCleanup:
    """Passthrough barrier that runs gc + optional CUDA cache clear."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": ("*",),
            },
            "optional": {
                "empty_cuda_cache": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("anything",)
    FUNCTION = "cleanup"
    DESCRIPTION = "Pass through any value after gc.collect() and optional torch.cuda.empty_cache()."

    def cleanup(self, anything, empty_cuda_cache=True):
        gc.collect()
        if empty_cuda_cache:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        return (anything,)


NODE_CLASS_MAPPINGS = {
    "LoboForgeAnySwitch": LoboForgeAnySwitch,
    "LoboForgeImpactSwitch": LoboForgeImpactSwitch,
    "LoboForgeMemoryCleanup": LoboForgeMemoryCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoboForgeAnySwitch": "LoboForge Any Switch",
    "LoboForgeImpactSwitch": "LoboForge Impact Switch",
    "LoboForgeMemoryCleanup": "LoboForge Memory Cleanup",
}
