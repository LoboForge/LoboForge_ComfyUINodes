import asyncio
import glob
import hashlib
import os
import random

from .dialog import pick_path

NODE_CLASS = "LoboForgeLoadFolderPrompts"

ORDER_FOLDER = "in folder order"
ORDER_RANDOM = "random"
ORDERS = (ORDER_FOLDER, ORDER_RANDOM)

_LEGACY_ORDER = {
    "in order": ORDER_FOLDER,
    "in folder order": ORDER_FOLDER,
    "random": ORDER_RANDOM,
}


def list_batch_prompts(directory_path, pattern):
    if not directory_path or not os.path.isdir(directory_path):
        return []
    directory_path = os.path.abspath(directory_path)
    pattern = pattern or "*.txt"
    search = os.path.join(glob.escape(directory_path), pattern)
    prompt_paths = []
    for file_name in glob.glob(search, recursive=True):
        if os.path.isfile(file_name):
            prompt_paths.append(os.path.abspath(file_name))
    prompt_paths.sort()
    return prompt_paths


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_how_many(how_many, file_count):
    how_many = int(how_many)
    if how_many == 0:
        return file_count
    return how_many


def normalize_order(order):
    if order in ORDERS:
        return order
    mapped = _LEGACY_ORDER.get(order)
    if mapped is not None:
        return mapped
    raise RuntimeError(
        f"LoboForge Load Folder Prompts: unknown order `{order}` "
        f"(use {ORDER_FOLDER!r} or {ORDER_RANDOM!r})"
    )


def select_prompt_id(order, seed, start_at, item_i, file_count):
    """Pick file index for item_i in the batch (always wraps with file_count)."""
    index = int(start_at) + int(item_i)
    if order == ORDER_RANDOM:
        # seed + per-item index so fan-out items differ; same seed+index stays stable.
        random.seed(int(seed) + index)
        return int(random.random() * file_count) % file_count
    return index % file_count


def read_prompt_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


class LoboForgeLoadFolderPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Folder of prompt/caption .txt files. Use choose folder.",
                }),
                "pattern": ("STRING", {
                    "default": "*.txt",
                    "multiline": False,
                    "tooltip": "File name filter. Default *.txt.",
                }),
                "how_many": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 150000,
                    "step": 1,
                    "tooltip": "0 = every matching file. N = load that many. One Queue fans out all of them.",
                }),
                "order": (list(ORDERS), {
                    "default": ORDER_FOLDER,
                    "tooltip": "in folder order = sorted names from Start at. random = shuffle picks (uses Seed).",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Only used when order is random.",
                }),
                "start_at": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 150000,
                    "step": 1,
                    "tooltip": "Skip this many files at the start of the sorted folder (0 = begin at first file).",
                }),
            },
            "optional": {
                "filename_text_extension": (["false", "true"], {
                    "tooltip": "false = stem only (foo.txt → foo) for SaveImage filename_prefix. true = keep basename with extension.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "filename", "file_count")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "load_folder_prompts"
    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Load prompt/caption text files from a folder in one Queue. "
        "how_many=0 loads every matching file; N loads that many. "
        "prompt and filename fan out downstream (OUTPUT_IS_LIST). "
        "filename defaults to the stem (no .txt) for SaveImage prefixes. "
        "Empty files yield an empty string."
    )

    def load_folder_prompts(
        self,
        folder,
        pattern,
        how_many,
        order,
        seed,
        start_at,
        filename_text_extension="false",
    ):
        pattern = pattern or "*.txt"
        if not folder or not os.path.isdir(folder):
            raise RuntimeError(f"LoboForge Load Folder Prompts: folder not found: {folder}")

        prompt_paths = list_batch_prompts(folder, pattern)
        if not prompt_paths:
            raise RuntimeError(
                f"LoboForge Load Folder Prompts: no files in `{folder}` matching `{pattern}`"
            )

        file_count = len(prompt_paths)
        count = resolve_how_many(how_many, file_count)
        if count < 1:
            raise RuntimeError("LoboForge Load Folder Prompts: how_many resolved to 0")

        order = normalize_order(order)
        prompts = []
        filenames = []
        strip_ext = filename_text_extension != "true"

        for item_i in range(count):
            prompt_id = select_prompt_id(order, seed, start_at, item_i, file_count)
            prompt_path = prompt_paths[prompt_id]
            text = read_prompt_text(prompt_path)
            filename = os.path.basename(prompt_path)
            if strip_ext:
                filename = os.path.splitext(filename)[0]
            prompts.append(text)
            filenames.append(filename)

        print(
            f"[LoboForge Load Folder Prompts] how_many={count} (requested={how_many}) "
            f"start_at={start_at} order={order} file_count={file_count} → {filenames}"
        )
        return (prompts, filenames, file_count)

    @classmethod
    def IS_CHANGED(
        cls,
        folder,
        pattern,
        how_many,
        order,
        seed,
        start_at,
        filename_text_extension="false",
    ):
        try:
            order = normalize_order(order)
        except RuntimeError:
            return float("NaN")
        if order == ORDER_RANDOM:
            return float("NaN")
        pattern = pattern or "*.txt"
        if not folder or not os.path.isdir(folder):
            return float("NaN")
        prompt_paths = list_batch_prompts(folder, pattern)
        if not prompt_paths:
            return float("NaN")
        file_count = len(prompt_paths)
        count = resolve_how_many(how_many, file_count)
        digests = []
        for item_i in range(count):
            prompt_id = select_prompt_id(ORDER_FOLDER, seed, start_at, item_i, file_count)
            selected = prompt_paths[prompt_id]
            if not os.path.isfile(selected):
                return float("NaN")
            try:
                digests.append(file_sha256(selected))
            except OSError:
                return float("NaN")
        return f"{filename_text_extension}," + ",".join(digests)


def _register_folder_route():
    if getattr(_register_folder_route, "done", False):
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        print(f"LoboForge Load Folder Prompts folder route was not registered ({exc}).")
        return
    if getattr(PromptServer, "instance", None) is None:
        return

    @PromptServer.instance.routes.post("/loboforge/load_folder_prompts/pick_folder")
    async def loboforge_pick_prompts_folder(request):
        try:
            result = await asyncio.to_thread(pick_path, "LoboForge Load Folder Prompts", True)
        except Exception as exc:
            result = {"error": f"Folder dialog failed: {exc}"}
        return web.json_response(result)

    _register_folder_route.done = True


_register_folder_route()

NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgeLoadFolderPrompts,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge Load Folder Prompts",
}
