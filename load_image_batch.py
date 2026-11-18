import asyncio
import glob
import hashlib
import os
import random

from PIL import Image

import node_helpers

from .dialog import pick_path
from .images import pil_to_image_tensor

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")
NODE_CLASS = "LoboForgeLoadImageBatch"

ORDER_FOLDER = "in folder order"
ORDER_RANDOM = "random"
ORDERS = (ORDER_FOLDER, ORDER_RANDOM)

# Legacy WAS / older LoboForge combo values → current order labels.
_LEGACY_ORDER = {
    "single_image": ORDER_FOLDER,
    "incremental_image": ORDER_FOLDER,
    "in order": ORDER_FOLDER,
    "in folder order": ORDER_FOLDER,
    "random": ORDER_RANDOM,
}


def list_batch_images(directory_path, pattern):
    if not directory_path or not os.path.isdir(directory_path):
        return []
    directory_path = os.path.abspath(directory_path)
    pattern = pattern or "*"
    if pattern in ("true", "false"):
        pattern = "*"
    search = os.path.join(glob.escape(directory_path), pattern)
    image_paths = []
    for file_name in glob.glob(search, recursive=True):
        if os.path.isfile(file_name) and file_name.lower().endswith(IMAGE_EXTENSIONS):
            image_paths.append(os.path.abspath(file_name))
    image_paths.sort()
    return image_paths


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
        f"LoboForge Load Folder Images: unknown order `{order}` "
        f"(use {ORDER_FOLDER!r} or {ORDER_RANDOM!r})"
    )


def select_image_id(order, seed, start_at, item_i, file_count):
    """Pick file index for item_i in the batch (always wraps with file_count)."""
    index = int(start_at) + int(item_i)
    if order == ORDER_RANDOM:
        # seed + per-item index so fan-out items differ; same seed+index stays stable.
        random.seed(int(seed) + index)
        return int(random.random() * file_count) % file_count
    return index % file_count


class LoboForgeLoadImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Folder of images. Use choose folder.",
                }),
                "pattern": ("STRING", {
                    "default": "*",
                    "multiline": False,
                    "tooltip": "File name filter. Leave as * for every image type we support.",
                }),
                "how_many": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 150000,
                    "step": 1,
                    "tooltip": "0 = every image in the folder. N = load that many. One Queue fans out all of them.",
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
                "allow_RGBA_output": (["false", "true"],),
            },
            "optional": {
                "filename_text_extension": (["false", "true"], {
                    "tooltip": (
                        "false = stem only (foo.png → foo) for SaveVideo / "
                        "LoboForge Save Video (Named) prefixes. true = keep basename."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "filename_text", "file_count")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "load_folder_images"
    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Load images from a folder in one Queue. "
        "how_many=0 loads every matching file; N loads that many. "
        "image and filename_text fan out downstream (OUTPUT_IS_LIST) — "
        "MiniMax H3 does not iterate a stacked batch; each list item is one ref. "
        "Use LoboForge Save Video (Named) or Video Filename Prefix so each item "
        "writes its own mp4. No separate counter node."
    )

    def load_folder_images(
        self,
        folder,
        pattern,
        how_many,
        order,
        seed,
        start_at,
        allow_RGBA_output,
        filename_text_extension="false",
    ):
        pattern = pattern or "*"
        if pattern in ("true", "false"):
            pattern = "*"
        if not folder or not os.path.isdir(folder):
            raise RuntimeError(f"LoboForge Load Folder Images: folder not found: {folder}")

        image_paths = list_batch_images(folder, pattern)
        if not image_paths:
            raise RuntimeError(
                f"LoboForge Load Folder Images: no images in `{folder}` matching `{pattern}`"
            )

        file_count = len(image_paths)
        count = resolve_how_many(how_many, file_count)
        if count < 1:
            raise RuntimeError("LoboForge Load Folder Images: how_many resolved to 0")

        order = normalize_order(order)
        images = []
        filenames = []
        allow_rgba = allow_RGBA_output == "true"
        strip_ext = filename_text_extension != "true"

        for item_i in range(count):
            image_id = select_image_id(order, seed, start_at, item_i, file_count)
            image_path = image_paths[image_id]
            image = node_helpers.pillow(Image.open, image_path)
            tensor = pil_to_image_tensor(image, allow_rgba=allow_rgba)
            filename = os.path.basename(image_path)
            if strip_ext:
                filename = os.path.splitext(filename)[0]
            images.append(tensor)
            filenames.append(filename)

        print(
            f"[LoboForge Load Folder Images] how_many={count} (requested={how_many}) "
            f"start_at={start_at} order={order} file_count={file_count} → {filenames}"
        )
        return (images, filenames, file_count)

    @classmethod
    def IS_CHANGED(
        cls,
        folder,
        pattern,
        how_many,
        order,
        seed,
        start_at,
        allow_RGBA_output,
        filename_text_extension="false",
    ):
        try:
            order = normalize_order(order)
        except RuntimeError:
            return float("NaN")
        if order == ORDER_RANDOM:
            return float("NaN")
        pattern = pattern or "*"
        if pattern in ("true", "false"):
            pattern = "*"
        if not folder or not os.path.isdir(folder):
            return float("NaN")
        image_paths = list_batch_images(folder, pattern)
        if not image_paths:
            return float("NaN")
        file_count = len(image_paths)
        count = resolve_how_many(how_many, file_count)
        digests = []
        for item_i in range(count):
            image_id = select_image_id(ORDER_FOLDER, seed, start_at, item_i, file_count)
            selected = image_paths[image_id]
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
        print(f"LoboForge Load Folder Images folder route was not registered ({exc}).")
        return
    if getattr(PromptServer, "instance", None) is None:
        return

    @PromptServer.instance.routes.post("/loboforge/load_image_batch/pick_folder")
    async def loboforge_pick_batch_folder(request):
        try:
            result = await asyncio.to_thread(pick_path, "LoboForge Load Folder Images", True)
        except Exception as exc:
            result = {"error": f"Folder dialog failed: {exc}"}
        return web.json_response(result)

    _register_folder_route.done = True


_register_folder_route()

NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgeLoadImageBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge Load Folder Images",
}
