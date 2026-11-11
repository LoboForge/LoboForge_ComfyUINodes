import asyncio
import glob
import hashlib
import json
import os
import random
import re

from PIL import Image, ExifTags

from .dialog import pick_path

NODE_CLASS = "LoboForgeLoadFolderImagePrompts"

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")

ORDER_FOLDER = "in folder order"
ORDER_RANDOM = "random"
ORDERS = (ORDER_FOLDER, ORDER_RANDOM)

_LEGACY_ORDER = {
    "in order": ORDER_FOLDER,
    "in folder order": ORDER_FOLDER,
    "random": ORDER_RANDOM,
}

_NEG_MARKERS = (
    "\nNegative prompt:",
    "\r\nNegative prompt:",
    "\nnegative prompt:",
    "\r\nnegative prompt:",
)
_STEPS_MARKERS = (
    "\nSteps:",
    "\r\nSteps:",
)

# PNG/WebP text keys preferred in this order (CivitAI / A1111 first).
_TEXT_KEY_PRIORITY = (
    "parameters",
    "Parameters",
    "prompt",
    "Prompt",
    "Comment",
    "comment",
    "UserComment",
    "Description",
    "description",
)


def list_batch_images(directory_path, pattern):
    if not directory_path or not os.path.isdir(directory_path):
        return []
    directory_path = os.path.abspath(directory_path)
    pattern = pattern or "*"
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
        f"LoboForge Load Folder Image Prompts: unknown order `{order}` "
        f"(use {ORDER_FOLDER!r} or {ORDER_RANDOM!r})"
    )


def select_image_id(order, seed, start_at, item_i, file_count):
    index = int(start_at) + int(item_i)
    if order == ORDER_RANDOM:
        random.seed(int(seed) + index)
        return int(random.random() * file_count) % file_count
    return index % file_count


def _decode_bytes(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, (bytes, bytearray)):
        return None
    raw = bytes(value)
    # EXIF UserComment often starts with charset code (8 bytes).
    if len(raw) >= 8:
        prefix = raw[:8]
        payload = raw[8:]
        if prefix in (b"ASCII\x00\x00\x00", b"UNICODE\x00", b"Unicode\x00"):
            if prefix.startswith(b"ASCII"):
                return payload.decode("ascii", errors="replace").rstrip("\x00")
            return payload.decode("utf-16", errors="replace").rstrip("\x00")
        if prefix == b"\x00\x00\x00\x00\x00\x00\x00\x00":
            return payload.decode("utf-8", errors="replace").rstrip("\x00")
    for encoding in ("utf-8", "utf-16-le", "latin-1"):
        try:
            return raw.decode(encoding).rstrip("\x00")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").rstrip("\x00")


def parse_a1111_positive(parameters_text):
    """Positive prompt from A1111/CivitAI `parameters` blob."""
    if not parameters_text or not isinstance(parameters_text, str):
        return ""
    text = parameters_text.replace("\r\n", "\n").replace("\r", "\n")
    cut = None
    for marker in _NEG_MARKERS:
        idx = text.find(marker.replace("\r\n", "\n"))
        if idx == -1:
            # also try case-insensitive Negative prompt mid-string
            continue
        cut = idx if cut is None else min(cut, idx)
    # Case-insensitive Negative prompt: anywhere as line start
    match = re.search(r"(?im)^Negative prompt:", text)
    if match is not None:
        cut = match.start() if cut is None else min(cut, match.start())
    if cut is None:
        for marker in ("\nSteps:",):
            idx = text.find(marker)
            if idx != -1:
                cut = idx if cut is None else min(cut, idx)
    if cut is None:
        # Sometimes Steps: is on the first line after a single-line prompt.
        match = re.search(r"(?m)^Steps:", text)
        if match is not None and match.start() > 0:
            cut = match.start()
    positive = text if cut is None else text[:cut]
    return positive.strip()


def _prompt_from_json_obj(obj, depth=0):
    """Best-effort positive text from JSON (CivitAI meta / ComfyUI prompt)."""
    if depth > 6 or obj is None:
        return ""
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return ""
        if s[0] in "{[":
            try:
                return _prompt_from_json_obj(json.loads(s), depth + 1)
            except (json.JSONDecodeError, TypeError):
                return s
        return s
    if isinstance(obj, dict):
        # CivitAI / NovelAI style
        for key in ("prompt", "Prompt", "positive", "Positive", "positive_prompt"):
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                return obj[key].strip()
        meta = obj.get("meta")
        if isinstance(meta, dict):
            got = _prompt_from_json_obj(meta, depth + 1)
            if got:
                return got
        # ComfyUI API prompt: node id → {class_type, inputs}
        candidates = []
        for node in obj.values():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            text = None
            for field in ("text", "string", "prompt", "positive"):
                val = inputs.get(field)
                if isinstance(val, str) and val.strip():
                    text = val.strip()
                    break
            if not text:
                continue
            title = ""
            meta = node.get("_meta")
            if isinstance(meta, dict):
                title = str(meta.get("title") or "").lower()
            is_neg = (
                "negative" in title
                or "neg" == title
                or "Negative" in class_type
                or class_type.endswith("Negative")
            )
            if is_neg:
                continue
            is_pos = "positive" in title or "pos" in title.split() or "CLIPTextEncode" in class_type
            if is_pos:
                candidates.insert(0, text)
            else:
                candidates.append(text)
        if candidates:
            return candidates[0]
        return ""
    return ""


def _looks_like_comfy_prompt_json(text):
    if not text or not isinstance(text, str):
        return False
    s = text.lstrip()
    if not s.startswith("{"):
        return False
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict) or not data:
        return False
    sample = next(iter(data.values()))
    return isinstance(sample, dict) and ("class_type" in sample or "inputs" in sample)


def extract_positive_from_text_blob(text):
    """Turn a raw metadata string into a positive prompt if possible."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    # A1111 / CivitAI parameters
    if "Negative prompt:" in text or re.search(r"(?m)^Steps:", text) or "\nSteps:" in text:
        return parse_a1111_positive(text)
    # JSON
    if text[0] in "{[":
        got = _prompt_from_json_obj(text)
        if got:
            # If ComfyUI graph JSON left us with nothing useful, got may still be empty.
            if _looks_like_comfy_prompt_json(text) and not got:
                return ""
            # Avoid returning entire huge workflow JSON as the "prompt".
            if got.lstrip().startswith("{") and _looks_like_comfy_prompt_json(got):
                return ""
            if len(got) > 20000 and got.lstrip().startswith("{"):
                return ""
            return got
        return ""
    return text


def _collect_text_candidates(img):
    """Yield candidate raw strings from PNG info + EXIF, priority first."""
    info = img.info or {}
    seen = set()

    def emit(raw, source):
        text = _decode_bytes(raw) if not isinstance(raw, str) else raw
        if not text or not str(text).strip():
            return
        key = (source, str(text)[:64])
        if key in seen:
            return
        seen.add(key)
        yield str(text), source

    for key in _TEXT_KEY_PRIORITY:
        if key in info:
            yield from emit(info[key], key)

    # Any other info keys that look prompt-related
    for key, value in info.items():
        kl = str(key).lower()
        if kl in {k.lower() for k in _TEXT_KEY_PRIORITY}:
            continue
        if kl in ("workflow", "exif"):
            # workflow handled below; raw exif bytes via getexif
            if kl == "workflow":
                yield from emit(value, "workflow")
            continue

    # EXIF UserComment / ImageDescription / XPComment
    try:
        exif = img.getexif()
    except Exception:
        exif = None
    if exif:
        wanted = {
            ExifTags.Base.UserComment.value if hasattr(ExifTags, "Base") else 37510: "UserComment",
            37510: "UserComment",
            270: "ImageDescription",  # ImageDescription
            40092: "XPComment",
        }
        for tag_id, name in wanted.items():
            if tag_id in exif:
                yield from emit(exif.get(tag_id), f"EXIF:{name}")
        # Walk IFD for UserComment if nested
        try:
            if hasattr(exif, "get_ifd"):
                for ifd_id in (0x8769, 0x8825):
                    try:
                        ifd = exif.get_ifd(ifd_id)
                    except Exception:
                        continue
                    if not ifd:
                        continue
                    for tag_id, name in ((37510, "UserComment"), (270, "ImageDescription")):
                        if tag_id in ifd:
                            yield from emit(ifd.get(tag_id), f"EXIF:{name}")
        except Exception:
            pass


def extract_prompt_from_image_path(path):
    """
    Best-effort positive generation prompt from image metadata.
    Prefer A1111/CivitAI `parameters`, then plain prompt text / JSON prompt fields,
    then simple ComfyUI text widgets. Returns "" if nothing recoverable.
    """
    try:
        with Image.open(path) as img:
            candidates = list(_collect_text_candidates(img))
    except Exception:
        return ""

    # Prefer parameters-like blobs first (already ordered), then parse.
    for text, source in candidates:
        source_l = source.lower()
        if source_l == "parameters" or source_l.endswith("parameters"):
            got = parse_a1111_positive(text)
            if got:
                return got
            # Some CivitAI exports store only positive + Steps on one blob
            got = extract_positive_from_text_blob(text)
            if got:
                return got
        elif source_l in ("prompt",) or source.endswith("prompt"):
            # Plain prompt string OR ComfyUI JSON
            if _looks_like_comfy_prompt_json(text):
                got = _prompt_from_json_obj(text)
                if got:
                    return got
                continue
            got = extract_positive_from_text_blob(text)
            if got:
                return got
        elif "usercomment" in source_l or source_l in ("comment", "description") or "xpcomment" in source_l:
            got = extract_positive_from_text_blob(text)
            if got:
                return got
        elif source_l == "workflow":
            # Workflow JSON rarely has the prompt as a top-level string; skip unless easy.
            got = _prompt_from_json_obj(text)
            if got and not got.lstrip().startswith("{"):
                return got

    # Final pass: any remaining candidate
    for text, _source in candidates:
        got = extract_positive_from_text_blob(text)
        if got:
            return got
    return ""


def write_prompt_sidecar(image_path, prompt_text):
    base, _ext = os.path.splitext(image_path)
    sidecar = base + ".txt"
    with open(sidecar, "w", encoding="utf-8") as handle:
        handle.write(prompt_text)
    return sidecar


class LoboForgeLoadFolderImagePrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Folder of AI images (CivitAI / A1111 / ComfyUI). Use choose folder.",
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
                    "tooltip": "0 = every matching image. N = load that many. One Queue fans out all of them.",
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
                "skip_missing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, skip images with no recoverable prompt. If false, emit empty strings for those files.",
                }),
                "write_sidecar_txt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, write prompt next to each image as basename.txt (off by default).",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "filename", "file_count", "missing_count")
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "load_folder_image_prompts"
    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Extract original generation prompts from image metadata (CivitAI / A1111 "
        "`parameters`, EXIF UserComment, simple ComfyUI text). "
        "Not vision captioning. how_many=0 processes every matching image; "
        "prompt and filename fan out (OUTPUT_IS_LIST)."
    )

    def load_folder_image_prompts(
        self,
        folder,
        pattern,
        how_many,
        order,
        seed,
        start_at,
        skip_missing=True,
        write_sidecar_txt=False,
    ):
        pattern = pattern or "*"
        if not folder or not os.path.isdir(folder):
            raise RuntimeError(
                f"LoboForge Load Folder Image Prompts: folder not found: {folder}"
            )

        image_paths = list_batch_images(folder, pattern)
        if not image_paths:
            raise RuntimeError(
                f"LoboForge Load Folder Image Prompts: no images in `{folder}` "
                f"matching `{pattern}`"
            )

        file_count = len(image_paths)
        count = resolve_how_many(how_many, file_count)
        if count < 1:
            raise RuntimeError(
                "LoboForge Load Folder Image Prompts: how_many resolved to 0"
            )

        order = normalize_order(order)
        prompts = []
        filenames = []
        missing_count = 0

        for item_i in range(count):
            image_id = select_image_id(order, seed, start_at, item_i, file_count)
            image_path = image_paths[image_id]
            name = os.path.basename(image_path)
            prompt_text = extract_prompt_from_image_path(image_path)

            if not prompt_text:
                missing_count += 1
                if skip_missing:
                    continue
                prompts.append("")
                filenames.append(name)
                continue

            prompts.append(prompt_text)
            filenames.append(name)
            if write_sidecar_txt:
                try:
                    write_prompt_sidecar(image_path, prompt_text)
                except OSError as exc:
                    print(
                        f"[LoboForge Load Folder Image Prompts] sidecar write failed "
                        f"for {name}: {exc}"
                    )

        if skip_missing and not prompts:
            raise RuntimeError(
                "LoboForge Load Folder Image Prompts: no recoverable prompts in "
                f"`{folder}` (checked {count} file picks; missing={missing_count})"
            )

        print(
            f"[LoboForge Load Folder Image Prompts] how_many={len(prompts)} "
            f"(requested={how_many}) start_at={start_at} order={order} "
            f"file_count={file_count} missing={missing_count} "
            f"skip_missing={skip_missing} → {filenames}"
        )
        return (prompts, filenames, file_count, missing_count)

    @classmethod
    def IS_CHANGED(
        cls,
        folder,
        pattern,
        how_many,
        order,
        seed,
        start_at,
        skip_missing=True,
        write_sidecar_txt=False,
    ):
        try:
            order = normalize_order(order)
        except RuntimeError:
            return float("NaN")
        if order == ORDER_RANDOM:
            return float("NaN")
        pattern = pattern or "*"
        if not folder or not os.path.isdir(folder):
            return float("NaN")
        image_paths = list_batch_images(folder, pattern)
        if not image_paths:
            return float("NaN")
        file_count = len(image_paths)
        count = resolve_how_many(how_many, file_count)
        digests = [f"skip={bool(skip_missing)}", f"side={bool(write_sidecar_txt)}"]
        for item_i in range(count):
            image_id = select_image_id(ORDER_FOLDER, seed, start_at, item_i, file_count)
            selected = image_paths[image_id]
            if not os.path.isfile(selected):
                return float("NaN")
            try:
                digests.append(file_sha256(selected))
            except OSError:
                return float("NaN")
        return ",".join(digests)


def _register_folder_route():
    if getattr(_register_folder_route, "done", False):
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        print(
            f"LoboForge Load Folder Image Prompts folder route was not registered ({exc})."
        )
        return
    if getattr(PromptServer, "instance", None) is None:
        return

    @PromptServer.instance.routes.post("/loboforge/load_folder_image_prompts/pick_folder")
    async def loboforge_pick_image_prompts_folder(request):
        try:
            result = await asyncio.to_thread(
                pick_path, "LoboForge Load Folder Image Prompts", True
            )
        except Exception as exc:
            result = {"error": f"Folder dialog failed: {exc}"}
        return web.json_response(result)

    _register_folder_route.done = True


_register_folder_route()

NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgeLoadFolderImagePrompts,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge Load Folder Image Prompts (from metadata)",
}
