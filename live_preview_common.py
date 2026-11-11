"""Shared live-preview helpers for LoboForge nodes (stock PreviewImage gallery)."""

import json
import os
import random

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from comfy_execution.utils import get_executing_context


def client_id():
    try:
        from server import PromptServer
    except Exception:
        return None
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return None
    return getattr(instance, "client_id", None)


def push_executed(unique_id, prompt_id, output, merge=False):
    """Push an intermediate executed so the stock gallery can grow mid list-map.

    List fan-out only emits a final executed after every item finishes. merge=True
    appends array fields (e.g. images) in the frontend; merge=False replaces.

    Uses the nested execution id (e.g. ``30:3``) for both node and display_node so
    leaf galleries and subgraph previewExposures keep working. The web extension
    ``liveImagePreview.js`` also mirrors tiles onto the parent SubgraphNode.
    """
    if unique_id is None:
        return
    try:
        from server import PromptServer
        instance = getattr(PromptServer, "instance", None)
        if instance is None:
            return
        # None sid = broadcast; needed if client_id is unset mid-prompt.
        sid = client_id()
        node_id = str(unique_id)
        instance.send_sync(
            "executed",
            {
                "node": node_id,
                "display_node": node_id,
                "output": output,
                "prompt_id": prompt_id,
                "merge": bool(merge),
            },
            sid,
        )
    except Exception as exc:
        print(f"[LoboForge live preview] intermediate executed push failed: {exc}")


def execution_ids(unique_id=None):
    ctx = get_executing_context()
    prompt_id = ctx.prompt_id if ctx is not None else None
    list_index = ctx.list_index if ctx is not None else None
    if unique_id is None and ctx is not None:
        unique_id = ctx.node_id
    return prompt_id, list_index, unique_id


def new_temp_prefix(tag="lobolive"):
    return f"_{tag}_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))


def save_image_tiles(images, output_dir, prefix_append, filename_text="", prompt=None, extra_pnginfo=None, compress_level=1):
    """Write IMAGE batch to temp like PreviewImage; return ui image entries."""
    filename_prefix = "loboforge_live" + prefix_append
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
    )
    results = []
    label = filename_text if isinstance(filename_text, str) else ""
    for batch_number, image in enumerate(images):
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value))
            if label:
                metadata.add_text("loboforge_filename", label)

        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.png"
        img.save(
            os.path.join(full_output_folder, file),
            pnginfo=metadata,
            compress_level=compress_level,
        )
        entry = {
            "filename": file,
            "subfolder": subfolder,
            "type": "temp",
        }
        if label:
            entry["label"] = label
        results.append(entry)
        counter += 1
    return results


def push_live_tiles(unique_id, prompt_id, list_index, results):
    """Grow the stock PreviewImage gallery as each list-map item finishes.

    Returns ui.images for this item only. execution.py concatenates those across
    the map for the final executed; intermediate pushes use merge so the gallery
    updates before the map completes.
    """
    # First (or only) item replaces; later items append via frontend merge.
    merge = list_index is not None and list_index > 0
    push_executed(unique_id, prompt_id, {"images": results}, merge=merge)
    return {"images": results}
