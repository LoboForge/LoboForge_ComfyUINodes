import asyncio
import io
import os
import sqlite3

from PIL import Image

import folder_paths
import node_helpers

from .dialog import pick_path
from .images import pil_to_image_tensor

NODE_CLASS = "LoboForgePromptRepository"
EMPTY_CHOICE = "(no prompts)"
PATH_COLUMNS = ("sample_image_path", "image_path", "sample_path")
_listed_db_path = ""


def connect_readonly(db_path):
    return sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)


def quote_ident(name):
    return '"' + str(name).replace('"', '""') + '"'


def table_columns(connection, table):
    rows = connection.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [row[1] for row in rows]


def prompt_labels(db_path):
    if not db_path or not os.path.isfile(db_path):
        return []
    connection = connect_readonly(db_path)
    try:
        columns = table_columns(connection, "prompts")
        if "id" not in columns:
            return []
        if "subject" in columns:
            rows = connection.execute("SELECT id, subject FROM prompts ORDER BY id").fetchall()
            labels = []
            for prompt_id, subject in rows:
                subject = (subject or "").strip()
                labels.append(f"{prompt_id} | {subject}" if subject else str(prompt_id))
            return labels
        rows = connection.execute("SELECT id FROM prompts ORDER BY id").fetchall()
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def choices_for_input():
    try:
        labels = prompt_labels(_listed_db_path)
    except sqlite3.Error:
        labels = []
    return labels or [EMPTY_CHOICE]


def parse_prompt_id(choice):
    text = "" if choice is None else str(choice).strip()
    if not text or text == EMPTY_CHOICE or text.startswith("("):
        return None
    head = text.split("|", 1)[0].strip()
    if head.isdigit():
        return int(head)
    return None


def mime_extension(mime, image_format):
    mime = (mime or "").lower()
    if "png" in mime or image_format == "PNG":
        return "png"
    if "webp" in mime or image_format == "WEBP":
        return "webp"
    if "gif" in mime or image_format == "GIF":
        return "gif"
    return "jpg"


def tensor_from_bytes(blob):
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if not blob:
        raise RuntimeError("LoboForge Prompt Repository: sample image is empty")
    try:
        image = node_helpers.pillow(Image.open, io.BytesIO(blob))
    except Exception as exc:
        raise RuntimeError(f"LoboForge Prompt Repository: sample image could not be decoded ({exc})") from exc
    return pil_to_image_tensor(image), image.format


def write_sample_blob(prompt_id, blob, mime, image_format):
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    extension = mime_extension(mime, image_format)
    path = os.path.join(temp_dir, f"loboforge_prompt_{prompt_id}.{extension}")
    payload = blob.tobytes() if isinstance(blob, memoryview) else blob
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def tensor_from_path(path):
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"LoboForge Prompt Repository: sample image not found: {path}")
    try:
        image = node_helpers.pillow(Image.open, path)
    except Exception as exc:
        raise RuntimeError(f"LoboForge Prompt Repository: sample image could not be decoded ({exc})") from exc
    return pil_to_image_tensor(image)


class LoboForgePromptRepository:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "db_path": ("STRING", {"default": "", "multiline": False}),
                "prompt": (choices_for_input(),),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("positive", "negative", "sample_image", "sample_image_path")
    FUNCTION = "load_prompt"
    CATEGORY = "LoboForge"
    DESCRIPTION = "Load one saved prompt from a local SQLite database. The menu lists id and subject only."

    def load_prompt(self, db_path, prompt):
        if not db_path or not os.path.isfile(db_path):
            raise RuntimeError(f"LoboForge Prompt Repository: database not found: {db_path}")
        prompt_id = parse_prompt_id(prompt)
        if prompt_id is None:
            raise RuntimeError("LoboForge Prompt Repository: no prompt selected")

        connection = connect_readonly(db_path)
        try:
            columns = table_columns(connection, "prompts")
            if not columns:
                raise RuntimeError("LoboForge Prompt Repository: prompts table not found")
            if "prompt_text" not in columns:
                raise RuntimeError("LoboForge Prompt Repository: prompt_text column not found")

            selected = ["prompt_text"]
            if "negative_prompt" in columns:
                selected.append("negative_prompt")
            path_column = next((name for name in PATH_COLUMNS if name in columns), None)
            if path_column:
                selected.append(path_column)
            if "sample_image" in columns:
                selected.append("sample_image")
            if "sample_image_mime" in columns:
                selected.append("sample_image_mime")

            query = "SELECT " + ", ".join(quote_ident(name) for name in selected) + " FROM prompts WHERE id = ?"
            row = connection.execute(query, (prompt_id,)).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError(f"LoboForge Prompt Repository: database read failed ({exc})") from exc
        finally:
            connection.close()

        if row is None:
            raise RuntimeError(f"LoboForge Prompt Repository: prompt id {prompt_id} not found")

        record = dict(zip(selected, row))
        positive = record.get("prompt_text") or ""
        negative = record.get("negative_prompt") or ""
        stored_path = (record.get(path_column) or "").strip() if path_column else ""
        blob = record.get("sample_image")

        if stored_path:
            tensor = tensor_from_path(stored_path)
            return (positive, negative, tensor, stored_path)
        if not blob:
            raise RuntimeError(f"LoboForge Prompt Repository: prompt id {prompt_id} has no sample image")
        tensor, image_format = tensor_from_bytes(blob)
        written = write_sample_blob(prompt_id, blob, record.get("sample_image_mime"), image_format)
        return (positive, negative, tensor, written)


def remember_db_path(path):
    global _listed_db_path
    _listed_db_path = path or ""


def _register_prompt_routes():
    if getattr(_register_prompt_routes, "done", False):
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        print(f"LoboForge Prompt Repository routes were not registered ({exc}).")
        return
    if getattr(PromptServer, "instance", None) is None:
        return

    @PromptServer.instance.routes.post("/loboforge/prompt_repository/pick_db")
    async def loboforge_pick_prompt_db(request):
        try:
            result = await asyncio.to_thread(pick_path, "LoboForge Prompt Repository", False)
        except Exception as exc:
            result = {"error": f"File dialog failed: {exc}"}
        return web.json_response(result)

    @PromptServer.instance.routes.post("/loboforge/prompt_repository/list")
    async def loboforge_list_prompts(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        db_path = (payload or {}).get("db_path") or ""
        remember_db_path(db_path)
        try:
            labels = await asyncio.to_thread(prompt_labels, db_path)
        except Exception as exc:
            return web.json_response({"error": f"Could not read prompt ids ({exc})", "choices": []})
        return web.json_response({"choices": labels})

    _register_prompt_routes.done = True


_register_prompt_routes()

NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgePromptRepository,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge Prompt Repository",
}
