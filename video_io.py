import os
from fractions import Fraction

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import InputImpl, Types
from comfy_api.latest._io import FolderType
from comfy_api.latest._ui import PreviewVideo, SavedResult


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resolve_under_output(path_or_rel):
    path = (path_or_rel or "").strip()
    if not path:
        raise RuntimeError("Empty path")
    if os.path.isabs(path):
        return path
    if path.startswith("output/") or path.startswith("output\\"):
        return os.path.join(folder_paths.get_output_directory(), path[7:])
    if path.startswith("input/") or path.startswith("input\\"):
        return os.path.join(folder_paths.get_input_directory(), path[6:])
    return os.path.join(folder_paths.get_output_directory(), path)


def _write_text(path, text):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text is not None else "")


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _tensor_to_pil(image):
    array = image.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA")
    return Image.fromarray(array[..., :3], mode="RGB")


def _save_image(image, path):
    _ensure_parent(path)
    _tensor_to_pil(image).save(path)


def _images_to_video(images, fps, audio=None):
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=images,
            audio=audio,
            frame_rate=Fraction(fps),
        )
    )


class LoboForgeSaveClipAndLastFrame:
    """Save clip + last frame under a segment folder and optional path sidecar txts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "run_name": ("STRING", {"default": "VideoDirector"}),
                "segment_id": ("STRING", {"default": "01"}),
                "filename_prefix": ("STRING", {"default": "clip"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "path_txt_dir": ("STRING", {"default": "output/VideoDirector/"}),
                "video_path_txt_name": ("STRING", {"default": ""}),
                "image_path_txt_name": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Encode images(+audio) to mp4, write last_frame.png beside it, "
        "optionally write path sidecar txt files for chaining."
    )
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "last_frame", "video_path", "last_frame_path", "segment_dir")
    OUTPUT_NODE = True
    FUNCTION = "save"

    def save(
        self,
        images,
        fps,
        run_name,
        segment_id,
        filename_prefix,
        audio=None,
        path_txt_dir="output/VideoDirector/",
        video_path_txt_name="",
        image_path_txt_name="",
    ):
        if images is None or images.shape[0] == 0:
            raise RuntimeError("LoboForge Save Clip And Last Frame: no images")

        out_root = folder_paths.get_output_directory()
        segment_dir = os.path.join(
            out_root,
            "VideoDirector",
            (run_name or "run").strip() or "run",
            "segments",
            (segment_id or "01").strip() or "01",
        )
        os.makedirs(segment_dir, exist_ok=True)

        prefix = (filename_prefix or "clip").strip() or "clip"
        video_path = os.path.join(segment_dir, f"{prefix}.mp4")
        last_frame_path = os.path.join(segment_dir, "last_frame.png")

        video = _images_to_video(images, float(fps), audio=audio)
        video.save_to(
            video_path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
        )

        last_frame = images[-1:].contiguous().clone()
        _save_image(last_frame, last_frame_path)

        txt_dir = _resolve_under_output(path_txt_dir) if path_txt_dir else None
        if txt_dir and video_path_txt_name:
            name = video_path_txt_name if video_path_txt_name.endswith(".txt") else f"{video_path_txt_name}.txt"
            _write_text(os.path.join(txt_dir, name), video_path)
        if txt_dir and image_path_txt_name:
            name = image_path_txt_name if image_path_txt_name.endswith(".txt") else f"{image_path_txt_name}.txt"
            _write_text(os.path.join(txt_dir, name), last_frame_path)

        # Frontend promoted/subgraph previews read ui.images + animated (PreviewVideo contract).
        subfolder = os.path.relpath(segment_dir, out_root).replace("\\", "/")
        video_ui = PreviewVideo([
            SavedResult(os.path.basename(video_path), subfolder, FolderType.output),
        ])
        return {
            "ui": video_ui.as_dict(),
            "result": (video, last_frame, video_path, last_frame_path, segment_dir),
        }


class LoboForgeLoadSegmentLastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_name": ("STRING", {"default": "VideoDirector"}),
                "segment_id": ("STRING", {"default": "01"}),
            },
            "optional": {
                "path_override": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "path")
    FUNCTION = "load"

    def load(self, run_name, segment_id, path_override=""):
        if path_override and path_override.strip():
            path = _resolve_under_output(path_override.strip())
        else:
            path = os.path.join(
                folder_paths.get_output_directory(),
                "VideoDirector",
                (run_name or "run").strip() or "run",
                "segments",
                (segment_id or "01").strip() or "01",
                "last_frame.png",
            )
        if not os.path.isfile(path):
            raise RuntimeError(f"LoboForge Load Segment Last Frame: missing {path}")
        image = Image.open(path)
        image = image.convert("RGB")
        array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return (tensor, path)


class LoboForgeSplitLastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "split_index": ("INT", {"default": -1, "min": -0x7fffffff, "max": 0x7fffffff}),
            }
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("IMAGE", "INT", "IMAGE", "INT")
    RETURN_NAMES = ("images_a", "a_count", "images_b", "b_count")
    FUNCTION = "split"

    def split(self, images, split_index):
        count = int(images.shape[0])
        index = int(split_index)
        if index < 0:
            index = count + index
        index = max(0, min(count, index))
        a = images[:index]
        b = images[index:]
        if a.shape[0] == 0:
            a = images[:1]
        if b.shape[0] == 0:
            b = images[-1:]
        return (a, int(a.shape[0]), b, int(b.shape[0]))


class LoboForgeLoadVideoPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.01}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "path")
    FUNCTION = "load"

    def load(self, path, force_rate=0.0, frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        path = _resolve_under_output((path or "").strip())
        if not os.path.isfile(path):
            raise RuntimeError(f"LoboForge Load Video Path: missing {path}")

        video = InputImpl.VideoFromFile(path)
        components = video.get_components()
        images = components.images
        audio = components.audio
        fps = float(components.frame_rate)

        skip = max(0, int(skip_first_frames))
        step = max(1, int(select_every_nth))
        if skip or step != 1:
            images = images[skip::step]
        cap = int(frame_load_cap)
        if cap > 0:
            images = images[:cap]

        force = float(force_rate)
        if force > 0:
            fps = force

        if images is None or images.shape[0] == 0:
            raise RuntimeError(f"LoboForge Load Video Path: no frames in {path}")

        return (images, audio, fps, path)


class LoboForgeImageBatchJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "join"

    def join(self, image_1, image_2=None, image_3=None, image_4=None, image_5=None, image_6=None, image_7=None, image_8=None):
        batches = [image_1]
        for img in (image_2, image_3, image_4, image_5, image_6, image_7, image_8):
            if img is not None:
                batches.append(img)
        return (torch.cat(batches, dim=0),)


class LoboForgeAudioConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "direction": (["right", "left"], {"default": "right"}),
            }
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concat"

    def concat(self, audio1, audio2, direction="right"):
        import comfy.audio

        if audio1 is None:
            return (audio2,)
        if audio2 is None:
            return (audio1,)

        waveform_1 = audio1["waveform"]
        waveform_2 = audio2["waveform"]
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]

        if waveform_1.shape[1] == 1:
            waveform_1 = waveform_1.repeat(1, 2, 1)
        if waveform_2.shape[1] == 1:
            waveform_2 = waveform_2.repeat(1, 2, 1)

        if sample_rate_1 != sample_rate_2:
            if sample_rate_1 > sample_rate_2:
                waveform_2 = comfy.audio.resample(waveform_2, sample_rate_2, sample_rate_1)
                out_rate = sample_rate_1
            else:
                waveform_1 = comfy.audio.resample(waveform_1, sample_rate_1, sample_rate_2)
                out_rate = sample_rate_2
        else:
            out_rate = sample_rate_1

        if direction == "left":
            waveform = torch.cat((waveform_2, waveform_1), dim=2)
        else:
            waveform = torch.cat((waveform_1, waveform_2), dim=2)
        return ({"waveform": waveform, "sample_rate": out_rate},)


class LoboForgeLoadImagePath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "path")
    FUNCTION = "load"

    def load(self, path):
        path = _resolve_under_output((path or "").strip())
        if not os.path.isfile(path):
            raise RuntimeError(f"LoboForge Load Image Path: missing {path}")
        image = Image.open(path)
        image = image.convert("RGB")
        array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return (tensor, path)


class LoboForgeLoadTextFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"

    def load(self, file_path):
        path = _resolve_under_output((file_path or "").strip())
        if not os.path.isfile(path):
            raise RuntimeError(f"LoboForge Load Text File: missing {path}")
        return (_read_text(path).strip(),)


class LoboForgeSaveText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "output_file_path": ("STRING", {"default": "output/VideoDirector/"}),
                "file_name": ("STRING", {"default": "note"}),
                "file_extension": (["txt", "json", "csv", "yaml", "yml"], {"default": "txt"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "save"

    def save(self, text, output_file_path, file_name, file_extension="txt", overwrite=True):
        directory = _resolve_under_output(output_file_path)
        os.makedirs(directory, exist_ok=True)
        name = (file_name or "note").strip() or "note"
        if name.endswith(f".{file_extension}"):
            filename = name
        else:
            filename = f"{name}.{file_extension}"
        path = os.path.join(directory, filename)
        if (not overwrite) and os.path.exists(path):
            return (text,)
        _write_text(path, text if text is not None else "")
        return (text,)


def video_filename_prefix(filename_text, folder="video/H3_batch"):
    """Build a SaveVideo-safe prefix: folder/stem (extension stripped, path sanitized)."""
    folder = (folder or "video/H3_batch").strip().replace("\\", "/")
    folder = folder.strip("/") or "video/H3_batch"
    raw = os.path.basename(str(filename_text or "").strip()) or "video"
    stem = os.path.splitext(raw)[0].strip() or "video"
    stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    return f"{folder}/{stem}"


class LoboForgeVideoFilenamePrefix:
    """Turn Load Folder Images filename_text into a SaveVideo filename_prefix."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "From LoboForge Load Folder Images (fan-out list item).",
                }),
                "folder": ("STRING", {
                    "default": "video/H3_batch",
                    "tooltip": "Subfolder under ComfyUI output/. One mp4 per source image stem.",
                }),
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "build"
    DESCRIPTION = (
        "Build output/<folder>/<stem>_#####_.mp4 prefixes from batch filenames. "
        "Wire to SaveVideo filename_prefix so each fan-out item gets its own file."
    )

    def build(self, filename_text, folder="video/H3_batch"):
        return (video_filename_prefix(filename_text, folder),)


class LoboForgeSaveVideoNamed:
    """Save VIDEO using the source image stem — one file per OUTPUT_IS_LIST fan-out item."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "From LoboForge Load Folder Images filename_text (same fan-out index).",
                }),
                "folder": ("STRING", {
                    "default": "video/H3_batch",
                    "tooltip": "Subfolder under ComfyUI output/.",
                }),
                "format": (["auto", "mp4", "mkv", "webm", "mov"], {"default": "mp4"}),
                "codec": (["auto", "h264", "av1", "prores"], {"default": "auto"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "LoboForge"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    DESCRIPTION = (
        "Save each fan-out VIDEO under output/<folder>/<stem>_#####_.mp4. "
        "Prefer this over a shared SaveVideo prefix when batching folder images."
    )

    def save(
        self,
        video,
        filename_text,
        folder="video/H3_batch",
        format="mp4",
        codec="auto",
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix = video_filename_prefix(filename_text, folder)
        format_name = format or "mp4"
        codec_name = codec or "auto"
        if format_name == "auto":
            format_name = {"av1": "webm", "prores": "mov"}.get(codec_name, "mp4")
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        metadata = None
        if extra_pnginfo or prompt is not None:
            metadata = {}
            if extra_pnginfo:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
        ext = Types.VideoContainer.get_extension(format_name)
        file = f"{filename}_{counter:05}_.{ext}"
        path = os.path.join(full_output_folder, file)
        video.save_to(
            path,
            format=Types.VideoContainer(format_name),
            codec=Types.VideoCodec(codec_name),
            metadata=metadata,
        )
        return {
            "ui": PreviewVideo([SavedResult(file, subfolder, FolderType.output)]).as_dict(),
            "result": (video,),
        }


NODE_CLASS_MAPPINGS = {
    "LoboForgeSaveClipAndLastFrame": LoboForgeSaveClipAndLastFrame,
    "LoboForgeLoadSegmentLastFrame": LoboForgeLoadSegmentLastFrame,
    "LoboForgeSplitLastFrame": LoboForgeSplitLastFrame,
    "LoboForgeLoadVideoPath": LoboForgeLoadVideoPath,
    "LoboForgeImageBatchJoin": LoboForgeImageBatchJoin,
    "LoboForgeAudioConcatenate": LoboForgeAudioConcatenate,
    "LoboForgeLoadImagePath": LoboForgeLoadImagePath,
    "LoboForgeLoadTextFile": LoboForgeLoadTextFile,
    "LoboForgeSaveText": LoboForgeSaveText,
    "LoboForgeVideoFilenamePrefix": LoboForgeVideoFilenamePrefix,
    "LoboForgeSaveVideoNamed": LoboForgeSaveVideoNamed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoboForgeSaveClipAndLastFrame": "LoboForge Save Clip And Last Frame",
    "LoboForgeLoadSegmentLastFrame": "LoboForge Load Segment Last Frame",
    "LoboForgeSplitLastFrame": "LoboForge Split Last Frame",
    "LoboForgeLoadVideoPath": "LoboForge Load Video Path",
    "LoboForgeImageBatchJoin": "LoboForge Image Batch Join",
    "LoboForgeAudioConcatenate": "LoboForge Audio Concatenate",
    "LoboForgeLoadImagePath": "LoboForge Load Image Path",
    "LoboForgeLoadTextFile": "LoboForge Load Text File",
    "LoboForgeSaveText": "LoboForge Save Text",
    "LoboForgeVideoFilenamePrefix": "LoboForge Video Filename Prefix",
    "LoboForgeSaveVideoNamed": "LoboForge Save Video (Named)",
}
