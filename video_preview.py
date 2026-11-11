import io
import os
import uuid
from fractions import Fraction

import folder_paths
from comfy_api.latest import InputImpl, Types
from comfy_api.latest._io import FolderType

NODE_CLASS = "LoboForgeVideoPreview"
NODE_CLASS_FRAMES = "LoboForgePreviewImageAudio"


def _folder_info_for_path(path):
    path = os.path.abspath(path)
    for folder_type, getter in (
        (FolderType.output, folder_paths.get_output_directory),
        (FolderType.temp, folder_paths.get_temp_directory),
        (FolderType.input, folder_paths.get_input_directory),
    ):
        base = os.path.abspath(getter())
        if path == base or path.startswith(base + os.sep):
            rel = os.path.relpath(os.path.dirname(path), base).replace("\\", "/")
            if rel == ".":
                rel = ""
            return folder_type, rel
    return FolderType.temp, ""


def _temp_preview_path():
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"loboforge_preview_{uuid.uuid4().hex}.mp4")


def _save_video_to_temp(video):
    path = _temp_preview_path()
    video.save_to(
        path,
        format=Types.VideoContainer.MP4,
        codec=Types.VideoCodec.H264,
        preset="ultrafast",
    )
    return path


def _preview_ui_for_path(video_path):
    # Legacy ui.video format for LoboForge.web/previewVideo.js only.
    # Do NOT return PreviewVideo (images+animated): that also triggers ComfyUI's
    # native media preview and doubles the player on the node.
    folder_type, subfolder = _folder_info_for_path(video_path)
    type_name = folder_type.value if hasattr(folder_type, "value") else str(folder_type)
    return {
        "video": [os.path.basename(video_path), type_name],
        "subfolder": subfolder or "",
    }


class LoboForgeVideoPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    CATEGORY = "LoboForge"
    DESCRIPTION = "Preview a VIDEO input with a short carousel of recent results."
    # Match core SaveVideo/PreviewImage: output sockets are required so validation
    # can index RETURN_TYPES when this node is used as an upstream link.
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True
    FUNCTION = "load_video"

    @classmethod
    def _resolve_video_path(cls, video):
        if isinstance(video, (str, os.PathLike)):
            return os.path.abspath(str(video))

        if hasattr(video, "get_stream_source"):
            source = video.get_stream_source()
            if isinstance(source, (str, os.PathLike)):
                return os.path.abspath(str(source))
            if isinstance(source, io.BytesIO):
                return _save_video_to_temp(video)

        if hasattr(video, "save_to"):
            return _save_video_to_temp(video)

        raise TypeError(
            f"LoboForge Video Preview expected a path or ComfyUI VIDEO input, got {type(video)!r}"
        )

    def load_video(self, video):
        video_path = self._resolve_video_path(video)
        return {"ui": _preview_ui_for_path(video_path), "result": (video,)}


class LoboForgePreviewImageAudio:
    """Encode IMAGE(+AUDIO) to a temp mp4 and show a real video player preview.

    Writes only under ComfyUI's temp folder for /view — nothing is saved to output/.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    CATEGORY = "LoboForge"
    DESCRIPTION = (
        "Preview VAE Decode frames as a real video (+ optional VAE Decode Audio). "
        "Does not write to the output folder — only a temp preview for the UI player."
    )
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True
    FUNCTION = "preview"

    def preview(self, images, fps, audio=None):
        if images is None or getattr(images, "shape", [0])[0] == 0:
            raise RuntimeError("LoboForge Preview Image+Audio: no images")

        components = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images,
                audio=audio,
                frame_rate=Fraction(fps),
            )
        )
        # Encode once to temp; return VideoFromFile so downstream path-based
        # previews (e.g. Hallo PreViewVideo) can resolve without re-encoding.
        video_path = _save_video_to_temp(components)
        return {
            "ui": _preview_ui_for_path(video_path),
            "result": (InputImpl.VideoFromFile(video_path),),
        }


NODE_CLASS_MAPPINGS = {
    NODE_CLASS: LoboForgeVideoPreview,
    NODE_CLASS_FRAMES: LoboForgePreviewImageAudio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_CLASS: "LoboForge Video Preview",
    NODE_CLASS_FRAMES: "LoboForge Preview Video (Image + Audio)",
}
