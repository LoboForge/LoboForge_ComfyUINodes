import numpy as np
import torch
from PIL import ImageOps

import node_helpers


def pil_to_image_tensor(image, allow_rgba=False):
    image = node_helpers.pillow(ImageOps.exif_transpose, image)
    if not allow_rgba or "A" not in image.getbands():
        image = image.convert("RGB")
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)
