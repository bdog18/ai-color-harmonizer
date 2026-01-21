from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image, ImageOps
from streamlit.runtime.uploaded_file_manager import UploadedFile

def load_image_rgb(
        src: Union[str, Path, UploadedFile],
        *,
        background: str = "white"
) -> np.ndarray:
    """
    Load an image into an RGB uint8 numpy array (H, W, 3), honoring EXIF orientation.
    Handles:
      - PNG/TIFF with alpha: composited over `background`
      - CMYK to RGB conversion
      - EXIF orientation via ImageOps.exif_transpose
    `src` can be a file path or a file-like object (for web/Streamlit).
    """
    im = Image.open(src)

    im = ImageOps.exif_transpose(im)

    if im.mode in ("RGBA", "LA"):
        bg_color = (255, 255, 255) if background == "white" else (0, 0, 0)
        rgb = Image.new("RGB", im.size, bg_color)
        rgb.paste(im, mask=im.split()[-1]) # alpha channel mask
        im = rgb
    else:
        im = im.convert("RGB")

    return np.asarray(im, dtype=np.uint8) # (H, W, 3) uint8

def save_image_rgb(arr: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save an RGB uint8 numpy array to disk (PNG). Creates parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


# Helpers

def is_valid_image_path(p: Union[str, Path]) -> bool:
    p = Path(p)
    return p.exists() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def ensure_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Coerce an arbitrary array into uint8 RGB if possible.
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError("Expected RGB with shape (H, W, 3)")
    return arr