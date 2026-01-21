from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from skimage.segmentation import slic
from skimage.util import img_as_float
import numpy as np
import cv2


@dataclass
class ImageFeatures:
    """
    Container for preprocessed data.
    """
    rgb_small: np.ndarray          # (Hs, Ws, 3) uint8
    hsv_small: np.ndarray          # (Hs, Ws, 3) float32, H in [0, 360), S,V in [0,1]
    lab_small: np.ndarray          # (Hs, Ws, 3) float32, CIELab using D65/2°
    sample_idx: np.ndarray         # (N,) flat indices into Hs*Ws
    samples_hsv: np.ndarray        # (N, 3) float32
    samples_lab: np.ndarray        # (N, 3) float32
    shape_small: Tuple[int, int]   # (Hs, Ws)


# ---------------- Resizing ----------------

def resize_longest(rgb: np.ndarray, max_long_edge: int = 640) -> np.ndarray:
    """
    Downscale keeping aspect ratio so max(H, W) == max_long_edge (no upscaling).
    Use INTER_AREA for downscale quality.
    """
    h, w = rgb.shape[:2]
    if max(h, w) <= max_long_edge:
        return rgb
    if h >= w:
        new_h = max_long_edge
        new_w = int(w * (max_long_edge / h))
    else:
        new_w = max_long_edge
        new_h = int(h * (max_long_edge / w))
    # OpenCV expects BGR, but resizing works for raw arrays
    out = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out


# ---------------- Color conversions ----------------
# OpenCV color spaces:
# - cv2.cvtColor expects RGB->HSV returns H in [0,180], S,V in [0,255] (uint8 or float ranges).
# - We'll standardize to H in degrees [0,360), S,V in [0,1] as float32.
# - For Lab, cv2 uses CIE L*a*b* with D65/2°. We'll output float32 in the usual ranges.

def rgb_to_hsv_norm(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.uint8)
    hsv_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H:[0,180], S:[0,255], V:[0,255]
    hsv = hsv_cv.astype(np.float32)
    hsv[..., 0] = hsv[..., 0] * 2.0                # to degrees [0,360)
    hsv[..., 1] = hsv[..., 1] / 255.0              # [0,1]
    hsv[..., 2] = hsv[..., 2] / 255.0              # [0,1]
    return hsv


def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # cv2 Lab: L in [0,100], a in [-128,127], b in [-128,127] (already perceptual)
    return lab


# ---------------- Sampling strategies ----------------

def sample_pixels(
    hsv: np.ndarray,
    lab: np.ndarray,
    n_samples: int = 30000,
    method: Literal["uniform", "grid", "superpixels"] = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return flat indices, HSV samples, Lab samples.
    - uniform: random unique pixels
    - grid: approximately uniform spatial coverage
    - superpixels: take centroids/medoids per SLIC region (requires scikit-image)
    """
    H, W = hsv.shape[:2]
    N = H * W
    rng = rng or np.random.default_rng(42)

    if method == "uniform":
        n = min(n_samples, N)
        idx = rng.choice(N, size=n, replace=False)

    elif method == "grid":
        # Pick a grid step so ~n_samples points are selected
        step = max(1, int(np.sqrt((H * W) / max(1, n_samples))))
        yy = np.arange(0, H, step)
        xx = np.arange(0, W, step)
        YY, XX = np.meshgrid(yy, xx, indexing="ij")
        idx = (YY * W + XX).reshape(-1)
        if idx.size > n_samples:
            idx = rng.choice(idx, size=n_samples, replace=False)

    elif method == "superpixels":
        # Use SLIC to reduce noise. We pick roughly n_samples superpixels.
        # Convert to float in [0,1] for SLIC
        img_float = img_as_float((hsv[..., :3].astype(np.float32)))  # HSV works fine for superpixels
        n_segments = max(50, min(n_samples, H * W // 100))  # heuristic bounds
        segments = slic(img_float, n_segments=n_segments, compactness=10, start_label=0)
        # Take one representative per segment (medoid-ish)
        flat_lab = lab.reshape(-1, 3)
        flat_seg = segments.reshape(-1)
        # sample one index per segment: the pixel closest to segment mean in Lab
        idx_list = []
        for s in range(segments.max() + 1):
            seg_idx = np.where(flat_seg == s)[0]
            if seg_idx.size == 0:
                continue
            seg_vecs = flat_lab[seg_idx]
            mu = seg_vecs.mean(axis=0, keepdims=True)
            d2 = ((seg_vecs - mu) ** 2).sum(axis=1)
            idx_list.append(seg_idx[np.argmin(d2)])
        idx = np.array(idx_list, dtype=np.int64)
        if idx.size > n_samples:
            idx = rng.choice(idx, size=n_samples, replace=False)

    else:
        raise ValueError(f"Unknown sampling method: {method}")

    hsv_s = hsv.reshape(-1, 3)[idx]
    lab_s = lab.reshape(-1, 3)[idx]
    return idx.astype(np.int64), hsv_s.astype(np.float32), lab_s.astype(np.float32)


# ---------------- Orchestrator ----------------

def preprocess_for_palette(
    rgb_u8: np.ndarray,
    *,
    max_long_edge: int = 640,
    pixel_sample: int = 30000,
    sample_method: Literal["uniform", "grid", "superpixels"] = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> ImageFeatures:
    """
    Full preprocessing for the pipeline:
      - resize to speed bounds
      - convert to HSV (deg, [0,1],[0,1]) and CIELab (float)
      - sample pixels
      - package outputs for clustering & rules
    """
    # 1) Resize
    small = resize_longest(rgb_u8, max_long_edge=max_long_edge)

    # 2) Colorspaces
    hsv = rgb_to_hsv_norm(small)         # float32, H in degrees
    lab = rgb_to_lab(small)              # float32 Lab

    # 3) Sampling
    idx, hsv_s, lab_s = sample_pixels(
        hsv, lab, n_samples=pixel_sample, method=sample_method, rng=rng
    )

    Hs, Ws = small.shape[:2]
    return ImageFeatures(
        rgb_small=small,
        hsv_small=hsv,
        lab_small=lab,
        sample_idx=idx,
        samples_hsv=hsv_s,
        samples_lab=lab_s,
        shape_small=(Hs, Ws),
    )
