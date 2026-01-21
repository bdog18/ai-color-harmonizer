from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2

from harmonizer.palette import PaletteColor


# ----------------------------
# Small helpers
# ----------------------------

def hex_to_rgb_u8(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def rgb_u8_to_hex(rgb: np.ndarray) -> str:
    r, g, b = [int(x) for x in rgb.tolist()]
    return f"#{r:02X}{g:02X}{b:02X}"


def _text_color_for_rgb(rgb01: np.ndarray) -> str:
    # Simple luminance heuristic for label readability
    return "white" if float(rgb01.mean()) < 0.5 else "black"


# ----------------------------
# Palette swatches
# ----------------------------

def plot_palette_swatches(
    palette: List[PaletteColor],
    *,
    title: str = "Palette",
    show_hex: bool = True,
    show_name: bool = True,
    show_share: bool = True,
    equal_width: bool = False,
    figsize: Tuple[float, float] = (10, 1.8),
):
    """
    Matplotlib swatch bar.
    - If equal_width=False: widths reflect share (sum to 1)
    - If equal_width=True: each swatch has width = 1/len(palette)
    Returns fig for notebook or Streamlit.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = 0.0

    n = max(1, len(palette))
    default_w = 1.0 / n

    for c in palette:
        w = default_w if equal_width else float(c.share)

        rgb01 = np.array(c.rgb, dtype=np.float32) / 255.0
        ax.add_patch(plt.Rectangle((x, 0), w, 1, color=rgb01))

        label_parts = []
        if show_hex:
            label_parts.append(c.hex)
        if show_name:
            name = getattr(c, "name", None)
            if name:
                label_parts.append(name)
        if show_share and (not equal_width):
            label_parts.append(f"{c.share*100:.0f}%")

        label = " • ".join(label_parts)

        if label:
            ax.text(
                x + w / 2,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=_text_color_for_rgb(rgb01),
            )

        x += w

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_accent_swatch(
    accent: str,
    title: str = "Accent Color",
    show_hex: bool = True,
    figsize: Tuple[float, float] = (10, 1.8),
):
    """
    Matplotlib single swatch.
    Returns fig for notebook or Streamlit.
    """
    fig, ax = plt.subplots(figsize=figsize)
    rgb_u8 = hex_to_rgb_u8(accent)
    rgb01 = rgb_u8.astype(np.float32) / 255.0
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=rgb01))

    if show_hex:
        ax.text(
            0.5,
            0.5,
            accent,
            ha="center",
            va="center",
            fontsize=9,
            color=_text_color_for_rgb(rgb01),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_harmony_swatches(
    hex_list: List[str],
    names_list: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    *,
    title: str = "Harmonies",
    figsize: Tuple[float, float] = (10, 1.8),
):
    """
    Plot a row of equally sized swatches given hex colors.
    """
    labels = labels or [""] * len(hex_list)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (hx, name, lab) in enumerate(zip(hex_list, names_list or [""] * len(hex_list), labels)):
        rgb01 = hex_to_rgb_u8(hx).astype(np.float32) / 255.0
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb01))
        if lab:
            ax.text(
                i + 0.5,
                0.5,
                f"{lab}" + (f"\n{hx}") + (f"\n{name}" if name else ""),
                ha="center",
                va="center",
                fontsize=9,
                color=_text_color_for_rgb(rgb01),
            )

    ax.set_xlim(0, len(hex_list))
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ----------------------------
# Quantization / overlays
# ----------------------------

def quantize_image_by_centers_hsv(
    rgb_small_u8: np.ndarray,
    sample_idx: np.ndarray,
    labels: np.ndarray,
    centers_hsv: np.ndarray,
) -> np.ndarray:
    """
    Recolor only sampled pixels using their cluster center color.
    This is a quick sanity visualization (not full per-pixel assignment).
    Returns RGB uint8 image same shape as rgb_small_u8.
    """
    H, W = rgb_small_u8.shape[:2]
    flat = rgb_small_u8.reshape(-1, 3).copy()

    # centers_hsv is (K,3) with H degrees, S/V in [0,1]
    # convert to RGB using OpenCV HSV convention
    hsv_cv = centers_hsv.astype(np.float32).copy()
    hsv_cv[:, 0] = hsv_cv[:, 0] / 2.0
    hsv_cv[:, 1] = hsv_cv[:, 1] * 255.0
    hsv_cv[:, 2] = hsv_cv[:, 2] * 255.0
    hsv_cv = np.clip(hsv_cv, 0, 255).astype(np.uint8)

    rgb_centers = cv2.cvtColor(hsv_cv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)

    flat[sample_idx] = rgb_centers[labels]
    return flat.reshape(H, W, 3)


def overlay_quantized(
    original_rgb_u8: np.ndarray,
    quant_rgb_u8: np.ndarray,
    *,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Alpha blend between original and quantized for visualization.
    """
    a = float(max(0.0, min(1.0, alpha)))
    out = (a * quant_rgb_u8.astype(np.float32) + (1 - a) * original_rgb_u8.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def show_side_by_side(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    title_a: str = "A",
    title_b: str = "B",
    figsize: Tuple[float, float] = (10, 5),
):
    """
    Quick notebook view: two images side-by-side.
    Returns fig.
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(img_a)
    axs[0].set_title(title_a)
    axs[0].axis("off")

    axs[1].imshow(img_b)
    axs[1].set_title(title_b)
    axs[1].axis("off")

    fig.tight_layout()
    return fig


# ----------------------------
# Distributions
# ----------------------------

def plot_hue_histogram(
    hsv_small: np.ndarray,
    *,
    bins: int = 36,
    figsize: Tuple[float, float] = (8, 3),
    title: str = "Hue Histogram",
):
    """
    Simple hue histogram weighted by saturation (so grays don't dominate).
    """
    H = hsv_small[..., 0].reshape(-1)  # degrees
    S = hsv_small[..., 1].reshape(-1)

    # Weight hue by saturation so neutrals don't overwhelm
    weights = S.astype(np.float32)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(H, bins=bins, range=(0, 360), weights=weights)
    ax.set_xlabel("Hue (degrees)")
    ax.set_ylabel("Weighted count (by saturation)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
