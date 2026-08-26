from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import webcolors as wc


@dataclass
class PaletteColor:
    name: str
    hex: str
    rgb: Tuple[int, int, int]        # 0-255
    hsv: Tuple[float, float, float]  # H in [0,360), S,V in [0,1]
    share: float                     # sums to 1
    salience: float                  # used for ordering


def hsv_to_rgb_u8(hsv: np.ndarray) -> np.ndarray:
    """
    hsv: (..., 3) float32 with H degrees [0,360), S,V in [0,1]
    returns RGB uint8 (...,3)
    """
    hsv = np.asarray(hsv, dtype=np.float32)
    hsv_cv = hsv.copy()
    hsv_cv[..., 0] = hsv_cv[..., 0] / 2.0        # to OpenCV hue [0,180)
    hsv_cv[..., 1] = hsv_cv[..., 1] * 255.0
    hsv_cv[..., 2] = hsv_cv[..., 2] * 255.0
    hsv_cv = np.clip(hsv_cv, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(hsv_cv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return rgb.astype(np.uint8)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def compute_salience(weights: np.ndarray, centers_hsv: np.ndarray) -> np.ndarray:
    """
    Simple, interpretable salience:
      salience = weight * (0.4 + 0.3*S + 0.3*V)

    - keeps dominance via weight
    - boosts saturated/bright colors slightly so accents can rise
    """
    w = weights.astype(np.float32)
    s = centers_hsv[:, 1].astype(np.float32)
    v = centers_hsv[:, 2].astype(np.float32)
    sal = w * (0.4 + 0.3 * s + 0.3 * v)
    return sal.astype(np.float32)


def css3_exact_name(rgb: tuple[int, int, int]) -> str | None:
    """Exact CSS3 name for an RGB tuple, or None if not named."""
    try:
        return wc.rgb_to_name(rgb, spec="css3")
    except ValueError:
        return None


def css3_closest_name(rgb: tuple[int, int, int]) -> str:
    """Closest CSS3 name to an RGB tuple (Euclidean distance in RGB)."""
    target = np.array(rgb, dtype=np.int16)

    best_name: str | None = None
    best_dist = 1_000_000_000

    # Use documented API: names() + name_to_rgb()
    for name in wc.names(spec="css3"):
        named_rgb = wc.name_to_rgb(name, spec="css3")  # returns IntegerRGB (tuple-like)
        named_rgb = np.array(named_rgb, dtype=np.int16)

        dist = int(np.sum((target - named_rgb) ** 2))  # squared Euclidean
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name or "unknown"


def css3_name(rgb: tuple[int, int, int]) -> str:
    """Exact CSS3 name if available, otherwise closest CSS3 name."""
    return css3_exact_name(rgb) or css3_closest_name(rgb)


def build_palette(
    centers_hsv: np.ndarray,
    weights: np.ndarray,
    *,
    sort_by: str = "salience",   # "share" or "salience"
) -> List[PaletteColor]:
    """
    Convert cluster centers + weights into a sorted palette.
    """
    centers_hsv = np.asarray(centers_hsv, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    # normalize weights defensively
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / total

    sal = compute_salience(weights, centers_hsv)
    rgb_u8 = hsv_to_rgb_u8(centers_hsv)  # (K,3)

    colors: List[PaletteColor] = []
    for hsv, w, s, rgb in zip(centers_hsv, weights, sal, rgb_u8):
        rgb_t = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        hsv_t = (float(hsv[0]), float(hsv[1]), float(hsv[2]))
        colors.append(
            PaletteColor(
                name=css3_name(rgb_t),
                hex=rgb_to_hex(rgb_t),
                rgb=rgb_t,
                hsv=hsv_t,
                share=float(w),
                salience=float(s),
            )
        )

    if sort_by == "share":
        colors.sort(key=lambda c: c.share, reverse=True)
    else:
        colors.sort(key=lambda c: c.salience, reverse=True)

    # renormalize shares after sorting (keeps sum=1 exactly)
    ssum = sum(c.share for c in colors)
    if ssum > 0:
        for i in range(len(colors)):
            colors[i].share = colors[i].share / ssum

    return colors


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _accent_candidate_reasons(
    palette: List[PaletteColor],
    *,
    min_share: float,
    hue_distance_deg: float,
    min_sat: float,
) -> List[Tuple[PaletteColor, List[str]]]:
    """
    For each non-dominant color, the reasons it fails the accent heuristic
    (empty list means it qualifies). Shared by find_accent_color and
    explain_accent_color so the two can't drift out of sync.
    """
    if not palette:
        return []
    base = palette[0]

    results: List[Tuple[PaletteColor, List[str]]] = []
    for c in palette[1:]:
        reasons = []
        if c.share < min_share:
            reasons.append(f"share {c.share:.3f} is below min_share {min_share:.3f}")
        if c.hsv[1] < min_sat:
            reasons.append(f"saturation {c.hsv[1]:.2f} is below min_sat {min_sat:.2f}")
        dist = _hue_dist(c.hsv[0], base.hsv[0])
        if dist < hue_distance_deg:
            reasons.append(f"hue is only {dist:.1f}° from the base color (needs ≥ {hue_distance_deg:.1f}°)")
        results.append((c, reasons))
    return results


def find_accent_color(
    palette: List[PaletteColor],
    *,
    min_share: float = 0.03,
    hue_distance_deg: float = 80.0,
    min_sat: float = 0.35,
) -> Optional[PaletteColor]:
    """
    Heuristic accent: a non-tiny, fairly saturated color far in hue from dominant.
    Returns a PaletteColor or None.
    """
    if not palette:
        return None

    candidates = [
        c
        for c, reasons in _accent_candidate_reasons(
            palette, min_share=min_share, hue_distance_deg=hue_distance_deg, min_sat=min_sat
        )
        if not reasons
    ]

    # pick the most "accent-y": high salience among candidates
    if not candidates:
        return None
    candidates.sort(key=lambda c: c.salience, reverse=True)
    return candidates[0]


def explain_accent_color(
    palette: List[PaletteColor],
    *,
    min_share: float = 0.03,
    hue_distance_deg: float = 80.0,
    min_sat: float = 0.35,
) -> List[str]:
    """
    Human-readable reasons why find_accent_color did or didn't return an
    accent, using the same thresholds (see _accent_candidate_reasons).
    """
    if not palette:
        return ["Palette is empty."]

    base = palette[0]
    lines = [f"Base color: {base.hex} (hue={base.hsv[0]:.1f}°, share={base.share:.2f})"]

    evaluated = _accent_candidate_reasons(
        palette, min_share=min_share, hue_distance_deg=hue_distance_deg, min_sat=min_sat
    )
    found_candidate = False
    for c, reasons in evaluated:
        if reasons:
            lines.append(f"{c.hex}: {'; '.join(reasons)}")
        else:
            lines.append(f"{c.hex}: qualifies as an accent candidate")
            found_candidate = True

    if not found_candidate:
        lines.append(
            "No other palette color is both saturated and far enough in hue "
            "from the base color, so no accent was found."
        )
    return lines
