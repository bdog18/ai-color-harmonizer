from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from harmonizer.palette import PaletteColor, css3_name, hsv_to_rgb_u8

@dataclass
class HarmonyResult:
    base_hex: str
    base_hex_name: str
    complementary: str
    complementary_name: str
    analogous: List[str]
    analogous_names: List[str]
    triadic: List[str]
    triadic_names: List[str]
    split_complementary: List[str]
    split_complementary_names: List[str]
    explain: List[str]


def wrap_hue(h: float) -> float:
    return float(h % 360.0)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def adjust_sv_to_style(
    h: float,
    s: float,
    v: float,
    *,
    style: str,
    ref_s: float,
    ref_v: float,
) -> Tuple[float, float, float]:
    """
    Keep hue, adjust S/V so harmonies feel consistent with the image/palette "finish".
    style: "pastel" | "muted" | "vibrant" | "dark" | "default"
    ref_s/ref_v: aggregated S/V from palette (e.g., weighted means)
    """
    if style == "pastel":
        s2 = min(s, 0.35)                 # keep low saturation
        v2 = max(v, max(ref_v, 0.80))     # keep bright
    elif style == "muted":
        s2 = min(s, max(ref_s, 0.28))
        v2 = v * 0.95 + ref_v * 0.05
    elif style == "vibrant":
        s2 = max(s, max(ref_s, 0.65))
        v2 = v * 0.90 + ref_v * 0.10
    elif style == "dark":
        s2 = max(s, ref_s * 0.9)
        v2 = min(v, min(ref_v, 0.35))
    else:
        # default: nudge toward palette average slightly
        s2 = 0.85 * s + 0.15 * ref_s
        v2 = 0.85 * v + 0.15 * ref_v

    return wrap_hue(h), clamp01(s2), clamp01(v2)


def hsv_to_hex(h: float, s: float, v: float) -> str:
    """
    Convert HSV (H degrees, S/V in [0,1]) to hex without external deps.
    """
    h = wrap_hue(h)
    s = clamp01(s)
    v = clamp01(v)

    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c

    if 0 <= h < 60:
        rp, gp, bp = c, x, 0
    elif 60 <= h < 120:
        rp, gp, bp = x, c, 0
    elif 120 <= h < 180:
        rp, gp, bp = 0, c, x
    elif 180 <= h < 240:
        rp, gp, bp = 0, x, c
    elif 240 <= h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x

    r = int(round((rp + m) * 255))
    g = int(round((gp + m) * 255))
    b = int(round((bp + m) * 255))

    return f"#{r:02X}{g:02X}{b:02X}"


def infer_style_from_palette(palette: List[PaletteColor]) -> Tuple[str, float, float]:
    """
    Infer a style label and compute reference S/V means.
    Returns (style, S_mean, V_mean).
    """
    shares = np.array([c.share for c in palette], dtype=np.float32)
    shares = shares / max(1e-9, shares.sum())
    S = np.array([c.hsv[1] for c in palette], dtype=np.float32)
    V = np.array([c.hsv[2] for c in palette], dtype=np.float32)

    s_mean = float(np.sum(S * shares))
    v_mean = float(np.sum(V * shares))

    # Simple, consistent with mood.py defaults
    if v_mean >= 0.80 and s_mean <= 0.35:
        style = "pastel"
    elif s_mean <= 0.30:
        style = "muted"
    elif v_mean <= 0.35:
        style = "dark"
    elif s_mean >= 0.60:
        style = "vibrant"
    else:
        style = "default"

    return style, s_mean, v_mean


def generate_harmonies(
    palette: List[PaletteColor],
    *,
    base_index: int = 0,            # usually dominant color
    analogous_step_deg: float = 30.0,
    triadic_step_deg: float = 120.0,
    split_comp_step_deg: float = 30.0,
) -> HarmonyResult:
    """
    Generate harmony palettes from a base palette color, adjusting S/V to match style.
    Outputs hex strings ready for UI.
    """
    if not palette:
        return HarmonyResult(
            base_hex="#000000",
            base_hex_name="unknown",
            complementary="#000000",
            complementary_name="unknown",
            analogous=[],
            analogous_names=[],
            triadic=[],
            triadic_names=[],
            split_complementary=[],
            split_complementary_names=[],
            explain=["empty palette"],
        )

    base = palette[min(base_index, len(palette) - 1)]
    base_h, base_s, base_v = base.hsv

    style, ref_s, ref_v = infer_style_from_palette(palette)
    explain = [f"style={style} (ref_s={ref_s:.2f}, ref_v={ref_v:.2f})"]

    # --- Harmony hues ---
    comp_h = wrap_hue(base_h + 180.0)
    ana1_h = wrap_hue(base_h - analogous_step_deg)
    ana2_h = wrap_hue(base_h + analogous_step_deg)

    tri1_h = wrap_hue(base_h + triadic_step_deg)
    tri2_h = wrap_hue(base_h - triadic_step_deg)

    sc1_h = wrap_hue(comp_h - split_comp_step_deg)
    sc2_h = wrap_hue(comp_h + split_comp_step_deg)

    # --- Adjust S/V to match style ---
    _, comp_s, comp_v = adjust_sv_to_style(comp_h, base_s, base_v, style=style, ref_s=ref_s, ref_v=ref_v)
    _, ana1_s, ana1_v = adjust_sv_to_style(ana1_h, base_s, base_v, style=style, ref_s=ref_s, ref_v=ref_v)
    _, ana2_s, ana2_v = adjust_sv_to_style(ana2_h, base_s, base_v, style=style, ref_s=ref_s, ref_v=ref_v)
    _, tri1_s, tri1_v = adjust_sv_to_style(tri1_h, base_s, base_v, style=style, ref_s=ref_s, ref_v=ref_v)
    _, tri2_s, tri2_v = adjust_sv_to_style(tri2_h, base_s, base_v, style=style, ref_s=ref_s, ref_v=ref_v)
    _, sc1_s, sc1_v = adjust_sv_to_style(sc1_h, base_s, base_v, style=style, ref_s=ref_s, ref_v=ref_v)
    _, sc2_s, sc2_v = adjust_sv_to_style(sc2_h, base_s, base_v, style=style, ref_s=ref_s, ref_v=ref_v)

    # --- Convert to hex ---
    base_hex = base.hex
    comp_hex = hsv_to_hex(comp_h, comp_s, comp_v)
    ana_hex = [hsv_to_hex(ana1_h, ana1_s, ana1_v), hsv_to_hex(ana2_h, ana2_s, ana2_v)]
    tri_hex = [hsv_to_hex(tri1_h, tri1_s, tri1_v), hsv_to_hex(tri2_h, tri2_s, tri2_v)]
    split_hex = [hsv_to_hex(sc1_h, sc1_s, sc1_v), hsv_to_hex(sc2_h, sc2_s, sc2_v)]

    explain.append(f"base_hsv=({base_h:.1f}, {base_s:.2f}, {base_v:.2f})")
    explain.append(f"comp_h={comp_h:.1f}, analogous=[{ana1_h:.1f},{ana2_h:.1f}], triadic=[{tri1_h:.1f},{tri2_h:.1f}]")

    base_rgb = hsv_to_rgb_u8(np.asarray((base_h, base_s, base_v), dtype=np.float32))[0]
    comp_rgb = hsv_to_rgb_u8(np.asarray((comp_h, comp_s, comp_v), dtype=np.float32))[0]
    ana1_rgb = hsv_to_rgb_u8(np.asarray((ana1_h, ana1_s, ana1_v), dtype=np.float32))[0]
    ana2_rgb = hsv_to_rgb_u8(np.asarray((ana2_h, ana2_s, ana2_v), dtype=np.float32))[0]
    tri1_rgb = hsv_to_rgb_u8(np.asarray((tri1_h, tri1_s, tri1_v), dtype=np.float32))[0]
    tri2_rgb = hsv_to_rgb_u8(np.asarray((tri2_h, tri2_s, tri2_v), dtype=np.float32))[0]
    sc1_rgb = hsv_to_rgb_u8(np.asarray((sc1_h, sc1_s, sc1_v), dtype=np.float32))[0]
    sc2_rgb = hsv_to_rgb_u8(np.asarray((sc2_h, sc2_s, sc2_v), dtype=np.float32))[0]

    return HarmonyResult(
        base_hex=base_hex,
        base_hex_name=css3_name((int(base_rgb[0]), int(base_rgb[1]), int(base_rgb[2]))),
        complementary=comp_hex,
        complementary_name=css3_name((int(comp_rgb[0]), int(comp_rgb[1]), int(comp_rgb[2]))),
        analogous=ana_hex,
        analogous_names=[css3_name((int(ana1_rgb[0]), int(ana1_rgb[1]), int(ana1_rgb[2]))),
                        css3_name((int(ana2_rgb[0]), int(ana2_rgb[1]), int(ana2_rgb[2])))],
        triadic=tri_hex,
        triadic_names=[css3_name((int(tri1_rgb[0]), int(tri1_rgb[1]), int(tri1_rgb[2]))),
                       css3_name((int(tri2_rgb[0]), int(tri2_rgb[1]), int(tri2_rgb[2])))],
        split_complementary=split_hex,
        split_complementary_names=[css3_name((int(sc1_rgb[0]), int(sc1_rgb[1]), int(sc1_rgb[2]))),
                                   css3_name((int(sc2_rgb[0]), int(sc2_rgb[1]), int(sc2_rgb[2])))],
        explain=explain,
    )