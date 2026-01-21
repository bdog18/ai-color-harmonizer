from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union
from pathlib import Path
import time
import numpy as np

from harmonizer.io import load_image_rgb
from harmonizer.preprocess import preprocess_for_palette
from harmonizer.cluster import cluster_colors, ClusterResult
from harmonizer.palette import build_palette, find_accent_color, PaletteColor
from harmonizer.mood import infer_mood, MoodResult
from harmonizer.harmony import generate_harmonies, HarmonyResult


@dataclass
class AnalysisResult:
    palette: list
    accent: Optional[str]
    mood: dict
    harmony: dict
    diagnostics: dict


def _palette_to_jsonable(palette: list[PaletteColor]) -> list[dict]:
    return [
        {
            "hex": c.hex,
            "share": float(c.share),
            "salience": float(c.salience),
            "rgb": {"r": int(c.rgb[0]), "g": int(c.rgb[1]), "b": int(c.rgb[2])},
            "hsv": {"h": float(c.hsv[0]), "s": float(c.hsv[1]), "v": float(c.hsv[2])},
        }
        for c in palette
    ]


def analyze_image(
    src: Union[str, Path, bytes, "BytesIO"],
    *,
    cfg: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      I/O -> preprocess -> cluster -> palette -> mood -> harmony -> JSONable dict

    src can be a filepath or file-like (Streamlit upload).
    cfg: dict loaded from YAML (optional). Supports:
      image.max_long_edge
      image.pixel_sample
      image.sample_method
      cluster.k_candidates
      cluster.deltaE_merge
      cluster.min_share
      random_seed
      harmony.analogous_step_deg
      harmony.triadic_step_deg
      harmony.split_comp_step_deg
      mood.* (used in infer_mood)
    """
    cfg = cfg or {}
    t0 = time.time()

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    # ---- 1) I/O ----
    t_io = time.time()
    rgb = load_image_rgb(src)
    t_io = time.time() - t_io

    # ---- 2) Preprocess ----
    img_cfg = cfg.get("image", {})
    max_long_edge = int(img_cfg.get("max_long_edge", 640))
    pixel_sample = int(img_cfg.get("pixel_sample", 30000))
    sample_method = str(img_cfg.get("sample_method", "grid"))

    t_pre = time.time()
    feats = preprocess_for_palette(
        rgb,
        max_long_edge=max_long_edge,
        pixel_sample=pixel_sample,
        sample_method=sample_method,  # "uniform" | "grid" | "superpixels"
        rng=rng,
    )
    t_pre = time.time() - t_pre

    # ---- 3) Cluster ----
    cl_cfg = cfg.get("cluster", {})
    k_candidates = tuple(cl_cfg.get("k_candidates", [4, 5, 6]))
    deltaE_merge = float(cl_cfg.get("deltaE_merge", 6.0))
    min_share = float(cl_cfg.get("min_share", 0.015))

    t_cl = time.time()
    clusters: ClusterResult = cluster_colors(
        feats.samples_lab,
        feats.samples_hsv,
        k_candidates=k_candidates,
        seed=seed,
        deltaE_merge=deltaE_merge,
        min_share=min_share,
    )
    t_cl = time.time() - t_cl

    # ---- 4) Palette ----
    t_pal = time.time()
    palette = build_palette(clusters.centers_hsv, clusters.weights, sort_by="salience")
    accent = find_accent_color(palette)
    t_pal = time.time() - t_pal

    # ---- 5) Mood ----
    t_mood = time.time()
    mood: MoodResult = infer_mood(palette, cfg=cfg)
    t_mood = time.time() - t_mood

    # ---- 6) Harmony ----
    harm_cfg = cfg.get("harmony", {})
    analogous_step_deg = float(harm_cfg.get("analogous_step_deg", 30.0))
    triadic_step_deg = float(harm_cfg.get("triadic_step_deg", 120.0))
    split_comp_step_deg = float(harm_cfg.get("split_comp_step_deg", 30.0))

    t_harm = time.time()
    harmony: HarmonyResult = generate_harmonies(
        palette,
        base_index=0,
        analogous_step_deg=analogous_step_deg,
        triadic_step_deg=triadic_step_deg,
        split_comp_step_deg=split_comp_step_deg,
    )
    t_harm = time.time() - t_harm

    # ---- 7) Assemble JSONable output ----
    out = AnalysisResult(
        palette=_palette_to_jsonable(palette),
        accent=accent.hex if accent else None,
        mood={
            "primary": mood.primary,
            "tags": mood.tags,
            "scores": mood.scores,
            "explain": mood.explain,
        },
        harmony={
            "base": harmony.base_hex,
            "complementary": harmony.complementary,
            "analogous": harmony.analogous,
            "triadic": harmony.triadic,
            "split_complementary": harmony.split_complementary,
            "explain": harmony.explain,
        },
        diagnostics={
            "timing_sec": {
                "io": t_io,
                "preprocess": t_pre,
                "cluster": t_cl,
                "palette": t_pal,
                "mood": t_mood,
                "harmony": t_harm,
                "total": time.time() - t0,
            },
            "image": {
                "original_shape": tuple(rgb.shape),
                "resized_shape": tuple(feats.rgb_small.shape),
                "pixel_sample": int(len(feats.sample_idx)),
                "sample_method": sample_method,
            },
            "cluster": {
                "k_candidates": list(k_candidates),
                "k_chosen": int(clusters.k_chosen),
                "final_k": int(len(clusters.weights)),
                "deltaE_merge": float(deltaE_merge),
                "min_share": float(min_share),
                "merge_count": int(len(clusters.debug.get("merge", {}).get("merges", []))),
                "silhouette_scores": clusters.debug.get("k_selection", {}).get("silhouette_scores", {}),
            },
        },
    )

    return asdict(out)
