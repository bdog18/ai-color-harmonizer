from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import streamlit as st
from typing import Literal

from harmonizer.io import load_image_rgb
from harmonizer.preprocess import preprocess_for_palette
from harmonizer.cluster import cluster_colors
from harmonizer.palette import build_palette, find_accent_color
from harmonizer.mood import infer_mood
from harmonizer.harmony import generate_harmonies
from harmonizer.viz import (
    plot_palette_swatches,
    plot_accent_swatch,
    plot_harmony_swatches,
    quantize_image_by_centers_hsv,
    overlay_quantized,
)

st.set_page_config(page_title="AI Color Harmonizer", layout="wide")
st.title("AI Color Harmonizer")
st.caption("Extract a dominant palette, infer mood, and generate harmony palettes from any image.")

# ----------------------------
# Sidebar controls (config knobs)
# ----------------------------
st.sidebar.header("Settings")

max_long_edge = st.sidebar.slider("Resize longest edge", 256, 1024, 640, step=32)
pixel_sample = st.sidebar.slider("Pixel sample count", 5000, 60000, 20000, step=5000)
sample_method: Literal["grid", "uniform", "superpixels"] = st.sidebar.selectbox("Sampling method", ["grid", "uniform", "superpixels"], index=0)  # type: ignore

k_candidates = st.sidebar.multiselect("K candidates", [3, 4, 5, 6, 7, 8], default=[4, 5, 6])
k_candidates = tuple(sorted(k_candidates)) if k_candidates else (4, 5, 6)

deltaE_merge = st.sidebar.slider("Merge threshold ΔE (Lab)", 2.0, 15.0, 6.0, step=0.5)
min_share = st.sidebar.slider("Min cluster share", 0.0, 0.08, 0.015, step=0.005)

overlay_alpha = st.sidebar.slider("Overlay strength", 0.0, 1.0, 0.65, step=0.05)

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

st.sidebar.divider()
st.sidebar.caption("Tip: If results look too 'samey', lower ΔE merge. If too noisy, raise min_share.")

# ----------------------------
# Upload
# ----------------------------
uploaded = st.file_uploader("Upload an image (JPG/PNG/WEBP)", type=["jpg", "jpeg", "png", "webp"])

if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

# ----------------------------
# Run pipeline (manual assembly so we can show intermediate visuals)
# ----------------------------
try:
    rgb = load_image_rgb(uploaded)
except Exception as e:
    st.error(f"Failed to load image: {e}")
    st.stop()

rng = np.random.default_rng(int(seed))

feats = preprocess_for_palette(
    rgb,
    max_long_edge=int(max_long_edge),
    pixel_sample=int(pixel_sample),
    sample_method=sample_method,
    rng=rng,
)

clusters = cluster_colors(
    feats.samples_lab,
    feats.samples_hsv,
    k_candidates=k_candidates,
    seed=int(seed),
    deltaE_merge=float(deltaE_merge),
    min_share=float(min_share),
)

palette = build_palette(clusters.centers_hsv, clusters.weights, sort_by="salience")
accent = find_accent_color(palette)
mood = infer_mood(palette, cfg=None)
harm = generate_harmonies(palette)

# Quantized overlay (sampled pixels recolored)
quant = quantize_image_by_centers_hsv(
    feats.rgb_small, feats.sample_idx, clusters.labels, clusters.centers_hsv
)
blend = overlay_quantized(feats.rgb_small, quant, alpha=float(overlay_alpha))

# Prepare JSON output (simple + readable)
result = {
    "palette": [
        {
            #"name": c.name,
            "hex": c.hex,
            "share": c.share,
            "salience": c.salience,
            "rgb": {"r": c.rgb[0], "g": c.rgb[1], "b": c.rgb[2]},
            "hsv": {"h": c.hsv[0], "s": c.hsv[1], "v": c.hsv[2]},
        }
        for c in palette
    ],
    "accent": accent.hex if accent else None,
    "mood": {
        "primary": mood.primary,
        "tags": mood.tags,
        "scores": mood.scores,
        "explain": mood.explain,
    },
    "harmony": {
        "base": harm.base_hex,
        "complementary": harm.complementary,
        "analogous": harm.analogous,
        "triadic": harm.triadic,
        "split_complementary": harm.split_complementary,
        "explain": harm.explain,
    },
    "diagnostics": {
        "image": {
            "original_shape": list(rgb.shape),
            "resized_shape": list(feats.rgb_small.shape),
            "sample_method": sample_method,
            "pixel_sample": int(len(feats.sample_idx)),
        },
        "cluster": {
            "k_candidates": list(k_candidates),
            "k_chosen": clusters.k_chosen,
            "final_k": int(len(clusters.weights)),
            "deltaE_merge": float(deltaE_merge),
            "min_share": float(min_share),
            "merge_count": int(len(clusters.debug.get("merge", {}).get("merges", []))),
            "silhouette_scores": clusters.debug.get("k_selection", {}).get("silhouette_scores", {}),
        },
    },
}

# ----------------------------
# Layout
# ----------------------------
img_col, overlay_col = st.columns([1, 1])

with img_col:
    st.subheader("Original")
    st.image(rgb, clamp=True, channels="RGB")


with overlay_col:
    st.subheader("Resized + Overlay (quantized samples)")
    st.image(blend, clamp=True, channels="RGB")
    
# Palette mood
mood_col = st.columns([1])[0]

with mood_col:
    st.subheader(f"Mood: **{str.title(mood.primary)}**")
    if mood.tags:
        st.write("Tags:", ", ".join([str.title(tag) for tag in mood.tags]))
    with st.expander("Mood explanation"):
        for line in mood.explain:
            st.write("-", str.title(line))

# Palette and harmonies
palette_col, harmonies_col = st.columns([1, 1])

with palette_col:
    st.subheader("Palette")
    fig = plot_palette_swatches(palette, title="Palette", show_share=False, show_name=False, equal_width=True)
    st.pyplot(fig, clear_figure=True)

    if accent:
        st.markdown(f"**Accent:**")
        fig = plot_accent_swatch(accent.hex, title="Accent Color", show_hex=True)
        st.pyplot(fig, clear_figure=True)

with harmonies_col:
    st.subheader("Harmonies")
    swatches = [harm.base_hex, harm.complementary] + harm.analogous + harm.triadic + harm.split_complementary
    names = [harm.base_hex_name, harm.complementary_name] + harm.analogous_names + harm.triadic_names + harm.split_complementary_names
    labels = ["base", "comp", "ana-", "ana+", "tri+", "tri-", "split-", "split+"]

    hfig = plot_harmony_swatches(swatches, labels=labels, title="Harmony Suggestions")
    st.pyplot(hfig, clear_figure=True)



st.divider()

# JSON output + download
st.subheader("JSON Output")

st.download_button(
    label="Download JSON",
    data=json.dumps(result, indent=2),
    file_name="color_harmonizer_result.json",
    mime="application/json",
)
st.json(result)
