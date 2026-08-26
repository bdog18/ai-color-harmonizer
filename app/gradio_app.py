from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import os
import tempfile
from typing import Optional

import gradio as gr
import numpy as np

from harmonizer.io import load_image_rgb
from harmonizer.preprocess import preprocess_for_palette
from harmonizer.cluster import cluster_colors
from harmonizer.palette import build_palette, find_accent_color, explain_accent_color
from harmonizer.mood import infer_mood
from harmonizer.harmony import generate_harmonies
from harmonizer.viz import (
    plot_palette_swatches,
    plot_accent_swatch,
    plot_harmony_swatches,
    quantize_image_by_centers_hsv,
    overlay_quantized,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K_CHOICES = [3, 4, 5, 6, 7, 8]
DEFAULT_K = [4, 5, 6]


def _mood_markdown(mood) -> str:
    lines = [f"### Mood: {mood.primary.title()}"]
    if mood.tags:
        lines.append("**Tags:** " + ", ".join(t.title() for t in mood.tags))
    return "\n\n".join(lines)


def _mood_explain_markdown(mood) -> str:
    return "\n".join(f"- {line.title()}" for line in mood.explain)


def run_harmonizer(
    image_path: Optional[str],
    max_long_edge: int,
    pixel_sample: int,
    sample_method: str,
    k_candidates: list,
    deltaE_merge: float,
    min_share: float,
    overlay_alpha: float,
    seed: int,
):
    if not image_path:
        raise gr.Error("Upload an image to begin.")

    k_candidates = tuple(sorted(k_candidates)) if k_candidates else tuple(DEFAULT_K)

    try:
        rgb = load_image_rgb(image_path)
    except Exception as e:
        logger.warning("Failed to load image %s: %s", image_path, e)
        raise gr.Error(f"Failed to load image: {e}")

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

    quant = quantize_image_by_centers_hsv(
        feats.rgb_small, feats.sample_idx, clusters.labels, clusters.centers_hsv
    )
    blend = overlay_quantized(feats.rgb_small, quant, alpha=float(overlay_alpha))

    result = {
        "palette": [
            {
                "hex": c.hex,
                "share": c.share,
                "salience": c.salience,
                "rgb": {"r": c.rgb[0], "g": c.rgb[1], "b": c.rgb[2]},
                "hsv": {"h": c.hsv[0], "s": c.hsv[1], "v": c.hsv[2]},
            }
            for c in palette
        ],
        "accent": accent.hex if accent else None,
        "accent_explain": None if accent else explain_accent_color(palette),
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
                # orjson (used by gr.JSON) rejects non-str dict keys, unlike json.dumps
                "silhouette_scores": {
                    str(k): v
                    for k, v in clusters.debug.get("k_selection", {}).get("silhouette_scores", {}).items()
                },
            },
        },
    }

    palette_fig = plot_palette_swatches(palette, title="Palette", show_share=False, show_name=False, equal_width=True)

    accent_fig = None
    accent_note = ""
    if accent:
        accent_fig = plot_accent_swatch(accent.hex, title="Accent Color", show_hex=True)
    else:
        reasons = explain_accent_color(palette)
        accent_note = "**No accent color found.**\n\n" + "\n".join(f"- {line}" for line in reasons)

    swatches = [harm.base_hex, harm.complementary] + harm.analogous + harm.triadic + harm.split_complementary
    names = [harm.base_hex_name, harm.complementary_name] + harm.analogous_names + harm.triadic_names + harm.split_complementary_names
    labels = ["base", "comp", "ana-", "ana+", "tri+", "tri-", "split-", "split+"]
    harmony_fig = plot_harmony_swatches(swatches, names_list=names, labels=labels, title="Harmony Suggestions")

    # Written to a temp file rather than returned in-memory: gr.File/DownloadButton
    # outputs need a path on disk to serve the download from.
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(result, tmp, indent=2)
    tmp.close()

    return (
        rgb,
        blend,
        _mood_markdown(mood),
        _mood_explain_markdown(mood),
        palette_fig,
        accent_fig,
        accent_note,
        harmony_fig,
        result,
        tmp.name,
    )


css = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}
#header {
    text-align: center;
    margin-bottom: 0;
}
#subheader {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin-bottom: 12px;
}
.settings-box {
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 8px 12px;
}
.results-box {
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 12px 16px;
}
"""

with gr.Blocks(title="AI Color Harmonizer") as iface:
    gr.Markdown("# \U0001f3a8 AI Color Harmonizer", elem_id="header")
    gr.Markdown(
        "Extract a dominant palette, infer mood, and generate harmony palettes from any image.",
        elem_id="subheader",
    )

    image_input = gr.Image(label="Upload an image (JPG/PNG/WEBP)", type="filepath")

    with gr.Accordion("Settings", open=False, elem_classes=["settings-box"]):
        with gr.Row():
            max_long_edge = gr.Slider(256, 1024, value=640, step=32, label="Resize longest edge")
            pixel_sample = gr.Slider(5000, 60000, value=20000, step=5000, label="Pixel sample count")
        with gr.Row():
            sample_method = gr.Dropdown(["grid", "uniform", "superpixels"], value="grid", label="Sampling method")
            k_candidates = gr.CheckboxGroup(K_CHOICES, value=DEFAULT_K, label="K candidates")
        with gr.Row():
            deltaE_merge = gr.Slider(2.0, 15.0, value=6.0, step=0.5, label="Merge threshold ΔE (Lab)")
            min_share = gr.Slider(0.0, 0.08, value=0.015, step=0.005, label="Min cluster share")
        with gr.Row():
            overlay_alpha = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Overlay strength")
            seed = gr.Number(value=42, precision=0, label="Random seed")
        gr.Markdown("_Tip: If results look too 'samey', lower ΔE merge. If too noisy, raise min_share._")

    run_button = gr.Button("Analyze", variant="primary")

    with gr.Row():
        original_output = gr.Image(label="Original")
        overlay_output = gr.Image(label="Resized + Overlay (quantized samples)")

    with gr.Group(elem_classes=["results-box"]):
        mood_output = gr.Markdown()
        with gr.Accordion("Mood explanation", open=False):
            mood_explain_output = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Palette")
            palette_output = gr.Plot(label="Palette", show_label=False)
            gr.Markdown("**Accent:**")
            accent_output = gr.Plot(label="Accent", show_label=False)
            accent_note_output = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Harmonies")
            harmony_output = gr.Plot(label="Harmonies", show_label=False)

    gr.Markdown("### JSON Output")
    json_output = gr.JSON()
    download_output = gr.File(label="Download JSON")

    inputs = [
        image_input,
        max_long_edge,
        pixel_sample,
        sample_method,
        k_candidates,
        deltaE_merge,
        min_share,
        overlay_alpha,
        seed,
    ]
    outputs = [
        original_output,
        overlay_output,
        mood_output,
        mood_explain_output,
        palette_output,
        accent_output,
        accent_note_output,
        harmony_output,
        json_output,
        download_output,
    ]
    run_button.click(fn=run_harmonizer, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(),
        css=css,
    )
