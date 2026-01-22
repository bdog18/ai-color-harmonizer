# AI Color Harmonizer

Extract dominant color palettes, infer mood, and generate harmonious color schemes from any image.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Palette Extraction** — Automatically extract dominant colors using K-means clustering in CIE Lab color space with silhouette-based K selection
- **Mood Inference** — Rule-based mood analysis (calming, energetic, cozy, minimal, dramatic, pastel, vibrant) with warm/cool and brightness tagging
- **Color Harmonies** — Generate complementary, analogous, triadic, and split-complementary palettes that match the image's visual style
- **Accent Detection** — Identify accent colors based on hue distance, saturation, and salience
- **Interactive Controls** — Fine-tune clustering parameters, sampling methods, and visualization options via the sidebar

## Demo

Upload any image to:

1. View the extracted color palette with share percentages
2. See the inferred mood with explanation
3. Explore generated harmony palettes
4. Download results as JSON

## Installation

### Local Setup

```bash
git clone https://github.com/yourusername/ai-color-harmonizer.git
cd ai-color-harmonizer
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker build -t ai-color-harmonizer .
docker run -p 8080:8080 ai-color-harmonizer
```

## Usage

1. **Upload** a JPG, PNG, or WEBP image
2. **Adjust** settings in the sidebar:
   - Resize longest edge (256–1024px)
   - Pixel sample count (5K–60K)
   - Sampling method (grid, uniform, superpixels)
   - K candidates for clustering
   - Merge threshold (ΔE in Lab)
   - Minimum cluster share
3. **View** results: original image, quantized overlay, palette, mood, and harmonies
4. **Download** the JSON output

## Project Structure

```
├── app/
│   └── streamlit_app.py      # Streamlit UI
├── harmonizer/
│   ├── io.py                 # Image loading/saving
│   ├── preprocess.py         # Resize, color conversion, sampling
│   ├── cluster.py            # K-means clustering, merge/prune
│   ├── palette.py            # Palette building, accent detection
│   ├── mood.py               # Mood inference rules
│   ├── harmony.py            # Color harmony generation
│   ├── viz.py                # Visualization utilities
│   └── pipeline.py           # End-to-end analysis pipeline
├── configs/
│   └── default.yaml          # Default configuration
├── demo_images/              # Sample images by mood category
├── notebooks/
│   └── 01_prototype.ipynb    # Development notebook
├── tests/
│   └── test_palette.py       # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## How It Works

### 1. Preprocessing
Images are resized and converted to HSV and CIE Lab color spaces. Pixels are sampled using grid, uniform random, or SLIC superpixel methods.

### 2. Clustering
K-means runs on Lab samples with automatic K selection via silhouette score. Similar clusters are merged if their ΔE distance is below threshold, and tiny clusters are pruned.

### 3. Palette Building
Cluster centers are converted to hex colors with CSS3 names. Colors are sorted by salience (weighted combination of share, saturation, and brightness).

### 4. Mood Inference
Rules analyze hue distribution, saturation, brightness, and contrast to score moods and assign tags like "warm", "cool", "pastel", or "vibrant".

### 5. Harmony Generation
Starting from the dominant color, harmonies are computed using standard color wheel relationships with saturation/brightness adjusted to match the image's style (pastel, muted, vibrant, or dark).

## Configuration

Settings can be adjusted via the sidebar or by editing [`configs/default.yaml`](configs/default.yaml):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_long_edge` | Resize target | 640 |
| `pixel_sample` | Samples for clustering | 30000 |
| `sample_method` | grid / uniform / superpixels | grid |
| `k_candidates` | K values to try | [4, 5, 6] |
| `deltaE_merge` | Cluster merge threshold | 6.0 |
| `min_share` | Minimum cluster share | 0.015 |

## Requirements

- Python 3.11+
- streamlit
- numpy
- opencv-python
- scikit-learn
- scikit-image
- matplotlib
- webcolors
- Pillow
- PyYAML

## License

MIT