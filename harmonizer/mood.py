from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from harmonizer.palette import PaletteColor


@dataclass
class MoodResult:
    primary: str
    tags: List[str]
    scores: Dict[str, float]
    explain: List[str]


# ----------------------------
# Hue helpers
# ----------------------------

def hue_bucket(h: float) -> str:
    """
    Map hue degrees -> coarse bucket.
    """
    h = float(h) % 360.0
    if h >= 345 or h < 15:
        return "red"
    if h < 45:
        return "orange"
    if h < 75:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 315:
        return "purple"
    return "magenta"


def is_warm(h: float) -> bool:
    # warm: red/orange/yellow-ish
    h = float(h) % 360.0
    return (h >= 315 or h < 90)


def is_cool(h: float) -> bool:
    # cool: green/cyan/blue/purple-ish (roughly)
    h = float(h) % 360.0
    return (90 <= h < 315)


def circ_mean_deg(hues: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted circular mean for degrees.
    """
    hues = np.deg2rad(hues.astype(np.float32))
    w = weights.astype(np.float32)
    w = w / max(1e-9, w.sum())
    x = np.sum(np.cos(hues) * w)
    y = np.sum(np.sin(hues) * w)
    mean = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0
    return float(mean)


def circ_dispersion_deg(hues: np.ndarray, weights: np.ndarray) -> float:
    """
    Rough circular dispersion proxy in degrees.
    Returns something like "how spread out are hues".
    """
    hues_r = np.deg2rad(hues.astype(np.float32))
    w = weights.astype(np.float32)
    w = w / max(1e-9, w.sum())
    x = np.sum(np.cos(hues_r) * w)
    y = np.sum(np.sin(hues_r) * w)
    R = np.sqrt(x * x + y * y)  # in [0,1], 1 means concentrated
    # convert to an intuitive-ish spread: 0->max spread, 1->min spread
    spread = (1.0 - float(R)) * 180.0
    return spread


# ----------------------------
# Mood inference
# ----------------------------

def infer_mood(
    palette: List[PaletteColor],
    *,
    cfg: Optional[dict] = None,
) -> MoodResult:
    """
    Rule-based mood inference using palette HSV + shares.

    cfg keys (optional):
      mood:
        s_low, s_high, v_dark, v_bright
        energetic_contrast (default 0.25)
        calming_hue_spread (default 60)
        vibrant_sat (default 0.70)
        pastel_sat_max (default 0.35)
        pastel_v_min (default 0.80)
    """
    if not palette:
        return MoodResult(primary="unknown", tags=[], scores={}, explain=["empty palette"])

    # Pull config with defaults
    mood_cfg = (cfg or {}).get("mood", {}) if cfg else {}
    s_low = float(mood_cfg.get("s_low", 0.30))
    s_high = float(mood_cfg.get("s_high", 0.60))
    v_dark = float(mood_cfg.get("v_dark", 0.35))
    v_bright = float(mood_cfg.get("v_bright", 0.80))

    energetic_contrast = float(mood_cfg.get("energetic_contrast", 0.25))
    calming_hue_spread = float(mood_cfg.get("calming_hue_spread", 60.0))
    vibrant_sat = float(mood_cfg.get("vibrant_sat", 0.70))
    pastel_sat_max = float(mood_cfg.get("pastel_sat_max", 0.35))
    pastel_v_min = float(mood_cfg.get("pastel_v_min", 0.80))

    # Vectorize palette
    shares = np.array([c.share for c in palette], dtype=np.float32)
    H = np.array([c.hsv[0] for c in palette], dtype=np.float32)
    S = np.array([c.hsv[1] for c in palette], dtype=np.float32)
    V = np.array([c.hsv[2] for c in palette], dtype=np.float32)

    shares = shares / max(1e-9, shares.sum())

    # Aggregate stats
    H_mean = circ_mean_deg(H, shares)
    H_spread = circ_dispersion_deg(H, shares)  # bigger = more diverse hues

    S_mean = float(np.sum(S * shares))
    V_mean = float(np.sum(V * shares))

    # Contrast-ish measures (weighted range)
    V_range = float(V.max() - V.min())
    S_range = float(S.max() - S.min())

    warm_share = float(np.sum(shares[[is_warm(h) for h in H]]))
    cool_share = 1.0 - warm_share

    # Neutral share: low saturation colors
    neutral_share = float(np.sum(shares[S < s_low]))

    explain: List[str] = []
    scores: Dict[str, float] = {}

    # ----------------------------
    # Tag rules (can coexist)
    # ----------------------------

    tags: List[str] = []

    # Warm / Cool tagging (relative)
    if warm_share >= 0.60:
        tags.append("warm")
        explain.append(f"warm_share={warm_share:.2f} >= 0.60")
    elif cool_share >= 0.60:
        tags.append("cool")
        explain.append(f"cool_share={cool_share:.2f} >= 0.60")

    # Dark / Bright tagging (absolute)
    if V_mean < v_dark:
        tags.append("dark")
        explain.append(f"V_mean={V_mean:.2f} < v_dark={v_dark:.2f}")
    elif V_mean > v_bright:
        tags.append("bright")
        explain.append(f"V_mean={V_mean:.2f} > v_bright={v_bright:.2f}")

    # Muted / Vibrant tagging
    if S_mean < s_low:
        tags.append("muted")
        explain.append(f"S_mean={S_mean:.2f} < s_low={s_low:.2f}")
    elif S_mean > s_high:
        tags.append("vibrant")
        explain.append(f"S_mean={S_mean:.2f} > s_high={s_high:.2f}")

    # Pastel tagging
    if (V_mean >= pastel_v_min) and (S_mean <= pastel_sat_max):
        tags.append("pastel")
        explain.append(f"pastel: V_mean={V_mean:.2f}>= {pastel_v_min:.2f} and S_mean={S_mean:.2f}<= {pastel_sat_max:.2f}")

    # ----------------------------
    # Primary mood scoring
    # ----------------------------
    # We'll compute a few mood scores and pick the max.
    # (This keeps it explainable but less brittle than if/else chains.)

    # Calming: cool-leaning + low-ish saturation + not too much hue diversity
    calming = 0.0
    calming += 0.6 * cool_share
    calming += 0.2 * (1.0 - min(1.0, S_mean / (s_high + 1e-9)))
    calming += 0.2 * (1.0 - min(1.0, H_spread / (calming_hue_spread + 1e-9)))
    if cool_share >= 0.60:
        explain.append("calming boost: cool-dominant")
    if H_spread <= calming_hue_spread:
        explain.append(f"calming boost: H_spread={H_spread:.1f} <= {calming_hue_spread:.1f}")

    # Energetic: warm-leaning + high saturation and/or strong contrast
    energetic = 0.0
    energetic += 0.6 * warm_share
    energetic += 0.2 * min(1.0, S_mean / (s_high + 1e-9))
    energetic += 0.2 * min(1.0, max(V_range, S_range) / (energetic_contrast + 1e-9))
    if max(V_range, S_range) >= energetic_contrast:
        explain.append(f"energetic boost: contrast={max(V_range,S_range):.2f} >= {energetic_contrast:.2f}")

    # Cozy: warm + mid brightness + mid saturation (not neon)
    cozy = 0.0
    cozy += 0.6 * warm_share
    cozy += 0.2 * (1.0 - abs(V_mean - 0.55) / 0.55)  # peak near 0.55
    cozy += 0.2 * (1.0 - abs(S_mean - 0.45) / 0.45)  # peak near 0.45

    # Minimal: lots of neutrals + low saturation + low contrast
    minimal = 0.0
    minimal += 0.5 * neutral_share
    minimal += 0.3 * (1.0 - min(1.0, S_mean / (s_low + 1e-9)))
    minimal += 0.2 * (1.0 - min(1.0, max(V_range, S_range) / 0.35))

    # Dramatic: darker + high contrast
    dramatic = 0.0
    dramatic += 0.6 * (1.0 - min(1.0, V_mean / (v_dark + 1e-9)))  # darker -> higher
    dramatic += 0.4 * min(1.0, max(V_range, S_range) / 0.35)

    # Pastel primary: pastel tag + gentle palette
    pastel = 0.0
    pastel += 0.7 if "pastel" in tags else 0.0
    pastel += 0.3 * (1.0 - min(1.0, max(V_range, S_range) / 0.35))

    # Vibrant primary: strong saturation + color diversity
    vibrant = 0.0
    vibrant += 0.6 * min(1.0, S_mean / (vibrant_sat + 1e-9))
    vibrant += 0.4 * min(1.0, H_spread / 90.0)

    scores = {
        "calming": float(calming),
        "energetic": float(energetic),
        "cozy": float(cozy),
        "minimal": float(minimal),
        "dramatic": float(dramatic),
        "pastel": float(pastel),
        "vibrant": float(vibrant),
    }

    primary = max(scores.items(), key=lambda kv: kv[1])[0]
    explain.insert(0, f"primary={primary} (scores={ {k: round(v,3) for k,v in scores.items()} })")

    # Deduplicate tags (preserve order)
    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]

    # Add some useful stat explanations
    explain.append(f"H_mean={H_mean:.1f}°, H_spread≈{H_spread:.1f}°")
    explain.append(f"S_mean={S_mean:.2f}, V_mean={V_mean:.2f}, neutral_share={neutral_share:.2f}")
    explain.append(f"V_range={V_range:.2f}, S_range={S_range:.2f}")

    return MoodResult(primary=primary, tags=tags, scores=scores, explain=explain)
