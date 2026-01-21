from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ----------------------------
# Data container for results
# ----------------------------

@dataclass
class ClusterResult:
    k_chosen: int
    centers_lab: np.ndarray      # (K, 3) float32
    centers_hsv: np.ndarray      # (K, 3) float32  (H in deg [0,360), S,V in [0,1])
    weights: np.ndarray          # (K,) float32      sums to 1
    labels: np.ndarray           # (N,) int32        label per sampled pixel
    kept_idx: np.ndarray         # (K,) int64        indices of clusters kept (after pruning/merging)
    debug: dict                  # misc diagnostics


# ----------------------------
# Helpers: distance + stats
# ----------------------------

def delta_e76(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """
    CIE76 DeltaE distance (Euclidean in Lab). Fast and good enough for this use-case.
    """
    d = lab_a - lab_b
    return float(np.sqrt(np.dot(d, d)))


def pairwise_deltae(centers_lab: np.ndarray) -> np.ndarray:
    """
    Pairwise DeltaE matrix between centers, shape (K, K).
    """
    # Efficient pairwise euclidean distances
    X = centers_lab.astype(np.float32)
    # (K,1,3) - (1,K,3) -> (K,K,3)
    diffs = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diffs * diffs, axis=-1))
    return D


def compute_cluster_means(
    labels: np.ndarray,
    samples_lab: np.ndarray,
    samples_hsv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute means and weights given labels.
    Returns: centers_lab (K,3), centers_hsv (K,3), weights (K,)
    """
    labels = labels.astype(np.int32)
    K = int(labels.max()) + 1
    N = labels.shape[0]

    centers_lab = np.zeros((K, 3), dtype=np.float32)
    centers_hsv = np.zeros((K, 3), dtype=np.float32)
    counts = np.zeros((K,), dtype=np.int64)

    for k in range(K):
        idx = np.where(labels == k)[0]
        counts[k] = idx.size
        if idx.size == 0:
            continue
        centers_lab[k] = samples_lab[idx].mean(axis=0)
        centers_hsv[k] = mean_hsv(samples_hsv[idx])

    weights = (counts / max(1, N)).astype(np.float32)
    return centers_lab, centers_hsv, weights


def mean_hsv(hsv: np.ndarray) -> np.ndarray:
    """
    Compute mean HSV where H is circular (degrees), S and V are linear.
    hsv: (n,3) with H degrees [0,360)
    """
    h = np.deg2rad(hsv[:, 0].astype(np.float32))
    s = hsv[:, 1].astype(np.float32)
    v = hsv[:, 2].astype(np.float32)

    # Circular mean for hue
    x = np.cos(h).mean()
    y = np.sin(h).mean()
    mean_h = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0

    return np.array([mean_h, float(s.mean()), float(v.mean())], dtype=np.float32)


# ----------------------------
# K selection + kmeans
# ----------------------------

def choose_k(
    samples_lab: np.ndarray,
    k_candidates: Sequence[int] = (4, 5, 6),
    *,
    seed: int = 42,
    max_points_for_score: int = 12000,
) -> Tuple[int, dict]:
    """
    Pick K using silhouette score on a subsample (no labels needed).
    Returns (best_k, debug_info)
    """
    X = samples_lab.astype(np.float32)
    rng = np.random.default_rng(seed)

    # silhouette can be expensive: subsample
    if X.shape[0] > max_points_for_score:
        idx = rng.choice(X.shape[0], size=max_points_for_score, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    scores = {}
    best_k = None
    best_score = -1.0

    for k in k_candidates:
        if k < 2:
            continue
        km = KMeans(n_clusters=int(k), n_init="auto", random_state=seed)
        labels = km.fit_predict(X_eval)
        # silhouette requires >1 cluster label present
        if len(np.unique(labels)) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(X_eval, labels))
        scores[int(k)] = score
        if score > best_score:
            best_score = score
            best_k = int(k)

    if best_k is None:
        best_k = int(k_candidates[0])

    debug = {"silhouette_scores": scores, "chosen_k": best_k}
    return best_k, debug


def run_kmeans(
    samples_lab: np.ndarray,
    k: int,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit kmeans on Lab samples. Returns (labels, centers_lab).
    """
    X = samples_lab.astype(np.float32)
    km = KMeans(n_clusters=int(k), n_init="auto", random_state=seed)
    labels = km.fit_predict(X).astype(np.int32)
    centers_lab = km.cluster_centers_.astype(np.float32)
    return labels, centers_lab


# ----------------------------
# Merge + prune
# ----------------------------

def merge_close_clusters(
    labels: np.ndarray,
    centers_lab: np.ndarray,
    samples_lab: np.ndarray,
    samples_hsv: np.ndarray,
    *,
    deltaE_merge: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Iteratively merge clusters whose Lab centers are within deltaE_merge.
    After merging, recompute means from samples (more accurate than averaging centers).
    Returns updated (labels, centers_lab, centers_hsv, weights, debug)
    """
    labels = labels.astype(np.int32)
    debug = {"merges": []}

    centers_lab, centers_hsv, weights = compute_cluster_means(
        labels, samples_lab, samples_hsv
    )

    while True:
        K = int(labels.max()) + 1
        if K <= 1:
            break

        centers_lab_cur, centers_hsv_cur, weights_cur = compute_cluster_means(
            labels, samples_lab, samples_hsv
        )

        D = pairwise_deltae(centers_lab_cur)
        np.fill_diagonal(D, np.inf)

        min_val = float(np.min(D))
        if not np.isfinite(min_val) or min_val > float(deltaE_merge):
            # no more merges needed
            centers_lab, centers_hsv, weights = centers_lab_cur, centers_hsv_cur, weights_cur
            break

        # Merge the closest pair (i, j): relabel j -> i (then compact labels)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        debug["merges"].append({"pair": (int(i), int(j)), "deltaE": min_val})

        labels = np.where(labels == j, i, labels)

        # Compact labels to 0..K-1 (important after merges)
        labels = relabel_compact(labels)

    return labels, centers_lab, centers_hsv, weights, debug


def prune_small_clusters(
    labels: np.ndarray,
    samples_lab: np.ndarray,
    samples_hsv: np.ndarray,
    *,
    min_share: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Drop clusters smaller than min_share of sampled pixels.
    Returns updated (labels, centers_lab, centers_hsv, weights, kept_idx, debug)
    """
    labels = labels.astype(np.int32)
    debug = {}

    centers_lab, centers_hsv, weights = compute_cluster_means(labels, samples_lab, samples_hsv)
    keep = np.where(weights >= float(min_share))[0].astype(np.int64)

    if keep.size == 0:
        # keep at least the largest cluster
        keep = np.array([int(np.argmax(weights))], dtype=np.int64)

    debug["kept_clusters"] = keep.tolist()
    debug["dropped_clusters"] = [int(i) for i in range(len(weights)) if i not in set(keep.tolist())]

    # Map old labels -> new labels
    mapping = {int(old): new for new, old in enumerate(keep.tolist())}
    new_labels = np.full_like(labels, fill_value=-1)

    for old, new in mapping.items():
        new_labels[labels == old] = new

    # If anything was dropped, those pixels become -1; assign them to nearest kept center (Lab)
    if np.any(new_labels == -1):
        kept_centers_lab = centers_lab[keep]
        dropped_idx = np.where(new_labels == -1)[0]
        X = samples_lab[dropped_idx].astype(np.float32)

        # nearest center
        dists = ((X[:, None, :] - kept_centers_lab[None, :, :]) ** 2).sum(axis=-1)
        nearest = np.argmin(dists, axis=1).astype(np.int32)
        new_labels[dropped_idx] = nearest

    new_labels = new_labels.astype(np.int32)

    # Recompute stats after reassignment
    centers_lab2, centers_hsv2, weights2 = compute_cluster_means(
        new_labels, samples_lab, samples_hsv
    )

    return new_labels, centers_lab2, centers_hsv2, weights2, keep, debug


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    """
    Remap labels so they are contiguous 0..K-1 in order of appearance.
    """
    labels = labels.astype(np.int32)
    uniq = np.unique(labels)
    mapping = {int(u): i for i, u in enumerate(uniq.tolist())}
    out = np.array([mapping[int(x)] for x in labels], dtype=np.int32)
    return out


# ----------------------------
# Main entry point
# ----------------------------

def cluster_colors(
    samples_lab: np.ndarray,
    samples_hsv: np.ndarray,
    *,
    k_candidates: Sequence[int] = (4, 5, 6),
    seed: int = 42,
    deltaE_merge: float = 6.0,
    min_share: float = 0.015,
) -> ClusterResult:
    """
    High-level clustering stage:
      1) choose K (silhouette)
      2) run kmeans on Lab
      3) merge close clusters by deltaE threshold
      4) prune tiny clusters
      5) recompute final centers + weights
    """
    debug_all = {}

    k, dbg_k = choose_k(samples_lab, k_candidates=k_candidates, seed=seed)
    debug_all.update({"k_selection": dbg_k})

    labels, centers_lab = run_kmeans(samples_lab, k=k, seed=seed)

    labels, centers_lab, centers_hsv, weights, dbg_merge = merge_close_clusters(
        labels,
        centers_lab,
        samples_lab=samples_lab,
        samples_hsv=samples_hsv,
        deltaE_merge=deltaE_merge,
    )
    debug_all.update({"merge": dbg_merge})

    labels, centers_lab, centers_hsv, weights, kept_idx, dbg_prune = prune_small_clusters(
        labels,
        samples_lab=samples_lab,
        samples_hsv=samples_hsv,
        min_share=min_share,
    )
    debug_all.update({"prune": dbg_prune})

    # Final compact relabel
    labels = relabel_compact(labels)
    centers_lab, centers_hsv, weights = compute_cluster_means(labels, samples_lab, samples_hsv)

    return ClusterResult(
        k_chosen=int(k),
        centers_lab=centers_lab.astype(np.float32),
        centers_hsv=centers_hsv.astype(np.float32),
        weights=weights.astype(np.float32),
        labels=labels.astype(np.int32),
        kept_idx=kept_idx.astype(np.int64),
        debug=debug_all,
    )
