"""
umap_reducer.py — Stage 4.5: Dimensionality Reduction via UMAP

Reduces high-dimensional embeddings (e.g. 1024-dim from bge-m3) to a compact
low-dimensional representation (default 8-dim) before passing to HDBSCAN.

Why this matters:
  - HDBSCAN suffers heavily from the curse of dimensionality on 1024-dim vectors.
  - UMAP with cosine metric preserves neighbourhood structure much better than PCA.
  - min_dist=0.0 produces tighter clusters ideal for density-based algorithms.

Cache strategy:
  - Reduced embeddings are cached as a .npy file.
  - A companion JSON stores the UMAP params used; if params change, cache is
    invalidated automatically and UMAP is re-fit.
"""

import hashlib
import json
import logging
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    UMAP_CACHE_FILE,
    UMAP_MIN_DIST,
    UMAP_METRIC,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    UMAP_PARAMS_CACHE_FILE,
    UMAP_RANDOM_STATE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def _params_signature(
    n_points: int,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> dict:
    """Return a JSON-serialisable dict capturing all UMAP parameters + data size."""
    return {
        "n_points": n_points,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
    }


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    metric: str = UMAP_METRIC,
    random_state: int = UMAP_RANDOM_STATE,
    use_cache: bool = True,
    cache_file: str = UMAP_CACHE_FILE,
    params_cache_file: str = UMAP_PARAMS_CACHE_FILE,
) -> np.ndarray:
    """
    Stage 4.5: Reduce embedding dimensions with UMAP.

    Args:
        embeddings:       Raw float32 embeddings of shape (N, D).
        n_components:     Target low-dimensional space size.
        n_neighbors:      UMAP n_neighbors; higher = more global structure.
        min_dist:         UMAP min_dist; 0.0 = maximally tight clusters.
        metric:           Distance metric passed to UMAP (cosine recommended).
        random_state:     Seed for reproducibility.
        use_cache:        Whether to load/save reduced embeddings to disk.
        cache_file:       Path for the .npy cache of reduced embeddings.
        params_cache_file: Path for the JSON cache of UMAP parameters.

    Returns:
        np.ndarray of shape (N, n_components), dtype float32.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    current_params = _params_signature(
        n_points=len(embeddings),
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    # ── Cache hit check ───────────────────────────────────────────────────────
    if use_cache and os.path.isfile(cache_file) and os.path.isfile(params_cache_file):
        with open(params_cache_file, "r", encoding="utf-8") as f:
            cached_params = json.load(f)
        if cached_params == current_params:
            reduced = np.load(cache_file)
            logger.info(
                f"Stage 4.5: Loaded cached UMAP output from {cache_file} "
                f"(shape={reduced.shape})"
            )
            return reduced
        else:
            logger.info("Stage 4.5: UMAP params changed — re-fitting UMAP.")

    # ── Fit UMAP ──────────────────────────────────────────────────────────────
    logger.info(
        f"Stage 4.5: Fitting UMAP on {len(embeddings)} points "
        f"({embeddings.shape[1]}→{n_components} dims, "
        f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}) …"
    )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=False,   # faster when RAM is available
        verbose=False,
    )
    reduced = reducer.fit_transform(embeddings).astype(np.float32)

    logger.info(
        f"Stage 4.5 complete: UMAP reduced {embeddings.shape} → {reduced.shape}"
    )

    # ── Persist cache ─────────────────────────────────────────────────────────
    if use_cache:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, reduced)
        with open(params_cache_file, "w", encoding="utf-8") as f:
            json.dump(current_params, f, indent=2)
        logger.info(f"UMAP cache saved to {cache_file}")

    return reduced
