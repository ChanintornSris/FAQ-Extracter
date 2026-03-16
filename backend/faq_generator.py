"""
faq_generator.py — Stages 8–10: FAQ Group Generation

Groups pre-extracted canonical FAQ pairs (from llm_extractor.py) by cluster.
Each cluster becomes a named group with a full list of {question, answer} FAQs.

Stage 8: LLM topic naming via topic_namer.name_cluster().
Stage 9: Sort FAQs within group by centroid proximity (most representative first).
Stage 10: Assemble final group output.
"""

import logging
import sys
import os
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import (
    FIELD_ANSWER,
    FIELD_CLEAN_QUESTION,
    FIELD_CLUSTER_ID,
    FIELD_QUESTION,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVEL,
    REPRESENTATIVE_Q_COUNT,
)
from backend.embedding_service import l2_normalize

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def _sort_faqs_by_centrality(
    faqs: list[dict],
    cluster_embs: np.ndarray,
) -> list[dict]:
    """
    Sort FAQ pairs so the most representative (closest to centroid) appear first.
    This makes the top entries in each group the "canonical" FAQs for that topic.
    """
    if len(faqs) <= 1:
        return faqs

    norm_embs = l2_normalize(cluster_embs)
    centroid = norm_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    sims = norm_embs @ centroid  # cosine similarity to centroid

    sorted_pairs = sorted(zip(sims, faqs), key=lambda x: -x[0])
    return [pair for _, pair in sorted_pairs]


def generate_groups(
    df: pd.DataFrame,
    full_embeddings: np.ndarray,
    name_fn: Callable[[list[str], int], str],
) -> list[dict[str, Any]]:
    """
    Produce the canonical group output list from clustered extracted FAQ pairs.

    Args:
        df:              DataFrame with: question, clean_question, answer, cluster_id.
                         Each row is a pre-extracted FAQ pair (from llm_extractor).
        full_embeddings: Full-dim (1024-dim) embeddings for each row in df.
        name_fn:         Callable(questions, cluster_idx) → str for LLM group naming.

    Returns:
        List of group dicts sorted by total_faqs descending:
        [
          {
            "group_id":   int,
            "group_name": str,        ← LLM-generated (e.g. "ปัญหาการเข้าระบบ MT5")
            "total_faqs": int,
            "faqs": [
              {"question": str, "answer": str},   ← real Q&A pairs
              ...
            ],
            # Legacy compat fields for the existing frontend/FAISS:
            "representative_questions": list[str],
            "suggested_admin_reply":    str,
            "total_questions":          int,
            "support_count":            int,
            "faq_question":             str,
            "faq_answer":               str,
            "cluster_id":               int,
          }
        ]
    """
    groups: list[dict] = []
    unique_clusters = sorted([c for c in df[FIELD_CLUSTER_ID].unique() if c != -1])

    logger.info(f"Stages 8–10: Generating groups for {len(unique_clusters)} clusters …")

    for idx, cluster_id in enumerate(unique_clusters):
        mask = df[FIELD_CLUSTER_ID] == cluster_id
        cluster_df = df[mask].copy()
        cluster_embs = full_embeddings[mask.values]

        # Collect Q&A pairs for this cluster
        raw_faqs = [
            {"question": row[FIELD_QUESTION], "answer": row[FIELD_ANSWER]}
            for _, row in cluster_df.iterrows()
        ]
        questions = cluster_df[FIELD_CLEAN_QUESTION].tolist()

        # Stage 8: LLM-generated abstract group name
        group_name = name_fn(questions, idx)

        # Stage 9: Sort FAQs by centrality (most representative first)
        sorted_faqs = _sort_faqs_by_centrality(raw_faqs, cluster_embs)

        # Representative questions = top REPRESENTATIVE_Q_COUNT for FAISS
        representative_questions = [f["question"] for f in sorted_faqs[:REPRESENTATIVE_Q_COUNT]]
        # Suggested reply = answer of the most representative FAQ
        suggested_admin_reply = sorted_faqs[0]["answer"] if sorted_faqs else ""

        groups.append(
            {
                "group_id": int(cluster_id),
                "group_name": group_name,
                "total_faqs": len(sorted_faqs),
                "faqs": sorted_faqs,
                # Legacy / backwards-compat fields (frontend + FAISS expect these)
                "representative_questions": representative_questions,
                "suggested_admin_reply": suggested_admin_reply,
                "total_questions": len(sorted_faqs),
                "support_count": len(sorted_faqs),
                "faq_question": group_name,
                "faq_answer": suggested_admin_reply,
                "cluster_id": int(cluster_id),
            }
        )

    # Sort by group size descending
    groups.sort(key=lambda x: x["total_faqs"], reverse=True)

    # Re-assign sequential group_ids after sort
    for new_id, g in enumerate(groups):
        g["group_id"] = new_id

    logger.info(
        f"Stages 8–10 complete: {len(groups)} groups, "
        f"{sum(g['total_faqs'] for g in groups)} total FAQ pairs."
    )
    return groups


def generate_faqs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    """Backwards-compat wrapper using the mock topic namer."""
    from backend.topic_namer import build_namer_fn
    return generate_groups(df, embeddings, build_namer_fn())
