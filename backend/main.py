"""
main.py — Pipeline Orchestrator v3 (LLM Batch FAQ Extraction + UMAP + Group Naming)

Pipeline stages:
  1.    Data Ingestion
  2.    Text Cleaning
  3.    Question Filtering
  3.5   LLM Batch FAQ Extraction  ← NEW (Ollama/Typhoon → canonical Q&A pairs)
  4.    Embed extracted FAQ questions (BAAI/bge-m3, 1024-dim)
  4.5   UMAP Reduction (1024→8)
  5.    Semantic Deduplication (on extracted FAQ embeddings)
  6.    HDBSCAN Clustering
  7.    Cluster Quality Filtering
  8.    LLM Group Naming (Ollama/Typhoon → short Thai category name)
  9-10. Group Assembly (Q&A pairs → grouped output with full FAQ lists)
  11.   FAISS Search Index
  12.   API Server (--serve flag)
  13.   Analytics

Usage:
  python backend/main.py --input data/conversations.json
  python backend/main.py --input data/conversations.json --serve
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import (
    ANALYTICS_OUTPUT_FILE,
    DEFAULT_INPUT_FILE,
    EMBEDDINGS_CACHE_FILE,
    EMBEDDINGS_IDS_CACHE_FILE,
    FAQ_EXTRACTION_BATCH_SIZE,
    FAQ_OUTPUT_FILE,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    API_HOST,
    API_PORT,
    UMAP_CACHE_FILE,
    UMAP_PARAMS_CACHE_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def run_pipeline(input_file: str) -> dict:
    """
    Execute all pipeline stages and return API state dict.

    Returns:
        {
            "faq_index": FAQSearchIndex,
            "groups": list[dict],
            "analytics": dict,
            "valid_questions": list[str],
            "valid_embeddings": np.ndarray,
        }
    """
    import numpy as np

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("FAQ Mining Pipeline v3 — LLM Extraction + UMAP + Group Naming")
    logger.info("=" * 70)

    # ── Stage 1: Data Ingestion ───────────────────────────────────────────────
    from backend.data_loader import load_dataset
    raw_df = load_dataset(input_file)

    # ── Stage 2: Text Cleaning ────────────────────────────────────────────────
    from backend.text_cleaner import clean_questions
    cleaned_df = clean_questions(raw_df)

    # ── Stage 3: Question Filtering ───────────────────────────────────────────
    from backend.question_filter import filter_questions
    valid_df = filter_questions(cleaned_df)

    if len(valid_df) == 0:
        raise RuntimeError("No valid questions after filtering. Please check your dataset.")

    # ── Stage 3.5: LLM Batch FAQ Extraction ──────────────────────────────────
    # Convert filtered Q&A pairs → canonical FAQ pairs via Ollama (Typhoon)
    from backend.llm_extractor import extract_all_faqs
    faq_df = extract_all_faqs(
        valid_df,
        batch_size=FAQ_EXTRACTION_BATCH_SIZE,
    )

    if len(faq_df) == 0:
        raise RuntimeError(
            "No FAQ pairs extracted. "
            "Check that Ollama is running and the Typhoon model is loaded. "
            "Run: ollama serve && ollama pull <model-name>"
        )

    logger.info(f"Stage 3.5 complete: {len(faq_df)} canonical FAQ pairs extracted.")

    # ── Stage 4: Embed Extracted FAQ Questions (bge-m3) ───────────────────────
    # Note: faq_df already has 'clean_question' column set by extract_all_faqs()
    from backend.embedding_service import generate_embeddings, l2_normalize
    full_embeddings, faq_df = generate_embeddings(
        faq_df,
        use_cache=True,
        cache_file=EMBEDDINGS_CACHE_FILE,
        ids_cache_file=EMBEDDINGS_IDS_CACHE_FILE,
    )

    # ── Stage 4.5: UMAP Dimensionality Reduction ──────────────────────────────
    from backend.umap_reducer import reduce_dimensions
    reduced_embeddings = reduce_dimensions(
        full_embeddings,
        use_cache=True,
        cache_file=UMAP_CACHE_FILE,
        params_cache_file=UMAP_PARAMS_CACHE_FILE,
    )

    # ── Stage 5: Semantic Dedup of Extracted FAQs ─────────────────────────────
    from backend.deduplication import deduplicate
    unique_faq_df, full_faq_with_flags = deduplicate(faq_df, full_embeddings)

    unique_mask = ~full_faq_with_flags["is_duplicate"].values
    unique_full_embs = full_embeddings[unique_mask]
    unique_reduced_embs = reduced_embeddings[unique_mask]

    if len(unique_faq_df) == 0:
        raise RuntimeError("All extracted FAQs were deduplicated. Lower FAQ_DEDUP_SIMILARITY_THRESHOLD.")

    # ── Stage 6: HDBSCAN Clustering ───────────────────────────────────────────
    from backend.clustering import run_clustering, filter_clusters
    clustered_faq_df = run_clustering(unique_faq_df, unique_reduced_embs)

    # ── Stage 7: Cluster Quality Filtering ────────────────────────────────────
    clustered_faq_df = filter_clusters(clustered_faq_df, unique_reduced_embs)

    n_groups = len([c for c in clustered_faq_df["cluster_id"].unique() if c != -1])
    logger.info(f"Clustering: {n_groups} FAQ groups identified.")

    # ── Stages 8–10: Group Generation (LLM naming + full FAQ lists) ───────────
    from backend.topic_namer import build_namer_fn
    from backend.faq_generator import generate_groups
    name_fn = build_namer_fn()
    groups = generate_groups(clustered_faq_df, unique_full_embs, name_fn)

    if not groups:
        logger.warning(
            "No groups generated. Try lowering CLUSTER_MIN_CLUSTER_SIZE or "
            "increasing FAQ_EXTRACTION_BATCH_SIZE."
        )

    # ── Stage 11: Build FAISS Search Index ────────────────────────────────────
    from backend.search_index import FAQSearchIndex
    faq_index = FAQSearchIndex()
    if groups:
        faq_index.build(groups)
        faq_index.save()

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(FAQ_OUTPUT_FILE), exist_ok=True)
    with open(FAQ_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"total_groups": len(groups), "groups": groups}, f, ensure_ascii=False, indent=2)
    logger.info(f"FAQ groups saved to: {FAQ_OUTPUT_FILE}")

    # ── Stage 13: Analytics ───────────────────────────────────────────────────
    from backend.analytics import generate_analytics
    analytics = generate_analytics(raw_df, full_faq_with_flags, unique_faq_df, clustered_faq_df, groups)

    os.makedirs(os.path.dirname(ANALYTICS_OUTPUT_FILE), exist_ok=True)
    with open(ANALYTICS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2)
    logger.info(f"Analytics saved to: {ANALYTICS_OUTPUT_FILE}")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s | {len(groups)} groups | "
        f"{sum(g.get('total_faqs', 0) for g in groups)} total FAQ pairs"
    )
    logger.info("=" * 70)

    # Expose canonical FAQ questions for /similar_questions endpoint
    valid_questions = unique_faq_df["clean_question"].tolist()
    valid_embeddings = l2_normalize(unique_full_embs)

    return {
        "faq_index": faq_index,
        "groups": groups,
        "faqs": groups,          # backwards compat
        "analytics": analytics,
        "valid_questions": valid_questions,
        "valid_embeddings": valid_embeddings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="FAQ Mining Pipeline v3 — LLM-extracted Q&A pairs, UMAP, HDBSCAN groups."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", type=str, default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    args = parser.parse_args()

    state = run_pipeline(args.input)

    if args.serve:
        import subprocess

        def kill_port(port: int):
            try:
                if sys.platform == "win32":
                    out = subprocess.check_output(
                        f"netstat -ano | findstr :{port}", shell=True, text=True
                    )
                    for line in out.strip().split("\n"):
                        if f":{port}" in line and "LISTENING" in line:
                            pid = line.strip().split()[-1]
                            subprocess.run(
                                f"taskkill /F /PID {pid}", shell=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            )
                else:
                    subprocess.run(
                        f"lsof -ti:{port} | xargs kill -9", shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
            except Exception:
                pass

        kill_port(args.port)

        from backend.api import app, set_pipeline_state
        set_pipeline_state(
            faq_index=state["faq_index"],
            faqs=state["groups"],
            analytics=state["analytics"],
            valid_questions=state["valid_questions"],
            valid_embeddings=state["valid_embeddings"],
        )
        import uvicorn
        logger.info(f"Starting API server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
