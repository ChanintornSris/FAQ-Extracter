"""
llm_extractor.py — Stage 3.5: LLM-Based FAQ Extraction from Conversation Batches

Uses Ollama (Typhoon/any local model) to read batches of customer-admin conversations
and extract canonical FAQ Q&A pairs from them.

Algorithm (MapReduce-style):
  1. Split raw conversations into batches of FAQ_EXTRACTION_BATCH_SIZE.
  2. For each batch → send to Ollama → parse JSON list of {question, answer}.
  3. Pool all extracted pairs across all batches.
  4. Cosine-similarity dedup (embedding-based) to remove near-identical questions.
  5. Return final DataFrame of {question, answer, row_id}.

The output of this stage feeds directly into Stage 4 (embedding) so that the rest
of the pipeline (UMAP → HDBSCAN → topic naming) operates on *canonical FAQ questions*,
not raw customer messages.
"""

import json
import logging
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import (
    FAQ_DEDUP_SIMILARITY_THRESHOLD,
    FAQ_EXTRACTION_BATCH_SIZE,
    FAQ_EXTRACTION_NUM_PREDICT,
    FAQ_PER_BATCH,
    FIELD_ANSWER,
    FIELD_CLEAN_QUESTION,
    FIELD_QUESTION,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    OLLAMA_TIMEOUT,
    TOPIC_NAMER_MODEL,
    TOPIC_NAMER_OLLAMA_URL,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# ─── Prompt templates ────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = (
    "คุณเป็น AI ที่ช่วยสกัด FAQ จากบทสนทนา คุณต้องตอบเป็น JSON array เท่านั้น ห้ามมีข้อความอื่น\n"
    "รูปแบบที่ต้องตอบ: [{\"question\": \"...\", \"answer\": \"...\"}]\n"
    "กฎ:\n"
    "1. สกัดเฉพาะคำถามที่ลูกค้าถามบ่อย ไม่ใช่คำถามเฉพาะบุคคล\n"
    "2. คำถามต้องเป็นภาษากลาง ครอบคลุม ตอบเป็น JSON เท่านั้น\n"
    "3. คำตอบต้องมาจาก admin ในบทสนทนา\n"
    "4. ห้ามสร้างคำตอบขึ้นมาเอง ใช้เฉพาะที่มีในบทสนทนา\n"
    "5. ห้ามมีข้อความอื่นนอกจาก JSON array"
)


def _format_conversations(conversations: list[dict]) -> str:
    """Format a list of conversation dicts into a readable text block."""
    lines = []
    for i, conv in enumerate(conversations, 1):
        q = str(conv.get("customer_message", conv.get("question", conv.get(FIELD_QUESTION, "")))).strip()
        a = str(conv.get("admin_reply", conv.get("answer", conv.get(FIELD_ANSWER, "")))).strip()
        if q and a:
            lines.append(f"[{i}] ลูกค้า: {q}")
            lines.append(f"    Admin: {a}")
    return "\n".join(lines)


def _parse_faq_json(raw: str) -> list[dict]:
    """
    Robustly parse the LLM's response into a list of {question, answer} dicts.
    Tries multiple strategies: direct parse → regex extraction → line-by-line.
    """
    if not raw:
        return []

    # Strategy 1: direct JSON parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return _validate_faq_list(data)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract JSON array from text using regex
    match = re.search(r'\[[\s\S]*?\]', raw)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate_faq_list(data)
        except json.JSONDecodeError:
            pass

    # Strategy 3: find individual JSON objects and wrap in array
    objects = re.findall(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', raw, re.DOTALL)
    if objects:
        try:
            data = [json.loads(obj) for obj in objects]
            return _validate_faq_list(data)
        except json.JSONDecodeError:
            pass

    logger.debug(f"Could not parse FAQ JSON. Raw response (first 200 chars): {raw[:200]}")
    return []


def _validate_faq_list(items: list) -> list[dict]:
    """Filter to valid {question, answer} dicts with non-empty strings."""
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if len(q) >= 5 and len(a) >= 5:
            valid.append({"question": q, "answer": a})
    return valid


def extract_faqs_from_batch(
    conversations: list[dict],
    model: str = TOPIC_NAMER_MODEL,
    ollama_url: str = TOPIC_NAMER_OLLAMA_URL,
    max_faqs: int = FAQ_PER_BATCH,
    num_predict: int = FAQ_EXTRACTION_NUM_PREDICT,
) -> list[dict]:
    """
    Send a batch of conversations to Ollama and extract FAQ Q&A pairs.

    Args:
        conversations: List of {"customer_message": ..., "admin_reply": ...} dicts.
        model:         Ollama model name.
        ollama_url:    Ollama /api/generate endpoint.
        max_faqs:      Maximum FAQ pairs to extract per batch.
        num_predict:   Max tokens for the LLM response.

    Returns:
        List of {"question": str, "answer": str} dicts. Empty list on failure.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    if not conversations:
        return []

    conv_text = _format_conversations(conversations)
    user_message = (
        f"บทสนทนาต่อไปนี้มีระหว่าง admin และลูกค้า\n"
        f"สกัดคู่ FAQ ที่พบบ่อยที่สุด {max_faqs} คู่ ตอบเป็น JSON array เท่านั้น:\n\n"
        f"{conv_text}\n\n"
        f"ตอบเป็น JSON array เท่านั้น ห้ามมีข้อความอื่น:"
    )

    # Llama-style instruction format (compatible with Typhoon)
    full_prompt = (
        f"[INST] <<SYS>>\n{_EXTRACT_SYSTEM}\n<</SYS>>\n\n"
        f"{user_message} [/INST]"
    )

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
            "stop": ["[INST]", "<<SYS>>"],
        },
    }

    try:
        resp = requests.post(ollama_url, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        faqs = _parse_faq_json(raw)
        logger.debug(f"Batch extracted {len(faqs)} FAQs from {len(conversations)} conversations.")
        return faqs

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Cannot reach Ollama at {ollama_url}. "
            "Ensure Ollama is running: `ollama serve`"
        )
        return []
    except Exception as exc:
        logger.warning(f"Batch extraction failed: {exc}")
        return []


def _embedding_dedup(
    faqs: list[dict],
    similarity_threshold: float = FAQ_DEDUP_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Remove near-duplicate FAQ pairs using cosine similarity on embedded questions.
    Keeps the pair with the longer answer when duplicates are found.

    Only runs if sentence-transformers is available (it always is in this pipeline).
    """
    if len(faqs) <= 1:
        return faqs

    from backend.embedding_service import encode_texts, l2_normalize

    questions = [f["question"] for f in faqs]
    try:
        raw_embs = encode_texts(questions, is_query=False, show_progress=False)
        norm_embs = l2_normalize(raw_embs)
    except Exception as exc:
        logger.warning(f"Embedding dedup failed, skipping: {exc}")
        return faqs

    sim_matrix = norm_embs @ norm_embs.T  # (N, N) cosine similarity

    kept = [True] * len(faqs)
    for i in range(len(faqs)):
        if not kept[i]:
            continue
        for j in range(i + 1, len(faqs)):
            if not kept[j]:
                continue
            if sim_matrix[i, j] >= similarity_threshold:
                # Keep the one with the longer answer
                if len(faqs[i]["answer"]) >= len(faqs[j]["answer"]):
                    kept[j] = False
                else:
                    kept[i] = False
                    break

    deduped = [f for f, k in zip(faqs, kept) if k]
    logger.info(
        f"Dedup: {len(faqs)} → {len(deduped)} FAQs "
        f"(removed {len(faqs) - len(deduped)} duplicates, threshold={similarity_threshold})"
    )
    return deduped


def extract_all_faqs(
    df: pd.DataFrame,
    batch_size: int = FAQ_EXTRACTION_BATCH_SIZE,
    max_faqs_per_batch: int = FAQ_PER_BATCH,
    model: str = TOPIC_NAMER_MODEL,
    ollama_url: str = TOPIC_NAMER_OLLAMA_URL,
    similarity_threshold: float = FAQ_DEDUP_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Stage 3.5: Full MapReduce FAQ extraction.

    1. Split df into batches of batch_size conversations.
    2. Call LLM on each batch to extract FAQ Q&A pairs.
    3. Pool all extracted pairs, run embedding-based dedup.
    4. Return a DataFrame with columns: question, clean_question, answer, row_id.

    Args:
        df:                    DataFrame with customer_message/question and admin_reply/answer.
        batch_size:            Conversations per LLM batch call.
        max_faqs_per_batch:    Max FAQ pairs per batch.
        model:                 Ollama model name.
        ollama_url:            Ollama endpoint.
        similarity_threshold:  Cosine threshold for dedup (0–1).

    Returns:
        DataFrame suitable for passing to generate_embeddings().
        Empty DataFrame if extraction fails completely.
    """
    records = df.to_dict(orient="records")
    n_batches = max(1, (len(records) + batch_size - 1) // batch_size)

    logger.info(
        f"Stage 3.5: LLM FAQ extraction — "
        f"{len(records)} conversations → {n_batches} batches (size={batch_size})"
    )

    all_faqs: list[dict] = []
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(records))
        batch = records[start:end]

        logger.info(f"  Batch {batch_idx + 1}/{n_batches}: rows {start}–{end - 1}")
        batch_faqs = extract_faqs_from_batch(
            batch, model=model, ollama_url=ollama_url,
            max_faqs=max_faqs_per_batch,
        )
        all_faqs.extend(batch_faqs)
        logger.info(f"  → {len(batch_faqs)} FAQs extracted (total so far: {len(all_faqs)})")

    if not all_faqs:
        logger.error(
            "Stage 3.5: No FAQs extracted. Check that Ollama is running and the model is loaded."
        )
        return pd.DataFrame(columns=["question", "clean_question", "answer", "row_id"])

    logger.info(f"Stage 3.5: Pooled {len(all_faqs)} raw FAQ pairs across all batches.")

    # Embedding-based deduplication
    all_faqs = _embedding_dedup(all_faqs, similarity_threshold=similarity_threshold)

    # Build output DataFrame compatible with generate_embeddings()
    faq_df = pd.DataFrame(all_faqs)  # columns: question, answer
    faq_df["clean_question"] = faq_df["question"]   # already canonical
    faq_df["row_id"] = range(len(faq_df))

    logger.info(
        f"Stage 3.5 complete: {len(faq_df)} unique FAQ pairs ready for embedding."
    )
    return faq_df
