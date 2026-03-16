"""
topic_namer.py — Stage 8: Generative Topic Naming via Local/Cloud LLM

Generates short, abstract Thai category names (1–3 words) per cluster.
Default provider: Ollama (local — no API key required).

Provider routing (set LLM_PROVIDER env var or edit config.py):
  "ollama"    → Local Ollama server at TOPIC_NAMER_OLLAMA_URL (default).
                Recommended model: hf.co/scb10x/typhoon-v1.5-8b-instruct-gguf:Q4_K_M
  "mock"      → No API call. Returns deterministic placeholder names.
  "openai"    → OpenAI ChatCompletion (gpt-4o-mini default).
  "anthropic" → Anthropic Messages API (claude-3-haiku default).
  "gemini"    → Google Generative AI (gemini-1.5-flash default).

Falls back to mock on any exception so the pipeline is never blocked.

Prompt contract (enforced in system message):
  Input:  5–10 representative Thai customer questions from one cluster.
  Output: Exactly ONE short Thai category label, 1–3 words, NO punctuation.
          Never a question. Never a full sentence.
  Examples: "ปัญหาการเข้าระบบ", "เวลาเปิดปิดตลาด", "สอบถามราคาสินทรัพย์"
"""

import logging
import os
import random
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    OLLAMA_TIMEOUT,
    TOPIC_MAX_TOKENS,
    TOPIC_NAMER_API_KEY,
    TOPIC_NAMER_MODEL,
    TOPIC_NAMER_OLLAMA_URL,
    TOPIC_NAMER_PROVIDER,
    TOPIC_SAMPLE_SIZE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# ─── Shared prompt components ────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "คุณเป็นผู้เชี่ยวชาญด้านการจัดหมวดหมู่คำถามลูกค้าภาษาไทยในธุรกิจหลักทรัพย์และฟินเทค "
    "งานของคุณคือดูคำถามของลูกค้าหลายข้อแล้วตั้งชื่อหมวดหมู่ที่ครอบคลุมทุกคำถาม "
    "กฎเหล็ก: "
    "1. ตอบเป็นภาษาไทยเท่านั้น "
    "2. ชื่อหมวดหมู่ต้องสั้น 1-3 คำ ห้ามเกิน "
    "3. ห้ามตอบเป็นประโยคคำถาม (ห้ามมีคำว่า ไหม อย่างไร เท่าไร ฯลฯ) "
    "4. ห้ามใส่เครื่องหมายวรรคตอนท้าย "
    "5. ตอบเพียงชื่อหมวดหมู่เท่านั้น ไม่มีคำอธิบายเพิ่มเติม ไม่มีคำทักทาย ไม่มีข้อความอื่น "
    "ตัวอย่างคำตอบที่ถูกต้อง: ปัญหาการเข้าระบบ, เวลาเปิดปิดตลาด, สอบถามราคาสินทรัพย์, "
    "การฝากถอนเงิน, ค่าธรรมเนียมการซื้อขาย"
)


def _build_user_message(questions: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return (
        f"คำถามลูกค้าในกลุ่มนี้:\n{numbered}\n\n"
        "ชื่อหมวดหมู่ (ตอบเฉพาะชื่อหมวดหมู่เท่านั้น 1-3 คำ):"
    )


def _sample_questions(questions: list[str], k: int = TOPIC_SAMPLE_SIZE) -> list[str]:
    """Return up to k questions, shuffled for diversity."""
    k = min(k, len(questions))
    return random.sample(questions, k)


def _clean_llm_output(raw: str) -> str:
    """
    Strip any conversational filler that a chat model might prepend.
    Keeps only the first non-empty line, then trims trailing punctuation.
    """
    for line in raw.strip().splitlines():
        line = line.strip()
        # Skip lines that look like filler
        if not line:
            continue
        if any(
            line.startswith(p)
            for p in ("ขอบคุณ", "แน่นอน", "สวัสดี", "ี:", ":", "sure", "of course", "okay")
        ):
            continue
        # Strip trailing Thai/western punctuation
        while line and line[-1] in ":.!?,;ๆ ":
            line = line[:-1]
        return line
    return raw.strip()


# ─── Provider implementations ────────────────────────────────────────────────

def _name_via_ollama(questions: list[str]) -> str:
    """
    Call local Ollama server using the /api/generate endpoint via requests.
    Combines system prompt + user message into a single prompt string because
    /api/generate is a completion endpoint (not a chat endpoint).

    Uses:
      temperature=0.1  — near-deterministic output
      num_predict=20   — hard token cap to prevent verbose responses
      stop=["\n", ".", "ๆ"]  — cut at first newline/sentence end
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required: pip install requests")

    samples = _sample_questions(questions)
    # Combine system + user into one completion prompt
    full_prompt = (
        f"[INST] <<SYS>>\n{_SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{_build_user_message(samples)} [/INST]"
    )

    payload = {
        "model": TOPIC_NAMER_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": TOPIC_MAX_TOKENS,
            "stop": ["\n", "\r", ".", "!", "?"],
        },
    }

    try:
        resp = requests.post(
            TOPIC_NAMER_OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,  # Extended timeout for local LLMs
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()
        result = _clean_llm_output(raw)
        if not result:
            raise ValueError(f"Empty response from Ollama. Raw: {repr(raw)}")
        return result

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {TOPIC_NAMER_OLLAMA_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )


def _name_via_mock(questions: list[str], cluster_idx: int) -> str:
    """Mock provider — returns legible placeholder without any API call."""
    label = f"กลุ่มที่ {cluster_idx + 1}"
    logger.debug(f"Mock namer: cluster {cluster_idx} → '{label}'")
    return label


def _name_via_openai(questions: list[str]) -> str:
    """OpenAI ChatCompletion via official openai package (>=1.0.0)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai>=1.0.0")

    client = OpenAI(api_key=TOPIC_NAMER_API_KEY)
    samples = _sample_questions(questions)
    response = client.chat.completions.create(
        model=TOPIC_NAMER_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(samples)},
        ],
        max_tokens=TOPIC_MAX_TOKENS,
        temperature=0.1,
    )
    return _clean_llm_output(response.choices[0].message.content)


def _name_via_anthropic(questions: list[str]) -> str:
    """Anthropic Messages API via anthropic package (>=0.25.0)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic>=0.25.0")

    client = anthropic.Anthropic(api_key=TOPIC_NAMER_API_KEY)
    samples = _sample_questions(questions)
    message = client.messages.create(
        model=TOPIC_NAMER_MODEL,
        max_tokens=TOPIC_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(samples)}],
    )
    return _clean_llm_output(message.content[0].text)


def _name_via_gemini(questions: list[str]) -> str:
    """Google Generative AI via google-generativeai package (>=0.5.0)."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("pip install google-generativeai>=0.5.0")

    genai.configure(api_key=TOPIC_NAMER_API_KEY)
    model = genai.GenerativeModel(
        model_name=TOPIC_NAMER_MODEL,
        system_instruction=_SYSTEM_PROMPT,
    )
    samples = _sample_questions(questions)
    response = model.generate_content(
        _build_user_message(samples),
        generation_config=genai.GenerationConfig(
            max_output_tokens=TOPIC_MAX_TOKENS,
            temperature=0.1,
        ),
    )
    return _clean_llm_output(response.text)


# ─── Public interface ─────────────────────────────────────────────────────────

def name_cluster(questions: list[str], cluster_idx: int = 0) -> str:
    """
    Generate a short abstract Thai category name for a cluster.

    Args:
        questions:   All questions in the cluster (sampled internally).
        cluster_idx: 0-based cluster index — used by mock provider only.

    Returns:
        A 1–3 word Thai category label, e.g. "ปัญหาการเข้าระบบ".
        Falls back to mock on any exception to keep the pipeline running.
    """
    provider = TOPIC_NAMER_PROVIDER.lower()

    try:
        if provider == "ollama":
            return _name_via_ollama(questions)
        elif provider == "openai":
            return _name_via_openai(questions)
        elif provider == "anthropic":
            return _name_via_anthropic(questions)
        elif provider == "gemini":
            return _name_via_gemini(questions)
        else:
            return _name_via_mock(questions, cluster_idx)

    except Exception as exc:
        logger.warning(
            f"Topic namer (provider={provider!r}) failed for cluster {cluster_idx}: {exc}. "
            "Falling back to mock."
        )
        return _name_via_mock(questions, cluster_idx)


def build_namer_fn() -> Callable[[list[str], int], str]:
    """Return a ready-to-use naming function bound to the configured provider."""
    provider = TOPIC_NAMER_PROVIDER.lower()
    model = TOPIC_NAMER_MODEL
    endpoint = TOPIC_NAMER_OLLAMA_URL if provider == "ollama" else "N/A"
    logger.info(
        f"Topic namer: provider='{provider}', model='{model}'"
        + (f", url='{endpoint}'" if provider == "ollama" else "")
    )
    return name_cluster
