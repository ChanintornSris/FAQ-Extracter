"""
config.py — Global Configuration File
All tunable variables, paths, and constants for the FAQ Mining System.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

# Input data
DEFAULT_INPUT_FILE = os.path.join(DATA_DIR, "conversations.json")

# Output files
FAQ_OUTPUT_FILE = os.path.join(DATA_DIR, "faqs.json")
ANALYTICS_OUTPUT_FILE = os.path.join(DATA_DIR, "analytics_report.json")
EMBEDDINGS_CACHE_FILE = os.path.join(DATA_DIR, "embeddings_cache.npy")
EMBEDDINGS_IDS_CACHE_FILE = os.path.join(DATA_DIR, "embeddings_ids_cache.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_FILE = os.path.join(DATA_DIR, "faq_index_meta.json")

# ─────────────────────────────────────────────
# STAGE 1 — DATA INGESTION
# ─────────────────────────────────────────────
# Expected field names in the raw dataset
RAW_CUSTOMER_FIELD = "customer_message"
RAW_ADMIN_FIELD = "admin_reply"

# Normalized field names used internally
FIELD_QUESTION = "question"
FIELD_ANSWER = "answer"
FIELD_CLEAN_QUESTION = "clean_question"
FIELD_CLUSTER_ID = "cluster_id"
FIELD_IS_DUPLICATE = "is_duplicate"
FIELD_CANONICAL_ID = "canonical_id"
FIELD_DUPLICATE_COUNT = "duplicate_count"

# ─────────────────────────────────────────────
# STAGE 2 — TEXT CLEANING
# ─────────────────────────────────────────────
# Thai greeting / filler phrases to strip
THAI_FILLER_PHRASES = [
    "สวัสดีครับ",
    "สวัสดีค่ะ",
    "สวัสดี",
    "ขอถามหน่อย",
    "ขอถามหน่อยครับ",
    "ขอถามหน่อยค่ะ",
    "สอบถามครับ",
    "สอบถามค่ะ",
    "สอบถาม",
    "มีเรื่องสอบถาม",
    "มีเรื่องสอบถามครับ",
    "มีเรื่องสอบถามค่ะ",
    "ขอสอบถามหน่อยครับ",
    "ขอสอบถามหน่อยค่ะ",
    "ขอสอบถาม",
    "รบกวนสอบถามครับ",
    "รบกวนสอบถามค่ะ",
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
]

# ─────────────────────────────────────────────
# STAGE 3 — QUESTION FILTERING
# ─────────────────────────────────────────────
MIN_QUESTION_LENGTH = 5       # Minimum characters after cleaning
MAX_QUESTION_LENGTH = 1000    # Maximum characters (to reject ultra-long noise)

# Regex patterns for emoji detection
EMOJI_PATTERN = (
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\\U0001f926-\\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+"
)

# ─────────────────────────────────────────────
# STAGE 4 — EMBEDDING (BAAI/bge-m3)
# ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32             # bge-m3 is larger; use smaller batches
EMBEDDING_DIMENSION = 1024            # bge-m3 output dimension
EMBEDDING_DEVICE = "cpu"              # Set to "cuda" if GPU available
EMBEDDING_SHOW_PROGRESS = True
# bge-m3 supports three retrieval methods; we use dense (default).
# Set to True to prepend instruction prefixes (recommended for asymmetric retrieval).
EMBEDDING_USE_INSTRUCTION = True
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_PASSAGE_PREFIX = ""         # bge-m3 passages need no prefix

# ─────────────────────────────────────────────
# STAGE 4.5 — UMAP DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────
UMAP_N_COMPONENTS = 8          # Target dimensionality (5–15 works well for HDBSCAN)
UMAP_N_NEIGHBORS = 15          # Controls local vs global structure balance
UMAP_MIN_DIST = 0.0            # 0.0 = tighter clusters (better for HDBSCAN)
UMAP_METRIC = "cosine"         # Use cosine on high-dim embeddings
UMAP_RANDOM_STATE = 42
UMAP_CACHE_FILE = os.path.join(DATA_DIR, "umap_reduced_cache.npy")
UMAP_PARAMS_CACHE_FILE = os.path.join(DATA_DIR, "umap_params_cache.json")

# ─────────────────────────────────────────────
# STAGE 3.5 — LLM BATCH FAQ EXTRACTION
# ─────────────────────────────────────────────
# How many conversations to send to Ollama per LLM call.
# Lower = more focused extraction; Higher = fewer API calls.
FAQ_EXTRACTION_BATCH_SIZE = 10
# Max FAQ pairs to extract per batch (LLM is instructed to return at most this many).
FAQ_PER_BATCH = 5
# Max tokens for the LLM to produce (JSON output needs more room than naming).
FAQ_EXTRACTION_NUM_PREDICT = 800
# Cosine similarity above which two extracted FAQ questions are considered duplicates.
# More relaxed than raw-question dedup (0.99) because extraction already canonicalises.
FAQ_DEDUP_SIMILARITY_THRESHOLD = 0.80

# ─────────────────────────────────────────────
# STAGE 5 — SEMANTIC DEDUPLICATION (of extracted FAQs)
# ─────────────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.95   # Near-exact match cutoff in embedding space
DEDUP_CHUNK_SIZE = 5000             # Process dedup in chunks for large datasets

# ─────────────────────────────────────────────
# STAGE 6 — CLUSTERING (HDBSCAN on UMAP output)
# ─────────────────────────────────────────────
CLUSTER_MIN_CLUSTER_SIZE = 2        # Minimum points to form a cluster
CLUSTER_MIN_SAMPLES = 2             # Controls how conservative HDBSCAN is
CLUSTER_METRIC = "euclidean"        # Euclidean on UMAP-reduced space
CLUSTER_SELECTION_METHOD = "eom"    # "eom" or "leaf"

# ─────────────────────────────────────────────
# STAGE 7 — CLUSTER QUALITY FILTERING
# ─────────────────────────────────────────────
CLUSTER_MIN_SIZE_THRESHOLD = 2      # Clusters smaller than this are discarded
CLUSTER_MAX_INTRA_DISTANCE = 2.0    # Max avg pairwise L2 distance in UMAP space

# ─────────────────────────────────────────────
# STAGE 8 — LLM TOPIC NAMING
# ─────────────────────────────────────────────
# Provider: "ollama" | "mock" | "openai" | "anthropic" | "gemini"
TOPIC_NAMER_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
TOPIC_NAMER_API_KEY = os.environ.get("LLM_API_KEY", "")           # Not needed for Ollama
TOPIC_NAMER_MODEL = os.environ.get("LLM_MODEL", "scb10x/llama3.1-typhoon2-8b-instruct")
TOPIC_NAMER_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
_timeout_env = os.environ.get("OLLAMA_TIMEOUT", "10800")
OLLAMA_TIMEOUT = None if _timeout_env.strip().lower() == "none" else int(_timeout_env)
TOPIC_SAMPLE_SIZE = 2               # How many questions to sample per cluster
TOPIC_MAX_TOKENS = 100               # LLM max tokens / num_predict for category name

# ─────────────────────────────────────────────
# STAGE 10 — REPRESENTATIVE QUESTIONS
# ─────────────────────────────────────────────
REPRESENTATIVE_Q_COUNT = 5          # Representative questions to surface per group

ANSWER_SIMILARITY_THRESHOLD = 0.70  # If max answer similarity < this → summarize
ANSWER_MIN_LENGTH = 5               # Minimum answer length in characters

# ─────────────────────────────────────────────
# STAGE 11 — FAISS SEARCH INDEX
# ─────────────────────────────────────────────
FAISS_TOP_K_DEFAULT = 5             # Default number of results per query
FAISS_USE_GPU = False               # Set True if faiss-gpu is installed
SEARCH_MAX_TOP_K = 50               # Hard cap on top_k

# ─────────────────────────────────────────────
# STAGE 12 — API SERVER
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True                   # Set False in production
API_TITLE = "FAQ Mining System API"
API_VERSION = "2.0.0"
API_DESCRIPTION = "Production-grade FAQ extraction from Thai customer support conversations."

CORS_ORIGINS = ["*"]  # Allow all origins for local development

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
