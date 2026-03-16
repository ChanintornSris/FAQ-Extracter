# FAQ Mining System — v3

> AI-powered FAQ extraction from Thai customer support conversations.  
> Uses **local LLM (Ollama)** to extract real Q&A pairs — no paid API, no internet required.

---

## ⚡ Quick Start (Complete Setup)

### Step 0 — Prerequisites Check

Open PowerShell and verify Python is installed:

```powershell
python --version   # Need 3.9 or higher
pip --version
```

If Python is missing → download from https://www.python.org/downloads/

---

### Step 1 — Install Ollama

**Option A (Recommended) — One command:**
```powershell
winget install Ollama.Ollama
```

**Option B — Download installer manually:**  
Go to: **https://ollama.com/download/OllamaSetup.exe**  
Run the `.exe` → Click Install → **Close and reopen PowerShell**

**Verify installation:**
```powershell
ollama --version
# Expected output: ollama version 0.x.x
```

> After installation, Ollama runs as a **background service automatically**.  
> You do NOT need to run `ollama serve` manually.

---

### Step 2 — Choose and Download a Language Model

Pick the model that fits your RAM:

| Model | Size | RAM Needed | Thai Quality | Recommendation |
|---|---|---|---|---|
| `scb10x/llama3.1-typhoon2-8b-instruct` | 5.0 GB | 8 GB+ | ✅ Best Thai | **Best overall — start here** |

**Download your chosen model** (example: typhoon):
```powershell
ollama pull scb10x/llama3.1-typhoon2-8b-instruct
```
> This takes 5–20 minutes depending on internet speed. Progress bar will appear.

**Verify the model downloaded:**
```powershell
ollama list
# Should show: scb10x/llama3.1-typhoon2-8b-instruct   ... (size) ... (date)
```

**Quick test — confirm Thai response:**
```powershell
ollama run scb10x/llama3.1-typhoon2-8b-instruct "สรุปในหนึ่งประโยค: ลูกค้าถามว่าโอนเงินยังไง"
```
Expected: a short Thai response. If you see Thai text → ✅ ready.

---

### Step 3 — Install Python Dependencies

```powershell
cd C:\Users\User\Desktop\context_extract
pip install -r requirements.txt
```

> First run downloads `BAAI/bge-m3` embedding model (~2.2 GB, automatic, cached).

---

### Step 4 — Configure the Model Name

Tell the system which Ollama model to use.

**Check the exact model name from step 2:**
```powershell
ollama list
```
The name shown here (e.g. `scb10x/llama3.1-typhoon2-8b-instruct`) is what you set below.

**Option A — Set per session (PowerShell):**
```powershell
$env:LLM_MODEL = "scb10x/llama3.1-typhoon2-8b-instruct"
```

**Option B — Set permanently in `backend/config.py`:**
```python
# Line ~153 in config.py
TOPIC_NAMER_MODEL = "scb10x/llama3.1-typhoon2-8b-instruct"   # ← change this
```

---

### Step 5 — Run the Pipeline

```powershell
# From the project directory:
python backend/main.py --input data/conversations.json --serve
```

Watch the console. You'll see stages progress:
```
Stage 1: Loading dataset...      → X records
Stage 2: Cleaning text...
Stage 3: Filtering questions...
Stage 3.5: LLM FAQ extraction... → calls Ollama for each batch
Stage 4: Embeddings (bge-m3)...  → downloads model first time
Stage 4.5: UMAP reduction...
...
Pipeline complete! X groups, Y FAQ pairs
Starting API at http://0.0.0.0:8000
```

---

### Step 6 — Open the UI

```
http://localhost:8000
```

---

## 🔄 Subsequent Runs (Already Set Up)

```powershell
$env:LLM_MODEL = "hf.co/scb10x/typhoon-v1.5-8b-instruct-gguf:Q4_K_M"
python backend/main.py --input data/conversations.json --serve
```

Embeddings and UMAP are **cached** — Stage 4 and 4.5 are skipped on repeat runs unless data changes.

---

## How It Works (Pipeline Overview)

```
Conversations
  → Stage 1-3:  Load → Clean → Filter valid Q&A pairs
  → Stage 3.5:  [LLM] Ollama reads batches of 30 conversations
                and extracts canonical {question, answer} pairs
  → Stage 4:    Embed FAQ questions (BAAI/bge-m3, 1024-dim)
  → Stage 4.5:  UMAP reduction (1024 → 8 dims)
  → Stage 5:    Deduplicate similar FAQs
  → Stage 6-7:  HDBSCAN clustering + quality filter
  → Stage 8:    [LLM] Ollama names each group
  → Stage 9-10: Assemble groups with full FAQ lists
  → Stage 11:   Build FAISS semantic search index
  → Output:     groups[{ group_name, faqs:[{question, answer}] }]
```

---

## Output Example

```json
{
  "total_groups": 5,
  "groups": [
    {
      "group_name": "ปัญหาการเข้าระบบ MT5",
      "total_faqs": 8,
      "faqs": [
        { "question": "MT5 เข้าระบบไม่ได้ทำอย่างไร", "answer": "กรุณารีเซ็ตรหัสผ่าน..." },
        { "question": "ลืมรหัสผ่าน MT5 ทำยังไง",      "answer": "ติดต่อ Support ที่..." }
      ]
    }
  ]
}
```

---

## Input File Format

Upload any file with customer questions and admin answers. Column names don't matter — the UI has a column mapping step.

**JSON:**
```json
[
  { "customer_message": "MT5 เข้าไม่ได้", "admin_reply": "กรุณารีเซ็ตรหัสผ่าน" }
]
```

**CSV:**
```csv
customer_message,admin_reply
"MT5 เข้าไม่ได้","กรุณารีเซ็ตรหัสผ่าน"
```

**Excel (.xlsx):** Any two columns, map them in the UI.

---

## Configuration Reference (`backend/config.py`)

### LLM Settings

| Setting | Default | Description |
|---|---|---|
| `TOPIC_NAMER_PROVIDER` | `"ollama"` | Change to `"mock"` to skip LLM (for testing only) |
| `TOPIC_NAMER_MODEL` | `"scb10x/typhoon..."` | **Change to match your `ollama list` output** |
| `TOPIC_NAMER_OLLAMA_URL` | `"http://localhost:11434/api/generate"` | Default Ollama endpoint |

### FAQ Extraction

| Setting | Default | Description |
|---|---|---|
| `FAQ_EXTRACTION_BATCH_SIZE` | `30` | Conversations per LLM call |
| `FAQ_PER_BATCH` | `8` | Max FAQ pairs extracted per batch |
| `FAQ_DEDUP_SIMILARITY_THRESHOLD` | `0.90` | Lower = keep more FAQ variants |

### Clustering

| Setting | Default | Description |
|---|---|---|
| `CLUSTER_MIN_CLUSTER_SIZE` | `3` | Min FAQs per group (lower for small datasets) |

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ollama: not recognized` | Ollama not installed or PowerShell not restarted | Close & reopen PowerShell after install |
| `"No FAQ pairs extracted"` | Ollama not running / model not pulled | Run `ollama list` to confirm model exists |
| Ollama running but no response | Model name wrong in config | Run `ollama list`, copy exact name to `LLM_MODEL` |
| Pipeline very slow | LLM calls take time | Normal — Stage 3.5 calls Ollama per batch; increase `FAQ_EXTRACTION_BATCH_SIZE` |
| Too few groups | Dataset too small or `CLUSTER_MIN_CLUSTER_SIZE` too high | Lower `CLUSTER_MIN_CLUSTER_SIZE` to 2 |
| bge-m3 download slow | First run only | 2.2 GB download, cached after first run |

---

## Project Structure

```
context_extract/
├── backend/
│   ├── config.py              ← All settings (edit this)
│   ├── main.py                ← Pipeline orchestrator
│   ├── api.py                 ← FastAPI REST + frontend
│   ├── data_loader.py         ← Stage 1: Load files
│   ├── text_cleaner.py        ← Stage 2: Clean text
│   ├── question_filter.py     ← Stage 3: Filter Q&A
│   ├── llm_extractor.py       ← Stage 3.5: LLM batch extraction [NEW]
│   ├── embedding_service.py   ← Stage 4: bge-m3 (1024-dim)
│   ├── umap_reducer.py        ← Stage 4.5: UMAP reduction [NEW]
│   ├── deduplication.py       ← Stage 5: Dedup
│   ├── clustering.py          ← Stage 6-7: HDBSCAN
│   ├── topic_namer.py         ← Stage 8: LLM group naming [NEW]
│   ├── faq_generator.py       ← Stage 9-10: Assemble output
│   ├── search_index.py        ← Stage 11: FAISS
│   └── analytics.py           ← Stage 13: Reports
├── frontend/
│   ├── index.html             ← Single-page app UI
│   ├── manual.html            ← User manual (this guide in Thai)
│   └── app.js                 ← UI logic
├── data/
│   ├── conversations.json     ← Sample dataset
│   ├── uploads/               ← Uploaded files
│   └── faqs.json              ← Output (auto-generated)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## API Reference

Base URL: `http://localhost:8000` | Swagger: `/docs`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/groups` | **Primary** — all FAQ groups with Q&A lists |
| GET | `/faqs` | Legacy alias for `/groups` |
| POST | `/search` | Semantic search: `{"query": "..."}` |
| GET | `/analytics` | Statistics report |
| GET | `/pipeline-status` | Poll pipeline progress |
| POST | `/upload` | Upload data file |
| POST | `/run-pipeline` | Trigger analysis |
| POST | `/faqs/relabel` | Move FAQ to different group |
| POST | `/faqs/delete` | Delete FAQ |
| GET | `/manual` | This user manual |

---

## Use Without UI (CLI)

```powershell
# Run pipeline + start server
$env:LLM_MODEL = "hf.co/scb10x/typhoon-v1.5-8b-instruct-gguf:Q4_K_M"
python backend/main.py --input data/conversations.json --serve

# Pipeline only (no server)
python backend/main.py --input data/conversations.json

# Test mode — no Ollama needed (mock LLM)
$env:LLM_PROVIDER = "mock"
python backend/main.py --input data/conversations.json --serve
```

---

## Docker Deployment

```powershell
docker build -t faq-miner-ai .
docker run -d -p 8000:8000 -v faq_data:/app/data `
  -e LLM_MODEL=scb10x/llama3.1-typhoon2-8b-instruct `
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate `
  --name faq_miner faq-miner-ai
```

> Ollama must run on the **host machine**. Use `host.docker.internal` instead of `localhost`.
