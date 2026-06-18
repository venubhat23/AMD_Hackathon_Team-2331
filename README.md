# 🛒 ShelfSense AI
### Computer Vision Understanding of Retail Shelf Images & Flyers


**Team 2331** — Yogaraaj K · Bhanu Kantimahanti · Venugopal Bhat

---

## 📌 What is ShelfSense AI?

ShelfSense AI is an end-to-end retail intelligence platform that uses a **vision-language model (Qwen-VL)** to extract structured product data from shelf photos and promotional flyers. Extracted data is stored as **per-product embeddings** in a vector database and made queryable through a **RAG-powered chatbot** — enabling store managers to ask natural language questions about their inventory, prices, and promotions.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📸 **Multi-Image Analyzer** | Upload single or batch images — analyzed in parallel via thread pool |
| 🧠 **Per-Product RAG** | Each product gets its own vector embedding for surgical semantic retrieval |
| 💬 **AI Chatbot** | Ask "show me hair products" or "what's low on stock?" — grounded in real scan data |
| 📦 **Inventory Dashboard** | Auto-classifies stock as ✅ Present / ⚠️ Low / ❌ Absent from quantity counts |
| 💰 **Price Change Tracker** | Detects price movements across scans with direction and % delta |
| 📋 **Planogram Compliance** | Cross-references flyer promotions against shelf stock — generates compliance score |
| 🗂️ **Session Persistence** | Full session management with SQLite-backed chat history across reloads |
| 🎭 **Demo Mode** | Runs complete app without GPU using pre-loaded realistic retail data |
| 🔁 **Image Deduplication** | MD5 hash check prevents re-analyzing identical images |
| 📊 **Analytics Dashboard** | Category breakdown, top brands, promotion types, inventory distribution |

---

## 🏗️ Project Structure

```
shelfsense-ai/
├── main.py                 # Streamlit UI — 7 pages
├── llm_service.py          # Qwen-VL image analysis + parallel processing
├── vector_store.py         # ChromaDB (per-product embeddings) + SQLite
├── chatbot_service.py      # RAG-powered chatbot with demo fallback
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── uploads/                # Auto-created — stores uploaded images
└── chroma_db/              # Auto-created — persistent vector DB + SQLite
    ├── (chromadb files)
    └── sessions.db         # SQLite: sessions, chat_history, inventory_log, price_history
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- GPU with ≥16GB VRAM for Qwen-VL-7B (or use Demo Mode without GPU)
- Git

### Step 1 — Clone & create virtual environment
```bash
git clone <your-repo-url>
cd shelfsense-ai

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows CMD
# venv\Scripts\Activate.ps1     # Windows PowerShell
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment
```bash
cp .env.example .env
# Edit .env if needed (see Demo Mode section below)
```

### Step 4 — Start vLLM server (separate terminal, requires GPU)

```bash
#vllm should be installed earlier with pip install vllm
vllm serve Qwen/Qwen2.5-VL-32B-Instruct  --port 8000
```

> ⚠️ First run downloads ~15GB model weights. Subsequent starts are fast.

### Step 5 — Launch the app
```bash
streamlit run main.py   --server.enableCORS false   --server.enableXsrfProtection false
```

App opens at: **http://localhost:8501**

If need to open in AMD jupyter notebook : **https://notebooks.amd.com/<pod-name>/proxy/8501/** --check here, in place of pod-name, please provide your pod name (eg: jupyter-hack-team-2331-260609205410-931e891d**
---

## 🎭 Demo Mode (No GPU Required)

Run the full app without any GPU or vLLM server using pre-loaded realistic retail data:

```bash
# Option A — environment variable
DEMO_MODE=true streamlit run main.py   --server.enableCORS false   --server.enableXsrfProtection false

# Option B — edit .env file
echo "DEMO_MODE=true" > .env
streamlit run main.py   --server.enableCORS false   --server.enableXsrfProtection false
```

**What demo mode provides:**
- 2 pre-loaded scans (shelf image + promotional flyer)
- 10 products with prices, quantities, inventory statuses
- Active promotions (BOGO, percentage off, combo offers)
- Chatbot with keyword-matched answers for common queries
- All 7 pages fully functional with populated data

> 💡 The app also **auto-falls back** to demo data if vLLM is unreachable — no crashes during live demos.

---

## 🧭 App Pages Walkthrough

### 1. 📸 Analyze Images
- Upload one or multiple shelf/flyer images
- Duplicate images detected via MD5 hash before wasting API call
- Parallel analysis for batch uploads
- Results show: products table, promotions, inventory status counts
- Price changes vs previous scans highlighted immediately

### 2. 💬 Product Chatbot
- Natural language queries grounded in your scanned product database
- Per-product RAG retrieval (not image-level) for precise answers
- Chat history persists to SQLite when a session is active
- Suggested question buttons for quick exploration

### 3. 📦 Inventory Status
- Full inventory table with Present / Low / Absent classification
- Low stock alert panel — products needing urgent attention
- Filter by status, search by product/brand name
- Export to CSV

### 4. 💰 Price Changes
- All price movements detected across scans
- Split into increases 🔺 and decreases 🔻
- Shows previous price, new price, delta amount and percentage
- Export price change log to CSV

### 5. 📋 Planogram Compliance
- Upload a flyer + a shelf image
- System fuzzy-matches promoted products against shelf stock
- Generates a compliance score (0–100%)
- Flags which promoted products are missing from shelf
- Adjustable match confidence threshold

### 6. 🕒 History
- All past scans with timestamps
- Expandable view per scan — products, promotions, inventory counts
- Price changes highlighted per scan

### 7. 📊 Analytics
- Category breakdown bar chart
- Inventory distribution chart
- Top brands ranking
- Promotion type breakdown
- Full aggregated products table

### 8. 🗂️ Sessions
- Group multiple scans into named audit sessions
- Restore any previous session to reload its chat history
- View full chat transcripts per session

---

## 🤖 AI Architecture

```
Image Input
    ↓
Pre-processing (blur check, contrast enhancement, auto-crop)
    ↓
Qwen2.5-VL-7B-Instruct via vLLM
Two-stage prompting:
  Stage 1 → Product detection
  Stage 2 → Price, quantity, promotions, inventory extraction
    ↓
Output Validation + Price Normalization + Category Normalization
    ↓
┌─────────────────────────────────────────────┐
│           Storage Layer                     │
│  ChromaDB — per-product vector embeddings   │
│  SQLite   — inventory, prices, sessions     │
└─────────────────────────────────────────────┘
    ↓
RAG Retrieval (all-MiniLM-L6-v2 embeddings)
    ↓
LLM Response Generation (Qwen via vLLM)
    ↓
Streamlit Dashboard
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Vision-Language Model | Qwen2.5-VL-7B-Instruct |
| LLM Serving | vLLM (OpenAI-compatible API) |
| Vector Database | ChromaDB (persistent, cosine similarity) |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 |
| Relational Store | SQLite |
| Frontend | Streamlit |
| Parallel Processing | Python ThreadPoolExecutor |
| Deduplication | MD5 hashing (hashlib) |
| Fuzzy Matching | difflib SequenceMatcher |
| Language | Python 3.10+ |

---

## 📦 Dependencies

```
streamlit>=1.35.0
Pillow>=10.0.0
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
orjson>=3.9.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Single image analysis | 8–15 seconds (GPU) |
| Batch of 3 images (parallel) | ~15 seconds total |
| RAG query response | < 2 seconds |
| Embedding generation | ~15ms per product |
| ChromaDB query (1000 products) | < 50ms |
| Demo mode response | < 100ms |
| GPU VRAM required | ~14–15GB (Qwen-VL-7B bf16) |
| CPU-only mode | Demo Mode — zero GPU needed |

---

## 🗃️ Database Schema

**ChromaDB collections:**
- `retail_products` — one document per product, with embedding
- `retail_images` — one document per scan (image-level summary)

**SQLite tables (`chroma_db/sessions.db`):**
- `sessions` — audit session records
- `chat_history` — per-session chat messages
- `inventory_log` — per-product inventory entries per scan
- `price_history` — price per product per scan (enables change detection)
- `image_hashes` — MD5 hashes for deduplication

---

## 🔮 Future Roadmap

- [ ] Expiry date extraction and freshness alerting
- [ ] Shelf Share of Voice — brand space % per image
- [ ] Mobile app for field agents
- [ ] Fine-tuning Qwen-VL on Indian retail product imagery
- [ ] ERP/POS system integration for sell-through correlation
- [ ] Competitor price intelligence from public flyers
- [ ] WhatsApp bot — photo → analysis in chat
- [ ] Predictive restocking using CV inventory + sales velocity

---

## 👥 Team

**Team 2331**

| Member | Role |
|---|---|
| Yogaraaj K| [Role] |
| Bhanu Kantimahanti| [Role] |
| Venugopal Bhat | [Role] |

---

## 📄 License

This project was built for hackathon purposes.

---

*Built with ❤️ during the hackathon — Team 2331*
