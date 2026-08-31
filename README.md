# OmniBrain — Agentic Multi-Modal RAG Orchestrator

**OmniBrain** is an Agentic Multi-Modal Retrieval-Augmented Generation (RAG) system engineered for complex financial reports and historical stock database analytics. Focused on **NVIDIA (NVDA)**, OmniBrain combines vector search over unstructured multi-page PDFs, multimodal vision reasoning over embedded charts/figures, and natural language Text-to-SQL querying against historical stock databases — powered by **Google Gemini Flash** via the free-tier Google AI Studio API.

---

## 🌟 Key Features

- **Unified Gemini Flash Stack**: Uses `gemini-2.0-flash` for text reasoning, routing, SQL generation, synthesis, and direct inline multimodal chart analysis.
- **Agentic Routing (LangGraph)**:
  - **Supervisor Router**: Dynamically routes user queries to the appropriate agent.
  - **Search Agent**: Semantic document retrieval using Qdrant vector store and `text-embedding-004`.
  - **Vision Agent**: Multimodal visual interpretation of financial charts and graphs via inline image inputs.
  - **SQL Agent**: Text-to-SQL generation targeting historical NVDA stock prices (1999–Present) and annual financials.
- **Self-RAG & Self-Correction**: Evaluates retrieved context relevance and automatically rewrites search queries if initial retrieval is inadequate.
- **Financial Guardrails**:
  - Enforces corporate finance/stock domain scope (refuses out-of-scope requests gracefully).
  - Validates response groundedness against supporting evidence.
- **Real Dataset Integration**: Built using real historical NVIDIA stock price data (`data/stock_data/NVDA.csv`).
- **Observability**: Execution tracing and metrics with Langfuse integration.
- **Interactive UI & REST API**: Full Streamlit frontend dashboard and FastAPI backend API.

---

## 🏗️ System Architecture

```
USER QUERY → Streamlit UI / FastAPI Backend → Financial Guardrails
           ↓
     LangGraph Supervisor Router
     ├── Search Agent ──> Qdrant Vector Store (text-embedding-004) ──> Self-RAG Evaluation
     ├── Vision Agent ──> Gemini Flash Multimodal (Page Charts / Figures)
     └── SQL Agent    ──> Text-to-SQL ──> SQLite (NVDA.csv Stock History DB)
           ↓
     Response Synthesis ──> Groundedness Check ──> Cited Final Answer
```

---

## 🛠️ Project Repository Structure

```
OmniBrain-Agentic-Multi-Modal-RAG-Orchestrator/
├── app/
│   ├── agents/
│   │   ├── state.py              # LangGraph workflow state definition
│   │   ├── supervisor.py         # Supervisor Agent (query router)
│   │   ├── search_agent.py       # Document Search Agent (Vector RAG)
│   │   ├── vision_agent.py       # Vision Agent (Chart Multimodal reasoning)
│   │   ├── sql_agent.py          # SQL Agent (Text-to-SQL for NVDA DB)
│   │   ├── self_rag.py           # Self-RAG evaluator & query rewriter
│   │   ├── synthesis.py          # Response synthesis node
│   │   ├── llm.py                # Gemini Flash client wrapper
│   │   └── graph.py              # Compiled LangGraph state machine
│   ├── ingestion/
│   │   ├── pdf_parser.py         # PyMuPDF text & chart extractor
│   │   └── embedder.py           # Gemini text-embedding-004 client
│   ├── retrieval/
│   │   └── vector_store.py       # Qdrant vector store client
│   ├── database/
│   │   ├── stock_db.py           # SQLite connection & schema manager
│   │   └── seed_stock_db.py      # Idempotent CSV-to-SQLite database seeder
│   ├── guardrails/
│   │   └── guardrail_manager.py # Financial scope & groundedness validation
│   ├── evaluation/
│   │   └── observability.py      # Langfuse observability tracer
│   ├── config.py                 # Settings & env var configuration
│   └── main.py                   # FastAPI server (/upload, /query, /documents, /health)
├── frontend/
│   └── app.py                    # Streamlit dashboard & interactive chat UI
├── data/
│   ├── documents/                # PDF report storage & extracted chart images
│   └── stock_data/
│       ├── NVDA.csv              # Required NVIDIA historical stock price CSV
│       └── stocks.db             # Pre-seeded SQLite database
├── scripts/
│   └── generate_sample_pdf.py    # Auto-generates sample NVIDIA report with charts & tables
├── tests/
│   ├── test_stock_db.py          # DB schema & seeding tests
│   ├── test_ingestion.py         # PDF parsing & Qdrant vector indexing tests
│   ├── test_agents.py            # LangGraph routing & agent tests
│   └── test_api.py               # FastAPI REST endpoint tests
├── .env.example                  # Environment configuration example
├── .gitignore                    # Git ignore file
├── README.md                     # Documentation
└── requirements.txt              # Project dependencies
```

---

## ⚡ Quick Start & Setup Guide

### 1. Requirements & Prerequisites
- Python 3.10+
- Google AI Studio API Key (Free Tier Gemini access)

### 2. Installation
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/sagarkapoor03/OmniBrain-Agentic-Multi-Modal-RAG-Orchestrator.git
cd OmniBrain-Agentic-Multi-Modal-RAG-Orchestrator

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your Google AI Studio API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
```

> **Note**: `data/stock_data/NVDA.csv` is the required dataset file and must be present at `data/stock_data/NVDA.csv` before running the seeding script.

### 4. Database Seeding
Seed the SQLite database with NVIDIA historical stock prices (1999–Present) and annual financials:
```bash
python -m app.database.seed_stock_db
```

### 5. Generate Sample NVIDIA PDF Report
Generate a multi-page sample NVIDIA FY2025 financial report containing text, financial tables, and revenue charts for zero-setup testing:
```bash
python scripts/generate_sample_pdf.py
```

---

## 🚀 Running the Application

### Option A: Run FastAPI Backend
Start the REST API server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at `http://localhost:8000/docs`.

### Option B: Run Streamlit UI Dashboard
Launch the interactive web dashboard:
```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
pytest tests/ -v
```

---

## 📊 Sample Queries to Try

- **SQL Stock Query**: *"What was NVIDIA's highest stock closing price in 2024?"*
- **Document Text Search**: *"What was NVIDIA's total revenue in FY2025 according to the report?"*
- **Vision Chart Analysis**: *"What does the NVIDIA revenue growth chart indicate?"*
- **Multi-Modal Hybrid Query**: *"Compare NVIDIA's annual revenue growth in the report with its historical stock performance."*
- **Guardrail Test**: *"How do I bake a chocolate cake?"* (Triggers out-of-scope refusal).

---

## 📄 License

MIT License
