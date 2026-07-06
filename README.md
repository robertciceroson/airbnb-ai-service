# Airbnb AI Service — FastAPI + LangGraph Agent

**A production-style AI service platform** built on top of the [Airbnb Price Prediction](https://github.com/robertciceroson/Airbnb-Price-Prediction) project — extending it from a Streamlit demo into a REST API with a multi-tool conversational AI agent for customer service.

---

## What It Does

This project exposes two REST endpoints backed by two distinct AI systems:

| Endpoint | What it does |
|---|---|
| `POST /predict` | Returns an estimated nightly Airbnb price for any NYC listing using a trained XGBoost model with seasonal adjustment |
| `POST /chat` | Routes user messages through a LangGraph agent that selects from three tools: price lookup, policy RAG search, or human handoff |

Real-world scenario: a traveler asks *"How much would a private room in Williamsburg cost in August, and what's the cancellation policy if I need to cancel 3 days before check-in?"* — the agent calls the price tool for the first part and the RAG policy tool for the second, in a single conversation turn.

---

## Architecture

```
Client (Streamlit / cURL / any HTTP client)
         │
         ▼
   FastAPI (uvicorn)
   ┌──────────────────────────────────┐
   │  GET  /health                    │
   │  POST /predict ─► XGBoost Model  │
   │  POST /chat    ─► LangGraph Agent│
   └────────────────┬─────────────────┘
                    │
         ┌──────────▼──────────┐
         │   LangGraph Agent    │
         │  (StateGraph + LLM)  │
         └──┬─────────┬────────┘
            │         │         │
     ┌──────▼─┐  ┌────▼────┐  ┌▼──────────┐
     │ Price  │  │ Policy  │  │  Human    │
     │ Tool   │  │ RAG Tool│  │ Handoff   │
     │(XGBoost│  │(FAISS + │  │  Tool     │
     │  API)  │  │LangChain│  │           │
     └────────┘  └─────────┘  └───────────┘
                      │
               ┌──────▼──────┐
               │ FAISS Index  │
               │ (Airbnb Help │
               │  Center docs)│
               └─────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| ML Model | XGBoost (trained on ~49K NYC listings, June 2026) |
| Agent Orchestration | LangGraph (StateGraph) |
| LLM | Llama 3.3 70B via Groq |
| RAG | LangChain + FAISS + HuggingFace Embeddings |
| Data Validation | Pydantic v2 |
| Model Serialization | joblib |
| Containerization | Docker |
| Frontend | Streamlit (two-tab: Price Predictor + Support Chat) |

---

## Project Structure

```
airbnb-ai-service/
├── app/
│   ├── main.py              # FastAPI app — /health, /predict, /chat
│   ├── config.py            # Centralized settings (pydantic-settings + .env)
│   ├── models/
│   │   ├── predictor.py     # XGBoost load-on-startup, predict()
│   │   └── schemas.py       # Pydantic request/response models
│   ├── agent/
│   │   ├── graph.py         # LangGraph StateGraph — agent + tool nodes
│   │   └── tools.py         # price_lookup, policy_search, human_handoff
│   └── rag/
│       └── ingest.py        # FAISS build + load + retriever
├── data/
│   ├── listings.csv         # NYC Airbnb dataset (Inside Airbnb, June 2026)
│   └── policies/            # Fetched Airbnb Help Center .txt files
├── models/                  # Saved XGBoost model + encoders (joblib)
├── vector_store/            # FAISS index files
├── train_and_save_model.py  # Step 1: train XGBoost, serialize to disk
├── ingest_policies.py       # Step 2: fetch Help Center pages, build FAISS
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier is sufficient)
- `listings.csv` from [Inside Airbnb](http://insideairbnb.com/get-the-data/) — New York City

### 1. Clone and install

```bash
git clone https://github.com/robertciceroson/airbnb-ai-service.git
cd airbnb-ai-service
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY
```

### 3. Place the dataset

```
data/listings.csv    ← NYC Airbnb listings from Inside Airbnb (June 2026)
```

### 4. Train the model

```bash
python train_and_save_model.py
```

Outputs `models/xgboost_model.joblib` and `models/encoders.joblib`. Takes ~30 seconds on a standard laptop.

### 5. Build the policy RAG index

```bash
python ingest_policies.py
```

Fetches 15 Airbnb Help Center pages, chunks them, embeds with `all-MiniLM-L6-v2`, and saves a FAISS index to `vector_store/`. Takes ~2 minutes on first run (embedding model downloads once).

### 6. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Reference

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "vector_store_loaded": true,
  "version": "1.0.0"
}
```

### `POST /predict`

**Request:**
```json
{
  "borough": "Manhattan",
  "neighbourhood": "Midtown",
  "room_type": "Entire home/apt",
  "minimum_nights": 2,
  "availability_365": 200,
  "number_of_reviews": 20,
  "reviews_per_month": 1.0,
  "calculated_host_listings_count": 1,
  "checkin_month": 7
}
```

**Response:**
```json
{
  "base_price": 241.50,
  "adjusted_price": 294.63,
  "seasonal_multiplier": 1.22,
  "season_label": "☀️ Summer peak",
  "neighbourhood_median": 265.0,
  "currency": "USD"
}
```

### `POST /chat`

**Request:**
```json
{
  "message": "What happens if I cancel 3 days before check-in on a moderate policy listing?",
  "conversation_id": "conv-abc-123",
  "history": []
}
```

**Response:**
```json
{
  "reply": "For a Moderate cancellation policy, cancelling 3 days before check-in means...",
  "tool_used": "policy_search",
  "conversation_id": "conv-abc-123",
  "sources": ["Airbnb Help Center — Moderate Cancellation Policy"]
}
```

---

## Agent Tools

| Tool | Trigger | What it does |
|---|---|---|
| `price_lookup` | Price / cost / rate questions | Calls XGBoost predictor with seasonal adjustment |
| `policy_search` | Cancellation / refund / rules / disputes | RAG retrieval over Airbnb Help Center docs |
| `human_handoff` | Frustration, account security, unresolvable issues | Returns structured escalation message with support links |

---

## Docker

```bash
# Build
docker build -t airbnb-ai-service .

# Run
docker run -p 8000:8000 --env-file .env airbnb-ai-service
```

> **Note:** Run `train_and_save_model.py` and `ingest_policies.py` before building the image so the model and vector store are included.

---

## Related Projects

- [Airbnb Price Prediction](https://github.com/robertciceroson/Airbnb-Price-Prediction) — Original EDA + ML pipeline + Streamlit demo (this project extends it)
- [HR Policy QA Bot](https://github.com/robertciceroson/HR-Policy-QA-Bot) — RAG pipeline with LangChain + FAISS + Llama 3.3 70B

---

## Author

**Robert C. Son** — AI Engineer | Business Analyst | Process Engineer
- GitHub: [github.com/robertciceroson](https://github.com/robertciceroson)
- LinkedIn: [linkedin.com/in/robert-son-0b33b3bb](https://linkedin.com/in/robert-son-0b33b3bb)
- Live demo (price predictor): [airbnb-price-prediction-a9hyny92wfihme4mnzkzte.streamlit.app](https://airbnb-price-prediction-a9hyny92wfihme4mnzkzte.streamlit.app)
