# Airbnb AI Service — FastAPI + LangGraph Agent + Support Chat

**A production-style AI service platform** built on top of the [Airbnb Price Prediction](https://github.com/robertciceroson/Airbnb-Price-Prediction) project — extending it from a Streamlit demo into a REST API with a multi-tool conversational AI agent for customer service.

🔗 **Live demo:** [airbnb-ai-service-tubhzkkmermmacr29pmakg.streamlit.app](https://airbnb-ai-service-tubhzkkmermmacr29pmakg.streamlit.app)

---

## What It Does

Two tabs, two AI systems:

| Tab | What it does |
|---|---|
| 💰 **Price Predictor** | Estimates the nightly Airbnb price for any NYC listing using a trained XGBoost model with seasonal adjustment. Calls a FastAPI backend hosted on Render. |
| 💬 **Support Chat** | Conversational AI agent (LangGraph + GPT-OSS 120B via Groq) that answers Airbnb policy and pricing questions using BM25 RAG over bundled policy documents. Runs entirely inside Streamlit Cloud — no Render timeout constraints. |

**Real-world scenario:** A traveler asks *"How much would a private room in Williamsburg cost in August, and what's the difference between a refund and Airbnb travel credit if my host cancels?"* — the agent calls the price tool for the first part and retrieves the exact policy for the second, in a single conversation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Cloud                         │
│                                                          │
│  Tab 1: Price Predictor                                  │
│    └─► POST /predict ─────────────────────────────────► │──┐
│                                                          │  │
│  Tab 2: Support Chat (LangGraph agent — runs here)       │  │  Render (Free Tier)
│    └─► load_agent() [cached]                             │  │  FastAPI /predict
│          ├─ price_lookup ────────────────────────────►   │──┘  XGBoost only
│          ├─ policy_search (BM25 over data/policies/)     │     DISABLE_AGENT=true
│          └─ human_handoff                                │
└─────────────────────────────────────────────────────────┘
```

**Why the agent runs in Streamlit, not Render:** Render's free tier has a 512 MB RAM limit and a 30-second request timeout — both too tight for a LangGraph + LLM agent. The FastAPI backend uses `DISABLE_AGENT=true` and only serves the XGBoost predictor. The LangGraph agent is built once, cached with `@st.cache_resource`, and runs entirely inside Streamlit Cloud.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| ML Model | XGBoost (trained on ~49K NYC listings, June 2026 — Inside Airbnb) |
| Agent Orchestration | LangGraph (StateGraph) |
| LLM | GPT-OSS 120B via Groq |
| RAG | BM25 sparse retrieval (rank-bm25 + LangChain) — no embeddings, no vector DB |
| Data Validation | Pydantic v2 |
| Model Serialization | joblib |
| Frontend | Streamlit (two-tab: Price Predictor + Support Chat) |
| Deployment | Render (FastAPI, predict-only) + Streamlit Cloud (full app + agent) |

---

## Support Chat — Features

The Support Chat tab runs a full LangGraph ReAct-style agent with three tools:

| Tool | When it fires | What it does |
|---|---|---|
| `policy_search` | Policy / cancellation / refund / rules / dispute questions | BM25 retrieval over 16 bundled Airbnb Help Center policy documents |
| `price_lookup` | Price / cost / rate questions | Calls the Render `/predict` endpoint (XGBoost + seasonal adjustment) |
| `human_handoff` | Frustration, account security, unresolvable issues | Returns a structured escalation message |

**Policy documents included** (`data/policies/`):

- Cancellation policies (Flexible, Moderate, Strict)
- Guest refund policy + host refund policy
- Refund vs. travel credit — what guests actually receive and when
- Extenuating circumstances / major disruptive events
- Resolution Center — how to dispute damage charges
- Security deposit, service fees, payment methods
- Check-in instructions, house rules, review policy

**UI features:**
- Session ID displayed per conversation (for debugging)
- Clear Chat button + red reminder hint to start fresh for new topics
- Suggested prompts: *What is the refund if my host cancels?* · *What is the difference between a refund and Airbnb travel credit?* · *How do I dispute a damage charge?*

---

## Project Structure

```
airbnb-ai-service/
├── app/
│   ├── main.py              # FastAPI app — /health, /predict, /chat (DISABLE_AGENT mode)
│   ├── config.py            # Centralized settings (pydantic-settings + .env)
│   ├── models/
│   │   ├── predictor.py     # XGBoost load-on-startup, predict()
│   │   └── schemas.py       # Pydantic request/response models
│   └── agent/
│       └── graph.py         # LangGraph StateGraph (used locally; disabled on Render)
├── data/
│   ├── listings.csv         # NYC Airbnb dataset (Inside Airbnb, June 2026)
│   └── policies/            # 16 Airbnb Help Center policy .txt files (BM25 source)
├── models/                  # Saved XGBoost model + encoders (joblib)
├── streamlit_app.py         # Two-tab Streamlit frontend (Price Predictor + Support Chat)
├── train_and_save_model.py  # Train XGBoost, serialize to disk
├── requirements.txt
├── .env.example
├── render.yaml              # Render deployment config
└── Procfile
```

---

## Setup & Local Development

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

### 5. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The Support Chat agent loads from `data/policies/` via BM25 — no separate index build step needed. Price lookups call `localhost:8000` by default.

### 6. (Optional) Run the FastAPI backend locally

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
  "vector_store_loaded": false,
  "version": "1.0.0"
}
```
> `vector_store_loaded` is `false` on Render (`DISABLE_AGENT=true`) — the agent runs in Streamlit Cloud, not here.

### `POST /predict`

**Request:**
```json
{
  "borough": "Manhattan",
  "neighbourhood": "Chelsea",
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
  "base_price": 747.00,
  "adjusted_price": 911.00,
  "seasonal_multiplier": 1.22,
  "season_label": "☀️ Summer peak",
  "neighbourhood_median": 610.0,
  "currency": "USD"
}
```

### `POST /chat`

> **Note:** `/chat` is available when running locally with `DISABLE_AGENT` unset or `false`. On the Render deployment it returns 503 (agent disabled). In production, the agent runs in Streamlit Cloud directly.

**Request:**
```json
{
  "message": "What is the difference between a refund and Airbnb travel credit when my host cancels?",
  "conversation_id": "conv-abc-123",
  "history": []
}
```

**Response:**
```json
{
  "reply": "When a host cancels your confirmed reservation, Airbnb issues a full cash refund to your original payment method — not travel credit...",
  "tool_used": "policy_search",
  "conversation_id": "conv-abc-123",
  "sources": []
}
```

---

## Deployment Notes

### Render (FastAPI — predict-only)
- Set `DISABLE_AGENT=true` in Render environment variables
- This skips loading LangChain/LangGraph at startup, keeping memory under 512 MB
- Only `/health` and `/predict` are active; `/chat` returns 503

### Streamlit Cloud
- Set `GROQ_API_KEY` and `API_BASE_URL` in Streamlit Secrets
- `API_BASE_URL` should point to the Render service URL
- The LangGraph agent and BM25 retriever load once on first chat and are cached for the session lifetime

---

## Related Projects

- [Airbnb Price Prediction](https://github.com/robertciceroson/Airbnb-Price-Prediction) — Original EDA + ML pipeline + Streamlit demo (this project extends it)
- [HR Policy QA Bot](https://github.com/robertciceroson/HR-Policy-QA-Bot) — RAG pipeline with LangChain + FAISS + Llama 3.3 70B

---

## Author

**Robert C. Son** — AI Engineer | AI Business Analyst | Process Engineer
- GitHub: [github.com/robertciceroson](https://github.com/robertciceroson)
- LinkedIn: [linkedin.com/in/robert-son-0b33b3bb](https://linkedin.com/in/robert-son-0b33b3bb)
- Live demo: [airbnb-ai-service-tubhzkkmermmacr29pmakg.streamlit.app](https://airbnb-ai-service-tubhzkkmermmacr29pmakg.streamlit.app)
