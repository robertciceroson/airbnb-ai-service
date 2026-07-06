"""
Airbnb AI Service — FastAPI application entry point.

Endpoints:
  GET  /health        — liveness check
  POST /predict       — XGBoost price prediction
  POST /chat          — LangGraph multi-tool AI agent (standard)
  POST /chat/stream   — same agent, SSE streaming (keepalive pings every 3s)

Run locally:
    uvicorn app.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.predictor import Predictor
from app.models.schemas import (
    PredictRequest, PredictResponse,
    ChatRequest, ChatResponse,
    HealthResponse,
)
from app.agent.graph import AirbnbAgent
from app.rag.ingest import get_retriever


# ── App state (shared across requests) ───────────────────────────────────────

class AppState:
    predictor: Predictor | None = None
    agent: AirbnbAgent | None = None
    vector_store_loaded: bool = False


state = AppState()


# ── Lifespan: startup / shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    print("🚀 Starting Airbnb AI Service…")

    # Load XGBoost model
    try:
        state.predictor = Predictor()
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   Run `python train_and_save_model.py` then restart.")

    # Load BM25 retriever from policy docs
    retriever = None
    try:
        retriever = get_retriever()
        state.vector_store_loaded = True
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   Run `python ingest_policies.py` then restart.")

    # Build LangGraph agent (works even if predictor/retriever are None)
    state.agent = AirbnbAgent(
        predictor=state.predictor,
        retriever=retriever,
    )
    print("✅ Agent ready.")

    yield  # ── App runs ──

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("👋 Shutting down.")


# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "REST API for NYC Airbnb price prediction (XGBoost) and "
        "AI-powered customer service chat (LangGraph + RAG)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Liveness check — confirms service is up and which components are loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=state.predictor is not None,
        vector_store_loaded=state.vector_store_loaded,
        version=settings.app_version,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict the estimated nightly Airbnb price for a NYC listing.

    - Uses XGBoost trained on ~49K current listings (June 2026 — Inside Airbnb)
    - Applies seasonal multiplier based on check-in month
    - Returns base price, adjusted price, seasonal label, and neighbourhood median
    """
    if state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_and_save_model.py and restart."
        )
    try:
        result = state.predictor.predict(
            borough=request.borough,
            neighbourhood=request.neighbourhood,
            room_type=request.room_type,
            minimum_nights=request.minimum_nights,
            availability_365=request.availability_365,
            number_of_reviews=request.number_of_reviews,
            reviews_per_month=request.reviews_per_month,
            calculated_host_listings_count=request.calculated_host_listings_count,
            checkin_month=request.checkin_month,
        )
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat Agent"])
async def chat(request: ChatRequest):
    """
    Multi-turn AI chat agent for Airbnb customer service.

    The agent automatically selects the right tool:
    - **price_lookup** — nightly rate questions
    - **policy_search** — cancellations, refunds, rules, disputes
    - **human_handoff** — escalation when needed

    Pass a stable `conversation_id` across turns to maintain context.
    """
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        history = [{"role": m.role, "content": m.content} for m in request.history]
        result  = state.agent.invoke(
            user_message=request.message,
            history=history,
        )
        return ChatResponse(
            reply=result["reply"],
            tool_used=result.get("tool_used", ""),
            conversation_id=request.conversation_id,
            sources=result.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat Agent"])
async def chat_stream(request: ChatRequest):
    """
    Streaming version of /chat using Server-Sent Events.

    Sends keepalive pings every 3 s while the agent processes, then emits the
    final result. Prevents Render free-tier's 30-second request timeout from
    killing long-running agent calls.

    Event types in the SSE stream:
      - {"type": "ping"}                    — keepalive, ignore in client
      - {"type": "result", "reply": ..., "tool_used": ..., "sources": [...], "conversation_id": ...}
      - {"type": "error",  "detail": ...}   — agent raised an exception
      - [DONE]                              — stream complete
    """
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    async def event_generator():
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None,
            lambda: state.agent.invoke(
                user_message=request.message,
                history=history,
            ),
        )

        # Send keepalive pings every 3 s while the agent thread runs.
        # asyncio.shield keeps the underlying Future running even when the
        # wait_for times out and cancels its wrapper.
        while not future.done():
            try:
                await asyncio.wait_for(asyncio.shield(future), timeout=3.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        # Agent finished — emit result or error
        try:
            result = future.result()
            payload = {
                "type": "result",
                "reply": result["reply"],
                "tool_used": result.get("tool_used", ""),
                "sources": result.get("sources", []),
                "conversation_id": request.conversation_id,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # tell Render's nginx not to buffer SSE
            "Connection": "keep-alive",
        },
    )
