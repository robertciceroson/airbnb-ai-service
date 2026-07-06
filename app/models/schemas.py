"""
Pydantic request/response schemas for all API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    borough: str = Field(..., example="Manhattan")
    neighbourhood: str = Field(..., example="Midtown")
    room_type: str = Field(..., example="Entire home/apt")
    minimum_nights: int = Field(default=2, ge=1, le=365)
    availability_365: int = Field(default=200, ge=0, le=365)
    number_of_reviews: int = Field(default=20, ge=0)
    reviews_per_month: float = Field(default=1.0, ge=0.0)
    calculated_host_listings_count: int = Field(default=1, ge=1)
    checkin_month: Optional[int] = Field(
        default=None,
        ge=1, le=12,
        description="Month number (1–12) for seasonal adjustment. Omit for base price only."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictResponse(BaseModel):
    base_price: float = Field(..., description="Raw XGBoost prediction (no seasonal adjustment)")
    adjusted_price: float = Field(..., description="Price after seasonal multiplier")
    seasonal_multiplier: float
    season_label: str
    neighbourhood_median: Optional[float] = Field(None, description="Median nightly price for this neighbourhood")
    currency: str = "USD"


# ── /chat ─────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str = Field(
        ...,
        description="Client-generated UUID — used to maintain conversation memory across turns."
    )
    history: list[Message] = Field(
        default_factory=list,
        description="Prior turns in this conversation (most recent last). Max 20 messages."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What is Airbnb's cancellation policy for a flexible listing?",
                "conversation_id": "conv-abc-123",
                "history": []
            }
        }
    }


class ChatResponse(BaseModel):
    reply: str
    tool_used: Optional[str] = Field(None, description="Which tool the agent invoked: price | policy | handoff | none")
    conversation_id: str
    sources: list[str] = Field(default_factory=list, description="Policy doc sources cited (if policy tool was used)")


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_store_loaded: bool
    version: str
