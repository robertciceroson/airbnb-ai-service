"""
XGBoost model: train-once-save, load-on-startup, predict-on-demand.

First run:  python train_and_save_model.py
Thereafter: the FastAPI startup event loads the saved artifacts.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from app.config import settings

# ── Seasonal constants (same as existing Streamlit app) ───────────────────────
SEASONAL_MULT = {
    1: 0.88, 2: 0.90, 3: 0.97,  4: 1.05,
    5: 1.08, 6: 1.18, 7: 1.22,  8: 1.20,
    9: 1.07, 10: 1.05, 11: 0.95, 12: 1.02,
}
SEASON_LABEL = {
    1: "❄️ Off-season",    2: "❄️ Off-season",
    3: "🌸 Shoulder",      4: "🌸 Spring",        5: "🌸 Spring peak",
    6: "☀️ Summer peak",   7: "☀️ Summer peak",   8: "☀️ Summer peak",
    9: "🍂 Fall shoulder", 10: "🍂 Fall shoulder",
    11: "❄️ Off-season",  12: "🎄 Holiday",
}

FEATURES = [
    "borough_enc", "neighbourhood_enc", "room_enc",
    "latitude", "longitude",
    "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365",
]


class Predictor:
    """Loads saved model + encoders and exposes a predict() method."""

    def __init__(self):
        self.model: XGBRegressor | None = None
        self.le_borough: LabelEncoder | None = None
        self.le_neighbourhood: LabelEncoder | None = None
        self.le_room: LabelEncoder | None = None
        self.neighbourhood_coords: pd.DataFrame | None = None
        self.neighbourhood_prices: pd.Series | None = None
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self):
        if not settings.model_path.exists() or not settings.encoders_path.exists():
            raise FileNotFoundError(
                "Trained model not found. Run `python train_and_save_model.py` first."
            )
        self.model = joblib.load(settings.model_path)
        encoders = joblib.load(settings.encoders_path)
        self.le_borough          = encoders["le_borough"]
        self.le_neighbourhood    = encoders["le_neighbourhood"]
        self.le_room             = encoders["le_room"]
        self.neighbourhood_coords  = encoders["neighbourhood_coords"]
        self.neighbourhood_prices  = encoders["neighbourhood_prices"]
        print("✅ XGBoost model loaded.")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        borough: str,
        neighbourhood: str,
        room_type: str,
        minimum_nights: int,
        availability_365: int,
        number_of_reviews: int,
        reviews_per_month: float,
        calculated_host_listings_count: int,
        checkin_month: int | None = None,
    ) -> dict:
        # Encode categoricals
        borough_enc       = int(self.le_borough.transform([borough])[0])
        neighbourhood_enc = int(self.le_neighbourhood.transform([neighbourhood])[0])
        room_enc          = int(self.le_room.transform([room_type])[0])

        # Lookup average coordinates for this neighbourhood
        coords = self.neighbourhood_coords.loc[neighbourhood]
        lat, lng = float(coords["latitude"]), float(coords["longitude"])

        input_df = pd.DataFrame(
            [[borough_enc, neighbourhood_enc, room_enc,
              lat, lng, minimum_nights, number_of_reviews,
              reviews_per_month, calculated_host_listings_count, availability_365]],
            columns=FEATURES,
        )

        base_price = float(self.model.predict(input_df)[0])

        # Seasonal adjustment
        month = checkin_month or 6          # default to June if not provided
        multiplier  = SEASONAL_MULT[month]
        adj_price   = base_price * multiplier
        season_lbl  = SEASON_LABEL[month]

        # Neighbourhood median
        median = self.neighbourhood_prices.get(neighbourhood, None)

        return {
            "base_price":           round(base_price, 2),
            "adjusted_price":       round(adj_price, 2),
            "seasonal_multiplier":  multiplier,
            "season_label":         season_lbl,
            "neighbourhood_median": float(median) if median is not None else None,
            "currency":             "USD",
        }

    # ── Helpers (used by the price tool in the agent) ─────────────────────────

    @property
    def valid_boroughs(self) -> list[str]:
        return list(self.le_borough.classes_)

    def valid_neighbourhoods(self, borough: str) -> list[str]:
        return [
            n for n in self.le_neighbourhood.classes_
            if n in self.neighbourhood_coords.index
        ]
