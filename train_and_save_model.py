"""
Run once to train the XGBoost model and save artifacts to disk.

Usage:
    python train_and_save_model.py

Outputs:
    models/xgboost_model.joblib
    models/encoders.joblib
"""
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

DATA_PATH    = Path("data/listings.csv")
MODEL_OUT    = Path("models/xgboost_model.joblib")
ENCODERS_OUT = Path("models/encoders.joblib")

FEATURES = [
    "borough_enc", "neighbourhood_enc", "room_enc",
    "latitude", "longitude",
    "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365",
]

DROP_COLS = ["id", "name", "host_id", "host_profile_id", "host_name",
             "last_review", "number_of_reviews_ltm", "license"]


def main():
    print("Loading data…")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=["neighbourhood_group", "neighbourhood", "room_type"])
    df["calculated_host_listings_count"] = df["calculated_host_listings_count"].fillna(1)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = df[df["price"] > 0]
    df = df[df["price"] <= df["price"].quantile(0.995)]
    df = df[df["minimum_nights"] <= df["minimum_nights"].quantile(0.99)]

    le_borough       = LabelEncoder()
    le_neighbourhood = LabelEncoder()
    le_room          = LabelEncoder()
    df["borough_enc"]       = le_borough.fit_transform(df["neighbourhood_group"])
    df["neighbourhood_enc"] = le_neighbourhood.fit_transform(df["neighbourhood"])
    df["room_enc"]          = le_room.fit_transform(df["room_type"])

    neighbourhood_coords = df.groupby("neighbourhood")[["latitude", "longitude"]].mean()
    neighbourhood_prices = df.groupby("neighbourhood")["price"].median().round(0)

    X, y = df[FEATURES], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("Training XGBoost…")
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"  R²  = {r2_score(y_test, y_pred):.4f}")
    print(f"  MAE = ${mean_absolute_error(y_test, y_pred):.2f}")

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    joblib.dump({
        "le_borough":           le_borough,
        "le_neighbourhood":     le_neighbourhood,
        "le_room":              le_room,
        "neighbourhood_coords": neighbourhood_coords,
        "neighbourhood_prices": neighbourhood_prices,
    }, ENCODERS_OUT)

    print(f"✅ Saved model → {MODEL_OUT}")
    print(f"✅ Saved encoders → {ENCODERS_OUT}")


if __name__ == "__main__":
    main()
