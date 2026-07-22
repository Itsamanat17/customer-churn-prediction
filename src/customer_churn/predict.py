"""Load a trained churn model and predict customer risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from customer_churn.config import FEATURE_COLUMNS, MODEL_PATH, SAMPLE_CUSTOMER


def risk_level(probability: float) -> str:
    """Map a churn probability into a business-friendly risk level."""
    if probability > 0.60:
        return "High"
    if probability > 0.30:
        return "Moderate"
    return "Low"


def recommendation(probability: float) -> str:
    """Return a retention recommendation for the predicted probability."""
    level = risk_level(probability)
    if level == "High":
        return "Offer retention incentive and priority support."
    if level == "Moderate":
        return "Monitor engagement and send targeted offers."
    return "Continue standard engagement."


def load_customer_data(input_file: str | Path | None = None, input_json: str | None = None) -> pd.DataFrame:
    """Load customer records from JSON, CSV, or the default sample profile."""
    if input_json:
        payload: Any = json.loads(input_json)
        records = payload if isinstance(payload, list) else [payload]
        return pd.DataFrame(records)

    if input_file:
        path = Path(input_file)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        records = payload if isinstance(payload, list) else [payload]
        return pd.DataFrame(records)

    return pd.DataFrame([SAMPLE_CUSTOMER])


def validate_customer_frame(customer_frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure customer input has the exact model feature columns."""
    missing = [column for column in FEATURE_COLUMNS if column not in customer_frame.columns]
    if missing:
        raise ValueError(f"Customer input is missing required fields: {missing}")
    return customer_frame[FEATURE_COLUMNS].copy()


def predict_customers(
    customer_frame: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
) -> list[dict[str, Any]]:
    """Predict churn for one or more customers."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run python train_model.py first.")

    model = joblib.load(model_path)
    valid_frame = validate_customer_frame(customer_frame)
    probabilities = model.predict_proba(valid_frame)[:, 1]
    predictions = model.predict(valid_frame)

    output = []
    for index, probability in enumerate(probabilities):
        output.append(
            {
                "row": index,
                "prediction": "Churn" if int(predictions[index]) == 1 else "No Churn",
                "churn_probability": float(probability),
                "risk_level": risk_level(float(probability)),
                "recommendation": recommendation(float(probability)),
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for prediction."""
    parser = argparse.ArgumentParser(description="Predict churn for customer records.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to churn_model.pkl.")
    parser.add_argument("--input-file", type=Path, default=None, help="JSON or CSV customer input.")
    parser.add_argument("--input-json", type=str, default=None, help="Inline JSON customer object.")
    return parser


def main() -> None:
    """Run prediction from the command line."""
    args = build_parser().parse_args()
    customers = load_customer_data(input_file=args.input_file, input_json=args.input_json)
    predictions = predict_customers(customers, model_path=args.model_path)
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()

