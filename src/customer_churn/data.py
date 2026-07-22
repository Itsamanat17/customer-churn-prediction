"""Data loading and synthetic customer churn data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from customer_churn.config import DATA_PATH, RANDOM_STATE


def generate_synthetic_churn_data(
    rows: int = 5000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a realistic synthetic telecom customer churn dataset."""
    np.random.seed(random_state)

    gender = np.random.choice(["Male", "Female"], rows)
    senior_citizen = np.random.choice([0, 1], rows, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], rows, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], rows, p=[0.30, 0.70])

    tenure = np.random.exponential(scale=24, size=rows).clip(0, 72).astype(int)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        rows,
        p=[0.55, 0.24, 0.21],
    )
    paperless_billing = np.random.choice(["Yes", "No"], rows, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], rows, p=[0.34, 0.44, 0.22])
    phone_service = np.random.choice(["Yes", "No"], rows, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        np.random.choice(["Yes", "No"], rows, p=[0.42, 0.58]),
    )
    online_security = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], rows, p=[0.29, 0.71]),
    )
    tech_support = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], rows, p=[0.29, 0.71]),
    )
    streaming_tv = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], rows, p=[0.38, 0.62]),
    )

    base_charge = np.where(
        internet_service == "Fiber optic",
        70,
        np.where(internet_service == "DSL", 45, 20),
    )
    monthly_charges = (
        base_charge
        + np.random.normal(15, 10, rows)
        + (multiple_lines == "Yes") * 8
        + (streaming_tv == "Yes") * 10
    ).clip(18, 120).round(2)
    total_charges = (monthly_charges * tenure + np.random.normal(0, 20, rows)).clip(0, None).round(2)

    logit = (
        -1.6
        + 1.9 * (contract == "Month-to-month")
        + 0.55 * (contract == "One year")
        - 0.035 * tenure
        + 0.012 * (monthly_charges - 60)
        + 0.55 * (internet_service == "Fiber optic")
        + 0.45 * (payment_method == "Electronic check")
        + 0.35 * (paperless_billing == "Yes")
        - 0.5 * (online_security == "Yes")
        - 0.5 * (tech_support == "Yes")
        - 0.3 * (partner == "Yes")
        - 0.25 * (dependents == "Yes")
        + 0.3 * (senior_citizen == 1)
        + np.random.normal(0, 0.6, rows)
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(rows) < churn_probability).astype(int)

    return pd.DataFrame(
        {
            "customerID": [f"C{10000 + index}" for index in range(rows)],
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def load_or_generate_dataset(
    data_path: str | Path | None = None,
    rows: int = 5000,
    random_state: int = RANDOM_STATE,
    save_generated: bool = True,
) -> pd.DataFrame:
    """Load a CSV dataset or generate one when no path is provided."""
    if data_path:
        return pd.read_csv(data_path)

    dataset = generate_synthetic_churn_data(rows=rows, random_state=random_state)
    if save_generated:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(DATA_PATH, index=False)
    return dataset
