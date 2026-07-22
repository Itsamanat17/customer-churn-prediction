"""Streamlit app for local customer churn prediction."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from customer_churn.config import FORM_OPTIONS, METRICS_PATH, MODEL_PATH, SAMPLE_CUSTOMER
from customer_churn.predict import predict_customers
from customer_churn.train import run_training


@st.cache_resource(show_spinner=False)
def ensure_model_exists() -> Path:
    """Train a default model if no saved model is available."""
    if not MODEL_PATH.exists():
        run_training(include_xgboost=False)
    return MODEL_PATH


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any] | None:
    """Load saved model metrics for the dashboard."""
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def build_customer_form() -> tuple[dict[str, Any], bool]:
    """Collect customer fields from Streamlit widgets."""
    customer = dict(SAMPLE_CUSTOMER)

    with st.form("customer_form"):
        left, middle, right = st.columns(3)

        with left:
            customer["gender"] = st.selectbox("Gender", FORM_OPTIONS["gender"], index=0)
            customer["SeniorCitizen"] = st.selectbox("Senior Citizen", FORM_OPTIONS["SeniorCitizen"], index=0)
            customer["Partner"] = st.selectbox("Partner", FORM_OPTIONS["Partner"], index=0)
            customer["Dependents"] = st.selectbox("Dependents", FORM_OPTIONS["Dependents"], index=1)
            customer["tenure"] = st.number_input("Tenure", min_value=0, max_value=72, value=6)

        with middle:
            customer["PhoneService"] = st.selectbox("Phone Service", FORM_OPTIONS["PhoneService"], index=0)
            customer["MultipleLines"] = st.selectbox("Multiple Lines", FORM_OPTIONS["MultipleLines"], index=0)
            customer["InternetService"] = st.selectbox(
                "Internet Service",
                FORM_OPTIONS["InternetService"],
                index=1,
            )
            customer["OnlineSecurity"] = st.selectbox("Online Security", FORM_OPTIONS["OnlineSecurity"], index=0)
            customer["TechSupport"] = st.selectbox("Tech Support", FORM_OPTIONS["TechSupport"], index=0)

        with right:
            customer["StreamingTV"] = st.selectbox("Streaming TV", FORM_OPTIONS["StreamingTV"], index=0)
            customer["Contract"] = st.selectbox("Contract", FORM_OPTIONS["Contract"], index=0)
            customer["PaperlessBilling"] = st.selectbox(
                "Paperless Billing",
                FORM_OPTIONS["PaperlessBilling"],
                index=0,
            )
            customer["PaymentMethod"] = st.selectbox("Payment Method", FORM_OPTIONS["PaymentMethod"], index=0)
            customer["MonthlyCharges"] = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=65.0,
                step=1.0,
            )
            customer["TotalCharges"] = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=390.0,
                step=10.0,
            )

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    return customer, submitted


def render_metrics(metrics: dict[str, Any] | None) -> None:
    """Show saved training metrics."""
    if not metrics:
        st.info("Model metrics will appear after training finishes.")
        return

    best_model = metrics["best_model"]
    best_metrics = metrics["results"][best_model]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Best Model", best_model)
    metric_columns[1].metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
    metric_columns[2].metric("Accuracy", f"{best_metrics['accuracy']:.3f}")
    metric_columns[3].metric("Churn Rate", f"{metrics['churn_rate']:.1%}")


def render_prediction(customer: dict[str, Any]) -> None:
    """Run prediction and render the business result."""
    results = predict_customers(pd.DataFrame([customer]), model_path=MODEL_PATH)
    prediction = results[0]
    probability = prediction["churn_probability"]
    percentage = probability * 100

    if prediction["risk_level"] == "High":
        st.error(f"High Risk: {percentage:.1f}% churn probability")
    elif prediction["risk_level"] == "Moderate":
        st.warning(f"Moderate Risk: {percentage:.1f}% churn probability")
    else:
        st.success(f"Low Risk: {percentage:.1f}% churn probability")

    result_columns = st.columns(3)
    result_columns[0].metric("Prediction", prediction["prediction"])
    result_columns[1].metric("Risk Level", prediction["risk_level"])
    result_columns[2].metric("Probability", f"{percentage:.1f}%")
    st.write(f"Recommendation: {prediction['recommendation']}")


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Customer Churn Prediction",
        layout="wide",
    )

    st.title("Customer Churn Prediction")
    st.caption("Local internship ML app built with Streamlit")

    with st.spinner("Checking trained model..."):
        ensure_model_exists()

    render_metrics(load_metrics())
    st.divider()

    st.subheader("Customer Profile")
    customer, submitted = build_customer_form()

    st.subheader("Prediction Result")
    if submitted:
        render_prediction(customer)
    else:
        st.info("Fill the customer profile and click Predict Churn.")

    with st.expander("Current customer input"):
        st.json(customer)


if __name__ == "__main__":
    main()
