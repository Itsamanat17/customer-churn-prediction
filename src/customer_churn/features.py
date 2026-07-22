"""Preprocessing utilities for churn prediction."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_churn.config import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def clean_churn_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Clean raw churn data and encode the target column."""
    cleaned = dataset.copy()

    for column in DROP_COLUMNS:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=[column])

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["TotalCharges"].median())

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Dataset must include a '{TARGET_COLUMN}' column.")

    if not is_numeric_dtype(cleaned[TARGET_COLUMN]):
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    if cleaned[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains values outside Yes/No or 1/0.")

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataset into model features and target values."""
    missing_features = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required feature columns: {missing_features}")

    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]
    return features, target


def build_preprocessor(features: pd.DataFrame | None = None) -> ColumnTransformer:
    """Build the preprocessing pipeline for numeric and categorical fields."""
    if features is None:
        numeric_features = NUMERIC_FEATURES
        categorical_features = CATEGORICAL_FEATURES
    else:
        numeric_features = [column for column in NUMERIC_FEATURES if column in features.columns]
        categorical_features = [column for column in CATEGORICAL_FEATURES if column in features.columns]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
