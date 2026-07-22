"""Model training and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from customer_churn.config import RANDOM_STATE
from customer_churn.features import build_preprocessor


def build_models(
    target_train: pd.Series,
    random_state: int = RANDOM_STATE,
    include_xgboost: bool = True,
) -> dict[str, Any]:
    """Create candidate models used in the project."""
    models: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    }

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return models

        negative_count = int((target_train == 0).sum())
        positive_count = int((target_train == 1).sum())
        scale_pos_weight = negative_count / max(positive_count, 1)

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    return models


def evaluate_predictions(
    target_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate the classification metrics used in the report."""
    return {
        "accuracy": float(accuracy_score(target_test, predictions)),
        "precision": float(precision_score(target_test, predictions, zero_division=0)),
        "recall": float(recall_score(target_test, predictions, zero_division=0)),
        "f1": float(f1_score(target_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target_test, probabilities)),
    }


def train_and_evaluate_models(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.Series,
    target_test: pd.Series,
    random_state: int = RANDOM_STATE,
    include_xgboost: bool = True,
) -> tuple[dict[str, dict[str, float]], dict[str, Pipeline]]:
    """Train candidate models and return metrics plus fitted pipelines."""
    preprocessor = build_preprocessor(features_train)
    models = build_models(target_train, random_state=random_state, include_xgboost=include_xgboost)
    results: dict[str, dict[str, float]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("prep", clone(preprocessor)),
                ("clf", classifier),
            ]
        )
        pipeline.fit(features_train, target_train)
        predictions = pipeline.predict(features_test)
        probabilities = pipeline.predict_proba(features_test)[:, 1]

        results[name] = evaluate_predictions(target_test, predictions, probabilities)
        fitted_pipelines[name] = pipeline

    return results, fitted_pipelines


def select_best_model(results: dict[str, dict[str, float]], metric: str = "roc_auc") -> str:
    """Select the best model name from a metrics dictionary."""
    if not results:
        raise ValueError("No model results were produced.")
    return max(results, key=lambda name: results[name][metric])


def get_feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.Series:
    """Return top feature importances for tree-based classifiers."""
    preprocessor = pipeline.named_steps["prep"]
    classifier = pipeline.named_steps["clf"]

    if not hasattr(classifier, "feature_importances_"):
        return pd.Series(dtype=float)

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

