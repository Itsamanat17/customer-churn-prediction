"""Train and save the customer churn prediction model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from customer_churn.config import (
    ASSETS_DIR,
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    ensure_project_dirs,
)
from customer_churn.data import load_or_generate_dataset
from customer_churn.features import clean_churn_data, split_features_target
from customer_churn.modeling import (
    get_feature_importance,
    select_best_model,
    train_and_evaluate_models,
)


def make_json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_evaluation_plots(
    results: dict[str, dict[str, float]],
    fitted_pipelines: dict[str, Any],
    best_model_name: str,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    assets_dir: Path = ASSETS_DIR,
) -> Path:
    """Save confusion matrix, ROC curves, metrics, and feature importance."""
    os.environ.setdefault("MPLCONFIGDIR", str(assets_dir / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns

    assets_dir.mkdir(parents=True, exist_ok=True)

    best_pipeline = fitted_pipelines[best_model_name]
    predictions = best_pipeline.predict(features_test)
    matrix = confusion_matrix(target_test, predictions)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 0],
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    axes[0, 0].set_title(f"Confusion Matrix - {best_model_name}")
    axes[0, 0].set_ylabel("Actual")
    axes[0, 0].set_xlabel("Predicted")

    for name, pipeline in fitted_pipelines.items():
        probabilities = pipeline.predict_proba(features_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(target_test, probabilities)
        axes[0, 1].plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC = {results[name]['roc_auc']:.3f})",
            linewidth=2,
        )
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Baseline")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    feature_importance = get_feature_importance(best_pipeline)
    if feature_importance.empty:
        axes[1, 0].axis("off")
        axes[1, 0].set_title("Feature Importance Unavailable")
    else:
        feature_importance.sort_values().plot(kind="barh", ax=axes[1, 0], color="#277da1")
        axes[1, 0].set_title(f"Top Feature Importances - {best_model_name}")
        axes[1, 0].set_xlabel("Importance")

    results_frame = pd.DataFrame(results).T
    results_frame.plot(kind="bar", ax=axes[1, 1], colormap="Set2")
    axes[1, 1].set_title("Model Evaluation Metrics")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
    axes[1, 1].legend(loc="lower right", fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = assets_dir / "model_evaluation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def run_training(
    rows: int = 5000,
    data_path: str | Path | None = None,
    output_dir: str | Path = MODELS_DIR,
    assets_dir: str | Path = ASSETS_DIR,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    include_xgboost: bool = True,
    save_plots: bool = True,
) -> dict[str, Any]:
    """Run the complete training workflow and save model artifacts."""
    ensure_project_dirs()
    output_dir = Path(output_dir)
    assets_dir = Path(assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_or_generate_dataset(
        data_path=data_path,
        rows=rows,
        random_state=random_state,
        save_generated=data_path is None,
    )
    cleaned = clean_churn_data(dataset)
    features, target = split_features_target(cleaned)

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    results, fitted_pipelines = train_and_evaluate_models(
        features_train,
        features_test,
        target_train,
        target_test,
        random_state=random_state,
        include_xgboost=include_xgboost,
    )
    best_model_name = select_best_model(results)
    best_pipeline = fitted_pipelines[best_model_name]

    model_path = output_dir / MODEL_PATH.name
    metrics_path = output_dir / METRICS_PATH.name
    joblib.dump(best_pipeline, model_path)

    predictions = best_pipeline.predict(features_test)
    report = classification_report(
        target_test,
        predictions,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )

    plot_path = None
    if save_plots:
        plot_path = save_evaluation_plots(
            results=results,
            fitted_pipelines=fitted_pipelines,
            best_model_name=best_model_name,
            features_test=features_test,
            target_test=target_test,
            assets_dir=assets_dir,
        )

    summary = {
        "best_model": best_model_name,
        "results": results,
        "classification_report": report,
        "dataset_rows": int(len(dataset)),
        "churn_rate": float(target.mean()),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "data_path": str(DATA_PATH if data_path is None else Path(data_path)),
        "plot_path": str(plot_path) if plot_path else None,
    }

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(make_json_safe(summary), metrics_file, indent=2)

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for model training."""
    parser = argparse.ArgumentParser(description="Train a customer churn prediction model.")
    parser.add_argument("--rows", type=int, default=5000, help="Synthetic rows to generate.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional CSV dataset path.")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR, help="Folder for model files.")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR, help="Folder for plots.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument("--no-xgboost", action="store_true", help="Train only Random Forest.")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving evaluation plots.")
    return parser


def main() -> None:
    """Run training from the command line."""
    args = build_parser().parse_args()
    summary = run_training(
        rows=args.rows,
        data_path=args.data_path,
        output_dir=args.output_dir,
        assets_dir=args.assets_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        include_xgboost=not args.no_xgboost,
        save_plots=not args.no_plots,
    )

    print("Training complete")
    print(f"Best model: {summary['best_model']}")
    print(f"Churn rate: {summary['churn_rate']:.2%}")
    print("Metrics:")
    for name, metrics in summary["results"].items():
        formatted = ", ".join(f"{metric}={value:.4f}" for metric, value in metrics.items())
        print(f"  {name}: {formatted}")
    print(f"Saved model: {summary['model_path']}")
    print(f"Saved metrics: {summary['metrics_path']}")
    if summary["plot_path"]:
        print(f"Saved plot: {summary['plot_path']}")


if __name__ == "__main__":
    main()

