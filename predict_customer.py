"""Convenience entry point for prediction from the project root."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from customer_churn.predict import main


if __name__ == "__main__":
    main()

