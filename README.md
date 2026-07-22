# Customer Churn Prediction

Local machine learning project converted from the notebook into normal Python code. It includes data generation, preprocessing, model training, prediction, saved artifacts, and a Streamlit web app.

## Project Structure

```text
customer-churn-prediction/
  app.py                         # Local Streamlit web app
  train_model.py                 # Train and save the model
  predict_customer.py            # Predict churn from JSON/CSV input
  requirements.txt               # Python dependencies
  environment.yml                # Conda environment
  src/customer_churn/            # Reusable ML package
  examples/sample_customer.json  # Example customer input
  docs/INTERNSHIP_REPORT.md      # Internship-ready report notes
```

## Setup

```bash
cd /home/abhinavgomra/project/customer-churn-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with Conda:

```bash
cd /home/abhinavgomra/project/customer-churn-prediction
conda env create -f environment.yml
conda activate customer-churn-prediction
```

## Train The Model

```bash
python train_model.py
```

This creates:

- `data/customers.csv`
- `models/churn_model.pkl`
- `models/metrics.json`
- `assets/model_evaluation.png`

To train only Random Forest:

```bash
python train_model.py --no-xgboost
```

## Predict From The Command Line

```bash
python predict_customer.py --input-file examples/sample_customer.json
```

You can also run it without input to use the built-in sample customer:

```bash
python predict_customer.py
```

## Run The Local Web App

```bash
streamlit run app.py
```

Open this URL in your browser:

```text
http://localhost:8501
```

If port `8501` is busy:

```bash
streamlit run app.py --server.port 8502
```

## Internship Summary

This project demonstrates an end-to-end customer churn prediction workflow:

- Synthetic telecom dataset generation
- Data cleaning and feature preprocessing
- Train-test split with stratification
- Random Forest and XGBoost model comparison
- ROC-AUC based best-model selection
- Model serialization with `joblib`
- CLI and local web app prediction interfaces
- Business recommendation based on churn probability
