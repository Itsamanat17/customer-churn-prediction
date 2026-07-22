# Customer Churn Prediction - Internship Project Report

## Objective

This project predicts whether a telecom customer is likely to churn. It helps a business identify high-risk customers early and take retention actions such as discounts, loyalty offers, support follow-up, or contract upgrade incentives.

## Dataset

The project uses a synthetic customer churn dataset with 5,000 customer records. The generated data includes demographics, services, account tenure, billing behavior, payment method, contract type, and churn status.

Target variable:

- `Churn = 0`: customer retained
- `Churn = 1`: customer churned

## Preprocessing

The training pipeline performs these steps:

- Drops `customerID`
- Converts `TotalCharges` to numeric values
- Fills missing `TotalCharges` values with the median
- Converts `Churn` from `Yes`/`No` into `1`/`0`
- Scales numerical features with `StandardScaler`
- Encodes categorical features with `OneHotEncoder`
- Uses a single scikit-learn `Pipeline` for preprocessing and model training

## Models

The project trains and compares:

- Random Forest Classifier
- XGBoost Classifier, when the `xgboost` package is installed

The best model is selected using ROC-AUC and saved to `models/churn_model.pkl`.

## Business Output

For each customer, the app returns:

- Churn classification
- Churn probability
- Risk level: Low, Moderate, or High
- Retention recommendation

## Deliverables

- Reusable Python package in `src/customer_churn`
- CLI training script: `python train_model.py`
- CLI prediction script: `python predict_customer.py`
- Local Streamlit web app: `streamlit run app.py`
- Saved metrics in `models/metrics.json`
- Evaluation chart in `assets/model_evaluation.png`
