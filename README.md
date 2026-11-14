# Telecom Customer Churn Prediction

## Overview
A machine learning application that predicts whether a telecom customer is likely to churn.

## Features
- Customer churn prediction
- Logistic Regression model
- Churn probability estimation
- Low, Medium and High risk categorization
- Rule-based customer retention recommendations
- Interactive Streamlit dashboard

## Model Performance
- Accuracy: 76%
- Churn Precision: 54%
- Churn Recall: 71%
- Churn F1-Score: 61%
- ROC-AUC: 0.8416

## Technology Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure
- dashboard/: Streamlit application
- models/: Trained model and configuration
- reports/: Project reports
- requirements.txt: Python dependencies

## Run Locally
Install dependencies:

pip install -r requirements.txt

Start the dashboard:

streamlit run dashboard/app.py

## Disclaimer
Predictions are estimates based on the trained model and should not be treated as guaranteed outcomes.
