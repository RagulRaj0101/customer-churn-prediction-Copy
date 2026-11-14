
import streamlit as st
import pandas as pd
import joblib
import json
import os

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# LOAD MODEL AND CONFIGURATION
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "model_config.json")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.json")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    with open(FEATURE_PATH, "r") as file:
        features = json.load(file)

    return model, config, features

model, config, feature_columns = load_model()
threshold = config["threshold"]

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("📊 Telecom Customer Churn Prediction")
st.write(
    "Predict customer churn probability and identify "
    "customers who may need retention support."
)

st.caption(
    "Machine Learning Model: Logistic Regression | "
    f"Decision Threshold: {threshold:.2f}"
)

st.divider()

# ------------------------------------------------
# CUSTOMER INPUT FORM
# ------------------------------------------------

st.sidebar.header("👤 Customer Information")

with st.sidebar.form("customer_form"):

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service", ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=20000.0,
        value=float(monthly_charges * tenure),
        step=10.0
    )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if submitted:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame(
        [customer_data],
        columns=feature_columns
    )

    churn_probability = model.predict_proba(input_df)[0][1]

    predicted_churn = int(churn_probability >= threshold)

    # Illustrative risk bands for dashboard display
    if churn_probability < 0.30:
        risk = "Low"
    elif churn_probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    # ------------------------------------------------
    # RESULTS
    # ------------------------------------------------

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )

    col2.metric(
        "Risk Category",
        risk
    )

    col3.metric(
        "Model Prediction",
        "Likely to Churn" if predicted_churn else "Likely to Stay"
    )

    st.progress(float(churn_probability))

    st.write(
        f"**Decision threshold:** {threshold:.2f}"
    )

    # ------------------------------------------------
    # RETENTION RECOMMENDATIONS
    # ------------------------------------------------

    st.subheader("💡 Retention Recommendations")

    recommendations = []

    if tenure <= 12:
        recommendations.append(
            "New customer: provide onboarding support "
            "and an early loyalty offer."
        )

    if contract == "Month-to-month":
        recommendations.append(
            "Offer a suitable one-year or two-year "
            "contract incentive."
        )

    if tech_support == "No" and internet_service != "No":
        recommendations.append(
            "Offer technical support assistance or "
            "a support plan."
        )

    if online_security == "No" and internet_service != "No":
        recommendations.append(
            "Explain available online security services."
        )

    if monthly_charges >= 80:
        recommendations.append(
            "Review the customer's plan and offer a "
            "suitable lower-cost alternative if available."
        )

    if payment_method == "Electronic check":
        recommendations.append(
            "Explain the convenience of automatic payment options."
        )

    if not recommendations:
        recommendations.append(
            "Continue regular customer engagement and "
            "monitor satisfaction."
        )

    for recommendation in recommendations:
        st.write("•", recommendation)

    st.caption(
        "Recommendations are rule-based suggestions, not "
        "guaranteed interventions or causal conclusions."
    )

else:

    st.info(
        "Enter customer information in the sidebar and "
        "click Predict Customer Churn."
    )

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction Project | "
    "Built with Python, Scikit-learn and Streamlit"
)

