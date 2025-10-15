import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Analyze and predict customer churn using machine-learning insights.")

# -------------------------------------------------------
# Cache model so it loads once
# -------------------------------------------------------
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------------------------------
# Sidebar inputs
# -------------------------------------------------------
st.sidebar.header("Enter Customer Information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
total = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

# Encode categorical values like training
senior_val = 1 if senior == "Yes" else 0
partner_val = 1 if partner == "Yes" else 0
dependents_val = 1 if dependents == "Yes" else 0

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
if st.button("🔮 Predict Churn"):
    X_input = pd.DataFrame([{
        'SeniorCitizen': senior_val,
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'Partner_Yes': partner_val,
        'Dependents_Yes': dependents_val
    }])

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    st.subheader("📈 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk of Churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Churn (Probability: {prob:.2f})")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit + Logistic Regression model")
