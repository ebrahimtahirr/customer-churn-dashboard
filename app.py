import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

# ------------------------------------------
# UI loads FIRST
# ------------------------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Analyze and predict customer churn using a simple Logistic Regression model.")

# ------------------------------------------
# Cached Model Loader
# ------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("churn_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Model failed to load: {e}")
        return None

model = load_model()

# ------------------------------------------
# Sidebar Inputs
# ------------------------------------------
st.sidebar.header("Enter Customer Information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
total = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

# Encode categorical variables
senior_val = 1 if senior == "Yes" else 0
partner_val = 1 if partner == "Yes" else 0
dependents_val = 1 if dependents == "Yes" else 0

# ------------------------------------------
# Prediction
# ------------------------------------------
if st.button("🔮 Predict Churn"):
    if model is None:
        st.error("Model not loaded. Please check if churn_model.pkl exists.")
    else:
        X_input = pd.DataFrame([{
            'SeniorCitizen': senior_val,
            'tenure': tenure,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'Partner_Yes': partner_val,
            'Dependents_Yes': dependents_val
        }])

        try:
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][1]

            st.subheader("📈 Prediction Result")
            if pred == 1:
                st.error(f"⚠️ High Risk of Churn (Probability: {prob:.2f})")
            else:
                st.success(f"✅ Low Risk of Churn (Probability: {prob:.2f})")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit + Logistic Regression model")
