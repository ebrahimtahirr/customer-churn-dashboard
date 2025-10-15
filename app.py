# =========================================================
# 📊 Customer Churn Prediction Dashboard
# =========================================================
import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# ✅ Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide",
    page_icon="📈"
)

# ---------------------------------------------------------
# 🧠 App Title
# ---------------------------------------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.write("""
This interactive dashboard uses a **Logistic Regression model**
to predict the likelihood of customer churn based on a few key inputs.
""")

# ---------------------------------------------------------
# ⚙️ Load Model (cached to prevent reloading)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("churn_model.joblib")
        return model
    except Exception as e:
        st.error(f"❌ Model failed to load: {e}")
        return None

model = load_model()

# ---------------------------------------------------------
# 🧮 Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("Enter Customer Information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
total = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

# Encode categorical values
senior_val = 1 if senior == "Yes" else 0
partner_val = 1 if partner == "Yes" else 0
dependents_val = 1 if dependents == "Yes" else 0

# ---------------------------------------------------------
# 🔮 Prediction Logic
# ---------------------------------------------------------
if st.button("🔮 Predict Churn"):
    if model is None:
        st.error("Model not loaded. Please ensure churn_model.joblib is in the repo.")
    else:
        # Build single-row dataframe for prediction
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
                st.error(f"⚠️ High Risk of Churn  \n**Probability:** {prob:.2f}")
            else:
                st.success(f"✅ Low Risk of Churn  \n**Probability:** {prob:.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------
# ℹ️ Footer
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**  
Built with Streamlit + Scikit-learn  
Model: Logistic Regression  
Author: Ebrahim Tahir
""")
