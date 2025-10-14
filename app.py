import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Analyze and predict customer churn using machine-learning insights.")

# Sidebar inputs
st.sidebar.header("Enter Customer Information")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
total = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 1500.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
internet = st.sidebar.selectbox("Internet Service", ["DSL","Fiber optic","No"])

# Build input row
input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly],
    'TotalCharges': [total],
    'Contract_One year': [1 if contract=="One year" else 0],
    'Contract_Two year': [1 if contract=="Two year" else 0],
    'PaymentMethod_Electronic check': [1 if payment=="Electronic check" else 0],
    'InternetService_Fiber optic': [1 if internet=="Fiber optic" else 0],
    'InternetService_No': [1 if internet=="No" else 0],
})

# Align columns with model training set
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

# Predict
if st.sidebar.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.error(f"⚠️ High Risk of Churn ({prob*100:.1f}% probability)")
    else:
        st.success(f"✅ Low Risk of Churn ({prob*100:.1f}% probability)")
