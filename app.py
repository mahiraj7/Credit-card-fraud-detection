import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")
features = joblib.load("fraud_features.pkl")

st.title("Credit card Fraud Detect using Isolation Forest")
st.write("Enter Transaction details to check whether it is Normal or Fraud")

time = st.number_input("Transaction Time", value=10000.0)
amount = st.number_input("Transaction Amount", value=100.0)

st.write("Enter V1 to V28 values. For beginner testing, keep default values.")

v_values = {}

for i in range(1, 29):
    v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

input_dict={
    "Time" : time,
    "Amount" : amount,
}

input_dict.update(v_values)

input_data = pd.DataFrame([input_dict])

input_data = input_data[features]

if st.button("Predict Transaction"):
    prediction = model.predict(input_data)

    if prediction[0] == -1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Normal Transaction")
