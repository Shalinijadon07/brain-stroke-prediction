import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load("models/best_model.pkl")

st.title("Brain Stroke Prediction App")

st.write("Enter patient details to predict stroke risk:")

# --- Input fields (MUST match training order!) ---

gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married?", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

age = st.number_input("Age", min_value=0, max_value=120, value=45)
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])


# --- Convert inputs EXACTLY like training notebook ---

gender_map = {"Male": 1, "Female": 0}
ever_married_map = {"No": 0, "Yes": 1}
work_type_map = {"children": 0, "Govt_job": 1, "Never_worked": 2, "Private": 3, "Self-employed": 4}
residence_map = {"Rural": 0, "Urban": 1}
smoking_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}
binary_map = {"No": 0, "Yes": 1}

# Final ordered feature list based on training:
features = np.array([[
    gender_map[gender],            # Cat
    ever_married_map[ever_married],# Cat
    work_type_map[work_type],      # Cat
    residence_map[residence_type], # Cat
    smoking_map[smoking_status],   # Cat
    binary_map[hypertension],      # Cat
    binary_map[heart_disease],     # Cat
    age,                           # Numeric
    avg_glucose_level,             # Numeric
    bmi                            # Numeric
]])

# --- Predict ---
if st.button("Predict Stroke Risk"):
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.error("⚠ High Risk of Stroke")
    else:
        st.success("✔ Low Risk of Stroke")
