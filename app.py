import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model.pkl')

st.title("Stroke Prediction Web App")
st.write("Enter patient details to predict the risk of stroke")

# Input fields (adjust according to your dataset features)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
avg_glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Map categorical input if necessary
smoking_mapping = {"never smoked":0, "formerly smoked":1, "smokes":2, "Unknown":3}
smoking_status = smoking_mapping[smoking_status]

# Collect features in the same order as your training
features = pd.DataFrame([[age, hypertension, heart_disease, avg_glucose, bmi, smoking_status]],
                        columns=['age','hypertension','heart_disease','avg_glucose_level','bmi','smoking_status'])

# Predict
if st.button("Predict Stroke Risk"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # probability of stroke
    st.write(f"Prediction: {'Stroke Risk' if prediction==1 else 'No Stroke Risk'}")
    st.write(f"Prediction Probability: {probability:.2f}")
