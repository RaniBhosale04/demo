import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Title and Description
st.title("Diabetes Prediction App")
st.write("Enter the patient's medical details below to predict the likelihood of diabetes.")

# Create input fields for the features required by the model
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=79.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)

with col2:
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    age = st.number_input("Age", min_value=1, max_value=120, value=33, step=1)

# Prediction Button
if st.button("Predict"):
    # Format the input data to match the exact feature names the model expects
    input_data = pd.DataFrame([[
        pregnancies, 
        glucose, 
        blood_pressure, 
        skin_thickness, 
        insulin, 
        bmi, 
        dpf, 
        age
    ]], columns=[
        'Pregnancies', 
        'Glucose', 
        'BloodPressure', 
        'SkinThickness', 
        'Insulin', 
        'BMI', 
        'DiabetesPedigreeFunction', 
        'Age'
    ])

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the results
    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ **Prediction: High risk of diabetes.** Please consult with a healthcare professional.")
    else:
        st.success("✅ **Prediction: Low risk of diabetes.**")
