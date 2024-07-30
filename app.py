import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('diabetis_classifier.sav')

# Create a function to predict diabetes
def predict_diabetes(input_data):
    input_data = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title("Diabetes Prediction")

# Create input fields for the features
st.sidebar.header("Input Features")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose", min_value=60, max_value=200, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=35, max_value=122, value=65)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=99, value=0)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=846, value=0)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=67.1, value=0.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
age = st.sidebar.number_input("Age", min_value=21, max_value=81, value=21)

# Store inputs into a dictionary
input_data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}

# When the user clicks the Predict button
if st.button("Predict"):
    result = predict_diabetes(input_data)
    if result == 1:
        st.write("The model predicts that the person has diabetes.")
    else:
        st.write("The model predicts that the person does not have diabetes.")
