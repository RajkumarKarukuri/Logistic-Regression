
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained logistic regression model
model = joblib.load("logreg_model.pkl")

# Title and description
st.title("🚢 Titanic Survival Prediction App")
st.write("This app predicts if a passenger survived the Titanic disaster based on input features.")

# User input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Map categorical values to numeric codes (must match how you trained the model)
sex_encoded = 1 if sex == "male" else 0  # Assuming male=1, female=0 during training
embarked_mapping = {"C": 0, "Q": 1, "S": 2}  # Adjust based on actual encoding
embarked_encoded = embarked_mapping[embarked]

# Create input DataFrame matching model expectations
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_encoded]
})

# Make prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    st.write("🎉 Survived" if prediction == 1 else "💀 Did not survive")
    st.write(f"Survival Probability: {prediction_proba:.2%}")
