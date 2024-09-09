import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

# Load data
data = pd.read_csv(r'C:\Users\yenbing\OneDrive\Documents\AI Assignment\diabetes_prediction_dataset.csv')

# Split data into features (X) and target variable (y)
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model for later use
dump(model, 'test.joblib')  # Save the model after training

# --- Streamlit UI ---

st.title('Diabetes Prediction App')

# Create UI elements for user input
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', ['Male', 'Female'])
hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
smoking_history = st.selectbox('Smoking History', ['Never', 'Current', 'Former'])
bmi = st.number_input('BMI', min_value=15, max_value=60)
hb_a1c_level = st.number_input('HbA1c Level', min_value=3, max_value=15)
blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, max_value=500)

# Button to trigger prediction
if st.button('Predict'):

    # Create a new data point based on user inputs
    new_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'hypertension': [1 if hypertension == 'Yes' else 0],
        'heart_disease': [1 if heart_disease == 'Yes' else 0],
        'smoking_history': [1 if smoking_history == 'Current' else (2 if smoking_history == 'Former' else 0)],
        'bmi': [bmi],
        'HbA1c_level': [hb_a1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

    # Load the trained model
    loaded_model = load('test.joblib')

    # Make a prediction
    prediction = loaded_model.predict(new_data)

    # Display the prediction with informative messages
    if prediction[0] == 1:
        st.error('Predicted: You are at a higher risk of diabetes. Please consult a healthcare professional.')
    else:
        st.success('Predicted: You are at a lower risk of diabetes based on this model. It is still recommended to maintain a healthy lifestyle and consult a healthcare professional for regular checkups.')
