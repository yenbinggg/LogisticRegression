import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from joblib import dump, load

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data from uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Handle missing values (if any)
    imputer = SimpleImputer(strategy='mean')  # You can change strategy as needed

    # Assuming the columns 'gender' and 'smoking_history' are categorical
    label_encoders = {}
    
    # Encode categorical columns
    for col in ['gender', 'smoking_history']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Store the encoder if needed later for inverse transform

    # Handle missing values for numerical columns
    data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = imputer.fit_transform(
        data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    )

    # Split data into features (X) and target variable (y)
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model for later use
    dump(model, 'test.joblib')

    # Load the trained model
    loaded_model = load('test.joblib')

    # --- Streamlit UI ---
    st.title('Diabetes Prediction App')

    # Ensure that selectbox options match the ones used during training
    gender_options = label_encoders['gender'].classes_.tolist()
    smoking_history_options = label_encoders['smoking_history'].classes_.tolist()

    # Create UI elements for user input
    age = st.number_input('Age', min_value=18, max_value=100)
    gender = st.selectbox('Gender', gender_options)
    hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
    heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
    smoking_history = st.selectbox('Smoking History', smoking_history_options)
    bmi = st.number_input('BMI', min_value=15, max_value=60)
    hb_a1c_level = st.number_input('HbA1c Level', min_value=3, max_value=15)
    blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, max_value=500)

    # Button to trigger prediction
    if st.button('Predict'):
        # Convert categorical data from user input into the same format used during training
        gender_encoded = label_encoders['gender'].transform([gender])[0]
        smoking_history_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]

        # Create a new data point based on user inputs
        new_data = pd.DataFrame({
            'age': [age], 
            'gender': [gender_encoded],  
            'hypertension': [1 if hypertension == 'Yes' else 0],
            'heart_disease': [1 if heart_disease == 'Yes' else 0],
            'smoking_history': [smoking_history_encoded],
            'bmi': [bmi],  
            'HbA1c_level': [hb_a1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })

        # Make a prediction
        prediction = loaded_model.predict(new_data)

        # Display the prediction with informative messages
        if prediction[0] == 1:
            st.error('Predicted: You are at a higher risk of diabetes. Please consult a healthcare professional.')
        else:
            st.success('Predicted: You are at a lower risk of diabetes based on this model. It is still recommended to maintain a healthy lifestyle and consult a healthcare professional for regular checkups.')
