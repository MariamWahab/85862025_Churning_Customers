import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from keras.models import load_model
import numpy as np

# Load the trained model
best_model_path = "bestmodel.h5"
best_model = load_model(best_model_path)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    
    # Replace empty strings with NaN and fill with mean for numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)

    # Mapping categorical columns
    cat_cols = {
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'OnlineSecurity': ['Yes', 'No'],
        'TechSupport': ['Yes', 'No'],
        'OnlineBackup': ['Yes', 'No'],
        'gender': ['Male', 'Female'],
        'InternetService': ['DSL', 'Fiber optic']
    }

    # Encode categorical columns
    label_encoders = {}
    for col, values in cat_cols.items():
        df[col] = df[col].apply(lambda x: values.index(x) if x in values else -1)  # Encode categorical columns
        label_encoders[col] = values

    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(df)
    return scaled_data


# Function to make predictions
def predict_churn(data):
    scaled_data = preprocess_input(data)
    predictions = best_model.predict(scaled_data)
    churn_probability = predictions[0][0]

    if churn_probability > 0.5:
        prediction = "Churn"
        confidence = churn_probability
    else:
        prediction = "No Churn"
        confidence = 1 - churn_probability

    return prediction, confidence

# Streamlit app
def main():
    st.title('Customer Churn Prediction')

    # Create input fields for user data
    st.write('Enter new customer data:')
    tenure = st.number_input('Tenure (Months):', min_value=0)
    monthly_charges = st.number_input('Monthly Charges:', min_value=0.0)
    total_charges = st.number_input('Total Charges:', min_value=0.0)
    contract = st.selectbox('Contract:', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method:', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    online_security = st.selectbox('Online Security:', ['Yes', 'No'])
    tech_support = st.selectbox('Tech Support:', ['Yes', 'No'])
    online_backup = st.selectbox('Online Backup:', ['Yes', 'No'])
    gender = st.selectbox('Gender:', ['Male', 'Female'])
    internet_service = st.selectbox('Internet Service:', ['DSL', 'Fiber optic'])

    user_input = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'OnlineSecurity': [online_security],
        'TechSupport': [tech_support],
        'OnlineBackup': [online_backup],
        'gender': [gender],
        'InternetService': [internet_service]
    })

    if st.button("Predict Churn"):
        prediction, confidence = predict_churn(user_input)
       

        st.write(f'Prediction: {prediction}, Confidence: {confidence}')

if __name__ == '__main__':
    main()
