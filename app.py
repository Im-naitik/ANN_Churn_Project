import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# Load model
model = tf.keras.models.load_model("model.h5")

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# App title
st.title("Customer Churn Prediction")

# User inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)

age = st.slider("Age", 18, 92, 35)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

tenure = st.slider("Tenure", 0, 10, 5)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Base input
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# Encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Combine input data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)],
    axis=1
)

# Add missing scaler columns automatically
for col in scaler.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Remove extra columns and arrange correct order
input_data = input_data[scaler.feature_names_in_]

# Scale input
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_probability = float(prediction[0][0])

# Output
st.subheader("Prediction Result")

if prediction_probability > 0.5:
    st.error("The customer is likely to churn")
else:
    st.success("The customer is not likely to churn")

st.write(f"Churn probability: {prediction_probability:.2%}")
