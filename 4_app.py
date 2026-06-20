import streamlit as st
import joblib
import pandas as pd
# Load model
model = joblib.load("knn_customer_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("Customer Segmentation using K-NN")
gender = st.selectbox( "Gender",["Female","Male"])
age = st.number_input("Age",18,70)
income = st.number_input("Annual Income",10000,200000)
score = st.number_input("Spending Score",1,100)
if st.button("Predict"):
    gender_value = 0
    if gender=="Male":
        gender_value = 1
    customer = pd.DataFrame(
    [
    [
    gender_value,
    age,
    income,
    score
    ]
    ],

    columns=[
    "Gender",
    "Age",
    "Annual_Income",
    "Spending_Score"
    ]

    )

    customer = scaler.transform(customer)
    result = model.predict(customer)

    if result[0]==0:
        segment="Low Value Customer"

    elif result[0]==1:
        segment="Medium Value Customer"

    else:
        segment="High Value Customer"


    st.success(segment)