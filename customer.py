# customer.py
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

@st.cache_resource
def load_customer_model():
    return joblib.load("models/satisfaction/satisfaction_best_model.pkl")

@st.cache_data
def load_customer_data():
    return pd.read_csv("data/passenger_cleaned.csv")

st.header("Customer Satisfaction Prediction")

model = load_customer_model()
df = load_customer_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Satisfaction Distribution")
    st.bar_chart(df["satisfaction"].value_counts())

with col2:
    st.subheader("Class vs Satisfaction")
    st.bar_chart(pd.crosstab(df["Class"], df["satisfaction"]))

st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", df["Gender"].unique())
    customer_type = st.selectbox("Customer Type", df["Customer Type"].unique())

with col2:
    travel_type = st.selectbox("Type of Travel", df["Type of Travel"].unique())
    travel_class = st.selectbox("Class", df["Class"].unique())

with col3:
    age = st.slider("Age", 5, 80, 25)
    flight_distance = st.slider("Flight Distance", 100, 5000, 500)

wifi = st.slider("Inflight Wifi", 0, 5, 3)
food = st.slider("Food & Drink", 0, 5, 3)
comfort = st.slider("Seat Comfort", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)

delay_dep = st.slider("Departure Delay", 0, 300, 10)
delay_arr = st.slider("Arrival Delay", 0, 300, 5)

total_service = wifi + food + comfort + cleanliness
total_delay = delay_dep + delay_arr

age_group = (
    "Teen" if age < 18 else
    "Young Adult" if age < 30 else
    "Adult" if age < 45 else
    "Senior" if age < 60 else
    "Elder"
    )

input_df = pd.DataFrame({
    "Gender": [gender],
    "Customer Type": [customer_type],
    "Age": [age],
    "Type of Travel": [travel_type],
    "Class": [travel_class],
    "Flight Distance": [flight_distance],
    "Inflight wifi service": [wifi],
    "Departure/Arrival time convenient": [3],
    "Ease of Online booking": [3],
    "Gate location": [3],
    "Food and drink": [food],
    "Online boarding": [3],
    "Seat comfort": [comfort],
    "Inflight entertainment": [3],
    "On-board service": [3],
    "Leg room service": [3],
    "Baggage handling": [3],
    "Checkin service": [3],
    "Inflight service": [3],
    "Cleanliness": [cleanliness],
    "Departure Delay in Minutes": [delay_dep],
    "Arrival Delay in Minutes": [delay_arr],
    "Total_Service_Score": [total_service],
    "Total_Delay": [total_delay],
    "Age_Group": [age_group]
    })

if st.button("Predict Satisfaction"):
    try:
        # FIX 2: Column alignment
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("Customer is Satisfied")
        else:
            st.error("Customer is Not Satisfied")

    except Exception as e:
        st.error("Prediction failed. Please check input values.")
