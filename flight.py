# flight.py
import streamlit as st
import pandas as pd
import joblib
import os

# Load Models (Cached)
@st.cache_resource
def load_flight_model():
    return joblib.load("models/flight/flight_best_model.pkl")

@st.cache_data
def load_flight_data():
    return pd.read_csv("data/flight_cleaned.csv")

st.header("Flight Price Prediction")

model = load_flight_model()
df = load_flight_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Price by Airline")
    st.bar_chart(df.groupby("Airline")["Price"].mean())

with col2:
    st.subheader("Stops vs Price")
    st.bar_chart(df.groupby("Total_Stops_Count")["Price"].mean())

st.subheader("Enter Flight Details")

col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox("Airline", df["Airline"].unique())
    source = st.selectbox("Source", df["Source"].unique())
    route = st.selectbox("Route", df["Route"].unique())

with col2:
    destination = st.selectbox("Destination", df["Destination"].unique())
    stops = st.slider("Total Stops", 0, 4, 1)

with col3:
    journey_day = st.slider("Journey Day", 1, 31, 10)
    journey_month = st.slider("Journey Month", 1, 12, 3)

dep_hour = st.slider("Departure Hour", 0, 23, 10)
arrival_hour = st.slider("Arrival Hour", 0, 23, 12)
duration = st.slider("Duration (minutes)", 30, 1500, 300)

# Input Data 
input_df = pd.DataFrame({
    "Airline": [airline],
    "Source": [source],
    "Destination": [destination],
    "Route": [route],
    "Additional_Info": ["No info"],
    "Journey_Day": [journey_day],
    "Journey_Month": [journey_month],
    "Dep_Hour": [dep_hour],
    "Arrival_Hour": [arrival_hour],
    "Duration_Hours": [duration // 60],
    "Duration_Minutes": [duration % 60],
    "Total_Duration_Minutes": [duration],
    "Total_Stops_Count": [stops]
    })

if st.button("Predict Flight Price"):
    try:
        # Column alignment
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: ₹ {round(prediction, 2)}")

    except Exception as e:
        st.error("Prediction failed. Please check input values.")
