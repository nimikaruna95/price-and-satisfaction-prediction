# Flight.py
import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models", "flight", "flight_best_model.pkl")

model = joblib.load(model_path) 

st.title("Flight Price Prediction")

# Load data (for dropdowns)
df = pd.read_csv("data/flight_cleaned.csv")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    st.bar_chart(df["Price"])

with col2:
    st.subheader("Airline vs Avg Price")
    airline_price = df.groupby("Airline")["Price"].mean()
    st.bar_chart(airline_price)

st.subheader("Stops vs Price")
st.bar_chart(df.groupby("Total_Stops_Count")["Price"].mean())

# Sidebar inputs
st.sidebar.header("Enter Flight Details")

airline = st.sidebar.selectbox("Airline", df["Airline"].unique())
source = st.sidebar.selectbox("Source", df["Source"].unique())
destination = st.sidebar.selectbox("Destination", df["Destination"].unique())
stops = st.sidebar.slider("Total Stops", 0, 4, 1)

journey_day = st.sidebar.slider("Journey Day", 1, 31, 10)
journey_month = st.sidebar.slider("Journey Month", 1, 12, 3)

dep_hour = st.sidebar.slider("Departure Hour", 0, 23, 10)
arrival_hour = st.sidebar.slider("Arrival Hour", 0, 23, 12)

duration_mins = st.sidebar.slider("Total Duration (minutes)", 30, 1500, 300)

# Create input dataframe
input_data = pd.DataFrame({
    "Airline": [airline],
    "Source": [source],
    "Destination": [destination],
    "Route_Start": [source[:3]],
    "Route_End": [destination[:3]],
    "Route_Stops_Count": [stops],
    "Journey_Day": [journey_day],
    "Journey_Month": [journey_month],
    "Journey_Year": [2019],
    "Dep_Hour": [dep_hour],
    "Arrival_Hour": [arrival_hour],
    "Duration_Hours": [duration_mins // 60],
    "Duration_Minutes": [duration_mins % 60],
    "Total_Duration_Minutes": [duration_mins],
    "Total_Stops_Count": [stops],
    "Additional_Info": ["No info"]
})

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Flight Price: ₹ {round(prediction, 2)}")