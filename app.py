# App.py
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Flight & Customer App",
    layout="wide"
)

st.title("Flight Price & Customer Satisfaction App")

# Project selection
project = st.sidebar.radio(
    "Select Project",
    ["Flight Price Prediction", "Customer Satisfaction"]
)

BASE_DIR = os.getcwd()

# FLIGHT PRICE PREDICTION
if project == "Flight Price Prediction":

    st.header("ight Price Prediction")

    # Load model
    model_path = os.path.join(BASE_DIR, "models", "flight", "flight_best_model.pkl")
    model = joblib.load(model_path)

    # Load data
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "flight_cleaned.csv"))

    # EDA
    st.subheader("Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df["Price"], bins=30)
        ax.set_title("Price Distribution")
        st.pyplot(fig)

    with col2:
        airline_price = df.groupby("Airline")["Price"].mean()
        st.bar_chart(airline_price)

    st.subheader("Stops vs Price")
    st.bar_chart(df.groupby("Total_Stops_Count")["Price"].mean())

    # Sidebar Inputs
    st.sidebar.title("Flight Input")

    airline = st.sidebar.selectbox("Airline", df["Airline"].unique())
    source = st.sidebar.selectbox("Source", df["Source"].unique())
    destination = st.sidebar.selectbox("Destination", df["Destination"].unique())

    stops = st.sidebar.slider("Total Stops", 0, 4, 1)

    journey_day = st.sidebar.slider("Journey Day", 1, 31, 10)
    journey_month = st.sidebar.slider("Journey Month", 1, 12, 3)

    dep_hour = st.sidebar.slider("Departure Hour", 0, 23, 10)
    arrival_hour = st.sidebar.slider("Arrival Hour", 0, 23, 12)

    duration_mins = st.sidebar.slider("Duration (minutes)", 30, 1500, 300)

    # Input Data
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
        st.success(f"💰 Estimated Flight Price: ₹ {round(prediction, 2)}")


# CUSTOMER SATISFACTION
elif project == "Customer Satisfaction":

    st.header("Customer Satisfaction Prediction")

    # Load model
    model_path = os.path.join(BASE_DIR, "models", "satisfaction", "satisfaction_best_model.pkl")
    data_path = os.path.join(BASE_DIR, "data", "passenger_cleaned.csv")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # EDA
    st.subheader("Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Satisfaction Distribution")
        st.bar_chart(df["satisfaction"].value_counts())

    with col2:
        st.subheader("Class vs Satisfaction")
        class_sat = pd.crosstab(df["Class"], df["satisfaction"])
        st.bar_chart(class_sat)

    st.subheader("Delay vs Satisfaction")
    delay_df = df.groupby("satisfaction")[[
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes"
    ]].mean()

    st.bar_chart(delay_df)

    # Sidebar Inputs
    st.sidebar.title("Customer Input")

    gender = st.sidebar.selectbox("Gender", df["Gender"].unique())
    customer_type = st.sidebar.selectbox("Customer Type", df["Customer Type"].unique())
    travel_type = st.sidebar.selectbox("Type of Travel", df["Type of Travel"].unique())
    travel_class = st.sidebar.selectbox("Class", df["Class"].unique())

    age = st.sidebar.slider("Age", 5, 80, 25)
    flight_distance = st.sidebar.slider("Flight Distance", 100, 5000, 500)

    wifi = st.sidebar.slider("Inflight Wifi", 0, 5, 3)
    food = st.sidebar.slider("Food & Drink", 0, 5, 3)
    comfort = st.sidebar.slider("Seat Comfort", 0, 5, 3)
    cleanliness = st.sidebar.slider("Cleanliness", 0, 5, 3)

    delay_dep = st.sidebar.slider("Departure Delay", 0, 300, 10)
    delay_arr = st.sidebar.slider("Arrival Delay", 0, 300, 5)

    total_service = wifi + food + comfort + cleanliness

    # Age group
    if age < 18:
        age_group = "Teen"
    elif age < 30:
        age_group = "Young Adult"
    elif age < 45:
        age_group = "Adult"
    elif age < 60:
        age_group = "Senior"
    else:
        age_group = "Elder"

    # Input Data
    input_data = pd.DataFrame({
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
        "Age_Group": [age_group]
    })

    # Prediction
    if st.button("Predict Satisfaction"):

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("Customer is Satisfied")
        else:
            st.error("Customer is Dissatisfied")

        # Feature importance
        try:
            importances = model.named_steps["model"].feature_importances_

            fig, ax = plt.subplots()
            ax.barh(range(len(importances)), importances)
            ax.set_title("Feature Importance")

            st.pyplot(fig)
        except:
            st.warning("Feature importance not available")

