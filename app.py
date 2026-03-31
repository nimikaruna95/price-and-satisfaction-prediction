# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Flight & Customer ML App",
    layout="wide"
)

st.title("Flight Price & Customer Satisfaction Prediction")

# Sidebar navigation
page = st.sidebar.radio("Select Module", ["Flight Price Prediction", "Customer Satisfaction"])

# Load Models (Cached)
@st.cache_resource
def load_flight_model():
    return joblib.load("models/flight/flight_best_model.pkl")

@st.cache_resource
def load_customer_model():
    return joblib.load("models/satisfaction/satisfaction_best_model.pkl")

@st.cache_data
def load_flight_data():
    return pd.read_csv("data/flight_cleaned.csv")

@st.cache_data
def load_customer_data():
    return pd.read_csv("data/passenger_cleaned.csv")

# Flight Module
if page == "Flight Price Prediction":

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

# Customer Module
else:

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
