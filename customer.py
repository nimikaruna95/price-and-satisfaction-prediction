# Customer.py
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models", "satisfaction", "satisfaction_best_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "passenger_cleaned.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

st.title("Customer Satisfaction Prediction")

st.header("Insights")

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

# Sidebar inputs
st.sidebar.header("Enter Customer Details")
 
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

# Input data
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

# Predictions
if st.button("Predict Satisfaction"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Results")

    if prediction == 1:
        st.success("Customer is Satisfied")
    else:
        st.error("Customer is Dissatisfied")

    # Feature Importance
    try:
        importances = model.named_steps["model"].feature_importances_

        fig, ax = plt.subplots()
        ax.barh(range(len(importances)), importances)
        ax.set_title("Feature Importance")

        st.pyplot(fig)
    except:
        st.warning("Feature importance not available")