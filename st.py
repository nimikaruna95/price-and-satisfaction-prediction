# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Flight & Customer ML App",
    layout="wide")

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

    st.subheader("Flight Data Dashboard")
    tab1, tab2, tab3 = st.tabs(["Visualizations", "Prediction", "Model Performance"])
    col1, col2 = st.columns(2)

    with tab1:

        fig = px.histogram(
            df,
            x="Price",
            nbins=40,
            title="Price Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            df.groupby("Airline", as_index=False)["Price"].mean(),
            x="Airline",
            y="Price",
            color="Price",
            title="Average Price by Airline"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            df,
            x="Source",
            y="Price",
            title="Source vs Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            df,
            x="Destination",
            y="Price",
            title="Destination vs Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            df.groupby("Total_Stops_Count", as_index=False)["Price"].mean(),
            x="Total_Stops_Count",
            y="Price",
            title="Stops vs Average Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            df.groupby("Journey_Month", as_index=False)["Price"].mean(),
            x="Journey_Month",
            y="Price",
            markers=True,
            title="Journey Month vs Average Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            df.groupby("Dep_Hour", as_index=False)["Price"].mean(),
            x="Dep_Hour",
            y="Price",
            markers=True,
            title="Departure Hour vs Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            df,
            x="Total_Duration_Minutes",
            y="Price",
            color="Airline",
            title="Duration vs Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.imshow(
            df.select_dtypes(include="number").corr(),
            text_auto=True,
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.pie(
            df,
            names="Airline",
            title="Flights by Airline"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:

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
            "Journey_Year": [2019],
            "Is_Weekend": [0],
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
                st.error(f"Prediction failed: {e}")

    with tab3:
        st.subheader("Flight Price Model Performance")
        metrics_path = "artifacts/flight/flight_model_metrics.csv"
        try:
            metrics_df = pd.read_csv(metrics_path)
            st.dataframe(metrics_df,use_container_width=True,hide_index=True)
            st.markdown("### Model Comparison")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="Test_RMSE",
                    title="Test RMSE Comparison",
                    text_auto=".2f")
                st.plotly_chart(fig,use_container_width=True)

            with col2:
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="Test_R2",
                    title="Test R² Score Comparison",
                    text_auto=".3f")
                st.plotly_chart(fig,use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="Test_MAE",
                    title="Test MAE Comparison",
                    text_auto=".2f")
                st.plotly_chart(fig,use_container_width=True)

            with col2:
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="Test_MAPE",
                    title="Test MAPE Comparison",
                    text_auto=".3f"
                )
                st.plotly_chart(fig,use_container_width=True)

            # Best model
            best_row = metrics_df.loc[metrics_df["Test_RMSE"].idxmin()]
            st.success(
                f"Best Flight Model: {best_row['Model']} | "
                f"RMSE: {best_row['Test_RMSE']:.2f} | "
                f"R²: {best_row['Test_R2']:.4f}")
        except FileNotFoundError:
            st.warning("Flight model metrics file not found. ""Run flight_mlflow.py first.")

# Customer Module
else:

    st.header("Customer Satisfaction Prediction")

    model = load_customer_model()
    df = load_customer_data()
    
    tab1, tab2, tab3 = st.tabs(["Visualizations", "Prediction", "Model Performance"])

    with tab1:

        # SATISFACTION DISTRIBUTION
        fig = px.histogram(
            df,
            x="satisfaction",
            color="satisfaction",
            title="Satisfaction Distribution",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # GENDER VS SATISFACTION
        fig = px.histogram(
            df,
            x="Gender",
            color="satisfaction",
            barmode="group",
            title="Gender vs Satisfaction",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # CUSTOMER TYPE VS SATISFACTION
        fig = px.histogram(
            df,
            x="Customer Type",
            color="satisfaction",
            barmode="group",
            title="Customer Type vs Satisfaction",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # TRAVEL CLASS VS SATISFACTION
        fig = px.histogram(
            df,
            x="Class",
            color="satisfaction",
            barmode="group",
            title="Travel Class vs Satisfaction",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # TYPE OF TRAVEL VS SATISFACTION
        fig = px.histogram(
            df,
            x="Type of Travel",
            color="satisfaction",
            barmode="group",
            title="Type of Travel vs Satisfaction",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # AGE DISTRIBUTION
        fig = px.histogram(
            df,
            x="Age",
            nbins=30,
            title="Age Distribution",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)

        # FLIGHT DISTANCE VS AGE
        fig = px.scatter(
            df,
            x="Flight Distance",
            y="Age",
            color="satisfaction",
            title="Flight Distance vs Age"
        )
        st.plotly_chart(fig, use_container_width=True)

        # DELAY ANALYSIS
        fig = px.scatter(
            df,
            x="Departure Delay in Minutes",
            y="Arrival Delay in Minutes",
            color="satisfaction",
            title="Delay Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)

        # SERVICE RATINGS
        ratings = [
            "Inflight wifi service",
            "Food and drink",
            "Seat comfort",
            "Cleanliness"
        ]

        # Convert service rating columns into long format
        ratings_long = df[ratings].melt(
            var_name="Service",
            value_name="Rating"
        )

        fig = px.box(
            ratings_long,
            x="Service",
            y="Rating",
            title="Service Ratings"
        )

        st.plotly_chart(fig, use_container_width=True)

        # CORRELATION HEATMAP
        correlation = df.select_dtypes(
            include="number"
        ).corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Customer Data Correlation Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
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

        # SERVICE RATINGS
        wifi = st.slider("Inflight Wifi Service",0, 5, 3)
        departure_arrival = st.slider("Departure/Arrival Time Convenient",0, 5, 3)
        online_booking = st.slider("Ease of Online Booking",0, 5, 3)
        gate_location = st.slider("Gate Location",0, 5, 3)
        food = st.slider("Food and Drink",0, 5, 3)
        online_boarding = st.slider("Online Boarding",0, 5, 3)
        comfort = st.slider("Seat Comfort",0, 5, 3)
        entertainment = st.slider("Inflight Entertainment",0, 5, 3)
        onboard_service = st.slider("On-board Service",0, 5, 3)
        leg_room = st.slider("Leg Room Service",0, 5, 3)
        baggage = st.slider("Baggage Handling",0, 5, 3)
        checkin = st.slider("Checkin Service",0, 5, 3)
        inflight_service = st.slider("Inflight Service",0, 5, 3)
        cleanliness = st.slider("Cleanliness",0, 5, 3)

        # DELAY
        delay_dep = st.slider("Departure Delay",0, 300, 10)
        delay_arr = st.slider("Arrival Delay",0, 300, 5)

        # TOTAL SERVICE SCORE
        total_service = (
            wifi
            + departure_arrival
            + online_booking
            + gate_location
            + food
            + online_boarding
            + comfort
            + entertainment
            + onboard_service
            + leg_room
            + baggage
            + checkin
            + inflight_service
            + cleanliness
        )

        # TOTAL DELAY
        total_delay = delay_dep + delay_arr

        age_group = (
            "Teen" if age < 18 else
            "Young Adult" if age < 30 else
            "Adult" if age < 45 else
            "Senior" if age < 60 else
            "Elder")

        input_df = pd.DataFrame({
            "Gender": [gender],
            "Customer Type": [customer_type],
            "Age": [age],
            "Type of Travel": [travel_type],
            "Class": [travel_class],
            "Flight Distance": [flight_distance],

            "Inflight wifi service": [wifi],
            "Departure/Arrival time convenient": [departure_arrival],
            "Ease of Online booking": [online_booking],
            "Gate location": [gate_location],
            "Food and drink": [food],
            "Online boarding": [online_boarding],
            "Seat comfort": [comfort],
            "Inflight entertainment": [entertainment],
            "On-board service": [onboard_service],
            "Leg room service": [leg_room],
            "Baggage handling": [baggage],
            "Checkin service": [checkin],
            "Inflight service": [inflight_service],
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
                st.error(f"Prediction failed: {e}")
                
    with tab3:
        st.subheader("Customer Satisfaction Model Performance")

        metrics_path = ("artifacts/satisfaction/""satisfaction_model_metrics.csv")

        try:
            metrics_df = pd.read_csv(metrics_path)
            st.dataframe(metrics_df,use_container_width=True,hide_index=True)
            st.markdown("### Model Comparison")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="Accuracy",
                    title="Accuracy Comparison",
                    text_auto=".3f"
                )
                st.plotly_chart(fig,use_container_width=True)

            with col2:
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="F1_Score",
                    title="F1 Score Comparison",
                    text_auto=".3f"
                )
                st.plotly_chart(fig,use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="Precision",
                    title="Precision Comparison",
                    text_auto=".3f"
                )
                st.plotly_chart(fig,use_container_width=True)

            with col2:
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="Recall",
                    title="Recall Comparison",
                    text_auto=".3f"
                )
                st.plotly_chart(fig,use_container_width=True)

            # Best model
            best_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]

            st.success(
                f"Best Customer Satisfaction Model: "
                f"{best_row['Model']} | "
                f"Accuracy: {best_row['Accuracy']:.4f} | "
                f"F1 Score: {best_row['F1_Score']:.4f}")

        except FileNotFoundError:
            st.warning("Customer satisfaction model metrics file not found. ""Run satisfaction_mlflow.py first.")