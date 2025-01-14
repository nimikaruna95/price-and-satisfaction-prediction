import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# App title
st.title("Flight Price Prediction App")

# cleaned data path
CLEANED_DATA_PATH = r"C:\Users\DELL\Desktop\flight_customer_app\flight_price\data\cleaned_flight_data.csv"

# data loading for performance
def load_cleaned_data():
    data = pd.read_csv(CLEANED_DATA_PATH)
    return data

# Main function
def main():
    # Load and display the data
    data = load_cleaned_data()
    st.write("### Dataset Preview", data.head())

    # Visualization section
    st.subheader("Visualizations")

    # Flight price trends by source and destination
    st.write("Average Price by Source and Destination")
    fig, ax = plt.subplots(figsize=(10, 6))
    data.groupby(['Source', 'Destination'])['Price'].mean().unstack().plot(kind='bar', ax=ax)
    ax.set_title("Average Flight Price by Source and Destination")
    ax.set_ylabel("Average Price")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Filter options
    st.subheader("Filter Data")
    source_filter = st.multiselect("Select Source", options=data['Source'].unique())
    destination_filter = st.multiselect("Select Destination", options=data['Destination'].unique())
    total_stops_filter = st.multiselect("Select Total Stops", options=data['Total_Stops'].unique())
    time_range = st.slider("Select Departure Time Range (Hour)", 0, 23, (0, 23))

    # Apply filters
    filtered_data = data.copy()
    if source_filter:
        filtered_data = filtered_data[filtered_data['Source'].isin(source_filter)]
    if destination_filter:
        filtered_data = filtered_data[filtered_data['Destination'].isin(destination_filter)]
    if total_stops_filter:
        filtered_data = filtered_data[filtered_data['Total_Stops'].isin(total_stops_filter)]
    filtered_data = filtered_data[
        (filtered_data['Dep_hour'] >= time_range[0]) & (filtered_data['Dep_hour'] <= time_range[1])
    ]
    st.write("### Filtered Data", filtered_data)

    # simple Linear Regression model
    def train_model(data):
        X = pd.get_dummies(data.drop(columns=['Price']), drop_first=True)
        y = data['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        # Save the trained model
        joblib.dump(model, r"C:\Users\DELL\Desktop\flight_customer_app\best_flight_price_model.pkl")
        return model, mae, X

    # training section
    st.subheader("Train Model")
    if st.button("Train Flight Price Model"):
        model, mae, X_features = train_model(data)
        st.success(f"Model trained successfully! MAE: ₹{mae:.2f}")

    # Prediction section
    st.subheader("Predict Flight Price")
    user_airline = st.selectbox("Select Airline", options=data['Airline'].unique())
    user_source = st.selectbox("Select Source", options=data['Source'].unique())
    user_destination = st.selectbox("Select Destination", options=data['Destination'].unique())
    user_dep_hour = st.slider("Select Departure Hour", 0, 23, 12)
    user_dep_minute = st.slider("Select Departure Minute", 0, 59, 0)
    user_arr_hour = st.slider("Select Arrival Hour", 0, 23, 15)
    user_arr_minute = st.slider("Select Arrival Minute", 0, 59, 30)
    user_duration_hours = st.number_input("Enter Duration (Hours)", min_value=0, value=2)
    user_duration_minutes = st.number_input("Enter Duration (Minutes)", min_value=0, value=30)
    user_total_stops = st.number_input("Enter Total Stops", min_value=0, value=0)

    # Prediction button
    if st.button("Predict Price"):
        user_input = pd.DataFrame({
            'Airline': [user_airline],
            'Source': [user_source],
            'Destination': [user_destination],
            'Total_Stops': [user_total_stops],
            'Journey_day': [1],  # Placeholder
            'Journey_month': [1],  # Placeholder
            'Journey_year': [2025],  # Placeholder
            'Dep_hour': [user_dep_hour],
            'Dep_minute': [user_dep_minute],
            'Arrival_hour': [user_arr_hour],
            'Arrival_minute': [user_arr_minute],
            'Duration_hours': [user_duration_hours],
            'Duration_minutes': [user_duration_minutes],
        })

        # One-hot encoding for prediction
        X_features = pd.get_dummies(data.drop(columns=['Price']), drop_first=True)
        user_input_encoded = pd.get_dummies(user_input, drop_first=True)
        user_input_encoded = user_input_encoded.reindex(columns=X_features.columns, fill_value=0)

        # Load the trained model and make prediction
        model = joblib.load(r"C:\Users\DELL\Desktop\flight_customer_app\best_flight_price_model.pkl")
        prediction = model.predict(user_input_encoded)
        st.write(f"### Predicted Flight Price: ₹{prediction[0]:,.2f}")

# Run the app
if __name__ == "__main__":
    main()


