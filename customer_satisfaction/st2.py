import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

CLEANED_DATA_PATH = r"C:\Users\DELL\Desktop\flight_customer_app\customer_satisfaction\data\cleaned_customer_data.csv"

def load_cleaned_data():
    data = pd.read_csv(CLEANED_DATA_PATH)
    return data

def load_model():
    try:
        model = joblib.load(r"C:\Users\DELL\Desktop\flight_customer_app\best_customer_satisfaction_model.pkl")
        return model
    except Exception as e:
        print(f"An error occurred: {e}")

def predict_customer_satisfaction(model, travel_type, gender, customer_type, food_and_drink, seat_comfort, 
                                   inflight_entertainment, on_board_service, leg_room_service, baggage_handling, 
                                   checkin_service, inflight_service, cleanliness):
    # Ensure all columns that the model expects are present, even if some are missing in the input
    input_data = pd.DataFrame({
        'id': [0],  # Default value for 'id'
        'Age': [30],  # Default value for 'Age'
        'Class': ['Eco'],  # Default value for 'Class'
        'Flight Distance': [500],  # Default value for 'Flight Distance'
        'Online boarding': [3],  # Default value for 'Online boarding'
        'Ease of Online booking': [3],  # Default value for 'Ease of Online booking'
        'Inflight wifi service': [3],  # Default value for 'Inflight wifi service'
        'Gate location': [3],  # Default value for 'Gate location'
        'Departure/Arrival time convenient': [3],  # Default value for 'Departure/Arrival time convenient'
        'Gender': [gender],
        'Type of Travel': [travel_type],
        'Food and drink': [food_and_drink],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [inflight_entertainment],
        'On-board service': [on_board_service],
        'Leg room service': [leg_room_service],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Customer Type': [customer_type],
        'Departure Delay in Minutes': [0],  # Default value for 'Departure Delay in Minutes'
        'Arrival Delay in Minutes': [0]  # Default value for 'Arrival Delay in Minutes'
    })

    # Convert categorical columns to the appropriate category type
    categorical_columns = ['Type of Travel', 'Gender', 'Customer Type', 'Class', 
                           'Inflight wifi service', 'Departure/Arrival time convenient', 
                           'Ease of Online booking', 'Gate location']
    for col in categorical_columns:
        input_data[col] = input_data[col].astype('category')

    # Ensure the order of columns matches the model's training order
    expected_columns = ['id', 'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 
                        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
                        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 
                        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 
                        'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    
    input_data = input_data[expected_columns] 

    prediction = model.predict(input_data)
    satisfaction = 'neutral or dissatisfied' if prediction[0] in [0, 1] else 'satisfied'
    return satisfaction

def load_data():
    data = pd.read_csv(CLEANED_DATA_PATH)
    return data

def main():
    data = load_cleaned_data()

    st.title("Customer Satisfaction Prediction")

    st.write("### Dataset Preview")
    st.write(data.head())

    st.subheader("Customer Satisfaction Trends")
    satisfaction_counts = data['satisfaction'].value_counts()
    st.write("Satisfaction Distribution in Data")
    fig, ax = plt.subplots()
    sns.barplot(x=satisfaction_counts.index, y=satisfaction_counts.values, ax=ax)
    ax.set_xlabel("Satisfaction Level")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    model = load_model()

    st.subheader("Predict Customer Satisfaction")
    travel_type = st.selectbox('Type of Travel', ['Eco', 'Business', 'Eco Plus'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'Disloyal Customer'])
    food_and_drink = st.slider('Food and drink', 1, 5)
    seat_comfort = st.slider('Seat comfort', 1, 5)
    inflight_entertainment = st.slider('Inflight entertainment', 1, 5)
    on_board_service = st.slider('On-board service', 1, 5)
    leg_room_service = st.slider('Leg room service', 1, 5)
    baggage_handling = st.slider('Baggage handling', 1, 5)
    checkin_service = st.slider('Checkin service', 1, 5)
    inflight_service = st.slider('Inflight service', 1, 5)
    cleanliness = st.slider('Cleanliness', 1, 5)

    if st.button('Predict Satisfaction'):
        result = predict_customer_satisfaction(model, travel_type, gender, customer_type, food_and_drink, seat_comfort, 
                                                inflight_entertainment, on_board_service, leg_room_service, baggage_handling, 
                                                checkin_service, inflight_service, cleanliness)

        if result:
            st.write(f'Predicted Customer Satisfaction: {result}')

            st.write("### Visualizing Predicted Satisfaction Level")
            satisfaction_values = [result]
            fig, ax = plt.subplots()
            sns.countplot(x=satisfaction_values, ax=ax)
            ax.set_title("Predicted Customer Satisfaction Level")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
