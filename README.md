# Flight Price and Customer Satisfaction Prediction Project

This project involves two predictive models for the travel and customer experience domains:

1. **Flight Price Prediction (Regression Model)**
2. **Customer Satisfaction Prediction (Classification Model)**

Both models are deployed via a **Streamlit** application for interactive data analysis and predictions, integrated with **MLflow** for model tracking and management.

---
## Project Structure

flight_customer_project/
│
├── app.py                          # Main Streamlit app (both projects)
├── flight.py                       # Streamlt app for flight_price
├── customer.py                     # Streamlt app for passenger_satisfaction  
│
├── data/
│   ├── Flight_price.csv
│   └── Passenger_Satisfaction.csv
│   └── flight_cleaned.csv
│   └── passenger_cleaned.csv
│
├── preprocessing/
│   ├── flight_preprocessing.py
│   └── satisfaction_preprocessing.py
│
├── training/
│   ├── train_flight_mlflow.py
│   └── train_satisfaction_mlflow.py
│
├── eda/
│   ├── flight_eda.py
│   ├── satisfaction_eda.py
│   └── flight/
│   └── customer/
│
├── models/
│   ├── flight/
│   │   ├── LinearRegression_model.pkl
│   │   ├── RandomForest_model.pkl
│   │   ├── XGBoost_model.pkl
│   │   └── flight_best_model.pkl
│   │
│   └── satisfaction/
│       ├── LogisticRegression_model.pkl
│       ├── DecisionTree_model.pkl
│       ├── RandomForest_model.pkl
│       ├── GradientBoosting_model.pkl
│       ├── KNN_model.pkl
│       ├── XGBoost_model.pkl
│       └── satisfaction_best_model.pkl
│
├── artifacts/
│   ├── flight/
│   │   └── *.png   # confusion matrix / plots
│   │
│   └── satisfaction/
│       └── *.png
│
├── mlruns_flight/                  # MLflow logs (flight)
├── mlruns_satisfaction/            # MLflow logs (customer)
│
├── requirements.txt
└── README.me

## Project 1: Flight Price Prediction (Regression)

### Project Title:
**Flight Price Prediction**

### Skills Acquired:
- Python
- Streamlit
- Machine Learning (Regression)
- Data Analysis
- MLflow

### Domain:
Travel and Tourism

### Problem Statement:
Predict flight ticket prices based on various factors such as departure time, source, destination, and airline type. The dataset is processed, cleaned, and used to train regression models, which are then deployed in a Streamlit app.

### Business Use Cases:
- Help travelers plan trips by predicting flight prices.
- Assist travel agencies with price optimization.
- Enable businesses to forecast travel budgets.
- Support airlines with pricing strategies.

### Approach:
1. **Data Preprocessing**:
   - Clean and preprocess the data.
   - Convert date/time columns and perform feature engineering.
2. **Flight Price Prediction**:
   - Perform EDA and train regression models (e.g., Linear Regression, Random Forest, XGBoost).
   - Integrate MLflow to log and track experiments.
3. **Streamlit App Development**:
   - Create an interactive app that predicts flight prices based on user inputs (route, time, date).

### Results:
- Cleaned dataset with high-accuracy flight price predictions.
- Streamlit app for interactive predictions and visualizations.
- MLflow integration for model tracking.

### Project Deliverables:
- Python scripts for preprocessing, training, and MLflow integration.
- Clean CSV file with processed flight data.
- Regression models logged and managed using MLflow.
- A Streamlit app for visualization and predictions.

### Technical Tags:
- Python, Data Cleaning, Feature Engineering, Machine Learning, Regression, Streamlit, MLflow

### Dataset:
- **Flight_Price.csv**  
  - Includes: Airline, Date_of_Journey, Source, Destination, Route, Departure/Arrival Times, Duration, Stops, Additional Info.

---

## Project 2: Customer Satisfaction Prediction (Classification)

### Project Title:
**Customer Satisfaction Prediction**

### Skills Acquired:
- Python
- Machine Learning (Classification)
- Data Analysis
- Streamlit
- MLflow

### Domain:
Customer Experience

### Problem Statement:
Predict customer satisfaction levels based on feedback, demographics, and service ratings. The dataset is processed, cleaned, and used to train classification models, which are deployed in a Streamlit app.

### Business Use Cases:
- Enhance customer experience by predicting dissatisfaction.
- Provide insights for service improvement.
- Support marketing with targeted customer groups.
- Assist management in customer retention strategies.

### Approach:
1. **Data Preprocessing**:
   - Clean and encode categorical data.
   - Normalize or standardize features.
2. **Customer Satisfaction Prediction**:
   - Perform EDA and train classification models (e.g., Logistic Regression, Random Forest, Gradient Boosting).
   - Integrate MLflow to track accuracy and F1-score metrics.
3. **Streamlit App Development**:
   - Build an app that predicts customer satisfaction based on user inputs.

### Results:
- Cleaned dataset with high-accuracy customer satisfaction predictions.
- Streamlit app for predicting satisfaction levels.
- MLflow integration for model tracking.

### Project Deliverables:
- Python scripts for preprocessing, training, and MLflow integration.
- Clean CSV file with processed customer data.
- Classification models logged and managed using MLflow.
- A Streamlit app for visualization and predictions.

### Technical Tags:
- Python, Data Cleaning, Feature Engineering, Machine Learning, Classification, Streamlit, MLflow

### Dataset:
- **Passenger_Satisfaction.csv**  
  - Includes: Gender, Customer Type, Age, Type of Travel, Class, Flight Distance, Satisfaction Levels, and Final Satisfaction Prediction.

---

## Streamlit App Development

The **Streamlit** app will feature two pages:
1. **Flight Price Prediction** - Users can input flight details to predict prices.
2. **Customer Satisfaction Prediction** - Users can input customer feedback data to predict satisfaction levels.

Both pages will integrate **MLflow** for model management and tracking.

---
