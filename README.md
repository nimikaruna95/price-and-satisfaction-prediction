This project aims to develop two predictive models in the travel and customer experience domains:

Flight Price Prediction (Regression Model)
Customer Satisfaction Prediction (Classification Model)
Both models will be deployed via a Streamlit application for interactive data analysis and predictions, integrated with MLflow for model tracking and management.

Project 1: Flight Price Prediction (Regression)
Project Title:
Flight Price Prediction

Skills Acquired:
Python
Streamlit
Machine Learning (Regression)
Data Analysis
MLflow
Domain:
Travel and Tourism

Problem Statement:
The goal is to predict flight ticket prices based on factors like departure time, source, destination, and airline type. By processing and cleaning a dataset, we train regression models to predict flight prices and deploy them in a Streamlit app.

Business Use Cases:
Assisting travelers in planning trips with predicted flight prices.
Supporting travel agencies with price optimization and marketing strategies.
Enabling businesses to forecast travel budgets.
Helping airlines optimize pricing strategies based on trends.
Approach:
Data Preprocessing:
Clean data, remove missing entries, and convert date/time columns.
Perform feature engineering (e.g., calculating price per minute).
Flight Price Prediction:
Conduct exploratory data analysis (EDA) to identify trends.
Use models like Linear Regression, Random Forest, and XGBoost.
Integrate MLflow to track experiments and models.
Streamlit App Development:
Create an interactive interface for flight price predictions based on user inputs (route, time, date).
Results:
Cleaned dataset and high-accuracy flight price predictions.
A functional Streamlit app displaying predictions and visualizations.
MLflow integration for model tracking and management.
Project Deliverables:
Python scripts for data preprocessing, training, and MLflow integration.
Clean CSV file containing processed data.
Regression models logged and managed using MLflow.
A Streamlit app for visualization and prediction.
Technical Tags:
Python, Data Cleaning, Feature Engineering, Machine Learning, Regression, Streamlit, MLflow
Dataset:
Flight_Price.csv
Includes: Airline, Date_of_Journey, Source, Destination, Route, Departure/Arrival times, Duration, Stops, and Additional Info.
Project 2: Customer Satisfaction Prediction (Classification)
Project Title:
Customer Satisfaction Prediction

Skills Acquired:
Python
Machine Learning (Classification)
Data Analysis
Streamlit
MLflow
Domain:
Customer Experience

Problem Statement:
The goal is to predict customer satisfaction levels based on feedback, demographics, and service ratings. By processing and cleaning the dataset, we train classification models to predict customer satisfaction levels and deploy the model in a Streamlit app.

Business Use Cases:
Enhancing customer experience by identifying and addressing dissatisfaction.
Providing insights for businesses to improve services.
Supporting marketing teams in identifying customer segments.
Assisting management with customer retention strategies.
Approach:
Data Preprocessing:
Clean and encode the dataset.
Normalize or standardize features as required.
Customer Satisfaction Prediction:
Perform EDA to understand feature relationships.
Use models like Logistic Regression, Random Forest, and Gradient Boosting.
Integrate MLflow to track metrics like accuracy and F1-score.
Streamlit App Development:
Create an interactive interface for users to input features and predict satisfaction levels.
Results:
Cleaned dataset and high-accuracy customer satisfaction predictions.
A functional Streamlit app that predicts customer satisfaction based on inputs.
MLflow integration for model tracking and management.
Project Deliverables:
Python scripts for data preprocessing, training, and MLflow integration.
Clean CSV file containing processed customer data.
Classification models logged and managed using MLflow.
A Streamlit app for visualization and prediction.
Technical Tags:
Python, Data Cleaning, Feature Engineering, Machine Learning, Classification, Streamlit, MLflow
Dataset:
Passenger_Satisfaction.csv
Includes: Gender, Customer Type, Age, Type of Travel, Class, Flight distance, Satisfaction levels (in various categories), and Satisfaction (final prediction).
Streamlit App Development:
A unified Streamlit app will be created with two separate pages:
Page 1: Flight Price Prediction
Page 2: Customer Satisfaction Prediction
Both pages will allow users to input specific details and view predictions along with relevant visualizations. The app will be integrated with MLflow for tracking model experiments and managing metadata.
