## Flight Price & Customer Satisfaction Prediction App

## Project Overview

This project is an **end-to-end Machine Learning application** that includes:

1. **Flight Price Prediction (Regression)**
2. **Customer Satisfaction Prediction (Classification)**

The application is built using **Python, Scikit-learn, MLflow, and Streamlit**, and demonstrates the complete ML lifecycle:

* Data preprocessing
* Feature engineering
* Model training
* Experiment tracking
* Deployment

---

## Features

### Flight Price Prediction

* Predicts flight ticket prices based on:

  * Airline, Source, Destination
  * Journey date
  * Departure & arrival time
  * Duration and number of stops
* Includes EDA visualizations:

  * Price distribution
  * Airline vs price
  * Stops vs price

---

### Customer Satisfaction Prediction

* Predicts whether a customer is:

  * **Satisfied**
  * **Dissatisfied**
* Based on:

  * Demographics
  * Travel details
  * Service ratings
  * Delay times
* Includes insights:

  * Satisfaction distribution
  * Class vs satisfaction
  * Delay vs satisfaction

---

## Machine Learning Workflow

### 🔹 Data Preprocessing

* Removed irrelevant columns (`id`, index columns)
* Handled missing values
* Converted date/time into numerical features
* Cleaned and standardized dataset

---

### Feature Engineering

#### Flight Dataset:

* Journey Day, Month, Year
* Departure & Arrival Hour
* Duration in minutes
* Route start/end
* Total stops

#### Customer Dataset:

* Total Service Score
* Age Group categorization

---

### Models Used

#### Regression (Flight Price)

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

#### Classification (Customer Satisfaction)

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* KNN
* XGBoost

---

### Model Selection

* Used **Cross Validation**
* Compared models using:

  * Accuracy
  * F1-score

* Selected **best performing model**

---

## MLflow Integration

* Logged:

  * Parameters
  * Metrics (Accuracy, F1-score, etc.)
  * Confusion Matrix (as artifacts)
* Tracked experiments
* Registered best model in **Model Registry**

---

## 💻 Streamlit Application

* Single app with **two projects**
* Sidebar navigation for switching
* Interactive UI with:

  * User inputs
  * Real-time predictions
  * Visual insights

---

## Project Structure

```
flight_customer_app/
│
├── app.py
├── data/
│   ├── flight_cleaned.csv
│   └── passenger_cleaned.csv
│
├── models/
│   ├── flight/
│   └── satisfaction/
│
├── preprocessing/
├── training/
├── eda/
└── mlruns/
```

---

## Installation

```bash
git clone <your-repo-link>
cd project-folder
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Results

* High accuracy achieved for both tasks
* Robust preprocessing pipeline
* Clean and interactive UI
* Efficient model tracking using MLflow

---

## Business Use Cases

* Helps travelers estimate flight costs
* Assists airlines in pricing strategies
* Improves customer experience analysis
* Supports decision-making for customer retention

---

## Key Highlights

* End-to-end ML project
* Real-world dataset handling
* MLflow experiment tracking
* Interactive Streamlit deployment
* Clean, modular, and scalable code

---

## Future Improvements

* Deploy on cloud (Streamlit Cloud / Render)
* Add real-time API integration
* Hyperparameter tuning
* Advanced visual dashboards

---

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
