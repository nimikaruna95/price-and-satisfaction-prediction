import pickle
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import pandas as pd
import os

def train_flight_price_model():
    try:
        # Path to the cleaned data CSV
        file_path = r"C:\\Users\\DELL\\Desktop\\flight_customer_app\\flight_price\\data\\cleaned_flight_data.csv"

        # Verify if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        # Load the cleaned data into a pandas DataFrame
        data = pd.read_csv(file_path)
        
        # Convert categorical columns to numerical using One-Hot Encoding
        data = pd.get_dummies(data, columns=['Airline', 'Source', 'Destination'], drop_first=True)
        
        # Define feature and target variables
        X = data.drop(['Price'], axis=1)
        y = data['Price']
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # models
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
        }

        # Train and log models with MLflow
        trained_models = {}
        best_model_name = None
        best_model = None
        best_mse = float('inf')

        for model_name, model in models.items():
            
            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mse ** 0.5

            # Log metrics and model with MLflow
            with mlflow.start_run():
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model")

                print(f"Model: {model_name} | MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

            # Save the model in the dictionary
            trained_models[model_name] = model

            # Save the best model
            if mse < best_mse:
                best_mse = mse
                best_model_name = model_name
                best_model = model

        # Save the best model
        if best_model:
            best_model_file = r"C:\\Users\\DELL\\Desktop\\flight_customer_app\\best_flight_price_model.pkl"
            joblib.dump(best_model, best_model_file)
            print(f"Best model saved: {best_model_name} with MSE: {best_mse:.4f}")

        # Save all trained models into a single file
        all_models_file = r"C:\\Users\\DELL\\Desktop\\flight_customer_app\\flight_price_models.pkl"
        with open(all_models_file, "wb") as f:
            pickle.dump(trained_models, f)
        print(f"All trained models saved to {all_models_file}")

        return trained_models

    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
train_flight_price_model()

