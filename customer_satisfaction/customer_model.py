import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib
import pickle

def train_and_save_customer_model():
    try:
        # Path to the cleaned data CSV
        file_path = r"C:\Users\DELL\Desktop\flight_customer_app\customer_satisfaction\data\cleaned_customer_data.csv"

        # Verify if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        # Load the cleaned data into a pandas DataFrame
        data = pd.read_csv(file_path)

        # categorical columns to be label encoded
        categorical_columns = [
            'Gender', 'Customer Type', 'Type of Travel', 'Class',
            'Food and drink', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness'
        ]

        # Label Encoding for categorical columns
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            if col in data.columns:
                data[col] = label_encoder.fit_transform(data[col])

        # Encode the target variable 'satisfaction'
        data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])

        # Define features (X) and target variable (y)
        X = data.drop(['satisfaction'], axis=1)
        y = data['satisfaction']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        # Train, evaluate, and log models
        trained_models = {}
        best_model = None
        best_model_name = None
        best_auc_roc = 0.0  # Initialize the best AUC-ROC score

        for model_name, model in models.items():
            
            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            # Log metrics and model with MLflow
            with mlflow.start_run():
                mlflow.log_param("model_type", model_name)
                mlflow.log_metric("train_score", train_score)
                mlflow.log_metric("test_score", test_score)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("auc_roc", auc_roc)
                mlflow.sklearn.log_model(model, f"{model_name}_customer_satisfaction_model")

                print(f"{model_name} - Train Score: {train_score:.4f}, Test Score: {test_score:.4f}, "
                      f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

                # Register the model in MLflow Model Registry
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}_customer_satisfaction_model"
                mlflow.register_model(model_uri, model_name)

            # Save the trained model
            trained_models[model_name] = model

            # Check if this model is the best based on AUC-ROC
            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_model_name = model_name
                best_model = model

        # Save the best model to a file
        if best_model:
            best_model_file = r"C:\Users\DELL\Desktop\flight_customer_app\best_customer_satisfaction_model.pkl"
            joblib.dump(best_model, best_model_file)
            print(f"Best model saved: {best_model_name} with AUC-ROC: {best_auc_roc:.4f}")

        # Save all trained models into a single file
        all_models_file = r"C:\Users\DELL\Desktop\flight_customer_app\customer_satisfaction_model.pkl"
        with open(all_models_file, "wb") as f:
            pickle.dump(trained_models, f)
        print(f"All trained models saved to {all_models_file}")

        return trained_models

    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to train, save, and log the models
trained_models = train_and_save_customer_model()
