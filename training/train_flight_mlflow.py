# flight_mlflow.py
# flight_mlflow.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from xgboost import XGBRegressor
from mlflow.models.signature import infer_signature

# Folders
MODEL_DIR = "models/flight"
ARTIFACT_DIR = "artifacts/flight"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# MLFLOW Setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Flight_Price_Prediction")

# Load Data
df = pd.read_csv("data/flight_cleaned.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

# Fix MLflow schema warning
num_cols = X.select_dtypes(exclude=["object"]).columns
X[num_cols] = X[num_cols].astype("float64")

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(max_iter=5000),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric="rmse"
    )
}

# Split function
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_rmse = float("inf")
best_model = None
best_model_name = None
best_model_uri = None

# Training Loop
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        mlflow.log_param("model_name", name)
        mlflow.log_param("num_features", X.shape[1])

        # Cross Validation
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error"
        )
        cv_rmse = -cv_scores.mean()
        mlflow.log_metric("cv_rmse", cv_rmse)

        # Train
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Metrices
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        # Actual VS Predicted
        plt.figure()
        sns.scatterplot(x=y_test, y=preds)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")

        pred_path = f"{ARTIFACT_DIR}/{name}_pred.png"
        plt.savefig(pred_path)
        plt.close()

        mlflow.log_artifact(pred_path)

        # Residual Plot 
        residuals = y_test - preds

        plt.figure()
        sns.histplot(residuals, kde=True)
        plt.title("Residual Distribution")

        res_path = f"{ARTIFACT_DIR}/{name}_residual.png"
        plt.savefig(res_path)
        plt.close()

        mlflow.log_artifact(res_path)

        # API (no warning)
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train.iloc[:5]
        )

        # Save locally
        joblib.dump(pipeline, f"{MODEL_DIR}/{name}_model.pkl")

        # Best model tracking
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_model_name = name
            best_model_uri = logged.model_uri

# Saving best model
joblib.dump(best_model, f"{MODEL_DIR}/flight_best_model.pkl")

print(f"\nBest Model: {best_model_name}")
print(f"Best RMSE: {best_rmse:.2f}")

# Register model
mlflow.register_model(best_model_uri, "Flight_Best_Model")

print("Flight training complete")
