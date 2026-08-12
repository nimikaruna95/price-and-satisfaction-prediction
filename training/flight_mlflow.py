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

# FOLDERS
MODEL_DIR = "models/flight"
ARTIFACT_DIR = "artifacts/flight"
metrics_results = []

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# MLFLOW SETUP
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Flight_Price_Prediction")

# LOAD DATA
df = pd.read_csv("data/flight_cleaned.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

# Fix MLflow schema warning
num_cols = X.select_dtypes(exclude=["object"]).columns
X[num_cols] = X[num_cols].astype("float64")

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# PREPROCESSOR
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# MODELS
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

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_rmse = float("inf")
best_model = None
best_model_name = None
best_model_uri = None

# TRAINING LOOP
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # PARAM LOGGING (FIXED)
        mlflow.log_param("model_name", name)
        mlflow.log_param("num_features", X.shape[1])

        # Safe random_state logging
        if hasattr(model, "random_state") and model.random_state is not None:
            mlflow.log_param("random_state", model.random_state)

        # Safe param logging
        if hasattr(model, "get_params"):
            params = model.get_params()

            clean_params = {
                k: v for k, v in params.items()
                if v is not None and k != "random_state"
            }

            mlflow.log_params(clean_params)

        # CROSS VALIDATION
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error"
        )
        cv_rmse = -cv_scores.mean()
        mlflow.log_metric("cv_rmse", cv_rmse)

        # TRAIN
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # METRICS
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)

        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2_score", r2)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_mape", mape)

        # STORE METRICS
        metrics_results.append({
            "Model": name,
            "CV_RMSE": cv_rmse,
            "Test_RMSE": rmse,
            "Test_R2": r2,
            "Test_MAE": mae,
            "Test_MAPE": mape
        })

        # PRINT METRICS TO TERMINAL
        print("\n" + "=" * 60)
        print(f"Flight Model: {name}")
        print("=" * 60)
        print(f"CV RMSE       : {cv_rmse:.4f}")
        print(f"Test RMSE     : {rmse:.4f}")
        print(f"Test R2 Score : {r2:.4f}")
        print(f"Test MAE      : {mae:.4f}")
        print(f"Test MAPE     : {mape:.4f}")

        # ACTUAL VS PREDICTED
        plt.figure()
        sns.scatterplot(x=y_test, y=preds)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")

        pred_path = f"{ARTIFACT_DIR}/{name}_pred.png"
        plt.savefig(pred_path)
        plt.close()
        mlflow.log_artifact(pred_path)

        # RESIDUAL PLOT
        residuals = y_test - preds

        plt.figure()
        sns.histplot(residuals, kde=True)
        plt.title("Residual Distribution")

        res_path = f"{ARTIFACT_DIR}/{name}_residual.png"
        plt.savefig(res_path)
        plt.close()
        mlflow.log_artifact(res_path)

        # FEATURE IMPORTANCE
        try:
            model_obj = pipeline.named_steps["model"]

            if hasattr(model_obj, "feature_importances_"):
                importances = model_obj.feature_importances_

                plt.figure()
                plt.bar(range(len(importances)), importances)
                plt.title("Feature Importance")

                fi_path = f"{ARTIFACT_DIR}/{name}_feature_importance.png"
                plt.savefig(fi_path)
                plt.close()

                mlflow.log_artifact(fi_path)

        except Exception:
            pass

        # MODEL LOG
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train.iloc[:5]
        )

        # Save locally
        joblib.dump(pipeline, f"{MODEL_DIR}/{name}_model.pkl")

        # BEST MODEL TRACKING
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_model_name = name
            best_model_uri = logged.model_uri

# SAVE BEST MODEL
joblib.dump(best_model, f"{MODEL_DIR}/flight_best_model.pkl")

print(f"\nBest Model: {best_model_name}")
print(f"Best RMSE: {best_rmse:.2f}")

# REGISTER MODEL
mlflow.register_model(best_model_uri, "Flight_Best_Model")

# SAVE ALL MODEL METRICS
metrics_df = pd.DataFrame(metrics_results)

metrics_path = f"{ARTIFACT_DIR}/flight_model_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\n" + "=" * 70)
print("ALL FLIGHT MODEL METRICS")
print("=" * 70)
print(metrics_df.to_string(index=False))

print(f"\nMetrics saved to: {metrics_path}")
print("Flight training complete")