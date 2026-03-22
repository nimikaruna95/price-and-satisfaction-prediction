# train_satisfaction_mlflow.py
import pandas as pd
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from mlflow.models.signature import infer_signature

# Folder Creation
MODEL_DIR = "models/satisfaction"
ARTIFACT_DIR = "artifacts/satisfaction"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Setup for the MLFlow 
mlflow.set_tracking_uri("file:./mlruns_satisfaction")
mlflow.set_experiment("Customer_Satisfaction")

# Loading the data
df = pd.read_csv("data/passenger_cleaned.csv")

# Drop unwanted columns
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Feature Extraction
X = df.drop("satisfaction", axis=1)

# For the Encode target for XGBoost
y = df["satisfaction"].map({
    "neutral or dissatisfied": 0,
    "satisfied": 1
})

print("Target values:", y.unique())

# Column types
categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

# For MLflow schema safety , Convert integers to float type
X[numerical] = X[numerical].astype("float64")

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
}

# Split Functions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_acc = 0
best_model = None
best_run_id = None

# Training of loop
for name, model in models.items():

    print(f"\n Training: {name}")

    with mlflow.start_run(run_name=name) as run:

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Cross-validation
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
            error_score='raise'   # For Debug Friendly
        )

        print(f"CV Accuracy: {scores.mean():.4f}")

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        preds = pipeline.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)

        print(f"Test Accuracy: {acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"{name} Confusion Matrix")

        cm_path = f"{ARTIFACT_DIR}/{name}_cm.png"
        plt.savefig(cm_path)
        plt.close()

        # MLFlow Logging
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.log_artifact(cm_path)

        # Model signature
        signature = infer_signature(X_train, pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature
        )

        # Saving in local path
        model_path = f"{MODEL_DIR}/{name}_model.pkl"
        joblib.dump(pipeline, model_path)

        # Best model tracking
        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_run_id = run.info.run_id

# Saving Best Model
best_model_path = f"{MODEL_DIR}/satisfaction_best_model.pkl"
joblib.dump(best_model, best_model_path)

print("\n Best model saved:", best_model_path)

# REgister for the  model
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "Satisfaction_Best_Model")
print("\n satisfaction Training Completed")
