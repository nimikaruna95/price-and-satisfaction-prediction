# satisfaction_mlflow.py
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

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from mlflow.models.signature import infer_signature

# Folders
MODEL_DIR = "models/satisfaction"
ARTIFACT_DIR = "artifacts/satisfaction"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# MLFLOW Setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Customer_Satisfaction")

# Load Data
df = pd.read_csv("data/passenger_cleaned.csv")

X = df.drop("satisfaction", axis=1)

y = df["satisfaction"].map({
    "neutral or dissatisfied": 0,
    "satisfied": 1
})

# Fix schema warning
num_cols = X.select_dtypes(exclude=["object"]).columns
X[num_cols] = X[num_cols].astype("float64")

categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

# Split function
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

best_acc = 0
best_model_uri = None
best_model_name = None
best_model = None

# Training Loop
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        mlflow.log_param("model_name", name)

        # Cross Validation
        cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        mlflow.log_metric("cv_accuracy", cv.mean())

        # Train
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # ROC-AUC (only if probability available)
        try:
            probs = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probs)
            mlflow.log_metric("roc_auc", roc_auc)
        except:
            pass

        # confusion matrix
        cm = confusion_matrix(y_test, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        cm_path = f"{ARTIFACT_DIR}/{name}_cm.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        # classification Report
        report = classification_report(y_test, preds)

        report_path = f"{ARTIFACT_DIR}/{name}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path)

        # API (no warning)
        logged = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_train, pipeline.predict(X_train))
        )

        # Save locally
        joblib.dump(pipeline, f"{MODEL_DIR}/{name}_model.pkl")

        # Best model tracking
        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_model_name = name
            best_model_uri = logged.model_uri

# Saving best model
joblib.dump(best_model, f"{MODEL_DIR}/satisfaction_best_model.pkl")

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_acc:.4f}")

# Register model
mlflow.register_model(best_model_uri, "Satisfaction_Best_Model")

print("Satisfaction training complete")
