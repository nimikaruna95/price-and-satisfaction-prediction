# satisfaction_preprocessing.py
import pandas as pd
import os


def preprocess_satisfaction_data(input_path,output_path):

    # READING FILE
    df = pd.read_csv(input_path)
    print("=" * 60)
    print("CUSTOMER SATISFACTION PREPROCESSING")
    print("=" * 60)
    print("Original Shape:", df.shape)

    df.columns = df.columns.str.strip()

    # REMOVE DUPLICATES
    duplicate_count = df.duplicated().sum()

    print("Duplicate Rows:",duplicate_count)
    df.drop_duplicates(inplace=True)

    # REMOVE UNNAMED / ID COLUMNS
    drop_cols = []
    for col in df.columns:
        if ("unnamed" in col.lower() or col.lower() == "id"):
            drop_cols.append(col)
    df.drop(columns=drop_cols,inplace=True,errors="ignore")

    print("Dropped columns:",drop_cols)

    # CLEAN TARGET
    df["satisfaction"] = (df["satisfaction"].astype(str).str.strip().str.lower())

    # HANDLE MISSING VALUES
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    # Arrival delay can legitimately be missing but,in some records, so fill it with 0.
    df["Arrival Delay in Minutes"] = (pd.to_numeric(df["Arrival Delay in Minutes"],errors="coerce").fillna(0))

    # Remove any remaining missing rows
    df.dropna(inplace=True)

    # SERVICE SCORE
    service_columns = [
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Inflight service",
        "Cleanliness"]

    df["Total_Service_Score"] = (df[service_columns].sum(axis=1))

    # TOTAL DELAY
    df["Total_Delay"] = (
        df["Departure Delay in Minutes"]
        +
        df["Arrival Delay in Minutes"]
    )

    # AGE GROUP
    df["Age_Group"] = pd.cut(df["Age"],bins=[0,18,30,45,60,100],labels=["Teen","Young Adult","Adult","Senior","Elder"],include_lowest=True
    )

    # Convert category to string so it is handled consistently by the ML pipeline.
    df["Age_Group"] = (df["Age_Group"].astype(str))

    # SAVE CLEANED DATA
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df.to_csv(output_path,index=False)

    # FINAL INFORMATION
    print("\nFinal Shape:",df.shape)
    print("\nFinal Columns:")

    for col in df.columns:
        print("-", col)
    print("\nCustomer cleaned CSV saved successfully:")
    print(output_path)
    print("=" * 60)

if __name__ == "__main__":
    preprocess_satisfaction_data("data/Passenger_Satisfaction.csv","data/passenger_cleaned.csv")
    print("Satisfaction preprocessing completed successfully.")