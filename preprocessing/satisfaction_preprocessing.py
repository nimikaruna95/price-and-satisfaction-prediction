# satisfaction_preprocessing.py
import pandas as pd
def preprocess_satisfaction_data(input_path, output_path):

    df = pd.read_csv(input_path)
    print("Original Shape:", df.shape)

    df.drop_duplicates(inplace=True)
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)

    # Fix of the target
    df["satisfaction"] = df["satisfaction"].str.strip().str.lower()

    # Fixing Column Names
    df.columns = df.columns.str.strip()

    print("Original Columns:", df.columns.tolist())

    # Removing the index column which is first column
    if df.columns[0] in ["", "Unnamed: 0"] or df.iloc[:, 0].equals(pd.Series(range(len(df)))):
        df.drop(df.columns[0], axis=1, inplace=True)
        print("Dropped first index column")

    # Remove Columns
    drop_cols = []

    for col in df.columns:
        if "unnamed" in col.lower() or col.strip() in ["id", ",id", ""]:
            drop_cols.append(col)

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    print("Dropped columns:", drop_cols)

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
        "Cleanliness"
    ]
    
    df["Total_Service_Score"] = df[service_columns].sum(axis=1)

    # Age group feature
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 45, 60, 100],
        labels=["Teen", "Young Adult", "Adult", "Senior", "Elder"]
    )  
    #df.columns = df.columns.str.strip()
    df.to_csv(output_path, index=False)
    print("customer cleaned csv file is saved.")

if __name__ == "__main__":
    preprocess_satisfaction_data("data/Passenger_Satisfaction.csv", "data/passenger_cleaned.csv")
    print("Satisfaction preprocessing completed successfully.")

