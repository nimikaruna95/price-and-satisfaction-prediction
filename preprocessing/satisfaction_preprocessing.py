# satisfaction_preprocessing.py
import pandas as pd

def preprocess_satisfaction_data(input_path, output_path):

    df = pd.read_csv(input_path)
    print("Original Shape:", df.shape)

    df.drop_duplicates(inplace=True)

    # Fill missing delay
    df["Arrival Delay in Minutes"].fillna(0, inplace=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean target
    df["satisfaction"] = df["satisfaction"].str.strip().str.lower()

    drop_cols = []

    for col in df.columns:
        if "unnamed" in col.lower() or col.lower() == "id":
            drop_cols.append(col)

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    print("Dropped columns:", drop_cols)

    # Service Score
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

    # Delay Feature 
    df["Total_Delay"] = (
        df["Departure Delay in Minutes"] +
        df["Arrival Delay in Minutes"]
    )

    # Age Group
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 45, 60, 100],
        labels=["Teen", "Young Adult", "Adult", "Senior", "Elder"]
    )

    df.to_csv(output_path, index=False)

    print("Customer cleaned csv file saved ")
    print("Final Shape:", df.shape)

if __name__ == "__main__":
    preprocess_satisfaction_data( "data/Passenger_Satisfaction.csv", "data/passenger_cleaned.csv"    )
    print("Satisfaction preprocessing completed successfully ")
