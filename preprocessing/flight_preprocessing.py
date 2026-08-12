# flight_preprocessing.py
import pandas as pd
import os

def preprocess_flight_data(input_path, output_path):

    # READING FILE 
    df = pd.read_csv(input_path)

    print("=" * 60)
    print("FLIGHT DATA PREPROCESSING")
    print("=" * 60)

    print("Original Shape:", df.shape)

    # REMOVE DUPLICATES
    duplicate_count = df.duplicated().sum()
    print("Duplicate Rows:", duplicate_count)

    df.drop_duplicates(inplace=True)

    # HANDLE MISSING VALUES
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    df.dropna(inplace=True)
    print("\nShape After Cleaning:", df.shape)

    # ROUTE CLEANING
    df["Route"] = df["Route"].str.replace("?","→",regex=False)
    
    # DATE FEATURES
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"],format="%d/%m/%Y")

    df["Journey_Day"] = df["Date_of_Journey"].dt.day
    df["Journey_Month"] = df["Date_of_Journey"].dt.month
    df["Journey_Year"] = df["Date_of_Journey"].dt.year

    # Weekend
    df["Is_Weekend"] = (df["Date_of_Journey"].dt.dayofweek >= 5).astype(int)

    # DEPARTURE TIME
    df["Dep_Hour"] = pd.to_datetime(df["Dep_Time"],format="%H:%M").dt.hour

    # ARRIVAL TIME
    df["Arrival_Time"] = df["Arrival_Time"].apply(lambda x: str(x).split()[0])
    df["Arrival_Hour"] = pd.to_datetime(df["Arrival_Time"],format="%H:%M").dt.hour

    # DURATION FEATURES
    def convert_duration(x):
        h = 0
        m = 0

        x = str(x).strip()
        if "h" in x:
            h = int(x.split("h")[0].strip())

        if "m" in x:
            if "h" in x:
                m = int(x.split("h")[1].replace("m", "").strip())
            else:
                m = int(x.replace("m", "").strip())
        total_minutes = h * 60 + m
        return h, m, total_minutes

    df[["Duration_Hours","Duration_Minutes","Total_Duration_Minutes"]] = df["Duration"].apply(lambda x: pd.Series(convert_duration(x)))
   
    # TOTAL STOPS
    df["Total_Stops_Count"] = df["Total_Stops"].map({
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4 })

    # Check unmapped values
    if df["Total_Stops_Count"].isnull().sum() > 0:
        print("\nWarning: Unknown Total_Stops values found.")
        print(df.loc[df["Total_Stops_Count"].isnull(),"Total_Stops"].value_counts())
        df.dropna(subset=["Total_Stops_Count"],inplace=True)

    # DROP THE UNUSED COLUMNS
    df.drop(columns=["Date_of_Journey","Dep_Time","Arrival_Time","Duration","Total_Stops"],inplace=True)

    # SAVE THE CLEANED DATA
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df.to_csv(output_path,index=False)

    # FINAL INFORMATION
    print("\nFinal Shape:", df.shape)
    print("\nFinal Columns:")
    for col in df.columns:
        print("-", col)
    print("\nFlight cleaned CSV saved successfully:")

    print(output_path)
    print("=" * 60)

if __name__ == "__main__":
    preprocess_flight_data("data/Flight_Price.csv","data/flight_cleaned.csv")
    print("Flight preprocessing completed successfully.")