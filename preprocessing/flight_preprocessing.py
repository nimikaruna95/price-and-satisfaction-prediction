# flight_preprocessing.py
import pandas as pd

def preprocess_flight_data(input_path, output_path):
    df = pd.read_csv(input_path)
    print("Original Shape:", df.shape)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Route Features
    df["Route"] = df["Route"].str.replace("?", "→", regex=False)

    # Date Features
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_Day"] = df["Date_of_Journey"].dt.day
    df["Journey_Month"] = df["Date_of_Journey"].dt.month

    # Weekend feature
    df["Is_Weekend"] = df["Date_of_Journey"].dt.dayofweek.apply(
        lambda x: 1 if x >= 5 else 0)

    # Time Features
    df["Dep_Hour"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.hour
    df["Arrival_Time"] = df["Arrival_Time"].apply(lambda x: x.split()[0])
    df["Arrival_Hour"] = pd.to_datetime(df["Arrival_Time"], format="%H:%M").dt.hour

    # Duration Feature Extraction
    def convert_duration(x):
        h, m = 0, 0

        if "h" in x:
            h = int(x.split("h")[0].strip())

        if "m" in x:
            if "h" in x:
                m = int(x.split("h")[1].replace("m", "").strip())
            else:
                m = int(x.replace("m", "").strip())

        total_minutes = h * 60 + m

        return h, m, total_minutes

    # Apply and create 3 columns
    df[["Duration_Hours", "Duration_Minutes", "Total_Duration_Minutes"]] = df["Duration"].apply(
    lambda x: pd.Series(convert_duration(x)))

    # Stops mapping
    df["Total_Stops_Count"] = df["Total_Stops"].map({
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    })

    # Drop unused columns
    df.drop(columns=[
        "Date_of_Journey",
        "Dep_Time",
        "Arrival_Time",
        "Duration",
        "Total_Stops", 
    ], inplace=True)

    df.to_csv(output_path, index=False)
    print("flight cleaned csv file is saved.")

if __name__ == "__main__":
    preprocess_flight_data("data/Flight_Price.csv", "data/flight_cleaned.csv")
    print("flight preprocessing completed successfully.")



