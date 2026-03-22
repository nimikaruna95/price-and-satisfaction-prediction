# flight_preprocessing.py
import pandas as pd
def preprocess_flight_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Cleaning 
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Route Features
    df["Route"] = df["Route"].str.replace("?", "→", regex=False)
    df["Route_List"] = df["Route"].str.split("→")

    df["Route_Start"] = df["Route_List"].apply(lambda x: x[0].strip())
    df["Route_End"] = df["Route_List"].apply(lambda x: x[-1].strip())
    df["Route_Stops_Count"] = df["Route_List"].apply(lambda x: len(x) - 1)

    # Date Features
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")

    df["Journey_Day"] = df["Date_of_Journey"].dt.day
    df["Journey_Month"] = df["Date_of_Journey"].dt.month
    df["Journey_Year"] = df["Date_of_Journey"].dt.year

    # Time Features
    df["Dep_Hour"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.hour

    df["Arrival_Time"] = df["Arrival_Time"].apply(lambda x: x.split()[0])
    df["Arrival_Hour"] = pd.to_datetime(df["Arrival_Time"], format="%H:%M").dt.hour

    # Duration convertion
    def convert_duration(x):
        h, m = 0, 0
        if "h" in x:
            h = int(x.split("h")[0])
        if "m" in x:
            m = int(x.split("h")[-1].replace("m", "").strip()) if "h" in x else int(x.replace("m", ""))
        return h, m, h * 60 + m

    df[["Duration_Hours", "Duration_Minutes", "Total_Duration_Minutes"]] = df["Duration"].apply(
        lambda x: pd.Series(convert_duration(x))
    )

    # Total Stops
    df["Total_Stops_Count"] = df["Total_Stops"].map({
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    })

    # Droping unused columns   
    drop_cols = [
        "Date_of_Journey",
        "Dep_Time",
        "Arrival_Time",
        "Duration",
        "Total_Stops",
        "Route",
        "Route_List"
    ]

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Saving csv file
    df.to_csv(output_path, index=False)
    print("flight cleaned csv file is saved.")

if __name__ == "__main__":
    preprocess_flight_data("data/Flight_price.csv", "data/flight_cleaned.csv")
    print("Flight preprocessing completed successfully.")




