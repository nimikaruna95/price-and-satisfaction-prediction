import pandas as pd

def load_and_preprocess_flight_data(file_path, save_cleaned_path=None):
    # Load the dataset
    df = pd.read_csv(file_path)

    df.info()

    # Total_Stops
    df['Total_Stops'] = df['Total_Stops'].replace('non-stop', 0)
    df['Total_Stops'] = df['Total_Stops'].apply(lambda x: int(str(x).split()[0]) if isinstance(x, str) else x)
    df['Total_Stops'].fillna(0, inplace=True)
    df['Total_Stops'] = df['Total_Stops'].astype(int)

    # Route
    if 'Route' in df.columns:
        df['Route'] = df['Route'].str.replace('?', '->')
        df['Route'] = df.groupby(['Source', 'Destination'])['Route'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
        )

    # Convert date and time columns to datetime
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], errors='coerce')
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], errors='coerce')
    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'], errors='coerce')

    # Feature Engineering
    df['Journey_day'] = df['Date_of_Journey'].dt.day
    df['Journey_month'] = df['Date_of_Journey'].dt.month
    df['Journey_year'] = df['Date_of_Journey'].dt.year
    df['Dep_hour'] = df['Dep_Time'].dt.hour
    df['Dep_minute'] = df['Dep_Time'].dt.minute
    df['Arrival_hour'] = df['Arrival_Time'].dt.hour
    df['Arrival_minute'] = df['Arrival_Time'].dt.minute

    # Split duration into hours and minutes
    def split_duration(duration):
        hours = 0
        minutes = 0
        if isinstance(duration, str):
            if 'h' in duration:
                hours = int(duration.split('h')[0])
                duration = duration.split('h')[1].strip()
            if 'm' in duration:
                minutes = int(duration.split('m')[0].strip())
        return hours, minutes

    df[['Duration_hours', 'Duration_minutes']] = df['Duration'].apply(lambda x: pd.Series(split_duration(x)))

    df.drop(columns=['Date_of_Journey', 'Route', 'Additional_Info','Dep_Time','Duration','Arrival_Time'], inplace=True, errors='ignore')

    df.dropna(inplace=True)

    if save_cleaned_path:
        df.to_csv(save_cleaned_path, index=False)

    return df

file_path = r"C:\Users\DELL\Desktop\flight_customer_app\flight_price\data\Flight_Price.csv"
save_cleaned_path = r"C:\Users\DELL\Desktop\flight_customer_app\flight_price\data\cleaned_flight_data.csv"

cleaned_data = load_and_preprocess_flight_data(file_path, save_cleaned_path)

print(cleaned_data.head())
cleaned_data.info()
cleaned_data.isna().sum()
