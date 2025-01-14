import pandas as pd

def load_and_preprocess_customer_data(file_path, save_cleaned_path=None):
    df = pd.read_csv(file_path)

    df.info()
    
    #cleaning and preprocessing
    df['Arrival Delay in Minutes'].fillna(0, inplace=True)
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].astype(int)
    df.dropna(inplace=True)

    # Clean column names by reomving extra spaces
    df.columns = df.columns.str.strip()

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    if save_cleaned_path:
        df.to_csv(save_cleaned_path, index=False)

    return df

file_path = r"C:\Users\DELL\Desktop\flight_customer_app\customer_satisfaction\data\Passenger_Satisfaction.csv"
save_cleaned_path = r"C:\Users\DELL\Desktop\flight_customer_app\customer_satisfaction\data\cleaned_customer_data.csv"

cleaned_data = load_and_preprocess_customer_data(file_path, save_cleaned_path)

print(cleaned_data.head())
cleaned_data.info()
cleaned_data.isna().sum()
