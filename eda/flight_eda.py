# flight_eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_flight_eda(path):

    df = pd.read_csv(path)
    output_dir = "eda/flight"
    os.makedirs(output_dir, exist_ok=True)

    print("Dataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

    # Price Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df["Price"], kde=True)
    plt.title("Flight Price Distribution")
    plt.savefig(f"{output_dir}//price_distribution.png")
    plt.close()
    
    # Airline vs Price
    plt.figure(figsize=(10,5))
    sns.boxplot(x="Airline", y="Price", data=df)
    plt.xticks(rotation=90)
    plt.title("Airline vs Price")
    plt.savefig(f"{output_dir}/airline_vs_price.png")
    plt.close()

    # Month vs Price
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Journey_Month", y="Price", data=df)
    plt.title("Journey Month vs Price")
    plt.savefig(f"{output_dir}//month_vs_price.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{output_dir}//correlation_heatmap.png")
    plt.close()

    print("Flight EDA plots are saved successfully.")

# Execute script
if __name__ == "__main__":
    perform_flight_eda("data/flight_cleaned.csv")
    print("Flight EDA completed successfully.")
