# flight_eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_flight_eda(path):
    df = pd.read_csv(path)

    output_dir = "eda/flight"

    os.makedirs(output_dir,exist_ok=True)

    print("=" * 60)
    print("FLIGHT EDA")
    print("=" * 60)

    print("Dataset Shape:",df.shape)
    print("\nColumns:")
    print(df.columns)

    # PRICE DISTRIBUTION
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Price"],kde=True)
    plt.title("Flight Price Distribution")
    plt.savefig(f"{output_dir}/price_distribution.png")
    plt.close()
    
    # AIRLINE VS PRICE
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="Airline",y="Price",data=df)
    plt.xticks(rotation=90)
    plt.title("Airline vs Price")
    plt.savefig(f"{output_dir}/airline_vs_price.png")
    plt.close()

    # MONTH VS PRICE
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Journey_Month",y="Price",data=df)
    plt.title("Journey Month vs Price")
    plt.savefig(f"{output_dir}/month_vs_price.png")
    plt.close()
    
    # SOURCE VS PRICE
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Source",y="Price",data=df)
    plt.title("Source vs Price")
    plt.savefig(f"{output_dir}/source_price.png")
    plt.close()

    # DESTINATION VS PRICE
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Destination",y="Price",data=df)
    plt.title("Destination vs Price")
    plt.savefig(f"{output_dir}/destination_price.png")
    plt.close()

    # STOPS VS PRICE
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Total_Stops_Count",y="Price",data=df)
    plt.title("Stops vs Price")
    plt.savefig(f"{output_dir}/stops_price.png")
    plt.close()

    # MONTH VS AVERAGE PRICE
    month_price = (df.groupby("Journey_Month")["Price"].mean().reset_index())
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=month_price,x="Journey_Month",y="Price",marker="o")
    plt.title("Journey Month vs Average Price")
    plt.savefig(f"{output_dir}/month_price.png")
    plt.close()

    # DEPARTURE HOUR VS PRICE
    dep_price = (df.groupby("Dep_Hour")["Price"].mean().reset_index())
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=dep_price,x="Dep_Hour",y="Price",marker="o")
    plt.title("Departure Hour vs Price")
    plt.savefig(f"{output_dir}/departure_price.png")
    plt.close()

    # DURATION VS PRICE
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df,x="Total_Duration_Minutes",y="Price",hue="Airline")
    plt.title("Duration vs Price")
    plt.savefig(f"{output_dir}/duration_price.png")
    plt.close()

    # CORRELATION HEATMAP
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.select_dtypes(include="number").corr(),annot=True,cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # AIRLINE DISTRIBUTION
    plt.figure(figsize=(8, 8))
    df["Airline"].value_counts().plot(kind="pie",autopct="%1.1f%%")

    plt.ylabel("")
    plt.title("Flights by Airline")
    plt.savefig(f"{output_dir}/airline_distribution.png")
    plt.close()

    print("\nFlight EDA plots saved successfully.")
    print("=" * 60)

if __name__ == "__main__":
    perform_flight_eda("data/flight_cleaned.csv")
    print("Flight EDA completed successfully.")