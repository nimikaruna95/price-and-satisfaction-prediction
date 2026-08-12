# satisfaction_eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def perform_satisfaction_eda(path):

    df = pd.read_csv(path)
    output_dir = "eda/customer"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("=" * 60)
    print("CUSTOMER SATISFACTION EDA")
    print("=" * 60)

    print("Dataset Shape:",df.shape)
    print("\nColumns:")
    print(df.columns)

    # SATISFACTION DISTRIBUTION
    plt.figure(figsize=(8, 5))
    sns.countplot(x="satisfaction",data=df)
    plt.title("Satisfaction Distribution")
    plt.xticks(rotation=15)
    plt.savefig(f"{output_dir}/satisfaction_distribution.png")
    plt.close()
    
    # GENDER VS SATISFACTION
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Gender",hue="satisfaction",data=df)
    plt.title("Gender vs Satisfaction")
    plt.savefig(f"{output_dir}/gender_satisfaction.png")
    plt.close()

    # CUSTOMER TYPE
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Customer Type",hue="satisfaction",data=df)
    plt.title("Customer Type vs Satisfaction")
    plt.savefig(f"{output_dir}/customer_type.png")
    plt.close()

    # TRAVEL CLASS
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Class",hue="satisfaction",data=df)
    plt.title("Travel Class vs Satisfaction")
    plt.savefig(f"{output_dir}/travel_class.png")
    plt.close()

    # TYPE OF TRAVEL
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Type of Travel",hue="satisfaction",data=df)
    plt.title("Type of Travel vs Satisfaction")
    plt.savefig(f"{output_dir}/travel_type.png")
    plt.close()

    # AGE DISTRIBUTION
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"],kde=True)
    plt.title("Age Distribution")
    plt.savefig(f"{output_dir}/age_distribution.png")
    plt.close()

    # FLIGHT DISTANCE
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="Flight Distance",y="Age",hue="satisfaction",data=df)
    plt.title("Flight Distance vs Age")
    plt.savefig(f"{output_dir}/flight_distance.png")
    plt.close()

    # DELAY ANALYSIS
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="Departure Delay in Minutes",y="Arrival Delay in Minutes",hue="satisfaction",data=df)
    plt.title("Delay Analysis")
    plt.savefig(f"{output_dir}/delay_analysis.png")
    plt.close()

    # SERVICE RATINGS
    ratings = ["Inflight wifi service","Food and drink","Seat comfort","Cleanliness"]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[ratings])
    plt.title("Service Ratings")
    plt.savefig(f"{output_dir}/service_ratings.png")
    plt.close()

    # CORRELATION HEATMAP
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.select_dtypes(include="number").corr(),annot=True,cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/customer_heatmap.png")
    plt.close()

    print("\nSatisfaction EDA plots saved successfully.")
    print("=" * 60)

if __name__ == "__main__":
    perform_satisfaction_eda("data/passenger_cleaned.csv")
    print("Satisfaction EDA completed successfully.")