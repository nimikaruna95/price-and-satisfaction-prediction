# satisfaction_eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_satisfaction_eda(path):

    df = pd.read_csv(path)

    # Eda folder Creation
    output_dir = "eda/customer"
    os.makedirs(output_dir, exist_ok=True)

    print("Dataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

    # Satisfaction Distribution
    plt.figure()
    sns.countplot(x="satisfaction", data=df)
    plt.title("Satisfaction Distribution")
    plt.savefig(f"{output_dir}//satisfaction_distribution.png")
    plt.close()

    # Gender vs Age vs Satisfaction
    plt.figure()
    sns.boxplot(x="Gender", y="Age", hue="satisfaction", data=df)
    plt.title("Gender vs Age vs Satisfaction")
    plt.savefig(f"{output_dir}//gender_age.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}//correlation.png")
    plt.close()

    print("Satisfaction EDA plots are saved successfully.")

# Execute script
if __name__ == "__main__":
    perform_satisfaction_eda("data/passenger_cleaned.csv")
    print("Satisfaction EDA completed and saved in 'eda/' folder.")
