import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("customer_segmentation_knn_dataset.csv")
# Check data
print(df.head())
print(df.info())
# Income vs Spending Score
plt.figure(figsize=(8,5))
plt.scatter(df["Annual_Income"],df["Spending_Score"],c=df["Segment"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()



# Segment distribution
plt.figure(figsize=(6,4))
df["Segment"].value_counts().plot(kind="bar")
plt.xlabel("Segment")
plt.ylabel("Customers")
plt.title("Customer Segment Count")
plt.show()

plt.hist(df["Age"],bins=20)
plt.xlabel("Age")
plt.ylabel("Customers")
plt.title("Age Distribution")
plt.show()
print(df.info())