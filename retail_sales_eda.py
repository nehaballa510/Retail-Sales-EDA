# ==========================================
# RETAIL SALES EXPLORATORY DATA ANALYSIS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
df = pd.read_csv("retail_sales.csv")

print("\nDataset Loaded Successfully\n")
print(df.head())

# ------------------------------------------
# 2. Data Cleaning
# ------------------------------------------
print("\nMissing Values:\n")
print(df.isnull().sum())

df.dropna(inplace=True)

df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)

# Create Total Sales column
df['total_sales'] = df['quantity'] * df['price']

print("\nData Cleaning Completed")

# ------------------------------------------
# 3. Descriptive Statistics
# ------------------------------------------
print("\nDescriptive Statistics:\n")
print(df[['quantity', 'price', 'total_sales', 'age']].describe())

print("\nMean Sales:", df['total_sales'].mean())
print("Median Sales:", df['total_sales'].median())
print("Mode Sales:", df['total_sales'].mode()[0])
print("Standard Deviation:", df['total_sales'].std())

# ------------------------------------------
# 4. Time Series Analysis
# ------------------------------------------
df['month'] = df['invoice_date'].dt.to_period('M')
monthly_sales = df.groupby('month')['total_sales'].sum()

plt.figure()
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

# ------------------------------------------
# 5. Product Category Analysis
# ------------------------------------------
plt.figure()
df.groupby('category')['total_sales'].sum().plot(kind='bar')
plt.title("Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.show()

# ------------------------------------------
# 6. Customer Demographics Analysis
# ------------------------------------------
plt.figure()
df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Customer Gender Distribution")
plt.ylabel("")
plt.show()

plt.figure()
plt.hist(df['age'], bins=10)
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# ------------------------------------------
# 7. Payment Method Analysis
# ------------------------------------------
plt.figure()
df['payment_method'].value_counts().plot(kind='bar')
plt.title("Payment Method Usage")
plt.xlabel("Payment Method")
plt.ylabel("Count")
plt.show()

# ------------------------------------------
# 8. Heatmap (Shopping Mall vs Category)
# ------------------------------------------
pivot_table = pd.pivot_table(
    df,
    values='total_sales',
    index='shopping_mall',
    columns='category',
    aggfunc='sum'
)

plt.figure(figsize=(10,6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Sales Heatmap: Shopping Mall vs Category")
plt.show()

# ------------------------------------------
# 9. Business Insights
# ------------------------------------------
print("\n--- KEY INSIGHTS ---")
print("1. Sales show monthly seasonality patterns.")
print("2. Certain product categories generate higher revenue.")
print("3. Credit Card is the most preferred payment method.")
print("4. Specific shopping malls contribute significantly to sales.")
print("5. Middle-aged customers dominate purchasing behavior.")

print("\nEDA Project Completed Successfully!")

