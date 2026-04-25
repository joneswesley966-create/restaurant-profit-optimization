# 1. Import LIABRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# 2. LOAD DATA

df = pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")

# Clean column names
df.columns = df.columns.str.strip()
print("Columns:",df.columns)

# 3. FEAUTURE ENGINEERING

# Total Profit

df['Total Profit'] = (df['InStoreNetProfit'] +
                      df['UberEatsNetProfit'] +
                      df['DoorDashNetProfit'] +
                      df['SelfDeliveryNetProfit'])
print("\n--- TOTAL PROFIT ---")
print(f"Total Profit: {df['Total Profit'].sum():,.2f}")

# 4. EDA

#Correlation Heatmap

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True)
, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Profit by segment

sns.barplot(x='Segment',
y='Total Profit', data=df)
plt.title("Profit by Segment")
plt.show()

# Orders vs Profit

sns.scatterplot(x='MonthlyOrders',
y='Total Profit', data=df)
plt.title("Orders vs Profit")
plt.show()


# 5. MODEL BUILDING (XGBOOST)

features = ['AOV', 'MonthlyOrders', 'CommissionRate', 'DeliveryCostPerOrder',
            'InStoreShare', 'UE_share', 'DD_share']

X = df[features]
y = df['Total Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBRegressor(n_estimators=200,
learning_rate=0.1)
model.fit(X_train, y_train)

print("\n✅ Model training completed")

predictions = model.predict(X_test)

print("\n 🔹 Sample Predictions:")
print(predictions[:5])

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print("\n📊 Model Performance:")
print("MAE:", mae)

# 7. FEAUTURE IMPORTANCE

importance = model.feature_importances_

plt.figure()
plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()


# 8. PROFIT OPTIMIZATION

best_profit = -999999
best_aov = 0

for aov in range(20, 100):
    input_data = [[aov, 1200, 0.25, 3.5, 0.3, 0.4, 0.2]]
    profit = model.predict(input_data) [0]

    if profit > best_profit:
        best_profit = profit
        best_aov = aov

print("\nBest AOV:", best_aov)
print("Maximum Predicted Profit:", int(best_profit))

# 9. KPI METRICS

print("\n--- KPIs ---")
print("Total Profit:",
df['Total Profit'].sum())
print("Average Profit:",
df['Total Profit'].mean())
print("Best Segment:",
df.groupby('Segment')
['Total Profit'].mean().idxmax())














