import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")
df.columns = df.columns.str.strip()

df['TotalProfit'] = (
    df['InStoreNetProfit'] +
    df['UberEatsNetProfit'] +
    df['DoorDashNetProfit'] +
    df['SelfDeliveryNetProfit']
)

features = ['AOV', 'MonthlyOrders', 'CommissionRate',
            'DeliveryCostPerOrder', 'InStoreShare', 'UE_share', 'DD_share']

X = df[features]
y = df['TotalProfit']

model = XGBRegressor(n_estimators=200, learning_rate=0.1)
model.fit(X, y)

# UI
st.title("Restaurant Profit Optimization")

aov = st.slider("AOV", 20, 100, 40)
orders = st.slider("Monthly Orders", 500, 3000, 1000)
commission = st.slider("Commission Rate", 0.1, 0.5, 0.25)
delivery = st.slider("Delivery Cost", 1.0, 6.0, 3.0)

instore = st.slider("InStore Share", 0.0, 1.0, 0.3)
ue = st.slider("UberEats Share", 0.0, 1.0, 0.4)
dd = st.slider("DoorDash Share", 0.0, 1.0, 0.2)

input_data = [[aov, orders, commission, delivery, instore, ue, dd]]
prediction = model.predict(input_data)[0]

st.subheader("Predicted Profit")
st.write(int(prediction))

# Recommendation
if prediction > 50000:
    st.success("High Profit Strategy ✅")
else:
    st.warning("Low Profit — adjust pricing or channels")