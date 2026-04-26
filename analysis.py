import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Restaurant Profit Optimization", layout="wide")

st.title("🍽️ Restaurant Profit Optimization Dashboard")
st.markdown("Predict and optimize profit using machine learning")


# Load data
df = pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")
df.columns = df.columns.str.strip()


df['TotalProfit'] = (
    df['InStoreNetProfit'] +
    df['UberEatsNetProfit'] +
    df['DoorDashNetProfit'] +
    df['SelfDeliveryNetProfit']
)


X = df[['AOV', 'MonthlyOrders', 'CommissionRate',
        'DeliveryCostPerOrder', 'InStoreShare', 'UE_share', 'DD_share']]
y = df['TotalProfit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

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


st.subheader("💰 Predicted Profit")
st.success(f"{int(prediction)}")

# Recommendation
if prediction > 50000:
    st.success("✅ High Profit Strategy")
else:
    st.warning("⚠️ Consider optimizing inputs")
