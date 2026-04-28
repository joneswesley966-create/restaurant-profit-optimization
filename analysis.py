import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Restaurant Profit Optimization", layout="wide")

st.title("🍽️ Restaurant Profit Optimization Dashboard")
st.markdown("Predict profit based on opreational inputs")


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
orders = st.slider("Monthly Orders", 0, 1000, 200)
commission = st.slider("Commission Rate", 0.1, 0.5, 0.25)
delivery = st.slider("Delivery Cost", 1.0, 6.0, 3.0)

instore = st.slider("InStore Share", 0.0, 1.0, 0.3)
ue = st.slider("UberEats Share", 0.0, 1.0, 0.4)
dd = st.slider("DoorDash Share", 0.0, 1.0, 0.2)

input_data = [[aov, orders, commission, delivery, instore, ue, dd]]
prediction = model.predict(input_data)[0]


st.sidebar.header("Enter Inputs")

aov = st.sidebar.slider("Average Order Value", 0, 100, 50)
orders = st.sidebar.slider("Monthly Orders", 0, 1000, 200)
commission = st.sidebar.slider("Commission Rate", 0.0, 1.0, 0.2)
delivery = st.sidebar.slider("Delivery Cost", 0, 50, 10)

from sklearn.metrics import r2_score

y_pred = model.predict(X_test)
st.write("Model Accuracy (R²):", r2_score(y_test, y_pred))

st.subheader("📊 Business Insight")

st.write("""
- Higher order volume increases profit
- High commission reduces profit margins
- Balanced delivery cost improves efficiency
""")



st.subheader("💰 Predicted Profit")
st.success(f"{int(prediction)}")
st.success("✅ Prediction Generated Successfully")

# Recommendation
if prediction > 50000:
    st.success("✅ High Profit Strategy")
else:
    st.warning("⚠️ Consider optimizing inputs")
