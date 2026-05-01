import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------------
# TITLE
# ---------------------------
st.title("🍽️ Restaurant Profit Optimization Dashboard")
st.markdown("Predict restaurant profit using machine learning")

# ---------------------------
# LOAD DATA SAFELY
# ---------------------------
file_path = "SkyCity Auckland Restaurants & Bars.csv"

if not os.path.exists(file_path):
    st.error("❌ SkyCity Auckland Restaurants & Bars.csv not found in repository!")
    st.stop()

df = pd.read_csv(file_path)

# ---------------------------
# CLEAN COLUMN NAMES
# ---------------------------
df.columns = df.columns.str.strip()

# DEBUG (remove later if you want)
st.write("Columns in dataset:", df.columns)

# ---------------------------
# RENAME COLUMNS (SAFE MAPPING)
# ---------------------------
df = df.rename(columns={
    'AverageOrderValue': 'AOV',
    'TotalOrders': 'Orders',
    'DeliveryCostPerOrder': 'DeliveryCost',
    'Commission': 'CommissionRate'
})

# ---------------------------
# CREATE TARGET
# ---------------------------
df['TotalProfit'] = (
    df['InStoreNetProfit'] +
    df['UberEatsNetProfit'] +
    df['DoorDashNetProfit'] +
    df['SelfDeliveryNetProfit']
)

# ---------------------------
# FEATURES (NOW SAFE)
# ---------------------------
features = ['AOV', 'Orders', 'CommissionRate', 'DeliveryCost']

missing = [col for col in features if col not in df.columns]

if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

X = df[features]
y = df['TotalProfit']

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# MODEL
# ---------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# PERFORMANCE
# ---------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"R² Score: {r2:.2f}")

# ---------------------------
# USER INPUT
# ---------------------------
st.sidebar.header("Enter Business Inputs")

aov = st.sidebar.slider("Average Order Value", 0, 100, 50)
orders = st.sidebar.slider("Monthly Orders", 0, 1000, 200)
commission = st.sidebar.slider("Commission Rate", 0.0, 1.0, 0.2)
delivery = st.sidebar.slider("Delivery Cost", 0, 50, 10)

# ---------------------------
# PREDICTION
# ---------------------------
input_data = pd.DataFrame({
    'AOV': [aov],
    'Orders': [orders],
    'CommissionRate': [commission],
    'DeliveryCost': [delivery]
})

prediction = model.predict(input_data)

st.subheader("💰 Predicted Profit")
st.success(f"Estimated Profit: {prediction[0]:.2f}")

# ---------------------------
# SCENARIO ANALYSIS
# ---------------------------
st.subheader("📈 Scenario Analysis")

low = input_data.copy()
low['CommissionRate'] = 0.1

high = input_data.copy()
high['CommissionRate'] = 0.3

low_profit = model.predict(low)[0]
high_profit = model.predict(high)[0]

st.write(f"Profit with LOW commission (10%): {low_profit:.2f}")
st.write(f"Profit with HIGH commission (30%): {high_profit:.2f}")

# ---------------------------
# INSIGHTS
# ---------------------------
st.subheader("📊 Business Insights")

st.write("""
- Higher order volume increases profit  
- High commission reduces margins  
- Delivery cost impacts profitability  
- Optimizing commission improves revenue  
""")
