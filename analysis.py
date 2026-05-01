import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------------
# TITLE
# ---------------------------
st.title("🍽️ Restaurant Profit Optimization Dashboard")
st.markdown("Predict restaurant profit using machine learning")

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("dataset.csv")

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
# FEATURES (IMPORTANT)
# ---------------------------
features = ['AOV', 'Orders', 'CommissionRate', 'DeliveryCost']
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
# MODEL PERFORMANCE
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
# PREDICTION (CORRECT WAY)
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
# SCENARIO ANALYSIS (UPGRADE)
# ---------------------------
st.subheader("📈 Scenario Analysis")

low_commission_data = input_data.copy()
low_commission_data['CommissionRate'] = 0.1

high_commission_data = input_data.copy()
high_commission_data['CommissionRate'] = 0.3

low_profit = model.predict(low_commission_data)[0]
high_profit = model.predict(high_commission_data)[0]

st.write(f"Profit with LOW commission (10%): {low_profit:.2f}")
st.write(f"Profit with HIGH commission (30%): {high_profit:.2f}")

# ---------------------------
# BUSINESS INSIGHTS
# ---------------------------
st.subheader("📊 Business Insights")

st.write("""
- Higher order volume significantly increases profit  
- High commission rates reduce overall profitability  
- Delivery cost has a moderate negative impact  
- Optimizing commission can improve margins  
""")

st.write("Low Commission Profit:", low_commission)
st.write("High Commission Profit:", high_commission)

st.subheader("💰 Predicted Profit")
st.success(f"{int(prediction)}")
st.success("✅ Prediction Generated Successfully")

# Recommendation
if prediction > 50000:
    st.success("✅ High Profit Strategy")
else:
    st.warning("⚠️ Consider optimizing inputs")
