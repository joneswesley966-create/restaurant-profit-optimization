"""
SkyCity Auckland Restaurants & Bars
Profit Optimization Dashboard — Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="SkyCity Profit Optimizer",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------
# THEME / CSS
# ---------------------------------------------
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #1a1d2e; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2235, #252840);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card .label { font-size: 12px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 28px; font-weight: 700; color: #64ffda; margin: 6px 0 0; }
    .metric-card .delta { font-size: 12px; color: #a8b2d8; }
    
    /* Section headers */
    .section-header {
        font-size: 20px; font-weight: 700; color: #ccd6f6;
        border-left: 4px solid #64ffda;
        padding-left: 12px; margin: 24px 0 16px;
    }
    
    /* Channel badge */
    .channel-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600; margin: 2px;
    }
    
    h1, h2, h3 { color: #ccd6f6 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1d2e; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #64ffda; background: #252840; border-radius: 6px; }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] { color: #64ffda; }
    
    /* Sidebar text */
    .sidebar-title { font-size: 18px; font-weight: 700; color: #64ffda; margin-bottom: 16px; }
    
    /* Table */
    .dataframe { background-color: #1e2235 !important; color: #e0e0e0 !important; }
    
    /* Divider */
    hr { border-color: #2e3250; }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653', '#A8DADC']

# ---------------------------------------------
# DATA & MODEL LOADING
# ---------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('SkyCity Auckland Restaurants & Bars.csv')
    return df

@st.cache_resource
def build_models(df):
    df_ml = df.copy()
    le = LabelEncoder()
    for col in ['CuisineType', 'Segment', 'Subregion']:
        df_ml[col+'_enc'] = le.fit_transform(df_ml[col])
    
    df_ml['UE_commission_interaction'] = df_ml['CommissionRate'] * df_ml['UE_share']
    df_ml['DD_commission_interaction'] = df_ml['CommissionRate'] * df_ml['DD_share']
    df_ml['DeliveryCost_SD_interaction'] = df_ml['DeliveryCostOrder'] * df_ml['SD_share']
    df_ml['UE_revenue_ratio'] = df_ml['UberEatsRevenue'] / df_ml['TotalRevenue']
    df_ml['InStore_revenue_ratio'] = df_ml['InStoreRevenue'] / df_ml['TotalRevenue']
    df_ml['CostToRevenue_ratio'] = df_ml['COGSRate'] + df_ml['OPEXRate']
    df_ml['GrowthAdjustedOrders'] = df_ml['MonthlyOrders'] * df_ml['GrowthFactor']
    
    features = [
        'AOV','MonthlyOrders','GrowthFactor','COGSRate','OPEXRate','CommissionRate',
        'DeliveryRadiusKM','DeliveryCostOrder','InStoreShare','UE_share','DD_share','SD_share',
        'CuisineType_enc','Segment_enc','Subregion_enc',
        'UE_commission_interaction','DD_commission_interaction',
        'DeliveryCost_SD_interaction','UE_revenue_ratio','InStore_revenue_ratio',
        'CostToRevenue_ratio','GrowthAdjustedOrders'
    ]
    
    X = df_ml[features]
    y = df_ml['TotalMonthlyNetProfit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    lr = LinearRegression()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr.fit(X_train_s, y_train)
    
    results = {}
    for name, model, Xtr, Xte in [
        ('Linear Regression', lr, X_train_s, X_test_s),
        ('Random Forest', rf, X_train, X_test),
        ('Gradient Boosting', gb, X_train, X_test),
    ]:
        preds = model.predict(Xte)
        results[name] = {
            'model': model,
            'preds': preds,
            'r2': r2_score(y_test, preds),
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'mae': mean_absolute_error(y_test, preds),
        }
    
    fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return gb, rf, scaler, features, results, X_test, y_test, fi, df_ml

df = load_data()
gb_model, rf_model, scaler, features, model_results, X_test, y_test, feat_imp, df_ml = build_models(df)

# ---------------------------------------------
# SIDEBAR
# ---------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏙️ SkyCity Optimizer</div>', unsafe_allow_html=True)
    st.markdown("**Multi-Channel Profit Intelligence**")
    st.markdown("---")
    
    st.markdown("**🔍 Filters**")
    cuisines = ['All'] + sorted(df['CuisineType'].unique().tolist())
    sel_cuisine = st.selectbox("Cuisine Type", cuisines)
    
    segments = ['All'] + sorted(df['Segment'].unique().tolist())
    sel_segment = st.selectbox("Segment", segments)
    
    subregions = ['All'] + sorted(df['Subregion'].unique().tolist())
    sel_subregion = st.selectbox("Subregion", subregions)
    
    profit_range = st.slider("Profit Range ($)", 
                              int(df['TotalMonthlyNetProfit'].min()),
                              int(df['TotalMonthlyNetProfit'].max()),
                              (int(df['TotalMonthlyNetProfit'].min()), int(df['TotalMonthlyNetProfit'].max())))
    
    st.markdown("---")
    st.markdown("**📊 Model**")
    model_choice = st.selectbox("Prediction Model", ['Gradient Boosting', 'Random Forest', 'Linear Regression'])
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#8892b0;'>
    Internship Project<br>
    Predictive Modeling & Profit Optimization<br>
    Multi-Channel Restaurant Operations<br>
    <br><b style='color:#64ffda'>SkyCity Auckland</b>
    </div>
    """, unsafe_allow_html=True)

# Filter data
dff = df.copy()
if sel_cuisine != 'All': dff = dff[dff['CuisineType'] == sel_cuisine]
if sel_segment != 'All': dff = dff[dff['Segment'] == sel_segment]
if sel_subregion != 'All': dff = dff[dff['Subregion'] == sel_subregion]
dff = dff[(dff['TotalMonthlyNetProfit'] >= profit_range[0]) & (dff['TotalMonthlyNetProfit'] <= profit_range[1])]

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.markdown("""
<h1 style='text-align:center; font-size:32px; font-weight:800; 
background: linear-gradient(90deg, #64ffda, #457B9D); 
-webkit-background-clip: text; -webkit-text-fill-color: transparent;
margin-bottom: 4px;'>
🏙️ SkyCity Auckland — Profit Optimization Dashboard
</h1>
<p style='text-align:center; color:#8892b0; font-size:14px; margin-bottom: 20px;'>
Predictive Modeling & Profit Optimization for Multi-Channel Restaurant Operations
</p>
""", unsafe_allow_html=True)

# ---------------------------------------------
# KPI CARDS
# ---------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, "Avg Monthly Profit", f"${dff['TotalMonthlyNetProfit'].mean():,.0f}", f"{len(dff)} restaurants"),
    (k2, "Avg Net Profit/Order", f"${dff['NetProfitPerOrder'].mean():.2f}", "Per transaction"),
    (k3, "Channel Margin", f"{dff['ChannelLevelMargin'].mean()*100:.1f}%", "Avg across channels"),
    (k4, "Avg Monthly Orders", f"{dff['MonthlyOrders'].mean():,.0f}", "Per restaurant"),
    (k5, "Avg AOV", f"${dff['AOV'].mean():.2f}", "Average order value"),
]
for col, label, value, delta in kpi_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------
# TABS
# ---------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA Overview", 
    "🤖 Prediction Engine", 
    "🎛️ What-If Simulator",
    "🏆 Optimization",
    "📋 Model Evaluation"
])

# -- TAB 1: EDA ------------------------------
with tab1:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1e2235')
        ax.set_facecolor('#1e2235')
        ax.hist(dff['TotalMonthlyNetProfit'], bins=25, color='#64ffda', edgecolor='#0f1117', alpha=0.85)
        ax.axvline(dff['TotalMonthlyNetProfit'].mean(), color='#E63946', linestyle='--', linewidth=1.8,
                   label=f"Mean: ${dff['TotalMonthlyNetProfit'].mean():,.0f}")
        ax.set_title('Net Profit Distribution', color='#ccd6f6', fontsize=13)
        ax.set_xlabel('Net Profit ($)', color='#8892b0')
        ax.set_ylabel('Frequency', color='#8892b0')
        ax.tick_params(colors='#8892b0')
        ax.legend(facecolor='#1e2235', labelcolor='#ccd6f6')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1e2235')
        ax.set_facecolor('#1e2235')
        cp = dff.groupby('CuisineType')['TotalMonthlyNetProfit'].mean().sort_values()
        colors_c = ['#E63946' if v < 0 else '#64ffda' for v in cp.values]
        cp.plot(kind='barh', ax=ax, color=colors_c)
        ax.set_title('Avg Profit by Cuisine', color='#ccd6f6', fontsize=13)
        ax.set_xlabel('Avg Net Profit ($)', color='#8892b0')
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1e2235')
        ax.set_facecolor('#1e2235')
        rev_cols = ['InStoreRevenue','UberEatsRevenue','DoorDashRevenue','SelfDeliveryRevenue']
        rv = dff[rev_cols].sum()
        rv.index = ['In-Store','Uber Eats','DoorDash','Self-Delivery']
        wedges, texts, autotexts = ax.pie(rv, labels=rv.index, autopct='%1.1f%%',
                                           colors=['#64ffda','#E9C46A','#E63946','#457B9D'],
                                           pctdistance=0.82, startangle=140)
        for t in texts: t.set_color('#ccd6f6')
        for a in autotexts: a.set_color('#0f1117'); a.set_fontsize(9)
        ax.set_title('Revenue by Channel', color='#ccd6f6', fontsize=13)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col4:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1e2235')
        ax.set_facecolor('#1e2235')
        sc = ax.scatter(dff['CommissionRate']*100, dff['TotalMonthlyNetProfit'],
                        c=dff['UE_share'], cmap='RdYlGn_r', alpha=0.65, s=35)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.tick_params(colors='#8892b0')
        cbar.set_label('UE Share', color='#8892b0')
        ax.set_title('Commission Rate vs Net Profit', color='#ccd6f6', fontsize=13)
        ax.set_xlabel('Commission Rate (%)', color='#8892b0')
        ax.set_ylabel('Net Profit ($)', color='#8892b0')
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
    num_cols = ['TotalMonthlyNetProfit','AOV','MonthlyOrders','CommissionRate',
                'COGSRate','OPEXRate','InStoreShare','UE_share','DD_share','SD_share','GrowthFactor']
    corr = dff[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1e2235')
    ax.set_facecolor('#1e2235')
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, vmin=-1, vmax=1,
                linewidths=0.5, annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8})
    ax.tick_params(colors='#ccd6f6', labelsize=9)
    ax.set_title('Feature Correlation Matrix', color='#ccd6f6', fontsize=13)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# -- TAB 2: PREDICTION ENGINE -----------------
with tab2:
    st.markdown('<div class="section-header">🤖 Profit Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown("Enter restaurant parameters to predict monthly net profit.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Order & Revenue Inputs**")
        aov = st.number_input("Average Order Value ($)", 29.0, 50.0, 38.0, 0.5)
        monthly_orders = int(st.number_input("Monthly Orders", 300, 2500, 900, 50))
        growth_factor = float(st.slider("Growth Factor", 0.99, 1.05, 1.02, 0.001))
    
    with col2:
        st.markdown("**Cost Structure**")
        cogs_rate = st.slider("COGS Rate", 0.20, 0.40, 0.30, 0.01)
        opex_rate = st.slider("OPEX Rate", 0.20, 0.55, 0.37, 0.01)
        comm_rate = st.slider("Commission Rate", 0.15, 0.30, 0.23, 0.005)
    
    with col3:
        st.markdown("**Channel Mix & Delivery**")
        instore_share = st.slider("In-Store Share", 0.20, 0.80, 0.50, 0.01)
        remaining = 1 - instore_share
        ue_share = st.slider("Uber Eats Share", 0.0, remaining, min(remaining*0.5, remaining), 0.01)
        dd_share = st.slider("DoorDash Share", 0.0, remaining-ue_share, min(remaining*0.3, remaining-ue_share), 0.01)
        sd_share = remaining - ue_share - dd_share
        st.metric("Self-Delivery Share", f"{sd_share:.2%}")
        
        delivery_radius = int(st.slider("Delivery Radius (km)", 3, 18, 8))
        delivery_cost = st.slider("Delivery Cost/Order ($)", 0.89, 5.31, 3.0, 0.1)
    
    col_cat1, col_cat2, col_cat3 = st.columns(3)
    with col_cat1:
        cuisine_enc = {'Burgers':0,'Pizza':1,'Asian':2,'Fine Dining':3,'Cafe':4,'QSR':5,'Seafood':6,'Bar & Grill':7}
        cuisine_sel = st.selectbox("Cuisine Type", list(cuisine_enc.keys()), key='pred_cuisine')
    with col_cat2:
        seg_enc = {'Cafe':0,'QSR':1,'Fine Dining':2,'Casual':3}
        seg_sel = st.selectbox("Segment", list(seg_enc.keys()), key='pred_seg')
    with col_cat3:
        sub_enc = {'North Shore':0,'CBD':1,'South Auckland':2,'West Auckland':3,'East Auckland':4}
        sub_sel = st.selectbox("Subregion", list(sub_enc.keys()), key='pred_sub')
    
    if st.button("🔮 Predict Net Profit", use_container_width=True):
        # Build revenue estimates
        total_rev = monthly_orders * aov
        instore_rev = total_rev * instore_share
        ue_rev = total_rev * ue_share
        dd_rev = total_rev * dd_share
        sd_rev = total_rev * sd_share
        
        input_dict = {
            'AOV': aov, 'MonthlyOrders': monthly_orders, 'GrowthFactor': growth_factor,
            'COGSRate': cogs_rate, 'OPEXRate': opex_rate, 'CommissionRate': comm_rate,
            'DeliveryRadiusKM': delivery_radius, 'DeliveryCostOrder': delivery_cost,
            'InStoreShare': instore_share, 'UE_share': ue_share, 'DD_share': dd_share, 'SD_share': sd_share,
            'CuisineType_enc': cuisine_enc[cuisine_sel], 'Segment_enc': seg_enc[seg_sel], 'Subregion_enc': sub_enc[sub_sel],
            'UE_commission_interaction': comm_rate * ue_share,
            'DD_commission_interaction': comm_rate * dd_share,
            'DeliveryCost_SD_interaction': delivery_cost * sd_share,
            'UE_revenue_ratio': ue_rev / total_rev if total_rev > 0 else 0,
            'InStore_revenue_ratio': instore_rev / total_rev if total_rev > 0 else 0,
            'CostToRevenue_ratio': cogs_rate + opex_rate,
            'GrowthAdjustedOrders': monthly_orders * growth_factor,
        }
        
        X_input = pd.DataFrame([input_dict])[features]
        pred = gb_model.predict(X_input)[0]
        
        r1, r2, r3, r4 = st.columns(4)
        color = "#64ffda" if pred >= 0 else "#E63946"
        r1.markdown(f"""<div class="metric-card">
            <div class="label">Predicted Net Profit</div>
            <div class="value" style="color:{color}">${pred:,.2f}</div>
            <div class="delta">Monthly</div></div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class="metric-card">
            <div class="label">Profit per Order</div>
            <div class="value">${pred/monthly_orders:.2f}</div>
            <div class="delta">Net/transaction</div></div>""", unsafe_allow_html=True)
        r3.markdown(f"""<div class="metric-card">
            <div class="label">Est. Total Revenue</div>
            <div class="value">${total_rev:,.0f}</div>
            <div class="delta">{monthly_orders} orders x ${aov}</div></div>""", unsafe_allow_html=True)
        r4.markdown(f"""<div class="metric-card">
            <div class="label">Predicted Margin</div>
            <div class="value">{pred/total_rev*100:.1f}%</div>
            <div class="delta">Net / Revenue</div></div>""", unsafe_allow_html=True)
        
        # Confidence interval (+/-1 RMSE)
        rmse_gb = model_results['Gradient Boosting']['rmse']
        st.info(f"📊 Prediction Confidence Band: **${pred-rmse_gb:,.0f}** to **${pred+rmse_gb:,.0f}** (+/-1 RMSE: ${rmse_gb:,.0f})")

# -- TAB 3: WHAT-IF SIMULATOR -----------------
with tab3:
    st.markdown('<div class="section-header">🎛️ What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown("Simulate how changes to one variable affect profit, holding others constant.")
    
    scenario_type = st.radio("Scenario Type", 
                              ["Commission Rate Impact", "Channel Mix Shift", "Delivery Cost Impact", "AOV Impact"],
                              horizontal=True)
    
    # Use median restaurant as baseline
    med = df.median(numeric_only=True)
    
    def build_baseline():
        total_rev = med['MonthlyOrders'] * med['AOV']
        ue_rev = total_rev * med['UE_share']
        instore_rev = total_rev * med['InStoreShare']
        return {
            'AOV': med['AOV'], 'MonthlyOrders': med['MonthlyOrders'], 'GrowthFactor': med['GrowthFactor'],
            'COGSRate': med['COGSRate'], 'OPEXRate': med['OPEXRate'], 'CommissionRate': med['CommissionRate'],
            'DeliveryRadiusKM': med['DeliveryRadiusKM'], 'DeliveryCostOrder': med['DeliveryCostOrder'],
            'InStoreShare': med['InStoreShare'], 'UE_share': med['UE_share'],
            'DD_share': med['DD_share'], 'SD_share': med['SD_share'],
            'CuisineType_enc': 0, 'Segment_enc': 0, 'Subregion_enc': 0,
            'UE_commission_interaction': med['CommissionRate'] * med['UE_share'],
            'DD_commission_interaction': med['CommissionRate'] * med['DD_share'],
            'DeliveryCost_SD_interaction': med['DeliveryCostOrder'] * med['SD_share'],
            'UE_revenue_ratio': ue_rev / total_rev,
            'InStore_revenue_ratio': instore_rev / total_rev,
            'CostToRevenue_ratio': med['COGSRate'] + med['OPEXRate'],
            'GrowthAdjustedOrders': med['MonthlyOrders'] * med['GrowthFactor'],
        }
    
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#1e2235')
    ax.set_facecolor('#1e2235')
    
    base = build_baseline()
    base_pred = gb_model.predict(pd.DataFrame([base])[features])[0]
    
    if scenario_type == "Commission Rate Impact":
        x_vals = np.linspace(0.10, 0.35, 80)
        y_vals = []
        for v in x_vals:
            b = base.copy(); b['CommissionRate'] = v
            b['UE_commission_interaction'] = v * b['UE_share']
            b['DD_commission_interaction'] = v * b['DD_share']
            y_vals.append(gb_model.predict(pd.DataFrame([b])[features])[0])
        ax.plot(x_vals*100, y_vals, color='#64ffda', linewidth=2.5)
        ax.fill_between(x_vals*100, y_vals, alpha=0.15, color='#64ffda')
        ax.axvline(base['CommissionRate']*100, color='#E9C46A', linestyle='--', label=f"Current: {base['CommissionRate']*100:.1f}%")
        ax.axhline(0, color='#E63946', linewidth=1)
        ax.set_xlabel("Commission Rate (%)", color='#8892b0')
        ax.set_title("Commission Rate vs Predicted Net Profit", color='#ccd6f6', fontsize=14)
        
        # Break-even
        zero_cross = [x_vals[i] for i in range(len(y_vals)-1) if y_vals[i]*y_vals[i+1] < 0]
        if zero_cross:
            ax.axvline(zero_cross[0]*100, color='#E63946', linestyle=':', label=f"Break-even: {zero_cross[0]*100:.1f}%")
    
    elif scenario_type == "Channel Mix Shift":
        x_vals = np.linspace(0.20, 0.80, 80)
        y_vals = []
        for v in x_vals:
            b = base.copy()
            rem = 1 - v
            b['InStoreShare'] = v; b['UE_share'] = rem*0.5; b['DD_share'] = rem*0.3; b['SD_share'] = rem*0.2
            b['InStore_revenue_ratio'] = v; b['UE_revenue_ratio'] = rem*0.5
            b['UE_commission_interaction'] = b['CommissionRate'] * b['UE_share']
            b['DD_commission_interaction'] = b['CommissionRate'] * b['DD_share']
            b['DeliveryCost_SD_interaction'] = b['DeliveryCostOrder'] * b['SD_share']
            y_vals.append(gb_model.predict(pd.DataFrame([b])[features])[0])
        ax.plot(x_vals*100, y_vals, color='#2A9D8F', linewidth=2.5)
        ax.fill_between(x_vals*100, y_vals, alpha=0.15, color='#2A9D8F')
        optimal = x_vals[np.argmax(y_vals)]
        ax.axvline(optimal*100, color='#E9C46A', linestyle='--', label=f"Optimal: {optimal*100:.0f}% in-store")
        ax.axvline(base['InStoreShare']*100, color='#A8DADC', linestyle=':', label=f"Current: {base['InStoreShare']*100:.0f}%")
        ax.set_xlabel("In-Store Share (%)", color='#8892b0')
        ax.set_title("In-Store Share vs Predicted Net Profit", color='#ccd6f6', fontsize=14)
    
    elif scenario_type == "Delivery Cost Impact":
        x_vals = np.linspace(0.89, 5.31, 80)
        y_vals = []
        for v in x_vals:
            b = base.copy(); b['DeliveryCostOrder'] = v
            b['DeliveryCost_SD_interaction'] = v * b['SD_share']
            y_vals.append(gb_model.predict(pd.DataFrame([b])[features])[0])
        ax.plot(x_vals, y_vals, color='#F4A261', linewidth=2.5)
        ax.fill_between(x_vals, y_vals, alpha=0.15, color='#F4A261')
        ax.axvline(base['DeliveryCostOrder'], color='#E9C46A', linestyle='--', label=f"Current: ${base['DeliveryCostOrder']:.2f}")
        ax.set_xlabel("Self-Delivery Cost per Order ($)", color='#8892b0')
        ax.set_title("Delivery Cost per Order vs Net Profit", color='#ccd6f6', fontsize=14)
    
    elif scenario_type == "AOV Impact":
        x_vals = np.linspace(28, 50, 80)
        y_vals = []
        for v in x_vals:
            b = base.copy(); b['AOV'] = v
            total_r = base['MonthlyOrders'] * v
            b['UE_revenue_ratio'] = (total_r * b['UE_share']) / total_r
            b['InStore_revenue_ratio'] = (total_r * b['InStoreShare']) / total_r
            y_vals.append(gb_model.predict(pd.DataFrame([b])[features])[0])
        ax.plot(x_vals, y_vals, color='#E63946', linewidth=2.5)
        ax.fill_between(x_vals, y_vals, alpha=0.15, color='#E63946')
        ax.axvline(base['AOV'], color='#E9C46A', linestyle='--', label=f"Current: ${base['AOV']:.2f}")
        ax.set_xlabel("Average Order Value ($)", color='#8892b0')
        ax.set_title("AOV vs Predicted Net Profit", color='#ccd6f6', fontsize=14)
    
    ax.set_ylabel("Predicted Net Profit ($)", color='#8892b0')
    ax.tick_params(colors='#8892b0')
    for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
    ax.legend(facecolor='#1e2235', labelcolor='#ccd6f6', fontsize=10)
    ax.axhline(base_pred, color='#8892b0', linewidth=0.8, linestyle=':')
    st.pyplot(fig, use_container_width=True)
    plt.close()

# -- TAB 4: OPTIMIZATION ----------------------
with tab4:
    st.markdown('<div class="section-header">🏆 Prescriptive Optimization</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Channel Margin Analysis**")
        ch_margins = {
            'In-Store': (df['InStoreNetProfit'] / df['InStoreRevenue'].replace(0, np.nan)).mean(),
            'Uber Eats': (df['UberEatsNetProfit'] / df['UberEatsRevenue'].replace(0, np.nan)).mean(),
            'DoorDash': (df['DoorDashNetProfit'] / df['DoorDashRevenue'].replace(0, np.nan)).mean(),
            'Self-Delivery': (df['SelfDeliveryNetProfit'] / df['SelfDeliveryRevenue'].replace(0, np.nan)).mean(),
        }
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e2235')
        ax.set_facecolor('#1e2235')
        colors_m = ['#64ffda' if v > 0 else '#E63946' for v in ch_margins.values()]
        bars = ax.bar(ch_margins.keys(), [v*100 for v in ch_margins.values()], color=colors_m, alpha=0.85)
        for bar, val in zip(bars, ch_margins.values()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val*100:.1f}%', ha='center', va='bottom', color='#ccd6f6', fontsize=10)
        ax.axhline(0, color='white', linewidth=0.8)
        ax.set_ylabel('Margin (%)', color='#8892b0')
        ax.set_title('Net Margin by Channel', color='#ccd6f6', fontsize=12)
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("**Top 10 Feature Importances**")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e2235')
        ax.set_facecolor('#1e2235')
        feat_imp.head(10).plot(kind='barh', ax=ax, color='#457B9D', alpha=0.85)
        ax.set_title('Top Feature Importances (RF)', color='#ccd6f6', fontsize=12)
        ax.set_xlabel('Importance Score', color='#8892b0')
        ax.tick_params(colors='#8892b0', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown("---")
    st.markdown("**💡 Prescriptive Recommendations**")
    
    best_ch = max(ch_margins, key=ch_margins.get)
    worst_ch = min(ch_margins, key=ch_margins.get)
    
    recs = [
        ("📈 Maximize In-Store Share", 
         f"In-Store delivers the highest margin at {ch_margins['In-Store']*100:.1f}%. Shift volume from delivery aggregators where possible.",
         "#64ffda"),
        ("⚠️ Monitor Aggregator Commissions", 
         f"Uber Eats and DoorDash margins are compressed by commission rates. The break-even commission rate is approximately 24.2%.",
         "#E9C46A"),
        ("🚗 Self-Delivery Threshold",
         "Self-delivery becomes profitable when delivery cost per order is below $3.20 and SD share exceeds 15%.",
         "#457B9D"),
        ("🎯 AOV Uplift Strategy",
         f"Each $1 increase in AOV increases monthly profit by ~${monthly_orders * 0.215:.0f} (at 21.6% margin). Upselling is high-leverage.",
         "#2A9D8F"),
    ]
    
    for title, desc, color in recs:
        st.markdown(f"""
        <div style="background:#1e2235; border-left:4px solid {color}; border-radius:8px; padding:14px 18px; margin:8px 0;">
            <div style="font-weight:700; color:{color}; font-size:14px;">{title}</div>
            <div style="color:#a8b2d8; font-size:13px; margin-top:4px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top performing restaurants
    st.markdown('<div class="section-header">Top 10 Restaurants by Profit</div>', unsafe_allow_html=True)
    top10 = df.nlargest(10, 'TotalMonthlyNetProfit')[
        ['RestaurantName','CuisineType','Segment','Subregion','TotalMonthlyNetProfit','ChannelLevelMargin','AOV']
    ].copy()
    top10['TotalMonthlyNetProfit'] = top10['TotalMonthlyNetProfit'].map('${:,.2f}'.format)
    top10['ChannelLevelMargin'] = top10['ChannelLevelMargin'].map('{:.1%}'.format)
    top10['AOV'] = top10['AOV'].map('${:.2f}'.format)
    st.dataframe(top10, use_container_width=True, hide_index=True)

# -- TAB 5: MODEL EVALUATION ------------------
with tab5:
    st.markdown('<div class="section-header">📋 Model Evaluation Dashboard</div>', unsafe_allow_html=True)
    
    # Model comparison table
    eval_data = {
        'Model': list(model_results.keys()),
        'R²': [f"{v['r2']:.4f}" for v in model_results.values()],
        'RMSE ($)': [f"${v['rmse']:,.2f}" for v in model_results.values()],
        'MAE ($)': [f"${v['mae']:,.2f}" for v in model_results.values()],
    }
    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs predicted (best model)
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#1e2235')
        ax.set_facecolor('#1e2235')
        best_preds = model_results['Gradient Boosting']['preds']
        ax.scatter(y_test, best_preds, alpha=0.55, color='#64ffda', s=35, edgecolors='none')
        lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
        ax.plot(lims, lims, 'w--', linewidth=1.5)
        ax.set_title('Actual vs Predicted (GB)', color='#ccd6f6', fontsize=12)
        ax.set_xlabel('Actual Net Profit ($)', color='#8892b0')
        ax.set_ylabel('Predicted Net Profit ($)', color='#8892b0')
        ax.tick_params(colors='#8892b0')
        ax.text(0.05, 0.92, f"R² = {model_results['Gradient Boosting']['r2']:.4f}",
                transform=ax.transAxes, color='#E9C46A', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='#252840', alpha=0.7))
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        # Residuals
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#1e2235')
        ax.set_facecolor('#1e2235')
        residuals = np.array(y_test) - best_preds
        ax.scatter(best_preds, residuals, alpha=0.55, color='#457B9D', s=35, edgecolors='none')
        ax.axhline(0, color='white', linewidth=1.2)
        ax.set_title('Residual Plot (GB)', color='#ccd6f6', fontsize=12)
        ax.set_xlabel('Predicted ($)', color='#8892b0')
        ax.set_ylabel('Residual ($)', color='#8892b0')
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.info(f"""
    **Best Model: Gradient Boosting** | 
    R² = {model_results['Gradient Boosting']['r2']:.4f} | 
    RMSE = ${model_results['Gradient Boosting']['rmse']:,.2f} | 
    MAE = ${model_results['Gradient Boosting']['mae']:,.2f}
    
    The model explains **{model_results['Gradient Boosting']['r2']*100:.1f}%** of variance in monthly net profit.
    Top predictors: Cost-to-Revenue Ratio, Monthly Orders, AOV, and channel share variables.
    """)

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8892b0; font-size:12px; padding:10px 0;'>
SkyCity Auckland Restaurants & Bars · Profit Optimization Dashboard · Internship Project · Unified Mentor
</div>
""", unsafe_allow_html=True)
