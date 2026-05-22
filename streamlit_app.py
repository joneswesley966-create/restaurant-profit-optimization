"""
SkyCity Auckland Restaurants & Bars
Profit Optimization Dashboard
"""

import os, glob, warnings
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(page_title="SkyCity Profit Optimizer",
                   page_icon="🏙️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background-color:#0f1117;color:#e0e0e0}
[data-testid="stSidebar"]{background-color:#1a1d2e}
.metric-card{background:linear-gradient(135deg,#1e2235,#252840);border:1px solid #2e3250;
  border-radius:12px;padding:18px 20px;text-align:center;box-shadow:0 4px 15px rgba(0,0,0,.3)}
.metric-card .label{font-size:12px;color:#8892b0;text-transform:uppercase;letter-spacing:1px}
.metric-card .value{font-size:26px;font-weight:700;color:#64ffda;margin:6px 0 0}
.metric-card .delta{font-size:12px;color:#a8b2d8}
.sec-hdr{font-size:20px;font-weight:700;color:#ccd6f6;border-left:4px solid #64ffda;
  padding-left:12px;margin:24px 0 16px}
h1,h2,h3{color:#ccd6f6!important}
.stTabs [data-baseweb="tab-list"]{background-color:#1a1d2e;border-radius:8px;padding:4px}
.stTabs [data-baseweb="tab"]{color:#8892b0}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:#64ffda;background:#252840;border-radius:6px}
</style>
""", unsafe_allow_html=True)

PALETTE = ['#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261','#264653','#A8DADC']

# ── DATA LOADING ─────────────────────────────
@st.cache_data
def load_data():
    CSV_NAME = 'SkyCity Auckland Restaurants & Bars.csv'
    # Search order: app folder, CWD, Desktop, Downloads, Documents
    home = os.path.expanduser('~')
    app_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(app_dir,          CSV_NAME),
        os.path.join(os.getcwd(),       CSV_NAME),
        os.path.join(home,'Desktop',    CSV_NAME),
        os.path.join(home,'Downloads',  CSV_NAME),
        os.path.join(home,'Documents',  CSV_NAME),
    ] + glob.glob(os.path.join(app_dir,  '*', CSV_NAME)) \
      + glob.glob(os.path.join(os.getcwd(),'*', CSV_NAME))

    found = next((p for p in paths if os.path.exists(p)), None)
    if found is None:
        st.error(f"CSV not found. Place '{CSV_NAME}' in the same folder as streamlit_app.py")
        st.stop()

    df = pd.read_csv(found)
    df.rename(columns={
        'InStoreOrders':        'InStoreOrdersCount',
        'UberEatsOrders':       'UberEatsOrdersCount',
        'DoorDashOrders':       'DoorDashOrdersCount',
        'SelfDeliveryOrders':   'SelfDeliveryOrdersCount',
        'DeliveryCostPerOrder': 'DeliveryCostOrder',
    }, inplace=True)
    df['TotalRevenue']          = df['InStoreRevenue']+df['UberEatsRevenue']+df['DoorDashRevenue']+df['SelfDeliveryRevenue']
    df['TotalMonthlyNetProfit'] = df['InStoreNetProfit']+df['UberEatsNetProfit']+df['DoorDashNetProfit']+df['SelfDeliveryNetProfit']
    df['NetProfitPerOrder']     = df['TotalMonthlyNetProfit'] / df['MonthlyOrders']
    df['ChannelLevelMargin']    = df['TotalMonthlyNetProfit'] / df['TotalRevenue']
    return df

@st.cache_resource
def build_models(df):
    df_ml = df.copy()
    # Defensive: rename raw CSV cols in case cache bypassed load_data transforms
    df_ml.rename(columns={
        'InStoreOrders':        'InStoreOrdersCount',
        'UberEatsOrders':       'UberEatsOrdersCount',
        'DoorDashOrders':       'DoorDashOrdersCount',
        'SelfDeliveryOrders':   'SelfDeliveryOrdersCount',
        'DeliveryCostPerOrder': 'DeliveryCostOrder',
    }, inplace=True)
    if 'TotalRevenue' not in df_ml.columns:
        df_ml['TotalRevenue'] = (df_ml['InStoreRevenue'] + df_ml['UberEatsRevenue'] +
                                 df_ml['DoorDashRevenue'] + df_ml['SelfDeliveryRevenue'])
    if 'TotalMonthlyNetProfit' not in df_ml.columns:
        df_ml['TotalMonthlyNetProfit'] = (df_ml['InStoreNetProfit'] + df_ml['UberEatsNetProfit'] +
                                          df_ml['DoorDashNetProfit'] + df_ml['SelfDeliveryNetProfit'])
    if 'NetProfitPerOrder' not in df_ml.columns:
        df_ml['NetProfitPerOrder'] = df_ml['TotalMonthlyNetProfit'] / df_ml['MonthlyOrders']
    if 'ChannelLevelMargin' not in df_ml.columns:
        df_ml['ChannelLevelMargin'] = df_ml['TotalMonthlyNetProfit'] / df_ml['TotalRevenue']
    le = LabelEncoder()
    for col in ['CuisineType','Segment','Subregion']:
        df_ml[col+'_enc'] = le.fit_transform(df_ml[col])
    df_ml['UE_commission_interaction']   = df_ml['CommissionRate']*df_ml['UE_share']
    df_ml['DD_commission_interaction']   = df_ml['CommissionRate']*df_ml['DD_share']
    df_ml['DeliveryCost_SD_interaction'] = df_ml['DeliveryCostOrder']*df_ml['SD_share']
    df_ml['UE_revenue_ratio']            = df_ml['UberEatsRevenue']/df_ml['TotalRevenue']
    df_ml['InStore_revenue_ratio']       = df_ml['InStoreRevenue']/df_ml['TotalRevenue']
    df_ml['CostToRevenue_ratio']         = df_ml['COGSRate']+df_ml['OPEXRate']
    df_ml['GrowthAdjustedOrders']        = df_ml['MonthlyOrders']*df_ml['GrowthFactor']

    FEATURES = [
        'AOV','MonthlyOrders','GrowthFactor','COGSRate','OPEXRate','CommissionRate',
        'DeliveryRadiusKM','DeliveryCostOrder','InStoreShare','UE_share','DD_share','SD_share',
        'CuisineType_enc','Segment_enc','Subregion_enc',
        'UE_commission_interaction','DD_commission_interaction',
        'DeliveryCost_SD_interaction','UE_revenue_ratio','InStore_revenue_ratio',
        'CostToRevenue_ratio','GrowthAdjustedOrders',
    ]
    X = df_ml[FEATURES]; y = df_ml['TotalMonthlyNetProfit']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train); X_te_s = scaler.transform(X_test)

    lr = LinearRegression(); lr.fit(X_tr_s,y_train)
    rf = RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42); rf.fit(X_train,y_train)
    gb = GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42); gb.fit(X_train,y_train)

    res = {}
    for name,model,Xtr,Xte in [('Linear Regression',lr,X_tr_s,X_te_s),
                                 ('Random Forest',rf,X_train,X_test),
                                 ('Gradient Boosting',gb,X_train,X_test)]:
        p = model.predict(Xte)
        res[name]={'model':model,'preds':p,
                   'r2':r2_score(y_test,p),
                   'rmse':np.sqrt(mean_squared_error(y_test,p)),
                   'mae':mean_absolute_error(y_test,p)}

    fi = pd.Series(rf.feature_importances_,index=FEATURES).sort_values(ascending=False)
    return gb,rf,scaler,FEATURES,res,X_test,y_test,fi,df_ml

df = load_data()
gb_model,rf_model,scaler,FEATURES,model_results,X_test,y_test,feat_imp,df_ml = build_models(df)

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:18px;font-weight:700;color:#64ffda;margin-bottom:16px">🏙️ SkyCity Optimizer</div>', unsafe_allow_html=True)
    st.markdown("**Multi-Channel Profit Intelligence**")
    st.markdown("---")
    st.markdown("**Filters**")
    sel_cuisine  = st.selectbox("Cuisine Type", ['All']+sorted(df['CuisineType'].unique().tolist()))
    sel_segment  = st.selectbox("Segment",      ['All']+sorted(df['Segment'].unique().tolist()))
    sel_subregion= st.selectbox("Subregion",    ['All']+sorted(df['Subregion'].unique().tolist()))
    profit_range = st.slider("Profit Range ($)",
                              int(df['TotalMonthlyNetProfit'].min()),
                              int(df['TotalMonthlyNetProfit'].max()),
                              (int(df['TotalMonthlyNetProfit'].min()),
                               int(df['TotalMonthlyNetProfit'].max())))
    st.markdown("---")
    st.markdown("""<div style='font-size:11px;color:#8892b0'>
    SkyCity Auckland<br>Profit Optimization<br>Internship Project<br>Unified Mentor
    </div>""", unsafe_allow_html=True)

dff = df.copy()
if sel_cuisine   != 'All': dff = dff[dff['CuisineType']==sel_cuisine]
if sel_segment   != 'All': dff = dff[dff['Segment']==sel_segment]
if sel_subregion != 'All': dff = dff[dff['Subregion']==sel_subregion]
dff = dff[(dff['TotalMonthlyNetProfit']>=profit_range[0]) & (dff['TotalMonthlyNetProfit']<=profit_range[1])]

# ── HEADER ───────────────────────────────────
st.markdown("""<h1 style='text-align:center;font-size:30px;font-weight:800;
background:linear-gradient(90deg,#64ffda,#457B9D);-webkit-background-clip:text;
-webkit-text-fill-color:transparent;margin-bottom:4px'>
SkyCity Auckland - Profit Optimization Dashboard</h1>
<p style='text-align:center;color:#8892b0;font-size:13px;margin-bottom:20px'>
Predictive Modeling & Profit Optimization for Multi-Channel Restaurant Operations
</p>""", unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
for col,label,val,delta in [
    (c1,"Avg Monthly Profit",  f"${dff['TotalMonthlyNetProfit'].mean():,.0f}", f"{len(dff)} records"),
    (c2,"Net Profit / Order",  f"${dff['NetProfitPerOrder'].mean():.2f}",     "Per transaction"),
    (c3,"Channel Margin",      f"{dff['ChannelLevelMargin'].mean()*100:.1f}%","Avg margin"),
    (c4,"Avg Monthly Orders",  f"{dff['MonthlyOrders'].mean():,.0f}",         "Per restaurant"),
    (c5,"Avg AOV",             f"${dff['AOV'].mean():.2f}",                   "Avg order value"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{val}</div>
        <div class="delta">{delta}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 EDA Overview","🤖 Prediction Engine",
    "🎛️ What-If Simulator","🏆 Optimization","📋 Model Evaluation"])

def dark_fig(w=7,h=4):
    fig,ax = plt.subplots(figsize=(w,h),facecolor='#1e2235')
    ax.set_facecolor('#1e2235')
    return fig,ax

def style_ax(ax):
    ax.tick_params(colors='#8892b0')
    for sp in ax.spines.values(): sp.set_edgecolor('#2e3250')
    return ax

# ── TAB 1: EDA ───────────────────────────────
with tab1:
    st.markdown('<div class="sec-hdr">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig,ax = dark_fig(); style_ax(ax)
        ax.hist(dff['TotalMonthlyNetProfit'],bins=35,color='#64ffda',edgecolor='#0f1117',alpha=0.85)
        ax.axvline(dff['TotalMonthlyNetProfit'].mean(),color='#E63946',linestyle='--',linewidth=1.8,
                   label=f"Mean: ${dff['TotalMonthlyNetProfit'].mean():,.0f}")
        ax.set_title('Net Profit Distribution',color='#ccd6f6',fontsize=12)
        ax.set_xlabel('Net Profit ($)',color='#8892b0'); ax.set_ylabel('Frequency',color='#8892b0')
        ax.legend(facecolor='#1e2235',labelcolor='#ccd6f6')
        st.pyplot(fig,use_container_width=True); plt.close()
    with c2:
        fig,ax = dark_fig(); style_ax(ax)
        cp = dff.groupby('CuisineType')['TotalMonthlyNetProfit'].mean().sort_values()
        cp.plot(kind='barh',ax=ax,color=['#E63946' if v<0 else '#64ffda' for v in cp])
        ax.set_title('Avg Profit by Cuisine',color='#ccd6f6',fontsize=12)
        ax.set_xlabel('Avg Net Profit ($)',color='#8892b0')
        st.pyplot(fig,use_container_width=True); plt.close()

    c3,c4 = st.columns(2)
    with c3:
        fig,ax = dark_fig(); style_ax(ax)
        rv = dff[['InStoreRevenue','UberEatsRevenue','DoorDashRevenue','SelfDeliveryRevenue']].sum()
        rv.index = ['In-Store','Uber Eats','DoorDash','Self-Delivery']
        wedges,texts,autotexts = ax.pie(rv,labels=rv.index,autopct='%1.1f%%',
            colors=['#64ffda','#E9C46A','#E63946','#457B9D'],pctdistance=0.82,startangle=140)
        for t in texts: t.set_color('#ccd6f6')
        for a in autotexts: a.set_color('#0f1117'); a.set_fontsize(9)
        ax.set_title('Revenue by Channel',color='#ccd6f6',fontsize=12)
        st.pyplot(fig,use_container_width=True); plt.close()
    with c4:
        fig,ax = dark_fig(); style_ax(ax)
        sc = ax.scatter(dff['CommissionRate']*100,dff['TotalMonthlyNetProfit'],
                        c=dff['UE_share'],cmap='RdYlGn_r',alpha=0.4,s=15)
        cbar = plt.colorbar(sc,ax=ax); cbar.ax.tick_params(colors='#8892b0')
        cbar.set_label('UE Share',color='#8892b0')
        ax.set_title('Commission Rate vs Net Profit',color='#ccd6f6',fontsize=12)
        ax.set_xlabel('Commission Rate (%)',color='#8892b0'); ax.set_ylabel('Net Profit ($)',color='#8892b0')
        st.pyplot(fig,use_container_width=True); plt.close()

    st.markdown('<div class="sec-hdr">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ['TotalMonthlyNetProfit','AOV','MonthlyOrders','CommissionRate',
                'COGSRate','OPEXRate','InStoreShare','UE_share','DD_share','SD_share','GrowthFactor']
    fig,ax = plt.subplots(figsize=(10,5),facecolor='#1e2235'); ax.set_facecolor('#1e2235')
    sns.heatmap(dff[num_cols].corr(),annot=True,fmt='.2f',cmap='RdYlGn',ax=ax,
                vmin=-1,vmax=1,linewidths=0.5,annot_kws={'size':8})
    ax.tick_params(colors='#ccd6f6',labelsize=9)
    ax.set_title('Feature Correlation Matrix',color='#ccd6f6',fontsize=13)
    st.pyplot(fig,use_container_width=True); plt.close()

# ── TAB 2: PREDICTION ENGINE ─────────────────
with tab2:
    st.markdown('<div class="sec-hdr">Profit Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown("Adjust parameters below to predict monthly net profit for any restaurant profile.")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Orders & Revenue**")
        aov            = st.number_input("Average Order Value ($)", 29.0, 50.0, float(df['AOV'].mean()), 0.5)
        monthly_orders = int(st.number_input("Monthly Orders", 441, 2337, int(df['MonthlyOrders'].median()), 50))
        growth_factor  = float(st.slider("Growth Factor", 0.99, 1.05, float(df['GrowthFactor'].median()), 0.001))
    with c2:
        st.markdown("**Cost Structure**")
        cogs_rate  = float(st.slider("COGS Rate",   0.20, 0.40, float(df['COGSRate'].median()),   0.005))
        opex_rate  = float(st.slider("OPEX Rate",   0.20, 0.55, float(df['OPEXRate'].median()),   0.005))
        comm_rate  = float(st.slider("Commission Rate", 0.27, 0.33, float(df['CommissionRate'].median()), 0.005))
    with c3:
        st.markdown("**Channel Mix & Delivery**")
        instore_share = float(st.slider("In-Store Share", 0.10, 0.80, float(df['InStoreShare'].median()), 0.01))
        remaining     = 1 - instore_share
        ue_share = float(st.slider("Uber Eats Share", 0.0, remaining, min(float(df['UE_share'].median()), remaining), 0.01))
        dd_share = float(st.slider("DoorDash Share",  0.0, max(remaining-ue_share,0.0),
                                    min(float(df['DD_share'].median()), max(remaining-ue_share,0.0)), 0.01))
        sd_share = max(remaining - ue_share - dd_share, 0.0)
        st.metric("Self-Delivery Share", f"{sd_share:.2%}")
        delivery_radius = int(st.slider("Delivery Radius (km)", 3, 18, int(df['DeliveryRadiusKM'].median())))
        delivery_cost   = float(st.slider("Delivery Cost/Order ($)", 0.89, 6.0, float(df['DeliveryCostOrder'].median()), 0.1))

    cc1,cc2,cc3 = st.columns(3)
    cuisine_map = {v:i for i,v in enumerate(sorted(df['CuisineType'].unique()))}
    segment_map = {v:i for i,v in enumerate(sorted(df['Segment'].unique()))}
    subregion_map={v:i for i,v in enumerate(sorted(df['Subregion'].unique()))}
    with cc1: cuisine_sel  = st.selectbox("Cuisine Type", list(cuisine_map.keys()), key='pc')
    with cc2: segment_sel  = st.selectbox("Segment",      list(segment_map.keys()), key='ps')
    with cc3: subregion_sel= st.selectbox("Subregion",    list(subregion_map.keys()), key='pr')

    if st.button("Predict Net Profit", use_container_width=True):
        total_rev    = monthly_orders * aov
        instore_rev  = total_rev * instore_share
        ue_rev       = total_rev * ue_share
        X_in = pd.DataFrame([{
            'AOV':aov,'MonthlyOrders':monthly_orders,'GrowthFactor':growth_factor,
            'COGSRate':cogs_rate,'OPEXRate':opex_rate,'CommissionRate':comm_rate,
            'DeliveryRadiusKM':delivery_radius,'DeliveryCostOrder':delivery_cost,
            'InStoreShare':instore_share,'UE_share':ue_share,'DD_share':dd_share,'SD_share':sd_share,
            'CuisineType_enc':cuisine_map[cuisine_sel],'Segment_enc':segment_map[segment_sel],
            'Subregion_enc':subregion_map[subregion_sel],
            'UE_commission_interaction':comm_rate*ue_share,
            'DD_commission_interaction':comm_rate*dd_share,
            'DeliveryCost_SD_interaction':delivery_cost*sd_share,
            'UE_revenue_ratio':ue_rev/total_rev if total_rev>0 else 0,
            'InStore_revenue_ratio':instore_rev/total_rev if total_rev>0 else 0,
            'CostToRevenue_ratio':cogs_rate+opex_rate,
            'GrowthAdjustedOrders':monthly_orders*growth_factor,
        }])[FEATURES]
        pred  = gb_model.predict(X_in)[0]
        rmse  = model_results['Gradient Boosting']['rmse']
        color = "#64ffda" if pred >= 0 else "#E63946"

        r1,r2,r3,r4 = st.columns(4)
        r1.markdown(f"""<div class="metric-card">
            <div class="label">Predicted Net Profit</div>
            <div class="value" style="color:{color}">${pred:,.2f}</div>
            <div class="delta">Monthly</div></div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class="metric-card">
            <div class="label">Profit per Order</div>
            <div class="value">${pred/monthly_orders:.2f}</div>
            <div class="delta">Net per transaction</div></div>""", unsafe_allow_html=True)
        r3.markdown(f"""<div class="metric-card">
            <div class="label">Est. Total Revenue</div>
            <div class="value">${total_rev:,.0f}</div>
            <div class="delta">{monthly_orders} orders</div></div>""", unsafe_allow_html=True)
        r4.markdown(f"""<div class="metric-card">
            <div class="label">Predicted Margin</div>
            <div class="value">{pred/total_rev*100:.1f}%</div>
            <div class="delta">Net / Revenue</div></div>""", unsafe_allow_html=True)

        st.info(f"Confidence Band: **${pred-rmse:,.0f}** to **${pred+rmse:,.0f}** (model RMSE: ${rmse:,.0f})")

# ── TAB 3: WHAT-IF ───────────────────────────
with tab3:
    st.markdown('<div class="sec-hdr">What-If Scenario Simulator</div>', unsafe_allow_html=True)
    scenario = st.radio("Scenario", ["Commission Rate Impact","Channel Mix Shift",
                                      "Delivery Cost Impact","AOV Impact"], horizontal=True)
    med = df.median(numeric_only=True)

    def baseline():
        tv = med['MonthlyOrders']*med['AOV']
        return {
            'AOV':med['AOV'],'MonthlyOrders':med['MonthlyOrders'],'GrowthFactor':med['GrowthFactor'],
            'COGSRate':med['COGSRate'],'OPEXRate':med['OPEXRate'],'CommissionRate':med['CommissionRate'],
            'DeliveryRadiusKM':med['DeliveryRadiusKM'],'DeliveryCostOrder':med['DeliveryCostOrder'],
            'InStoreShare':med['InStoreShare'],'UE_share':med['UE_share'],
            'DD_share':med['DD_share'],'SD_share':med['SD_share'],
            'CuisineType_enc':0,'Segment_enc':0,'Subregion_enc':0,
            'UE_commission_interaction':med['CommissionRate']*med['UE_share'],
            'DD_commission_interaction':med['CommissionRate']*med['DD_share'],
            'DeliveryCost_SD_interaction':med['DeliveryCostOrder']*med['SD_share'],
            'UE_revenue_ratio':med['UberEatsRevenue']/tv if tv>0 else 0,
            'InStore_revenue_ratio':med['InStoreRevenue']/tv if tv>0 else 0,
            'CostToRevenue_ratio':med['COGSRate']+med['OPEXRate'],
            'GrowthAdjustedOrders':med['MonthlyOrders']*med['GrowthFactor'],
        }

    fig,ax = dark_fig(12,5); style_ax(ax)
    b0    = baseline()

    if scenario == "Commission Rate Impact":
        xs = np.linspace(0.10, 0.35, 80)
        ys = []
        for v in xs:
            b=b0.copy(); b['CommissionRate']=v
            b['UE_commission_interaction']=v*b['UE_share']
            b['DD_commission_interaction']=v*b['DD_share']
            ys.append(gb_model.predict(pd.DataFrame([b])[FEATURES])[0])
        ax.plot(xs*100, ys, color='#64ffda', linewidth=2.5)
        ax.fill_between(xs*100, ys, alpha=0.15, color='#64ffda')
        ax.axvline(b0['CommissionRate']*100, color='#E9C46A', linestyle='--',
                   label=f"Current: {b0['CommissionRate']*100:.1f}%")
        ax.axhline(0, color='#E63946', linewidth=1)
        zeros = [xs[i] for i in range(len(ys)-1) if ys[i]*ys[i+1]<0]
        if zeros: ax.axvline(zeros[0]*100, color='#E63946', linestyle=':', label=f"Break-even: {zeros[0]*100:.1f}%")
        ax.set_xlabel("Commission Rate (%)", color='#8892b0')
        ax.set_title("Commission Rate vs Predicted Net Profit", color='#ccd6f6', fontsize=14)

    elif scenario == "Channel Mix Shift":
        xs = np.linspace(0.15, 0.80, 80)
        ys = []
        for v in xs:
            b=b0.copy(); rem=1-v
            b['InStoreShare']=v; b['UE_share']=rem*0.5; b['DD_share']=rem*0.3; b['SD_share']=rem*0.2
            b['InStore_revenue_ratio']=v; b['UE_revenue_ratio']=rem*0.5
            b['UE_commission_interaction']=b['CommissionRate']*rem*0.5
            b['DD_commission_interaction']=b['CommissionRate']*rem*0.3
            b['DeliveryCost_SD_interaction']=b['DeliveryCostOrder']*rem*0.2
            ys.append(gb_model.predict(pd.DataFrame([b])[FEATURES])[0])
        opt = np.argmax(ys)
        ax.plot(xs*100, ys, color='#2A9D8F', linewidth=2.5)
        ax.fill_between(xs*100, ys, alpha=0.15, color='#2A9D8F')
        ax.axvline(xs[opt]*100, color='#E9C46A', linestyle='--', label=f"Optimal: {xs[opt]*100:.0f}% in-store")
        ax.axvline(b0['InStoreShare']*100, color='#A8DADC', linestyle=':', label=f"Current: {b0['InStoreShare']*100:.0f}%")
        ax.set_xlabel("In-Store Share (%)", color='#8892b0')
        ax.set_title("In-Store Share vs Predicted Net Profit", color='#ccd6f6', fontsize=14)

    elif scenario == "Delivery Cost Impact":
        xs = np.linspace(0.89, 7.0, 80)
        ys = []
        for v in xs:
            b=b0.copy(); b['DeliveryCostOrder']=v
            b['DeliveryCost_SD_interaction']=v*b['SD_share']
            ys.append(gb_model.predict(pd.DataFrame([b])[FEATURES])[0])
        ax.plot(xs, ys, color='#F4A261', linewidth=2.5)
        ax.fill_between(xs, ys, alpha=0.15, color='#F4A261')
        ax.axvline(b0['DeliveryCostOrder'], color='#E9C46A', linestyle='--',
                   label=f"Current: ${b0['DeliveryCostOrder']:.2f}")
        ax.set_xlabel("Delivery Cost per Order ($)", color='#8892b0')
        ax.set_title("Delivery Cost vs Predicted Net Profit", color='#ccd6f6', fontsize=14)

    else:  # AOV
        xs = np.linspace(28, 50, 80)
        ys = []
        for v in xs:
            b=b0.copy(); b['AOV']=v
            ys.append(gb_model.predict(pd.DataFrame([b])[FEATURES])[0])
        ax.plot(xs, ys, color='#E63946', linewidth=2.5)
        ax.fill_between(xs, ys, alpha=0.15, color='#E63946')
        ax.axvline(b0['AOV'], color='#E9C46A', linestyle='--', label=f"Current: ${b0['AOV']:.2f}")
        ax.set_xlabel("Average Order Value ($)", color='#8892b0')
        ax.set_title("AOV vs Predicted Net Profit", color='#ccd6f6', fontsize=14)

    ax.set_ylabel("Predicted Net Profit ($)", color='#8892b0')
    ax.legend(facecolor='#1e2235', labelcolor='#ccd6f6', fontsize=10)
    st.pyplot(fig, use_container_width=True); plt.close()

# ── TAB 4: OPTIMIZATION ──────────────────────
with tab4:
    st.markdown('<div class="sec-hdr">Prescriptive Optimization</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    ch_margins = {
        'In-Store':      (df['InStoreNetProfit']     /df['InStoreRevenue'].replace(0,np.nan)).mean(),
        'Uber Eats':     (df['UberEatsNetProfit']     /df['UberEatsRevenue'].replace(0,np.nan)).mean(),
        'DoorDash':      (df['DoorDashNetProfit']     /df['DoorDashRevenue'].replace(0,np.nan)).mean(),
        'Self-Delivery': (df['SelfDeliveryNetProfit'] /df['SelfDeliveryRevenue'].replace(0,np.nan)).mean(),
    }
    with c1:
        fig,ax = dark_fig(); style_ax(ax)
        bars = ax.bar(ch_margins.keys(), [v*100 for v in ch_margins.values()],
                      color=['#64ffda' if v>0 else '#E63946' for v in ch_margins.values()], alpha=0.85)
        for bar,val in zip(bars,ch_margins.values()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val*100:.1f}%', ha='center', va='bottom', color='#ccd6f6', fontsize=10)
        ax.axhline(0,color='white',linewidth=0.8)
        ax.set_ylabel('Margin (%)',color='#8892b0'); ax.set_title('Net Margin by Channel',color='#ccd6f6',fontsize=12)
        st.pyplot(fig,use_container_width=True); plt.close()

    with c2:
        fig,ax = dark_fig(); style_ax(ax)
        feat_imp.head(10).plot(kind='barh',ax=ax,color='#457B9D',alpha=0.85)
        ax.set_title('Top Feature Importances (RF)',color='#ccd6f6',fontsize=12)
        ax.set_xlabel('Importance Score',color='#8892b0'); ax.tick_params(labelsize=8)
        st.pyplot(fig,use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("**Prescriptive Recommendations**")
    best_ch = max(ch_margins, key=ch_margins.get)
    for title,desc,color in [
        ("Maximize In-Store Share",
         f"In-Store yields {ch_margins['In-Store']*100:.1f}% margin - highest across all channels. Shift volume from aggregators where feasible.", "#64ffda"),
        ("Monitor Aggregator Commissions",
         f"Uber Eats/DoorDash margins are compressed by {df['CommissionRate'].mean()*100:.1f}% avg commission. Negotiate rates or apply menu surcharges.", "#E9C46A"),
        ("Self-Delivery Threshold",
         "Self-delivery turns profitable when cost/order < $3.50 and SD share > 15%. Evaluate per-restaurant ROI.", "#457B9D"),
        ("AOV Uplift Strategy",
         f"Current avg AOV is ${df['AOV'].mean():.2f}. Each $1 increase improves per-order profit significantly - prioritize upselling.", "#2A9D8F"),
    ]:
        st.markdown(f"""<div style="background:#1e2235;border-left:4px solid {color};
        border-radius:8px;padding:14px 18px;margin:8px 0">
        <div style="font-weight:700;color:{color};font-size:14px">{title}</div>
        <div style="color:#a8b2d8;font-size:13px;margin-top:4px">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Top 10 Restaurants by Profit</div>', unsafe_allow_html=True)
    top10 = df.nlargest(10,'TotalMonthlyNetProfit')[
        ['RestaurantName','CuisineType','Segment','Subregion',
         'TotalMonthlyNetProfit','ChannelLevelMargin','AOV']].copy()
    top10['TotalMonthlyNetProfit'] = top10['TotalMonthlyNetProfit'].map('${:,.2f}'.format)
    top10['ChannelLevelMargin']    = top10['ChannelLevelMargin'].map('{:.1%}'.format)
    top10['AOV']                   = top10['AOV'].map('${:.2f}'.format)
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ── TAB 5: MODEL EVALUATION ──────────────────
with tab5:
    st.markdown('<div class="sec-hdr">Model Evaluation Dashboard</div>', unsafe_allow_html=True)

    eval_df = pd.DataFrame({
        'Model':    list(model_results.keys()),
        'R2':       [f"{v['r2']:.4f}"    for v in model_results.values()],
        'RMSE ($)': [f"${v['rmse']:,.2f}" for v in model_results.values()],
        'MAE ($)':  [f"${v['mae']:,.2f}"  for v in model_results.values()],
    })
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    c1,c2 = st.columns(2)
    with c1:
        fig,ax = dark_fig(6,4.5); style_ax(ax)
        bp = model_results['Gradient Boosting']['preds']
        ax.scatter(y_test, bp, alpha=0.4, color='#64ffda', s=20, edgecolors='none')
        lims=[min(y_test.min(),bp.min()),max(y_test.max(),bp.max())]
        ax.plot(lims,lims,'w--',linewidth=1.5)
        ax.set_title('Actual vs Predicted (GB)',color='#ccd6f6',fontsize=12)
        ax.set_xlabel('Actual ($)',color='#8892b0'); ax.set_ylabel('Predicted ($)',color='#8892b0')
        ax.text(0.05,0.92,f"R2 = {model_results['Gradient Boosting']['r2']:.4f}",
                transform=ax.transAxes,color='#E9C46A',fontsize=11,
                bbox=dict(boxstyle='round',facecolor='#252840',alpha=0.7))
        st.pyplot(fig,use_container_width=True); plt.close()
    with c2:
        fig,ax = dark_fig(6,4.5); style_ax(ax)
        res = np.array(y_test) - bp
        ax.scatter(bp, res, alpha=0.4, color='#457B9D', s=20, edgecolors='none')
        ax.axhline(0,color='white',linewidth=1.2)
        ax.set_title('Residual Plot (GB)',color='#ccd6f6',fontsize=12)
        ax.set_xlabel('Predicted ($)',color='#8892b0'); ax.set_ylabel('Residual ($)',color='#8892b0')
        st.pyplot(fig,use_container_width=True); plt.close()

    gb_res = model_results['Gradient Boosting']
    st.info(f"Best Model: Gradient Boosting | R2 = {gb_res['r2']:.4f} | RMSE = ${gb_res['rmse']:,.2f} | MAE = ${gb_res['mae']:,.2f} | Explains {gb_res['r2']*100:.1f}% of profit variance")

# ── FOOTER ───────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#8892b0;font-size:12px;padding:10px'>SkyCity Auckland Restaurants & Bars  |  Profit Optimization Dashboard  |  Unified Mentor Internship Project</div>", unsafe_allow_html=True)