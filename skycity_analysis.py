"""
SkyCity Auckland Restaurants & Bars
Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations
Author: Jones (Intern)
Dataset: 1,696 real restaurant-month records
"""

import os, glob, pickle, json, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

# =============================================================
# 0. CONFIGURATION  — auto-finds CSV in 5 locations
# =============================================================
CSV_NAME   = 'SkyCity Auckland Restaurants & Bars.csv'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = SCRIPT_DIR
HOME       = os.path.expanduser('~')

SEARCH_PATHS = [
    os.path.join(SCRIPT_DIR,         CSV_NAME),
    os.path.join(os.getcwd(),        CSV_NAME),
    os.path.join(HOME, 'Desktop',    CSV_NAME),
    os.path.join(HOME, 'Downloads',  CSV_NAME),
    os.path.join(HOME, 'Documents',  CSV_NAME),
] + glob.glob(os.path.join(SCRIPT_DIR,  '*', CSV_NAME)) \
  + glob.glob(os.path.join(os.getcwd(), '*', CSV_NAME))

CSV_PATH = next((p for p in SEARCH_PATHS if os.path.exists(p)), None)

plt.style.use('seaborn-v0_8-darkgrid')
PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653', '#A8DADC']
sns.set_palette(PALETTE)

# =============================================================
# 1. LOAD & PREPARE DATA
# =============================================================
if CSV_PATH is None:
    searched = '\n'.join(f'  - {p}' for p in SEARCH_PATHS[:5])
    raise FileNotFoundError(
        f"\nCould not find '{CSV_NAME}' in:\n{searched}\n\n"
        f"Copy '{CSV_NAME}' into:\n  {SCRIPT_DIR}\n"
    )

df = pd.read_csv(CSV_PATH)

# Rename real CSV columns to match expected names
df.rename(columns={
    'InStoreOrders':        'InStoreOrdersCount',
    'UberEatsOrders':       'UberEatsOrdersCount',
    'DoorDashOrders':       'DoorDashOrdersCount',
    'SelfDeliveryOrders':   'SelfDeliveryOrdersCount',
    'DeliveryCostPerOrder': 'DeliveryCostOrder',
}, inplace=True)

# Derive calculated columns
df['TotalRevenue']          = (df['InStoreRevenue'] + df['UberEatsRevenue'] +
                               df['DoorDashRevenue'] + df['SelfDeliveryRevenue'])
df['TotalMonthlyNetProfit'] = (df['InStoreNetProfit'] + df['UberEatsNetProfit'] +
                               df['DoorDashNetProfit'] + df['SelfDeliveryNetProfit'])
df['NetProfitPerOrder']     = df['TotalMonthlyNetProfit'] / df['MonthlyOrders']
df['ChannelLevelMargin']    = df['TotalMonthlyNetProfit'] / df['TotalRevenue']

print("=" * 60)
print("SKYCITY AUCKLAND - PROFIT OPTIMIZATION ANALYSIS")
print("=" * 60)
print(f"\nLoaded: {CSV_PATH}")
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# =============================================================
# 2. EDA
# =============================================================
print("\n-- DESCRIPTIVE STATISTICS --")
print(df[['TotalMonthlyNetProfit', 'NetProfitPerOrder', 'ChannelLevelMargin',
          'AOV', 'MonthlyOrders', 'COGSRate', 'OPEXRate', 'CommissionRate']].describe().round(3))

fig = plt.figure(figsize=(20, 22))
fig.suptitle("SkyCity Auckland - Exploratory Data Analysis", fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['TotalMonthlyNetProfit'], bins=40, color=PALETTE[0], edgecolor='white', alpha=0.85)
ax1.axvline(df['TotalMonthlyNetProfit'].mean(), color='black', linestyle='--',
            label=f"Mean: ${df['TotalMonthlyNetProfit'].mean():,.0f}")
ax1.set_title('Net Profit Distribution', fontsize=11)
ax1.set_xlabel('Net Profit ($)'); ax1.set_ylabel('Frequency'); ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
df.groupby('CuisineType')['TotalMonthlyNetProfit'].mean().sort_values().plot(kind='barh', ax=ax2, color=PALETTE[1])
ax2.set_title('Avg Profit by Cuisine Type', fontsize=11); ax2.set_xlabel('Avg Net Profit ($)')

ax3 = fig.add_subplot(gs[0, 2])
df.groupby('Segment')['TotalMonthlyNetProfit'].mean().sort_values().plot(kind='bar', ax=ax3, color=PALETTE[2], rot=15)
ax3.set_title('Avg Profit by Segment', fontsize=11); ax3.set_ylabel('Avg Net Profit ($)')

ax4 = fig.add_subplot(gs[1, 0])
rev_totals = df[['InStoreRevenue','UberEatsRevenue','DoorDashRevenue','SelfDeliveryRevenue']].sum()
rev_totals.index = ['In-Store', 'Uber Eats', 'DoorDash', 'Self-Delivery']
ax4.pie(rev_totals, labels=rev_totals.index, autopct='%1.1f%%',
        colors=PALETTE[:4], startangle=140, pctdistance=0.82)
ax4.set_title('Revenue Mix by Channel', fontsize=11)

ax5 = fig.add_subplot(gs[1, 1])
prof_totals = df[['InStoreNetProfit','UberEatsNetProfit','DoorDashNetProfit','SelfDeliveryNetProfit']].sum()
prof_totals.index = ['In-Store', 'Uber Eats', 'DoorDash', 'Self-Delivery']
prof_totals.plot(kind='bar', ax=ax5, color=[PALETTE[2] if v>0 else PALETTE[0] for v in prof_totals.values], rot=15)
ax5.axhline(0, color='black', linewidth=0.8)
ax5.set_title('Total Net Profit by Channel', fontsize=11); ax5.set_ylabel('Net Profit ($)')

ax6 = fig.add_subplot(gs[1, 2])
sc = ax6.scatter(df['CommissionRate']*100, df['TotalMonthlyNetProfit'],
                 c=df['UE_share'], cmap='RdYlGn_r', alpha=0.4, s=15)
plt.colorbar(sc, ax=ax6, label='UE Share')
ax6.set_title('Commission Rate vs Net Profit', fontsize=11)
ax6.set_xlabel('Commission Rate (%)'); ax6.set_ylabel('Net Profit ($)')

ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(df['AOV'], df['TotalMonthlyNetProfit'], alpha=0.3, color=PALETTE[4], s=15)
m, b = np.polyfit(df['AOV'], df['TotalMonthlyNetProfit'], 1)
x_line = np.linspace(df['AOV'].min(), df['AOV'].max(), 100)
ax7.plot(x_line, m*x_line+b, color='black', linewidth=1.8)
ax7.set_title('AOV vs Net Profit', fontsize=11)
ax7.set_xlabel('Average Order Value ($)'); ax7.set_ylabel('Net Profit ($)')

ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(df['MonthlyOrders'], df['TotalMonthlyNetProfit'], alpha=0.3, color=PALETTE[5], s=15)
ax8.set_title('Monthly Orders vs Net Profit', fontsize=11)
ax8.set_xlabel('Monthly Orders'); ax8.set_ylabel('Net Profit ($)')

ax9 = fig.add_subplot(gs[2, 2])
num_cols = ['TotalMonthlyNetProfit','AOV','MonthlyOrders','CommissionRate',
            'COGSRate','OPEXRate','InStoreShare','UE_share','DD_share','SD_share','GrowthFactor']
corr = df[num_cols].corr()
sns.heatmap(corr[['TotalMonthlyNetProfit']].sort_values('TotalMonthlyNetProfit'),
            annot=True, fmt='.2f', cmap='RdYlGn', ax=ax9, vmin=-1, vmax=1, linewidths=0.5)
ax9.set_title('Correlation with Net Profit', fontsize=11)

ax10 = fig.add_subplot(gs[3, 0])
df.groupby('Subregion')['TotalMonthlyNetProfit'].mean().sort_values().plot(kind='barh', ax=ax10, color=PALETTE[6])
ax10.set_title('Avg Profit by Subregion', fontsize=11)

ax11 = fig.add_subplot(gs[3, 1])
seg_labels = sorted(df['Segment'].unique())
bp_data = [df[df['Segment']==s]['NetProfitPerOrder'].values for s in seg_labels]
ax11.boxplot(bp_data, labels=seg_labels, patch_artist=True,
             boxprops=dict(facecolor=PALETTE[1], alpha=0.7))
ax11.set_title('Net Profit/Order by Segment', fontsize=11)
ax11.set_ylabel('Net Profit per Order ($)'); ax11.tick_params(axis='x', rotation=15)

ax12 = fig.add_subplot(gs[3, 2])
ch_margin = {
    'In-Store':      (df['InStoreNetProfit']     / df['InStoreRevenue'].replace(0,np.nan)).mean(),
    'Uber Eats':     (df['UberEatsNetProfit']     / df['UberEatsRevenue'].replace(0,np.nan)).mean(),
    'DoorDash':      (df['DoorDashNetProfit']     / df['DoorDashRevenue'].replace(0,np.nan)).mean(),
    'Self-Delivery': (df['SelfDeliveryNetProfit'] / df['SelfDeliveryRevenue'].replace(0,np.nan)).mean(),
}
ax12.bar(ch_margin.keys(), [v*100 for v in ch_margin.values()],
         color=[PALETTE[2] if v>0 else PALETTE[0] for v in ch_margin.values()])
ax12.axhline(0, color='black', linewidth=0.8)
ax12.set_title('Avg Net Margin by Channel', fontsize=11); ax12.set_ylabel('Margin (%)')
ax12.tick_params(axis='x', rotation=10)

eda_path = os.path.join(OUT_DIR, 'eda_analysis.png')
plt.savefig(eda_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\nEDA chart saved: {eda_path}")

# =============================================================
# 3. FEATURE ENGINEERING
# =============================================================
df_ml = df.copy()
le = LabelEncoder()
for col in ['CuisineType', 'Segment', 'Subregion']:
    df_ml[col+'_enc'] = le.fit_transform(df_ml[col])

df_ml['UE_commission_interaction']   = df_ml['CommissionRate'] * df_ml['UE_share']
df_ml['DD_commission_interaction']   = df_ml['CommissionRate'] * df_ml['DD_share']
df_ml['DeliveryCost_SD_interaction'] = df_ml['DeliveryCostOrder'] * df_ml['SD_share']
df_ml['UE_revenue_ratio']            = df_ml['UberEatsRevenue']  / df_ml['TotalRevenue']
df_ml['InStore_revenue_ratio']       = df_ml['InStoreRevenue']   / df_ml['TotalRevenue']
df_ml['CostToRevenue_ratio']         = df_ml['COGSRate'] + df_ml['OPEXRate']
df_ml['GrowthAdjustedOrders']        = df_ml['MonthlyOrders'] * df_ml['GrowthFactor']

# =============================================================
# 4. MODEL DEVELOPMENT
# =============================================================
FEATURES = [
    'AOV','MonthlyOrders','GrowthFactor','COGSRate','OPEXRate','CommissionRate',
    'DeliveryRadiusKM','DeliveryCostOrder','InStoreShare','UE_share','DD_share','SD_share',
    'CuisineType_enc','Segment_enc','Subregion_enc',
    'UE_commission_interaction','DD_commission_interaction',
    'DeliveryCost_SD_interaction','UE_revenue_ratio','InStore_revenue_ratio',
    'CostToRevenue_ratio','GrowthAdjustedOrders',
]

X = df_ml[FEATURES]
y = df_ml['TotalMonthlyNetProfit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)
X_test_s   = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=5, random_state=42),
}

results = {}
print("\n-- MODEL EVALUATION --")
print(f"{'Model':<22} {'RMSE':>10} {'MAE':>10} {'R2':>8}")
print("-" * 52)

for name, model in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if name == 'Linear Regression' else (X_train, X_test)
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"{name:<22} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best      = results[best_name]
print(f"\nBest model: {best_name}  (R2 = {best['r2']:.4f})")

rf_model = results['Random Forest']['model']
gb_model = results['Gradient Boosting']['model']
fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

# =============================================================
# 5. MODEL VISUALIZATIONS
# =============================================================
fig2, axes = plt.subplots(2, 3, figsize=(20, 12))
fig2.suptitle("SkyCity Auckland - Model Evaluation & Insights", fontsize=16, fontweight='bold')

ax = axes[0, 0]
ax.scatter(y_test, best['preds'], alpha=0.4, color=PALETTE[0], s=20)
lims = [min(y_test.min(), best['preds'].min()), max(y_test.max(), best['preds'].max())]
ax.plot(lims, lims, 'k--', linewidth=1.5)
ax.set_title(f'{best_name}: Actual vs Predicted', fontsize=11)
ax.set_xlabel('Actual Net Profit ($)'); ax.set_ylabel('Predicted Net Profit ($)')
ax.text(0.05, 0.92, f"R2 = {best['r2']:.4f}", transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axes[0, 1]
residuals = np.array(y_test) - best['preds']
ax.scatter(best['preds'], residuals, alpha=0.3, color=PALETTE[1], s=20)
ax.axhline(0, color='black', linewidth=1.2)
ax.set_title(f'{best_name}: Residuals', fontsize=11)
ax.set_xlabel('Predicted ($)'); ax.set_ylabel('Residual ($)')

ax = axes[0, 2]
r2s  = [results[m]['r2']   for m in results]
bars = ax.bar(range(3), r2s, color=PALETTE[:3], alpha=0.85, width=0.5)
ax.set_xticks(range(3)); ax.set_xticklabels(['LR', 'RF', 'GB'])
ax.set_title('Model R2 Comparison', fontsize=11); ax.set_ylabel('R2'); ax.set_ylim(0, 1.1)
for bar, score in zip(bars, r2s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{score:.4f}', ha='center', va='bottom', fontsize=9)

ax = axes[1, 0]
fi.head(12).plot(kind='barh', ax=ax, color=PALETTE[3])
ax.set_title('Top 12 Feature Importances (RF)', fontsize=11); ax.set_xlabel('Importance')

ax = axes[1, 1]
sample = X_test.copy().iloc[[0]]
comm_rates   = np.linspace(0.10, 0.35, 60)
profits_comm = []
for cr in comm_rates:
    s = sample.copy()
    s['CommissionRate']            = cr
    s['UE_commission_interaction'] = cr * float(s['UE_share'].iloc[0])
    s['DD_commission_interaction'] = cr * float(s['DD_share'].iloc[0])
    profits_comm.append(gb_model.predict(s)[0])
ax.plot(comm_rates*100, profits_comm, color=PALETTE[0], linewidth=2.5)
ax.fill_between(comm_rates*100, profits_comm, alpha=0.15, color=PALETTE[0])
ax.axvline(float(sample['CommissionRate'].iloc[0])*100, color='black', linestyle='--',
           label=f"Current: {float(sample['CommissionRate'].iloc[0])*100:.1f}%")
ax.set_title('Scenario: Commission Rate vs Profit', fontsize=11)
ax.set_xlabel('Commission Rate (%)'); ax.set_ylabel('Predicted Net Profit ($)'); ax.legend()

ax = axes[1, 2]
instore_shares = np.linspace(0.20, 0.80, 60)
profits_mix    = []
for ins in instore_shares:
    s   = sample.copy(); rem = 1 - ins
    s['InStoreShare']              = ins
    s['UE_share']                  = rem * 0.5
    s['DD_share']                  = rem * 0.3
    s['SD_share']                  = rem * 0.2
    s['InStore_revenue_ratio']     = ins
    s['UE_revenue_ratio']          = rem * 0.5
    s['UE_commission_interaction'] = float(s['CommissionRate'].iloc[0]) * rem * 0.5
    s['DD_commission_interaction'] = float(s['CommissionRate'].iloc[0]) * rem * 0.3
    s['DeliveryCost_SD_interaction']= float(s['DeliveryCostOrder'].iloc[0]) * rem * 0.2
    profits_mix.append(gb_model.predict(s)[0])
opt = int(np.argmax(profits_mix))
ax.plot(instore_shares*100, profits_mix, color=PALETTE[2], linewidth=2.5)
ax.fill_between(instore_shares*100, profits_mix, alpha=0.15, color=PALETTE[2])
ax.axvline(instore_shares[opt]*100, color='red', linestyle='--',
           label=f"Optimal: {instore_shares[opt]*100:.0f}%")
ax.set_title('Scenario: In-Store Share vs Profit', fontsize=11)
ax.set_xlabel('In-Store Share (%)'); ax.set_ylabel('Predicted Net Profit ($)'); ax.legend()

plt.tight_layout()
model_path = os.path.join(OUT_DIR, 'model_evaluation.png')
plt.savefig(model_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Model chart saved: {model_path}")

# =============================================================
# 6. KPI SUMMARY
# =============================================================
neg_df        = df[df['TotalMonthlyNetProfit'] < 0]
breakeven_comm = neg_df['CommissionRate'].mean()*100 if len(neg_df) > 0 else float('nan')
best_ch        = max(ch_margin, key=ch_margin.get)

print("\n-- KEY PERFORMANCE INDICATORS --")
print(f"  Avg Monthly Net Profit:        ${df['TotalMonthlyNetProfit'].mean():>10,.2f}")
print(f"  Profit Std Dev (sensitivity):  ${df['TotalMonthlyNetProfit'].std():>10,.2f}")
print(f"  Avg Channel Margin:            {df['ChannelLevelMargin'].mean()*100:>9.2f}%")
print(f"  Break-Even Commission Rate:    {breakeven_comm:>9.2f}%")
print(f"  Best Model R2:                 {best['r2']:>10.4f}")
print(f"  Best Model RMSE:               ${best['rmse']:>10.2f}")
print(f"  Top profit driver:             {fi.index[0]}")
print(f"  Most profitable channel:       {best_ch} ({ch_margin[best_ch]*100:.1f}% margin)")

# =============================================================
# 7. SAVE ARTIFACTS
# =============================================================
with open(os.path.join(OUT_DIR,'rf_model.pkl'),  'wb') as f: pickle.dump(rf_model, f)
with open(os.path.join(OUT_DIR,'gb_model.pkl'),  'wb') as f: pickle.dump(gb_model, f)
with open(os.path.join(OUT_DIR,'scaler.pkl'),    'wb') as f: pickle.dump(scaler,   f)

meta = {
    'features':     FEATURES,
    'ch_margins':   ch_margin,
    'fi':           fi.to_dict(),
    'model_results':{k:{'r2':v['r2'],'rmse':v['rmse'],'mae':v['mae']} for k,v in results.items()},
    'kpis': {
        'avg_profit': df['TotalMonthlyNetProfit'].mean(),
        'std_profit': df['TotalMonthlyNetProfit'].std(),
        'avg_margin': df['ChannelLevelMargin'].mean(),
        'avg_npo':    df['NetProfitPerOrder'].mean(),
    }
}
with open(os.path.join(OUT_DIR,'feature_meta.json'),'w') as f: json.dump(meta, f, indent=2)

print("\nArtifacts saved to:", OUT_DIR)
print("Analysis complete. Ready for Streamlit dashboard.")