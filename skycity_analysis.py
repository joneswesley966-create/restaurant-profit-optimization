"""
SkyCity Auckland Restaurants & Bars
Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations
Author: Jones (Intern)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. STYLE
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653', '#A8DADC']
sns.set_palette(PALETTE)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('/home/claude/SkyCity_Auckland_Restaurants_Bars.csv')
print("=" * 60)
print("SKYCITY AUCKLAND — PROFIT OPTIMIZATION ANALYSIS")
print("=" * 60)
print(f"\nDataset: {df.shape[0]} restaurant-months × {df.shape[1]} features")
print("\nColumn List:")
for c in df.columns:
    print(f"  • {c}")

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n── DESCRIPTIVE STATISTICS ──")
print(df[['TotalMonthlyNetProfit','NetProfitPerOrder','ChannelLevelMargin',
          'AOV','MonthlyOrders','COGSRate','OPEXRate','CommissionRate']].describe().round(3))

fig = plt.figure(figsize=(20, 22))
fig.suptitle("SkyCity Auckland — Exploratory Data Analysis", fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# 2a. Profit distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['TotalMonthlyNetProfit'], bins=30, color=PALETTE[0], edgecolor='white', alpha=0.85)
ax1.set_title('Total Monthly Net Profit Distribution', fontsize=11)
ax1.set_xlabel('Net Profit ($)')
ax1.set_ylabel('Frequency')
ax1.axvline(df['TotalMonthlyNetProfit'].mean(), color='black', linestyle='--', label=f"Mean: ${df['TotalMonthlyNetProfit'].mean():,.0f}")
ax1.legend(fontsize=9)

# 2b. Profit by Cuisine
ax2 = fig.add_subplot(gs[0, 1])
cuisine_profit = df.groupby('CuisineType')['TotalMonthlyNetProfit'].mean().sort_values()
cuisine_profit.plot(kind='barh', ax=ax2, color=PALETTE[1])
ax2.set_title('Avg Monthly Profit by Cuisine Type', fontsize=11)
ax2.set_xlabel('Avg Net Profit ($)')

# 2c. Profit by Segment
ax3 = fig.add_subplot(gs[0, 2])
seg_profit = df.groupby('Segment')['TotalMonthlyNetProfit'].mean().sort_values()
seg_profit.plot(kind='bar', ax=ax3, color=PALETTE[2], rot=15)
ax3.set_title('Avg Profit by Segment', fontsize=11)
ax3.set_ylabel('Avg Net Profit ($)')

# 2d. Channel Revenue Mix (pie)
ax4 = fig.add_subplot(gs[1, 0])
rev_cols = ['InStoreRevenue','UberEatsRevenue','DoorDashRevenue','SelfDeliveryRevenue']
rev_totals = df[rev_cols].sum()
rev_totals.index = ['In-Store','Uber Eats','DoorDash','Self-Delivery']
ax4.pie(rev_totals, labels=rev_totals.index, autopct='%1.1f%%',
        colors=PALETTE[:4], startangle=140, pctdistance=0.82)
ax4.set_title('Total Revenue by Channel', fontsize=11)

# 2e. Channel Profit contribution
ax5 = fig.add_subplot(gs[1, 1])
prof_cols = ['InStoreNetProfit','UberEatsNetProfit','DoorDashNetProfit','SelfDeliveryNetProfit']
prof_totals = df[prof_cols].sum()
prof_totals.index = ['In-Store','Uber Eats','DoorDash','Self-Delivery']
colors_p = [PALETTE[2] if v > 0 else PALETTE[0] for v in prof_totals]
prof_totals.plot(kind='bar', ax=ax5, color=colors_p, rot=15)
ax5.set_title('Total Net Profit by Channel', fontsize=11)
ax5.set_ylabel('Net Profit ($)')
ax5.axhline(0, color='black', linewidth=0.8)

# 2f. Commission Rate vs Profit
ax6 = fig.add_subplot(gs[1, 2])
sc = ax6.scatter(df['CommissionRate'], df['TotalMonthlyNetProfit'],
                 c=df['UE_share'], cmap='RdYlGn_r', alpha=0.6, s=30)
plt.colorbar(sc, ax=ax6, label='UE Share')
ax6.set_title('Commission Rate vs Net Profit', fontsize=11)
ax6.set_xlabel('Commission Rate')
ax6.set_ylabel('Net Profit ($)')

# 2g. AOV vs Profit
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(df['AOV'], df['TotalMonthlyNetProfit'], alpha=0.5, color=PALETTE[4], s=30)
m, b = np.polyfit(df['AOV'], df['TotalMonthlyNetProfit'], 1)
x_line = np.linspace(df['AOV'].min(), df['AOV'].max(), 100)
ax7.plot(x_line, m*x_line+b, color='black', linewidth=1.5)
ax7.set_title('AOV vs Net Profit', fontsize=11)
ax7.set_xlabel('Average Order Value ($)')
ax7.set_ylabel('Net Profit ($)')

# 2h. Monthly Orders vs Profit
ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(df['MonthlyOrders'], df['TotalMonthlyNetProfit'], alpha=0.5, color=PALETTE[5], s=30)
ax8.set_title('Monthly Orders vs Net Profit', fontsize=11)
ax8.set_xlabel('Monthly Orders')
ax8.set_ylabel('Net Profit ($)')

# 2i. Correlation heatmap
ax9 = fig.add_subplot(gs[2, 2])
num_cols = ['TotalMonthlyNetProfit','AOV','MonthlyOrders','CommissionRate',
            'COGSRate','OPEXRate','InStoreShare','UE_share','DD_share','SD_share','GrowthFactor']
corr = df[num_cols].corr()
sns.heatmap(corr[['TotalMonthlyNetProfit']].sort_values('TotalMonthlyNetProfit'),
            annot=True, fmt='.2f', cmap='RdYlGn', ax=ax9, vmin=-1, vmax=1, linewidths=0.5)
ax9.set_title('Feature Correlation with Net Profit', fontsize=11)

# 2j. Profit by Subregion
ax10 = fig.add_subplot(gs[3, 0])
sub_profit = df.groupby('Subregion')['TotalMonthlyNetProfit'].mean().sort_values()
sub_profit.plot(kind='barh', ax=ax10, color=PALETTE[6])
ax10.set_title('Avg Profit by Subregion', fontsize=11)

# 2k. Net Profit per Order by Segment
ax11 = fig.add_subplot(gs[3, 1])
bp_data = [df[df['Segment']==s]['NetProfitPerOrder'].values for s in df['Segment'].unique()]
ax11.boxplot(bp_data, labels=df['Segment'].unique(), patch_artist=True,
             boxprops=dict(facecolor=PALETTE[1], alpha=0.7))
ax11.set_title('Net Profit/Order by Segment', fontsize=11)
ax11.set_ylabel('Net Profit per Order ($)')
ax11.tick_params(axis='x', rotation=15)

# 2l. Channel Mix efficiency
ax12 = fig.add_subplot(gs[3, 2])
channel_margin = {
    'In-Store': (df['InStoreNetProfit'] / df['InStoreRevenue'].replace(0, np.nan)).mean(),
    'Uber Eats': (df['UberEatsNetProfit'] / df['UberEatsRevenue'].replace(0, np.nan)).mean(),
    'DoorDash': (df['DoorDashNetProfit'] / df['DoorDashRevenue'].replace(0, np.nan)).mean(),
    'Self-Delivery': (df['SelfDeliveryNetProfit'] / df['SelfDeliveryRevenue'].replace(0, np.nan)).mean(),
}
ax12.bar(channel_margin.keys(), channel_margin.values(),
         color=[PALETTE[2] if v > 0 else PALETTE[0] for v in channel_margin.values()])
ax12.set_title('Avg Margin by Channel', fontsize=11)
ax12.set_ylabel('Margin')
ax12.axhline(0, color='black', linewidth=0.8)
ax12.tick_params(axis='x', rotation=10)

plt.savefig('/home/claude/eda_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✓ EDA chart saved")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df_ml = df.copy()
le = LabelEncoder()
for col in ['CuisineType', 'Segment', 'Subregion']:
    df_ml[col+'_enc'] = le.fit_transform(df_ml[col])

# Engineered features per project spec
df_ml['UE_commission_interaction'] = df_ml['CommissionRate'] * df_ml['UE_share']
df_ml['DD_commission_interaction'] = df_ml['CommissionRate'] * df_ml['DD_share']
df_ml['DeliveryCost_SD_interaction'] = df_ml['DeliveryCostOrder'] * df_ml['SD_share']
df_ml['UE_revenue_ratio'] = df_ml['UberEatsRevenue'] / df_ml['TotalRevenue']
df_ml['InStore_revenue_ratio'] = df_ml['InStoreRevenue'] / df_ml['TotalRevenue']
df_ml['CostToRevenue_ratio'] = (df_ml['COGSRate'] + df_ml['OPEXRate'])
df_ml['GrowthAdjustedOrders'] = df_ml['MonthlyOrders'] * df_ml['GrowthFactor']

# ─────────────────────────────────────────────
# 4. MODEL DEVELOPMENT
# ─────────────────────────────────────────────
features = [
    'AOV', 'MonthlyOrders', 'GrowthFactor', 'COGSRate', 'OPEXRate', 'CommissionRate',
    'DeliveryRadiusKM', 'DeliveryCostOrder', 'InStoreShare', 'UE_share', 'DD_share', 'SD_share',
    'CuisineType_enc', 'Segment_enc', 'Subregion_enc',
    'UE_commission_interaction', 'DD_commission_interaction',
    'DeliveryCost_SD_interaction', 'UE_revenue_ratio', 'InStore_revenue_ratio',
    'CostToRevenue_ratio', 'GrowthAdjustedOrders'
]

X = df_ml[features]
y = df_ml['TotalMonthlyNetProfit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
}

results = {}
print("\n── MODEL EVALUATION ──")
print(f"{'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
print("-" * 52)

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"{name:<22} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]
print(f"\n✓ Best model: {best_name} (R² = {best['r2']:.4f})")

# Feature importance
rf_model = results['Random Forest']['model']
fi = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 5. MODEL VISUALIZATION
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(20, 12))
fig2.suptitle("SkyCity Auckland — Model Evaluation & Insights", fontsize=16, fontweight='bold')

# Actual vs Predicted for best model
ax = axes[0, 0]
ax.scatter(y_test, best['preds'], alpha=0.6, color=PALETTE[0], s=40)
lims = [min(y_test.min(), best['preds'].min()), max(y_test.max(), best['preds'].max())]
ax.plot(lims, lims, 'k--', linewidth=1.5)
ax.set_title(f'{best_name}: Actual vs Predicted', fontsize=11)
ax.set_xlabel('Actual Net Profit ($)')
ax.set_ylabel('Predicted Net Profit ($)')
ax.text(0.05, 0.92, f"R² = {best['r2']:.4f}", transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Residuals
ax = axes[0, 1]
residuals = y_test - best['preds']
ax.scatter(best['preds'], residuals, alpha=0.5, color=PALETTE[1], s=30)
ax.axhline(0, color='black', linewidth=1.2)
ax.set_title(f'{best_name}: Residual Plot', fontsize=11)
ax.set_xlabel('Predicted ($)')
ax.set_ylabel('Residual ($)')

# Model comparison bar
ax = axes[0, 2]
model_names = list(results.keys())
r2_scores = [results[m]['r2'] for m in model_names]
rmse_scores = [results[m]['rmse'] for m in model_names]
x_pos = np.arange(len(model_names))
bars = ax.bar(x_pos, r2_scores, color=PALETTE[:3], alpha=0.8, width=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(['LR', 'RF', 'GB'], fontsize=10)
ax.set_title('Model R² Comparison', fontsize=11)
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1.1)
for bar, score in zip(bars, r2_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.4f}', ha='center', va='bottom', fontsize=9)

# Feature importance (top 12)
ax = axes[1, 0]
fi.head(12).plot(kind='barh', ax=ax, color=PALETTE[3])
ax.set_title('Top 12 Feature Importances (RF)', fontsize=11)
ax.set_xlabel('Importance Score')

# Scenario: Commission Rate Impact
ax = axes[1, 1]
comm_rates = np.linspace(0.10, 0.35, 50)
sample = X_test.copy().iloc[[0]]
profits_comm = []
for cr in comm_rates:
    s_mod = sample.copy()
    s_mod['CommissionRate'] = cr
    s_mod['UE_commission_interaction'] = cr * s_mod['UE_share']
    s_mod['DD_commission_interaction'] = cr * s_mod['DD_share']
    profits_comm.append(rf_model.predict(s_mod)[0])
ax.plot(comm_rates * 100, profits_comm, color=PALETTE[0], linewidth=2.5)
ax.fill_between(comm_rates * 100, profits_comm, alpha=0.15, color=PALETTE[0])
ax.set_title('Scenario: Commission Rate → Profit', fontsize=11)
ax.set_xlabel('Commission Rate (%)')
ax.set_ylabel('Predicted Net Profit ($)')
ax.axvline(sample['CommissionRate'].values[0]*100, color='black', linestyle='--', label='Current')
ax.legend()

# Scenario: Channel Mix Shift
ax = axes[1, 2]
instore_shares = np.linspace(0.20, 0.80, 50)
profits_mix = []
for ins in instore_shares:
    s_mod = sample.copy()
    remaining = 1 - ins
    s_mod['InStoreShare'] = ins
    s_mod['UE_share'] = remaining * 0.5
    s_mod['DD_share'] = remaining * 0.3
    s_mod['SD_share'] = remaining * 0.2
    s_mod['InStore_revenue_ratio'] = ins
    s_mod['UE_revenue_ratio'] = remaining * 0.5
    s_mod['UE_commission_interaction'] = s_mod['CommissionRate'] * s_mod['UE_share']
    s_mod['DD_commission_interaction'] = s_mod['CommissionRate'] * s_mod['DD_share']
    s_mod['DeliveryCost_SD_interaction'] = s_mod['DeliveryCostOrder'] * s_mod['SD_share']
    profits_mix.append(rf_model.predict(s_mod)[0])
ax.plot(instore_shares * 100, profits_mix, color=PALETTE[2], linewidth=2.5)
ax.fill_between(instore_shares * 100, profits_mix, alpha=0.15, color=PALETTE[2])
optimal_idx = np.argmax(profits_mix)
ax.axvline(instore_shares[optimal_idx]*100, color='red', linestyle='--',
           label=f'Optimal: {instore_shares[optimal_idx]*100:.0f}%')
ax.set_title('Scenario: In-Store Share → Profit', fontsize=11)
ax.set_xlabel('In-Store Share (%)')
ax.set_ylabel('Predicted Net Profit ($)')
ax.legend()

plt.tight_layout()
plt.savefig('/home/claude/model_evaluation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Model evaluation chart saved")

# ─────────────────────────────────────────────
# 6. KPI SUMMARY
# ─────────────────────────────────────────────
print("\n── KEY PERFORMANCE INDICATORS ──")
print(f"  Predicted Net Profit (avg):      ${df['TotalMonthlyNetProfit'].mean():>10,.2f}")
print(f"  Profit Sensitivity Index (std):  ${df['TotalMonthlyNetProfit'].std():>10,.2f}")
print(f"  Channel Mix Efficiency (avg):    {df['ChannelLevelMargin'].mean()*100:>9.2f}%")
print(f"  Break-Even Comm Rate (approx):   {(df[df['TotalMonthlyNetProfit']<0]['CommissionRate'].mean())*100:>9.2f}%")
print(f"  Best Model R²:                   {best['r2']:>10.4f}")
print(f"  Best Model RMSE:                 ${best['rmse']:>10.2f}")
print(f"\n  Top profit driver:               {fi.index[0]}")
print(f"  Second driver:                   {fi.index[1]}")
print(f"  Third driver:                    {fi.index[2]}")

# Optimal channel mix recommendation
ch_margins = {
    'In-Store': (df['InStoreNetProfit'] / df['InStoreRevenue'].replace(0, np.nan)).mean(),
    'Uber Eats': (df['UberEatsNetProfit'] / df['UberEatsRevenue'].replace(0, np.nan)).mean(),
    'DoorDash': (df['DoorDashNetProfit'] / df['DoorDashRevenue'].replace(0, np.nan)).mean(),
    'Self-Delivery': (df['SelfDeliveryNetProfit'] / df['SelfDeliveryRevenue'].replace(0, np.nan)).mean(),
}
best_ch = max(ch_margins, key=ch_margins.get)
print(f"\n  Most profitable channel:         {best_ch} ({ch_margins[best_ch]*100:.1f}% margin)")

print("\n✓ Analysis complete. Ready for Streamlit dashboard.")

# Save model artifacts for streamlit
import pickle
with open('/home/claude/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('/home/claude/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

feature_meta = {
    'features': features,
    'ch_margins': ch_margins,
    'fi': fi.to_dict(),
    'model_results': {k: {'r2': v['r2'], 'rmse': v['rmse'], 'mae': v['mae']} for k, v in results.items()},
    'kpis': {
        'avg_profit': df['TotalMonthlyNetProfit'].mean(),
        'std_profit': df['TotalMonthlyNetProfit'].std(),
        'avg_margin': df['ChannelLevelMargin'].mean(),
        'avg_npo': df['NetProfitPerOrder'].mean(),
    }
}
import json
with open('/home/claude/feature_meta.json', 'w') as f:
    json.dump(feature_meta, f)

df.to_csv('/home/claude/SkyCity_Auckland_Restaurants_Bars.csv', index=False)
print("✓ Model artifacts saved")














