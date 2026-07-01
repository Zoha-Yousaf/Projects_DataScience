# ============================================================
# Project : Sales Prediction using Python
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv('Advertising.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("=" * 58)
print("          SALES PREDICTION — ADVERTISING SPEND")
print("=" * 58)
print(f"\nDataset Shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDescriptive stats:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── 2. EXPLORATORY DATA ANALYSIS ────────────────────────────
print("\n--- Correlation with Sales ---")
print(df.corr()['Sales'].sort_values(ascending=False))

# ── 3. FEATURE ENGINEERING ──────────────────────────────────
# Total advertising budget
df['Total_Ad_Spend']   = df['TV'] + df['Radio'] + df['Newspaper']
# TV share of budget
df['TV_Share']         = df['TV'] / df['Total_Ad_Spend']
# TV × Radio interaction (historically high correlators)
df['TV_Radio_Interact']= df['TV'] * df['Radio']

features = ['TV', 'Radio', 'Newspaper',
            'Total_Ad_Spend', 'TV_Share', 'TV_Radio_Interact']
X = df[features]
y = df['Sales']

print(f"\nFeatures used : {features}")
print(f"\nTarget (Sales) stats:  Min={y.min():.2f}  Max={y.max():.2f}  Mean={y.mean():.2f}")

# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain : {len(X_train)} samples  |  Test : {len(X_test)} samples")

# ── 5. TRAIN MODELS ──────────────────────────────────────────
models = {
    'Linear Regression'  : LinearRegression(),
    'Ridge Regression'   : Ridge(alpha=1.0),
    'Random Forest'      : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting'  : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n--- Model Results ---")
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds         = mdl.predict(X_test)
    mae           = mean_absolute_error(y_test, preds)
    rmse          = np.sqrt(mean_squared_error(y_test, preds))
    r2            = r2_score(y_test, preds)
    cv_scores     = cross_val_score(mdl, X, y, cv=5, scoring='r2')
    results[name] = {'model': mdl, 'preds': preds,
                     'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv_scores.mean()}
    print(f"\n{name}")
    print(f"  MAE     : {mae:.4f}")
    print(f"  RMSE    : {rmse:.4f}")
    print(f"  R²      : {r2:.4f}")
    print(f"  CV R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

best_name = max(results, key=lambda k: results[k]['R2'])
best      = results[best_name]
print(f"\n{'=' * 58}")
print(f"  Best Model : {best_name}  (R² = {best['R2']:.4f})")
print(f"{'=' * 58}")

# ── 6. VISUALISATIONS ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Sales Prediction — Advertising Spend Analysis',
             fontsize=16, fontweight='bold')

# Plot 1 – TV vs Sales
ax = axes[0, 0]
ax.scatter(df['TV'], df['Sales'], alpha=0.6, color='#3498db', s=40)
m, b = np.polyfit(df['TV'], df['Sales'], 1)
x_line = np.linspace(df['TV'].min(), df['TV'].max(), 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=2)
ax.set_xlabel('TV Advertising ($000s)')
ax.set_ylabel('Sales (units)')
ax.set_title(f'TV vs Sales  (r={df["TV"].corr(df["Sales"]):.2f})')

# Plot 2 – Radio vs Sales
ax = axes[0, 1]
ax.scatter(df['Radio'], df['Sales'], alpha=0.6, color='#2ecc71', s=40)
m, b = np.polyfit(df['Radio'], df['Sales'], 1)
x_line = np.linspace(df['Radio'].min(), df['Radio'].max(), 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=2)
ax.set_xlabel('Radio Advertising ($000s)')
ax.set_ylabel('Sales (units)')
ax.set_title(f'Radio vs Sales  (r={df["Radio"].corr(df["Sales"]):.2f})')

# Plot 3 – Newspaper vs Sales
ax = axes[0, 2]
ax.scatter(df['Newspaper'], df['Sales'], alpha=0.6, color='#e67e22', s=40)
m, b = np.polyfit(df['Newspaper'], df['Sales'], 1)
x_line = np.linspace(df['Newspaper'].min(), df['Newspaper'].max(), 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=2)
ax.set_xlabel('Newspaper Advertising ($000s)')
ax.set_ylabel('Sales (units)')
ax.set_title(f'Newspaper vs Sales  (r={df["Newspaper"].corr(df["Sales"]):.2f})')

# Plot 4 – Actual vs Predicted (best model)
ax = axes[1, 0]
ax.scatter(y_test, best['preds'], alpha=0.7, color='#9b59b6', s=50)
lims = [min(y_test.min(), best['preds'].min()),
        max(y_test.max(), best['preds'].max())]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Sales')
ax.set_ylabel('Predicted Sales')
ax.set_title(f'Actual vs Predicted ({best_name})\nR² = {best["R2"]:.4f}')
ax.legend(fontsize=8)

# Plot 5 – Correlation heatmap
ax = axes[1, 1]
corr = df[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Correlation Heatmap')

# Plot 6 – Model R² comparison
ax = axes[1, 2]
names_list  = list(results.keys())
r2_list     = [results[n]['R2'] for n in names_list]
bar_colors  = ['#e74c3c' if n == best_name else '#bdc3c7' for n in names_list]
bars = ax.bar(names_list, r2_list, color=bar_colors, edgecolor='white')
for bar, score in zip(bars, r2_list):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005, f'{score:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title('Model Comparison (R² Score)')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1.1)
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('sales_prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] sales_prediction_results.png")

# ── 7. BUSINESS INSIGHTS ────────────────────────────────────
print("\n" + "=" * 58)
print("  BUSINESS INSIGHTS")
print("=" * 58)
tv_corr   = df['TV'].corr(df['Sales'])
rad_corr  = df['Radio'].corr(df['Sales'])
news_corr = df['Newspaper'].corr(df['Sales'])
print(f"1. TV Advertising correlation with Sales     : {tv_corr:.3f}  ← HIGHEST")
print(f"2. Radio Advertising correlation with Sales  : {rad_corr:.3f}")
print(f"3. Newspaper correlation with Sales          : {news_corr:.3f}  ← LOWEST")
print(f"4. Best model R²                             : {best['R2']:.4f}")
print(f"\n→ RECOMMENDATION: Invest primarily in TV advertising for maximum")
print(f"  sales impact. Radio provides a moderate boost.")
print(f"  Newspaper has the weakest effect on sales.")

# ── 8. SAMPLE PREDICTION ────────────────────────────────────
print("\n--- Sample Prediction ---")
sample = pd.DataFrame({
    'TV': [150], 'Radio': [30], 'Newspaper': [20],
    'Total_Ad_Spend': [200], 'TV_Share': [0.75],
    'TV_Radio_Interact': [4500]
})
pred_sales = best['model'].predict(sample)[0]
print(f"Budget: TV=$150K  Radio=$30K  Newspaper=$20K")
print(f"Predicted Sales : {pred_sales:.2f} units")
print("\nProject Complete!")
