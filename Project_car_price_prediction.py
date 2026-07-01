# ============================================================
#Project: Car Price Prediction with Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv('car_data.csv')

print("=" * 58)
print("        CAR PRICE PREDICTION — ML REGRESSION")
print("=" * 58)
print(f"\nDataset Shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDescriptive stats:\n{df.describe()}")

# ── 2. FEATURE ENGINEERING ───────────────────────────────────
# Car Age from Year (dataset circa 2020)
df['Car_Age'] = 2020 - df['Year']

# Price Depreciation ratio
df['Price_Ratio'] = df['Selling_Price'] / df['Present_Price']

# Encode categorical columns
cat_cols = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']
le = LabelEncoder()
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

# Feature matrix
feature_cols = ['Present_Price', 'Driven_kms', 'Car_Age',
                'Owner', 'Fuel_Type_enc', 'Selling_type_enc',
                'Transmission_enc']
X = df[feature_cols]
y = df['Selling_Price']

print(f"\nFeatures used : {feature_cols}")
print(f"\nTarget (Selling_Price) stats:")
print(f"  Min: {y.min():.2f}  Max: {y.max():.2f}  Mean: {y.mean():.2f}")

# ── 3. TRAIN / TEST SPLIT ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain samples : {len(X_train)}  |  Test samples : {len(X_test)}")

# ── 4. TRAIN MULTIPLE MODELS ─────────────────────────────────
models = {
    'Linear Regression'        : LinearRegression(),
    'Random Forest'            : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting'        : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds        = mdl.predict(X_test)
    mae          = mean_absolute_error(y_test, preds)
    rmse         = np.sqrt(mean_squared_error(y_test, preds))
    r2           = r2_score(y_test, preds)
    results[name] = {'model': mdl, 'preds': preds, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"\n{name}")
    print(f"  MAE  : {mae:.4f} Lakhs")
    print(f"  RMSE : {rmse:.4f} Lakhs")
    print(f"  R²   : {r2:.4f}")

# Best model = highest R²
best_name  = max(results, key=lambda k: results[k]['R2'])
best       = results[best_name]
print(f"\n{'=' * 58}")
print(f"  Best Model : {best_name}  (R² = {best['R2']:.4f})")
print(f"{'=' * 58}")

# ── 5. VISUALISATIONS ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Car Price Prediction — Results', fontsize=16, fontweight='bold')

# Plot 1 – Actual vs Predicted (best model)
ax = axes[0, 0]
ax.scatter(y_test, best['preds'], alpha=0.6, color='#3498db', s=50)
lims = [min(y_test.min(), best['preds'].min()),
        max(y_test.max(), best['preds'].max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
ax.set_xlabel('Actual Price (Lakhs)')
ax.set_ylabel('Predicted Price (Lakhs)')
ax.set_title(f'Actual vs Predicted ({best_name})\nR² = {best["R2"]:.4f}')
ax.legend()

# Plot 2 – Residuals
ax = axes[0, 1]
residuals = y_test.values - best['preds']
ax.scatter(best['preds'], residuals, alpha=0.6, color='#e67e22', s=50)
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Predicted Price')
ax.set_ylabel('Residual')
ax.set_title('Residual Plot')

# Plot 3 – Feature Importance (Random Forest)
ax = axes[1, 0]
rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
colors_fi = plt.cm.viridis(np.linspace(0.2, 0.9, len(feat_imp)))
feat_imp.plot(kind='barh', ax=ax, color=colors_fi)
ax.set_title('Feature Importance (Random Forest)')
ax.set_xlabel('Importance Score')

# Plot 4 – Model comparison bar
ax = axes[1, 1]
model_names = list(results.keys())
r2_scores   = [results[n]['R2']  for n in model_names]
bar_colors  = ['#e74c3c' if n == best_name else '#95a5a6' for n in model_names]
bars = ax.bar(model_names, r2_scores, color=bar_colors, edgecolor='white')
for bar, score in zip(bars, r2_scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005, f'{score:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Model Comparison (R² Score)')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1.05)
ax.tick_params(axis='x', rotation=10)

plt.tight_layout()
plt.savefig('car_price_prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] car_price_prediction_results.png")

# ── 6. SAMPLE PREDICTION ─────────────────────────────────────
print("\n--- Sample Prediction ---")
sample_data = pd.DataFrame({
    'Present_Price'   : [6.0],
    'Driven_kms'      : [40000],
    'Car_Age'         : [5],
    'Owner'           : [0],
    'Fuel_Type_enc'   : [0],    # Petrol
    'Selling_type_enc': [0],    # Dealer
    'Transmission_enc': [0],    # Manual
})
pred_price = best['model'].predict(sample_data)[0]
print(f"Car   : 5-year-old Petrol, 40,000 km, Present Price 6L")
print(f"Predicted Selling Price : ₹ {pred_price:.2f} Lakhs")
print("\nProject Complete!")
