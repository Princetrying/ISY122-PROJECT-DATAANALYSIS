import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load the cleaned dataset
df = pd.read_csv(r"C:\Users\msu-wone\Downloads\wfp_food_prices_phl (cleaned).csv")

# Quick overview
print("Shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# Convert DATE column to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

# Extract year and month
df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month

# Block 2 — Average Rice Price Trend Over Time
monthly_avg = df.groupby('DATE')['PRICE'].mean().reset_index()

plt.figure(figsize=(14, 5))
plt.plot(monthly_avg['DATE'], monthly_avg['PRICE'], color='steelblue', linewidth=2)
plt.title('Average Rice Price in the Philippines (2019 – March2026)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Average Price (PHP per KG)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('price_trend.png')
plt.show()
print("Chart saved as price_trend.png!")

# Block 3 — Average Price per Region
region_avg = df.groupby('ADMIN1')['PRICE'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
region_avg.plot(kind='bar', color='steelblue')
plt.title('Average Rice Price per Region (2019 – March2026)', fontsize=14)
plt.xlabel('Region')
plt.ylabel('Average Price (PHP per KG)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('price_per_region.png')
plt.show()
print("Chart saved as price_per_region.png!")

# Block 4 — Average Price per Rice Type
commodity_avg = df.groupby('COMMODITY')['PRICE'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
commodity_avg.plot(kind='bar', color='steelblue')
plt.title('Average Price per Rice Type (2019 – March2026)', fontsize=14)
plt.xlabel('Rice Type')
plt.ylabel('Average Price (PHP per KG)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('price_per_commodity.png')
plt.show()
print("Chart saved as price_per_commodity.png!")

# Block 5 — Correlation Analysis
df['REGION_CODE'] = df['ADMIN1'].astype('category').cat.codes
df['COMMODITY_CODE'] = df['COMMODITY'].astype('category').cat.codes
df['PRICETYPE_CODE'] = df['PRICETYPE'].astype('category').cat.codes

corr_df = df[['YEAR', 'MONTH', 'REGION_CODE',
              'COMMODITY_CODE', 'PRICETYPE_CODE', 'PRICE']]

corr_matrix = corr_df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Rice Price Variables', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Heatmap saved as correlation_heatmap.png!")

# Block 6 — Feature Engineering
df = df.sort_values(['ADMIN1', 'COMMODITY', 'DATE']).reset_index(drop=True)

# ✅ Create target variable FIRST
df['PRICE_NEXT'] = df.groupby(['ADMIN1', 'COMMODITY'])['PRICE'].shift(-1)
df['PRICE_DIRECTION'] = (df['PRICE_NEXT'] > df['PRICE']).astype(int)

# Lag features
df['PRICE_LAG1'] = df.groupby(['ADMIN1', 'COMMODITY'])['PRICE'].shift(1)
df['PRICE_LAG2'] = df.groupby(['ADMIN1', 'COMMODITY'])['PRICE'].shift(2)
df['PRICE_LAG3'] = df.groupby(['ADMIN1', 'COMMODITY'])['PRICE'].shift(3)

# Rolling averages
df['PRICE_ROLL3'] = df.groupby(['ADMIN1', 'COMMODITY'])['PRICE'].transform(lambda x: x.rolling(3).mean())
df['PRICE_ROLL6'] = df.groupby(['ADMIN1', 'COMMODITY'])['PRICE'].transform(lambda x: x.rolling(6).mean())

# Price change features
df['PRICE_CHANGE']     = df['PRICE'] - df['PRICE_LAG1']
df['PRICE_PCT_CHANGE'] = df.groupby(['ADMIN1', 'COMMODITY'])['PRICE'].pct_change()

# Is price above its 3-month average?
df['ABOVE_ROLLING_AVG'] = (df['PRICE'] > df['PRICE_ROLL3']).astype(int)

# Season
df['QUARTER'] = df['DATE'].dt.quarter

# Drop NaN rows
df = df.dropna()

print(f"Dataset size after feature engineering: {df.shape}")
print("Target distribution:")
print(df['PRICE_DIRECTION'].value_counts())

# Block 6 — Feature Engineering (Create Target Variable)
df = df.sort_values('DATE')
df['PRICE_NEXT'] = df.groupby(['ADMIN1', 'COMMODITY'])['PRICE'].shift(-1)
df['PRICE_DIRECTION'] = (df['PRICE_NEXT'] > df['PRICE']).astype(int)
df = df.dropna(subset=['PRICE_NEXT'])
print("\nTarget variable distribution:")
print(df['PRICE_DIRECTION'].value_counts())


# Block 7 — Prepare Features and Split Data
features = [
    'YEAR', 'MONTH', 'QUARTER',
    'REGION_CODE', 'COMMODITY_CODE', 'PRICETYPE_CODE',
    'PRICE_LAG1', 'PRICE_LAG2', 'PRICE_LAG3',
    'PRICE_ROLL3', 'PRICE_ROLL6',
    'PRICE_CHANGE', 'PRICE_PCT_CHANGE',
    'ABOVE_ROLLING_AVG'
]

X = df[features]
y = df['PRICE_DIRECTION']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Block 8 — Train and Evaluate Random Forest
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['DOWN (0)', 'UP (1)'],
            yticklabels=['DOWN (0)', 'UP (1)'])
plt.title('Confusion Matrix — XGBoost Classifier', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as confusion_matrix.png!")

# Block 10 — Compare Multiple ML Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss')
}

results = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = round(acc * 100, 2)
    print(f"{name}: {acc * 100:.2f}%")

plt.figure(figsize=(10, 5))
plt.barh(list(results.keys()), list(results.values()), color='steelblue')
plt.title('Model Accuracy Comparison', fontsize=14)
plt.xlabel('Accuracy (%)')
plt.xlim(40, 100)
for i, (k, v) in enumerate(results.items()):
    plt.text(v + 0.3, i, f'{v}%', va='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
print("\nBest model:", max(results, key=results.get),
      "with", max(results.values()), "%")