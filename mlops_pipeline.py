# mlops_pipeline.py
"""
End-to-End MLOps Pipeline:
1. Load dataset
2. Train ML model
3. Evaluate and save
4. Reload and predict sample
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os

MODEL_TYPE = "RandomForest"  # Options: LinearRegression / DecisionTree / RandomForest
DATA_PATH = "data.csv"

print("📥 Loading dataset...")
data = pd.read_csv(DATA_PATH)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Dataset loaded: {len(data)} samples, {X.shape[1]} features")

if MODEL_TYPE == "LinearRegression":
    model = LinearRegression()
elif MODEL_TYPE == "DecisionTree":
    model = DecisionTreeClassifier(random_state=42)
elif MODEL_TYPE == "RandomForest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    raise ValueError("Invalid model type selected")

print(f"🚀 Training {MODEL_TYPE}...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
if MODEL_TYPE == "LinearRegression":
    mse = mean_squared_error(y_test, y_pred)
    print(f"📊 Mean Squared Error: {mse:.4f}")
else:
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 Accuracy: {acc:.4f}")

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")
print("💾 Model saved to artifacts/model.pkl")

# Verify reload and predict
loaded = joblib.load("artifacts/model.pkl")
sample = [list(X.iloc[0])]
pred = loaded.predict(sample)
print("\n🔍 Sample Prediction:")
print(f"Input: {sample}")
print(f"Output: {pred[0]}")

print("\n✅ Pipeline completed successfully!")
