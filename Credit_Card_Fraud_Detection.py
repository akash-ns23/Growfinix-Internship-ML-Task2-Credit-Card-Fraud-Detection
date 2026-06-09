import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Create sample dataset
np.random.seed(42)

normal = 95
fraud = 5

amount = np.concatenate([
    np.random.randint(100, 5000, normal),
    np.random.randint(10000, 50000, fraud)
])

time = np.concatenate([
    np.random.randint(1, 86400, normal),
    np.random.randint(1, 86400, fraud)
])

target = np.concatenate([
    np.zeros(normal),
    np.ones(fraud)
])

df = pd.DataFrame({
    "Amount": amount,
    "Time": time,
    "Target": target.astype(int)
})

print("First 5 rows of dataset:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Features
X = df[["Amount", "Time"]]
y = df["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Isolation Forest
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X_train)

# Predict
pred = model.predict(X_test)

# Convert predictions
pred = np.where(pred == -1, 1, 0)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("\n==============================")
print("RESULT")
print("==============================")
print("Isolation Forest Accuracy:", accuracy)

print("\nTask Completed Successfully!")
