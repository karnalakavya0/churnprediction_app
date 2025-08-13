import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load data directly from local CSV
df = pd.read_csv("data.csv")
print("Dataset loaded successfully. Shape:", df.shape)

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(0, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df.drop(['customerID'], axis=1), drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Model
model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model.fit(X_train_res, y_train_res)

# Evaluation
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
feature_names = X.columns.tolist()
# Save model and scaler
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved!")
