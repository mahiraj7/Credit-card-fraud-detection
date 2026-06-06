import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("creditcard.csv")
print(data.head())

print("Dataset Shape:", data.shape)

print("Missing Values:")
print(data.isnull().sum())

print("Transaction Count:")
print(data["Class"].value_counts())

X= data.drop("Class", axis=1)
y= data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
)

model = IsolationForest(
    n_estimators=100,
    contamination=0.002,
    random_state=42
)

model.fit(X_train)
predictions = model.predict(X_test)

y_pred = [1 if value == -1 else 0 for value in predictions]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "fraud_model.pkl")
joblib.dump(X.columns, "fraud_features.pkl")

print("Model saved successfully!")
