import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

input_file = r"E:\project AI\vehicle-tracking-deai\data\processed\vehicles_features.csv"
df = pd.read_csv(input_file)

df["target"] = (df["acceleration"] > 0).astype(int)

X = df[["speed"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

model_file = r"E:\project AI\vehicle-tracking-deai\data\processed\vehicle_model.pkl"
joblib.dump(model, model_file)
print(f"Model saved at {model_file}")