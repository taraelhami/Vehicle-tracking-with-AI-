import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

# file pass in my computer
input_file = r"E:\project AI\vehicle-tracking-deai\data\processed\vehicles_clean.csv"

df = pd.read_csv(input_file)

# normalize
feature_cols = ["x", "y", "speed"]

scaler = MinMaxScaler()


df[feature_cols] = scaler.fit_transform(df[feature_cols])


processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"
os.makedirs(processed_dir, exist_ok=True)

joblib.dump(scaler, os.path.join(processed_dir, "scaler.save"))

# save normalized data
output_file = os.path.join(processed_dir, "vehicles_normalized.csv")
df.to_csv(output_file, index=False)

print("✅ Data normalized successfully")
print(f"📁 Saved at: {output_file}")
print(df[["x", "y", "speed"]].describe())