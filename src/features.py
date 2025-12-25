# src/features.py
import pandas as pd
import os

input_file = r"E:\project AI\vehicle-tracking-deai\data\processed\vehicles_clean.csv"
df = pd.read_csv(input_file)

# calculate acceleration for each vehicle
df["acceleration"] = df.groupby("vehicle_id")["speed"].diff()

# delete NaN
df = df.dropna()

# saving
output_file = r"E:\project AI\vehicle-tracking-deai\data\processed\vehicles_features.csv"
df.to_csv(output_file, index=False)

print(f"Features created and saved at {output_file}")
