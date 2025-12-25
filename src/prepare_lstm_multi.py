import pandas as pd
import numpy as np
import os

processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"
input_file = os.path.join(processed_dir, "vehicles_normalized.csv")

df = pd.read_csv(input_file)
df = df.iloc[::5]
seq_length = 5
vehicle_ids = df['vehicle_id'].unique()

X_list, y_list = [], []

for vid in vehicle_ids:
    vehicle = df[df['vehicle_id'] == vid][['x','y','speed']].values
    for i in range(len(vehicle) - seq_length):
        X_list.append(vehicle[i:i+seq_length])
        y_list.append(vehicle[i+seq_length,:2]) 

X = np.array(X_list)
y = np.array(y_list)

np.save(os.path.join(processed_dir, "X_lstm_multi.npy"), X)
np.save(os.path.join(processed_dir, "y_lstm_multi.npy"), y)

print(f"Multi-vehicle sequences created: X={X.shape}, y={y.shape}")
 