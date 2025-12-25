import pandas as pd
import numpy as np
import os

processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"
input_file = os.path.join(processed_dir, "vehicles_features.csv")

df = pd.read_csv(input_file)

vehicle = df[df["vehicle_id"] == df["vehicle_id"].iloc[0]]

data = vehicle[["x","y","speed"]].values

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, :2])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(data, seq_length)

np.save(os.path.join(processed_dir, "X_lstm.npy"), X)
np.save(os.path.join(processed_dir, "y_lstm.npy"), y)

print(f"LSTM sequences created: X.shape={X.shape}, y.shape={y.shape}")
