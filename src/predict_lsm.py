import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"
model_file = os.path.join(processed_dir, "lstm_model.h5")

model = load_model(model_file, compile=False)

X = np.load(os.path.join(processed_dir, "X_lstm.npy"))
y = np.load(os.path.join(processed_dir, "y_lstm.npy"))


predictions = model.predict(X)


real = y


plt.figure(figsize=(10,6))
plt.plot(real[:,0], real[:,1], label="Real Path", marker='o')
plt.plot(predictions[:,0], predictions[:,1], label="Predicted Path", marker='x')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Vehicle Trajectory Prediction")
plt.legend()
plt.grid(True)

results_dir = r"E:\project AI\vehicle-tracking-deai\results\trajectory_plots"
os.makedirs(results_dir, exist_ok=True)
plt_file = os.path.join(results_dir, "vehicle_predicted_vs_real.png")
plt.savefig(plt_file)
plt.show()

print(f"Trajectory prediction plot saved at {plt_file}")
