# src/plot_trajectory.py
import pandas as pd
import matplotlib.pyplot as plt
import os

input_file = r"E:\project AI\vehicle-tracking-deai\data\processed\vehicles_clean.csv"
df = pd.read_csv(input_file)


vehicle = df[df["vehicle_id"] == df["vehicle_id"].iloc[0]]

plt.figure(figsize=(8,6))
plt.plot(vehicle["x"], vehicle["y"], marker='o')
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title(f"Trajectory of Vehicle {vehicle['vehicle_id'].iloc[0]}")
plt.grid(True)

# ذخیره نمودار
results_dir = r"E:\project AI\vehicle-tracking-deai\results\trajectory_plots"
os.makedirs(results_dir, exist_ok=True)
plt_file = os.path.join(results_dir, f"vehicle_{vehicle['vehicle_id'].iloc[0]}_trajectory.png")
plt.savefig(plt_file)
plt.show()

print(f"Trajectory plot saved at {plt_file}")
