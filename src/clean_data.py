import pandas as pd
import os

# Path to the raw input CSV containing vehicle tracking data
input_file = r"E:\project AI\vehicle-tracking-deai\data\vehicles_initial.csv"

df = pd.read_csv(input_file)

#delete bad datas
df = df.dropna()

# sort vehicle_id and frame
df = df.sort_values(by=["vehicle_id", "frame"])

# create processed folder
processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"
os.makedirs(processed_dir, exist_ok=True)

# Path for the cleaned output CSV
output_file = os.path.join(processed_dir, "vehicles_clean.csv")

# Save the cleaned DataFrame to CSV without the index
df.to_csv(output_file, index=False)


print(f"Clean data saved at {output_file}")
