import pandas as pd
import os

input_file = r"E:\project AI\vehicle-tracking-deai\data\vehicles_initial.csv"
df = pd.read_csv(input_file)

#delete bad datas
df = df.dropna()

# sort vehicle_id and frame
df = df.sort_values(by=["vehicle_id", "frame"])

# create processed folder
processed_dir = r"E:\project AI\vehicle-tracking-deai\data\processed"
os.makedirs(processed_dir, exist_ok=True)

output_file = os.path.join(processed_dir, "vehicles_clean.csv")
df.to_csv(output_file, index=False)

print(f"Clean data saved at {output_file}")