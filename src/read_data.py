import pandas as pd

file_path = r"E:\project AI\vehicle-tracking-deai\data\trajectories-0750am-0805am.txt"

df = pd.read_csv(file_path, delim_whitespace=True, header=None)

df = df[[0, 1, 5, 6, 8]]
df.columns = ["vehicle_id", "frame", "x", "y", "speed"]

print("اولین ۵ خط داده:")
print(df.head())
print("\nاطلاعات کلی:")
print(df.info())

df.to_csv(r"E:\project AI\vehicle-tracking-deai\data\vehicles_initial.csv", index=False)