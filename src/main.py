import pandas as pd


file_path = r"E:\project AI\vehicle-tracking-deai\data\trajectories-0750am-0805am.txt"

columns = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
    "Local_X", "Local_Y", "Global_X", "Global_Y",
    "Vehicle_Length", "Vehicle_Width", "Vehicle_Class",
    "Velocity", "Acceleration", "Lane_ID",
    "Preceding", "Following", "Space_Headway", "Time_Headway"
]



df = pd.read_csv(file_path, sep="\t", header=None)
print(df.head())
print(df.info())
