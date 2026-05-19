import pandas as pd
import os

file_path = r'g:\My Drive\Projects\FoPra\Li-Ion-Batterie\Cleaned Data\cleaned_profile_data.csv'

try:
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

print(f"Columns: {df.columns.tolist()}")
print(f"Unique Cycles: {df['Cycle'].unique()}")
print(f"Unique Electrode Types: {df['Electrode Type'].unique()}")
print(f"Unique Directions: {df['Direction'].unique()}")
