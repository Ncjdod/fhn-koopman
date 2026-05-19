import pandas as pd
import os

file_path = r'g:\My Drive\Projects\FoPra\Li-Ion-Batterie\Cleaned Data\cleaned_raw_curves.csv'

try:
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

print(f"Unique Electrode Types: {df['Electrode Type'].unique()}")

target_cycles = [2, 5, 9]

for cycle in target_cycles:
    print(f"\n--- Cycle {cycle} ---")
    for electrode in df['Electrode Type'].unique():
        subset = df[(df['Cycle'] == cycle) & (df['Electrode Type'] == electrode)]
        print(f"Electrode: {electrode}, Rows: {len(subset)}")
        if not subset.empty:
            print(f"  Voltage Mean: {subset['Voltage(V)'].mean()}")
            print(f"  Capacity Max: {subset['Specific Capacity (mAh/g)'].max()}")
