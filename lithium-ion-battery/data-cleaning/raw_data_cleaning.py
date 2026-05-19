import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- 1. MASS CONFIGURATION (Graphite Only) ---

# Original total active layer masses
total_mass_cal = 3.71
total_mass_uncal = 12.5 - 9.68  # 2.82 mg

# Apply 0.95 factor for Graphite content
MASS_CAL_GRAPHITE = total_mass_cal * 0.95
MASS_UNCAL_GRAPHITE = total_mass_uncal * 0.95

print(f"Graphite Mass (Calendered): {MASS_CAL_GRAPHITE:.4f} mg")
print(f"Graphite Mass (Uncalendered): {MASS_UNCAL_GRAPHITE:.4f} mg")

# File Names
FILES = {
    "Calendered": {
        "raw": BASE_DIR / "calendered_rawdata.csv",
        "mass": MASS_CAL_GRAPHITE,
    },
    "Uncalendered": {
        "raw": BASE_DIR / "uncalendered_rawdata.csv",
        "mass": MASS_UNCAL_GRAPHITE,
    },
}

# --- 2. RAW DATA PROCESSING FUNCTION ---

def process_raw_data(filepath, mass_mg, label):
    print(f"Reading {filepath}...")
    if not filepath.exists():
        print(f"WARNING: File not found: {filepath}")
        return pd.DataFrame()

    try:
        # Load raw data
        # Raw data often has units in the second row, but here we assume standard header
        df = pd.read_csv(filepath)
        
        # Clean columns
        df.columns = df.columns.str.strip()
        
        # We need: Cycle, Current, Capacity, Voltage
        # Filter for active steps only (Charge or Discharge)
        # Usually identifiable by 'status' containing "Chg" or Current != 0
        # Looking at snippet: status="CCCV DChg" or "CCCV Chg"
        
        # Filter 1: Drop rows where status is 'Rest'
        mask_active = df['status'].str.contains('Chg', case=False, na=False)
        df_active = df[mask_active].copy()
        
        # Ensure numeric
        cols_to_numeric = ['Cycle', 'Current(A)', 'Capacity(Ah)', 'Voltage(V)']
        for col in cols_to_numeric:
            df_active[col] = pd.to_numeric(df_active[col], errors='coerce')
        
        df_active = df_active.dropna(subset=cols_to_numeric)
        
        # Determine Direction
        # Current < 0 -> Discharge (Lithiation)
        # Current > 0 -> Charge (Delithiation)
        # (Note: In some half-cell setups signs are flipped, but Voltage trend confirms)
        df_active['Direction'] = np.where(df_active['Current(A)'] < 0, 'Discharge', 'Charge')
        
        # Calculate Specific Capacity (mAh/g) based on Graphite Mass
        # Capacity(Ah) * 1e6 / mg
        df_active['Specific Capacity (mAh/g)'] = (df_active['Capacity(Ah)'] * 1e6) / mass_mg
        
        # Select final columns for plotting
        clean_df = df_active[[
            'Cycle', 
            'Direction', 
            'Voltage(V)', 
            'Specific Capacity (mAh/g)', 
            'Current(A)',
            'status'
        ]].copy()
        
        # Cast Cycle to int for cleaner filtering later
        clean_df['Cycle'] = clean_df['Cycle'].astype(int)
        
        clean_df['Electrode Type'] = label
        
        return clean_df

    except Exception as e:
        print(f"Error processing raw data for {label}: {e}")
        return pd.DataFrame()

# --- 3. EXECUTION ---

print("\n--- STARTING RAW DATA PROCESSING ---")

df_cal = process_raw_data(FILES['Calendered']['raw'], FILES['Calendered']['mass'], "Calendered")
df_uncal = process_raw_data(FILES['Uncalendered']['raw'], FILES['Uncalendered']['mass'], "Uncalendered")

# Combine
final_df = pd.concat([df_cal, df_uncal], ignore_index=True)

# Export
output_filename = "cleaned_raw_curves.csv"
final_df.to_csv(output_filename, index=False)

print(f"\nSUCCESS! Processed {len(final_df)} rows.")
print(f"Data saved to: {output_filename}")
print("\nTIP for Plotting:")
print("1. Filter by 'Electrode Type' (Calendered/Uncalendered)")
print("2. Filter by 'Cycle' (e.g., Cycle 1 for Formation, Cycle 10 for C-rate test)")
print("3. Plot X='Specific Capacity (mAh/g)', Y='Voltage(V)'")