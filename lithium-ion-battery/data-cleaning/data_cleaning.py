import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- 1. CONFIGURATION ---

# Active Mass (mg) - calculated from your notes
MASS_CAL = 3.71 *0.95
MASS_UNCAL = (12.5 - 9.68) * 0.95

BASE_DIR = Path(__file__).resolve().parent

# File Names (Must match your local files exactly)
FILES = {
    "Calendered": {
        "overview": BASE_DIR / "calendered_overview.csv",
        "profiles": BASE_DIR / "calendered_dQdE.csv",
        "mass": MASS_CAL
    },
    "Uncalendered": {
        "overview": BASE_DIR / "uncalendered_overview.csv",
        "profiles": BASE_DIR / "uncalendered_dQdE.csv",
        "mass": MASS_UNCAL
    }
}

# --- 2. FUNCTION: PROCESS OVERVIEW (Cycle Data) ---
def process_overview(filepath, mass_mg, label):
    print(f"Reading {filepath}...")
    if not os.path.exists(filepath):
        print(f"WARNING: File not found: {filepath}")
        return pd.DataFrame()

    try:
        # Read file
        df = pd.read_csv(filepath)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Filter: 'Cycle number' must be numeric. This removes "SEI formation" text rows
        df['Cycle number'] = pd.to_numeric(df['Cycle number'], errors='coerce')
        df = df.dropna(subset=['Cycle number'])
        df['Cycle number'] = df['Cycle number'].astype(int)
        
        # Select columns - handling potential whitespace in headers
        # We try to find the columns that contain "Chg Cap" and "DChg Cap"
        chg_col = [c for c in df.columns if "Chg Cap(Ah)" in c][0]
        dchg_col = [c for c in df.columns if "DChg Cap(Ah)" in c][0]
        
        cols_to_keep = ['Cycle number', chg_col, dchg_col]
        
        # Check for C-rate
        crate_col = [c for c in df.columns if "C-rate" in c]
        if crate_col:
            cols_to_keep.append(crate_col[0])
            
        clean_df = df[cols_to_keep].copy()
        
        # Standardize column names for the output
        clean_df = clean_df.rename(columns={
            chg_col: 'Charge Capacity (Ah)',
            dchg_col: 'Discharge Capacity (Ah)'
        })
        if crate_col:
            clean_df = clean_df.rename(columns={crate_col[0]: 'C-rate'})
        
        # Calculations
        # Capacity: Ah * 1e6 / mg = mAh/g
        clean_df['Specific Charge Capacity (mAh/g)'] = pd.to_numeric(clean_df['Charge Capacity (Ah)']) * 1e6 / mass_mg
        clean_df['Specific Discharge Capacity (mAh/g)'] = pd.to_numeric(clean_df['Discharge Capacity (Ah)']) * 1e6 / mass_mg
        
        # Efficiency
        clean_df['Coulombic Efficiency (%)'] = (clean_df['Discharge Capacity (Ah)'] / clean_df['Charge Capacity (Ah)']) * 100
        
        clean_df['Electrode Type'] = label
        return clean_df
        
    except Exception as e:
        print(f"Error processing overview for {label}: {e}")
        return pd.DataFrame()

# --- 3. FUNCTION: PROCESS PROFILES (Voltage & dQ/dV) ---
def process_profiles(filepath, mass_mg, label):
    print(f"Reading {filepath}...")
    if not os.path.exists(filepath):
        print(f"WARNING: File not found: {filepath}")
        return pd.DataFrame()

    try:
        # Read with header at row 2 (The row starting with "E vs Li/Li+...")
        raw_df = pd.read_csv(filepath, header=2) 
        
        processed_chunks = []
        
        total_cols = raw_df.shape[1]
        cycle_count = 1
        
        # Loop through columns in blocks of 4 (Red_V, Red_Q, Ox_V, Ox_Q)
        for i in range(0, total_cols, 4):
            if i+3 >= total_cols: break
            
            # Extract block
            block = raw_df.iloc[:, i:i+4].copy()
            
            # Drop empty rows (where data ends for this cycle)
            block = block.dropna(how='all')
            
            if block.empty: continue

            # --- REDUCTION (Discharge) ---
            # Col 0: Voltage, Col 1: Capacity
            red_df = block.iloc[:, 0:2].copy()
            red_df.columns = ['Voltage (V)', 'Capacity (Ah)']
            red_df['Direction'] = 'Discharge'
            red_df['Cycle'] = cycle_count
            
            # --- OXIDATION (Charge) ---
            # Col 2: Voltage, Col 3: Capacity
            ox_df = block.iloc[:, 2:4].copy()
            ox_df.columns = ['Voltage (V)', 'Capacity (Ah)']
            ox_df['Direction'] = 'Charge'
            ox_df['Cycle'] = cycle_count
            
            # Combine
            cycle_df = pd.concat([red_df, ox_df])
            
            # Force numeric
            cycle_df['Voltage (V)'] = pd.to_numeric(cycle_df['Voltage (V)'], errors='coerce')
            cycle_df['Capacity (Ah)'] = pd.to_numeric(cycle_df['Capacity (Ah)'], errors='coerce')
            cycle_df = cycle_df.dropna()
            
            # Specific Capacity Calculation
            cycle_df['Specific Capacity (mAh/g)'] = cycle_df['Capacity (Ah)'] * 1e6 / mass_mg
            
            # Differential Capacity (dQ/dV) Calculation
            # Simple difference method
            cycle_df['dQ'] = cycle_df['Specific Capacity (mAh/g)'].diff()
            cycle_df['dV'] = cycle_df['Voltage (V)'].diff()
            
            # Calculate dQ/dV, handling division by zero/noise
            with np.errstate(divide='ignore', invalid='ignore'):
                cycle_df['dQ/dV (mAh/V)'] = cycle_df['dQ'] / cycle_df['dV']
            
            # Clean up infinite values from flat voltage steps
            cycle_df['dQ/dV (mAh/V)'] = cycle_df['dQ/dV (mAh/V)'].replace([np.inf, -np.inf], np.nan)
            
            processed_chunks.append(cycle_df)
            cycle_count += 1
            
        if not processed_chunks:
            return pd.DataFrame()
            
        full_profile_df = pd.concat(processed_chunks)
        full_profile_df['Electrode Type'] = label
        
        # Cleanup columns
        cols_to_export = ['Electrode Type', 'Cycle', 'Direction', 'Voltage (V)', 'Specific Capacity (mAh/g)', 'dQ/dV (mAh/V)']
        return full_profile_df[cols_to_export]

    except Exception as e:
        print(f"Error processing profiles for {label}: {e}")
        return pd.DataFrame()

# --- 4. EXECUTION ---

print("--- STARTING DATA CLEANING ---")

# 1. Process Overview Data
print("\n1. Processing Cycle Data (Overview)...")
df_overview_cal = process_overview(FILES['Calendered']['overview'], FILES['Calendered']['mass'], 'Calendered')
df_overview_uncal = process_overview(FILES['Uncalendered']['overview'], FILES['Uncalendered']['mass'], 'Uncalendered')

final_overview = pd.concat([df_overview_cal, df_overview_uncal], ignore_index=True)
final_overview.to_csv("cleaned_cycle_data.csv", index=False)
print(f"-> Saved 'cleaned_cycle_data.csv' with {len(final_overview)} rows.")

# 2. Process Profile Data
print("\n2. Processing Profile Data (dQdE)...")
df_profile_cal = process_profiles(FILES['Calendered']['profiles'], FILES['Calendered']['mass'], 'Calendered')
df_profile_uncal = process_profiles(FILES['Uncalendered']['profiles'], FILES['Uncalendered']['mass'], 'Uncalendered')

final_profiles = pd.concat([df_profile_cal, df_profile_uncal], ignore_index=True)
final_profiles.to_csv("cleaned_profile_data.csv", index=False)
print(f"-> Saved 'cleaned_profile_data.csv' with {len(final_profiles)} rows.")

print("\n--- DONE! ---")