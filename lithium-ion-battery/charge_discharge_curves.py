import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Use raw string for path to avoid escape sequence warnings
file_path = r'Cleaned Data\cleaned_raw_curves.csv'
if not os.path.exists(file_path):
    # Fallback for absolute path if running from different cwd
    file_path = r'g:\My Drive\Projects\FoPra\Li-Ion-Batterie\Cleaned Data\cleaned_raw_curves.csv'

df_charge_discharge = pd.read_csv(file_path)

CYCLE_TITLES = {
    2: "SEI Formation",
    5: "Stress Prep",
    9: "Rate Capability C/2"
}

def plot_combined_cycle(cycle_number):
    print(f"Plotting Combined Cycle {cycle_number}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define styles for distinction
    styles = {
        'Calendered': {'color': 'blue', 'Charge': '-', 'Discharge': '--'},
        'Uncalendered': {'color': 'orange', 'Charge': '-', 'Discharge': '--'}
    }

    has_data = False
    for electrode in ['Calendered', 'Uncalendered']:
        # Filter data for the specific cycle and electrode
        df_subset = df_charge_discharge[
            (df_charge_discharge['Cycle'] == cycle_number) & 
            (df_charge_discharge['Electrode Type'] == electrode)
        ]
        
        if df_subset.empty:
            print(f"  No data for {electrode}")
            continue
            
        has_data = True
        
        # Plot Charge and Discharge
        for direction in ['Charge', 'Discharge']:
            df_direction = df_subset[df_subset['Direction'] == direction]
            if not df_direction.empty:
                label = f"{electrode} - {direction}"
                linestyle = styles[electrode][direction]
                color = styles[electrode]['color']
                
                ax.plot(
                    df_direction['Specific Capacity (mAh/g)'], 
                    df_direction['Voltage(V)'], 
                    label=label,
                    color=color,
                    linestyle=linestyle
                )

    if not has_data:
        plt.close(fig)
        return None, None

    ax.set_xlabel('Specific Capacity (mAh/g)')
    ax.set_ylabel('Voltage (V)')
    
    # Set custom title
    title_text = CYCLE_TITLES.get(cycle_number, f'Cycle {cycle_number}')
    ax.set_title(title_text)
    ax.legend()
    ax.grid(True)
    
    # Save figure
    filename = f"Cycle_{cycle_number}_Charge_Discharge.png"
    fig.savefig(filename)
    print(f"  Saved {filename}")
    
    return fig, ax

# Vectorize/Iterate over specific cycles
target_cycles = [2, 5, 9]

all_figures = []
all_axes = []

for cycle in target_cycles:
    fig, ax = plot_combined_cycle(cycle)
    if fig:
        all_figures.append(fig)
        all_axes.append(ax)

# Now all_figures and all_axes contain the objects in memory
print(f"Generated {len(all_figures)} plots.")
plt.show()
