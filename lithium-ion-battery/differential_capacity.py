import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file path
file_path = r'Cleaned Data\cleaned_profile_data.csv'
if not os.path.exists(file_path):
    file_path = r'g:\My Drive\Projects\FoPra\Li-Ion-Batterie\Cleaned Data\cleaned_profile_data.csv'

# Load data
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

CYCLE_TITLES = {
    1: "Cycle 1",
    2: "Cycle 2"
}

def plot_differential_capacity(cycle_number):
    print(f"Plotting Differential Capacity for Cycle {cycle_number}")
    
    # Filter for cycle
    df_cycle = df[df['Cycle'] == cycle_number]
    
    if df_cycle.empty:
        print(f"No data found for Cycle {cycle_number}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Styles
    styles = {
        'Calendered': {'color': 'blue', 'Charge': '-', 'Discharge': '--'},
        'Uncalendered': {'color': 'orange', 'Charge': '-', 'Discharge': '--'}
    }

    has_data = False
    for electrode in ['Calendered', 'Uncalendered']:
        for direction in ['Charge', 'Discharge']:
            # Filter for electrode and direction
            subset = df_cycle[
                (df_cycle['Electrode Type'] == electrode) & 
                (df_cycle['Direction'] == direction)
            ]
            
            if not subset.empty:
                # Filter out extreme values (outliers > 1000 or < -1000)
                subset = subset[subset['dQ/dV (mAh/V)'].abs() <= 1000]
                
                if subset.empty:
                    continue

                has_data = True
                label = f"{electrode} - {direction}"
                color = styles[electrode]['color']
                linestyle = styles[electrode][direction]
                
                # Plot dQ/dV vs Voltage
                ax.plot(
                    subset['Voltage (V)'], 
                    subset['dQ/dV (mAh/V)'], 
                    label=label,
                    color=color,
                    linestyle=linestyle
                )

    if not has_data:
        print("No data to plot.")
        plt.close(fig)
        return

    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('dQ/dV (mAh/V)')
    
    title = f"Differential Capacity - {CYCLE_TITLES.get(cycle_number, f'Cycle {cycle_number}')}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Save
    filename = f"DiffCap_Cycle_{cycle_number}.png"
    fig.savefig(filename)
    print(f"Saved {filename}")

# Run for Cycles 1 and 2
for cycle in [1, 2]:
    plot_differential_capacity(cycle)

plt.show()