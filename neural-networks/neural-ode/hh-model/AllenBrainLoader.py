"""
Allen Brain Dataset Loader (H5PY Fallback - Fixed)

Directly reads Allen NWB v1 files using h5py. Correctly scales voltage (V -> mV) and current (A -> pA).
"""

import requests
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Allen Brain API
# ============================================================
DOWNLOAD_URL = "http://api.brain-map.org/api/v2/well_known_file_download/491316386"
SPECIMEN_ID = 485909730

def download_nwb(save_dir="allen_data"):
    """Download NWB file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"specimen_{SPECIMEN_ID}.nwb")
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
        print(f"File exists: {filepath}")
        return filepath
    
    print(f"Downloading from {DOWNLOAD_URL}...")
    try:
        resp = requests.get(DOWNLOAD_URL, stream=True)
        resp.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def find_sweeps(f):
    """List available sweeps."""
    sweeps = []
    if 'acquisition' in f:
        # Check timeseries first
        ts = f['acquisition'].get('timeseries', {})
        for key in ts.keys():
            if 'Sweep_' in key:
                sweeps.append(key)
    return sorted(sweeps, key=lambda x: int(x.split('_')[1]))

def get_sweep_data(f, sweep_name):
    """Extract voltage (mV) and current (pA)."""
    try:
        # Voltage
        v_node = f['acquisition']['timeseries'][sweep_name]
        raw_v = v_node['data'][()]
        voltage_mV = raw_v * 1000 # V -> mV
        
        # Time
        starting_time = v_node['starting_time'][()] if 'starting_time' in v_node else 0.0
        rate = v_node['starting_time'].attrs.get('rate', 200000.0)
        time_ms = (starting_time + np.arange(len(raw_v)) / rate) * 1000
        
        # Current
        # Try finding corresponding stimulus sweep
        c_node = None
        if 'stimulus' in f and 'presentation' in f['stimulus']:
            pres = f['stimulus']['presentation']
            if sweep_name in pres:
                c_node = pres[sweep_name]
            
        if c_node:
            raw_c = c_node['data'][()]
            current_pA = raw_c * 1e12 # A -> pA
            
            # Crop to match length
            min_len = min(len(voltage_mV), len(current_pA))
            voltage_mV = voltage_mV[:min_len]
            current_pA = current_pA[:min_len]
            time_ms = time_ms[:min_len]
        else:
            current_pA = np.zeros_like(voltage_mV)
            
        return time_ms, voltage_mV, current_pA
        
    except Exception as e:
        print(f"Error reading {sweep_name}: {e}")
        return None, None, None

def plot_sweep(filepath):
    print(f"Opening {filepath}...")
    with h5py.File(filepath, 'r') as f:
        sweeps = find_sweeps(f)
        print(f"Found {len(sweeps)} sweeps.")
        
        # Find a sweep with spikes (max V > 0 mV)
        target = None
        for sweep_name in sweeps:
            # Skip test pulses (low numbers usually)
            if int(sweep_name.split('_')[1]) < 10: continue
            
            t, v, c = get_sweep_data(f, sweep_name)
            if v is not None and np.max(v) > 0:
                print(f"Found spiking sweep: {sweep_name} (Max V: {np.max(v):.1f} mV)")
                target = sweep_name
                break
        
        if not target and sweeps:
            target = sweeps[-1]
            
        if target:
            t, v, c = get_sweep_data(f, target)
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Voltage
            ax[0].plot(t, v, 'b', lw=0.8)
            ax[0].set_ylabel("Voltage (mV)")
            ax[0].set_title(f"Allen Specimen {SPECIMEN_ID} - {target}")
            ax[0].grid(True, alpha=0.3)
            
            # Current
            ax[1].plot(t, c, 'r', lw=0.8)
            ax[1].set_ylabel("Current (pA)")
            ax[1].set_xlabel("Time (ms)")
            ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('allen_data_preview.png')
            print("Saved allen_data_preview.png")

if __name__ == "__main__":
    fpath = download_nwb()
    if fpath:
        plot_sweep(fpath)
