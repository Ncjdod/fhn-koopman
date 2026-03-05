"""
Allen Brain Data Loading & Preprocessing for Phase 2

Handles:
  - NWB file download from Allen Brain API
  - Sweep extraction (voltage mV, current pA, time ms)
  - Downsampling and windowing around spiking activity
  - Smooth dV/dt computation via Savitzky-Golay filter
"""

import os
import requests
import h5py
import numpy as np
from scipy.signal import decimate, savgol_filter


# ================================================================
# Allen Brain API
# ================================================================

DOWNLOAD_URL = "http://api.brain-map.org/api/v2/well_known_file_download/491316386"
SPECIMEN_ID = 485909730


def download_nwb(save_dir=None):
    """
    Download Allen Brain NWB file (cached).

    Args:
        save_dir: Directory to save file (default: allen_data/ next to this script)

    Returns:
        filepath: Path to NWB file, or None on failure
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allen_data")
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"specimen_{SPECIMEN_ID}.nwb")

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
        print(f"Allen data cached: {filepath}")
        return filepath

    print(f"Downloading Allen Brain data...")
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
    """List available sweep names in NWB file."""
    sweeps = []
    if 'acquisition' in f:
        ts = f['acquisition'].get('timeseries', {})
        for key in ts.keys():
            if 'Sweep_' in key:
                sweeps.append(key)
    return sorted(sweeps, key=lambda x: int(x.split('_')[1]))


def get_sweep_data(f, sweep_name):
    """
    Extract time (ms), voltage (mV), current (pA) from one sweep.

    Returns:
        time_ms, voltage_mV, current_pA — or (None, None, None) on error
    """
    try:
        v_node = f['acquisition']['timeseries'][sweep_name]
        raw_v = v_node['data'][()]
        voltage_mV = raw_v * 1000  # V -> mV

        starting_time = v_node['starting_time'][()] if 'starting_time' in v_node else 0.0
        rate = v_node['starting_time'].attrs.get('rate', 200000.0)
        time_ms = (starting_time + np.arange(len(raw_v)) / rate) * 1000

        # Stimulus current
        c_node = None
        if 'stimulus' in f and 'presentation' in f['stimulus']:
            pres = f['stimulus']['presentation']
            if sweep_name in pres:
                c_node = pres[sweep_name]

        if c_node is not None:
            raw_c = c_node['data'][()]
            current_pA = raw_c * 1e12  # A -> pA
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


# ================================================================
# Preprocessing
# ================================================================

def _find_best_sweep(f, sweeps):
    """Find the sweep with the most spikes."""
    best_sweep = None
    best_n_spikes = 0

    for sweep_name in sweeps:
        t, v, c = get_sweep_data(f, sweep_name)
        if v is None:
            continue
        crossings = np.diff(np.sign(v - 0.0))
        n_spikes = np.sum(crossings > 0)
        if n_spikes > best_n_spikes:
            best_n_spikes = n_spikes
            best_sweep = sweep_name

    return best_sweep, best_n_spikes


def load_allen_data(downsample=20, window_pre=5.0, window_post=50.0):
    """
    Load, preprocess, and window Allen Brain electrophysiology data.

    Steps:
        1. Download NWB file (cached)
        2. Find sweep with most spikes
        3. Downsample (FIR anti-aliasing)
        4. Window around first spike [first_spike - pre, first_spike + post]

    Args:
        downsample:   Downsampling factor (200kHz / 20 = 10kHz)
        window_pre:   ms before first spike to include
        window_post:  ms after first spike to include

    Returns:
        t_ms:  (T,) time in ms (starting from 0)
        V_mV:  (T,) voltage in mV
        I_pA:  (T,) current in pA
    """
    print("\n--- Loading Allen Brain Data ---")

    filepath = download_nwb()
    assert filepath is not None, "Failed to download NWB file"

    with h5py.File(filepath, 'r') as f:
        sweeps = find_sweeps(f)
        print(f"Found {len(sweeps)} sweeps")

        best_sweep, best_n_spikes = _find_best_sweep(f, sweeps)
        print(f"Best sweep: {best_sweep} ({best_n_spikes} spikes)")

        t_raw, v_raw, c_raw = get_sweep_data(f, best_sweep)

    # Zero-base time
    t_raw = t_raw - t_raw[0]

    # Downsample
    v_ds = decimate(v_raw, downsample, ftype='fir')
    c_ds = decimate(c_raw, downsample, ftype='fir')
    t_ds = t_raw[::downsample][:len(v_ds)]

    # Window around first spike
    crossings = np.diff(np.sign(v_ds - 0.0))
    spike_idx = np.where(crossings > 0)[0]

    if len(spike_idx) > 0:
        first_spike_t = t_ds[spike_idx[0]]
        t_start = max(first_spike_t - window_pre, t_ds[0])
        t_end = min(first_spike_t + window_post, t_ds[-1])
    else:
        print("WARNING: No spikes found. Using first 55ms.")
        t_start = t_ds[0]
        t_end = t_ds[0] + 55.0

    mask = (t_ds >= t_start) & (t_ds <= t_end)
    t_train = t_ds[mask] - t_ds[mask][0]  # start from 0
    v_train = v_ds[mask]
    c_train = c_ds[mask]

    n_spikes_win = np.sum(np.diff(np.sign(v_train - 0.0)) > 0)
    print(f"Window: {t_train[-1]:.1f}ms, {len(t_train)} points, {n_spikes_win} spikes")
    print(f"V: [{v_train.min():.1f}, {v_train.max():.1f}] mV")
    print(f"I: [{c_train.min():.1f}, {c_train.max():.1f}] pA")

    return (
        np.array(t_train, dtype=np.float32),
        np.array(v_train, dtype=np.float32),
        np.array(c_train, dtype=np.float32),
    )


def compute_dVdt(t_ms, V_mV, window=5, polyorder=3):
    """
    Compute dV/dt from voltage trace using Savitzky-Golay filter.

    Much more robust than raw finite differences for noisy electrophysiology.
    The filter smooths and differentiates in one step.

    Args:
        t_ms:      (T,) time in ms
        V_mV:      (T,) voltage in mV
        window:    Filter window length (odd integer, >=polyorder+2)
        polyorder: Polynomial order for fitting

    Returns:
        dVdt: (T,) in mV/ms
    """
    # Average dt for uniform grid assumption (after downsampling, nearly uniform)
    dt_mean = np.mean(np.diff(t_ms))

    # Savitzky-Golay derivative: deriv=1 gives first derivative
    # Result is in units of V_mV per sample, divide by dt to get per ms
    dVdt = savgol_filter(V_mV, window_length=window, polyorder=polyorder,
                         deriv=1, delta=dt_mean)

    return dVdt.astype(np.float32)


# ================================================================
# Quick test
# ================================================================
if __name__ == "__main__":
    print("Allen Brain Data Loader — Test")
    print("=" * 50)

    t, V, I = load_allen_data()
    print(f"\nLoaded: {len(t)} points, {t[-1]:.1f}ms")

    dVdt = compute_dVdt(t, V)
    print(f"dV/dt range: [{dVdt.min():.1f}, {dVdt.max():.1f}] mV/ms")
    print(f"dV/dt std:   {dVdt.std():.1f} mV/ms")

    print("\nData loader OK!")
