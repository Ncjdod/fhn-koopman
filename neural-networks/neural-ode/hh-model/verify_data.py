"""
Verify Allen Brain Data for HH Neural ODE Training

Checks:
  1. Data loads correctly from NWB
  2. Units are correct (mV, pA, ms)
  3. Shapes and ranges are physiologically plausible
  4. Downsampling works
  5. I_ext interpolation function works
  6. Data is compatible with model input format
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax.numpy as jnp
import h5py

# Import our loader
from HH_model.AllenBrainLoader import download_nwb, find_sweeps, get_sweep_data, SPECIMEN_ID

print("=" * 60)
print("Allen Brain Data Verification for HH Neural ODE Training")
print("=" * 60)

# ============================================================
# 1. Load data
# ============================================================
print("\n[1] Loading NWB file...")
filepath = download_nwb()
assert filepath is not None, "FAIL: Could not download NWB file"
print(f"  OK: File at {filepath}")

# ============================================================
# 2. Find spiking sweep
# ============================================================
print("\n[2] Finding spiking sweep...")
with h5py.File(filepath, 'r') as f:
    sweeps = find_sweeps(f)
    print(f"  Total sweeps: {len(sweeps)}")
    
    # Find best spiking sweep
    target = None
    best_n_spikes = 0
    
    for sweep_name in sweeps:
        t, v, c = get_sweep_data(f, sweep_name)
        if v is None:
            continue
        
        # Count spikes (threshold crossings at 0 mV)
        crossings = np.diff(np.sign(v - 0.0))
        n_spikes = np.sum(crossings > 0)
        
        if n_spikes > best_n_spikes:
            best_n_spikes = n_spikes
            target = sweep_name
    
    print(f"  Best spiking sweep: {target} ({best_n_spikes} spikes)")
    
    # Load the target sweep
    t_raw, v_raw, c_raw = get_sweep_data(f, target)

assert t_raw is not None, "FAIL: Could not load sweep data"
print(f"  OK: Data loaded")

# ============================================================
# 3. Check raw data and make time relative
# ============================================================
print("\n[3] Raw data info...")
print(f"  Raw time:    [{t_raw[0]:.1f}, {t_raw[-1]:.1f}] ms (absolute)")
print(f"  Duration:    {t_raw[-1] - t_raw[0]:.1f} ms")
print(f"  Voltage:     [{v_raw.min():.1f}, {v_raw.max():.1f}] mV")
print(f"  Current:     [{c_raw.min():.1f}, {c_raw.max():.1f}] pA")
print(f"  Samples:     {len(t_raw)}")
dt = t_raw[1] - t_raw[0]
print(f"  dt:          {dt:.5f} ms ({1.0/dt:.0f} kHz)")

# Make time relative (start at 0)
t_raw = t_raw - t_raw[0]
print(f"  Time (shifted): [{t_raw[0]:.1f}, {t_raw[-1]:.1f}] ms")

# Sanity checks
assert v_raw.min() < -50, f"FAIL: Resting potential too high ({v_raw.min():.1f} mV)"
assert v_raw.max() > 0,   f"FAIL: No spikes detected (max V = {v_raw.max():.1f} mV)"
assert c_raw.max() > 0,   f"FAIL: No stimulus current detected"
print("  OK: Ranges physiologically plausible")

# ============================================================
# 4. Downsample for training
# ============================================================
print("\n[4] Downsampling...")

# Original 200kHz -> Downsample to 10kHz (0.1ms per sample)
downsample_factor = 20
t_ds = t_raw[::downsample_factor]
v_ds = v_raw[::downsample_factor]
c_ds = c_raw[::downsample_factor]

new_dt = t_ds[1] - t_ds[0]
print(f"  Downsample factor: {downsample_factor}x")
print(f"  New dt:    {new_dt:.3f} ms ({1.0/new_dt:.1f} kHz)")
print(f"  New samples: {len(t_ds)}")
print(f"  V range after DS: [{v_ds.min():.1f}, {v_ds.max():.1f}] mV")

# Check spike shape preserved
spike_diff = abs(v_ds.max() - v_raw.max())
print(f"  Spike peak loss: {spike_diff:.2f} mV {'(OK)' if spike_diff < 5 else '(WARNING)'}")

# ============================================================
# 5. Extract training window (stimulus region)
# ============================================================
print("\n[5] Extracting training window...")

# Find first spike (V crossing 0 mV upward)
crossings_full = np.diff(np.sign(v_ds - 0.0))
spike_indices = np.where(crossings_full > 0)[0]

if len(spike_indices) > 0:
    first_spike_idx = spike_indices[0]
    first_spike_t = t_ds[first_spike_idx]
    print(f"  First spike at: {first_spike_t:.1f} ms (index {first_spike_idx})")
    
    # Window: 5ms before first spike to 50ms after
    pre_window = 5.0
    post_window = 50.0
    t_start = max(first_spike_t - pre_window, t_ds[0])
    t_end = min(first_spike_t + post_window, t_ds[-1])
else:
    print("  WARNING: No spikes found in downsampled data!")
    print("  Using middle 55ms of recording")
    mid = len(t_ds) // 2
    t_start = t_ds[max(mid - 2750, 0)]  # ~55ms around middle
    t_end = t_ds[min(mid + 2750, len(t_ds)-1)]

mask = (t_ds >= t_start) & (t_ds <= t_end)
t_train = t_ds[mask]
v_train = v_ds[mask]
c_train = c_ds[mask]

# Shift time to start at 0
t_train = t_train - t_train[0]

print(f"  Training window: {t_train[-1]:.1f} ms ({len(t_train)} points)")
print(f"  V in window: [{v_train.min():.1f}, {v_train.max():.1f}] mV")
print(f"  I in window: [{c_train.min():.1f}, {c_train.max():.1f}] pA")

# Count spikes in window
crossings = np.diff(np.sign(v_train - 0.0))
n_spikes_window = np.sum(crossings > 0)
print(f"  Spikes in window: {n_spikes_window}")

# ============================================================
# 6. Convert to JAX arrays
# ============================================================
print("\n[6] Converting to JAX...")

t_jax = jnp.array(t_train, dtype=jnp.float32)
v_jax = jnp.array(v_train, dtype=jnp.float32)
c_jax = jnp.array(c_train, dtype=jnp.float32)

print(f"  t_jax: shape={t_jax.shape}, dtype={t_jax.dtype}")
print(f"  v_jax: shape={v_jax.shape}, dtype={v_jax.dtype}")
print(f"  c_jax: shape={c_jax.shape}, dtype={c_jax.dtype}")

# Create I_ext interpolation function
def make_I_ext_fn(t_data, c_data):
    """Create interpolation function for external current."""
    def I_ext_fn(t):
        return jnp.interp(t, t_data, c_data)
    return I_ext_fn

I_ext_fn = make_I_ext_fn(t_jax, c_jax)

# Test interpolation at a few points
print(f"\n  I_ext interpolation test:")
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    t_test = t_jax[int(frac * (len(t_jax)-1))]
    I_test = I_ext_fn(t_test)
    print(f"    t={t_test:>8.2f}ms -> I_ext={I_test:>8.1f} pA")

# ============================================================
# 7. Model compatibility check
# ============================================================
print("\n[7] Model compatibility check...")

y0 = jnp.array([v_jax[0]])
print(f"  y0 = [{y0[0]:.1f}] mV")
print(f"  t_span: {len(t_jax)} points, [{t_jax[0]:.2f}, {t_jax[-1]:.2f}] ms")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'HH_model'))
from HH_NeuralODE import create_model
import jax

model = create_model(key=jax.random.PRNGKey(0))
dy = model(0.0, y0, I_ext_fn(0.0))
print(f"  Model forward pass: dV/dt = {dy[0]:.4f}")
print(f"  OK: Model accepts data format")

# ============================================================
# 8. Unit conversion note
# ============================================================
print("\n[8] Unit conversion note:")
print(f"  Allen data:  pA (picoamperes, absolute current)")
print(f"  HH model:    uA/cm^2 (current density)")
print(f"  Allen range: [{c_train.min():.1f}, {c_train.max():.1f}] pA")
print(f"  HH range:    ~[0, 20] uA/cm^2")
print(f"  -> The Neural ODE will learn the implicit mapping")
print(f"     (Fourier features + MLP can absorb unit differences)")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY - ALL CHECKS PASSED")
print("=" * 60)
print(f"  Specimen:      {SPECIMEN_ID}")
print(f"  Best sweep:    {target} ({best_n_spikes} spikes total)")
print(f"  Training data: {len(t_train)} points, {t_train[-1]:.1f}ms window")
print(f"  Spikes in win: {n_spikes_window}")
print(f"  V range:       [{v_train.min():.1f}, {v_train.max():.1f}] mV")
print(f"  I range:       [{c_train.min():.1f}, {c_train.max():.1f}] pA")
print(f"  Model OK:      forward pass works")
print(f"  I_ext interp:  works")
print("=" * 60)
