"""Quick test: train 500 epochs with new architecture, then evaluate."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Phase1Config
from train_field import train_phase1

# Phase 1: 500 epochs to test new architecture
p1 = Phase1Config()
p1.n_epochs = 500
p1.log_every = 50
p1.plot_every = 9999
p1.val_every = 100
p1.checkpoint_every = 250

print("Training with new architecture (Fourier + gate-safe + clipping)...")
model, history = train_phase1(config=p1)

# Now run evaluation
print("\n\n")
from evaluate import run_all
from hh_reference import HHReference

hh = HHReference()
save_dir = os.path.join(os.path.dirname(p1.checkpoint_dir), "eval_plots")
run_all(model, hh, save_dir)
