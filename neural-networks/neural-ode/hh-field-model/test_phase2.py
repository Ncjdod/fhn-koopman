"""
Quick end-to-end test: Phase 1 (100 epochs) -> Phase 2 (50 epochs)
Just verifies everything runs without error.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Phase1Config, Phase2Config
from train_field import train_phase1
from train_boundary import train_phase2

# Phase 1: small run
p1 = Phase1Config()
p1.n_epochs = 100
p1.batch_size = 1024
p1.log_every = 25
p1.plot_every = 9999   # skip plots
p1.val_every = 50
p1.checkpoint_every = 50

print("=" * 60)
print("TEST: Phase 1 (100 epochs, batch=1024)")
print("=" * 60)
model, history = train_phase1(config=p1)

# Phase 2: small run
p2 = Phase2Config()
p2.n_epochs = 50
p2.log_every = 10
p2.plot_every = 9999
p2.checkpoint_every = 9999

print("\n\n")
print("=" * 60)
print("TEST: Phase 2 (50 epochs)")
print("=" * 60)
model2, latent, history2 = train_phase2(model=model, config_p1=p1, config_p2=p2)

if model2 is not None:
    print("\n\nTEST PASSED: Both phases ran successfully!")
else:
    print("\n\nTEST FAILED: Phase 2 returned None")
