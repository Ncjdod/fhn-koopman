"""
Configuration for HH Vector Field Learning.

Two-phase training:
  Phase 1: Distill analytical HH vector field into NN (pure regression)
  Phase 2: Fine-tune with Allen Brain data as boundary condition
"""

import os


class Phase1Config:
    """Hyperparameters for vector field distillation."""

    # --- Random seed ---
    seed = 42

    # --- Model architecture ---
    hidden_dim = 256
    n_layers = 4          # hidden layers
    activation = "tanh"
    n_fourier = 128       # Fourier feature pairs (input dim = 2*n_fourier)
    fourier_sigma = (1.0, 5.0)  # multi-scale: (low, high) frequency bands
    head_dim = 32         # gate output head width (m, h, n)
    v_head_dim = 64       # voltage output head width (larger for complex V dynamics)

    # --- Training ---
    n_epochs = 10000
    batch_size = 32768    # online-generated, reused for inner_steps
    inner_steps = 3       # gradient steps per batch (amortize data generation)
    lr = 1e-3
    lr_min = 1e-5         # cosine schedule floor
    weight_decay = 1e-4   # AdamW regularization

    # --- Sampling ---
    physiological_fraction = 0.85  # 15% uniform (covers edges without extreme outliers dominating)
    # State-space bounds
    V_range = (-100.0, 60.0)       # mV
    m_range = (0.0, 1.0)
    h_range = (0.0, 1.0)
    n_range = (0.0, 1.0)
    I_ext_range = (-10.0, 150.0)   # uA/cm^2
    # Physiological sampling distribution
    V_mean = -65.0
    V_std = 30.0
    gate_std = 0.15                # std around x_inf(V)

    # --- Logging ---
    log_every = 50
    plot_every = 500
    checkpoint_every = 1000
    val_every = 200                # integration validation

    # --- Paths ---
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints"
    )


class Phase2Config:
    """Hyperparameters for Allen Brain boundary condition fine-tuning."""

    # --- Random seed ---
    seed = 42

    # --- Training ---
    n_epochs = 2000
    model_lr = 1e-4       # smaller LR to preserve learned field
    latent_lr = 1e-2      # larger LR for latent gating variables
    conversion_lr = 1e-3  # unit conversion factor

    # --- Loss weights ---
    field_weight = 0.1    # anti-forgetting: keeps HH field loss in the mix
    gating_consistency_weight = 1.0
    smooth_weight = 0.01  # penalize jerky latent variables

    # --- Anti-forgetting sampling ---
    field_batch_size = 2048

    # --- Data preprocessing ---
    downsample = 20
    window_pre = 5.0      # ms before first spike
    window_post = 50.0    # ms after first spike
    dVdt_smooth_window = 5  # Savitzky-Golay polynomial window (odd int)
    dVdt_smooth_order = 3   # polynomial order

    # --- Unit conversion ---
    membrane_area_cm2_init = 2e-5  # ~2000 um^2 (typical cortical soma)

    # --- Logging ---
    log_every = 20
    plot_every = 200
    checkpoint_every = 500

    # --- Paths ---
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints"
    )
