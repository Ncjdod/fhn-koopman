# FitzHugh-Nagumo with Koopman Operators

This project models neural membrane excitability dynamics and dynamic mode decomposition (DMD) / Koopman operator representations of FHN time series.

* **Method**: Adaptive ODE integrations via Diffrax with parameter optimization via Optax (JAX), coupled with Dynamic Mode Decomposition (DMD) and Hankel-alternative Koopman analysis.
* **Results**: Computes trajectories and reconstructs dynamical phase states. Generated result plots can be found under `plots/`.
