"""Models (plan step 4 on).

Done:
  - backbone.py: MeshGraphNets encode-process-decode (shared by A, B, C, D)
  - model_a.py: MeshGraphNet baseline (target: dT)
  - model_b.py: Fixed linear heat diffusion + shared Model A correction (target: dT)
  - model_c.py: Enthalpy formulation + exact material table (target: dH)
  - model_d.py: Strictly conservative edge fluxes with exact zero internal energy change (target: dH)
  - gcn.py: Standard and residual Graph Convolutional Network baselines (target: dT)
  - linear_filter.py: Fixed discrete Laplacian baseline, and the area-lumped LF-T filter with its constrained Ridge
    fit and selected predictor (linear_filter_experiment.py, linear_filter_rollout.py)
  - train.py: Training, validation on rollouts, checkpoints, resume (python -m models.train)

Planned:
  - LF-H filter, unet.py. A-dual is model A with a second output head.

Needs torch and torch_geometric; CI (numpy only) never imports this package.
"""
