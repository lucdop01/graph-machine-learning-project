"""Rollout and evaluation harness (plan step 3).
metrics.py (numpy only: nRMSE, melting-range error, melt-pool IoU, solidification duration, energy), rollout.py
(predictor interface, rollout loop, per-run and split evaluation, selection score, reference predictors, size bins,
speedup benchmark, command line), torch_predictor.py (wraps a PyTorch Geometric model for rollouts).
Check 8 (verify/harness.py) tests the harness with the solver's own frames as predictions."""
