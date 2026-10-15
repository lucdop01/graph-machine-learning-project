"""Models (plan step 4 on).
Done: backbone.py (MeshGraphNets encode-process-decode, shared by A-D), model_a.py (MeshGraphNet, predicts dT),
model_b.py (fixed linear diffusion + shared Model A correction), linear_filter.py (fixed baseline),
train.py (training, validation on rollouts, checkpoints, resume; python -m models.train).
Planned: fitted LF-T/LF-H filters, unet.py, gcn.py (edge-weighted), model_c.py (enthalpy + exact table), model_d.py (conservative edge fluxes). A-dual is model A with a
second output head.
Needs torch and torch_geometric; CI (numpy only) never imports this package."""
