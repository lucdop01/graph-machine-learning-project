# Temperature or Energy? Building Melting Physics into Graph Neural Networks for Melt-Pool Simulation

CS4350 group project: Jack Gulley, Satwik Miyyapuram, Luc Dop, Lei Wang.

We compare graph neural networks that differ only in how much physics is built into them: none (GCN, MeshGraphNet),
linear heat diffusion (MAM-PhyGNN style), an enthalpy output with an exact material table, and exact heat conservation.
Baselines are a U-Net on a regular grid and linear graph filters. The data comes from our own verified solver for a laser
moving over a thin Ti-6Al-4V plate, with melting.

## Status

| Part | State |
|---|---|
| Solver (`solver/`) | Done, all four verification checks pass |
| Data generator (`data/generate.py`) | Done, smoke-tested on every split |
| Graphs, models, training, analysis | To do (`graphs/`, `models/`, `analysis/` list the planned modules) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy h5py pyyaml matplotlib jupyter     # enough for the solver and data
# for the models: install torch for your CUDA version (https://pytorch.org), then
pip install torch_geometric
```

## Quick start

All commands run from the repo root.

```bash
python -m verify.run_all                     # all four solver checks, about 2 minutes
python -m data.generate --only week1         # 50 constant-property runs for pipeline debugging, a few minutes
python -m data.generate                      # all ~275 runs, under an hour, about 1.7 GB
python -m data.generate --limit 1            # one run per split, smoke test
```

Generated runs go to `data/runs/` (git-ignored) and the index to `data/splits.json` (commit this one).
The data is fully determined by the seeds in `generate.py`; host a copy of `data/runs/` outside GitHub and link it here.

## Solver

2D thin plate (2 mm x 1 mm, 100 um thick) of Ti-6Al-4V, heat conduction with melting in enthalpy form,
explicit finite differences on a 10 um grid, insulated edges, Gaussian laser (r = 50 um, absorptivity 0.35)
moving along a polyline, 0.2 ms of cooling after the laser switches off.

- Material data: temperature-dependent k and c_p from *Numerical study to analyze the influence of process parameters on
  temperature and stress field in powder bed fusion of Ti-6Al-4V*, Materials 18(2), 368 (2025). Melting between
  1,878 and 1,928 K with latent heat 286 kJ/kg. Properties are held constant above the liquidus; density is constant.
- Scope: conduction only. No liquid flow, evaporation or surface heat loss. Runs that reach 3,660 K (boiling) are rejected.
- Output: temperature every 10 solver steps (13.3 us). Melted fraction and enthalpy follow exactly from temperature
  (`solver/ti64.py`).

**Convention for model inputs.** The solver applies the laser at the start of each solver step. Build the per-node
laser energy for a model step by summing the Gaussian source at those same solver step times.

## Verification

| Check | Script | Criterion | Result |
|---|---|---|---|
| Energy balance | `verify/energy_balance.py` | error < 1% | +0.01% to +0.04% on 3 runs |
| Moving heat source vs Rosenthal (Gaussian spot) | `verify/rosenthal.py` | < 2% of temperature rise | <= 0.3% |
| Melting front vs Neumann's Stefan solution | `verify/stefan.py` | < 2% after 20 cells | 0.26% |
| Grid (20, 10, 5 um) and time-step refinement | `verify/convergence.py` | 10 vs 5 um peak T < 2% | 0.29%; halving the time step changes little |

The 10 um data differs from the 5 um solution by about 7.4 K on average in the melting range. Treat differences between
models below that on the fine-grid test as noise. The checks also run on GitHub Actions when `solver/` or `verify/`
changes (`.github/workflows/verify.yml`).

## Data splits

| Split | Runs | Purpose |
|---|---|---|
| week1 | 50 | constant properties, no melting; pipeline debugging only |
| train / val | 150 / 20 | straight tracks, 50-195 W, 0.5-1.2 m/s, hot corner held out |
| test_interp | 30 | unseen settings in range |
| test_extrap | 15 | 170-195 W at 0.50-0.65 m/s, the largest melt pools |
| test_path | 20 | L-turns and zigzags, never seen in training |
| test_fine | 30 | test_interp settings on a 5 um grid; only the ground truth changes |
| test_large | 10 | 4 mm x 2 mm plate, for transfer and partitioned rollouts |

New-mesh tests reuse test_interp runs on a differently jittered mesh, so they need no extra solver runs.

## Layout

```
solver/      ti64.py (material table), paths.py (laser paths), solver.py
verify/      energy_balance.py, rosenthal.py, stefan.py, convergence.py, run_all.py
data/        generate.py, splits.json, runs/ (git-ignored)
graphs/      planned: mesh, grid, laplacian, partition, dataset
models/      planned: linear filters, U-Net, GCN, MeshGraphNet backbone, models A-D
analysis/    planned: probes, attribution, spectra
configs/     one yaml per model and ablation
notebooks/   project.ipynb (course requirement)
results/     metrics.csv, figures/ (git-ignored)
```
