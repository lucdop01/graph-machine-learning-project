# Temperature or Energy? Building Melting Physics into Graph Neural Networks for Melt-Pool Simulation

CS4350 group project: Jack Gulley, Satwik Miyyapuram, Luc Dop, Lei Wang.

We compare graph neural networks that differ only in how much physics is built into them: none (GCN, MeshGraphNet),
linear heat diffusion (MAM-PhyGNN style), an enthalpy output with an exact material table, and exact heat conservation.
Baselines are a U-Net on a regular grid and linear graph filters. The data comes from our own verified solver for a laser
moving over a thin Ti-6Al-4V plate, with melting.

## Status

| Part | State |
|---|---|
| Solver (`solver/`) | Done, all six checks pass (four accuracy checks, cooling, solidification metric) |
| Data generator (`data/generate.py`) | Done, cools every run until fully solid; full run checked on every split |
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
python -m verify.run_all                     # all six solver checks, about 9 minutes
python -m verify.run_all --quick             # the four accuracy checks only, about 2 minutes
python -m data.generate --only week1         # 50 constant-property runs for pipeline debugging, a few minutes
python -m data.generate                      # all ~275 runs plus week1, under an hour, about 1.3 GB
python -m data.generate --limit 1            # one run per split, smoke test
```

Generated runs go to `data/runs/` (git-ignored) and the index to `data/splits.json` (commit this one).
The data is fully determined by the seeds in `generate.py`; host a copy of `data/runs/` outside GitHub and link it here.

## Solver

2D thin plate (2 mm x 1 mm, 100 um thick) of Ti-6Al-4V, heat conduction with melting in enthalpy form,
explicit finite differences on a 10 um grid, insulated edges, Gaussian laser (r = 50 um, absorptivity 0.35)
moving along a polyline. After the laser switches off, every run except week1 keeps cooling until the whole plate is
below the solidus: at least 0.2 ms, at most 5 ms. The slowest run below boiling needs 1.9 ms (zigzag, 195 W, 0.65 m/s).
A run still liquid at 5 ms would be kept but flagged incomplete; none is. week1 keeps a fixed 0.2 ms and has no
completion flag, since it uses constant properties without melting.

- Material data: temperature-dependent k and c_p from *Numerical study to analyze the influence of process parameters on
  temperature and stress field in powder bed fusion of Ti-6Al-4V*, Materials 18(2), 368 (2025). Melting between
  1,878 and 1,928 K with latent heat 286 kJ/kg. Properties are held constant above the liquidus; density is constant.
- Scope: conduction only. No liquid flow, evaporation or surface heat loss. Runs that reach 3,660 K (boiling) are rejected.
- Output: temperature every 10 solver steps (13.3 us). Melted fraction and enthalpy follow exactly from temperature
  (`solver/ti64.py`). The last saved frame is the final state. Each run file also stores `cooling_complete`,
  `t_cool_actual` (cooling time actually simulated), `t_end`, `terminal_frame` (index of the final frame),
  `terminal_saved`, `n_steps` and `T_max`. `splits.json` repeats `complete`, `t_cool` and `frames` per run.
- Event recording: `simulate(..., events=points)` records melting and solidification times at every solver step at
  given points (`solver/events.py`). The generator does not store events; see the evaluation notes.

**Convention for model inputs.** The solver applies the laser at the start of each solver step. Build the per-node
laser energy for a model step by summing the Gaussian source at those same solver step times.

## Verification

| Check | Script | Criterion | Result |
|---|---|---|---|
| Energy balance | `verify/energy_balance.py` | error < 1% | +0.01% to +0.04% on 3 runs |
| Moving heat source vs Rosenthal (Gaussian spot) | `verify/rosenthal.py` | < 2% of temperature rise | <= 0.5% |
| Melting front vs Neumann's Stefan solution | `verify/stefan.py` | < 2% after 20 cells | 0.26% |
| Grid (20, 10, 5 um) and time-step refinement | `verify/convergence.py` | 10 vs 5 um peak T < 2% | 0.29%; halving the time step changes little |
| Cooling until solid, every path and split | `verify/cooling.py` | complete within half the 5 ms cap | slowest 1.86 ms; one zigzag corner boils and is rejected |
| Solidification duration: grid, time step, output cadence | `verify/solidification.py` | halving the time step < 1% | 0.14%; grid and cadence reported below |

In the melting range, the 10 um data differs from the 5 um solution by about 7.4 K on average. This is the resolution
sensitivity of the 10 um data. It is not an error that every model must make: the models learn the 10 um solver, and
the main comparisons are made against it. On test_fine, show this solver-to-solver gap next to the model errors.
The checks also run on GitHub Actions when `solver/`, `verify/` or `data/generate.py` changes
(`.github/workflows/verify.yml`).

## Evaluation notes

These change or sharpen the evaluation plan in the proposal.

**Solidification duration.** Defined per point in `solver/events.py`: the time from the last fall through the liquidus
(1,928 K) to the next fall through the solidus (1,878 K). Between two states the temperature is taken as linear in
time. Points that only partly melt (peak between 1,878 and 1,928 K) have no duration; they count in melt-pool IoU
through the melted fraction. A point that melts again uses its last solidification. A later reheat that stays below
the liquidus does not change it. Remelting happens on zigzags (136 of 3,882 fully melted points at 190 W, 0.66 m/s).

**Evaluate at mesh nodes, not solver cells.** The solver's own cells step through the melting range one at a time, so
per-cell durations depend on the cell size. At 150 W and 0.65 m/s, the median per-cell duration is 74.1 us on the
10 um grid and 58.4 us on the 5 um grid. At jittered points with bilinear weights, which is what the mesh nodes get,
the medians are much closer (`verify/solidification.py`).

| Change | 195 W, 0.5 m/s | 150 W, 0.65 m/s | 120 W, 0.65 m/s | zigzag 190 W, 0.66 m/s |
|---|---|---|---|---|
| Reference median duration (10 um, every step) | 118.6 us | 43.6 us | 39.8 us | 203.0 us |
| Time step halved | +0.14% | +0.14% | +0.01% | |
| Grid 5 um | +2.5% | +6.6% | +0.8% | |
| Grid 20 um | +28% | +38% | +32% | |
| Sampled every 13.3 us (saved frames) | +0.0% | +0.9% | -0.3% | +0.2% |
| Sampled every 26.7 us (E3 small step) | +0.1% | +1.8% | -4.7% | +0.2% |
| Sampled every 66.7 us (model step) | +0.0% | -2.7% | -18% | +0.4% |
| Sampled every 133 us (E3 large step) | -2.9% | -27% | -36% | -2.8% |
| Fully melted points missed at the model step | 7 of 1,403 | 53 of 796 | 15 of 337 | 8 of 3,882 |

Rows show the change of the median duration against the reference, over points where both define one. Per point,
at the model step the median absolute difference is 2.7%, 13.8%, 17.6% and 1.4%. "Missed" points melt in the solver
but never show a temperature above the liquidus at the sampled times, so no interpolation can recover them.

**Consequences for the plan.**

1. Score solidification duration for models against the solver sampled at the model's own output times, with the same
   nodes, bilinear weights and event rules. A perfect model then scores zero. Report the sampling error above next to it.
2. Pools whose median solidification takes less than one model step are not resolved at 66.7 us. This includes
   150 W at 0.65 m/s and every smaller pool. The 26.7 us step of E3 resolves them (median within 4.7%, at most 1% of points missed). E3 at 26.7 us for
   models A and D is therefore required if the report makes a claim about solidification. E3 at 133 us stays optional
   and is not scored on solidification duration.
3. test_fine keeps solidification duration as a diagnostic. Next to the model numbers, report the gap between the 10 um
   and 5 um solvers, computed on the same nodes, at the same times and with the same event rules. Same-grid metrics
   stay the primary comparison.
4. Reference events at every solver step are only needed to measure sampling error. Saved frames are within 1% of them
   in the median (row "13.3 us"). If needed at mesh nodes, rerun the runs with `events=<node coordinates>`.

**Energy consistency of graph preprocessing (check when `graphs/` exists).** Bilinear interpolation of temperature
does not preserve total enthalpy, mostly because enthalpy jumps by the latent heat across the melting range. For
every saved frame, compare the mesh energy, the sum over nodes of rho d V_i H(T_i) with V_i the lumped node area, with
the solver energy, the same sum over grid cells. Report the largest difference relative to the cumulative laser
input. Target: below 1%, the same as the solver's energy check. If it fails, interpolate enthalpy instead of
temperature, or use a conservative remap. Model D conserves the energy of its own graph exactly. Its energy error
against the interpolated solver data is not zero by design: it includes this preprocessing mismatch, so report the
two together.

**Boiling.** The boiling guard stays. Straight tracks never boil (hottest 3,415 K at 195 W, 0.5 m/s). Zigzags do:
170 W at 0.5 m/s reaches 3,785 K and is rejected, and 195 W at 0.65 m/s reaches 3,628 K.

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
solver/      ti64.py (material table), paths.py (laser paths), solver.py, events.py (melting and solidification events)
verify/      energy_balance.py, rosenthal.py, stefan.py, convergence.py, cooling.py, solidification.py, run_all.py
data/        generate.py, splits.json, runs/ (git-ignored)
graphs/      planned: mesh, grid, laplacian, partition, dataset
models/      planned: linear filters, U-Net, GCN, MeshGraphNet backbone, models A-D
analysis/    planned: probes, attribution, spectra
configs/     one yaml per model and ablation
notebooks/   project.ipynb (course requirement)
results/     metrics.csv, figures/ (git-ignored)
```
