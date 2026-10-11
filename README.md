# Temperature or Energy? Building Melting Physics into Graph Neural Networks for Melt-Pool Simulation

CS4350 group project: Jack Gulley, Satwik Miyyapuram, Luc Dop, Lei Wang, Giorgio Bernardini.

We compare graph neural networks that differ only in how much physics is built into them: none (GCN, MeshGraphNet),
linear heat diffusion (MAM-PhyGNN style), an enthalpy output with an exact material table, and exact heat conservation.
Baselines are a U-Net on a regular grid and linear graph filters. The data comes from our own verified solver for a laser
moving over a thin Ti-6Al-4V plate, with melting.

## Status

| Part | State |
|---|---|
| Solver (`solver/`) | Done, all six checks pass (four accuracy checks, cooling, solidification metric) |
| Data generator (`data/generate.py`) | Done, cools every run until fully solid; full run checked on every split |
| Graph preprocessing (`graphs/`) | Done: meshes, node data, PyTorch Geometric dataset; energy check passes (check 7) |
| U-Net grid, Laplacian, partitions, models, training, analysis | To do (`graphs/`, `models/`, `analysis/` list the planned modules) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy h5py pyyaml matplotlib jupyter     # enough for the solver, data and graph preprocessing
# for the models: install torch for your CUDA version (https://pytorch.org), then
pip install torch_geometric
```

## Quick start

All commands run from the repo root.

```bash
python -m verify.run_all                     # all seven checks (six solver, one graph), about 11 minutes
python -m verify.run_all --quick             # the four accuracy checks only, about 2 minutes
python -m data.generate --only week1         # 50 constant-property runs for pipeline debugging, a few minutes
python -m data.generate                      # all ~275 runs plus week1, under an hour, about 1.3 GB
python -m data.generate --limit 1            # one run per split, smoke test
python -m graphs.preprocess                  # every run onto its mesh, data/graphs/seed0/ (git-ignored)
python -m graphs.preprocess --splits test_interp --mesh-seed 1   # the new-mesh test
python -m verify.graph_energy --dataset      # graph energy check over every run in data/splits.json
python -m graphs.dataset --stride 5          # train normalisation statistics, sample counts per split
python -m graphs.figures                     # mesh and interpolation figures in results/figures/
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
- Output: temperature every 10 solver steps (13.3 us; 26.5 us for week1, whose constant properties allow a 2x larger
  time step). Melted fraction and enthalpy follow exactly from temperature
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
| Graph preprocessing keeps the energy (see Graphs) | `verify/graph_energy.py` | mesh vs solver energy and laser input < 1% | own runs at most 0.42%; full dataset (355 checks) at most 0.84%, laser at most 0.17% |

In the melting range, the 10 um data differs from the 5 um solution by about 7.4 K on average. This is the resolution
sensitivity of the 10 um data. It is not an error that every model must make: the models learn the 10 um solver, and
the main comparisons are made against it. On test_fine, show this solver-to-solver gap next to the model errors.
The checks also run on GitHub Actions when `solver/`, `verify/` or `data/generate.py` changes
(`.github/workflows/verify.yml`).

## Graphs

**Meshes** (`graphs/mesh.py`). One mesh per laser path, cached in `data/meshes/<path>_seed<k>.npz`. Commit these
files so every machine uses the same meshes. test_fine uses the straight mesh of test_interp, since only its ground
truth changes. The new-mesh test uses seed 1. With d the distance to the laser path:

- Fine band, d up to 120 um: a 10 um lattice, each node moved by up to 20% of the spacing per coordinate. The farthest
  solver cell that passes the solidus is 55 um from the path for a straight track at 195 W, 0.5 m/s, 75 um for a
  zigzag at 195 W, 0.65 m/s and 85 um for an L-turn at 170 W, 0.5 m/s (at the corner and the track end). Through the
  bilinear weights a node sees cell centres up to 14 um away, so the band leaves about 20 um of margin. Check 7
  confirms for every run that no node that passes the solidus lies outside the band.
- Transition: the target spacing grows as 10 um + 0.2 (d - 120 um) and reaches 50 um at 320 um. Points come from
  jittered lattices at 15, 22 and 33 um, cleaned where two lattices meet, and are then relaxed with a spring smoother
  (DistMesh style). Only these nodes move, by up to about 50 um. A few end up to 4 um inside the band.
- Coarse zone: a 50 um lattice with the same jitter.
- Plate edges: nodes exactly on the edges, moved along the edge only, with fixed corners. The triangles cover the
  plate exactly.

Triangles come from a Delaunay triangulation, and every triangle side is an edge in both directions. A node's area is
a third of the area of each adjacent triangle, so the areas sum to the plate area. Node volume is area times thickness.

| Mesh | Nodes (seed 0) | Smallest triangle angle, seeds 0 / 1 / 2 |
|---|---|---|
| straight, 2 x 1 mm | 5,172 | 23.2 / 25.3 / 23.1 deg |
| l_turn | 5,369 | 24.6 / 24.7 / 22.6 deg |
| zigzag | 8,371 | 23.4 / 21.8 / 24.6 deg |
| straight_large, 4 x 2 mm | 11,044 | 23.2 / 21.9 / 23.2 deg |

A 20% jittered lattice on its own gives 22 to 23 deg, so the transition does not produce the worst triangles.

**Node data** (`graphs/preprocess.py`). For every saved frame, the node temperature comes from the solver temperature
with the fixed bilinear weights of `solver/events.py`, the same weights the evaluation uses. The laser energy per volume
(J/m^3) is stored per frame interval: the sum of dt q(x_i, t_n) over the solver steps in the interval, with
t_n = n dt while t_n <= laser_off, exactly as the solver applies it. A model step of s frames sums s intervals, so the
stride is chosen when loading. Files go to `data/graphs/seed<k>/<run>.h5` (git-ignored). The node temperature stored
there is the ground truth for training and evaluation. Each file records the mesh parameters, a fingerprint of the
exact mesh (node positions and triangles) and a hash of its node data. A file that no longer fits the current mesh,
has the wrong array shapes or is older than its run file is rewritten by `graphs.preprocess` and refused by
`graphs.dataset` and `verify.graph_energy --processed`.

**Dataset** (`graphs/dataset.py`). `MeltPoolGraphs(split, stride)` gives one PyTorch Geometric sample per start frame.
Node features are temperature, melted fraction, laser energy of the step and a boundary flag. Edge features are
x_source - x_target and its length. The raw fields T, T_next, H, H_next, E and V and the index of each edge's reverse
(for model D's antisymmetric fluxes) come along. The stride is a whole number of saved frames, at least 1. Normalisation
statistics come from the train split only and are saved to `data/graphs/norm_stride<s>.json`, together with the mean
and spread of the targets dT and dH and a fingerprint of the training data. A dataset refuses statistics made from
other training data, another interpolation, another stride or another frame interval (week1 frames are twice as
long, so it needs its own statistics). Statistics are never recomputed silently, because a trained model must keep the
statistics it was trained with; `python -m graphs.dataset --stride <s> --recompute` rebuilds them.

**Energy check** (`verify/graph_energy.py`, check 7). For every frame it compares the mesh energy with the solver
energy. The difference is reported (a) relative to the laser input so far, from the frame where 5% of the run's input
has arrived, and (b) relative to the run's total input, over every frame. It also compares the laser input on the mesh
with the solver's per model step (5 and 2 frames). Results on the check's own runs, with temperature interpolation:

| Run | Split | (a) | (b) | Laser per step | Farthest melted node |
|---|---|---|---|---|---|
| straight 195 W, 0.50 m/s | test_extrap | 0.42% | 0.33% | 0.13% | 62 um |
| straight 50 W, 1.20 m/s | train | 0.11% | 0.10% | 0.11% | no melting |
| straight 120 W, 0.80 m/s | test_interp | 0.41% | 0.20% | 0.12% | 11 um |
| same run, mesh seed 1 | new mesh | 0.40% | 0.20% | 0.15% | 11 um |
| l_turn 170 W, 0.50 m/s | test_path | 0.35% | 0.25% | 0.16% | 82 um |
| zigzag 195 W, 0.65 m/s | test_path | 0.23% | 0.15% | 0.15% | 72 um |
| straight 150 W, 0.80 m/s, 5 um grid | test_fine | 0.35% | 0.24% | 0.12% | 22 um |
| straight_large 195 W, 0.65 m/s | test_large | 0.30% | 0.30% | 0.15% | 42 um |

Over the full dataset (every run in `data/splits.json`, and test_interp again on mesh seed 1), worst value per split:

| Split | Runs | (a) | (b) | Laser per step | Farthest melted node |
|---|---|---|---|---|---|
| week1 | 50 | 0.26% | 0.26% | 0.11% | no melting |
| train | 150 | 0.83% | 0.31% | 0.13% | 49 um |
| val | 20 | 0.75% | 0.30% | 0.13% | 42 um |
| test_interp | 30 | 0.84% | 0.31% | 0.12% | 52 um |
| test_interp, mesh seed 1 | 30 | 0.81% | 0.31% | 0.15% | 52 um |
| test_extrap | 15 | 0.56% | 0.33% | 0.13% | 60 um |
| test_path | 20 | 0.63% | 0.22% | 0.17% | 72 um |
| test_fine | 30 | 0.51% | 0.33% | 0.12% | 52 um |
| test_large | 10 | 0.67% | 0.29% | 0.16% | 32 um |

All 355 checks pass. "Laser per step" is the larger of the 5-frame and 2-frame values.

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

**Energy consistency of graph preprocessing.** Check 7 (see Graphs) compares, for every saved frame, the mesh
energy, the sum over nodes of rho d A_i H(T_i) with A_i the lumped node area, with the solver energy, the same sum
over grid cells. The target is below 1%, the same as the solver's energy check, and all 325 runs pass. There
are two sources of error. While the melt pool forms, bilinear temperature misses part of the latent heat at the melt
front. This error peaks at about a tenth of the run's input, at up to 0.84% of the input so far, and then fades.
Interpolating enthalpy (`--interp H`) cuts it to about 0.1%. Late in the run, both methods drift upwards by 0.1 to
0.35% of the total input, because the heat spreads into the graded and 50 um zones, where linear interpolation between
nodes overestimates a curved temperature field. Temperature interpolation stays the default, since it passes and the
solidification numbers above use bilinear temperature at nodes. The margin is small, so if a later change pushes the
check over 1%, rerun the preprocessing with `--interp H`. The laser input on the mesh is within 0.17% of the
solver's for every model step.
Model D conserves the energy of its own graph exactly. Its energy error against the interpolated solver data is not
zero by design, since it includes this preprocessing mismatch, so report the two together.

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
verify/      energy_balance.py, rosenthal.py, stefan.py, convergence.py, cooling.py, solidification.py,
             graph_energy.py, run_all.py
data/        generate.py, splits.json, meshes/ (commit), runs/ and graphs/ (git-ignored)
graphs/      mesh.py, preprocess.py, dataset.py, figures.py; planned: grid, laplacian, partition
models/      planned: linear filters, U-Net, GCN, MeshGraphNet backbone, models A-D
analysis/    planned: probes, attribution, spectra
configs/     one yaml per model and ablation
notebooks/   project.ipynb (course requirement)
results/     metrics.csv, figures/ (git-ignored)
```
