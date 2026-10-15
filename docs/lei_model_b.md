# Lei: Model B and mesh transfer

Report-ready method text and the checkpoint-free spectral figure are in
docs/lei_report_draft.md. Settings to confirm with teammates are in
docs/lei_team_alignment.md; no team message has been sent.

Aligned with **Melt-Pool GNN Plan (4)**, received 13 October 2026.

As of 15 October: Model B is registered in the shared trainer and checkpoint
loader. The YAML is executable and matches A's training settings. Local physics,
training, checkpoint and resume checks pass. The formal three-seed gate is
**pending**: this host has no CUDA/MPS device and no generated full dataset.
The user selected local verification plus runnable GPU experiment scripts.

## Deliverables and current status

- Fixed linear diffusion and source branch plus the exact shared Model A backbone.
- Same parameter count, training noise, rotation/reflection augmentation, schedule
  and validation selection criterion as A; train-only normalisation.
- One shared training loop, registered model construction and checkpoint loading.
- Three-seed gate runner, mesh-transfer runner, teacher-forced correction maps,
  and spectral response comparison with the exact homogeneous heat kernel.
- Gate numbers, trained-model interpretation and final report conclusions await
  real checkpoints. Local fixtures are not gate evidence.

## Integration

```python
from models.model_b import ModelB
from evaluation.torch_predictor import TorchPredictor

# stats are train-only dataset.stats; default correction is the shared ModelA.
model = ModelB(stats=stats)
predictor = TorchPredictor(model, stats, stride=5, target="dT", name="B")
```

The correction has exactly A's architecture and receives A's existing x and
edge_attr. It outputs one residual per node in training dT standard-deviation
units. An output of zero adds zero physical correction; do not centre the
residual a second time. Train against the original normalised total dT target.
All optimisation and checkpoint selection belong to the shared runner.

The fixed branch uses k=20 W/(m K), cp=600 J/(kg K), rho=4420 kg/m³;
alpha=k/(rho cp). These match week1 and deliberately omit phase change.
With positive stiffness K, diffusion is -alpha M^-1 K T. Signed cotangents
preserve the finite-element operator. Explicit substeps use a conservative
spectral bound; their count changes with dt and mesh size. The integrated laser
energy is divided evenly over these substeps. The NumPy fixed filter uses the
same discretisation and material constants. The learned correction can violate
conservation; Model B does not guarantee conserved total enthalpy.

Jojo owns the shared Laplacian/filter in the new plan. The modules provided here
are a working integration implementation to reconcile with that shared version
before training, rather than a second independently tuned baseline.

## Gate on 19 October

Before running: fix nRMSE as the primary metric. For each of val and test_interp
on mesh seed 1, B must improve over A by strictly more than
max(std(A), std(B)), using sample standard deviations across seeds 0, 1, 2.
Both domains must pass. Report melting-range error and LF alongside it; they
do not override this preregistered nRMSE rule. This is a chosen concrete rule
within the plan's nRMSE-or-melting-error criterion, not a statistical significance test.

Use stride 5, the same runs per domain, same A backbone and hyperparameters,
same short training budget, and checkpoints selected only on val selection_score.
Export CSV rows `model,seed,domain,nrmse`, model A/B/LF, domain val/new_mesh.
Each of the six groups requires all three seeds. Repeating deterministic LF
records comparison provenance; it is not a stochastic LF training result.

```bash
python -m analysis.model_b_gate results/gate.csv --out results/lei/gate.json
```

If go: equal sweep trials, best config on validation, final three-seed runs.
If no-go: retain fixed LF and run spectra analysis. In either case report all
numbers and the failed gate conditions; do not tune the rule after seeing results.

## Mesh-transfer study, 27 October–2 November

Compare each trained model on the same test_interp runs, graph seeds 0 and 1,
without fine-tuning or refitting normalisation. Compare test_fine errors with
the existing cross predictor's paired 10 µm vs 5 µm solver gap. This gap
is resolution context, not an error floor for the model. In this repository
test_fine refines the solver grid, not the GNN mesh. Describe the study as
new-mesh transfer and solver-resolution sensitivity. Solidification on test_fine
is diagnostic only; the plan permits quantitative mid-pool solidification at stride 2.

```bash
python -m evaluation.rollout --predictor laplacian --split val --stride 5
python -m evaluation.rollout --predictor laplacian --split test_interp --mesh-seed 1 --stride 5
python -m evaluation.rollout --predictor laplacian --split test_fine --stride 5
python -m evaluation.rollout --predictor cross --split test_fine --stride 5
python -m analysis.spectra --mesh straight --seeds 0 1
python -m verify.model_b
```

Rollout CLI requires generated/preprocessed data and train-only size bins as
described in README. configs/model_b.yaml now runs through models.train.

## Report method draft

Model B augments the shared message-passing model with a fixed linear heat
diffusion prediction on the triangular input mesh. We use a mass-lumped
finite-element stiffness operator and insulated boundaries, with diffusivity
fixed from the constant material properties. An identical learned backbone
predicts the remaining temperature increment. Geometry enters in physical
units, enabling evaluation on independently sampled meshes without refitting
normalisation. We select checkpoints by validation rollout nRMSE and compare
three training seeds. The decision rule is fixed before the gate experiments.

Results paragraph: fill A/B/LF means ± sample std on both gate domains, decision,
test_interp seed-0/seed-1 errors, test_fine errors and solver gap after actual runs.

## Validation performed locally

`python -m verify.model_b` passes the operator symmetry/PSD and constant-state
checks, integrated-source energy balance, NumPy/PyTorch parity on two graph
resolutions, batched step durations, normalisation, correction gradients and
gate pass/fail/missing-seed cases.

A separate real solver → RunGraph → SampleBuilder → ModelB → TorchPredictor
smoke uses one constant-property week1 run (100 W, 1 m/s, 20 µs cooling).
At stride 5 the PyTorch zero-correction step differs from the NumPy filter by
at most 0.00131 K. Its fixed-filter rollout nRMSE is 0.09467 and energy budget
residual is below 3.2e-15 relative to input (`results/lei/smoke.json`). These are
smoke diagnostics, not three-seed Model B performance results.

`results/lei/spectra.json` records the first eight mass-normalised eigenvalues
and the largest eigenvalue for straight graph seeds 0 and 1. Both have the
expected approximately zero constant mode. This is mesh spectrum groundwork;
learned-model spectral behaviour still needs checkpoints.

## Reproducible commands

Activate the repo virtual environment locally, or install requirements on the
GPU host. Run everything from the repository root. The formal gate runs A and B
from scratch with augmentation on, so it does not reuse the old unaugmented A
results. Default budget is 20,000 steps per seed, seeds 0/1/2, stride 5.

```bash
python -m verify.model_b
python -m verify.model_b_training
python -m analysis.run_model_b_gate --prepare-data
python -m analysis.run_model_b_gate --check
python -m analysis.run_model_b_gate --train
# Continue interrupted jobs using the same settings:
python -m analysis.run_model_b_gate --train --resume
# Re-score completed jobs without retraining:
python -m analysis.run_model_b_gate
```

Data preparation regenerates the full train (150), val (20), test_interp (30)
splits and preprocesses the new mesh. It never uses --limit. Existing training
statistics are checked, never silently replaced. If a team dataset already
exists, skip preparation and run --check. A CPU smoke is explicitly available
through verify.model_b_training; changing the architecture/steps for that test
must not replace the formal gate.

Outputs under results/lei/gate include scores.csv, decision.json, report.md,
per-run metrics, configs and a manifest with checkpoint hashes and data
fingerprint. The runner refuses incomplete or incompatible checkpoints. --check
is read-only and diagnoses missing data without starting training. LF is
our fixed branch with no fitted parameters; reconcile it with Jojo's version
before a team comparison. The extra least-squares LF-T/LF-H baselines mentioned
in TA Feedback are team scope, not implemented here.

After full training, interpretation and transfer use actual checkpoint paths:

```bash
python -m analysis.model_b_corrections --checkpoint checkpoints/b_gate_20k_aug_b_seed0/best.pt --device cuda
python -m analysis.spectra --mesh straight --seeds 0 1 --out results/lei/spectral_response.json
python -m analysis.plot_model_b_spectra
python -m data.generate --only test_fine
python -m graphs.preprocess --splits test_fine
python -m analysis.mesh_transfer --checkpoints checkpoints/b_gate_20k_aug_a_seed0/best.pt checkpoints/b_gate_20k_aug_b_seed0/best.pt --device cuda
```

Repeat transfer for all three final seeds and every available team model. The
transfer tool supports every checkpoint registered in the shared trainer; C/D
must first be merged/registered by their owners. Speedup uses the existing
benchmark CLI separately, on the same hardware. No-go spectra compare actual
fixed-filter response against exp(-alpha dt lambda); zero mismatch would not
prove a linear model captures phase change. Correction maps sample ground-truth
input states and measure area-weighted absolute learned increments, so they are
interpretation diagnostics rather than autoregressive evaluation scores.

The training integration check uses a single physical simulation, width 16,
two layers and four CPU steps. It exercises the existing shared trainer with
noise and augmentation, then simulates interruption at step 2 and checks the
resumed step-4 weights bit for bit against uninterrupted training. It also
checks the A/B parameter counts, paired initial weights and rotation-invariant
physics. See results/lei/training_smoke.json; its score is not a gate result.
