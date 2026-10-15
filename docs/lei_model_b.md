# Lei: Model B and mesh transfer

Aligned with **Melt-Pool GNN Plan (4)**, received 13 October 2026.

15 October integration note: main now includes Model A and the shared training
runner. This merge preserves the verified prototype; registering Model B and
adapting its YAML to that runner are still pending. References below to missing
Model A/training describe the original 13 October implementation snapshot.

## Deliverables and current status

- Implemented a mass-lumped cotangent diffusion operator, fixed linear filter,
  and Model B wrapper accepting the team's Model A as its correction network.
- Extended SampleBuilder with physical node area, cotangent weights and step dt;
  training and TorchPredictor rollouts use exactly the same input path.
- Added a preregistered three-seed gate and a mesh spectra command for the no-go branch.
- The shared Model A, training runner and trained checkpoints do not exist in
  the fetched repository. Therefore the prototype is verified, but the actual
  gate, full-data sweep and learned-model transfer results are **pending**.
  No synthetic check is a trained-model experimental result.

## Integration

```python
from models.model_b import ModelB
from evaluation.torch_predictor import TorchPredictor

# model_a is supplied by Luc; stats are train-only dataset.stats.
model = ModelB(correction=model_a, stats=stats)
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
the existing cross predictor's paired 10 µm vs 5 µm solver gap. In this repository
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
described in README. Config YAML is an integration specification until the
shared runner implements a config reader.

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
