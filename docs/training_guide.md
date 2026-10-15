# Melt-Pool GNN Unified Training & Terminology Guide

This guide documents the unified training harness, standardized terminology conventions, model architectures, configuration schema, and execution workflows for the melt-pool graph machine learning project.

---

## 1. Unified Terminology & Conventions

To prevent confusion across codebase contributions, configs, analysis notebooks, and reports, the following canonical terms and conventions are enforced across all modules:

| Canonical Term | Definition | Allowed Values / Schema | Prohibited Synonyms (Do Not Use) |
|---|---|---|---|
| **`target`** | The physical increment quantity predicted by the neural network | `"dT"`: Temperature increment $(T_{k+s} - T_k)$ [K]<br>`"dH"`: Specific enthalpy increment $(H_{k+s} - H_k)$ [J/kg] | `vars`, `predictands`, `outputs`, `targets`, `labels` |
| **`stats`** | Normalization dictionary computed from the training split (`data/graphs/norm_stride<s>.json`) | Dict containing keys:<br>- `stats["node"]`: mean/std for `T`, `phi`, `E`<br>- `stats["edge"]`: mean/std for `rel`, `dist`<br>- `stats["target"]`: mean/std for `dT` or `dH`<br>- `stats["train_fingerprint"]`: dataset hash | `norm_stats`, `statistics`, `vars`, `scalers` |
| **`stride`** | Cadence of saved solver frames spanned by a single model step ($s \ge 1$) | Default: `5` ($66.7\ \mu\text{s}$, primary model step)<br>Fine: `2` ($26.7\ \mu\text{s}$, E3 solidification step) | `step_size`, `frame_gap`, `skip`, `cadence` |
| **`device`** | Compute execution device for model tensors and training loops | Default: `"cpu"`<br>GPU: `"cuda"` | `hardware`, `accelerator`, `backend` |
| **`precision`** | Numerical floating point precision for training matrix products | Default: `"fp32"` (32-bit float, optimal for CPU)<br>GPU: `"bf16"` (bfloat16 mixed precision autocast) | `dtype`, `mode`, `mixed_prec` |
| **`noise`** | Gaussian noise standard deviation multiplier added to input temperatures during training | Non-negative float (e.g. `0.03` corresponds to $0.03 \times \sigma_T \approx 10.7\text{ K}$) | `jitter`, `temp_noise`, `sigma` |
| **`augment`** | Spatial rotational and reflection data augmentation | Boolean: `true` or `false` | `augmentation`, `rotate`, `transform` |

### Detailed Rules
1. **Target Normalization:**
   Loss is always computed on the normalized target:
   $$\mathcal{L} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \left( \frac{\Delta y_i - \mu_y}{\sigma_y} - \widehat{\Delta y}_i \right)^2$$
   where $y \in \{T, H\}$ as determined by the model's `target` attribute.
2. **Noise Injection:**
   During training, input node temperature $T_0$ is perturbed by $T \leftarrow T + \mathcal{N}(0, (\text{noise} \cdot \sigma_T)^2)$, where $\sigma_T \approx 358.1\text{ K}$ is the train set standard deviation. Phase fraction $\phi$ and specific enthalpy $H$ are re-derived from the noisy temperature, and the network is supervised to predict the transition to the uncorrupted future state $y_{k+s} - y_{\text{noisy}}$, teaching error correction across multi-step rollouts.
3. **Data Augmentation:**
   When `augment: true`, each training graph undergoes random uniform rotation $\mathbf{R} \in \text{SO}(2)$ and horizontal reflection ($y \to -y$) with probability 0.5. Node positions and relative edge displacement vectors $\mathbf{x}_i - \mathbf{x}_j$ are transformed accordingly. Node scalar features, edge distances, cell volumes, and target increments are strictly invariant.

---

## 2. Model Architecture Catalog

All architectures conform to a unified initialization interface and are registered in `train_harness.py`:

```python
model = build_model(model_cfg, stats=stats, stride=stride)
```

| Model Key | Class | Target | Parameters | Description | Primary Config |
|---|---|---|---|---|---|
| **`model_a`** | `ModelA` | `"dT"` | 2,331,905 | MeshGraphNet baseline with Encode-Process-Decode (Pfaff et al.). 15 processor blocks, latent width 128. | `configs/model_a.yaml` |
| **`model_b`** | `ModelB` | `"dT"` | 2,331,905 | Physics-guided linear heat diffusion residual. Combines analytic discrete heat diffusion with a neural correction predicting deviation $\Delta T - \Delta T_{\text{diff}}$. | `configs/model_b.yaml` |
| **`model_c`** | `ModelC` | `"dH"` | 2,331,905 | Enthalpy formulation predicting specific enthalpy change $\Delta H$. During rollout, mapped to $T$ via exact material inversion $T(H)$, preventing phase-change gradient stalls. | `configs/model_c.yaml` |
| **`model_d`** | `ModelD` | `"dH"` | 2,337,411 | Strictly conservative edge-flux GNN. Predicts antisymmetric edge heat fluxes $F_{ij} = -F_{ji}$. Guarantees exact $0.0$ net internal energy change bit for bit. | `configs/model_d.yaml` |
| **`gcn`** | `GCNModel` | `"dT"` | ~250,000 | Isotropic Graph Convolutional Network baseline with normalize edge weighting. | `configs/gcn.yaml` |
| **`rgcn`** | `RGCNModel` | `"dT"` | ~250,000 | Residual Graph Convolutional Network baseline with skip connections. | `configs/gcn.yaml` |
| **`linear_filter`** | `LinearFilter` | `"dT"` | 0 | Analytic discrete Laplace-Beltrami diffusion operator (parameter-free physical baseline). | N/A |

---

## 3. Configuration Schema

Every model run is governed by a YAML configuration file under `configs/`:

```yaml
model:
  name: model_a            # registered model key: model_a, model_b, model_c, model_d, gcn, rgcn
  width: 128               # latent dimension of node/edge MLPs
  layers: 15               # number of message passing processor blocks
  hidden_layers: 2         # hidden layers inside each MLP block

data:
  stride: 5                # frames per model step (5 frames = 66.7 us; 2 frames = 26.7 us)
  mesh_seed: 0             # jittered mesh seed (0 for canonical training)

training:
  steps: 100000            # total optimization steps
  batch_size: 4            # number of graphs per batch
  lr: 3.0e-4               # initial learning rate
  lr_end: 1.0e-6           # final decayed learning rate (exponential schedule)
  noise: 0.03              # temperature noise multiplier (fraction of sigma_T)
  seed: 0                  # random seed for model weights and dataset permutation
  val_every: 2000          # evaluation rollout interval (mean nRMSE on 20 val runs)
  log_every: 100           # metric streaming interval to stdout and log.csv
  num_workers: 0           # DataLoader workers (0 required on Windows and for exact resume)
  device: cpu              # default compute device: "cpu" (use "cuda" on GPU)
  precision: fp32          # numerical precision: "fp32" (use "bf16" on CUDA GPU)
  augment: true            # rotational SO(2) and reflection data augmentation
```

---

## 4. Training Workflows

Both entry points (`python -m models.train` and `python train_harness.py`) share identical configuration parsing, validation, and execution engines.

### 4.1 Standard Training (CPU Default)

Run on CPU using the configuration defaults:

```bash
# Model A baseline
python -m models.train --config configs/model_a.yaml --name model_a_seed0

# Model B physics-residual
python -m models.train --config configs/model_b.yaml --name model_b_seed0

# Model C enthalpy formulation
python -m models.train --config configs/model_c.yaml --name model_c_seed0

# Model D conservative edge fluxes
python -m models.train --config configs/model_d.yaml --name model_d_seed0

# GCN / RGCN baseline
python -m models.train --config configs/gcn.yaml --name gcn_seed0
```

### 4.2 GPU Acceleration

To train on GPU (NVIDIA CUDA or AMD ROCm), specify `--device cuda` and `--precision bf16`:

```bash
python -m models.train --config configs/model_a.yaml --device cuda --precision bf16 --name model_a_cuda
```

### 4.3 Hyperparameter Overrides

CLI flags override settings in the YAML configuration without modifying the file:

```bash
# Smoke run for 2,000 steps with custom learning rate
python -m models.train --config configs/model_a.yaml --steps 2000 --lr 1e-4 --name smoke_test

# Adjust noise or batch size
python -m models.train --config configs/model_a.yaml --noise 0.01 --steps 10000 --name test_low_noise
```

### 4.4 Resuming an Interrupted Run

If a training job is stopped, interrupted, or preempted, resume it cleanly:

```bash
python -m models.train --resume --name model_a_seed0
```

When `num_workers: 0` is set, state restoration (optimizer moments, RNG states, data permutation counters) resumes with **bit-for-bit exactness** on CPU.

### 4.5 Multi-Seed Evaluation

To obtain rigorous scientific benchmarks with mean $\pm$ standard deviation across independent initializations (e.g., seeds 0, 1, 2):

```bash
python train_harness.py --config configs/model_a.yaml --multi-seed --seeds 0 1 2
```

This generates `checkpoints/<model>_eval_seed0`, `_seed1`, `_seed2` and aggregates validation scores upon completion.

### 4.6 Hyperparameter Sweeps

To execute a fixed-budget sweep exploring learning rates and noise scales:

```bash
python train_harness.py --config configs/model_a.yaml --sweep --trials 6
```

Results are saved to `results/sweeps/<model_name>_sweep.csv` and rendered in a summary ranking table.

---

## 5. Evaluation & Rollouts

Model selection is strictly dictated by the **selection score**: mean rollout nRMSE across all 20 validation runs at the model's training stride.

### 5.1 Evaluating Checkpoints

```bash
# Evaluate on validation split (default device is CPU)
python -m evaluation.rollout --split val --checkpoint checkpoints/model_a_seed0/best.pt

# Evaluate on test splits
python -m evaluation.rollout --split test_interp --checkpoint checkpoints/model_a_seed0/best.pt
python -m evaluation.rollout --split test_extrap --checkpoint checkpoints/model_a_seed0/best.pt
python -m evaluation.rollout --split test_path --checkpoint checkpoints/model_a_seed0/best.pt
python -m evaluation.rollout --split test_fine --checkpoint checkpoints/model_a_seed0/best.pt
python -m evaluation.rollout --split test_large --checkpoint checkpoints/model_a_seed0/best.pt

# GPU-accelerated evaluation
python -m evaluation.rollout --split test_interp --checkpoint checkpoints/model_a_seed0/best.pt --device cuda
```

### 5.2 Key Evaluation Metrics

1. **`nrmse`**: Normalized root mean squared error against ground truth temperature:
   $$\text{nRMSE} = \frac{\sqrt{\sum_{i, k} (T_{i, k} - \hat{T}_{i, k})^2}}{\sqrt{\sum_{i, k} (T_{i, k} - 300\text{ K})^2}}$$
2. **`melt_mae`**: Mean absolute error specifically within the critical melting zone ($1,800\text{ K} \le T \le 2,000\text{ K}$).
3. **`iou` & `iou_soft`**: Melt pool intersection over union (binary $\phi > 0.5$ and continuous $\phi$).
4. **`sol_med` & `sol_med_abs`**: Median signed and absolute relative error in solidification duration.
5. **`energy_a` & `energy_b`**: Mesh energy conservation error relative to laser heat input:
   - `energy_a`: Peak deviation after 5% laser delivery.
   - `energy_b`: Peak deviation across entire run.
   - For Model D, net internal conduction energy is mathematically guaranteed to be $0.0$ bit for bit.
6. **`speedup`**: Ratio of solver simulation runtime to neural rollout runtime:
   ```bash
   python -m evaluation.rollout --benchmark --checkpoint checkpoints/model_a_seed0/best.pt
   ```

---

## 6. Verification & Quality Assurance

Run verification tests before submitting code or starting training runs:

```bash
# 1. Mock training test (verifies all architectures, forward/backward passes, checkpoint saving)
python -m verify.mock_train

# 2. Model B unit tests
python -m verify.model_b
python -m verify.model_b_training

# 3. Code formatting & linting
ruff check .
```
