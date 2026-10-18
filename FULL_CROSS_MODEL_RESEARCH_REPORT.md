# Comprehensive Benchmark & Graph ML Research Report: All Models, 2D Solvers & Node-Level Visualizations

**Course**: CS4350 Graph Machine Learning — TU Delft  
**Author**: Satwik Miyyapuram  
**Date**: October 18, 2026  
**Repository Branch**: [`satwik/model-b-d-training`](file:///scratch/smiyyapuram/graph-machine-learning-project) (Merged into [`origin/main`](file:///scratch/smiyyapuram/graph-machine-learning-project) via PR #5)  
**Hardware & Checkpoint Bundle**: 191 MB archive at [`models_b_d_checkpoints_and_report.zip`](file:///scratch/smiyyapuram/graph-machine-learning-project/models_b_d_checkpoints_and_report.zip)  

---

## 1. Executive Summary & Master Leaderboard

We evaluate **9 distinct methodologies** for simulating melt-pool thermodynamics during laser powder bed fusion (LPBF) additive manufacturing of Ti-6Al-4V alloy. The models span non-learned physical baselines, regular-grid 2D convolutional networks (CNNs), and four Graph Neural Network (GNN) architectures designed with contrasting physical inductive biases.

### Comprehensive Cross-Model Leaderboard

All models are evaluated under identical closed-loop autoregressive rollouts with temporal stride $s = 5$ frames ($66.7\ \mu\text{s}$ per step) on the held-out validation set (`val`, 20 diverse tracks, $P \in [50, 195]\text{ W}$, $v \in [0.5, 1.2]\text{ m/s}$):

| Rank | Model / Baseline | Paradigm / Method | Target | Parameters | Validation nRMSE | Melt MAE (1800–2000 K) | Energy Conservation Drift | Rollout Speed | Checkpoint Location / Source |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| — | **Oracle** | Exact Finite-Difference Replay | — | 0 | **`0.00000`** | `0.00 K` | $0.000\%$ | — | Ground truth simulation files |
| **1** | **Model B** | **Spectral-Spatial Hybrid GNN** | $\Delta T$ | 2,331,905 | **`0.00289 ± 0.00030`** | **`3.82 K`** | $+0.28\%$ drift | ~18 ms/step | [`checkpoints/model_b_seed{0,1,2}/best.pt`](file:///scratch/smiyyapuram/graph-machine-learning-project/checkpoints/) |
| **2** | **Model A** | Pure Spatial MPNN (MeshGraphNets) | $\Delta T$ | 2,331,905 | **`0.00364 ± 0.00006`** | `4.34 K` | $+0.88\%$ drift | ~18 ms/step | [`checkpoints/model_a_seed0/best.pt`](file:///scratch/smiyyapuram/graph-machine-learning-project/checkpoints/) |
| **3** | **Model C** | Direct Enthalpy GNN + $T(H)$ Inversion | $\Delta H$ | 2,331,905 | **`0.00487 ± 0.00093`** | `5.12 K` | $+0.42\%$ drift | ~19 ms/step | Branch [`origin/jack/model-c-full-runs`](file:///scratch/smiyyapuram/graph-machine-learning-project) |
| — | *Cross-Solver (10 vs 5 $\mu\text{m}$)* | *Physical Discretization Gap* | $T$ | *0* | *`0.00507`* | *`4.61 K`* | *+0.02%* | — | 2D FD solver resolution gap |
| **4** | **Model D** | **Skew-Symmetric Conservative Flux GNN** | $\Delta H$ | 2,365,058 | **`0.00616 ± 0.00010`** | `6.24 K` | **`0.000000 J` (Exact)** | ~21 ms/step | [`checkpoints/model_d_seed{0,1,2}/best.pt`](file:///scratch/smiyyapuram/graph-machine-learning-project/checkpoints/) |
| **5** | **U-Net (Grid)** | 2D Euclidean CNN (5 $\mu\text{m}$ Raster) | $\Delta T$ | 488,705 | **`0.00702`** | `8.09 K` | $-0.27\%$ drift | ~13 ms/step | Branch [`origin/u-net`](file:///scratch/smiyyapuram/graph-machine-learning-project) |
| **6** | **Linear Filter (LF-T)** | Discrete Cotangent Laplacian (No ML) | $\Delta T$ | 0 | **`0.34033`** | `142.5 K` | $-0.20\%$ drift | ~2 ms/step | Branch [`origin/laplacian-filter-baseline`](file:///scratch/smiyyapuram/graph-machine-learning-project) |
| **7** | **Persistence** | Trivial Frozen State ($T_{t+1} = T_t$) | $T$ | 0 | **`1.00000`** | `1,620 K` | $-100.0\%$ | 0 ms/step | Cold start baseline |

---

## 2. Inventory: 2D Numerical Solver & Available Team Checkpoints

### A. The 2D Numerical Simulation Solver (`solver/solver.py`)
* **Mechanism**: Solves the 2D non-linear Stefan heat conduction problem on a uniform Cartesian staggered grid with temperature-dependent conductivity $k(T)$, specific heat $c_p(T)$, and latent heat of fusion ($L = 286\text{ kJ/kg}$) using the enthalpy formulation.
* **Discretizations**:
  1. **Standard Resolution**: $10\ \mu\text{m}$ grid cells ($200 \times 100$ cells for $2 \times 1\text{ mm}$ domain).
  2. **Fine Resolution**: $5\ \mu\text{m}$ grid cells ($400 \times 200$ cells, 4x computational cost).
* **The Physical Resolution Bound**: The difference between the $10\ \mu\text{m}$ solver and the $5\ \mu\text{m}$ solver is **`0.00507` nRMSE** (Melt MAE $4.61\text{ K}$). **Model B (`0.00289`) and Model A (`0.00364`) achieve errors smaller than this physical discretization gap.**

### B. 2D Euclidean CNN Baseline: U-Net (`origin/u-net`, Giorgio Bernardini)
* **Architecture**: 4-level U-Net encoder-decoder with skip connections operating on rasterized $5\ \mu\text{m}$ grids ($400 \times 200 = 80,000$ pixels).
* **Available Checkpoints**:
  * `checkpoints/unet_early_stop_retry_group_s5_seed0.pt` (Best validation checkpoint, 488,705 parameters).
  * `checkpoints/unet_full_group_s5_seed0.pt`
* **Generalization Failure**: While U-Net achieves `0.00702` on in-distribution validation, it suffers **catastrophic collapse on out-of-distribution geometry**:
  * On larger plates (`test_large`, $4 \times 2\text{ mm}$), U-Net error explodes to **`0.34978` nRMSE** with a **$+118.16\%$ energy accumulation** (unbounded blowup).
  * On unseen laser paths (`test_path`, zigzags & L-turns), error jumps to **`0.08160` nRMSE** (Melt MAE $79.5\text{ K}$).
  * **Core Reason**: Standard 2D convolution kernels are translation-equivariant on uniform Euclidean grids but lack inductive biases for unstructured metric spaces, variable graph boundaries, and coordinate-free relative distances.

### C. Direct Enthalpy GNN: Model C (`origin/jack/model-c-full-runs`, Jack Gulley)
* **Architecture**: 15-block EncodeProcessDecode MPNN predicting extensive specific enthalpy change $\Delta H$, inverted via thermodynamic table $T(H)$.
* **Available Checkpoints & Results**:
  * Validated across 3 seeds (`results/eval/model_c_seed*.json`): Seed 0 (`0.00412`), Seed 1 (`0.00594`), Seed 2 (`0.00455`), aggregate **`0.00487 ± 0.00093`**.
  * Shows that direct enthalpy prediction stabilizes phase transitions but suffers from gradient discontinuity at the mushy zone boundaries ($T_S = 1878\text{ K}, T_L = 1928\text{ K}$).

### D. Linear Cotangent Laplacian Filter: LF-T (`origin/laplacian-filter-baseline`, Giorgio Bernardini)
* **Mechanism**: Explicit integration of the mass-lumped cotangent Laplace-Beltrami operator ($\mathbf{M}^{-1}\mathbf{K}$) with no learnable parameters.
* **Results**: Validation nRMSE **`0.34033`**, proving that pure linear heat diffusion completely fails in the presence of latent heat phase transitions.

### E. Model B & Model D 3-Seed Checkpoints (`satwik/model-b-d-training`, Satwik Miyyapuram)
* **All 6 checkpoints packaged in [`models_b_d_checkpoints_and_report.zip`](file:///scratch/smiyyapuram/graph-machine-learning-project/models_b_d_checkpoints_and_report.zip)**:
  * `checkpoints/model_b_seed0/{best,last}.pt` (Best step 88k, nRMSE `0.00312`)
  * `checkpoints/model_b_seed1/{best,last}.pt` (Best step 92k, nRMSE **`0.00256`** — Project Record)
  * `checkpoints/model_b_seed2/{best,last}.pt` (Best step 98k, nRMSE `0.00300`)
  * `checkpoints/model_d_seed0/{best,last}.pt` (Best step 96k, nRMSE **`0.00609`** — Exact $0.000000\text{ J}$ drift)
  * `checkpoints/model_d_seed1/{best,last}.pt` (Best step 74k, nRMSE `0.00628`)
  * `checkpoints/model_d_seed2/{best,last}.pt` (Best step 84k, nRMSE `0.00613`)

---

## 3. Node-Level Graph Visualizations & Cross-Model Comparison

We generated a specialized 9-panel, publication-grade visualization showing the **adaptive graph mesh, discrete graph nodes, message-passing edges, and thermal fields**:

![Graph Nodes and Heat Prediction Comparison](figures/graph_nodes_heat_comparison.png)
*(High-resolution figure saved at `docs/figures/graph_nodes_heat_comparison.png`)*

### Detailed Panel Breakdown

1. **Panel (A) — Adaptive Graph Mesh Discretization**:
   * Shows the full $2.0 \times 1.0\text{ mm}$ domain containing **$N = 5,172$ nodes** and **$E = 30,552$ directed edges**.
   * Color-coded by mesh zones: **Fine Band** ($h = 10\ \mu\text{m}$, green), **Transition Zone** ($h = 15\text{–}33\ \mu\text{m}$, orange), and **Coarse Bulk** ($h = 50\ \mu\text{m}$, purple).
   * Highlights the drastic node economy: only 5,172 nodes compared to 80,000 pixels in the $5\ \mu\text{m}$ U-Net grid (15.5x fewer degrees of freedom).
2. **Panel (B) — Close-Up Graph Topology & Message-Passing Channels**:
   * Zoomed into the $320 \times 160\ \mu\text{m}$ Region of Interest (ROI) around the laser focus.
   * Explicitly displays **individual graph vertices** as discrete blue markers connected by Delaunay triangulation edge channels, along which edge messages $\mathbf{m}_{ij}$ are exchanged.
3. **Panel (C) — Discrete Node Temperatures (Ground Truth)**:
   * Every graph node is plotted as an individual scatter dot colored by its actual temperature ($T_{\text{node}} \in [300, 2200]\text{ K}$), overlaid with solidus ($T_S = 1878\text{ K}$, cyan) and liquidus ($T_L = 1928\text{ K}$, white) phase contours.
   * Demonstrates that the GNN directly models discrete non-Euclidean vertex states rather than continuous pixel arrays.
4. **Panels (D, E, F) — Thermal Field Predictions (GT vs. Model B vs. Model D)**:
   * **Ground Truth (D)**: Crisp teardrop melt pool with maximum temperature $T_{\max} = 2174\text{ K}$.
   * **Model B (E)**: Perfectly captures the elongated trailing thermal tail and the steep advancing temperature front (nRMSE `0.00289`).
   * **Model D (F)**: Captures the full melt-pool geometry and solidus envelope with zero long-term energy accumulation (nRMSE `0.00616`).
5. **Panels (G, H) — Absolute Local Error Maps ($|T_{\text{pred}} - T_{\text{true}}|$):**
   * **Model B (G)**: Maximum local error inside the melt-pool ROI is restricted to **$< 15\text{ K}$** (under $0.7\%$ relative error).
   * **Model D (H)**: Maximum local error is bounded below **$< 24\text{ K}$**, localized tightly around the mushy zone boundary where edge flux discretization interacts with the non-linear enthalpy inversion $T(H)$.
6. **Panel (I) — Centerline Profile with Discrete Node Markers**:
   * Slices along the laser track ($y \approx 500\ \mu\text{m}$), explicitly plotting the **discrete graph nodes** as distinct markers (circles for GT, triangles for Model B, squares for Model D).
   * Shows that both models track the non-linear latent heat plateau and steep Gaussian temperature rise across the discrete nodes with sub-millimeter precision.

---

## 4. Reframing Around Graph ML Research Questions (For the Professor)

To satisfy the professor's requirement that the final project report be grounded in **Graph Machine Learning research**, we organize the paper around four fundamental theoretical questions:

```
                          [ Graph ML Research Questions ]
                                         |
     +-------------------+-------------------+-------------------+-------------------+
     |                   |                   |                   |                   |
   [ RQ1 ]             [ RQ2 ]             [ RQ3 ]             [ RQ4 ]
Spectral-Spatial    Skew-Symmetric       Extensive vs.       Out-of-Distribution
 Hybrid Coupling     Edge Flux MPNN      Intensive Targets     Graph Transfer
 (Oversmoothing)     (Conservation)     (State Inversion)     (CNN vs. GNN)
```

### RQ1: Does explicit spectral-spatial decomposition mitigate oversmoothing in deep message-passing physical surrogates?
* **Theoretical Foundation**: Repeated message-passing aggregations in deep MPNNs act as low-pass filters on graph signals ($\mathbf{H}^{(l+1)} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\mathbf{H}^{(l)}$), collapsing high-frequency features (oversmoothing).
* **Finding**: By computing the smooth low-frequency background diffusion via an explicit discrete Laplace-Beltrami operator ($\mathbf{M}^{-1}\mathbf{K}$), the 15-layer spatial MPNN is freed from propagating global smooth gradients and focuses entirely on localized high-frequency phase transitions. This achieves the project-best accuracy (**`0.00289` nRMSE**).

### RQ2: Can strict physical conservation laws be encoded into GNNs via skew-symmetric edge message passing?
* **Theoretical Foundation**: Standard node-level GNN regressions $\Delta u_i = f_\theta(\mathbf{x}_i, \mathcal{N}_i)$ violate the First Law of Thermodynamics because errors sum constructively across nodes over long autoregressive rollouts.
* **Finding**: Enforcing exact bit-for-bit skew symmetry on edge messages ($F_{ij} = -F_{ji}$) guarantees that internal fluxes sum identically to zero ($\sum_i \sum_{j \in \mathcal{N}(i)} F_{ji} \equiv 0$). Model D achieves **exact $0.000000\text{ J}$ internal energy accumulation**, providing provable long-term rollout stability with an inter-seed standard deviation of just $\pm 0.00010$.

### RQ3: What are the representational trade-offs between intensive ($T$) and extensive ($H$) nodal graph states?
* **Theoretical Foundation**: Temperature $T$ is an intensive thermodynamic property that experiences a derivative discontinuity during first-order phase change ($\partial T / \partial t \to 0$ between $T_S$ and $T_L$). Enthalpy $H$ is extensive and strictly monotonic ($\partial H / \partial t = q_{\text{in}} + \nabla \cdot (k \nabla T)$).
* **Finding**: While predicting extensive $\Delta H$ enforces exact energy conservation (Model D) and smooth monotonic transport (Model C), the non-linear inversion $T(H)$ amplifies small nodal enthalpy errors in the high-heat-capacity mushy zone. Direct intensive prediction $\Delta T$ (Model B) achieves higher pointwise regression accuracy at the expense of unconstrained energy drift.

### RQ4: How does unstructured graph message passing compare to Euclidean grid convolutions in geometric generalization?
* **Theoretical Foundation**: CNN kernels $\mathbf{W} \in \mathbb{R}^{k \times k}$ are tied to regular coordinate grids. GNNs operate on coordinate differences $(\mathbf{p}_i - \mathbf{p}_j)$ and topological adjacency.
* **Finding**: When evaluated on domain scaling (`test_large`, $4 \times 2\text{ mm}$ plate) and path geometry changes (`test_path`), U-Net collapses catastrophically (nRMSE `0.34978`, $+118\%$ energy blowup). In contrast, all GNN architectures generalize zero-shot across large plates (nRMSE `0.00850`) and complex L-turns/zigzags (`0.01240`).

---

## 5. Ready-to-Use Artifact Links

1. **Master Checkpoint Package**:
   * [`models_b_d_checkpoints_and_report.zip`](file:///scratch/smiyyapuram/graph-machine-learning-project/models_b_d_checkpoints_and_report.zip) (191 MB — Contains all 6 trained checkpoints, configs, training logs, and evaluation reports).
2. **Key Visualizations**:
   * **Node-Level & Mesh Comparison**: [`results/figures/graph_nodes_heat_comparison.png`](file:///scratch/smiyyapuram/graph-machine-learning-project/results/figures/graph_nodes_heat_comparison.png) (2.0 MB)
   * **Thermal Evolution Across Regimes**: [`results/figures/melt_pool_thermal_evolution.png`](file:///scratch/smiyyapuram/graph-machine-learning-project/results/figures/melt_pool_thermal_evolution.png) (385 KB)
   * **Global Heat Overview**: [`results/figures/heat_prediction_overview.png`](file:///scratch/smiyyapuram/graph-machine-learning-project/results/figures/heat_prediction_overview.png) (1.0 MB)
3. **Pull Request**:
   * Merged into `origin/main`: [PR #5 on GitHub](https://github.com/lucdop01/graph-machine-learning-project/pull/5)
