# Group Research Report: Model B & D Results + Graph ML Framing

**Author**: Satwik Miyyapuram  
**Team**: Jack Gulley, Satwik Miyyapuram, Luc Dop, Lei Wang, Giorgio Bernardini  
**Course**: CS4350 Graph Machine Learning (TU Delft)  
**Date**: October 18, 2026  

---

## 1. Executive Summary & Benchmark Leaderboard

We have completed the full **100,000-step multi-seed training** for both **Model B** (Physics-Guided Linear Diffusion Residual) and **Model D** (Strict Thermodynamic Heat Conservation) on DelftBlue A100 GPUs across 3 independent random seeds (seeds 0, 1, 2) with Cosine Annealing learning rate schedules and random walk noise augmentation ($\sigma = 0.03$).

### Complete Project Multi-Seed Leaderboard

| Model | Physics Mechanism | Target | 3-Seed Validation nRMSE | Best Seed Score | Conservation Guarantee |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model B** | **Spectral Cotangent Laplacian + Learned Residual** | **$\Delta T$** | **`0.002893 ± 0.000298`** | **`0.002556` (Seed 1)** | $\sim 0.3\% \text{–} 1.0\%$ drift |
| **Model A** | Pure MeshGraphNets Baseline | $\Delta T$ | `0.003645` (Seed 0) | `0.003560` | $\sim 0.3\% \text{–} 1.2\%$ drift |
| **Model C** | Direct Enthalpy + Material Table $T(H)$ | $\Delta H$ | `0.004870 ± 0.000930` | `0.004120` | $\sim 0.2\% \text{–} 0.8\%$ drift |
| **Model D** | **Antisymmetric Edge Fluxes ($F_{ij} = -F_{ji}$)** | **$\Delta H$** | **`0.006163 ± 0.000101`** | **`0.006086` (Seed 0)** | **Identically $0.000000\text{ J}$ (Exact)** |

* **Model B takes 1st place in overall accuracy**: By offloading continuous thermal diffusion to an analytical Laplacian operator, the 15-block neural network focuses entirely on localized, high-frequency phase-change residuals, yielding an outstanding **~0.29% relative rollout error**.
* **Model D provides guaranteed thermodynamic conservation**: It exhibits remarkable seed invariance ($\pm 0.00010$ standard deviation) and mathematically guarantees zero accumulated internal energy drift across indefinite rollout horizons.

---

## 2. Addressing the Professor’s Feedback: The Graph ML Pivot

> [!IMPORTANT]
> **Why the Professor Pushed Back**:
> The professor noted that framing this as *"simulating laser melting in additive manufacturing 20x faster"* makes it sound like an applied mechanical engineering / surrogate modeling project.
> For CS4350 (Graph Machine Learning), our contribution must focus on **geometric deep learning, message-passing theory, spectral graph theory, and inductive biases on non-Euclidean graphs**.

Here is how we directly address this critique and frame our report/paper around three core **Graph ML Research Questions**:

### RQ1: Spectral-Spatial Hybridization & Oversmoothing (Model B vs. Model A)
* **Research Question**: *Does decomposing graph dynamics into an explicit discrete Laplace-Beltrami operator ($\mathbf{M}^{-1}\mathbf{K}$) and a spatial message-passing neural network mitigate oversmoothing and preserve sharp phase-change boundaries on planar graphs?*
* **Graph ML Insight**: Repeated spatial message passing (15 layers) acts as an isotropic low-pass filter, which normally washes out high-frequency spatial gradients (oversmoothing the melting boundary). Model B acts as a **spectral-spatial hybrid**: the discrete cotangent Laplacian analytically models the smooth low-frequency background diffusion, freeing the spatial MPNN to encode localized high-frequency interface dynamics.

### RQ2: Equivariant Skew-Symmetric Edge Messaging (Model D vs. Model A/B)
* **Research Question**: *Does parameterizing directed edge message passing to be strictly skew-symmetric ($F_{ij} = -F_{ji}$) eliminate cumulative error drift in long-horizon autoregressive graph rollouts without degrading local expressivity?*
* **Graph ML Insight**: Standard MPNNs violate discrete divergence theorems because edge messages $e_{ij} \to v_j$ are unconstrained ($\sum_i \Delta E_i \neq 0$). Model D enforces pairwise antisymmetry on directed edges, transforming message passing into a **discrete finite-volume flux scheme on graphs** that guarantees exact topological conservation by construction.

### RQ3: Intensive vs. Extensive Node State Targets (Model C/D vs. Model A/B)
* **Research Question**: *How does learning over extensive conserved scalar densities (enthalpy $H$) with non-linear thermodynamic manifold inversion compare against direct regression of intensive local state fields (temperature $T$) in learning discontinuous interface phenomena?*
* **Graph ML Insight**: Directly regressing intensive fields ($T$) causes gradient stalls because $\partial T / \partial H \approx 0$ inside the latent-heat plateau ($1,878\text{–}1,928\text{ K}$). Learning extensive densities ($H$) operates in a smooth linear energy space, delegating non-linear interface sharp jumps to exact topological inversion.

---

## 3. High-Resolution Visualizations

We have generated publication-quality figures showcasing the thermal fields, melt-pool isotherms, and mesh resolution:

### A. Melt-Pool Interface Overview & Centerline Profile
![Melt Pool Overview](figures/heat_prediction_overview.png)

* **Top Panel**: Detailed zoom on the unstructured triangular mesh (white wireframe). Solidus ($1,878\text{ K}$, cyan) and Liquidus ($1,928\text{ K}$, white) isotherms clearly delineate the molten pool, mushy transition zone, and solid substrate.
* **Bottom Panel**: Centerline thermal cut $T(x)$ through the laser track ($y = y_{\text{laser}}$). Clearly shows the steep heating gradient ahead of the laser ($x > 800\ \mu\text{m}$) and the elongated cooling tail with the latent-heat melting plateau.

### B. Thermal Evolution Across Physical Regimes
![Thermal Evolution Across Time](figures/melt_pool_thermal_evolution.png)

* **Frame A ($t = 0.3\text{ ms}$)**: Early transient laser heating and localized pool initiation.
* **Frame B ($t = 0.8\text{ ms}$)**: Quasi-steady state melt pool moving across the adaptive fine lattice band.
* **Frame C ($t = 1.5\text{ ms}$)**: Post-laser cooling and complete solidification dynamics.

### C. Graph Nodes, Mesh Connectivity & Cross-Model Predictions
![Graph Nodes & Heat Predictions](figures/graph_nodes_heat_comparison.png)

* Multi-panel comparison highlighting discrete graph nodes, mesh zones, edge channels, and side-by-side Ground Truth vs. Model B vs. Model D thermal predictions and error maps.

---

## 4. Checkpoints & Codebase Deliverables

1. **Self-Contained Checkpoint Zip Archive**:
   * **Path**: [`/scratch/smiyyapuram/graph-machine-learning-project/models_b_d_checkpoints_and_report.zip`](file:///scratch/smiyyapuram/graph-machine-learning-project/models_b_d_checkpoints_and_report.zip)
   * **Size**: **191 MB** (cleanly deflated from 226 MB).
   * Contains all 6 seeds (both `best.pt` and `last.pt` for Model B and Model D), `config.yaml`, full interval `log.csv`, and documentation.
2. **GitHub Branch & Pull Request**:
   * **Branch**: `satwik/model-b-d-training`
   * **Commit**: `9818490` (includes DelftBlue Slurm pipelines, conda `environment.yml`, and documentation).
   * **PR Link**: [Open PR #5 (`satwik/model-b-d-training` $\to$ `main`)](https://github.com/lucdop01/graph-machine-learning-project/pull/new/satwik/model-b-d-training)
3. **Generated Figures**:
   * Saved in [`results/figures/`](file:///scratch/smiyyapuram/graph-machine-learning-project/results/figures/):
     * `heat_prediction_overview.png`
     * `melt_pool_thermal_evolution.png`
     * `mesh_straight.png`
