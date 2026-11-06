# Model B results

> Updated 6 November 2026. Report-ready results text based on the current augmented A checkpoints and Luc’s evaluation snapshots. Training and rollout credit remains with the original model trainers and evaluator. This update audits and analyses their files; it does not claim newly executed training or rollouts.

## Comparison

We compared the diffusion-residual Model B with the current augmented MeshGraphNet Model A using three training seeds per model. Both use a 100,000-step budget, stride 5, a 15-block width-128 backbone, the same optimizer settings, noise and rotation/reflection augmentation. Each checkpoint was selected using validation rollout nRMSE. Per-run score aggregation and evaluation metadata were verified against the raw CSV and JSON files. Normalization agrees to roundoff and training-data fingerprints match. Training runtimes differ (A: PyTorch 2.9.1+ROCm; B: PyTorch 2.5.1), so runtime effects are not isolated. Uncertainty below denotes sample standard deviation over three seeds, not a confidence interval.

On validation, B achieves nRMSE 0.002893 ± 0.000298 versus 0.003608 ± 0.000233 for A, a 19.8% lower mean. Melting-range MAE over true temperatures of 1800–2000 K is 3.707 ± 0.199 K for B versus 4.343 ± 0.289 K for A. The train-fitted LF-T K5 and fixed physical diffusion filter have validation nRMSE 0.340331 and 0.150486, respectively. B’s lower mean error extends to all seven evaluated domains, but these results alone do not establish a causal oversmoothing mechanism.

## Mesh transfer

With fixed weights, test_interp nRMSE increases from 0.004312 ± 0.000267 to 0.011776 ± 0.001489 for A when the learning mesh changes from seed 0 to seed 1. B increases from 0.003077 ± 0.000244 to 0.009905 ± 0.002296. B retains a 15.9% lower new-mesh mean error, but its relative degradation is larger (3.22× versus 2.73×). Its new-mesh nRMSE improvement over A is smaller than the larger seed standard deviation. The evidence therefore supports lower absolute error on this mesh, not uniformly greater robustness to remeshing. New-mesh melting MAE is 4.697 ± 0.220 K for B versus 8.689 ± 0.365 K for A.

## Fine-resolution solver and path generalization

On test_fine, B has nRMSE 0.004632 ± 0.000124 versus 0.005394 ± 0.000089 for A, a 14.1% lower mean. The paired 10–5 µm solver gap is 0.005073 nRMSE and 4.610 K melting MAE. It provides resolution-sensitivity context and is not a lower bound on learned-model error. test_fine changes the solver truth resolution, not the GNN mesh. Solidification remains diagnostic at stride 5.

On unseen paths, B achieves nRMSE 0.076476 ± 0.008090 compared with 0.099304 ± 0.006750 for A. D achieves 0.011847 ± 0.000143, outperforming both B and C (0.109287 ± 0.007318); the earlier report’s claim that C is best on paths should be removed.

## Gate and limitations

These are full-training results, not the originally scheduled short-budget gate. The old automated rule required nRMSE gains beyond seed spread on both validation and new mesh; the written plan also allowed melting-range improvement. Descriptively, the nRMSE comparison exceeds the larger seed standard deviation on validation but not the new mesh, while melting-MAE improvements exceed it on both. No rule is retroactively selected and no preregistered Go decision is claimed. The team should explicitly decide whether to retain or drop the formal short-training gate.

Only three seeds and one new-mesh realization are available. U-Net has one seed and uses a regular grid sampled at graph nodes; its mesh score does not test a learned graph operator. Error means and seed spreads do not establish statistical significance.

## Files

- [Full audited comparison and transfer tables](../results/lei/model_b_comparison/report.md)
- [Scores](../results/lei/model_b_comparison/scores.csv)
- [Source provenance and hashes](../results/lei/model_b_comparison/provenance.json)
- [Comparison figure](../results/lei/model_b_comparison/comparison.png)
- [Mesh-transfer figure](../results/lei/model_b_comparison/mesh_transfer.png)

Reproduce with `python -m analysis.compare_model_b`. Run `git fetch origin` first if the pinned pending-PR commit is not present locally.
