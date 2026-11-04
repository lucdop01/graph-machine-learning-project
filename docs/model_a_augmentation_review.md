# Model A augmentation results

> **Note, 4 November 2026.** The augmented training runs `a_aug_seed0`-`a_aug_seed2` are shared as
> `results/models/model_a_seed0`-`model_a_seed2`, and their evaluation files are `results/eval/model_a_seed*`. Only the
> run name stored in each `best.pt` changed; the weights are identical, so the shared files' hashes differ from the
> training-run hashes recorded in the numerical summary. The test splits were re-run from the shared checkpoints on
> 4 November and agree with the 29 October files to within GPU run-to-run noise. The seed-0 diagnostic of the augmented
> run is `results/eval/model_a_seed0_path_diagnostic.csv`.
>
> A second review recomputed every number below from the raw files and adds four points. (1) On the straight +x track,
> the training situation, the one-step energy ratio moves from 0.977 to 1.025, so augmentation turns a small deficit
> into a small excess of similar size rather than removing the bias. (2) Mesh topology and boundary distance change with
> the path, so the zigzag result does not isolate the hot surroundings as the cause. (3) The spurious melt pools cover
> 1-2 x 10^-4 mm^2, a few nodes above the melting threshold. (4) The new-mesh domain, where augmentation costs most, is
> one of the two Model B gate domains, and every other model trains with augmentation.

Random rotation and reflection during training substantially improve Model A on unseen L-turns, while zigzags
beside previously heated tracks remain inaccurate. Across three seeds, test_path nRMSE falls from 0.1728 to 0.0993
(42.5%), and melt-pool IoU rises from 0.302 to 0.390. Every path run improves in nRMSE for every seed. Ordinary
interpolation accuracy is nearly unchanged, but accuracy on the unseen mesh seed worsens substantially.

Training finished on 14 October 2026. Evaluation and this review are dated 29 October 2026. The three original runs
remain the ablation; each of the three augmented runs is evaluated at its best validation checkpoint.

## Training and evaluation

Model A is the temperature-increment MeshGraphNet with width 128 and 15 message-passing blocks. The augmented runs
use the same 150 training trajectories, straight tracks moving +x from a cold plate, and the same 66.7 us model step.
Each training sample receives one shared orthogonal transformation: a uniform rotation over 0-360 degrees and a
reflection with probability 0.5. Raw edge vectors and node positions transform together; edge lengths, node volumes,
scalar features and targets remain unchanged. Validation and evaluation receive no augmentation.

For each matching seed, the checkpoint configurations differ only in the augmentation flag. Both settings use
100,000 Adam steps, batch size 4, learning rate 3e-4 decaying to 1e-6, input noise 0.03 times the training temperature
spread, and bf16 matrix products in training with fp32 evaluation. Hyperparameters were kept from the original
tuning. Augmentation was introduced after inspecting the original test_path failure, so these results are a disclosed
follow-up experiment on the existing test set. Training still contains no turning or multiple-track trajectories.

Evaluation used `E:\venvs\gml-rocm\Scripts\python.exe`, Python 3.12, PyTorch 2.9.1+rocm7.2.1 and the AMD Radeon
RX 7900 XTX. The device argument is `cuda` for this ROCm build. The project-local `.venv` contains CPU-only PyTorch
and is unsuitable for these GPU commands.

All seven evaluations were completed for every augmented seed: val, test_interp, test_interp with mesh seed 1,
test_extrap, test_path, test_fine and test_large. This produced 21 CSV/JSON pairs, plus two diagnostics comparing
`a_seed0` and `a_aug_seed0` on one straight track, one L-turn and one zigzag. Split sizes are respectively
20, 30, 30, 15, 20, 30 and 10 runs. The path split contains ten L-turns and ten zigzags.

| Augmented seed | Best checkpoint step | Logged validation nRMSE | Re-evaluated validation nRMSE |
|---|---|---|---|
| 0 | 90,000 | 0.003640350916 | 0.003640352187 |
| 1 | 84,000 | 0.003823117485 | 0.003823116897 |
| 2 | 98,000 | 0.003359586201 | 0.003359585813 |

## Evaluation checks

All expected new files and run identifiers are present. nRMSE and energy metrics are finite on every run. CSV values
reproduce the JSON nRMSE means, melting-range MAE, IoU, and energy medians and maxima. The same checks pass for the
21 original CSV/JSON pairs used as the comparison. The two diagnostic files align on 132 rows with identical run,
frame, time, source energy and truth energy changes.

The six best checkpoints have the expected names, validation-selected steps and scores, matching seed settings and
the same training fingerprint, `a134269f75ce9513`. The largest difference between a repeated augmented validation
score and its training log is 1.27e-9, negligible relative to the reported errors. The numerical review stores the
exact scores, checkpoint metadata and SHA-256 hashes in
[`model_a_augmentation_summary.json`](../results/eval/model_a_augmentation_summary.json).

Values below are means and sample standard deviations over three training seeds. Within a seed, nRMSE is averaged
over runs; melting-range MAE pools the true 1,800-2,000 K node/frame pairs, and IoU pools area-weighted intersections
and unions across runs. Energy error (a) is the maximum absolute error divided by cumulative laser input after 5%
of the run's input has arrived, followed by a median over runs and then a mean over seeds. Final energy error is
signed and divided by the run's total laser input. Energy errors compare predictions with the stored mesh truth.

## Accuracy across test splits

| Split | Original nRMSE | Augmented nRMSE | Relative change |
|---|---|---|---|
| val | 0.00360 +/- 0.00007 | 0.00361 +/- 0.00023 | +0.1% |
| test_interp | 0.00433 +/- 0.00015 | 0.00431 +/- 0.00027 | -0.5% |
| test_interp, mesh seed 1 | 0.00675 +/- 0.00028 | 0.01178 +/- 0.00149 | +74.4% |
| test_extrap | 0.01363 +/- 0.00021 | 0.01558 +/- 0.00059 | +14.3% |
| test_path | 0.17284 +/- 0.00721 | 0.09930 +/- 0.00675 | -42.5% |
| test_fine | 0.00564 +/- 0.00018 | 0.00539 +/- 0.00009 | -4.3% |
| test_large | 0.01039 +/- 0.00149 | 0.01141 +/- 0.00054 | +9.8% |

| Split | Augmented melting-range MAE | Augmented IoU | Augmented energy error (a), median |
|---|---|---|---|
| test_interp | 7.1 +/- 0.2 K | 0.961 +/- 0.001 | 1.38 +/- 0.61% |
| test_interp, mesh seed 1 | 8.7 +/- 0.4 K | 0.952 +/- 0.002 | 1.10 +/- 0.15% |
| test_extrap | 24.1 +/- 0.7 K | 0.910 +/- 0.003 | 1.32 +/- 0.47% |
| test_path | 252.2 +/- 24.0 K | 0.390 +/- 0.030 | 9.56 +/- 1.22% |
| test_fine | 7.9 +/- 0.2 K | 0.951 +/- 0.004 | 1.44 +/- 0.51% |
| test_large | 4.9 +/- 0.3 K | 0.969 +/- 0.001 | 2.42 +/- 0.91% |

The mean validation and ordinary interpolation nRMSE stay almost unchanged. Ordinary test_interp melting-range MAE
increases from 6.5 to 7.1 K, IoU falls from 0.964 to 0.961, and energy error (a) increases from 0.89% to 1.38%.
The largest regression is on the new mesh seed: 88 of the 90 run/seed comparisons worsen in nRMSE. Parameter
extrapolation also worsens. The results support improved generalization to path changes; other distribution shifts
show different trade-offs.

On test_large, the mean across seeds of the median signed final energy error decreases from +3.82% to +0.64%,
even though nRMSE increases. All augmented seeds predict a melt pool on `test_large_006`, which has no true pool;
augmented seed 2 also does so on `test_interp_027` with mesh seed 1. These failures accompany the lower energy drift.

## L turns and zigzags

Each family contains ten runs. Family metrics use the same pooling rules as the split summaries. In the energy
column, each seed contributes its median signed final error across that family's runs.

| Path family | Setting | nRMSE | Melting-range MAE | IoU | Median final energy error |
|---|---|---|---|---|---|
| L-turn | Original | 0.1039 +/- 0.0083 | 84.4 +/- 5.8 K | 0.636 +/- 0.012 | -10.34 +/- 1.38% |
| L-turn | Augmented | 0.0360 +/- 0.0015 | 16.6 +/- 0.4 K | 0.868 +/- 0.005 | -1.79 +/- 0.05% |
| Zigzag | Original | 0.2417 +/- 0.0062 | 333.7 +/- 6.8 K | 0.286 +/- 0.003 | -30.86 +/- 0.68% |
| Zigzag | Augmented | 0.1626 +/- 0.0124 | 267.0 +/- 25.5 K | 0.368 +/- 0.031 | -22.02 +/- 1.63% |

L-turn nRMSE decreases by 65.4%, with a much smaller energy deficit and a substantial recovery of melt-pool shape.
Across all augmented L-turn runs and seeds, final energy errors range from -2.55% to -1.19% of total laser input.
L-turn nRMSE nevertheless remains well above ordinary interpolation error.

Zigzag nRMSE decreases by 32.7%, but the remaining field and energy errors are large. Final energy errors across
augmented zigzags range from -28.53% to -13.50%. The combined path split improves on every run for every seed, while
remaining unsuitable for a claim of accurate generalization to arbitrary scan paths. Its median absolute relative
solidification-duration error falls from about 49.5% to 33.4%; about 35.0% of nodes that fully melt in the sampled
truth still fail to fully melt in the augmented predictions, compared with 44.4% originally.

## One step energy diagnostics

The diagnostics start each prediction from the true state, separating one-step model error from the accumulation
of rollout errors. The ratio below is the sum of predicted energy increments divided by the sum of laser input over
the indicated leg. It is therefore weighted by source energy. Leg labels follow changes in the source centroid;
the first step, which has no previous centroid to define a direction, is excluded.

| Seed 0 example and leg | Steps | Original ratio | Augmented ratio | Truth ratio |
|---|---|---|---|---|
| Straight track, +x | 2-29 | 0.977 | 1.025 | 1.002 |
| L-turn, first +x | 2-15 | 0.984 | 1.028 | 1.001 |
| L-turn, +y | 16-24 | 0.592 | 0.956 | 1.002 |
| Zigzag, first +x | 2-21 | 0.981 | 1.005 | 1.001 |
| Zigzag, first +y connector | 22-24 | 0.707 | 0.883 | 1.003 |
| Zigzag, -x beside the first track | 25-45 | 0.405 | 0.643 | 1.000 |
| Zigzag, second +y connector | 46-48 | 0.403 | 0.660 | 1.005 |
| Zigzag, second +x beside a hot track | 49-69 | 0.268 | 0.274 | 1.001 |

The L-turn +y ratio approaches the physical energy balance after augmentation. The zigzag -x leg improves but still
loses much of the net input, and the second +x leg barely improves even though +x is the original training direction.
This supports a remaining failure on unfamiliar thermal context. Mesh topology and boundary relationships also
change with the paths, so the experiment does not isolate hot surroundings as the sole cause.

These ratios measure the combined learned diffusion and laser response. They cannot distinguish underestimated
source heating from a spurious energy sink in the diffusion update. A source-off comparison at fixed true states
would separate those two model contributions.

The stored mesh truth has small energy-budget differences from the solver because of preprocessing. Ratios are
close to, but not exactly, 1; the final partial laser step of the zigzag has a truth ratio of 0.908. The table includes
partial steps and uses source weighting, so it should not be compared directly with selected full-step ranges in
the original informal diagnosis. The diagnostics cover seed 0 and three example runs; the family-level rollout
results above cover all three seeds and all twenty unseen paths.

## Results to preserve and next comparisons

Keep both training settings and all seeds in the report. The augmented setting gives much better L-turn predictions
without adding turning paths to training. The nearly unchanged second +x diagnostic identifies a remaining failure
for the comparison with architectures that include conservation or material laws. Exact conservation in Model D
will constrain the net energy error; temperature, melting and heat redistribution still require evaluation.

Use the same declared augmented protocol for the remaining model comparison and retain the original Model A as the
ablation. Track the mesh-seed regression explicitly. Any experiment with hot initial fields or multiple tracks
should be reported separately because it changes which thermal states are covered during training. For an independent
final assessment after this test-informed change, freeze the protocol before evaluating fresh held-out trajectories.

The raw reports and numerical summary belong in version control. The three augmented `best.pt` files, with their
configurations and training logs, are shared in `results/models/model_a_seed0`-`model_a_seed2`. The original runs'
checkpoints and every `last.pt`, which retains optimizer and RNG state for continuation, remain local. Their metadata
and hashes are recorded in the numerical summary.
