# Settings to align before the Model B gate

This is a draft for Lei to discuss with teammates. No message has been sent.
The following values describe the current code, not an agreement by others.

| Topic | Current implementation | Confirm with |
|---|---|---|
| Geometry | Signed cotangent stiffness, lumped area, insulated edges, SI units | Jojo |
| Fixed baseline | Same source integration/substeps/constants as B, no fitted parameters | Jojo; avoid a different independently tuned filter |
| Material constants | k=20, cp=600, rho=4420; alpha fixed | Team |
| Backbone | A's width 128, 15 blocks, 2 hidden layers; identical parameter count | Luc/training owner |
| Augmentation | Noise 0.03 of train T spread, rotations/reflections on for both A and B | Luc/training owner |
| Short budget | 20,000 steps per model/seed, 3 seeds, batch 4, same LR decay | Training owner; old unaugmented A results are not the matched comparison |
| Selection | Mean val rollout nRMSE only, stride 5 | Shared harness owner |
| Gate domains | 20 val runs and 30 test_interp runs on graph seed 1 | Jojo/harness owner |
| Pass rule | B nRMSE improvement > max sample std(A,B) in both domains | Team before running; already recorded locally |
| Deadline | Gate verdict planned for 19 October 2026 | Team; GPU access still pending |
| If Go | Same number of sweep trials as other models, then 3 final seeds | Team; exact sweep trial count not yet agreed |
| Interpretation | Fixed-filter spectral response now available; correction maps need a trained B | Satwik/interpretability owner |

## Message draft

Hi all, Model B is now integrated into the shared training/checkpoint pipeline
on main. It uses the same Model A backbone and parameter count, plus fixed
cotangent diffusion and the integrated laser source. The local training,
checkpoint/resume and evaluation integration checks pass.

Before the gate, could we align Jojo's Laplacian/filter with the implementation
in main and confirm the short budget? The current script uses 20k steps for
each of seeds 0/1/2, stride 5, and identical noise/rotation augmentation for A
and B. It scores val and test_interp on mesh seed 1 and reports nRMSE plus
melting-range MAE. The recorded rule uses nRMSE improvement greater than
max(std_A, std_B) in both domains. No formal gate result exists yet.

My DelftBlue GPU application is pending. Does anyone already have access to
a GPU environment where the matched A/B gate can run before 19 October?
