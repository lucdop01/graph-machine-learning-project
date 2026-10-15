# Lei scope and acceptance checklist

Checked on 15 October 2026 against the local Melt-Pool GNN Plan (4), the fetched
team repository, Project_proposal-1.pdf, Proposal.docx and TA Feedback.docx.
The PDF and Proposal.docx are course proposal guidance/templates, not a filled
technical specification. TA Feedback contains project-specific Q&A and some
team responses/proposals; these are distinguished from explicit instructor comments.
The latest named-owner plan defines Lei's allocation. Earlier broader team
plans do not silently assign every model or baseline to Lei.

| Requirement | Source and interpretation | Implementation or remaining work |
|---|---|---|
| B = fixed linear heat diffusion + learned correction, same backbone as A | Plan (4), Lei prototype; TA Feedback RQ3 | models/model_b.py; shared ModelA, fixed k/(rho cp) |
| Shared mesh Laplacian and filter | Plan (4), Jojo dependency | Working cotangent/filter implementation provided; reconcile with Jojo before experiments |
| No separate training loop; config + seed defines run | Plan (4), Interfaces | models/train.py registration, configs/model_b.yaml, checkpoint loading |
| Same settings and parameter count as A | Plan prototype and Fair comparison | Same width/layers, optimiser, noise and augmentation; paired initialization/count checked locally |
| Gate by 19 October, three short seeds, stride 5, val plus test_interp mesh 1 | Plan (4), Lei gate and rules | analysis/run_model_b_gate.py, 20k steps by default, full 150/20/30 run counts |
| Rule written before results | Plan gate | nRMSE improvement > max sample std(A,B) in both domains; nRMSE preregistered as primary; report melt MAE too |
| Go: full data, equal sweep budget and three final seeds | Plan conditional branch | Pending actual gate decision and agreed team sweep budget; do not infer Go from local tests |
| No-go: fixed filter and take over spectra | Plan conditional branch | Fixed filter and spectral-response tool ready; branch decision still pending |
| Frequency response vs exact heat kernel | TA Feedback interpretability response | analysis/spectra.py compares substepped fixed filter with exp(-alpha dt lambda) of the same graph operator |
| B correction concentration in 1800–2000 K band | TA Feedback interpretability response | analysis/model_b_corrections.py: area-weighted absolute correction share vs band area share, teacher-forced diagnostic |
| New mesh: test_interp graph seeds 0 and 1 | Plan (4), mesh transfer | analysis/mesh_transfer.py evaluates unchanged checkpoints, no refitting |
| test_fine only changes solver truth; show solver-to-solver gap | Plan and TA Feedback fine-test clarification | Same mesh; existing CrossSolver pairing, gap included beside model scores; not an error floor |
| Solidification on test_fine diagnostic only | Latest plan; clarified evaluation notes | Transfer interpretation explicitly limits claims; no new solidification accuracy claims |
| Report gate, B and mesh transfer | Plan writing row | Method/run guide ready; gate report generated from real results; final conclusions pending experiments |
| Traditional + advanced baseline, one ML challenge, bounded experiments | Course templates | Team scope: fixed diffusion + Model A; spectra/corrections address interpretation. Not all team deliverables assigned to Lei |
| Fitted LF-T/LF-H, U-Net, C/D, probes, A-dual, explanations, scaling | TA Feedback broader project response | Team-level follow-up, not implemented or claimed complete by this change |

## What is verified locally

The physics component test covers conservation of the fixed linear temperature
branch, operator properties, numerical parity and gate decision edge cases.
The training integration test uses the real solver and shared trainer with a
small backbone; interrupt/resume matches uninterrupted weights exactly. It
also checks A checkpoint compatibility and exercises the gate scorer on six
temporary fixture checkpoints, map export and transfer export. Fixture gate
numbers stay in a temporary directory and are deleted; they are not evidence
for a scientific Go/No-go decision.

## Current execution boundary

No CUDA or MPS device is available in this session, and the full generated
dataset is absent. The user chose local validation plus runnable experiment
scripts. Therefore formal gate/full training, checkpoint-based correction
results and final transfer comparisons remain pending. No result is invented.
No messages to teammates or remote training jobs are sent by these scripts.
