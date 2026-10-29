# Full evaluation of the existing shared weights

Submit from Lei's DelftBlue checkout:

```bash
git pull --ff-only
bash lei_submit_full_eval.sh
```

The first job prepares/reuses train, val, test_interp and test_fine data, statistics,
size bins and test_interp mesh seed 1. Existing valid simulation files are reused.
Do not run other writers on these data directories concurrently. The submission
script refuses to duplicate a running lei_prepare_eval/lei_full_prepare/lei_full_eval.

After preparation succeeds, five array tasks score A seed 0, B seeds 0/1/2, and
the two filters. At most two tasks run concurrently. Domains: val mesh 0,
test_interp mesh 0, test_interp mesh 1, test_fine mesh 0. The filter task also
computes the paired solver-to-solver gap. Every completed trajectory is saved.
No model training is started. Disconnecting SSH does not cancel batch jobs.

LF-T source and fit are retained from team branch commit
3528c6cdd40b7b4e8b9f1cebd5f9abab27e22df2, in separate analysis modules; only its
module import and default artifact location were adjusted. It does not replace
main's fixed physical filter. Both operators passed a numerical parity check on
straight meshes 0 and 1; LF-T also passed the constant-state zero-source check.
The complete cluster rollout has not been executed locally: generated data are
not present locally. Actual cluster package/GPU compatibility remains to be
confirmed by the running jobs.

Outputs: results/lei/full_eval, including per-run CSV, summary JSON, completion
markers, scores.csv, status.json and report.md. Summary is scheduled after the
evaluation array finishes, even if individual evaluation tasks fail. If data
preparation fails, the dependent array will not start; inspect its log and cancel
the blocked array/summary before submitting again. Clear/archive old output files
before a fresh campaign; do not mix outputs from multiple campaigns.

Monitor:

```bash
squeue -u "$USER"
tail -n 40 lei_full_prepare_JOBID.out
tail -n 40 lei_full_eval_ARRAYID_0.out
```

This is a complete evaluation campaign for the **currently available** A/B
weights. It is not a completed formal gate: augmented A seeds 1/2 are missing,
and the shared checkpoints have 100k-step budgets rather than the original
20k-step short-budget gate. A verdict must not be inferred from partial outputs.
The original from-scratch paired gate remains available through
analysis.run_model_b_gate; do not run it merely to duplicate existing training.
Training/tuning, additional spectral interpretation and final report writing
are outside this batch campaign.
