"""Check 8: the rollout and evaluation harness (evaluation/) gives a perfect model a perfect score.

The solver's own frames are fed in as predictions, so the harness is proven before any model can confuse the picture.
Each predictor runs at strides 5 and 2 (66.7 and 26.7 us; week1 frames are twice as long):
  oracle        the true frames: nRMSE 0, melting-range MAE 0, IoU 1 (hard and soft, overall and per size bin),
                energy error 0, solidification error 0 with 0 missed, 0 spurious and the same unfinished nodes as
                the truth. Asserted to 1e-6.
  offset +10 K  the true frames plus 10 K: melting-range MAE exactly 10 K; nRMSE and the energy errors equal closed
                forms computed here, nRMSE = 10 sqrt(N) / ||T - 300 K|| and dE = sum rho d A [H(T + 10) - H(T)].
  persistence   T_next = T: nRMSE equal to ||T_0 - T|| / ||T - 300 K||, which is 1 because frame 0 is exactly 300 K
                (also checked); IoU 0 and every melted node missed, so clearly non-zero errors.
  torch path    (needs torch and torch_geometric, skipped without them, so not in CI) the oracle through
                TorchPredictor with a dummy module that reads the normalised input temperature, de-normalises it and
                outputs the normalised true dT, then the normalised true dH. This proves the normalisation and
                de-normalisation with the training statistics: every predicted node temperature must match the truth
                to 0.01 K (float32 rounding gives about 1e-4 K), and nRMSE, the melting-range error and the energy
                errors must be zero up to that rounding. IoU, solidification and the missed and spurious counts are
                printed but not gated: they follow from the temperatures, and on a pool that barely passes the solidus
                a rounding error of 3e-5 K already moves the soft IoU by about 1e-3. The statistics are the saved
                data/graphs/norm_stride<s>.json, loaded with all its checks (graphs.dataset.load_stats), so a stale
                file fails. Only in CI mode, and only when that file is absent, are statistics computed from the
                check's own runs instead (printed as such).
Printed, not gated: the cadence diagnostic of solidification per run (the truth at the output times against the truth
at every saved frame), the oracle's energy budget (the preprocessing mismatch) and, for test_fine, the 10 vs 5 um gap.

Usage (from the repo root):
  python -m verify.harness             CI mode: simulates 5 runs (numpy, scipy and h5py only), about 2 minutes
  python -m verify.harness --dataset   the processed val and test splits in data/graphs/seed0; needs
                                       data/graphs/size_bins.json (python -m evaluation.rollout --size-bins)
  options: --splits ..., --mesh-seed 1 (with --dataset)
Pass: every assertion above, for every run and stride."""
import argparse
import os
import time

import numpy as np

from solver.solver import simulate
from solver.ti64 import CONST_PROPS, RHO, T0
from data.generate import COOL
from graphs.mesh import PATHS
from graphs.preprocess import mesh_for_run
from graphs.dataset import RunGraph, TORCH_MISSING, compute_stats, load_runs, load_stats, stats_file
from evaluation.rollout import (SPLITS, Offset, Oracle, Persistence, compute_size_bins, evaluate_run, evaluate_split,
                                fmt, load_size_bins, rollout)

STRIDES = (5, 2)
TOL = 1e-6                                     # oracle and closed forms
TOL32 = dict(T=1e-2, nrmse=1e-5, melt_mae=1e-2, energy=1e-5)   # torch path: float32 rounding (T, MAE in K)

# split (sets the material table), path, P (W), v (m/s), extra simulate() arguments
CASES = [
    ("val", "straight", 120.0, 0.65, COOL),            # mid-size pool, the largest cadence error at the model step
    ("val", "straight", 150.0, 0.65, COOL),            # mid-size pool
    ("test_extrap", "straight", 195.0, 0.50, COOL),    # largest straight pool
    ("test_path", "zigzag", 190.0, 0.66, COOL),        # large pool, remelting
    ("week1", "straight", 130.0, 0.55, dict(props=CONST_PROPS)),   # no melting, 26.52 us frames
]

failures = []


def expect(ok, what):
    if not ok:
        failures.append(what)
        print(f"    FAIL: {what}")
    return ok


def close(x, y, tol):
    return x is not None and np.isfinite(x) and abs(x - y) <= tol


# ---------------------------------------------------------------- assertions per predictor

def check_oracle(rows, summary, tol, label="oracle"):
    """Metrics that are defined must be exact. Runs without melting leave the melting metrics undefined, so the
    pooled values must show that the melting metrics were exercised at all."""
    for r in rows:
        name = f"{label} {r['run']} stride {r['stride']}"
        expect(close(r["nrmse"], 0.0, tol["nrmse"]), f"{name}: nRMSE {r['nrmse']}")
        for k in ("a", "b", "end"):
            expect(close(r[f"energy_{k}"], 0.0, tol["energy"]), f"{name}: energy ({k}) {r[f'energy_{k}']}")
        if "melt_mae" not in r:
            continue
        if r["melt_points"] > 0:
            expect(close(r["melt_mae"], 0.0, tol["melt_mae"]), f"{name}: melting-range MAE {r['melt_mae']}")
        if r["union_mm2"] > 0:
            expect(close(r["iou"], 1.0, tol["iou"]), f"{name}: IoU {r['iou']}")
        if r["union_soft_mm2"] > 0:
            expect(close(r["iou_soft"], 1.0, tol["iou"]), f"{name}: soft IoU {r['iou_soft']}")
        if r["sol_compared"] > 0:
            expect(close(r["sol_med"], 0.0, tol["sol"]) and close(r["sol_med_abs"], 0.0, tol["sol"]),
                   f"{name}: solidification {r['sol_med']} / {r['sol_med_abs']} over {r['sol_compared']} nodes")
        expect(r["sol_missed"] == 0 and r["sol_spurious"] == 0 and r["sol_unfinished"] == r["sol_unfinished_ref"]
               and r["sol_compared"] == r["sol_melted_ref"] - r["sol_unfinished_ref"],
               f"{name}: compared {r['sol_compared']}, missed {r['sol_missed']}, spurious {r['sol_spurious']}, "
               f"unfinished {r['sol_unfinished']} vs {r['sol_unfinished_ref']}")
    melting = [r for r in rows if "melt_mae" in r]
    if melting and any(r["peak_area_mm2"] > 0 for r in melting):      # the melting metrics were exercised
        expect(sum(r["melt_points"] for r in melting) > 0 and sum(r["sol_compared"] for r in melting) > 0
               and sum(r["union_mm2"] for r in melting) > 0, f"{label}: no melting-range point, pool or duration")
    if "iou" in summary:
        expect(close(summary["iou"]["iou"], 1.0, tol["iou"]), f"{label}: pooled IoU {summary['iou']['iou']}")
        for b, v in summary.get("iou_by_bin", {}).items():
            if b != "none":
                expect(close(v["iou"], 1.0, tol["iou"]), f"{label}: IoU in bin {b} {v['iou']}")
            else:
                expect(v["runs_with_predicted_pool"] == 0, f"{label}: predicted pool in bin none")


def check_offset(rows, runs, dT=10.0):
    by_name = {r.name: r for r in runs}
    for r in rows:
        run, name = by_name[r["run"]], f"offset {r['run']} stride {r['stride']}"
        f = r["stride"] * np.arange(1, r["frames_scored"] + 1)
        T = run.T_frames(f)
        cf = dT * np.sqrt(T.size) / np.sqrt(((T - T0) ** 2).sum())
        expect(close(r["nrmse"], cf, 1e-9 * cf), f"{name}: nRMSE {r['nrmse']} vs closed form {cf}")
        dE = RHO * float(run.attrs["d"]) * ((run.H_from_T(T + dT) - run.H_from_T(T)) * run.mesh.area).sum(axis=1)
        cum, total = run.laser_input()[f], run.laser_input()[-1]
        late = cum >= 0.05 * total
        cfe = dict(a=np.max(np.abs(dE[late]) / cum[late]), b=np.max(np.abs(dE)) / total, end=dE[-1] / total)
        for k, v in cfe.items():
            expect(close(r[f"energy_{k}"], v, 1e-9 * abs(v)), f"{name}: energy ({k}) {r[f'energy_{k}']} vs {v}")
        if "melt_mae" in r and r["melt_points"] > 0:
            expect(close(r["melt_mae"], dT, TOL), f"{name}: melting-range MAE {r['melt_mae']}")


def check_persistence(rows, runs):
    by_name = {r.name: r for r in runs}
    for r in rows:
        run, name = by_name[r["run"]], f"persistence {r['run']} stride {r['stride']}"
        f = r["stride"] * np.arange(1, r["frames_scored"] + 1)
        T, T_start = run.T_frames(f), run.T(0)
        expect((T_start == T0).all(), f"{name}: frame 0 is not 300 K everywhere")
        cf = np.sqrt(((T_start - T) ** 2).sum() / ((T - T0) ** 2).sum())
        expect(close(r["nrmse"], cf, 1e-12), f"{name}: nRMSE {r['nrmse']} vs closed form {cf}")
        expect(r["energy_b"] > 0.1, f"{name}: energy (b) {r['energy_b']} is not clearly non-zero")
        if "melt_mae" in r:
            if r["melt_points"] > 0:
                expect(r["melt_mae"] > 100, f"{name}: melting-range MAE {r['melt_mae']} K")
            if r["union_mm2"] > 0:
                expect(r["iou"] == 0.0, f"{name}: IoU {r['iou']}")
            expect(r["sol_missed"] == r["sol_melted_ref"], f"{name}: {r['sol_missed']} of {r['sol_melted_ref']} missed")


def line(label, s):
    out = f"  {label:12s} nRMSE {fmt(s['nrmse_mean'], digits=4):>8s}"
    if "melt_mae" in s:
        sol = s["solidification"]
        out += (f"  melt MAE {fmt(s['melt_mae'], digits=4):>7s} K  IoU {fmt(s['iou']['iou'], digits=4):>6s} / "
                f"{fmt(s['iou']['iou_soft'], digits=4):>6s}  sol {fmt(sol['med'], True):>7s} "
                f"(missed {sol['missed']}, spurious {sol['spurious']}, unfinished {sol['unfinished']})")
    e = s["energy"]
    out += f"  energy (a) max {fmt(e['a']['max'], True)}, (b) max {fmt(e['b']['max'], True)}"
    print(out)


# ---------------------------------------------------------------- one group of runs

def check_runs(runs, bins, partners=None, title=""):
    """All three reference predictors at both strides; returns the oracle rows per stride for the diagnostics."""
    oracle_rows = {}
    for stride in STRIDES:
        steps = sorted({round(stride * r.frame_us, 1) for r in runs})
        print(f"\n{title}stride {stride} (" + ", ".join(f"{x} us" for x in steps) + ")")
        rows, s = evaluate_split(runs, Oracle(), stride, bins, partners)
        line("oracle", s)
        check_oracle(rows, s, dict(nrmse=TOL, melt_mae=TOL, iou=TOL, sol=TOL, energy=TOL))
        oracle_rows[stride] = rows
        b = s["budget"]
        print(f"  {'':12s} oracle energy budget (mesh energy minus mesh laser input, the preprocessing mismatch): "
              f"(a) max {fmt(b['a']['max'], True)}, (b) max {fmt(b['b']['max'], True)}")
        rows, s = evaluate_split(runs, Offset(10.0), stride, bins, partners)
        line("offset +10 K", s)
        check_offset(rows, runs)
        rows, s = evaluate_split(runs, Persistence(), stride, bins, partners)
        line("persistence", s)
        check_persistence(rows, runs)
    return oracle_rows


def print_cadence(runs, oracle_rows):
    """Solidification at the output times against every saved frame (same nodes, rules), per run and stride."""
    print("\ncadence diagnostic: truth at the output times vs the truth at every saved frame (not gated)")
    head = "".join(f" | stride {s}: median   |med|  missed unfin." for s in STRIDES)
    print(f"  {'run':28s} {'pool mm2':>8s} {'melted':>6s}{head}")
    for i, run in enumerate(runs):
        if not run.melts:
            continue
        rs = [oracle_rows[s][i] for s in STRIDES]
        cells = "".join(f" |        {fmt(r['cad_med'], True):>7s} {fmt(r['cad_med_abs'], True):>7s} "
                        f"{r['cad_missed']:6d} {r['cad_unfinished']:6d}" for r in rs)
        print(f"  {run.name:28s} {rs[0]['peak_area_mm2']:8.4f} {rs[0]['cad_melted_ref']:6d}{cells}")
    if any("gap_med" in r for r in oracle_rows[STRIDES[0]]):
        print("\n10 vs 5 um gap (test_fine): the paired 10 um run against the 5 um truth at the output times")
        for i, run in enumerate(runs):
            rs = [oracle_rows[s][i] for s in STRIDES]
            print(f"  {run.name:28s}" + "".join(f" | stride {r['stride']}: median {fmt(r['gap_med'], True):>7s} "
                                                 f"|med| {fmt(r['gap_med_abs'], True):>7s} missed {r['gap_missed']} "
                                                 f"spurious {r['gap_spurious']}" for r in rs))


# ---------------------------------------------------------------- torch path

def torch_path(runs, ci_mode):
    if TORCH_MISSING:
        print(f"\ntorch path: skipped ({TORCH_MISSING})")
        return
    import torch
    from evaluation.torch_predictor import TorchPredictor

    class TruthModule(torch.nn.Module):
        """Dummy model: de-normalises the input temperature and returns the normalised true change to frame k + s."""

        def __init__(self, stats, stride, target):
            super().__init__()
            self.stats, self.stride, self.target = stats, stride, target
            self.runs = {r.name: r for r in runs}

        def forward(self, g):
            run, k, n = self.runs[g.run], int(g.k), self.stats["node"]["T"]
            T_in = g.x[:, 0].double().cpu().numpy() * n["std"] + n["mean"]
            T_next = run.T(k + self.stride)
            d = T_next - T_in if self.target == "dT" else run.H_from_T(T_next) - run.H_from_T(T_in)
            m = self.stats["target"][self.target]
            return torch.tensor((d - m["mean"]) / m["std"], dtype=torch.float32)

    melting = [r for r in runs if r.melts]                 # the training statistics are for 13.34 us frames
    print(f"\ntorch path: the oracle through TorchPredictor (torch {torch.__version__}), {len(melting)} runs")
    for stride in STRIDES:
        if os.path.exists(stats_file(stride)) or not ci_mode:
            stats, source = load_stats(stride), f"{stats_file(stride)} (checked against the processed train split)"
        else:
            stats, source = compute_stats(melting, stride), "computed from this check's own runs (no saved file)"
        print(f"  stride {stride}: statistics {source}")
        for target in ("dT", "dH"):
            pred = TorchPredictor(TruthModule(stats, stride, target), stats, stride, target, name=f"oracle_{target}")
            dT_max = 0.0
            for run in melting:
                ro = rollout(run, pred, stride)
                dT_max = max(dT_max, float(np.abs(ro["T"] - run.T_frames(ro["frames"])).max()))
            rows = [evaluate_run(run, pred, stride)[0] for run in melting]
            worst = lambda key: np.nanmax([abs(r[key] - (1.0 if "iou" in key else 0.0)) for r in rows])
            print(f"    target {target}: worst |T error| {dT_max:.2e} K, nRMSE {worst('nrmse'):.2e}, melt MAE "
                  f"{worst('melt_mae'):.2e} K, energy (a) {worst('energy_a'):.2e}, (b) {worst('energy_b'):.2e}; "
                  f"not gated: |1 - IoU| {worst('iou'):.2e} (soft {worst('iou_soft'):.2e}), |sol| "
                  f"{worst('sol_med_abs'):.2e}, missed {sum(r['sol_missed'] for r in rows)}, "
                  f"spurious {sum(r['sol_spurious'] for r in rows)}")
            label = f"torch {target} stride {stride}"
            expect(dT_max <= TOL32["T"], f"{label}: node temperature off by {dT_max} K")
            for r in rows:
                expect(close(r["nrmse"], 0.0, TOL32["nrmse"]), f"{label} {r['run']}: nRMSE {r['nrmse']}")
                if r["melt_points"] > 0:
                    expect(close(r["melt_mae"], 0.0, TOL32["melt_mae"]), f"{label} {r['run']}: MAE {r['melt_mae']}")
                for k in ("a", "b", "end"):
                    expect(close(r[f"energy_{k}"], 0.0, TOL32["energy"]),
                           f"{label} {r['run']}: energy ({k}) {r[f'energy_{k}']}")


# ---------------------------------------------------------------- modes

def ci_runs():
    runs = []
    for split, path, P, v, kw in CASES:
        t0 = time.time()
        sim = simulate(P, v, PATHS[path][0], **kw)
        name = f"{path} {P:.0f} W {v:.2f} m/s" + (" week1" if split == "week1" else "")
        sim.update(split=split, path=path, name=name)
        run = RunGraph.from_simulation(sim, mesh_for_run(sim, 0))
        runs.append(run)
        print(f"  {run.name:28s} {split:11s} {run.mesh.n_nodes:6d} nodes {run.last + 1:4d} frames "
              f"({run.frame_us:.2f} us)   {time.time() - t0:.0f} s", flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="store_true", help="the processed val and test splits instead of own runs")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--mesh-seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    if not a.dataset:
        print(f"check 8, CI mode: {len(CASES)} simulated runs, strides {STRIDES}")
        runs = ci_runs()
        bins = compute_size_bins(runs, "check 8 CI runs")
        edges = ", ".join(f"{e * 1e6:.4f}" for e in bins["edges_m2"])
        print(f"  size bins from these runs (CI only): edges {edges} mm^2")
        oracle_rows = check_runs(runs, bins)
        print_cadence(runs, oracle_rows)
        torch_path(runs, ci_mode=True)
    else:
        bins = load_size_bins()
        print(f"check 8 on the processed splits {', '.join(a.splits)} (mesh seed {a.mesh_seed}); size bins "
              f"{', '.join(f'{e * 1e6:.4f}' for e in bins['edges_m2'])} mm^2 from the train split")
        val = None
        for split in a.splits:
            runs = load_runs(split, a.mesh_seed)
            partners = load_runs("test_interp", a.mesh_seed) if split == "test_fine" else None
            oracle_rows = check_runs(runs, bins, partners, title=f"{split} ({len(runs)} runs), ")
            print_cadence(runs, oracle_rows)
            if split == "val":
                val = runs
        if val is not None:
            torch_path(val, ci_mode=False)
    print(f"\ncheck 8 took {time.time() - t_start:.0f} s")
    print("PASS" if not failures else f"FAIL: {len(failures)} assertions failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
