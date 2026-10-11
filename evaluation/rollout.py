"""Rollout and evaluation harness (plan step 3).

A predictor gets the true initial state (frame 0, 300 K everywhere), the mesh and the laser energy of each step, and
predicts the whole run from its own outputs. Its outputs are compared with the stored node truth (graphs/preprocess.py)
at every output frame. Checkpoints are selected on the validation rollout error, selection_score().

Predictor interface:
  step(run, T, k, stride) -> T_next   node temperature (K, numpy, shape (nodes,)) at frame k + stride, from the
                                      predictor's own state T at frame k. A model may use run.mesh, run.V,
                                      run.laser(k, stride), run.t, run.phi and the material table (run.H_from_T,
                                      run.T_from_H), never the stored truth. Models C and D convert H to T inside
                                      their wrapper (evaluation/torch_predictor.py).
  reset(run)                          optional, called before each rollout
  name                                used in reports and output file names
Reference predictors (they read the truth on purpose):
  oracle        the true frame k + s. Every metric is exactly zero.
  persistence   T_next = T. From the cold start it stays at 300 K, so its nRMSE is ||T_0 - T|| / ||T - 300 K||, which
                is 1 when frame 0 is exactly 300 K.
  offset        the true frame k + s plus 10 K everywhere
  cross         test_fine only: the frames of the paired 10 um test_interp run (same P and v, from the attributes) as
                predictions of the 5 um truth. Gives the 10 vs 5 um solver gap for every metric.

Rollout: steps of s saved frames from frame 0 up to the last multiple of s at or before the final frame. Leftover
frames (fewer than s) are not scored. A non-finite prediction stops the rollout with an error naming the step.

Metrics per run (evaluation/metrics.py), at the output frames k >= 1 against the stored node truth:
  nrmse            ||T_hat - T|| / ||T - 300 K|| over every node and output frame, each node weighted equally
  melt_mae         mean |T_hat - T| (K) over (node, frame) where the true T is 1,800-2,000 K
  iou, iou_soft    melt pool (melted fraction above 0.5), node areas, intersections and unions summed over the output
                   frames; soft IoU on the continuous melted fraction. NaN only when the union is empty.
  sol_*            final solidification duration (solver/events.py rules) of the prediction against the truth at the
                   same output times, so a perfect model scores 0: median signed (sol_med) and absolute (sol_med_abs)
                   relative error over nodes where both define a duration, and counts of fully melted nodes that are
                   missed, spurious or unfinished (melted but not yet solid at the rollout horizon)
  energy_a/b/end   mesh energy sum rho d A_i H(T_i) of the prediction minus that of the truth. As check 7, but relative
                   to the laser input on the mesh (sum V_i E_i), since processed files do not store the solver's input
                   (the two differ by at most 0.17% per step): a = max |dE| / input so far, from the frame where 5% of
                   the run's input has arrived; b = max |dE| / total input; end = signed dE / total input at the last
                   output frame
  budget_a/b/end   the same for the prediction's mesh energy minus the laser input so far. The oracle shows the
                   preprocessing mismatch; model D should give about 0.
Diagnostics next to them:
  cad_*            cadence error of solidification: the truth at the output times against the truth at every saved
                   frame (same rules and counts). cad_unfinished counts nodes still solidifying at the horizon, kept
                   apart from nodes the coarser sampling missed.
  gap_*            test_fine only: the paired 10 um run against the 5 um truth at the output times
  peak_area_mm2, size_bin   the true run's largest melt pool over every saved frame, and its size bin
week1 has no melting (constant properties) and saves a frame every 26.52 us: melt_mae, IoU, solidification and the
size bin are not computed for it.

Melt-pool size bins: edges are the terciles of the peak melt-pool area over the train runs (mesh seed 0) that have a
pool. They are computed once (--size-bins) and saved to data/graphs/size_bins.json with the training-data fingerprint
of the statistics; a mismatch raises an error. Runs without a pool form the bin "none": their IoU is 0 if the
prediction has a pool and undefined otherwise, so the summary also counts the runs with a predicted pool.

Split summary: nrmse_mean is the mean over runs (the selection score); melt_mae is pooled over every (node, frame) of
the split; IoU pools intersections and unions over the runs, overall and per size bin; solidification medians are
taken over every compared node of the split and the counts are summed; energy and budget give the median and the
maximum over runs (for the signed end value, the median and the largest absolute value).

Usage (from the repo root, after python -m graphs.preprocess):
  python -m evaluation.rollout --size-bins                                   size bins from the train split, once
  python -m evaluation.rollout --split val --stride 5 --predictor oracle     per-run CSV and summary JSON
  python -m evaluation.rollout --split test_fine --predictor cross           10 vs 5 um gap, every metric
  python -m evaluation.rollout --benchmark --splits val test_path --runs 2 --predictor persistence
  options: --mesh-seed 1 (new-mesh test), --out results/eval
Output: results/eval/<predictor>_<split>[_seed<k>]_s<stride>.csv (one row per run) and .json (split summary).
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from solver.events import events_from_frames
from solver.solver import simulate
from solver.ti64 import CONST_PROPS
from data.generate import COOL
from graphs.mesh import PATHS, mesh_name
from graphs.preprocess import OUT, processed_file
from graphs.dataset import (INDEX, RunGraph, check_stride, current_train_fingerprint, load_runs, run_name,
                            train_fingerprint)
from evaluation import metrics as M

BIN_NAMES = ("small", "medium", "large")
SPLITS = ("val", "test_interp", "test_extrap", "test_path", "test_fine", "test_large")


# ---------------------------------------------------------------- predictors

class Oracle:
    name = "oracle"

    def step(self, run, T, k, stride):
        return run.T(k + stride)


class Persistence:
    name = "persistence"

    def step(self, run, T, k, stride):
        return np.array(T, dtype=np.float64)


class Offset:
    name = "offset"

    def __init__(self, dT=10.0):
        self.dT = float(dT)

    def step(self, run, T, k, stride):
        return run.T(k + stride) + self.dT


class CrossSolver:
    """test_fine: the paired 10 um run's frames as predictions of the 5 um truth (README evaluation note 3)."""
    name = "cross"

    def __init__(self, partners):
        self.partners, self.pair = partners, None

    def reset(self, run):
        self.pair = find_pair(run, self.partners)

    def step(self, run, T, k, stride):
        return self.pair.T(k + stride)


def find_pair(run, partners):
    """The run among partners with the same laser power and speed (from the attributes, not the index), checked to be
    comparable: same path, thickness, laser-off time, mesh (name, seed, fingerprint), interpolation and saved times."""
    P, v = float(run.attrs["P"]), float(run.attrs["v"])
    hits = [p for p in partners if np.isclose(float(p.attrs["P"]), P, rtol=1e-12, atol=0)
            and np.isclose(float(p.attrs["v"]), v, rtol=1e-12, atol=0)]
    if len(hits) != 1:
        raise ValueError(f"{run.name}: {len(hits)} partner runs with P = {P} W and v = {v} m/s, need exactly 1")
    p = hits[0]
    same = ("path", "mesh", "mesh_seed", "mesh_fp", "interp")
    problems = [f"{key} {p.attrs[key]} vs {run.attrs[key]}" for key in same if str(p.attrs[key]) != str(run.attrs[key])]
    problems += [f"{key} {p.attrs[key]} vs {run.attrs[key]}" for key in ("d", "laser_off")
                 if not np.isclose(float(p.attrs[key]), float(run.attrs[key]), rtol=1e-12, atol=0)]
    if p.last != run.last or not np.allclose(p.t, run.t, rtol=1e-9, atol=0):
        problems.append(f"saved times differ ({p.last + 1} vs {run.last + 1} frames)")
    if problems:
        raise ValueError(f"{run.name} and its pair {p.name} are not comparable: " + "; ".join(problems))
    return p


def make_predictor(name, partners=None):
    if name == "oracle":
        return Oracle()
    if name == "persistence":
        return Persistence()
    if name == "offset":
        return Offset(10.0)
    if name == "cross":
        return CrossSolver(partners)
    raise ValueError(f"unknown predictor {name!r}")


# ---------------------------------------------------------------- rollout and metrics

def horizon(run, stride):
    """The last scored frame: the last multiple of the stride at or before the final frame."""
    stride = check_stride(stride)
    return run.last // stride * stride


def rollout(run, predictor, stride):
    """Predict the whole run from frame 0. Returns frames (output frame indices), t (s), T (frames, nodes) in K and
    wall (s, predictor time only)."""
    stride = check_stride(stride)
    n = run.last // stride
    if n < 1:
        raise ValueError(f"{run.name}: {run.last + 1} frames leave no step of {stride} frames")
    if hasattr(predictor, "reset"):
        predictor.reset(run)
    out = np.empty((n, run.mesh.n_nodes))
    T = run.T(0)
    wall = 0.0
    for j in range(n):
        k = j * stride
        t0 = time.perf_counter()
        T = np.asarray(predictor.step(run, T, k, stride), dtype=np.float64)
        wall += time.perf_counter() - t0
        if T.shape != out.shape[1:]:
            raise ValueError(f"{predictor.name} returned shape {T.shape} at step {j + 1}, expected {out.shape[1:]}")
        if not np.isfinite(T).all():
            raise FloatingPointError(f"{predictor.name} returned a non-finite temperature at step {j + 1} of {n} "
                                     f"(frame {k} -> {k + stride}) of {run.name}")
        out[j] = T
        T = out[j].copy()                           # the predictor gets its own output, never a shared buffer
    frames = stride * np.arange(1, n + 1)
    return dict(frames=frames, t=run.t[frames], T=out, wall=wall, stride=stride)


def run_cache(run):
    """Per-run values that do not depend on the predictor: the truth's events at every saved frame and its peak
    melt-pool area. Kept on the run object."""
    if getattr(run, "_eval_cache", None) is None:
        T_all = run.T_frames(np.arange(run.last + 1))
        run._eval_cache = dict(events=events_from_frames(T_all, run.t),
                               peak=M.peak_pool_area(T_all, run.mesh.area) if run.melts else np.nan)
    return run._eval_cache


def size_bin(peak, edges):
    if edges is None or np.isnan(peak):
        return ""
    if peak <= 0:
        return "none"
    return BIN_NAMES[int(np.searchsorted(edges, peak, side="left"))]


def _sol(prefix, rel, counts):
    return {prefix + "med": M.rel_summary(rel)["med"], prefix + "med_abs": M.rel_summary(rel)["med_abs"],
            **{prefix + c: counts[c] for c in M.SOL_COUNTS}}


def evaluate_run(run, predictor, stride, edges=None, pair=None):
    """One rollout scored with every metric. edges: size-bin edges (m^2) or None; pair: the 10 um run of a test_fine
    run, for the gap diagnostic. Returns (row, parts): row for the per-run CSV, parts for pooling over a split."""
    ro = rollout(run, predictor, stride)
    f = ro["frames"]
    T_hat, T = ro["T"], run.T_frames(f)
    row = dict(run=run.name, split=run.attrs["split"], mesh=f"{run.mesh.name} seed {run.mesh.seed}",
               predictor=predictor.name, stride=ro["stride"], step_us=ro["stride"] * run.frame_us,
               frames_scored=len(f), horizon=int(f[-1]), last_frame=run.last, wall_s=ro["wall"])
    parts = dict(melts=run.melts)

    sse, ssr = M.nrmse_sums(T_hat, T)
    row["nrmse"], parts["nrmse"] = M.nrmse(sse, ssr), (sse, ssr)

    d = float(run.attrs["d"])
    cum = run.laser_input()
    e_hat = M.mesh_energy(T_hat, run.H_from_T, run.mesh.area, d)
    e_true = M.mesh_energy(T, run.H_from_T, run.mesh.area, d)
    row.update({"energy_" + k: v for k, v in M.energy_errors(e_hat, e_true, cum[f], cum[-1]).items()})
    row.update({"budget_" + k: v for k, v in M.energy_errors(e_hat, cum[f], cum[f], cum[-1]).items()})

    if run.melts:
        cache = run_cache(run)
        s, c = M.melting_range_sums(T_hat, T)
        row["melt_mae"], row["melt_points"], parts["melt"] = M.ratio(s, c), c, (s, c)
        iou = M.iou_sums(T_hat, T, run.mesh.area)
        row["iou"], row["iou_soft"] = M.ratio(iou["inter"], iou["union"]), M.ratio(iou["inter_soft"], iou["union_soft"])
        row["union_mm2"], row["union_soft_mm2"] = iou["union"] * 1e6, iou["union_soft"] * 1e6
        row["peak_area_mm2"], row["size_bin"] = cache["peak"] * 1e6, size_bin(cache["peak"], edges)
        parts["iou"], parts["bin"] = iou, row["size_bin"]

        t_out = np.r_[run.t[0], ro["t"]]
        T0 = run.T(0)[None]
        ev_true = events_from_frames(np.vstack([T0, T]), t_out)
        ev_hat = events_from_frames(np.vstack([T0, T_hat]), t_out)
        parts["sol"] = M.solidification(ev_hat, ev_true)
        parts["cad"] = M.solidification(ev_true, cache["events"])
        row.update(_sol("sol_", *parts["sol"]))
        row.update(_sol("cad_", *parts["cad"]))
        if pair is not None:
            ev_pair = events_from_frames(np.vstack([pair.T(0)[None], pair.T_frames(f)]), t_out)
            parts["gap"] = M.solidification(ev_pair, ev_true)
            row.update(_sol("gap_", *parts["gap"]))
    return row, parts


def _pool_sol(parts, key):
    got = [p[key] for p in parts if key in p]
    if not got:
        return None
    rel = np.concatenate([g[0] for g in got])
    out = M.rel_summary(rel)
    out.update({c: int(sum(g[1][c] for g in got)) for c in M.SOL_COUNTS})
    return out


def _pool_iou(parts):
    s = {k: sum(p["iou"][k] for p in parts) for k in ("inter", "union", "inter_soft", "union_soft")}
    return dict(runs=len(parts), iou=M.ratio(s["inter"], s["union"]),
                iou_soft=M.ratio(s["inter_soft"], s["union_soft"]),
                runs_with_union=sum(p["iou"]["union"] > 0 for p in parts))


def _med_max(rows, key):
    """Median and maximum over runs; for a signed value (end) the maximum of the absolute value."""
    v = np.array([r[key] for r in rows], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return dict(median=np.nan, max=np.nan)
    return dict(median=float(np.median(v)), max=float(np.abs(v).max()))


def summarise(rows, parts, bins=None):
    """Split summary (see the module docstring)."""
    nr = np.array([r["nrmse"] for r in rows], float)
    s = dict(predictor=rows[0]["predictor"], split=rows[0]["split"], mesh=sorted({r["mesh"] for r in rows}),
             stride=rows[0]["stride"], step_us=rows[0]["step_us"], runs=len(rows),
             nrmse_mean=float(nr.mean()), nrmse_median=float(np.median(nr)), nrmse_max=float(nr.max()))
    melt = [p for p in parts if p["melts"]]
    if melt:
        s["melt_mae"] = M.ratio(sum(p["melt"][0] for p in melt), sum(p["melt"][1] for p in melt))
        s["melt_points"] = int(sum(p["melt"][1] for p in melt))
        s["iou"] = _pool_iou(melt)
        if bins is not None:
            s["iou_by_bin"] = {}
            for name in BIN_NAMES + ("none",):
                group = [p for p in melt if p["bin"] == name]
                if group:
                    s["iou_by_bin"][name] = _pool_iou(group)
            if "none" in s["iou_by_bin"]:      # the true run has no pool: a union means a predicted pool
                s["iou_by_bin"]["none"]["runs_with_predicted_pool"] = s["iou_by_bin"]["none"].pop("runs_with_union")
        for key, name in (("sol", "solidification"), ("cad", "cadence"), ("gap", "gap_10_vs_5um")):
            pooled = _pool_sol(melt, key)
            if pooled is not None:
                s[name] = pooled
    for group in ("energy", "budget"):
        s[group] = {k: _med_max(rows, f"{group}_{k}") for k in ("a", "b", "end")}
    s["wall_s"] = float(sum(r["wall_s"] for r in rows))
    s["steps"] = int(sum(r["frames_scored"] for r in rows))
    if bins is not None:
        s["size_bins"] = bins
    return s


def evaluate_split(runs, predictor, stride, bins=None, partners=None):
    """Every run of a split. bins: size_bins.json content (or None); partners: the 10 um runs for test_fine's gap
    diagnostic (or None). Returns (rows, summary)."""
    edges = None if bins is None else np.asarray(bins["edges_m2"])
    rows, parts = [], []
    for run in runs:
        pair = find_pair(run, partners) if partners is not None else None
        r, p = evaluate_run(run, predictor, stride, edges, pair)
        rows.append(r)
        parts.append(p)
    return rows, summarise(rows, parts, bins)


def selection_score(runs, predictor, stride):
    """Checkpoint selection: mean rollout nRMSE over the runs (the 20 val runs) at the model's stride."""
    scores = []
    for run in runs:
        ro = rollout(run, predictor, stride)
        scores.append(M.nrmse(*M.nrmse_sums(ro["T"], run.T_frames(ro["frames"]))))
    return float(np.mean(scores))


# ---------------------------------------------------------------- size bins

def size_bins_file(root=OUT):
    return os.path.join(root, "size_bins.json")


def compute_size_bins(runs, source):
    """Terciles of the peak melt-pool area over the runs that have a pool."""
    peaks = np.array([run_cache(r)["peak"] for r in runs if r.melts])
    pos = np.sort(peaks[peaks > 0])
    if pos.size < 3:
        raise ValueError(f"only {pos.size} runs with a melt pool, need at least 3 for terciles")
    return dict(edges_m2=np.quantile(pos, [1 / 3, 2 / 3]).tolist(), source=source, runs=int(peaks.size),
                runs_with_pool=int(pos.size),
                definition="terciles of the peak melt-pool area (melted fraction above 0.5, node areas, every saved "
                           "frame) over the runs with a pool; small <= edge 1 < medium <= edge 2 < large; "
                           "none = no pool")


def save_size_bins(root=OUT, index=INDEX):
    runs = load_runs("train", 0, root, index, preload=False)
    bins = compute_size_bins(runs, "train split, mesh seed 0")
    bins["train_fingerprint"] = train_fingerprint({r.name: r.attrs for r in runs})
    with open(size_bins_file(root), "w") as f:
        json.dump(bins, f, indent=1)
    return bins


def load_size_bins(root=OUT, index=INDEX):
    """The saved size bins, checked against the current processed train split. Never recomputed silently."""
    fn = size_bins_file(root)
    fix = "; run python -m evaluation.rollout --size-bins"
    if not os.path.exists(fn):
        raise FileNotFoundError(f"{fn} missing" + fix)
    with open(fn) as f:
        bins = json.load(f)
    if bins.get("train_fingerprint") != current_train_fingerprint(root, index):
        raise ValueError(f"{fn} was computed from different training data than the current processed train split" + fix)
    return bins


# ---------------------------------------------------------------- output

CSV_FIELDS = (["run", "split", "mesh", "predictor", "stride", "step_us", "frames_scored", "horizon", "last_frame",
               "size_bin", "peak_area_mm2", "nrmse", "melt_mae", "melt_points", "iou", "iou_soft", "union_mm2"]
              + [p + k for p in ("sol_", "cad_", "gap_") for k in ("med", "med_abs") + M.SOL_COUNTS]
              + [p + k for p in ("energy_", "budget_") for k in ("a", "b", "end")] + ["wall_s"])


def _clean(x):
    """NaN -> None for strict JSON."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def write_results(rows, summary, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, stem + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, CSV_FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(_clean(summary), f, indent=1)


def fmt(x, pct=False, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{100 * x:.{digits}g}%" if pct else f"{x:.{digits}g}"


def sol_text(s):
    if s is None:
        return "-"
    return (f"median {fmt(s['med'], True)} (|.| {fmt(s['med_abs'], True)}) over {s['compared']} nodes; missed "
            f"{s['missed']}, spurious {s['spurious']}, unfinished {s['unfinished']} (ref {s['unfinished_ref']}) "
            f"of {s['melted_ref']} melted")


def print_summary(s):
    print(f"\n{s['predictor']} on {s['split']} ({', '.join(s['mesh'])}), stride {s['stride']} "
          f"({s['step_us']:.1f} us), {s['runs']} runs, {s['steps']} steps")
    print(f"  nRMSE mean {fmt(s['nrmse_mean'], digits=4)} (selection score), median {fmt(s['nrmse_median'])}, "
          f"max {fmt(s['nrmse_max'])}")
    if "melt_mae" in s:
        print(f"  melting-range MAE {fmt(s['melt_mae'])} K over {s['melt_points']} (node, frame) pairs")
        print(f"  IoU {fmt(s['iou']['iou'], digits=4)}, soft {fmt(s['iou']['iou_soft'], digits=4)}")
        for name, b in s.get("iou_by_bin", {}).items():
            extra = f", {b['runs_with_predicted_pool']} with a predicted pool" if name == "none" else ""
            print(f"    {name:6s} {b['runs']:3d} runs: IoU {fmt(b['iou'], digits=4)}, "
                  f"soft {fmt(b['iou_soft'], digits=4)}" + extra)
        for key in ("solidification", "cadence", "gap_10_vs_5um"):
            if key in s:
                print(f"  {key:15s} {sol_text(s[key])}")
    for g in ("energy", "budget"):
        e = s[g]
        print(f"  {g:7s} (a) median {fmt(e['a']['median'], True)} max {fmt(e['a']['max'], True)}; (b) median "
              f"{fmt(e['b']['median'], True)} max {fmt(e['b']['max'], True)}; "
              f"end median {fmt(e['end']['median'], True)}")
    print(f"  predictor time {s['wall_s']:.2f} s")


# ---------------------------------------------------------------- speedup benchmark

def simulate_like(row):
    """Re-run the solver with the generator's settings (data/generate.py) for one row of data/splits.json."""
    split = row["split"]
    large = split == "test_large"
    wp = PATHS[mesh_name(row["path"], 4e-3 if large else 2e-3, 2e-3 if large else 1e-3)][0]
    kw = dict(props=CONST_PROPS) if split == "week1" else dict(COOL)
    if np.isclose(float(row["dx"]), 5e-6):
        kw.update(nx=400, ny=200, dx=5e-6, save_every=40)
    elif large:
        kw.update(nx=400, ny=200)
    return simulate(float(row["P"]), float(row["v"]), wp, **kw)


def benchmark(splits, n_runs, predictor, stride, mesh_seed=0, root=OUT, index=INDEX):
    """Solver time against rollout time for the first n_runs runs of each split (both on this machine). The speedup is
    only meaningful for a model; truth-based predictors are reported as n/a."""
    truth_based = isinstance(predictor, (Oracle, Offset, CrossSolver))
    rows = json.load(open(index))
    out = []
    for split in splits:
        for row in [r for r in rows if r["split"] == split][:n_runs]:
            t0 = time.perf_counter()
            sim = simulate_like(row)
            t_solver = time.perf_counter() - t0
            if len(sim["t"]) != row["frames"]:
                raise RuntimeError(f"{run_name(row)}: re-simulation gave {len(sim['t'])} frames, data/splits.json "
                                   f"says {row['frames']}")
            run = RunGraph(processed_file(run_name(row), mesh_seed, root))
            ro = rollout(run, predictor, stride)
            speed = "n/a" if truth_based else f"{t_solver / ro['wall']:.1f}x"
            print(f"{run.name:24s} solver {t_solver:7.2f} s ({run.last} frames)   {predictor.name} rollout "
                  f"{ro['wall']:7.3f} s ({len(ro['frames'])} steps of {stride})   speedup {speed}", flush=True)
            out.append(dict(run=run.name, solver_s=t_solver, rollout_s=ro["wall"], steps=len(ro["frames"])))
    return out


# ---------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--predictor", default="oracle", choices=("oracle", "persistence", "offset", "cross"))
    ap.add_argument("--mesh-seed", type=int, default=0)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--size-bins", action="store_true", help="compute the size bins from the train split and exit")
    ap.add_argument("--benchmark", action="store_true", help="time solver vs rollout")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), help="splits for --benchmark")
    ap.add_argument("--runs", type=int, default=2, help="runs per split for --benchmark")
    a = ap.parse_args()

    if a.size_bins:
        b = save_size_bins()
        print(f"{size_bins_file()}: {b['runs_with_pool']} of {b['runs']} train runs have a melt pool; edges "
              + ", ".join(f"{e * 1e6:.4f} mm^2" for e in b["edges_m2"]))
        return
    stride = check_stride(a.stride)
    partners = None
    if a.predictor == "cross" or a.split == "test_fine":
        if a.split != "test_fine":
            raise SystemExit("--predictor cross is defined for --split test_fine only")
        partners = load_runs("test_interp", a.mesh_seed)
    predictor = make_predictor(a.predictor, partners)
    if a.benchmark:
        benchmark(a.splits, a.runs, predictor, stride, a.mesh_seed)
        return
    runs = load_runs(a.split, a.mesh_seed)
    bins = load_size_bins()
    rows, summary = evaluate_split(runs, predictor, stride, bins, partners)
    summary["mesh_seed"] = a.mesh_seed
    stem = f"{a.predictor}_{a.split}" + (f"_seed{a.mesh_seed}" if a.mesh_seed else "") + f"_s{stride}"
    write_results(rows, summary, a.out, stem)
    for r in rows:
        print(f"  {r['run']:24s} nRMSE {fmt(r['nrmse'], digits=4):>9s}  melt MAE {fmt(r.get('melt_mae')):>7s} K  "
              f"IoU {fmt(r.get('iou'), digits=4):>6s}  sol {fmt(r.get('sol_med'), True):>7s}  "
              f"cadence {fmt(r.get('cad_med'), True):>7s}  energy (a) {fmt(r['energy_a'], True):>7s}  "
              f"budget (a) {fmt(r['budget_a'], True):>7s}")
    print_summary(summary)
    print(f"\nwrote {os.path.join(a.out, stem)}.csv and .json")


if __name__ == "__main__":
    main()
