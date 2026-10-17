"""Latent-heat plateau of a rollout (plan: model C prototype check).

While a node melts or solidifies, its enthalpy changes by the latent heat (286 kJ/kg) and its temperature stays in
the melting range, 1,878 to 1,928 K. A model with the exact material table (C, D) gets this from H -> T; a dT model (A,
B) has to learn it. Over the scored frames of each rollout (every stride-th frame) this counts, for predictor and
truth alike:
  dwell   node-frames with T in the melting range. Ratio prediction / truth: 1 is the right plateau, below 1 the
          nodes pass through the range too fast (too little latent heat), above 1 they stay too long
  hit     of the truth's node-frames in the range, the fraction where the prediction is in the range as well
  jumps   steps in which a node goes from below the solidus to above the liquidus, or back, in one step. The solver's
          own frames do this too at stride 5 near the laser, so compare the counts, not with zero
Runs that never reach the solidus are skipped. The oracle gives dwell ratio 1, hit 1 and equal jumps (the check of
this script).

--dh: the one-step enthalpy change instead, to see where a model's dH is biased. Every step starts from the true
temperature at every start frame (no rollout, so errors do not accumulate). The predicted dH is H(T_pred) - H(T) with
the run's material table for every predictor, so a dT model (A) and a dH model (C) are compared on the same quantity
(for C it is its own output, up to the table inversion). Node-steps are grouped by the true dH (kJ/kg) and by the
phase at the start of the step (solid, melting range, liquid). Per group: count, mean true and predicted dH, the ratio
of their sums (below 1: too little heating) and the mean absolute error. Only steps that change H by at least 1 kJ/kg
in truth or prediction are counted, so the cold plate does not dilute the groups. The oracle gives ratio 1 and error 0.

--crossings: the one-step steps (again from the true temperature) in which the truth crosses the whole melting range,
from below the solidus to above the liquidus (melting) or back (solidifying). For each direction, where the
prediction ends: across the range as well, inside it, or short of it (still on the starting side). A model that
lands inside instead of across adds plateau frames that the truth does not have. The oracle crosses every time.

Usage (from the repo root):
  python -m analysis.plateau --checkpoint checkpoints/<name>/best.pt [more checkpoints] [--split val]
  python -m analysis.plateau --predictor oracle
  python -m analysis.plateau --dh --checkpoint checkpoints/<name>/best.pt [more checkpoints]
  python -m analysis.plateau --crossings --checkpoint checkpoints/<name>/best.pt [more checkpoints]
"""
import argparse
import itertools
from typing import Any

import numpy as np

from evaluation.rollout import checkpoint_predictor, make_predictor, rollout
from graphs.dataset import load_runs
from solver.ti64 import T_L, T_S

DH_BINS = (-np.inf, 0.0, 10e3, 50e3, 150e3, 300e3, np.inf)          # J/kg, true dH of the step
DH_ACTIVE = 1e3                                                     # J/kg, smallest |dH| counted
PHASES = ("solid", "melting", "liquid")


def plateau_counts(T_prev, T):
    """dwell (node-frames in range), and jumps across the whole range, for frames T (n, nodes) after T_prev (nodes)."""
    inside = (T >= T_S) & (T <= T_L)
    before = np.vstack([T_prev[None], T[:-1]])
    jumps = ((before < T_S) & (T > T_L)) | ((before > T_L) & (T < T_S))
    return inside, int(jumps.sum())


def evaluate(runs, predictor, stride):
    tot = {"dwell_pred": 0, "dwell_true": 0, "hit": 0, "jumps_pred": 0, "jumps_true": 0, "runs": 0}
    for run in runs:
        if not run.melts:
            continue
        frames = stride * np.arange(1, run.last // stride + 1)
        truth = run.T_frames(frames)
        if truth.max() < T_S:
            continue
        r = rollout(run, predictor, stride)
        T0 = run.T(0)
        in_p, j_p = plateau_counts(T0, r["T"])
        in_t, j_t = plateau_counts(T0, truth)
        tot["dwell_pred"] += int(np.sum(in_p))
        tot["dwell_true"] += int(np.sum(in_t))
        tot["hit"] += int(np.sum(in_p & in_t))
        tot["jumps_pred"] += j_p
        tot["jumps_true"] += j_t
        tot["runs"] += 1
    return tot


def report(name, tot):
    if not tot["runs"]:
        print(f"{name}: no run reaches the solidus")
        return
    ratio = tot["dwell_pred"] / tot["dwell_true"]
    hit = tot["hit"] / tot["dwell_true"]
    print(f"{name}: {tot['runs']} melting runs, dwell ratio {ratio:.3f} ({tot['dwell_pred']} / {tot['dwell_true']} "
          f"node-frames), hit {hit:.3f}, jumps {tot['jumps_pred']} (truth {tot['jumps_true']})")


def dh_pairs(runs, predictor, stride):
    """True and predicted one-step dH (J/kg) and the phase index at the start, over the active node-steps."""
    true, pred, phase = [], [], []
    for run in runs:
        if not run.melts:
            continue
        if hasattr(predictor, "reset"):
            predictor.reset(run)
        for k in run.starts(stride):
            T = run.T(k)
            H = run.H_from_T(T)
            dh_t = run.H_from_T(run.T(k + stride)) - H
            dh_p = run.H_from_T(np.asarray(predictor.step(run, T, k, stride), dtype=np.float64)) - H
            on = (np.abs(dh_t) >= DH_ACTIVE) | (np.abs(dh_p) >= DH_ACTIVE)
            true.append(dh_t[on])
            pred.append(dh_p[on])
            phase.append(np.digitize(T[on], (T_S, T_L)))             # 0 below T_S, 1 in range (T_L inclusive), 2 above
    return np.concatenate(true), np.concatenate(pred), np.concatenate(phase)


def crossing_counts(runs, predictor, stride):
    """Per direction ("melting", "solidifying"): how many true one-step crossings the prediction crosses as well,
    ends inside the range, or ends short of it."""
    out = {d: {"across": 0, "inside": 0, "short": 0} for d in ("melting", "solidifying")}
    for run in runs:
        if not run.melts:
            continue
        if hasattr(predictor, "reset"):
            predictor.reset(run)
        for k in run.starts(stride):
            T, T_next = run.T(k), run.T(k + stride)
            up = (T < T_S) & (T_next > T_L)
            down = (T > T_L) & (T_next < T_S)
            if not (up.any() or down.any()):
                continue
            P = np.asarray(predictor.step(run, T, k, stride), dtype=np.float64)
            inside = (P >= T_S) & (P <= T_L)
            for d, m, across in (("melting", up, P > T_L), ("solidifying", down, P < T_S)):
                out[d]["across"] += int((m & across).sum())
                out[d]["inside"] += int((m & inside).sum())
                out[d]["short"] += int((m & ~across & ~inside).sum())
    return out


def report_crossings(name, out):
    parts = []
    for d, c in out.items():
        n = sum(c.values())
        parts.append(f"{d} {n}: " + ", ".join(f"{k} {v / n:.3f}" for k, v in c.items()) if n else f"{d}: none")
    print(f"{name}: true one-step crossings; " + "; ".join(parts))


def report_dh(name, true, pred, phase):
    print(f"{name}: {len(true)} active node-steps (|dH| >= {DH_ACTIVE / 1e3:g} kJ/kg)")
    print(f"  {'group':>18s} {'count':>9s} {'true kJ/kg':>11s} {'pred kJ/kg':>11s} {'ratio':>7s} {'MAE kJ/kg':>10s}")
    rows = [(f"dH {lo / 1e3:g} to {hi / 1e3:g}", (true >= lo) & (true < hi)) for lo, hi in itertools.pairwise(DH_BINS)]
    rows += [(f"start {p}", phase == i) for i, p in enumerate(PHASES)]
    rows.append(("all", np.ones_like(true, dtype=bool)))
    for label, m in rows:
        if not m.any():
            print(f"  {label:>18s} {0:9d}")
            continue
        t, p = true[m], pred[m]
        ratio = p.sum() / t.sum() if t.sum() != 0 else np.nan
        print(f"  {label:>18s} {m.sum():9d} {t.mean() / 1e3:11.2f} {p.mean() / 1e3:11.2f} {ratio:7.3f} "
              f"{np.abs(p - t).mean() / 1e3:10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", default=[])
    ap.add_argument("--predictor", choices=("oracle",))
    ap.add_argument("--dh", action="store_true", help="one-step dH bias instead of the rollout plateau")
    ap.add_argument("--crossings", action="store_true", help="where the prediction ends on true one-step crossings")
    ap.add_argument("--split", default="val")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--mesh-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if not a.checkpoint and not a.predictor:
        ap.error("give --checkpoint or --predictor")
    runs = load_runs(a.split, a.mesh_seed)
    predictors: list[tuple[Any, int]] = [(make_predictor(a.predictor), a.stride)] if a.predictor else []
    for path in a.checkpoint:
        predictor, stride, _ = checkpoint_predictor(path, a.device, a.stride)
        predictors.append((predictor, stride))
    for predictor, stride in predictors:
        if a.crossings:
            report_crossings(predictor.name, crossing_counts(runs, predictor, stride))
        elif a.dh:
            report_dh(predictor.name, *dh_pairs(runs, predictor, stride))
        else:
            report(predictor.name, evaluate(runs, predictor, stride))


if __name__ == "__main__":
    main()
