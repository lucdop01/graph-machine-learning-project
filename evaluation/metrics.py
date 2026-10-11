"""Rollout metrics (plan step 3). numpy only, so check 8 runs in CI without torch.

Every function compares predicted node temperatures with the stored node truth at the same output frames, arrays of
shape (frames, nodes) in K. Frame 0, the shared initial state, is not scored. Functions that are pooled over runs
return sums, so a split summary can add them up (evaluation/rollout.py).

  nRMSE              ||T_hat - T|| / ||T - 300 K|| over every node and frame, each node weighted equally
  melting range      mean |T_hat - T| over (node, frame) where the true T is 1,800-2,000 K, each node weighted equally
  melt-pool IoU      nodes with melted fraction above 0.5, weighted by the lumped node area; soft IoU on the continuous
                     melted fraction, sum A min(phi_hat, phi) / sum A max(phi_hat, phi). Undefined only for an empty
                     union, so a predicted pool on a run without a true pool scores 0.
  solidification     final solidification duration per node (solver/events.py) against a reference sequence
  energy             mesh energy sum rho d A_i H(T_i) against a reference, relative to the laser input on the mesh
"""
import numpy as np

from solver.events import duration
from solver.ti64 import RHO, T0, liquid_fraction

MELT_RANGE = (1800.0, 2000.0)   # K, the melting-range error (proposal)
PHI_POOL = 0.5                  # a node is in the melt pool when its melted fraction is above this
ENERGY_FRAC = 0.05              # energy (a) is gated from the frame where 5% of the run's input has arrived (check 7)


def ratio(num, den):
    return num / den if den > 0 else np.nan


def nrmse_sums(T_hat, T):
    """Sums of squares for nRMSE: (sum (T_hat - T)^2, sum (T - 300 K)^2)."""
    return float(((T_hat - T) ** 2).sum()), float(((T - T0) ** 2).sum())


def nrmse(sse, ssr):
    return float(np.sqrt(ratio(sse, ssr)))


def melting_range_sums(T_hat, T):
    """(sum |T_hat - T|, count) over (node, frame) where the true T is within MELT_RANGE (inclusive)."""
    m = (T >= MELT_RANGE[0]) & (T <= MELT_RANGE[1])
    return float(np.abs(T_hat - T)[m].sum()), int(m.sum())


def iou_sums(T_hat, T, area):
    """Area-weighted intersections and unions (m^2) of the melt pools, summed over frames; hard and soft."""
    phi_hat, phi = liquid_fraction(T_hat), liquid_fraction(T)
    p, q = phi_hat > PHI_POOL, phi > PHI_POOL
    return dict(inter=float(((p & q) * area).sum()), union=float(((p | q) * area).sum()),
                inter_soft=float((np.minimum(phi_hat, phi) * area).sum()),
                union_soft=float((np.maximum(phi_hat, phi) * area).sum()))


def peak_pool_area(T, area):
    """Largest melt-pool area (m^2, nodes with melted fraction above 0.5) over the given frames."""
    return float(((liquid_fraction(T) > PHI_POOL) * area).sum(axis=1).max())


SOL_COUNTS = ("melted_ref", "melted", "compared", "missed", "spurious", "unfinished_ref", "unfinished")


def solidification(ev, ref):
    """Final solidification duration of ev against ref, both solver/events.py results at the same nodes.
    Returns the relative errors (d_ev - d_ref) / d_ref over nodes where both define a duration, and counts:
      melted_ref, melted      nodes fully melted at least once (rose through the liquidus) in ref, in ev
      compared                nodes where both define a duration
      missed                  fully melted in ref, never above the liquidus in ev (lost between the sampled times)
      spurious                fully melted in ev, never in ref
      unfinished_ref, unfinished   fully melted but not yet solid at the end of the sequence, so no duration
                              (for example still solidifying at the rollout horizon)"""
    d_ref, d_ev = duration(ref), duration(ev)
    m_ref, m_ev = ref["n_melt"] >= 1, ev["n_melt"] >= 1
    both = np.isfinite(d_ref) & np.isfinite(d_ev)
    rel = (d_ev[both] - d_ref[both]) / d_ref[both]
    counts = dict(melted_ref=int(m_ref.sum()), melted=int(m_ev.sum()), compared=int(both.sum()),
                  missed=int((m_ref & ~m_ev).sum()), spurious=int((m_ev & ~m_ref).sum()),
                  unfinished_ref=int((m_ref & ~np.isfinite(d_ref)).sum()),
                  unfinished=int((m_ev & ~np.isfinite(d_ev)).sum()))
    return rel, counts


def rel_summary(rel):
    """Median signed and median absolute relative error (NaN if no node is compared)."""
    if rel.size == 0:
        return dict(med=np.nan, med_abs=np.nan)
    return dict(med=float(np.median(rel)), med_abs=float(np.median(np.abs(rel))))


def mesh_energy(T, H_from_T, area, d):
    """Mesh energy (J) per frame: sum over nodes of rho d A_i H(T_i). T has shape (frames, nodes)."""
    return RHO * d * (H_from_T(T) * area).sum(axis=-1)


def energy_errors(e, e_ref, cum, total):
    """Energy difference dE = e - e_ref (J, one value per output frame), as in check 7 but relative to the laser input
    on the mesh. cum: input so far at the same frames; total: the run's total input.
      a    max |dE| / input so far, over frames where the input so far is at least 5% of the total
      b    max |dE| / total input, over every frame
      end  signed dE / total input at the last frame"""
    if not total > 0:
        return dict(a=np.nan, b=np.nan, end=np.nan)
    dE = np.asarray(e, float) - np.asarray(e_ref, float)
    late = cum >= ENERGY_FRAC * total
    return dict(a=float(np.max(np.abs(dE[late]) / cum[late])) if late.any() else np.nan,
                b=float(np.max(np.abs(dE)) / total), end=float(dE[-1] / total))
