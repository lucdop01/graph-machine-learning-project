"""PyTorch Geometric samples from processed runs (plan step 2).

A sample is one model step of one run: the state at saved frame k and the state at frame k + stride, on the run's
mesh. The stride is a whole number of saved frames, at least 1. Every split except week1 saves a frame every 13.34 us,
so 5 frames = 66.7 us (model step) and 2 frames = 26.7 us (E3 small step). week1 saves every 26.52 us, because its
constant properties allow a 2x larger solver step. Every start frame from 0 to the last frame minus the stride gives a
sample, cooling included.

Node features x, normalised with training statistics:
  T          temperature at frame k (K)
  phi        melted fraction at frame k (solver/ti64.py liquid_fraction; 0 for week1, which has no melting)
  E          laser energy per volume received during the step (J/m^3): the sum of dt * q over the solver steps
  boundary   1 on the plate edge, 0 elsewhere (not normalised)
Edge features edge_attr, normalised: x_source - x_target (2 values) and its length. edge_index[0] is the source.
Raw fields for targets, losses and physics:
  T, T_next (K), H, H_next (J/kg, from the material table), E (J/m^3), V (node volume, m^3), pos (m), boundary,
  rev_edge (index of each edge's reverse, for antisymmetric edge fluxes; offset correctly when batched),
  t (time of frame k, s), k, stride, run (run name).
Model C's extra features (cp(T), harmonic-mean conduction flux) follow from T and the edge vectors at load time.

Normalisation statistics come from the train split on mesh seed 0 only and are saved to
data/graphs/norm_stride<s>.json: mean and standard deviation of T, phi and E, of the edge features, and of the
one-step changes dT and dH (the targets of the temperature and enthalpy models). Every node of every sample counts
once. Other splits and other mesh seeds read the saved file, so no test data reaches the statistics.

Consistency checks (each raises an error that names the fix):
  processed file   same mesh parameters and mesh fingerprint as the current mesh, array shapes that match it, and
                   not older than its run file when the run file is present (graphs/preprocess.py processed_problems)
  statistics       same mesh parameters, stride and interpolation, the same frame interval as the dataset, and a
                   fingerprint of the training data (run names and data hashes of the train split, mesh seed 0) equal
                   to that of the current training files. Statistics are never recomputed silently, since a trained
                   model must keep the statistics it was trained with; recompute with --recompute.
Statistics passed in by hand (stats=...) are checked for stride, interpolation and frame interval only.

Usage (after python -m graphs.preprocess):
  python -m graphs.dataset --stride 5            train statistics (computed once) and sample counts per split
  from graphs.dataset import MeltPoolGraphs
  train = MeltPoolGraphs("train", stride=5)       # computes and saves the statistics on first use
  test = MeltPoolGraphs("test_path", stride=5)    # reads them
  loader = torch_geometric.loader.DataLoader(train, batch_size=4, shuffle=True)
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import hashlib
import numbers

from solver.ti64 import liquid_fraction
from graphs.mesh import PARAMS, get_mesh
from graphs.preprocess import OUT, material, processed_file, processed_problems, read_processed_header

NODE_FEATURES = ("T", "phi", "E", "boundary")
EDGE_FEATURES = ("dx", "dy", "dist")
TARGETS = ("dT", "dH")
INDEX = "data/splits.json"


def check_stride(stride):
    if isinstance(stride, bool) or not isinstance(stride, numbers.Integral) or stride < 1:
        raise ValueError(f"stride must be a whole number of saved frames, at least 1, not {stride!r}")
    return int(stride)


def run_name(row):
    return os.path.splitext(os.path.basename(row["file"]))[0]


def train_fingerprint(attrs_by_run):
    """Hash of the training data behind the statistics: run names, data hashes, interpolation and mesh."""
    keys = ("data_hash", "interp", "mesh", "mesh_seed", "mesh_fp")
    rows = sorted([name] + [str(a.get(k)) for k in keys] for name, a in attrs_by_run.items())
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:16]


def current_train_fingerprint(root=OUT, index=INDEX):
    """Fingerprint of the processed train split on mesh seed 0, from the file attributes only."""
    attrs, missing = {}, []
    for row in json.load(open(index)):
        if row["split"] == "train":
            fn = processed_file(run_name(row), 0, root)
            if os.path.exists(fn):
                attrs[run_name(row)] = read_processed_header(fn)[0]
            else:
                missing.append(fn)
    if missing:
        raise FileNotFoundError(f"{len(missing)} processed train runs missing (first {missing[0]}), so the training "
                                f"statistics cannot be checked; run python -m graphs.preprocess --splits train")
    return train_fingerprint(attrs)


class GraphSample(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "rev_edge":                       # indexes edges, so it shifts by the edge count when batched
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


class RunGraph:
    """One processed run (graphs/preprocess.py) and its mesh. Also the entry point for rollouts (plan step 3)."""

    def __init__(self, fn, preload=True):
        self.file = fn
        self.name = os.path.splitext(os.path.basename(fn))[0]
        self.attrs, shape_T, shape_E, n_frames = read_processed_header(fn)
        problems = processed_problems(fn, self.attrs, shape_T, shape_E, n_frames)
        if problems:
            raise ValueError(f"{fn}: " + "; ".join(problems) + ". Rerun python -m graphs.preprocess --force "
                             f"--splits {self.attrs.get('split', '<split>')} --mesh-seed {self.attrs.get('mesh_seed', 0)}")
        with h5py.File(fn, "r") as f:
            self.t = f["t"][:]
            self._T = f["T"][:] if preload else None
            self._E = f["E_frame"][:] if preload else None
        self.mesh = get_mesh(self.attrs["mesh"], int(self.attrs["mesh_seed"]))
        self.last = len(self.t) - 1                 # index of the final (fully solid) frame
        self.frame_us = float(self.t[1] * 1e6)       # 13.34 us, 26.52 us for week1
        self.H_from_T, self.T_from_H = material(self.attrs["split"])
        self.melts = self.attrs["split"] != "week1"
        self.V = self.mesh.area * float(self.attrs["d"])

    def _read(self, key, a, b):
        cached = self._T if key == "T" else self._E
        if cached is not None:
            return cached[a:b]
        with h5py.File(self.file, "r") as f:
            return f[key][a:b]

    def T(self, k):
        """Node temperature at frame k (K)."""
        return self._read("T", k, k + 1)[0].astype(np.float64)

    def laser(self, k, stride):
        """Laser energy per volume (J/m^3) of the model step from frame k to frame k + stride."""
        stride = check_stride(stride)
        if not 0 <= k <= self.last - stride:
            raise IndexError(f"no model step of {stride} frames starts at frame {k} (last frame {self.last})")
        return self._read("E_frame", k, k + stride).astype(np.float64).sum(axis=0)

    def phi(self, T):
        return liquid_fraction(T) if self.melts else np.zeros_like(T)

    def starts(self, stride):
        return range(0, self.last - check_stride(stride) + 1)


def stats_file(stride, root=OUT):
    return os.path.join(root, f"norm_stride{stride}.json")


def compute_stats(runs, stride):
    """Mean and standard deviation over every node of every sample of the given (training) runs."""
    stride = check_stride(stride)
    frames = {round(r.frame_us, 6) for r in runs}
    if len(frames) != 1:
        raise ValueError(f"runs with different frame intervals {sorted(frames)} us cannot share statistics")
    acc = {key: np.zeros(3) for key in ("T", "phi", "E") + TARGETS}   # count, sum, sum of squares

    def add(key, x):
        acc[key] += (x.size, x.sum(), (x ** 2).sum())

    n_samples = 0
    for run in runs:
        T = run._read("T", 0, run.last + 1).astype(np.float64)
        Ec = np.concatenate([np.zeros((1, T.shape[1])), np.cumsum(run._read("E_frame", 0, run.last).astype(np.float64), axis=0)])
        k = np.arange(run.last - stride + 1)
        add("T", T[k])
        add("phi", run.phi(T[k]))
        add("E", Ec[k + stride] - Ec[k])
        add("dT", T[k + stride] - T[k])
        H = run.H_from_T(T)
        add("dH", H[k + stride] - H[k])
        n_samples += len(k)
    def ms(a):   # a constant feature (phi when no training run melts, e.g. week1) gets std 1, not 0
        std = float(np.sqrt(max(a[2] / a[0] - (a[1] / a[0]) ** 2, 0.0)))
        return dict(mean=a[1] / a[0], std=std if std > 0 else 1.0)
    meshes = {(r.mesh.name, r.mesh.seed): r.mesh for r in runs}
    rel = np.concatenate([m.edge_vectors()[0] for m in meshes.values()])
    dist = np.linalg.norm(rel, axis=1)
    return dict(stride=stride, frame_us=runs[0].frame_us, split=sorted({r.attrs["split"] for r in runs}),
                runs=len(runs), samples=n_samples, meshes=[f"{n} seed {s}" for n, s in meshes],
                interp=sorted({r.attrs["interp"] for r in runs}), mesh_params=PARAMS,
                train_fingerprint=train_fingerprint({r.name: r.attrs for r in runs}),
                node={key: ms(acc[key]) for key in ("T", "phi", "E")},
                edge=dict(rel=dict(mean=rel.mean(axis=0).tolist(), std=rel.std(axis=0).tolist()),
                          dist=dict(mean=float(dist.mean()), std=float(dist.std()))),
                target={key: ms(acc[key]) for key in TARGETS})


class MeltPoolGraphs(Dataset):
    """All one-step samples of one split at one stride. Pass stats to override the saved training statistics."""

    def __init__(self, split, stride=5, mesh_seed=0, root=OUT, index=INDEX, stats=None, preload=True):
        stride = check_stride(stride)
        rows = [r for r in json.load(open(index)) if r["split"] == split]
        if not rows:
            raise ValueError(f"no runs of split {split!r} in {index}")
        files = [processed_file(run_name(r), mesh_seed, root) for r in rows]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} processed runs missing (first {missing[0]}); run "
                                    f"python -m graphs.preprocess --splits {split} --mesh-seed {mesh_seed}")
        self.runs = [RunGraph(f, preload) for f in files]
        self.split, self.stride = split, stride
        self.index = [(i, k) for i, run in enumerate(self.runs) for k in run.starts(stride)]
        frames = {round(r.frame_us, 6) for r in self.runs}
        if len(frames) != 1:
            raise ValueError(f"split {split} mixes frame intervals {sorted(frames)} us")
        if stats is None:
            fn = stats_file(stride, root)
            if not os.path.exists(fn):
                if split != "train" or mesh_seed != 0:
                    raise FileNotFoundError(f"{fn} missing: run python -m graphs.dataset --stride {stride} first")
                with open(fn, "w") as f:
                    json.dump(compute_stats(self.runs, stride), f, indent=1)
            with open(fn) as f:
                stats = json.load(f)
            fix = f"; recompute with python -m graphs.dataset --stride {stride} --recompute"
            if stats.get("mesh_params") != PARAMS:
                raise ValueError(f"{fn} was computed for other mesh parameters" + fix)
            if stats.get("train_fingerprint") != current_train_fingerprint(root, index):
                raise ValueError(f"{fn} was computed from different training data than the current processed train "
                                 f"split (data/graphs/seed0)" + fix)
        self._check_stats(stats, stride)
        self.stats = stats
        self._static = {}

    def _check_stats(self, stats, stride):
        interp = sorted({r.attrs["interp"] for r in self.runs})
        if stats.get("stride") != stride:
            raise ValueError(f"statistics are for stride {stats.get('stride')}, not {stride}")
        if stats.get("interp") != interp:
            raise ValueError(f"statistics come from {stats.get('interp')}-interpolated data, split {self.split} "
                             f"is {interp}-interpolated")
        frame_us = self.runs[0].frame_us
        if not np.isclose(stats.get("frame_us", np.nan), frame_us, rtol=1e-6):
            raise ValueError(f"split {self.split} saves a frame every {frame_us:.2f} us, the statistics assume "
                             f"{stats.get('frame_us', float('nan')):.2f} us, so stride {stride} would be a different "
                             f"model step (week1 needs its own statistics, passed as stats=...)")

    def __len__(self):
        return len(self.index)

    def _mesh_tensors(self, run):
        key = (run.mesh.name, run.mesh.seed)
        if key not in self._static:
            m, s = run.mesh, self.stats["edge"]
            rel, dist = m.edge_vectors()
            ea = np.c_[(rel - s["rel"]["mean"]) / s["rel"]["std"], (dist - s["dist"]["mean"]) / s["dist"]["std"]]
            self._static[key] = dict(edge_index=torch.from_numpy(m.edge_index),
                                     edge_attr=torch.tensor(ea, dtype=torch.float32),
                                     rev_edge=torch.from_numpy(m.rev), pos=torch.tensor(m.pos, dtype=torch.float32),
                                     boundary=torch.from_numpy(m.boundary))
        return self._static[key]

    def sample(self, run, k):
        s, n = self.stride, self.stats["node"]
        T0, T1, E = run.T(k), run.T(k + s), run.laser(k, s)
        phi = run.phi(T0)
        st = self._mesh_tensors(run)
        z = lambda x, key: (x - n[key]["mean"]) / n[key]["std"]
        x = np.c_[z(T0, "T"), z(phi, "phi"), z(E, "E"), st["boundary"].numpy()]
        f32 = lambda a: torch.tensor(a, dtype=torch.float32)
        return GraphSample(x=f32(x), **st, T=f32(T0), T_next=f32(T1), H=f32(run.H_from_T(T0)),
                           H_next=f32(run.H_from_T(T1)), E=f32(E), V=f32(run.V), t=float(run.t[k]), k=k, stride=s,
                           run=run.name)

    def __getitem__(self, i):
        r, k = self.index[i]
        return self.sample(self.runs[r], k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test_interp", "test_extrap", "test_path",
                                                    "test_fine", "test_large"])
    ap.add_argument("--recompute", action="store_true", help="recompute the train statistics")
    a = ap.parse_args()
    if a.recompute and os.path.exists(stats_file(a.stride)):
        os.remove(stats_file(a.stride))
    train = MeltPoolGraphs("train", a.stride, preload=False)
    st = train.stats
    print(f"stride {a.stride} ({a.stride * st['frame_us']:.1f} us), statistics from {st['runs']} train runs, "
          f"{st['samples']} samples ({stats_file(a.stride)})")
    for group in ("node", "target"):
        for key, v in st[group].items():
            print(f"  {key:4s} mean {v['mean']:12.4g}  std {v['std']:12.4g}")
    print(f"  edge |dx| mean {st['edge']['dist']['mean'] * 1e6:.2f} um, std {st['edge']['dist']['std'] * 1e6:.2f} um")
    for split in a.splits:
        try:
            ds = MeltPoolGraphs(split, a.stride, preload=False)
        except (ValueError, FileNotFoundError) as e:
            print(f"{split}: skipped ({e})")
            continue
        g = ds[0]
        print(f"{split:12s} {len(ds.runs):4d} runs {len(ds):7d} samples; first sample {g.num_nodes} nodes, "
              f"{g.num_edges} edges, x {tuple(g.x.shape)}, edge_attr {tuple(g.edge_attr.shape)}")
