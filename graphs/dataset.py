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

SampleBuilder builds these inputs from any state T0 at frame k. Training samples pass the true frame k (and the true
frame k + stride as the target); rollouts (evaluation/torch_predictor.py) pass the model's own previous output and get
no T_next or H_next. So the features are built one way only.

Training noise (MeltPoolGraphs noise_std, in K, default 0): Gaussian noise is added to the true temperature at frame k
before the sample is built, so the features (T, phi) and the raw T and H use the noisy state, and the target
T_next - T is the true next frame minus the noisy input: the model learns to correct its own errors (Pfaff et al.).
The noise comes from torch's global random generator (seeded by the training script; a DataLoader seeds each worker).
With noise_std 0 no random number is drawn and the samples are exactly those without noise.

Training augmentation (MeltPoolGraphs augment, default False): each training sample gets one random orthogonal
transform Q, a reflection (y -> -y) with probability 0.5 followed by a rotation by an angle drawn uniformly from
0-360 degrees. Q is applied to the raw edge vectors x_source - x_target before they are normalised, and to pos, so
the whole sample is rotated or reflected: plate, mesh, temperature and laser field together. Edge lengths, node
features (all direction-free), volumes and targets do not change. Conduction is isotropic and the plate edges are
adiabatic, so a transformed sample is as valid as the original; the model can no longer tie its predictions to the
coordinate frame (all train tracks move in +x). Q comes from torch's global random generator after the noise, so a
resumed run draws the same transforms, and with augment False no extra number is drawn. Validation, rollouts and every
evaluation use the samples as they are (SampleBuilder.sample without rotation). The edge statistics stay those of
the untransformed train split (the two component scales differ by 0.2%).

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
                   model must keep the statistics it was trained with; recompute with --recompute. load_stats() reads
                   the saved file with these checks.
Statistics passed in by hand (stats=...) are checked for stride, interpolation and frame interval only (check_stats).

torch and torch_geometric are optional for this module: without them (CI, the numpy-only evaluation harness) RunGraph,
load_runs, load_stats and compute_stats work, and SampleBuilder and MeltPoolGraphs raise an error.

Usage (after python -m graphs.preprocess):
  python -m graphs.dataset --stride 5            train statistics (computed once) and sample counts per split
  from graphs.dataset import MeltPoolGraphs
  train = MeltPoolGraphs("train", stride=5)       # computes and saves the statistics on first use
  test = MeltPoolGraphs("test_path", stride=5)    # reads them
  loader = torch_geometric.loader.DataLoader(train, batch_size=4, shuffle=True)
"""
import argparse
import hashlib
import json
import numbers
import os

import h5py
import numpy as np

try:                                  # optional, see the module docstring; other import errors are not hidden
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
    TORCH_MISSING = None
except ModuleNotFoundError as e:
    if e.name not in ("torch", "torch_geometric"):
        raise
    torch, Dataset, Data = None, object, object
    TORCH_MISSING = str(e)

from solver.ti64 import liquid_fraction
from graphs.mesh import PARAMS, get_mesh
from graphs.preprocess import (OUT, data_hash, grid_of, material, node_laser_energy, node_temperature, processed_file,
                               processed_problems, read_processed_header)

NODE_FEATURES = ("T", "phi", "E", "boundary")
EDGE_FEATURES = ("dx", "dy", "dist")
TARGETS = ("dT", "dH")
INDEX = "data/splits.json"


def require_torch(what):
    if TORCH_MISSING:
        raise ImportError(f"{what} needs torch and torch_geometric ({TORCH_MISSING})")


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
    """One processed run (graphs/preprocess.py) and its mesh. Also the entry point for rollouts (evaluation/)."""

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
        self._setup(get_mesh(self.attrs["mesh"], int(self.attrs["mesh_seed"])))

    @classmethod
    def from_simulation(cls, sim, mesh, interp="T"):
        """A run held in memory, built from simulate() output without processed files (check 8 in CI). sim needs
        'name', 'split' and 'path' set, as in verify/graph_energy.py. The node data is computed and stored as float32
        exactly as graphs.preprocess writes it."""
        if not np.allclose(mesh.waypoints, sim["waypoints"], rtol=0, atol=1e-12):
            raise ValueError(f"{sim['name']}: laser path differs from mesh {mesh.name}")
        self = cls.__new__(cls)
        T = node_temperature(sim, mesh, interp)
        E = node_laser_energy(sim, mesh.pos).astype(np.float32)
        self.file, self.name = None, sim["name"]
        self.attrs = {key: sim[key] for key in ("P", "v", "d", "dt", "dx", "laser_off", "t_end", "terminal_frame",
                                                "path", "split")}
        self.attrs.update(mesh=mesh.name, mesh_seed=mesh.seed, interp=interp, save_every=grid_of(sim)[3],
                          mesh_params=json.dumps(PARAMS, sort_keys=True), mesh_fp=mesh.fingerprint(),
                          data_hash=data_hash(T, E))
        self.t = np.asarray(sim["t"], float)
        self._T, self._E = T, E
        self._setup(mesh)
        return self

    def _setup(self, mesh):
        self.mesh = mesh
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

    def T_frames(self, frames):
        """Node temperature at several frames (K), shape (len(frames), nodes)."""
        frames = np.asarray(frames, dtype=int)
        a = int(frames.min())
        return self._read("T", a, int(frames.max()) + 1)[frames - a].astype(np.float64)

    def laser(self, k, stride):
        """Laser energy per volume (J/m^3) of the model step from frame k to frame k + stride."""
        stride = check_stride(stride)
        if not 0 <= k <= self.last - stride:
            raise IndexError(f"no model step of {stride} frames starts at frame {k} (last frame {self.last})")
        return self._read("E_frame", k, k + stride).astype(np.float64).sum(axis=0)

    def laser_input(self):
        """(frames,) laser energy on the mesh (J) that has arrived up to each frame: cumulative sum of V_i E_i."""
        per_interval = self._read("E_frame", 0, self.last).astype(np.float64) @ self.V
        return np.concatenate([[0.0], np.cumsum(per_interval)])

    def phi(self, T):
        return liquid_fraction(T) if self.melts else np.zeros_like(T)

    def starts(self, stride):
        return range(0, self.last - check_stride(stride) + 1)


def load_runs(split, mesh_seed=0, root=OUT, index=INDEX, preload=True):
    """Every processed run of one split, in the order of data/splits.json."""
    rows = [r for r in json.load(open(index)) if r["split"] == split]
    if not rows:
        raise ValueError(f"no runs of split {split!r} in {index}")
    files = [processed_file(run_name(r), mesh_seed, root) for r in rows]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} processed runs missing (first {missing[0]}); run "
                                f"python -m graphs.preprocess --splits {split} --mesh-seed {mesh_seed}")
    return [RunGraph(f, preload) for f in files]


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


def load_stats(stride, root=OUT, index=INDEX):
    """The saved training statistics of a stride, checked against the current mesh parameters and the current
    processed train split. Raises if the file is missing or stale; never computes or recomputes it."""
    stride = check_stride(stride)
    fn = stats_file(stride, root)
    if not os.path.exists(fn):
        raise FileNotFoundError(f"{fn} missing: run python -m graphs.dataset --stride {stride} first")
    with open(fn) as f:
        stats = json.load(f)
    fix = f"; recompute with python -m graphs.dataset --stride {stride} --recompute"
    if stats.get("mesh_params") != PARAMS:
        raise ValueError(f"{fn} was computed for other mesh parameters" + fix)
    if stats.get("train_fingerprint") != current_train_fingerprint(root, index):
        raise ValueError(f"{fn} was computed from different training data than the current processed train "
                         f"split (data/graphs/seed0)" + fix)
    if stats.get("stride") != stride:
        raise ValueError(f"{fn} holds statistics for stride {stats.get('stride')}, not {stride}" + fix)
    return stats


def check_stats(stats, stride, runs, what):
    """Statistics fit these runs: same stride, interpolation and frame interval. what names the runs in errors."""
    interp = sorted({r.attrs["interp"] for r in runs})
    if stats.get("stride") != stride:
        raise ValueError(f"statistics are for stride {stats.get('stride')}, not {stride}")
    if stats.get("interp") != interp:
        raise ValueError(f"statistics come from {stats.get('interp')}-interpolated data, {what} is "
                         f"{interp}-interpolated")
    frame_us = runs[0].frame_us
    if not np.isclose(stats.get("frame_us", np.nan), frame_us, rtol=1e-6):
        raise ValueError(f"{what} saves a frame every {frame_us:.2f} us, the statistics assume "
                         f"{stats.get('frame_us', float('nan')):.2f} us, so stride {stride} would be a different "
                         f"model step (week1 needs its own statistics, passed as stats=...)")


class SampleBuilder:
    """Model input for the step from frame k to frame k + stride, built from a given state T0 (K) at frame k.
    T_next (the true frame k + stride) adds the training targets; rollouts leave it out. device: where the tensors
    are made (the mesh tensors are cached there)."""

    def __init__(self, stats, stride, device=None):
        require_torch("SampleBuilder")
        self.stats, self.stride, self.device = stats, check_stride(stride), device
        self._static, self._geometry = {}, {}

    def _edge_attr(self, rel, dist):
        s = self.stats["edge"]
        ea = np.c_[(rel - s["rel"]["mean"]) / s["rel"]["std"], (dist - s["dist"]["mean"]) / s["dist"]["std"]]
        return torch.tensor(ea, dtype=torch.float32)

    def mesh_tensors(self, run):
        key = (run.mesh.name, run.mesh.seed)
        if key not in self._static:
            m = run.mesh
            rel, dist = m.edge_vectors()
            st = dict(edge_index=torch.from_numpy(m.edge_index), edge_attr=self._edge_attr(rel, dist),
                      rev_edge=torch.from_numpy(m.rev), pos=torch.tensor(m.pos, dtype=torch.float32),
                      boundary=torch.from_numpy(m.boundary))
            self._static[key] = {k: v.to(self.device) for k, v in st.items()} if self.device is not None else st
            self._geometry[key] = (rel, dist, np.asarray(m.pos, dtype=np.float64))
        return self._static[key]

    def transformed(self, run, rotation):
        """edge_attr and pos of the run's mesh after the orthogonal 2 x 2 transform rotation (new tensors; the
        cached ones are not touched)."""
        self.mesh_tensors(run)
        rel, dist, pos = self._geometry[(run.mesh.name, run.mesh.seed)]
        Q = np.asarray(rotation, dtype=np.float64)
        if Q.shape != (2, 2) or not np.allclose(Q @ Q.T, np.eye(2), atol=1e-12):
            raise ValueError(f"rotation must be an orthogonal 2 x 2 matrix, not {Q.tolist()}")
        out = dict(edge_attr=self._edge_attr(rel @ Q.T, dist), pos=torch.tensor(pos @ Q.T, dtype=torch.float32))
        return {k: v.to(self.device) for k, v in out.items()} if self.device is not None else out

    def sample(self, run, k, T0, T_next=None, rotation=None):
        """rotation: None, or an orthogonal 2 x 2 matrix applied to the edge vectors and pos (training augmentation,
        see the module docstring)."""
        s, n = self.stride, self.stats["node"]
        T0 = np.asarray(T0, dtype=np.float64)
        E = run.laser(k, s)
        phi = run.phi(T0)
        st = self.mesh_tensors(run)
        if rotation is not None:
            st = {**st, **self.transformed(run, rotation)}
        z = lambda x, key: (x - n[key]["mean"]) / n[key]["std"]
        x = np.c_[z(T0, "T"), z(phi, "phi"), z(E, "E"), run.mesh.boundary]
        f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        raw = dict(T=f32(T0), H=f32(run.H_from_T(T0)), E=f32(E), V=f32(run.V))
        if T_next is not None:
            raw.update(T_next=f32(T_next), H_next=f32(run.H_from_T(np.asarray(T_next, dtype=np.float64))))
        return GraphSample(x=f32(x), **st, **raw, t=float(run.t[k]), k=k, stride=s, run=run.name)


def random_orthogonal():
    """A random orthogonal 2 x 2 matrix from torch's global generator: a reflection y -> -y with probability 0.5,
    then a rotation by a uniform angle (together uniform over all rotations and reflections)."""
    u = torch.rand(2, dtype=torch.float64).tolist()
    c, s = np.cos(2 * np.pi * u[0]), np.sin(2 * np.pi * u[0])
    return np.array([[c, -s], [s, c]]) @ np.diag([1.0, -1.0 if u[1] < 0.5 else 1.0])


class MeltPoolGraphs(Dataset):
    """All one-step samples of one split at one stride. Pass stats to override the saved training statistics.
    noise_std: standard deviation (K) of the training noise on the input temperature; augment: a random rotation and
    reflection per sample (both: see the module docstring; for training only)."""

    def __init__(self, split, stride=5, mesh_seed=0, root=OUT, index=INDEX, stats=None, preload=True, noise_std=0.0,
                 augment=False):
        require_torch("MeltPoolGraphs")
        stride = check_stride(stride)
        noise_std = float(noise_std)
        if not (np.isfinite(noise_std) and noise_std >= 0):
            raise ValueError(f"noise_std must be a finite number of kelvin, at least 0, not {noise_std!r}")
        self.noise_std, self.augment = noise_std, bool(augment)
        self.runs = load_runs(split, mesh_seed, root, index, preload)
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
            stats = load_stats(stride, root, index)
        check_stats(stats, stride, self.runs, f"split {split}")
        self.stats = stats
        self.builder = SampleBuilder(stats, stride)

    def __len__(self):
        return len(self.index)

    def sample(self, run, k):
        T0 = run.T(k)
        if self.noise_std > 0:
            T0 = T0 + self.noise_std * torch.randn(T0.shape, dtype=torch.float64).numpy()
        rotation = random_orthogonal() if self.augment else None     # after the noise: same noise with or without
        return self.builder.sample(run, k, T0, run.T(k + self.stride), rotation)

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
