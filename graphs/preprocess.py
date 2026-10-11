"""Solver runs on the mesh (plan step 2): node temperature at every saved frame and the laser energy per frame interval.

Node temperature: the solver temperature at the node with the fixed bilinear weights of solver/events.py
(bilinear_sampler), the same weights the evaluation uses. interp="H" interpolates enthalpy instead and converts
back with the material table (kept as an option; temperature passes the energy check, verify/graph_energy.py).
The node temperature stored here is the ground truth for training and evaluation.

Laser energy: the solver adds dt * q(x, t_i) / rho to the enthalpy at t_i = i * dt for every step with
t_i <= laser_off. The laser energy per volume a node receives (J/m^3) is the same sum over the same solver steps,
with q evaluated at the node. It is stored per frame interval (the solver steps between two saved frames), so the
model step is chosen when loading: the energy E_i of a model step from frame k to frame k + s is the sum of
intervals k to k + s - 1.

Usage (from the repo root):
  python -m graphs.preprocess                              every run in data/splits.json on mesh seed 0
  python -m graphs.preprocess --splits test_interp --mesh-seed 1    new-mesh test: same runs, other jitter
  options: --force (rewrite existing files), --interp H
Output: data/graphs/seed<k>/<run>.h5 (git-ignored) with T (frames, nodes), E_frame (frames - 1, nodes), t, attrs.

Each file records the mesh parameters, a fingerprint of the exact mesh (node positions and triangles) and a hash of
its own node data (used by the training-statistics fingerprint in graphs/dataset.py). processed_problems() checks a
file against the current mesh: parameters, fingerprint, array shapes, and that the run file is not newer (when the run
file is present). An existing file is reused only if it passes these checks and has the requested interpolation;
otherwise it is rewritten. graphs/dataset.py refuses to load a file that fails them.
"""
import argparse
import functools
import hashlib
import json
import os
import time

import h5py
import numpy as np

from solver.paths import Path
from solver.events import bilinear_sampler
from solver import ti64
from graphs.mesh import PARAMS, get_mesh, mesh_name

OUT = "data/graphs"


def material(split):
    """Enthalpy table of a run: week1 uses constant properties without melting, every other split Ti-6Al-4V."""
    if split == "week1":
        return ti64.CONST_PROPS["H_from_T"], ti64.CONST_PROPS["T_from_H"]
    return ti64.H_from_T, ti64.T_from_H


def load_run(fn):
    """A run file as the dict simulate() returns (T stays float32, as stored)."""
    with h5py.File(fn, "r") as f:
        run = {key: f[key][:] for key in ("T", "t", "laser", "waypoints")}
        run.update({key: (val.item() if hasattr(val, "item") else val) for key, val in f.attrs.items()})
    run["file"] = fn
    run["name"] = os.path.splitext(os.path.basename(fn))[0]
    return run


def grid_of(run):
    """nx, ny, dx and save_every of a run (save_every follows from the frame times)."""
    _, nx, ny = run["T"].shape
    return nx, ny, float(run["dx"]), int(round(float(run["t"][1]) / float(run["dt"])))


def mesh_for_run(run, seed=0):
    nx, ny, dx, _ = grid_of(run)
    mesh = get_mesh(mesh_name(run["path"], nx * dx, ny * dx), seed)
    if not np.allclose(mesh.waypoints, run["waypoints"], rtol=0, atol=1e-12):
        raise ValueError(f"{run.get('name', 'run')}: laser path differs from mesh {mesh.name}")
    return mesh


def node_temperature(run, mesh, interp="T"):
    """(frames, nodes) temperature at the mesh nodes, float32."""
    nx, ny, dx, _ = grid_of(run)
    sample = bilinear_sampler(mesh.pos, nx, ny, dx)
    T = run["T"].astype(float)
    if interp == "T":
        return sample(T).astype(np.float32)
    if interp == "H":
        HfT, TfH = material(run["split"])
        return TfH(sample(HfT(T))).astype(np.float32)
    raise ValueError(f"interp must be 'T' or 'H', not {interp!r}")


def laser_schedule(run):
    """Solver steps with the laser on, as the solver applies them: step i at t_i = i * dt while t_i <= laser_off.
    Returns (frame interval of each step, laser position of each step, q0 = peak power density in W/m^3)."""
    _, _, _, save_every = grid_of(run)
    path = Path(run["waypoints"], float(run["v"]))
    if not np.isclose(path.duration, float(run["laser_off"]), rtol=1e-12, atol=0):
        raise ValueError("laser_off does not match the path length and speed")
    n_steps = (len(run["t"]) - 1) * save_every                            # steps up to the last saved frame
    i = np.arange(n_steps)
    t = i * float(run["dt"])
    on = t <= float(run["laser_off"])
    xy = np.array([path.pos(ti) for ti in t[on]]).reshape(-1, 2)
    q0 = 2 * ti64.ETA * float(run["P"]) / (np.pi * ti64.R_SPOT ** 2 * float(run["d"]))
    return i[on] // save_every, xy, q0


def node_laser_energy(run, points, schedule=None):
    """(frames - 1, points) laser energy per volume (J/m^3) received in each frame interval."""
    interval, xy, q0 = schedule or laser_schedule(run)
    dt = float(run["dt"])
    E = np.zeros((len(run["t"]) - 1, len(points)))
    for k in np.unique(interval):
        c = xy[interval == k]                                              # laser positions in this interval
        r2 = ((points[None, :, :] - c[:, None, :]) ** 2).sum(axis=2)
        E[k] = dt * q0 * np.exp(-2 * r2 / ti64.R_SPOT ** 2).sum(axis=0)
    return E


def grid_laser_input(run, schedule=None):
    """(frames - 1,) laser energy (J) the solver adds to the whole plate in each frame interval.
    The Gaussian is separable, so the sum over cells is a product of two 1D sums."""
    interval, xy, q0 = schedule or laser_schedule(run)
    nx, ny, dx, _ = grid_of(run)
    x, y = (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx
    sx = np.exp(-2 * (x[None, :] - xy[:, :1]) ** 2 / ti64.R_SPOT ** 2).sum(axis=1)
    sy = np.exp(-2 * (y[None, :] - xy[:, 1:]) ** 2 / ti64.R_SPOT ** 2).sum(axis=1)
    per_step = float(run["dt"]) * q0 * sx * sy * dx ** 2 * float(run["d"])
    return np.bincount(interval, weights=per_step, minlength=len(run["t"]) - 1)


def processed_file(run_name, seed=0, root=OUT):
    return os.path.join(root, f"seed{seed}", f"{run_name}.h5")


def data_hash(T, E):
    """Short hash of the stored node arrays (float32, as written)."""
    h = hashlib.sha1()
    for a in (T, E):
        h.update(np.ascontiguousarray(a, dtype="<f4").tobytes())
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=None)
def _mesh_fingerprint(name, seed):
    return get_mesh(name, seed).fingerprint()


def processed_problems(path, attrs, shape_T, shape_E, n_frames):
    """Reasons why a processed file does not fit the current mesh and run file (empty list: it fits)."""
    missing = [key for key in ("mesh", "mesh_seed", "mesh_params", "mesh_fp", "data_hash", "interp") if key not in attrs]
    if missing:
        return [f"made by an older graphs/preprocess.py (no {', '.join(missing)})"]
    problems = []
    if attrs["mesh_params"] != json.dumps(PARAMS, sort_keys=True):
        problems.append("made with other mesh parameters")
    else:
        fp = _mesh_fingerprint(str(attrs["mesh"]), int(attrs["mesh_seed"]))
        if attrs["mesh_fp"] != fp:
            problems.append(f"made on a different {attrs['mesh']} seed {attrs['mesh_seed']} mesh "
                            f"(fingerprint {attrs['mesh_fp']}, current {fp})")
        n = get_mesh(str(attrs["mesh"]), int(attrs["mesh_seed"])).n_nodes
        if tuple(shape_T) != (n_frames, n) or tuple(shape_E) != (n_frames - 1, n):
            problems.append(f"array shapes T {tuple(shape_T)}, E_frame {tuple(shape_E)} do not match "
                            f"{n_frames} frames and {n} nodes")
    src = str(attrs.get("source", ""))
    if src and os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(path):
        problems.append(f"its run file {src} is newer")
    return problems


def read_processed_header(path):
    """Attributes and array shapes of a processed file, without reading the arrays."""
    with h5py.File(path, "r") as f:
        attrs = {key: (val.item() if hasattr(val, "item") else val) for key, val in f.attrs.items()}
        return attrs, f["T"].shape, f["E_frame"].shape, f["t"].shape[0]


def process_run(fn, seed=0, interp="T", root=OUT, force=False):
    """Write one processed run. Returns the output file name."""
    out = processed_file(os.path.splitext(os.path.basename(fn))[0], seed, root)
    if os.path.exists(out) and not force:
        attrs, sT, sE, nf = read_processed_header(out)
        if (attrs.get("interp") == interp and attrs.get("mesh_seed") == seed
                and not processed_problems(out, attrs, sT, sE, nf)):
            return out
    run = load_run(fn)
    mesh = mesh_for_run(run, seed)
    T = node_temperature(run, mesh, interp)
    E = node_laser_energy(run, mesh.pos).astype(np.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as f:
        n = mesh.n_nodes
        f.create_dataset("T", data=T, chunks=(1, n), compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("E_frame", data=E, chunks=(1, n), compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("t", data=run["t"])
        for key in ("P", "v", "d", "dt", "dx", "laser_off", "t_end", "terminal_frame", "path", "split"):
            f.attrs[key] = run[key]
        if "cooling_complete" in run:
            f.attrs["cooling_complete"] = run["cooling_complete"]
        f.attrs.update(mesh=mesh.name, mesh_seed=seed, interp=interp, source=fn, save_every=grid_of(run)[3],
                       mesh_params=json.dumps(PARAMS, sort_keys=True), mesh_fp=mesh.fingerprint(),
                       data_hash=data_hash(T, E))
    os.replace(tmp, out)
    return out


def main(splits=None, seed=0, interp="T", force=False, index="data/splits.json"):
    rows = [r for r in json.load(open(index)) if splits is None or r["split"] in splits]
    t0 = time.time()
    for n, row in enumerate(rows, 1):
        out = process_run(row["file"], seed, interp, force=force)
        if n % 25 == 0 or n == len(rows):
            print(f"{n}/{len(rows)} runs ({time.time() - t0:.0f} s), last: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+")
    ap.add_argument("--mesh-seed", type=int, default=0)
    ap.add_argument("--interp", choices=("T", "H"), default="T")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.splits, a.mesh_seed, a.interp, a.force)
