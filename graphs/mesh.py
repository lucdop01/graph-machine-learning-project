"""Adaptive jittered triangle mesh, one per laser path (plan step 2).

All lengths are in metres. d is the distance from a point to the laser path polyline (round caps at the ends).

  fine band   d <= BAND = 120 um: a 10 um lattice, each node moved by up to 20% of the spacing per coordinate.
              The farthest solver cell that passes the solidus is 55 um from the path for a straight track at
              195 W, 0.5 m/s, 75 um for a zigzag at 195 W, 0.65 m/s and 85 um for an L-turn at 170 W, 0.5 m/s
              (corner and track end). Through the bilinear weights a node sees cell centres up to 14 um away, so
              every node that ever passes the solidus lies inside the band. verify/graph_energy.py checks this
              for every run.
  transition  target spacing h = 10 um + GRADE * (d - BAND), reaching 50 um at d = 320 um. Points come from
              jittered lattices at 15, 22 and 33 um, each used where the target spacing is closest to it. A point
              too close to a point of a finer lattice is dropped. The transition nodes (outside the band, not on
              the 50 um lattice, not on the edge) are then relaxed with a spring smoother (DistMesh style), so the
              triangles keep a good shape. The fine band and the 50 um zone are not moved. Relaxed nodes can
              drift a few um into the band (closest 116 um); they keep zone 1, since zones follow construction.
  coarse      a 50 um lattice, jittered like the fine band.
  edge        nodes exactly on the plate edges at the local target spacing, jittered along the edge only, with
              the four corners fixed. The triangles therefore cover the plate exactly.

Triangles come from scipy's Delaunay triangulation. Graph edges are the triangle sides, in both directions.
Node areas are lumped: a third of the area of every adjacent triangle, so they sum to the plate area.

Usage: python -m graphs.mesh            builds (or loads) every mesh and prints node counts and triangle quality
Meshes are cached in data/meshes/<name>_seed<k>.npz. Commit these files so every machine uses the same meshes.
"""
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from solver.paths import straight, l_turn, zigzag

H_FINE, H_COARSE = 10e-6, 50e-6     # node spacing near the track and far from it
BAND = 120e-6                       # half-width of the fine band around the path
GRADE = 0.2                         # spacing growth per unit distance outside the band
JITTER = 0.2                        # random shift per coordinate, as a fraction of the local spacing
N_LEVELS = 4                        # lattices at 10, 15, 22, 33 and 50 um
MIN_DIST = 0.6                      # drop a point closer than MIN_DIST * h to a point of a finer lattice
RELAX_ITERS, RELAX_STEP, RELAX_PRESSURE = 150, 0.2, 1.2
MESH_DIR = "data/meshes"
PARAMS = dict(H_FINE=H_FINE, H_COARSE=H_COARSE, BAND=BAND, GRADE=GRADE, JITTER=JITTER, N_LEVELS=N_LEVELS,
              MIN_DIST=MIN_DIST, RELAX_ITERS=RELAX_ITERS, RELAX_STEP=RELAX_STEP, RELAX_PRESSURE=RELAX_PRESSURE,
              version=1)

# name: (waypoints, Lx, Ly). test_fine uses "straight" (same plate, only the solver grid differs).
PATHS = {
    "straight": (straight(), 2e-3, 1e-3),
    "l_turn": (l_turn(), 2e-3, 1e-3),
    "zigzag": (zigzag(), 2e-3, 1e-3),
    "straight_large": (straight(4e-3, 2e-3), 4e-3, 2e-3),
}
ZONES = ("fine", "transition", "coarse")


@dataclass
class Mesh:
    name: str
    seed: int
    pos: np.ndarray          # (n, 2) node positions
    tri: np.ndarray          # (m, 3) triangles
    edge_index: np.ndarray   # (2, E) directed edges (source, target), both directions of every triangle side
    rev: np.ndarray          # (E,) index of the reverse edge: edge_index[:, rev[e]] == edge_index[::-1, e]
    area: np.ndarray         # (n,) lumped node area, m^2 (node volume = area * plate thickness)
    boundary: np.ndarray     # (n,) node lies on the plate edge
    zone: np.ndarray         # (n,) by construction: 0 fixed 10 um lattice, 1 relaxed transition, 2 fixed 50 um lattice
    dist: np.ndarray         # (n,) distance to the laser path of the final node position
    waypoints: np.ndarray
    Lx: float
    Ly: float

    @property
    def n_nodes(self):
        return len(self.pos)

    @property
    def n_edges(self):
        return self.edge_index.shape[1]

    def fingerprint(self):
        """Short hash of the geometry (node positions and triangles). Processed runs store it, so node data can never
        be paired with a different mesh, even one built with the same parameters on another machine."""
        h = hashlib.sha1()
        for a in (self.pos.astype("<f8"), self.tri.astype("<i8")):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]

    def edge_vectors(self):
        """Relative position of every edge, sender minus receiver (x_source - x_target), and its length."""
        rel = self.pos[self.edge_index[0]] - self.pos[self.edge_index[1]]
        return rel, np.linalg.norm(rel, axis=1)

    def quality(self):
        """Smallest triangle angle (degrees) overall and per zone (a triangle belongs to its outermost node's zone)."""
        a = triangle_angles(self.pos, self.tri).min(axis=1)
        z = self.zone[self.tri].max(axis=1)
        out = dict(min=float(a.min()), p01=float(np.percentile(a, 0.1)), median=float(np.median(a)))
        for k, name in enumerate(ZONES):
            if (z == k).any():
                out[name] = float(a[z == k].min())
        return out

    def summary(self):
        q = self.quality()
        counts = ", ".join(f"{name} {int((self.zone == k).sum())}" for k, name in enumerate(ZONES))
        return (f"{self.name} seed {self.seed}: {self.n_nodes} nodes ({counts}, edge {int(self.boundary.sum())}), "
                f"{len(self.tri)} triangles, {self.n_edges} directed edges, min angle {q['min']:.1f} deg "
                f"(0.1% quantile {q['p01']:.1f}, per zone " + ", ".join(f"{z} {q[z]:.1f}" for z in ZONES if z in q)
                + f"), area sum / plate {self.area.sum() / (self.Lx * self.Ly) - 1:+.1e}")


def path_distance(points, waypoints):
    """Distance from each point to the polyline."""
    p = np.asarray(points, float)
    w = np.asarray(waypoints, float)
    d = np.full(len(p), np.inf)
    for a, b in zip(w[:-1], w[1:]):
        ab = b - a
        s = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(p - (a + s[:, None] * ab), axis=1))
    return d


def target_spacing(points, waypoints):
    return np.clip(H_FINE + GRADE * (path_distance(points, waypoints) - BAND), H_FINE, H_COARSE)


def triangle_angles(pos, tri):
    """(m, 3) interior angles in degrees."""
    def angle(u, v, w):
        x, y = v - u, w - u
        return np.degrees(np.arctan2(np.abs(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]), (x * y).sum(axis=1)))
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    return np.stack([angle(a, b, c), angle(b, c, a), angle(c, a, b)], axis=1)


def triangle_areas(pos, tri):
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    return 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])


def _edge_nodes(waypoints, Lx, Ly, rng):
    """Nodes on the four plate edges at the local target spacing. Corners are fixed, the rest move along the edge."""
    corners = np.array([(0.0, 0.0), (Lx, 0.0), (Lx, Ly), (0.0, Ly)])
    out = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        s = np.linspace(0.0, L, int(round(L / 1e-6)) + 1)               # 1 um samples along the edge
        h = target_spacing(a + s[:, None] * u, waypoints)
        cells = np.concatenate([[0.0], np.cumsum(np.diff(s) / (0.5 * (h[1:] + h[:-1])))])   # integral of 1/h
        n = max(1, int(round(cells[-1])))
        sk = np.interp(np.linspace(0.0, cells[-1], n + 1), cells, s)
        hk = np.interp(sk, s, h)
        sk[1:-1] += rng.uniform(-JITTER, JITTER, n - 1) * hk[1:-1]
        out.append(a + sk[:-1, None] * u)                               # the end corner starts the next edge
    return np.concatenate(out)                                          # exactly on the edges: u is axis-aligned


def _unique_edges(tri):
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def _relax(pos, movable, waypoints, Lx, Ly):
    """Spring smoother (Persson and Strang's DistMesh): edges shorter than the target spacing push their nodes apart.
    Only movable nodes move; they are surrounded by fixed nodes, so the region they fill does not change."""
    p = pos.copy()
    lo, hi = 0.25 * H_FINE, np.array([Lx, Ly]) - 0.25 * H_FINE
    for it in range(RELAX_ITERS):
        if it % 5 == 0:                                                  # refresh the connectivity
            E = _unique_edges(Delaunay(p).simplices)
            E = E[movable[E].any(axis=1)]
        vec = p[E[:, 1]] - p[E[:, 0]]
        L = np.linalg.norm(vec, axis=1)
        h = target_spacing(0.5 * (p[E[:, 0]] + p[E[:, 1]]), waypoints)
        L0 = RELAX_PRESSURE * h * np.sqrt((L ** 2).sum() / (h ** 2).sum())
        f = (np.maximum(L0 - L, 0.0) / L)[:, None] * vec
        force = np.zeros_like(p)
        np.add.at(force, E[:, 0], -f)
        np.add.at(force, E[:, 1], f)
        p[movable] = np.clip(p[movable] + RELAX_STEP * force[movable], lo, hi)
    return p


def build_mesh(name, seed=0):
    """Build the mesh for one laser path. Same name and seed give the same mesh on the same machine."""
    waypoints, Lx, Ly = PATHS[name]
    waypoints = np.asarray(waypoints, float)
    rng = np.random.default_rng(seed)
    r = (H_COARSE / H_FINE) ** (1.0 / N_LEVELS)
    spacings = H_FINE * r ** np.arange(N_LEVELS + 1)

    edge = _edge_nodes(waypoints, Lx, Ly, rng)
    pts, level = [edge], [np.full(len(edge), -1)]
    tree = cKDTree(edge)
    for lev, s in enumerate(spacings):
        nx_, ny_ = int(np.floor(Lx / s + 1e-9)), int(np.floor(Ly / s + 1e-9))
        X, Y = np.meshgrid(np.arange(1, nx_ + 1) * s, np.arange(1, ny_ + 1) * s, indexing="ij")
        P = np.c_[X.ravel(), Y.ravel()]
        P = P[(P[:, 0] < Lx - 0.5 * s) & (P[:, 1] < Ly - 0.5 * s)]       # interior lattice points only
        h = target_spacing(P, waypoints)
        nearest = np.abs(np.log(h[:, None] / spacings[None, :])).argmin(axis=1)
        P = P[nearest == lev]                                            # lattice used where it matches the target
        P = P + rng.uniform(-JITTER, JITTER, P.shape) * s
        h = target_spacing(P, waypoints)
        to_edge = np.minimum.reduce([P[:, 0], Lx - P[:, 0], P[:, 1], Ly - P[:, 1]])
        gap, _ = tree.query(P)
        P = P[(to_edge > 0.5 * h) & (gap > MIN_DIST * h)]
        pts.append(P)
        level.append(np.full(len(P), lev))
        tree = cKDTree(np.concatenate(pts))
    pos, level = np.concatenate(pts), np.concatenate(level)
    boundary = level == -1
    dist = path_distance(pos, waypoints)
    movable = ~boundary & (dist > BAND) & (level < N_LEVELS)
    pos = _relax(pos, movable, waypoints, Lx, Ly)
    zone = np.where(dist <= BAND, 0, np.where(movable, 1, 2))          # by construction, before the move
    zone[boundary] = np.where(dist[boundary] <= BAND, 0,
                              np.where(target_spacing(pos[boundary], waypoints) < H_COARSE, 1, 2))
    dist = path_distance(pos, waypoints)                                # relaxed nodes have moved

    dl = Delaunay(pos)
    tri = dl.simplices.astype(np.int64)
    areas = triangle_areas(pos, tri)
    if len(np.unique(tri)) != len(pos) or len(dl.coplanar):
        raise RuntimeError(f"{name} seed {seed}: some nodes are not mesh vertices")
    if areas.min() <= 0 or abs(areas.sum() / (Lx * Ly) - 1) > 1e-12:
        raise RuntimeError(f"{name} seed {seed}: triangles do not cover the plate exactly")
    area = np.zeros(len(pos))
    np.add.at(area, tri.ravel(), np.repeat(areas / 3.0, 3))

    und = _unique_edges(tri)
    ei = np.concatenate([und, und[:, ::-1]]).T
    order = np.lexsort((ei[1], ei[0]))                                   # sort by source, then target
    ei = ei[:, order]
    key = ei[0] * len(pos) + ei[1]
    rev = np.searchsorted(key, ei[1] * len(pos) + ei[0])
    return Mesh(name, seed, pos, tri, ei.astype(np.int64), rev.astype(np.int64), area, boundary, zone.astype(np.int8),
                dist, waypoints, float(Lx), float(Ly))


def mesh_file(name, seed=0, root=MESH_DIR):
    return os.path.join(root, f"{name}_seed{seed}.npz")


def save_mesh(mesh, root=MESH_DIR):
    os.makedirs(root, exist_ok=True)
    np.savez_compressed(mesh_file(mesh.name, mesh.seed, root), pos=mesh.pos, tri=mesh.tri, edge_index=mesh.edge_index,
                        rev=mesh.rev, area=mesh.area, boundary=mesh.boundary, zone=mesh.zone, waypoints=mesh.waypoints, Lx=mesh.Lx, Ly=mesh.Ly, params=json.dumps(PARAMS, sort_keys=True))


def get_mesh(name, seed=0, root=MESH_DIR, cache=True):
    """Load the cached mesh if it was built with the current parameters, otherwise build and cache it."""
    fn = mesh_file(name, seed, root)
    if cache and os.path.exists(fn):
        f = np.load(fn)
        if str(f["params"]) == json.dumps(PARAMS, sort_keys=True):
            return Mesh(name, seed, f["pos"], f["tri"], f["edge_index"], f["rev"], f["area"], f["boundary"],
                        f["zone"], path_distance(f["pos"], f["waypoints"]), f["waypoints"], float(f["Lx"]),
                        float(f["Ly"]))
    mesh = build_mesh(name, seed)
    if cache:
        save_mesh(mesh, root)
    return mesh


def mesh_name(path, Lx, Ly):
    """Mesh for a run, from its path attribute and plate size."""
    for name, (_, lx, ly) in PATHS.items():
        if name.split("_large")[0] == path and np.isclose(lx, Lx) and np.isclose(ly, Ly):
            return name
    raise KeyError(f"no mesh for path {path!r} on a {Lx*1e3:g} x {Ly*1e3:g} mm plate")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache")
    a = ap.parse_args()
    for seed in a.seeds:
        for name in PATHS:
            m = build_mesh(name, seed) if a.rebuild else get_mesh(name, seed)
            if a.rebuild:
                save_mesh(m)
            print(m.summary())
