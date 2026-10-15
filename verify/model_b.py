"""Physics and integration checks: python -m verify.model_b (no external dataset)."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from scipy.spatial import Delaunay

from graphs.laplacian import cotangent_weights, stiffness
from graphs.dataset import GraphSample, SampleBuilder, RunGraph, compute_stats
from models.model_b import LinearDiffusion, ModelB
from models.linear_filter import LinearFilter
from analysis.model_b_gate import decide
from solver.ti64 import RHO, CP_CONST


def square(n):
    xy = np.linspace(0, 1e-3, n)
    x, y = np.meshgrid(xy, xy)
    pos = np.c_[x.ravel(), y.ravel()]
    tri = Delaunay(pos).simplices
    und = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
    ei = np.concatenate([und, und[:, ::-1]]).T
    u, v = pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]]
    a = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    area = np.zeros(len(pos))
    np.add.at(area, tri.ravel(), np.repeat(a / 3, 3))
    return SimpleNamespace(pos=pos, tri=tri, edge_index=ei, area=area, n_nodes=len(pos),
                           fingerprint=lambda: f"square{n}")


def graph(mesh, dt=1e-5):
    f = lambda x: torch.tensor(x, dtype=torch.float64)
    return GraphSample(num_nodes=mesh.n_nodes, T=f(300 + 20 * np.cos(mesh.pos[:, 0] / 1e-3 * np.pi)),
                       E=f(np.full(mesh.n_nodes, 1e6)), area=f(mesh.area), dt=dt,
                       edge_index=torch.tensor(mesh.edge_index),
                       cotangent_weight=f(cotangent_weights(mesh.pos, mesh.tri, mesh.edge_index)))


def main():
    diffusion = LinearDiffusion()
    for n in (5, 9):
        m = square(n)
        K = stiffness(m).toarray()
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        np.testing.assert_allclose(K.sum(axis=1), 0, atol=1e-14)
        assert np.linalg.eigvalsh(K).min() > -1e-12
        g = graph(m)
        d = diffusion(g)
        energy = float((g.area * d).sum())
        expected = float((g.area * g.E).sum()) / (RHO * CP_CONST)
        np.testing.assert_allclose(energy, expected, rtol=1e-10)
        run = SimpleNamespace(mesh=m, t=np.array([0, g.dt]), laser=lambda k, s: g.E.numpy())
        np.testing.assert_allclose(LinearFilter().step(run, g.T.numpy(), 0, 1),
                                   (g.T + d).numpy(), atol=1e-10)
        g.E.zero_()
        g.T.fill_(400)
        torch.testing.assert_close(diffusion(g), torch.zeros_like(g.T))
    g1, g2 = graph(square(5)), graph(square(9), 2e-5)
    batch = Batch.from_data_list([g1, g2])
    torch.testing.assert_close(diffusion(batch), torch.cat([diffusion(g1), diffusion(g2)]))

    class Correction(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        def forward(self, g):
            return self.value.expand(g.num_nodes)

    model = ModelB(Correction(), {"target": {"dT": {"mean": 2., "std": 3.}}}).double()
    out = model(batch)
    torch.testing.assert_close(out * 3 + 2, diffusion(batch))
    out.square().mean().backward()
    assert model.correction.value.grad is not None and torch.isfinite(model.correction.value.grad)
    rows = [dict(model=m, domain=d, seed=s, nrmse=v + s * .001)
            for m, v in (("A", .2), ("B", .1), ("LF", .3)) for d in ("val", "new_mesh") for s in (0, 1, 2)]
    assert decide(rows)["decision"] == "go"
    for r in rows:
        if r["model"] == "B" and r["domain"] == "new_mesh":
            r["nrmse"] = .25
    assert decide(rows)["decision"] == "no-go"
    try:
        decide(rows[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("gate accepted missing seed")
    print("PASS: stiffness, constant state, source energy, baseline parity on two resolutions,")
    print("      batched dt, output normalisation, correction gradients, gate go/no-go/missing data")


if __name__ == "__main__":
    main()
