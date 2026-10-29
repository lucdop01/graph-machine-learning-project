"""Short CUDA/PyG/Model B probe, with optional CPU verification.

python -m verify.gpu_probe --device cuda
This uses a tiny synthetic graph, not trained-model evaluation or a benchmark.
"""
import argparse
import json
import torch
import torch_geometric
from torch_geometric.data import Data
from graphs.laplacian import cotangent_weights
from models.model_b import ModelB
import numpy as np


def probe(device):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the GPU job; inspect nvidia-smi and PyTorch runtime/driver compatibility")
    pos = np.array([[0., 0.], [1e-4, 0.], [0., 1e-4], [1e-4, 1e-4]])
    tri = np.array([[0, 1, 2], [1, 3, 2]])
    edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    ei = np.concatenate([edges, edges[:, ::-1]]).T
    g = Data(num_nodes=4, x=torch.randn(4, 4), edge_attr=torch.randn(10, 3),
             edge_index=torch.tensor(ei), T=torch.tensor([300., 310., 320., 330.]),
             E=torch.full((4,), 1e6), area=torch.tensor([1/6, 1/3, 1/3, 1/6]) * 1e-8,
             cotangent_weight=torch.tensor(cotangent_weights(pos, tri, ei), dtype=torch.float32), dt=1e-6)
    stats = {"target": {"dT": {"mean": 0., "std": 100.}}}
    model = ModelB(stats=stats, width=16, layers=2).to(device)
    g = g.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    out = model(g)
    loss = out.square().mean()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                     for p in model.parameters()):
        raise FloatingPointError("non-finite Model B loss or gradients")
    opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    result = dict(status="PASS", device=device, torch=str(torch.__version__), cuda_build=torch.version.cuda,
                  pyg=torch_geometric.__version__, model_b_forward_backward=True, loss=float(loss.detach().cpu()),
                  gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                  note="Tiny synthetic integration probe; not training or gate evidence")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    probe(parser.parse_args().device)
