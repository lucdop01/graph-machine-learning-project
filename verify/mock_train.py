"""Smoke / mock test for the unified training harness.

Usage (from repo root):
    python -m verify.mock_train
Pass: all training harness stages (model build, train step, loss, backward, metrics,
      checkpointing, atomic write, and checkpoint reload) execute without error.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch_geometric.data import Data

from graphs.dataset import EDGE_FEATURES, NODE_FEATURES
from models.train_harness import (
    Interval,
    assert_weights_finite,
    build_checkpoint,
    build_model,
    execute_train_step,
    generate_sweep_candidates,
    init_run_files,
    rng_state,
    save_atomic,
)


def create_mock_batch(num_nodes: int = 16, num_edges: int = 32) -> Any:
    """Create a minimal synthetic PyTorch Geometric Data batch for mock testing."""
    x = torch.randn(num_nodes, len(NODE_FEATURES), dtype=torch.float32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.int64)
    edge_attr = torch.randn(num_edges, len(EDGE_FEATURES), dtype=torch.float32)
    T = torch.full((num_nodes,), 300.0, dtype=torch.float32)
    T_next = T + torch.randn(num_nodes, dtype=torch.float32) * 5.0
    H = torch.full((num_nodes,), 0.0, dtype=torch.float32)
    H_next = H + torch.randn(num_nodes, dtype=torch.float32) * 2000.0
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, T=T, T_next=T_next, H=H, H_next=H_next)


def main():
    print("\n============================================================")
    print("Running Mock Training Harness Verification")
    print("============================================================\n")

    # 1. Model instantiation (Model A and GCN baseline)
    for model_name in ("model_a", "gcn"):
        m_cfg = {"name": model_name, "width": 16, "layers": 2, "hidden_layers": 1}
        test_model = build_model(m_cfg)
        n_params = sum(p.numel() for p in test_model.parameters())
        print(f"1. Model '{model_name}' instantiated successfully ({n_params:,} parameters)")
    model_cfg = {"name": "model_a", "width": 16, "layers": 2, "hidden_layers": 1}
    model = build_model(model_cfg)

    # 2. Forward pass, loss calculation, backward pass, optimizer step
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = create_mock_batch()
    print("2. Testing forward pass, loss, backward pass, and optimizer step...")
    loss, gn = execute_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        target="dT",
        mean=0.0,
        std=10.0,
        precision="fp32",
        device=device,
        lr=1e-3,
    )
    print(f"   Step 1 completed: loss = {float(loss.detach()):.4e}, grad_norm = {float(gn.detach()):.4e}")
    if not (torch.isfinite(loss) and torch.isfinite(gn)):
        raise ValueError("Mock step produced non-finite loss or gradient norm")

    # 3. Interval metrics accumulator and host transfer verification
    print("3. Testing Interval accumulator and host transfer verification...")
    interval = Interval(device)
    for step in range(1, 4):
        step_loss, step_gn = execute_train_step(
            model, optimizer, create_mock_batch(), "dT", 0.0, 10.0, "fp32", device, 1e-3
        )
        interval.add(step, step_loss, step_gn)
    stats_row, n_steps = interval.flush()
    print(f"   Interval flushed {n_steps} steps: train_loss = {stats_row['train_loss']:.4e}, grad_norm = {stats_row['grad_norm']:.4e}")

    # 4. Atomic checkpoint persistence and validation
    mock_run_name = "mock_test_run"
    mock_files = init_run_files(mock_run_name)
    os.makedirs(os.path.dirname(mock_files["best.pt"]), exist_ok=True)
    print(f"4. Testing atomic checkpoint creation at {mock_files['best.pt']}...")
    assert_weights_finite(model, 3)
    mock_stats = {
        "train_fingerprint": "mock_fp",
        "target": {"dT": {"mean": 0.0, "std": 10.0}},
    }
    mock_bins = {"train_fingerprint": "mock_fp", "bins": [0, 10]}
    ckpt = build_checkpoint(
        name=mock_run_name,
        model=model,
        cfg={"model": model_cfg, "data": {"stride": 5}, "training": {"precision": "fp32"}},
        target="dT",
        stride=5,
        stats=mock_stats,
        bins=mock_bins,
        step=3,
        score=0.0123,
        best=0.0123,
        best_step=3,
        elapsed=1.23,
        precision="fp32",
        optimizer=optimizer,
        rng_data=rng_state(),
    )
    save_atomic(ckpt, mock_files["best.pt"])
    save_atomic(ckpt, mock_files["last.pt"])
    print("   Checkpoints saved successfully")

    # 5. Checkpoint loading
    print(f"5. Testing loading checkpoint from {mock_files['last.pt']}...")
    loaded_ckpt = torch.load(mock_files["last.pt"], map_location="cpu", weights_only=True)
    if loaded_ckpt["step"] != 3 or loaded_ckpt["best_score"] != 0.0123:
        raise ValueError("Loaded checkpoint metadata mismatch")
    print("   Checkpoint loaded and validated successfully")

    # 6. Sweep candidate generation
    print("6. Testing sweep candidate generation...")
    candidates = generate_sweep_candidates(4)
    print(f"   Generated {len(candidates)} candidates: {candidates}")

    # Clean up mock checkpoints
    for f in (mock_files["best.pt"], mock_files["last.pt"]):
        if os.path.exists(f):
            os.remove(f)
    mock_dir = os.path.dirname(mock_files["best.pt"])
    if os.path.exists(mock_dir) and not os.listdir(mock_dir):
        os.rmdir(mock_dir)

    print("\n============================================================")
    print("PASS: Mock training verification completed successfully.")
    print("============================================================\n")


if __name__ == "__main__":
    main()
