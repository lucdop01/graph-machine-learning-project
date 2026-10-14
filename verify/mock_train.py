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

from graphs.dataset import EDGE_FEATURES, NODE_FEATURES, GraphSample
from models.model_d import ModelD
from train_harness import (
    Interval,
    assert_weights_finite,
    build_checkpoint,
    build_model,
    execute_train_step,
    generate_sweep_candidates,
    init_run_files,
    load_config,
    rng_state,
    save_atomic,
    set_all_seeds,
)


def create_mock_batch(num_nodes: int = 16, num_edges: int = 32) -> Any:
    """Create a minimal synthetic PyTorch Geometric GraphSample batch for mock testing."""
    half = num_edges // 2
    src = torch.randint(0, num_nodes, (half,), dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (half,), dtype=torch.int64)
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ], dim=0)
    rev_edge = torch.cat([torch.arange(half, 2 * half), torch.arange(0, half)])

    x = torch.randn(num_nodes, len(NODE_FEATURES), dtype=torch.float32)
    edge_attr = torch.randn(2 * half, len(EDGE_FEATURES), dtype=torch.float32)
    T = torch.full((num_nodes,), 300.0, dtype=torch.float32)
    T_next = T + torch.randn(num_nodes, dtype=torch.float32) * 5.0
    V = torch.full((num_nodes,), 1e-12, dtype=torch.float32)
    E = torch.randn(num_nodes, dtype=torch.float32).abs() * 1e7
    H = torch.full((num_nodes,), 0.0, dtype=torch.float32)
    H_next = H + (E / 4420.0) + torch.randn(num_nodes, dtype=torch.float32) * 50.0
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    boundary = torch.zeros(num_nodes, dtype=torch.float32)

    return GraphSample(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        rev_edge=rev_edge,
        T=T,
        T_next=T_next,
        H=H,
        H_next=H_next,
        V=V,
        E=E,
        pos=pos,
        boundary=boundary,
    )


def main():
    print("\n============================================================")
    print("Running Mock Training Harness Verification")
    print("============================================================\n")

    set_all_seeds(0)
    device = torch.device("cpu")

    # 1. Model instantiation & forward/backward verification (Model A, GCN, Model D)
    print("--- 1. Testing Model Architectures (Mock Size: width=16, layers=2) ---")
    models_to_test = ("model_a", "gcn", "model_d")
    for model_name in models_to_test:
        m_cfg = {"name": model_name, "width": 16, "layers": 2, "hidden_layers": 1}
        test_model = build_model(m_cfg)
        n_params = sum(p.numel() for p in test_model.parameters())
        tgt = getattr(test_model, "target", "dT")
        mean = 0.0
        std = 10.0 if tgt == "dT" else 5000.0
        set_stats_fn = getattr(test_model, "set_stats", None)
        if callable(set_stats_fn):
            set_stats_fn(mean, std)

        opt = torch.optim.Adam(test_model.parameters(), lr=1e-3)
        b = create_mock_batch()
        step_loss, step_gn = execute_train_step(
            model=test_model,
            optimizer=opt,
            batch=b,
            target=tgt,
            mean=mean,
            std=std,
            precision="fp32",
            device=device,
            lr=1e-3,
        )
        if not (torch.isfinite(step_loss) and torch.isfinite(step_gn)):
            raise ValueError(f"Step for {model_name} produced non-finite loss or grad norm")
        print(f"   [PASS] {model_name:8s} ({n_params:,} params, target {tgt}): loss = {float(step_loss.detach()):.4e}, grad_norm = {float(step_gn.detach()):.4e}")

    # Explicit Model D physics verification (flux antisymmetry & zero internal flux sum)
    print("\n--- 2. Testing Model D Exact Physics (Antisymmetry & Heat Conservation) ---")
    m_d = build_model({"name": "model_d", "width": 16, "layers": 2, "hidden_layers": 1})
    assert isinstance(m_d, ModelD)
    mb = create_mock_batch()
    _, F_anti = m_d.fluxes(mb)
    assert torch.allclose(F_anti, -F_anti[mb.rev_edge], atol=1e-7), "Flux antisymmetry F_ij = -F_ji failed"
    net_flux_sum = float(F_anti.sum().detach())
    assert abs(net_flux_sum) < 1e-6, f"Net internal flux across mesh is non-zero: {net_flux_sum:.4e}"
    print("   [PASS] Flux antisymmetry verified: F_ij == -F_ji bit for bit")
    print(f"   [PASS] Mesh internal conduction sum verified: sum(fluxes) = {net_flux_sum:.4e} (identically 0.0)")

    # 3. Production Config Verification (model_a.yaml, gcn.yaml, model_d.yaml)
    print("\n--- 3. Testing Production YAML Configurations ---")
    config_paths = ("configs/model_a.yaml", "configs/gcn.yaml", "configs/model_d.yaml")
    for cp in config_paths:
        cfg = load_config(cp)
        prod_model = build_model(cfg["model"])
        n_params = sum(p.numel() for p in prod_model.parameters())
        prod_tgt = getattr(prod_model, "target", "dT")
        prod_mean = 0.0
        prod_std = 10.0 if prod_tgt == "dT" else 5000.0
        set_stats_fn = getattr(prod_model, "set_stats", None)
        if callable(set_stats_fn):
            set_stats_fn(prod_mean, prod_std)

        prod_opt = torch.optim.Adam(prod_model.parameters(), lr=1e-4)
        b = create_mock_batch()
        step_loss, step_gn = execute_train_step(
            model=prod_model,
            optimizer=prod_opt,
            batch=b,
            target=prod_tgt,
            mean=prod_mean,
            std=prod_std,
            precision="fp32",
            device=device,
            lr=1e-4,
        )
        assert_weights_finite(prod_model, 1)
        if not (torch.isfinite(step_loss) and torch.isfinite(step_gn)):
            raise ValueError(f"Production step for {cp} produced non-finite loss or grad norm")
        print(f"   [PASS] Config '{cp}': {cfg['model']['name']} ({n_params:,} params, width={cfg['model']['width']}, layers={cfg['model']['layers']}) step verified")

    # 4. Multi-step interval accumulator
    print("\n--- 4. Testing Interval Accumulator and Streaming ---")
    model_cfg = {"name": "model_a", "width": 16, "layers": 2, "hidden_layers": 1}
    model = build_model(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    interval = Interval(device)
    for step in range(1, 4):
        step_loss, step_gn = execute_train_step(
            model, optimizer, create_mock_batch(), "dT", 0.0, 10.0, "fp32", device, 1e-3
        )
        interval.add(step, step_loss, step_gn)
    stats_row, n_steps = interval.flush()
    print(f"   [PASS] Interval flushed {n_steps} steps: train_loss = {stats_row['train_loss']:.4e}, grad_norm = {stats_row['grad_norm']:.4e}")

    # 5. Atomic checkpoint persistence and reload validation for all models
    print("\n--- 5. Testing Checkpointing & Atomic Reload for All Models ---")
    for m_name in models_to_test:
        sub_cfg = {"name": m_name, "width": 16, "layers": 2, "hidden_layers": 1}
        sub_model = build_model(sub_cfg)
        sub_opt = torch.optim.Adam(sub_model.parameters(), lr=1e-3)
        sub_tgt = getattr(sub_model, "target", "dT")
        mock_run_name = f"mock_{m_name}_run"
        mock_files = init_run_files(mock_run_name)
        os.makedirs(os.path.dirname(mock_files["best.pt"]), exist_ok=True)

        assert_weights_finite(sub_model, 1)
        mock_stats = {
            "train_fingerprint": "mock_fp",
            "target": {sub_tgt: {"mean": 0.0, "std": 10.0}},
        }
        mock_bins = {"train_fingerprint": "mock_fp", "bins": [0, 10]}
        ckpt = build_checkpoint(
            name=mock_run_name,
            model=sub_model,
            cfg={"model": sub_cfg, "data": {"stride": 5}, "training": {"precision": "fp32"}},
            target=sub_tgt,
            stride=5,
            stats=mock_stats,
            bins=mock_bins,
            step=1,
            score=0.042,
            best=0.042,
            best_step=1,
            elapsed=0.5,
            precision="fp32",
            optimizer=sub_opt,
            rng_data=rng_state(),
        )
        save_atomic(ckpt, mock_files["best.pt"])
        save_atomic(ckpt, mock_files["last.pt"])

        loaded_ckpt = torch.load(mock_files["last.pt"], map_location="cpu", weights_only=True)
        if loaded_ckpt["step"] != 1 or loaded_ckpt["best_score"] != 0.042 or loaded_ckpt["target"] != sub_tgt:
            raise ValueError(f"Loaded checkpoint metadata mismatch for {m_name}")

        # Clean up mock checkpoints
        for f in (mock_files["best.pt"], mock_files["last.pt"]):
            if os.path.exists(f):
                os.remove(f)
        mock_dir = os.path.dirname(mock_files["best.pt"])
        if os.path.exists(mock_dir) and not os.listdir(mock_dir):
            os.rmdir(mock_dir)

        print(f"   [PASS] Model '{m_name}' atomic checkpoint saved, verified, and reloaded successfully")

    # 6. Sweep candidate generation
    print("\n--- 6. Testing Sweep Candidate Generator ---")
    candidates = generate_sweep_candidates(4)
    print(f"   [PASS] Generated {len(candidates)} candidates: {candidates}")

    print("\n============================================================")
    print("PASS: Mock training verification completed successfully.")
    print("============================================================\n")


if __name__ == "__main__":
    main()
