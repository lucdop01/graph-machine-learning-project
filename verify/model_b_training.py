"""CPU integration check of the actual shared trainer, checkpoint and resume.

Uses one in-memory physical simulation and a small A backbone. Only data I/O
is injected; all optimisation, sampling, augmentation, rollout selection and
checkpointing run through models.train. This is not a gate experiment.
"""
import copy
import csv
import json
from pathlib import Path
import tempfile
import os
from unittest.mock import patch
import numpy as np
import torch

from graphs.mesh import get_mesh
from graphs.dataset import MeltPoolGraphs, RunGraph, SampleBuilder, compute_stats
from solver.solver import simulate
from models import train as runner


class SimulatedStop(Exception):
    pass


def main():
    torch.set_num_threads(2)
    mesh = get_mesh("straight")
    sim = simulate(100., 1., mesh.waypoints, t_cool=2e-5)
    sim.update(name="lei_training_fixture", split="train", path="straight")
    run = RunGraph.from_simulation(sim, mesh)
    # Production load_stats reads JSON; reproduce that native-number contract.
    stats = json.loads(json.dumps(compute_stats([run], 5)))
    bins = {"train_fingerprint": stats["train_fingerprint"], "edges_m2": [1e-9, 2e-9]}

    def dataset(split, stride, stats, noise_std, augment):
        ds = MeltPoolGraphs.__new__(MeltPoolGraphs)
        ds.runs, ds.stride, ds.stats = [run], stride, stats
        ds.index = [(0, k) for k in (0, 1, 2, 3)]
        ds.noise_std, ds.augment = noise_std, augment
        ds.builder = SampleBuilder(stats, stride)
        return ds

    cfg = runner.load_config("configs/model_b.yaml")
    cfg["model"].update(width=16, layers=2)
    cfg["training"].update(steps=4, batch_size=2, val_every=2, log_every=1,
                           device="cpu", precision="fp32", augment=True)
    ca = runner.load_config("configs/model_a.yaml")
    assert cfg["data"] == ca["data"]
    # Verify architecture parameter matching independently of the tiny training fixture.
    torch.manual_seed(17)
    a = runner.build_model(dict(name="model_a", width=16, layers=2, hidden_layers=2))
    torch.manual_seed(17)
    b = runner.build_model(cfg["model"], stats)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.correction.state_dict()[key], rtol=0, atol=0)

    g = SampleBuilder(stats, 5).sample(run, 0, run.T(0))
    baseline = b.diffusion(g)
    angle = .7
    Q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = SampleBuilder(stats, 5).sample(run, 0, run.T(0), rotation=Q)
    torch.testing.assert_close(baseline, b.diffusion(rotated), rtol=0, atol=0)

    with tempfile.TemporaryDirectory(prefix="lei_training_") as tmp:
        with (patch.object(runner, "CKPT_DIR", tmp),
              patch.object(runner, "load_stats", return_value=stats),
              patch.object(runner, "load_size_bins", return_value=bins),
              patch.object(runner, "load_runs", return_value=[run]),
              patch.object(runner, "MeltPoolGraphs", side_effect=dataset)):
            full_score = runner.train(copy.deepcopy(cfg), "uninterrupted")
            real_save = runner.save

            def stop_after_checkpoint(obj, fn, **kwargs):
                real_save(obj, fn, **kwargs)
                if fn.endswith("last.pt") and obj["step"] == 2:
                    raise SimulatedStop()

            with patch.object(runner, "save", side_effect=stop_after_checkpoint):
                try:
                    runner.train(copy.deepcopy(cfg), "resumed")
                except SimulatedStop:
                    pass
                else:
                    raise AssertionError("simulated interruption did not fire")
            _, ckpt = runner.load_checkpoint(str(Path(tmp) / "resumed/last.pt"))
            assert ckpt["step"] == 2
            resumed_score = runner.train(copy.deepcopy(cfg), "resumed", ckpt=ckpt)
            full_model, full = runner.load_checkpoint(str(Path(tmp) / "uninterrupted/last.pt"))
            resumed_model, resumed = runner.load_checkpoint(str(Path(tmp) / "resumed/last.pt"))
            for key, value in full_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
            assert full_score == resumed_score and full["step"] == resumed["step"] == 4
            assert full["target"] == resumed["target"] == "dT"
            torch.testing.assert_close(full_model.eval()(g), resumed_model.eval()(g), rtol=0, atol=0)
            with (Path(tmp) / "uninterrupted/log.csv").open() as f:
                log_rows = list(csv.DictReader(f))
            first_loss, last_loss = float(log_rows[0]["train_loss"]), float(log_rows[-1]["train_loss"])
            assert last_loss < first_loss
            # A checkpoints still load through the same factory after B registration.
            a_path = Path(tmp) / "a_roundtrip.pt"
            torch.save(dict(model_config=dict(name="model_a", width=16, layers=2, hidden_layers=2),
                            stats=stats, model=a.state_dict()), a_path)
            loaded_a, _ = runner.load_checkpoint(str(a_path))
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, loaded_a.state_dict()[key], rtol=0, atol=0)

            # Full gate scoring on six temporary fixture checkpoints. These intentionally
            # reuse fixture weights, so no scientific seed comparison is exported.
            from analysis import run_model_b_gate as gate
            from analysis import model_b_corrections as maps
            from analysis import mesh_transfer as transfer
            gate_cfg = {"B": copy.deepcopy(cfg), "A": copy.deepcopy(cfg)}
            gate_cfg["A"]["model"] = dict(name="model_a", width=16, layers=2, hidden_layers=2)
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                for m in ("A", "B"):
                    for seed in (0, 1, 2):
                        fixture = copy.deepcopy(full)
                        expected = copy.deepcopy(gate_cfg[m])
                        expected["training"]["seed"] = seed
                        fixture.update(config=expected, model_config=expected["model"])
                        if m == "A":
                            fixture["model"] = a.state_dict()
                        dest = Path("checkpoints") / f"fixture_{m.lower()}_seed{seed}"
                        dest.mkdir(parents=True)
                        torch.save(fixture, dest / "best.pt")
                        torch.save(fixture, dest / "last.pt")
                with (patch.object(gate, "gate_configs", return_value=gate_cfg),
                      patch.object(gate, "check_data", return_value=(stats, bins)),
                      patch.object(gate, "load_runs", return_value=[run]),
                      patch.object(gate.subprocess, "check_output", side_effect=OSError("fixture has no Git"))):
                    gate.main(["--steps", "4", "--device", "cpu", "--precision", "fp32",
                               "--prefix", "fixture", "--out", "gate_results"])
                with Path("gate_results/scores.csv").open() as f:
                    scored = list(csv.DictReader(f))
                assert len(scored) == 18
                assert json.loads(Path("gate_results/manifest.json").read_text())["status"] == "complete"
                b_checkpoint = str(Path("checkpoints/fixture_b_seed0/best.pt"))
                with (patch.object(maps, "load_runs", return_value=[run]),
                      patch.object(maps, "load_stats", return_value=stats),
                      patch("sys.argv", ["maps", "--checkpoint", b_checkpoint, "--frames-per-run", "2"])):
                    maps.main()
                diagnostics = json.loads(Path("results/lei/corrections/summary.json").read_text())
                assert 0 <= diagnostics["area_fraction_in_band"] <= 1
                assert len(diagnostics["samples"]) == 2
                with (patch.object(transfer, "load_runs", return_value=[run]),
                      patch.object(transfer, "load_stats", return_value=stats),
                      patch.object(transfer, "load_size_bins", return_value=bins),
                      patch("sys.argv", ["transfer", "--checkpoints", b_checkpoint])):
                    transfer.main()
                assert len(json.loads(Path("results/lei/mesh_transfer/summary.json").read_text())) == 7
            finally:
                os.chdir(previous)
    result = dict(status="pass", fixture="one physical simulation, width 16, 2 layers, 4 CPU training steps",
                  gate_evidence=False, parameters_matched=True, identical_seed_initialisation=True,
                  rotation_invariant_physics=True, checkpoint_roundtrip=True,
                  model_a_checkpoint_compatible=True, gate_scoring_pipeline=True,
                  correction_map_export=True, mesh_transfer_export=True,
                  interrupted_resume_bit_exact=True, best_val_score=full_score)
    result.update(first_train_loss=first_loss, last_train_loss=last_loss)
    Path("results/lei").mkdir(parents=True, exist_ok=True)
    Path("results/lei/training_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
