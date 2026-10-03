"""Speed and memory check (docs/PLAN.md, Gate 3A): PPO decisions/s and peak GPU memory for the Phase 2 network and
the Phase 3 networks (dense and MoE cores, random weights: speed does not depend on them), then BC rows/s for the
Phase 2 network and the MoE one. Run it with the PC otherwise idle (~15 min, full GPU):

  .venv/Scripts/python scripts/speed_check.py
"""
import argparse
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.model import new_net  # noqa: E402
from trainer.test_model import DIMS  # noqa: E402

PHASE2 = ROOT / "checkpoints" / "bc" / "update0060000.pt"
PHASE2_PPO_SPEED = 190  # decisions/s of ppo-bc2 (16 workers x 2 games), docs/PLAN.md Phase 2 results


def gpu_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def run(name, module, flags):
    """Runs one trainer command; returns its mean speed over the logged points after the first two, and the peak GPU
    memory meanwhile (MiB, the whole card)."""
    stop, peak = threading.Event(), [0]

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_memory() or 0)
            stop.wait(0.5)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        subprocess.run([sys.executable, "-m", module, "--run", name, "--port", "0", *flags], cwd=ROOT, check=True)
    finally:
        stop.set()
        sampler.join()
    acc = EventAccumulator(str(ROOT / "runs" / name))
    acc.Reload()
    tag = "perf/decisions_per_s" if module == "trainer.ppo" else "perf/samples_per_s"
    points = [e.value for e in acc.Scalars(tag)][2:]
    return sum(points) / max(len(points), 1), peak[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ppo-workers", type=int, default=16)
    p.add_argument("--ppo-games", type=int, default=2)
    p.add_argument("--ppo-updates", type=int, default=10)
    p.add_argument("--ppo-minibatches", type=int, help="PPO minibatches per epoch (default: the trainer's, 8 for ranked)")
    p.add_argument("--bc-workers", type=int, default=14)
    p.add_argument("--bc-games", type=int, default=6)
    p.add_argument("--bc-updates", type=int, default=30)
    p.add_argument("--skip-bc", action="store_true", help="only the PPO runs")
    a = p.parse_args(argv)
    speed = ROOT / "checkpoints" / "speed"
    speed.mkdir(parents=True, exist_ok=True)
    for core in ("dense", "moe"):
        args = {"arch": "v2", "core": core, "widths": "64,128,192", "run": f"speed-{core}"}
        torch.manual_seed(0)
        torch.save({"model": new_net(DIMS, args).state_dict(), "args": args, "update": 0}, speed / f"{core}.pt")
    idle = gpu_memory()
    ppo = ["--setup", "ranked1v1", "--kl-coef", "0.2", "--lr", "1e-4", "--ent-coef", "0", "--workers",
           str(a.ppo_workers), "--games-per-worker", str(a.ppo_games), "--updates", str(a.ppo_updates)]
    if a.ppo_minibatches:
        ppo += ["--minibatches", str(a.ppo_minibatches)]
    bc = ["--workers", str(a.bc_workers), "--games-per-worker", str(a.bc_games), "--val-workers", "0", "--updates",
          str(a.bc_updates), "--log-every", str(max(1, a.bc_updates // 6))]
    runs = {"speed-ppo-v1": ("trainer.ppo", ppo + ["--init", str(PHASE2), "--opponent", str(PHASE2)]),
            "speed-ppo-dense": ("trainer.ppo", ppo + ["--init", str(speed / "dense.pt"), "--opponent",
                                                      str(speed / "dense.pt")]),
            "speed-ppo-moe": ("trainer.ppo", ppo + ["--init", str(speed / "moe.pt"), "--opponent", str(speed / "moe.pt")]),
            "speed-bc-v1": ("trainer.bc", bc + ["--init", str(PHASE2)]),
            "speed-bc-moe": ("trainer.bc", bc + ["--core", "moe"])}
    if a.skip_bc:
        runs = {name: spec for name, spec in runs.items() if spec[0] == "trainer.ppo"}
    results = {}
    try:
        for name, (module, flags) in runs.items():
            results[name] = run(name, module, flags)
    finally:
        for name in runs:
            shutil.rmtree(ROOT / "checkpoints" / name, ignore_errors=True)
            shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        shutil.rmtree(speed, ignore_errors=True)
    v1 = results["speed-ppo-v1"][0]
    print(f"\nGPU memory before the runs: {idle} MiB (other programs)")
    for name, (rate, peak) in results.items():
        unit = "decisions/s" if "ppo" in name else "rows/s"
        extra = f"  {rate / v1:.0%} of v1, {rate / PHASE2_PPO_SPEED:.0%} of Phase 2's {PHASE2_PPO_SPEED}" \
            if "ppo" in name and name != "speed-ppo-v1" else ""
        print(f"{name:16} {rate:8.1f} {unit:12} peak {peak} MiB{extra}")


if __name__ == "__main__":
    main()
