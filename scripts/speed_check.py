"""Speed and memory check (docs/PLAN.md, 3B). It measures:
- PPO decisions/s and peak GPU memory for the Phase 2 network and a v4 network. The v4 network has every action and
  the SwiGLU experts the BC refresh trains, with random weights, or the checkpoint given with --v4. Gate 3B times the
  BC clone: a random network acts on most decisions, which costs more simulation;
- with --v5, a fresh v5 network (step 6: 20 actions, the same SwiGLU experts) in the same PPO setup, as a share of v4's
  decisions/s with its own peak memory. The budget (design, "Speed and memory"): at least 95% of v4's and no spill;
- BC rows/s for the v4 network.
Gate 3B: v4 keeps at least 75% of the Phase 2 network's decisions/s in the same run. Run it with the PC otherwise idle
(~15 min, full GPU):

  .venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12
  .venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12 --skip-bc --v4 checkpoints/bc3b-v4/latest.pt
A code A/B (PLAN 4A, "Speed work") times only the clone: add --skip-v1.
Step 6's check of the v5 network against v4 (PPO only, ~15 min):
  .venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12 --skip-bc --skip-v1 --v5
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
from trainer.test_model import DIMS, DIMS_V5  # noqa: E402

PHASE2 = ROOT / "checkpoints" / "bc" / "update0060000.pt"
PHASE2_PPO_SPEED = 190  # decisions/s of ppo-bc2 (16 workers x 2 games), docs/PLAN.md Phase 2 results


def gpu_memory(field="used"):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu=memory.{field}", "--format=csv,noheader,nounits"],
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
    tags = ("perf/rollout_share", "perf/env_share", "perf/gpu_reserved_gb", "perf/gpu_allocated_gb")
    last = {t: acc.Scalars(t)[-1].value for t in tags if t in acc.Tags()["scalars"]}
    return sum(points) / max(len(points), 1), peak[0], last


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
    p.add_argument("--skip-v1", action="store_true", help="no Phase 2 network run (the code A/B probes)")
    p.add_argument("--v5", action="store_true", help="also time a fresh v5 network, as a share of v4's decisions/s")
    p.add_argument("--v4", help="a v4 checkpoint to time in PPO instead of random weights (Gate 3B: the BC clone)")
    a = p.parse_args(argv)
    speed = ROOT / "checkpoints" / "speed"
    speed.mkdir(parents=True, exist_ok=True)
    v4 = a.v4
    if v4 is None:
        args = {"arch": "v4", "core": "moe-swiglu", "widths": "64,128,192", "run": "speed-v4"}
        torch.manual_seed(0)
        v4 = str(speed / "v4.pt")
        torch.save({"model": new_net(DIMS, args).state_dict(), "args": args, "update": 0}, v4)
    v5 = str(speed / "v5.pt")
    if a.v5:
        args = {"arch": "v5", "core": "moe-swiglu", "widths": "64,128,192", "run": "speed-v5"}
        torch.manual_seed(0)
        torch.save({"model": new_net(DIMS_V5, args).state_dict(), "args": args, "update": 0}, v5)
    idle = gpu_memory()
    ppo = ["--setup", "ranked1v1", "--kl-coef", "0.2", "--lr", "1e-4", "--ent-coef", "0", "--workers",
           str(a.ppo_workers), "--games-per-worker", str(a.ppo_games), "--updates", str(a.ppo_updates)]
    if a.ppo_minibatches:
        ppo += ["--minibatches", str(a.ppo_minibatches)]
    bc = ["--workers", str(a.bc_workers), "--games-per-worker", str(a.bc_games), "--val-workers", "0", "--updates",
          str(a.bc_updates), "--log-every", str(max(1, a.bc_updates // 6))]
    runs = {"speed-ppo-v1": ("trainer.ppo", ppo + ["--init", str(PHASE2), "--opponent", str(PHASE2)]),
            "speed-ppo-v4": ("trainer.ppo", ppo + ["--init", v4, "--opponent", v4]),
            "speed-ppo-v5": ("trainer.ppo", ppo + ["--init", v5, "--opponent", v5]),
            "speed-bc-v4": ("trainer.bc", bc + ["--core", "moe-swiglu"])}
    if not a.v5:
        del runs["speed-ppo-v5"]
    if a.skip_bc:
        runs = {name: spec for name, spec in runs.items() if spec[0] == "trainer.ppo"}
    if a.skip_v1:
        del runs["speed-ppo-v1"]
    results = {}
    try:
        for name, (module, flags) in runs.items():
            results[name] = run(name, module, flags)
    finally:
        for name in runs:
            shutil.rmtree(ROOT / "checkpoints" / name, ignore_errors=True)
            shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        shutil.rmtree(speed, ignore_errors=True)
    v1 = results["speed-ppo-v1"][0] if "speed-ppo-v1" in results else None
    print(f"\nGPU memory before the runs: {idle} MiB (other programs)")
    for name, (rate, peak, last) in results.items():
        unit = "decisions/s" if "ppo" in name else "rows/s"
        extra = f"  {rate / v1:.0%} of v1, {rate / PHASE2_PPO_SPEED:.0%} of Phase 2's {PHASE2_PPO_SPEED}" \
            if "ppo" in name and name != "speed-ppo-v1" and v1 else ""
        if last:
            extra += (f"  (rollouts {last['perf/rollout_share']:.0%} of the time, the simulation "
                      f"{last['perf/env_share']:.0%}, "
                      f"allocator peak {last['perf/gpu_reserved_gb'] * 1024:.0f} MiB, "
                      f"{last['perf/gpu_allocated_gb'] * 1024:.0f} in use)")
        print(f"{name:20} {rate:8.1f} {unit:12} peak {peak} MiB{extra}")
    rate, peak, last = results["speed-ppo-v4"]
    share = f"{rate / v1:.0%} of the Phase 2 network's decisions/s in this run (Gate 3B: >= 75%)" if v1 else \
        f"{rate:.1f} decisions/s"
    print(f"\nPPO, v4: {share}; its own peak {peak - (idle or 0)} MiB")
    if a.v5:
        rate5, peak5, last5 = results["speed-ppo-v5"]
        card = gpu_memory("total")
        reserved = last5.get("perf/gpu_reserved_gb", 0) * 1024  # the allocator's peak: past the card when it spills
        fits = "SPILLED" if card and reserved > card else "no spill"
        print(f"PPO, v5: {rate5 / rate:.0%} of v4's decisions/s ({rate5:.1f} against {rate:.1f}; budget >= 95%); its "
              f"own peak {peak5 - (idle or 0)} MiB against v4's {peak - (idle or 0)}; {fits}: allocator peak "
              f"{reserved:.0f} of {card} MiB")


if __name__ == "__main__":
    main()
