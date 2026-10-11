"""Where PPO's time goes (PLAN 4A, "Speed work"). It runs the real loop (trainer.ppo, as speed_check.py runs it, with
the 3B clone) for 5 updates and records update 4 with torch.profiler. Every update is also timed without the profiler,
by part. From the trace, per phase (rollout, learning) and per part (the innermost label open on the main thread), it
reports the CPU time, the time the GPU sat idle, the syncs (the main thread waiting for the GPU), and the GPU time and
kernels launched. ~2 min, with the PC idle:

  .venv/Scripts/python scripts/profile_ppo.py [trainer.ppo flags, which override the defaults below]
  .venv/Scripts/python scripts/profile_ppo.py --analyze runs/profile/trace.json   # the tables of a saved trace

Results go to runs/profile/<date>-<time>.json, and the trace (~0.4 GB) to runs/profile/trace.json.
"""
import bisect
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "profile"
RUN = "profile-v4"
CKPT = "checkpoints/bc3b-v4/latest.pt"
ARGS = ["--run", RUN, "--port", "0", "--setup", "ranked1v1", "--kl-coef", "0.2", "--lr", "1e-4", "--ent-coef", "0",
        "--workers", "16", "--games-per-worker", "2", "--minibatches", "12", "--updates", "5", "--init", CKPT,
        "--opponent", CKPT]
SYNC = {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"}  # a blocking copy ends in one too
COLUMNS = ("cpu_ms", "gpu_idle_ms", "sync_ms", "syncs", "gpu_ms", "kernels")


def analyze(path):
    """{phase: {"wall_ms", "gpu_idle_ms", "parts": {label: COLUMNS}}} from a trace this script saved."""
    ev = [e for e in json.load(open(path, encoding="utf-8"))["traceEvents"] if e.get("ph") == "X"]
    ann = [e for e in ev if e.get("cat") == "user_annotation"]
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    runtime = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    launch = {e["args"]["correlation"]: e for e in runtime if "correlation" in e.get("args", {})}
    main = next(e for e in ann if e["name"] == "ppo.learn")["tid"]
    result = {}
    for phase in ("rollout", "learn"):
        p = next(e for e in ann if e["name"] == "ppo." + phase)
        lo, hi = p["ts"], p["ts"] + p["dur"]
        points, stack, segs, last = [], [], [], lo  # segs: (start, end, the innermost label open on the main thread)
        for i, e in enumerate(a for a in ann if a["tid"] == main and lo <= a["ts"] < hi and a["dur"] > 0):
            points += [(e["ts"], 1, i, e["name"]), (e["ts"] + e["dur"], 0, i, e["name"])]
        for t, opens, i, name in sorted(points, key=lambda x: (x[0], x[1])):
            if t > last:
                segs.append((last, t, stack[-1][1] if stack else "(none)"))
            last = t
            stack = stack + [(i, name)] if opens else [s for s in stack if s[0] != i]
        starts = [s[0] for s in segs]
        label_at = lambda t: segs[max(bisect.bisect_right(starts, t) - 1, 0)][2]
        idle, cur = [], lo
        for s, e in sorted((k["ts"], k["ts"] + k["dur"]) for k in kernels if lo <= k["ts"] < hi):
            if s > cur:
                idle.append((cur, s))
            cur = max(cur, e)
        idle += [(cur, hi)] if hi > cur else []
        parts, j = defaultdict(lambda: dict.fromkeys(COLUMNS, 0.0)), 0
        for s0, e0, name in segs:
            parts[name]["cpu_ms"] += (e0 - s0) / 1e3
            while j < len(idle) and idle[j][1] <= s0:
                j += 1
            k = j
            while k < len(idle) and idle[k][0] < e0:
                parts[name]["gpu_idle_ms"] += (min(e0, idle[k][1]) - max(s0, idle[k][0])) / 1e3
                k += 1
        for r in runtime:
            if r["tid"] == main and lo <= r["ts"] < hi and r["name"] in SYNC:
                parts[label_at(r["ts"])]["sync_ms"] += r["dur"] / 1e3
                parts[label_at(r["ts"])]["syncs"] += 1
        for k in kernels:  # the backward pass's kernels come from the autograd thread while the main thread waits in it
            r = launch.get(k.get("args", {}).get("correlation"))
            if r is not None and lo <= r["ts"] < hi:
                parts[label_at(r["ts"])]["gpu_ms"] += k["dur"] / 1e3
                parts[label_at(r["ts"])]["kernels"] += 1
        result[phase] = {"wall_ms": p["dur"] / 1e3, "gpu_idle_ms": sum(e - s for s, e in idle) / 1e3,
                         "parts": dict(sorted(parts.items(), key=lambda kv: -kv[1]["cpu_ms"]))}
    return result


def print_tables(result):
    for phase, r in result.items():
        print(f"\n{phase}: {r['wall_ms']:.0f} ms, GPU idle {r['gpu_idle_ms']:.0f} ms "
              f"({r['gpu_idle_ms'] / r['wall_ms']:.0%}); per part (exclusive):")
        print(f"{'':36}" + "".join(f"{c:>12}" for c in COLUMNS))
        for name, v in r["parts"].items():
            print(f"{name:36}" + "".join(f"{v[c]:12.0f}" for c in COLUMNS))


def main():
    import trainer.ppo as ppo
    from trainer import control, env, model

    argv = ARGS + sys.argv[1:]
    a = ppo.parse_args(argv)
    rows = (a.steps or 48) * a.workers * a.games_per_worker
    OUT.mkdir(parents=True, exist_ok=True)
    trace = OUT / "trace.json"
    cur, updates, phase = defaultdict(float), [], {"rf": None}

    def labelled(fn, name, sync=False):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            try:
                with record_function(name):
                    return fn(*args, **kwargs)
            finally:
                cur[name] += time.perf_counter() - t
                if sync and phase["rf"] is not None:
                    cur[phase["name"] + "_sync"] += time.perf_counter() - t
        return wrapper

    def net(fn):
        grad, nograd = labelled(fn, "ppo.net_grad"), labelled(fn, "ppo.net_nograd")
        return lambda self, obs: (grad if torch.is_grad_enabled() else nograd)(self, obs)

    def begin(name):
        end()
        phase.update(rf=record_function("ppo." + name), name=name, t=time.perf_counter())
        phase["rf"].__enter__()

    def end():
        if phase["rf"] is not None:
            phase["rf"].__exit__(None, None, None)
            cur[phase["name"]] += time.perf_counter() - phase["t"]
            phase["rf"] = None

    env.VecEnv.step = labelled(env.VecEnv.step, "ppo.env_step")  # its own time: the workers' simulation and the pipes
    env.VecEnv._absorb = labelled(env.VecEnv._absorb, "ppo.env_absorb")  # the observations
    model.Policy.forward = net(model.Policy.forward)
    model.PolicyV1.forward = net(model.PolicyV1.forward)
    for m in ("__init__", "log_prob", "kl", "entropy", "sample"):
        setattr(model.ActionDist, m, labelled(getattr(model.ActionDist, m), "ppo.dist_" + m.strip("_")))
    torch.Tensor.backward = labelled(torch.Tensor.backward, "ppo.backward")
    torch.nn.utils.clip_grad_norm_ = labelled(torch.nn.utils.clip_grad_norm_, "ppo.clip")
    for m in ("item", "tolist", "cpu", "__bool__"):  # explicit waits for the GPU (Adam's .item() are on the CPU)
        fn = getattr(torch.Tensor, m)
        timed = labelled(fn, "ppo.sync_" + m.strip("_"), sync=True)
        setattr(torch.Tensor, m, lambda self, *args, fn=fn, timed=timed, **kw: (timed if self.is_cuda else fn)(
            self, *args, **kw))
    batches = ppo.HostRows.batches

    def host_batches(self, idx, device):
        it = batches(self, idx, device)
        while True:
            with record_function("ppo.host_next"):
                t = time.perf_counter()
                o = next(it, None)
                cur["ppo.host_next"] += time.perf_counter() - t
            if o is None:
                return
            yield o

    wait, empty_cache, summary = control.Control.wait, torch.cuda.empty_cache, ppo.Outcomes.summary

    def control_wait(self):  # the first one of an update starts its rollout
        if phase["rf"] is None:
            begin("rollout")
        return wait(self)

    def learn_begins():  # trainer.ppo empties the cache once per update, between the rollout and learning
        begin("learn")
        empty_cache()

    def learn_ends(self):  # and asks for the outcomes right after learning
        end()
        updates.append(dict(cur))
        cur.clear()
        prof.step()
        return summary(self)

    ppo.HostRows.batches = host_batches
    control.Control.wait, torch.cuda.empty_cache, ppo.Outcomes.summary = control_wait, learn_begins, learn_ends
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule=schedule(wait=2, warmup=1, active=1, repeat=1),
                   on_trace_ready=lambda p: p.export_chrome_trace(str(trace)))
    prof.start()
    try:
        ppo.main(argv)
    finally:
        prof.stop()
        for d in ("checkpoints", "runs"):
            shutil.rmtree(ROOT / d / RUN, ignore_errors=True)
    keys = ("rollout", "rollout_sync", "ppo.env_step", "ppo.env_absorb", "learn", "learn_sync", "ppo.host_next",
            "ppo.backward")
    print(f"\n{rows} decisions per update; seconds per update, without the profiler but update 4:")
    print("update  dec/s" + "".join(f"{k.replace('ppo.', ''):>14}" for k in keys))
    for i, u in enumerate(updates, 1):
        print(f"{i:>6} {rows / (u['rollout'] + u['learn']):6.1f}" + "".join(f"{u.get(k, 0):14.2f}" for k in keys)
              + ("  (profiled)" if i == 4 else ""))
    result = {"args": argv, "updates": updates, "profile": analyze(trace) if trace.exists() else None}
    if result["profile"]:
        print_tables(result["profile"])
    path = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"\nsaved {path.relative_to(ROOT)}; the trace in {trace.relative_to(ROOT)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--analyze"]:
        print_tables(analyze(sys.argv[2]))
    else:
        main()
