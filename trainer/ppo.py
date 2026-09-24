"""PPO training (CleanRL-style) with pause/save/resume, for a PC that is also used daily.

  Train:   .venv/Scripts/python -m trainer.ppo --run onion
  Resume:  .venv/Scripts/python -m trainer.ppo --run onion --resume
  Control: http://127.0.0.1:8787 or python -m trainer.ctl pause|resume|save|stop|throttle|status
  Charts:  .venv/Scripts/tensorboard --logdir runs
"""
import argparse
import collections
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trainer.control import Control, serve
from trainer.env import GAME_DIR, ROOT, VecEnv
from trainer.model import ACT_NAMES, ActionDist, Policy

KEEP_ON_RESUME = ("resume", "port", "duty", "gpu_temp_limit", "updates")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="onion", help="name for checkpoints/<run> and runs/<run>")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/latest.pt")
    p.add_argument("--map", default="Onion")
    p.add_argument("--normal-size", action="store_true", help="Normal map size instead of Compact")
    p.add_argument("--nations", type=int, default=3)
    p.add_argument("--difficulty", default="Easy")
    p.add_argument("--bots", type=int, default=0)
    p.add_argument("--timer", type=int, default=5, help="game timer in minutes")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--steps", type=int, default=128, help="decisions per game per update")
    p.add_argument("--updates", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--widths", default="64,128,192", help="CNN stage widths")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=25, help="updates between checkpoints")
    p.add_argument("--port", type=int, default=8787, help="control page port (0 = any free port)")
    p.add_argument("--duty", type=float, default=1.0, help="start throttled: fraction of time spent working")
    p.add_argument("--gpu-temp-limit", type=int, default=0, help="auto-pause at this GPU temperature in C (0 = off)")
    return p.parse_args(argv)


def env_config(a):
    return {"map": a.map, "compact": not a.normal_size, "nations": a.nations, "difficulty": a.difficulty,
            "bots": a.bots, "timerMinutes": a.timer, "seats": 1}


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation. rewards, values, dones: [T, N]; dones[t] = the episode ended after step t
    (the next observation already belongs to a new game). last_value: [N], the value after the last step."""
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * live - values[t]
        running = delta + gamma * lam * live * running
        adv[t] = running
    return adv


def save_checkpoint(path, **state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash during the save never corrupts the previous checkpoint


def game_commit():
    return subprocess.run(["git", "-C", str(GAME_DIR), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


class Outcomes:
    """Rolling episode results, and how well the value head predicted each finished game's reward, by game phase:
    sign accuracy (trivially high while one outcome dominates, e.g. an agent that rarely wins) and the correlation
    between value and final reward (the gate's measure; rank-based rewards keep it defined without wins)."""

    def __init__(self, rows, window=200):
        self.episodes = collections.deque(maxlen=window)
        self.pending = [[] for _ in range(rows)]
        self.samples = collections.deque(maxlen=50_000)  # (phase bucket, value, final reward)

    def record(self, progress, values):
        for row, (p, v) in enumerate(zip(progress, values)):
            self.pending[row].append((p, v))

    def finish(self, infos, seats):
        for info in infos:
            row = info["env"] * seats + info["seat"]
            self.samples.extend((min(int(p * 3), 2), v, info["reward"]) for p, v in self.pending[row])
            self.pending[row] = []
            self.episodes.append(info)

    def summary(self):
        if not self.episodes:
            return {}
        ep = list(self.episodes)
        out = {"episodes/win_rate": float(np.mean([e["rank"] == 1 for e in ep])),
               "episodes/mean_reward": float(np.mean([e["reward"] for e in ep])),
               "episodes/mean_rank": float(np.mean([e["rank"] for e in ep])),
               "episodes/ticks": float(np.mean([e["ticks"] for e in ep]))}
        s = np.array(self.samples, float).reshape(-1, 3)
        for bucket, name in enumerate(("early", "mid", "late")):
            v, r = s[s[:, 0] == bucket, 1], s[s[:, 0] == bucket, 2]
            if len(v) > 0:
                out[f"value/outcome_acc_{name}"] = float(np.mean((v > 0) == (r > 0)))
            if len(v) > 1 and v.std() > 0 and r.std() > 0:
                out[f"value/outcome_corr_{name}"] = float(np.corrcoef(v, r)[0, 1])
        return out


def main(argv=None):
    a = parse_args(argv)
    ckpt_dir = ROOT / "checkpoints" / a.run
    state = None
    if a.resume:
        state = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        a = argparse.Namespace(**{**state["args"], **{k: getattr(a, k) for k in KEEP_ON_RESUME}})
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run_dir = ROOT / "runs" / a.run

    envs = VecEnv(env_config(a), a.workers, a.games_per_worker, seed=f"{a.run}-{a.seed}", device=device,
                  episodes=state["episodes"] if state else None, log_dir=run_dir / "workers")
    model = Policy(envs.dims, tuple(int(w) for w in a.widths.split(","))).to(device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    update = global_step = 0
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        update, global_step = state["update"], state["global_step"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    control = Control(duty=a.duty, gpu_temp_limit=a.gpu_temp_limit)
    server = serve(control, a.port)
    control.note(f"control page on http://127.0.0.1:{server.server_address[1]}")
    writer = SummaryWriter(run_dir)
    commit = game_commit()

    def checkpoint():
        # Games in progress are dropped on quit; a resume starts each slot at its next episode, so no seed replays.
        s = dict(model=model.state_dict(), optimizer=opt.state_dict(), args=vars(a), update=update,
                 global_step=global_step, episodes=[e + 1 for e in envs.episodes], game_commit=commit,
                 rng={"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all()})
        save_checkpoint(ckpt_dir / "latest.pt", **s)
        if update % 250 == 0 and update > 0:
            save_checkpoint(ckpt_dir / f"update{update:06d}.pt", **s)
        control.note(f"saved checkpoint at update {update}")

    T, N = a.steps, envs.N
    obs = envs.obs
    buf = {k: torch.zeros((T,) + tuple(v.shape), dtype=v.dtype, device=device) for k, v in obs.items()}
    actions = torch.zeros(T, N, 4, dtype=torch.long, device=device)
    logps, values, rewards, dones = (torch.zeros(T, N, device=device) for _ in range(4))
    outcomes = Outcomes(N)
    act_counts, invalid_count = np.zeros(envs.T), 0

    def rollout():
        """Collects T steps from every game; False if a stop was requested meanwhile."""
        nonlocal obs, global_step, invalid_count
        for step in range(T):
            if not control.wait():
                return False
            t0 = time.perf_counter()
            for k, v in obs.items():
                buf[k][step] = v
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(obs)
                dist = ActionDist(out, obs)
                act = dist.sample()
                actions[step], logps[step], values[step] = act, dist.log_prob(act), out["value"].float()
            act_np = act.cpu().numpy()
            outcomes.record(obs["scalars"][:, 0].tolist(), values[step].tolist())
            obs, rew, done, infos, invalid = envs.step(act_np)
            rewards[step] = torch.from_numpy(rew).to(device)
            dones[step] = torch.from_numpy(done).to(device)
            outcomes.finish(infos, envs.S)
            np.add.at(act_counts, act_np[:, 0], 1)
            invalid_count += int(invalid.sum())
            global_step += N
            control.throttle(time.perf_counter() - t0)
        return True

    def learn():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            last_value = model(obs)["value"].float()
        adv = gae(rewards, values, dones, last_value, a.gamma, a.gae_lambda)
        ret = adv + values
        flat = {k: v.flatten(0, 1) for k, v in buf.items()}
        b_act, b_logp, b_adv, b_ret = actions.flatten(0, 1), logps.flatten(), adv.flatten(), ret.flatten()
        stats = collections.defaultdict(list)
        for _ in range(a.epochs):
            for mb in torch.randperm(T * N, device=device).split(T * N // a.minibatches):
                o = {k: v[mb] for k, v in flat.items()}
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(o)
                dist = ActionDist(out, o)
                logp, ent = dist.log_prob(b_act[mb]), dist.entropy()
                log_ratio = logp - b_logp[mb]
                ratio = log_ratio.exp()
                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                v_loss = 0.5 * (out["value"].float() - b_ret[mb]).pow(2).mean()
                loss = pg_loss - a.ent_coef * ent.mean() + a.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    stats["loss/policy"].append(pg_loss.item())
                    stats["loss/value"].append(v_loss.item())
                    stats["loss/entropy"].append(ent.mean().item())
                    stats["loss/approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["loss/clip_frac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
        metrics = {k: float(np.mean(v)) for k, v in stats.items()}
        var = ret.var().item()
        metrics["value/explained_variance"] = float("nan") if var == 0 else 1 - (ret - values).var().item() / var
        return metrics

    control.set_state("running")
    try:
        while update < a.updates:
            t0 = time.perf_counter()
            if not rollout():
                break
            t1 = time.perf_counter()
            metrics = learn()
            control.throttle(time.perf_counter() - t1)
            update += 1
            metrics.update(outcomes.summary())
            metrics["perf/decisions_per_s"] = T * N / (time.perf_counter() - t0)
            metrics["actions/invalid_rate"] = invalid_count / (T * N)
            for i, name in enumerate(ACT_NAMES):
                metrics[f"actions/{name}"] = float(act_counts[i] / (T * N))
            act_counts[:], invalid_count = 0, 0
            for k, v in metrics.items():
                writer.add_scalar(k, v, global_step)
            control.update(update=update, global_step=global_step, metrics=metrics)
            if update % a.save_every == 0 or control.take_save():
                checkpoint()
    except KeyboardInterrupt:
        control.note("interrupted")
    finally:
        control.set_state("saving")
        checkpoint()
        envs.close()
        writer.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
