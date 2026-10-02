"""Plays policies against each other on fixed seeds (the same games every time, so results compare directly).

  .venv/Scripts/python -m trainer.eval --a checkpoints/bc/latest.pt --b scripted      (ranked 1v1, 256 games)
  .venv/Scripts/python -m trainer.eval --a checkpoints/ppo/latest.pt --b checkpoints/bc/latest.pt
  .venv/Scripts/python -m trainer.eval --setup onion --a random                        (Phase 1: one seat vs 3 nations)

A policy is "random", "scripted" or a checkpoint path. The results are the first policy's; in 2-seat setups the two
policies swap seats from one game slot to the next.
"""
import argparse
import json
import sys

import numpy as np
import torch

from trainer.control import REPLAYS
from trainer.env import ONION, RANKED_1V1, VecEnv
from trainer.model import ActionDist, RandomPolicy, load_net
from trainer.ppo import game_commit

SETUPS = {"onion": ONION, "ranked1v1": RANKED_1V1}


def summarize(results):
    n = len(results)
    reward = np.array([r["reward"] for r in results], float)
    win = np.array([r["rank"] == 1 for r in results], float)
    rank = np.array([r["rank"] for r in results], float)
    se = lambda v: float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    seat = np.array([r.get("seat", 0) for r in results])
    by_seat = {f"win_rate_seat{s}": float(win[seat == s].mean()) for s in np.unique(seat)} if len(set(seat)) > 1 else {}
    return {"episodes": n, "win_rate": float(win.mean()), "win_rate_se": se(win), **by_seat,
            "mean_reward": float(reward.mean()), "mean_reward_se": se(reward), "mean_rank": float(rank.mean())}


class Scripted:
    """The scripted expander (PLAN Phase 1 diagnosis): spawns on a random legal cell at the first decision (and when
    forced), then every `every` decisions attacks unowned land, or else a random attackable bot, with troop bin
    `bin`; otherwise waits."""

    def __init__(self, bin=1, every=4, seed=0):
        self.bin, self.every, self.rng, self.step = bin, every, np.random.default_rng(seed), 0

    def __call__(self, obs):
        types, attack, cells = (obs[k].cpu().numpy() for k in ("type_mask", "attack_mask", "cell_mask"))
        bots = obs["players"][..., 5].cpu().numpy() > 0.5  # player feature 5: bot (bridge/obs.ts)
        rows = []
        for i in range(types.shape[0]):
            act = [0, 0, 0, 0]
            if types[i, 1] and (self.step == 0 or not types[i, 0]) and cells[i, 0].any():
                act = [1, int(self.rng.choice(np.flatnonzero(cells[i, 0]))), 0, 0]
            elif types[i, 2] and self.step % self.every == 0:
                targets = [0] if attack[i, 0] else np.flatnonzero(attack[i] & bots[i])
                if len(targets):
                    act = [2, 0, int(self.rng.choice(targets)), self.bin]
            rows.append(act)
        self.step += 1
        return torch.tensor(rows, dtype=torch.long, device=obs["type_mask"].device)


def load_policy(spec, dims, device="cuda"):
    """obs -> actions [n, 4] for "random", "scripted" or a checkpoint path (sampling, as in training)."""
    if spec == "scripted":
        return Scripted()
    model = RandomPolicy(dims).to(device) if spec == "random" else load_net(spec, dims, device)

    def act(obs):
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                out = model(obs)
            return ActionDist(out, obs).sample()

    return act


def seats_for(E, S):
    """seats[e][p]: the seat policy p plays in game slot e (the policies rotate through the seats)."""
    return [[(e + p) % S for p in range(S)] for e in range(E)]


def play(specs, config, episodes, workers, games_per_worker, device="cuda", seed="eval", record=None):
    """Plays the first ceil(episodes / games) episodes of every game slot; returns the summary of policy 0, and the IDs
    of the games saved with `record` ({"dir", "gitCommit", "games"}). Sampling uses a fixed torch seed, so a given set
    of policies always gives the same result."""
    torch.manual_seed(0)
    envs = VecEnv(config, workers, games_per_worker, seed=seed, device=device, record=record)
    try:
        S, E = envs.S, envs.E
        if len(specs) != S:
            raise ValueError(f"{S} seats need {S} policies, got {len(specs)}")
        policies = [load_policy(s, envs.dims, device) for s in specs]
        seats = seats_for(E, S)
        rows = [torch.tensor([e * S + seats[e][p] for e in range(E)], device=device) for p in range(S)]
        per_slot = -(-episodes // E)
        counts, results, seen, obs = np.zeros(E, int), [], [], envs.obs
        while counts.min() < per_slot:
            act = torch.zeros(envs.N, 4, dtype=torch.long, device=device)
            for p, policy in enumerate(policies):
                act[rows[p]] = policy({k: v[rows[p]] for k, v in obs.items()})
            obs, _, _, infos, _ = envs.step(act.cpu().numpy())
            seen += infos
            for info in infos:
                e = info["env"]
                if info["seat"] == seats[e][0] and counts[e] < per_slot:
                    counts[e] += 1
                    results.append(info)
                    if len(results) % E == 0:  # no control page here: show signs of life
                        print(f"{len(results)}/{per_slot * E} games", file=sys.stderr, flush=True)
    finally:
        envs.close()
    return {**summarize(results), "recorded": sorted({i["recorded"] for i in seen if i.get("recorded")})}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="the scored policy: random, scripted or a checkpoint path")
    p.add_argument("--b", help="its opponent in 2-seat setups")
    p.add_argument("--setup", choices=sorted(SETUPS), default="ranked1v1")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--seed", default="eval")
    p.add_argument("--record", type=int, default=0, help="games saved to replays/ per worker, to watch them")
    a = p.parse_args(argv)
    config = SETUPS[a.setup]
    if config["seats"] == 2 and a.b is None:
        p.error("--b is required in 2-seat setups")
    record = {"dir": str(REPLAYS), "gitCommit": game_commit(), "games": a.record} if a.record > 0 else None
    result = play([a.a, a.b][:config["seats"]], config, a.episodes, a.workers, a.games_per_worker, seed=a.seed,
                  record=record)
    print(json.dumps({"a": a.a, "b": a.b, "setup": a.setup, **result}, indent=2))
    for game_id in result["recorded"]:
        print(f"watch: python -m trainer.watch (or a running trainer), then http://localhost:9000/game/{game_id}")


if __name__ == "__main__":
    main()
