"""Evaluate a policy on a fixed set of games (the same seeds for every policy, so results compare directly).

  .venv/Scripts/python -m trainer.eval --policy random --like checkpoints/onion/latest.pt
  .venv/Scripts/python -m trainer.eval --policy checkpoints/onion/latest.pt
"""
import argparse
import json

import numpy as np
import torch

from trainer.env import VecEnv
from trainer.model import ActionDist, Policy, RandomPolicy
from trainer.ppo import env_config, parse_args


def summarize(results):
    n = len(results)
    reward = np.array([r["reward"] for r in results], float)
    win = np.array([r["rank"] == 1 for r in results], float)
    rank = np.array([r["rank"] for r in results], float)
    se = lambda v: float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"episodes": n, "win_rate": float(win.mean()), "win_rate_se": se(win), "mean_reward": float(reward.mean()),
            "mean_reward_se": se(reward), "mean_rank": float(rank.mean())}


def evaluate(make_policy, config, episodes, workers, games_per_worker, device="cuda", seed="eval"):
    """Plays the first ceil(episodes / games) episodes of every game slot; `make_policy(dims)` builds the policy.
    Sampling uses a fixed torch seed, so a given policy always gives the same result."""
    torch.manual_seed(0)
    envs = VecEnv(config, workers, games_per_worker, seed=seed, device=device)
    try:
        policy = make_policy(envs.dims).to(device).eval()
        per_slot = -(-episodes // envs.E)
        counts, results, obs = np.zeros(envs.E, int), [], envs.obs
        while counts.min() < per_slot:
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = policy(obs)
                act = ActionDist(out, obs).sample()
            obs, _, _, infos, _ = envs.step(act.cpu().numpy())
            for info in infos:
                if counts[info["env"]] < per_slot:
                    counts[info["env"]] += 1
                    results.append(info)
    finally:
        envs.close()
    return summarize(results)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="'random' or a checkpoint path")
    p.add_argument("--like", help="for --policy random: take the game setup from this checkpoint")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--seed", default="eval")
    a = p.parse_args(argv)
    ckpt_path = a.like if a.policy == "random" else a.policy
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path else None
    train_args = argparse.Namespace(**ckpt["args"]) if ckpt else parse_args([])
    if a.policy == "random":
        make = RandomPolicy
    else:
        def make(dims):
            model = Policy(dims, tuple(int(w) for w in train_args.widths.split(",")))
            model.load_state_dict(ckpt["model"])
            return model.to(memory_format=torch.channels_last)
    result = evaluate(make, env_config(train_args), a.episodes, a.workers, a.games_per_worker, seed=a.seed)
    print(json.dumps({"policy": a.policy, "update": ckpt["update"] if ckpt and a.policy != "random" else None,
                      **result}, indent=2))


if __name__ == "__main__":
    main()
