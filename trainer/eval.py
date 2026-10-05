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
from pathlib import Path

import numpy as np
import torch

from trainer.control import REPLAYS
from trainer.env import ONION, RANKED_1V1, VecEnv
from trainer.model import ATTACK, BOAT, BUILD, SPAWN, ActionDist, RandomPolicy, load_net
from trainer.ppo import game_commit

SETUPS = {"onion": ONION, "ranked1v1": RANKED_1V1}
TROOP_TYPES, GOLD_TYPES = [ATTACK, BOAT], [BUILD]  # the two kinds of action the bins would split (PLAN decision 11)


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


def label(spec):
    """A policy's name for the players it plays in recorded games (letters, digits and spaces: the game's username
    rules)."""
    if spec in ("random", "scripted"):
        return spec.capitalize()
    ckpt = torch.load(spec, map_location="cpu", weights_only=False)
    return f"{'PPO' if 'kl_coef' in ckpt['args'] else 'Clone'} {ckpt['update']}"


def tag(spec):
    """A policy's short name in folder names: a checkpoint's run and update (runs reuse update numbers)."""
    if spec in ("random", "scripted"):
        return spec
    ckpt = torch.load(spec, map_location="cpu", weights_only=False)
    return f"{ckpt['args'].get('run', Path(spec).parent.name)} {ckpt['update']}"


def replay_dir(specs):
    """Where a match's recorded games go: one folder per matchup, since fixed seeds give every match the same game
    IDs."""
    return REPLAYS / " vs ".join(tag(s) for s in specs)


def name_players(path, by_seat):
    """Renames a recorded game's players by seat (clientID AGENT000 is seat 0). Names do not enter the game's hashes,
    so the record still replays in sync."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    for p in rec["info"]["players"]:
        p["username"] = by_seat[int(p["clientID"].removeprefix("AGENT"))]
    Path(path).write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")


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
            dist = ActionDist(out, obs)
            act.type_probs = dist.type.probs  # for Cooldown's meters
            return dist.sample()

    return act


class Cooldown:
    """Policy A in the bandwidth test (PLAN 3B): after a decision that acts, the row's next `n` decisions become no-ops,
    outside the spawn phase (where spawn is legal). Over those decisions it counts the share that acts, the share of
    actions that follow an action, the share of sampled actions the cooldown dropped, and, for networks, the share of
    decisions with >= 25% on both a troop and a gold action. With n = 0 it only counts."""

    def __init__(self, policy, n):
        self.policy, self.n = policy, n
        self.left = self.prev = None
        self.counts = dict.fromkeys(("decisions", "acted", "follows", "wanted", "blocked", "both"), 0)

    def __call__(self, obs):
        a = self.policy(obs)
        live = ~obs["type_mask"][:, SPAWN]
        if self.left is None:
            self.left = torch.zeros(len(a), dtype=torch.long, device=a.device)
            self.prev = torch.zeros(len(a), dtype=torch.bool, device=a.device)
        wanted = live & (a[:, 0] != 0)
        blocked = wanted & (self.left > 0)
        a[blocked] = 0
        acted = live & (a[:, 0] != 0)
        probs = getattr(self.policy, "type_probs", None)
        both = torch.zeros_like(acted)
        if probs is not None:
            both = (probs[:, TROOP_TYPES].sum(1) >= 0.25) & (probs[:, GOLD_TYPES].sum(1) >= 0.25)
        for key, rows in (("decisions", live), ("acted", acted), ("follows", acted & self.prev), ("wanted", wanted),
                          ("blocked", blocked), ("both", both & live)):
            self.counts[key] += int(rows.sum())
        self.left = torch.where(acted, self.n, (self.left - 1).clamp(min=0))
        self.prev = acted
        return a

    def meters(self):
        c = self.counts
        share = lambda k, n: c[k] / c[n] if c[n] else 0.0
        m = {"decisions": c["decisions"], "act_share": share("acted", "decisions"),
             "follow_share": share("follows", "acted"), "blocked_share": share("blocked", "wanted")}
        if getattr(self.policy, "type_probs", None) is not None:
            m["both_share"] = share("both", "decisions")
        return m


def seats_for(E, S):
    """seats[e][p]: the seat policy p plays in game slot e (the policies rotate through the seats)."""
    return [[(e + p) % S for p in range(S)] for e in range(E)]


def play(specs, config, episodes, workers, games_per_worker, device="cuda", seed="eval", record=None, a_cooldown=0):
    """Plays the first ceil(episodes / games) episodes of every game slot; returns the summary of policy 0, its meters
    (Cooldown, with `a_cooldown` no-ops after each of its actions), and the IDs of the games saved with `record`
    ({"dir", "gitCommit", "games"}). Sampling uses a fixed torch seed, so a given set of policies always gives the same
    result."""
    torch.manual_seed(0)
    envs = VecEnv(config, workers, games_per_worker, seed=seed, device=device, record=record)
    try:
        S, E = envs.S, envs.E
        if len(specs) != S:
            raise ValueError(f"{S} seats need {S} policies, got {len(specs)}")
        policies = [load_policy(s, envs.dims, device) for s in specs]
        policies[0] = Cooldown(policies[0], a_cooldown)
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
    recorded = {i["recorded"]: seats[i["env"]] for i in seen if i.get("recorded")}
    if recorded:
        names = [label(s) for s in specs]
        for game_id, slot_seats in recorded.items():
            name_players(Path(record["dir"]) / f"{game_id}.json", {s: names[p] for p, s in enumerate(slot_seats)})
    return {**summarize(results), "meters": policies[0].meters(), "recorded": sorted(recorded)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="the scored policy: random, scripted or a checkpoint path")
    p.add_argument("--b", help="its opponent in 2-seat setups")
    p.add_argument("--setup", choices=sorted(SETUPS), default="ranked1v1")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--seed", default="eval")
    p.add_argument("--record", type=int, default=0, help="games saved per worker into replays/<A> vs <B>/, to watch them")
    p.add_argument("--a-cooldown", type=int, default=0,
                   help="after each action of policy A, its next N decisions are no-ops (the bandwidth test, PLAN 3B)")
    a = p.parse_args(argv)
    config = SETUPS[a.setup]
    if config["seats"] == 2 and a.b is None:
        p.error("--b is required in 2-seat setups")
    specs = [a.a, a.b][:config["seats"]]
    folder = replay_dir(specs)
    record = {"dir": str(folder), "gitCommit": game_commit(), "games": a.record} if a.record > 0 else None
    result = play(specs, config, a.episodes, a.workers, a.games_per_worker, seed=a.seed, record=record,
                  a_cooldown=a.a_cooldown)
    print(json.dumps({"a": a.a, "b": a.b, "setup": a.setup, "a_cooldown": a.a_cooldown, **result}, indent=2))
    if result["recorded"]:
        print(f'watch: python -m trainer.watch --dir "{folder.name}", then open')
        for game_id in result["recorded"]:
            print(f"  http://localhost:9000/game/{game_id}")


if __name__ == "__main__":
    main()
