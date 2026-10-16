"""Plays policies against each other on fixed seeds (the same games every time, so results compare directly).

  .venv/Scripts/python -m trainer.eval --a checkpoints/bc/latest.pt --b scripted      (ranked 1v1, 256 games)
  .venv/Scripts/python -m trainer.eval --a checkpoints/ppo/latest.pt --b checkpoints/bc/latest.pt
  .venv/Scripts/python -m trainer.eval --setup onion --a random                        (Phase 1: one seat vs 3 nations)
  .venv/Scripts/python -m trainer.eval --setup ffa --a checkpoints/ppo/latest.pt --maps seen   (4N: one seat vs nations)

A policy is "random", "scripted" or a checkpoint path. The results are the first policy's; in 2-seat setups the two
policies swap seats from one game slot to the next. The setups ffa and hvn (4N) play the public rules, no timer, on
--maps: FFA with policy A in one seat against the map's Hard nations, Humans vs Nations with it in all four (--b is not
used). Gate 4N's bars read decided_win_rate, the games won and not cut by the env's backstop (win_rate counts a lead
at the backstop as a win). Every result is printed and saved to runs/eval/<date>-<time>.json.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

from trainer.control import REPLAYS
from trainer.env import ONION, RANKED_1V1, VecEnv, game_commit, gate_config
from trainer.model import ATTACK, BOAT, BUILD, RETREAT, SPAWN, ActionDist, RandomPolicy, load_net

SETUPS = {"onion": ONION, "ranked1v1": RANKED_1V1}
GATES = ("ffa", "hvn")  # Gate 4N's games, from trainer.env.gate_config
RESULTS = Path(__file__).resolve().parents[1] / "runs" / "eval"  # every match's result, as runs/score/ for BC
TROOP_TYPES = [ATTACK, BOAT, RETREAT]  # the two kinds of action the bins would split (PLAN decision 11)
GOLD_TYPES = list(range(BUILD, RETREAT))  # the nine builds and upgrade


def summarize(results):
    n = len(results)
    reward = np.array([r["reward"] for r in results], float)
    win = np.array([r["rank"] == 1 for r in results], float)
    rank = np.array([r["rank"] for r in results], float)
    se = lambda v: float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    seat = np.array([r.get("seat", 0) for r in results])
    by_seat = {f"win_rate_seat{s}": float(win[seat == s].mean()) for s in np.unique(seat)} if len(set(seat)) > 1 else {}
    cut = np.array([r.get("cut", False) for r in results], float)  # games the env's backstop ended, decided by tiles
    decided = win * (1 - cut)  # Gate 4N's win: the public rules play a cut game on, so a lead at the backstop is none
    return {"episodes": n, "win_rate": float(win.mean()), "win_rate_se": se(win), **by_seat,
            "decided_win_rate": float(decided.mean()), "decided_win_rate_se": se(decided),
            "mean_reward": float(reward.mean()), "mean_reward_se": se(reward), "mean_rank": float(rank.mean()),
            "cut_share": float(cut.mean())}


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
    forced), then every `every` decisions attacks unowned land, or else a random attackable bot, sending `share`
    percent of its troops; otherwise waits."""

    def __init__(self, share=25, every=4, seed=0):
        self.share, self.every, self.rng, self.step = share, every, np.random.default_rng(seed), 0

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
                    act = [2, 0, int(self.rng.choice(targets)), self.share]
            rows.append(act)
        self.step += 1
        return torch.tensor(rows, dtype=torch.long, device=obs["type_mask"].device)


def load_policy(spec, dims, device="cuda", two_bins=False):
    """obs -> actions [n, 4] for "random", "scripted" or a checkpoint path (sampling, as in training); with `two_bins`,
    [n, 8] from two_bin_sample."""
    if spec == "scripted":
        return Scripted()
    model = RandomPolicy(dims).to(device) if spec == "random" else load_net(spec, dims, device)

    def act(obs):
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                out = model(obs)
            dist = ActionDist(out, obs)
            act.type_probs = dist.type.probs  # for Cooldown's meters
            return two_bin_sample(dist, obs) if two_bins else dist.sample()

    return act


def two_bin_sample(dist, obs):
    """The upward probe of the bandwidth test (PLAN 3B): per decision a troop action (attack, boat, retreat or no-op)
    and then a gold action (a build, a nuke, an upgrade or no-op), [n, 8]. Each kind keeps the policy's own
    probability, as bins trained by imitation would, so the bot acts as often as before and only acting in parallel
    is new. In the spawn phase the first action is an ordinary sample and the second a no-op."""
    p = dist.type.probs
    u = torch.rand(len(p), 2, device=p.device)
    first, second = dist.complete(pick(p, TROOP_TYPES, u[:, 0])), dist.complete(pick(p, GOLD_TYPES, u[:, 1]))
    single = torch.cat([dist.sample(), torch.zeros_like(first)], 1)
    return torch.where(obs["type_mask"][:, SPAWN, None], single, torch.cat([first, second], 1))


def pick(probs, kinds, u):
    """Per row, one of `kinds` with the policy's own probability of it (u uniform in [0, 1)), else no-op (0)."""
    kinds = [k for k in kinds if k < probs.shape[1]]
    i = (u[:, None] >= probs[:, kinds].cumsum(1)).sum(1)
    return torch.tensor(kinds + [0], device=probs.device)[i]


class Cooldown:
    """Policy A in the bandwidth test (PLAN 3B): after a decision that acts, the row's next `n` decisions become no-ops,
    outside the spawn phase (where spawn is legal). Over those decisions it counts the share that acts, the share with
    two actions (two_bin_sample), the share of actions that follow an action, the share of sampled actions the cooldown
    dropped, and, for networks, the share of decisions with >= 25% on both a troop and a gold action. With n = 0 it
    only counts."""

    def __init__(self, policy, n):
        self.policy, self.n = policy, n
        self.left = self.prev = None
        self.counts = dict.fromkeys(("decisions", "acted", "parallel", "follows", "wanted", "blocked", "both"), 0)

    def __call__(self, obs):
        a = self.policy(obs)
        live = ~obs["type_mask"][:, SPAWN]
        if self.left is None:
            self.left = torch.zeros(len(a), dtype=torch.long, device=a.device)
            self.prev = torch.zeros(len(a), dtype=torch.bool, device=a.device)
        wanted = live & (a[:, 0::4] != 0).any(1)  # every 4th entry is an action's type
        blocked = wanted & (self.left > 0)
        a[blocked] = 0
        actions = (a[:, 0::4] != 0).sum(1)
        acted = live & (actions > 0)
        probs = getattr(self.policy, "type_probs", None)
        both = torch.zeros_like(acted)
        if probs is not None:
            troop, gold = ([k for k in kinds if k < probs.shape[1]] for kinds in (TROOP_TYPES, GOLD_TYPES))
            both = (probs[:, troop].sum(1) >= 0.25) & (probs[:, gold].sum(1) >= 0.25)
        for key, rows in (("decisions", live), ("acted", acted), ("parallel", live & (actions > 1)),
                          ("follows", acted & self.prev), ("wanted", wanted), ("blocked", blocked),
                          ("both", both & live)):
            self.counts[key] += int(rows.sum())
        self.left = torch.where(acted, self.n, (self.left - 1).clamp(min=0))
        self.prev = acted
        return a

    def meters(self):
        c = self.counts
        share = lambda k, n: c[k] / c[n] if c[n] else 0.0
        m = {"decisions": c["decisions"], "act_share": share("acted", "decisions"),
             "parallel_share": share("parallel", "decisions"), "follow_share": share("follows", "acted"),
             "blocked_share": share("blocked", "wanted")}
        if getattr(self.policy, "type_probs", None) is not None:
            m["both_share"] = share("both", "decisions")
        return m


def seats_for(E, S):
    """seats[e][p]: the seat policy p plays in game slot e (the policies rotate through the seats)."""
    return [[(e + p) % S for p in range(S)] for e in range(E)]


def play(specs, config, episodes, workers, games_per_worker, device="cuda", seed="eval", record=None, a_cooldown=0,
         a_two_bins=False, wait=None):
    """Plays the first ceil(episodes / games) episodes of every game slot; returns the summary of policy 0, its meters
    (Cooldown, with `a_cooldown` no-ops after each of its actions; `a_two_bins`: a troop and a gold action per
    decision; and Gate 3B's invalid actions: of its decisions that act, the share the bridge refuses), and the IDs of
    the games saved with `record` ({"dir", "gitCommit", "games"}). Sampling uses a fixed torch seed, so a given set of
    policies always gives the same result.
    - A later policy may be a list: game slot e plays its entry e % len, and the result adds policy 0's summary against
      each entry ("by_b"; the trainer's panel, PLAN 4A).
    - `wait` is called before every step (the trainer's pause and stop): once it returns False the match ends early,
      returning None."""
    torch.manual_seed(0)
    envs = VecEnv(config, workers, games_per_worker, seed=seed, device=device, record=record)
    try:
        S, E = envs.S, envs.E
        if len(specs) != S:
            raise ValueError(f"{S} seats need {S} policies, got {len(specs)}")
        seats = seats_for(E, S)
        of_slot = [[s[e % len(s)] if isinstance(s, list) else s for e in range(E)] for s in specs]
        groups = []  # (policy, its rows), one per seat policy and spec; policy 0 is one spec
        for p in range(S):
            for spec in dict.fromkeys(of_slot[p]):
                policy = load_policy(spec, envs.dims, device, two_bins=a_two_bins and p == 0)
                rows = [e * S + seats[e][p] for e in range(E) if of_slot[p][e] == spec]
                groups.append((Cooldown(policy, a_cooldown) if p == 0 else policy, torch.tensor(rows, device=device)))
        meter, a_rows = groups[0]
        per_slot = -(-episodes // E)
        counts, results, seen, obs = np.zeros(E, int), [], [], envs.obs
        a_cpu, acted, refused = a_rows.cpu().numpy(), 0, 0
        while counts.min() < per_slot:
            if wait is not None and not wait():
                return None
            act = torch.zeros(envs.N, 8 if a_two_bins else 4, dtype=torch.long, device=device)
            for policy, rows in groups:
                out = policy({k: v[rows] for k, v in obs.items()})
                act[rows, :out.shape[1]] = out
            obs, _, _, infos, invalid = envs.step(act.cpu().numpy())
            acted += int((act[a_rows, 0::4] != 0).any(1).sum())  # every 4th entry is an action's type
            refused += int(invalid[a_cpu].sum())
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
    recorded = {i["recorded"]: i["env"] for i in seen if i.get("recorded")}
    if recorded:
        names = {s: label(s) for row in of_slot for s in row}
        for game_id, e in recorded.items():
            name_players(Path(record["dir"]) / f"{game_id}.json", {seats[e][p]: names[of_slot[p][e]] for p in range(S)})
    meters = {**meter.meters(), "actions": acted, "invalid_share": refused / acted if acted else 0.0}
    out = {**summarize(results), "meters": meters, "recorded": sorted(recorded)}
    if S > 1 and isinstance(specs[1], list):
        out["by_b"] = {b: summarize([r for r in results if of_slot[1][r["env"]] == b]) for b in dict.fromkeys(specs[1])}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="the scored policy: random, scripted or a checkpoint path")
    p.add_argument("--b", help="its opponent in 2-seat setups")
    p.add_argument("--setup", choices=[*sorted(SETUPS), *GATES], default="ranked1v1")
    p.add_argument("--maps", choices=("heldout", "seen", "training"),
                   help="the maps of --setup ffa and hvn (default heldout): Gate 4N's 4 held-out maps, 4 seen ones, or "
                        "training's whole list")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--seed", default="eval")
    p.add_argument("--record", type=int, default=0, help="games saved per worker into replays/<A> vs <B>/, to watch them")
    p.add_argument("--a-cooldown", type=int, default=0,
                   help="after each action of policy A, its next N decisions are no-ops (the bandwidth test, PLAN 3B)")
    p.add_argument("--a-two-bins", action="store_true",
                   help="policy A takes a troop and a gold action per decision (the bandwidth test's upward probe)")
    a = p.parse_args(argv)
    if a.setup in GATES:
        a.maps = a.maps or "heldout"
        config = gate_config(a.setup, a.maps)
    elif a.maps:
        p.error("--maps applies to --setup ffa and hvn")
    else:
        config = SETUPS[a.setup]
    if config["seats"] == 2 and a.b is None:
        p.error("--b is required in 2-seat setups")
    specs = [a.a] * config["seats"] if a.setup in GATES else [a.a, a.b][:config["seats"]]
    folder = replay_dir(specs[:1] if a.setup in GATES else specs)  # one policy in every seat: one name, one load
    record = {"dir": str(folder), "gitCommit": game_commit(), "games": a.record} if a.record > 0 else None
    result = play(specs, config, a.episodes, a.workers, a.games_per_worker, seed=a.seed, record=record,
                  a_cooldown=a.a_cooldown, a_two_bins=a.a_two_bins)
    text = json.dumps({"a": a.a, "b": a.b, "setup": a.setup, "maps": a.maps, "seed": a.seed, "a_cooldown": a.a_cooldown,
                       "a_two_bins": a.a_two_bins, **result}, indent=2)
    out = RESULTS / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")  # first, so a closed terminal cannot lose it
    print(text)
    print(f"saved {out}")
    if result["recorded"]:
        print(f'watch: python -m trainer.watch --dir "{folder.name}", then open')
        for game_id in result["recorded"]:
            print(f"  http://localhost:9000/game/{game_id}")


if __name__ == "__main__":
    main()
