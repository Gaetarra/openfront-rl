"""PPO training (CleanRL-style) with pause/save/resume, for a PC that is also used daily.

  Train:   .venv/Scripts/python -m trainer.ppo --run onion
  Ranked:  .venv/Scripts/python -m trainer.ppo --run ppo-bc --setup ranked1v1 --init checkpoints/bc/latest.pt
               --opponent checkpoints/bc/latest.pt --kl-coef 0.05
  Resume:  .venv/Scripts/python -m trainer.ppo --run onion --resume
  Control: http://127.0.0.1:8787 or python -m trainer.ctl pause|resume|save|stop|throttle|status
  Charts:  .venv/Scripts/tensorboard --logdir runs
"""
import argparse
import collections
import glob
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trainer.control import Control, serve
from trainer.env import PLAYLISTS, RANKED_1V1, ROOT, VecEnv, frozen_eval_config, game_commit
from trainer.eval import play
from trainer.league import ROTATE, League, name, panel_table, ratings
from trainer.model import ACT_NAMES, ATTENDED, CORES, ActionDist, arch_of, load_net, new_net

SESSION = ("port", "duty", "gpu_temp_limit")  # settings of this sitting, not of the run: a resume takes them as given
MAY_CHANGE = ("updates", "kl_coef", "eval_games", "eval_every", "eval_games_public")  # a resume may give new values
WHOLE = ("ffa", "hvn")  # setups where every seat is the learner's: the opponents are the game's own nations (4N)
REF_OUTPUTS = ("type", "maps", "ptr", "retreat", "shares", "bins")  # what ActionDist reads of the KL reference
SNAPSHOT = 250  # updates between kept checkpoints
AVERAGED = 5  # kept checkpoints in each weight average, --init counting as update 0: the last 1,000 updates. Averages
# join the league's pool and play the panel: PPO's snapshots scatter around a stronger mean policy (lr4a's five scored
# 0.778 against the 3B clone, their average 0.945; PLAN 4A, "Checks before the next league")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="onion", help="name for checkpoints/<run> and runs/<run>")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/latest.pt")
    p.add_argument("--map", default="Onion")
    p.add_argument("--normal-size", action="store_true", help="Normal map size instead of Compact")
    p.add_argument("--nations", type=int, default=3)
    p.add_argument("--difficulty", default="Easy")
    p.add_argument("--bots", type=int, default=0)
    p.add_argument("--timer", type=int, default=5, help="game timer in minutes")
    p.add_argument("--setup", choices=("onion", "ranked1v1"), default="onion",
                   help="ranked1v1: the official ranked 1v1 playlist (the map, size, nation, bot and timer flags are ignored)")
    p.add_argument("--mix", nargs="+", metavar="SETUP=WORKERS",
                   help="instead of --setup, workers on several of the public playlists (ranked1v1, ffa, hvn), in the "
                        "order given, e.g. ranked1v1=8 ffa=4 hvn=2: one learner seat in each ranked game, every seat in "
                        "an FFA or Humans vs Nations game. --workers becomes their sum (4N)")
    p.add_argument("--seats", type=int, default=1, help="agent seats in the onion setup")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--steps", type=int, default=None,
                   help="decisions per game per update (default: 48 for ranked1v1 and --mix, which keeps the run under "
                        "8 GiB of VRAM; 128 otherwise)")
    p.add_argument("--updates", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatches", type=int, default=None, help="default: 8 for ranked1v1 and --mix, 4 otherwise")
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--widths", default="64,128,192", help="CNN stage widths")
    p.add_argument("--core", choices=CORES, default="moe", help="the core of a new network (not with --init)")
    p.add_argument("--init", help="start from this checkpoint's weights, e.g. checkpoints/bc/latest.pt")
    p.add_argument("--opponent", help="frozen checkpoint that plays every seat but the learner's (2-seat setups)")
    p.add_argument("--league", nargs="+",
                   help="instead of --opponent, a pool of frozen checkpoints (globs expand): PFSP picks the opponent of "
                        "new games every 25 updates, and the learner's weight averages join it every 250 (PLAN 4A)")
    p.add_argument("--panel", nargs="+",
                   help="frozen checkpoints that every 250th update's weight average (and --init) plays on fixed games "
                        "while training waits; the first anchors the Bradley-Terry rating (PLAN 4A)")
    p.add_argument("--eval-games", type=int, default=128,
                   help="games per panel evaluation, split over the panel by game slot (given with --resume, it replaces "
                        "the run's)")
    p.add_argument("--eval-every", type=int, default=SNAPSHOT,
                   help=f"updates between frozen evaluations, a multiple of {SNAPSHOT} (given with --resume, it replaces "
                        "the run's)")
    p.add_argument("--eval-games-public", type=int, default=32,
                   help="games per frozen evaluation against the nations, for each of ffa and hvn in a --mix (given with "
                        "--resume, it replaces the run's)")
    p.add_argument("--kl-coef", type=float, default=None,
                   help="KL penalty toward the --init policy (default 0; given with --resume, it replaces the run's)")
    p.add_argument("--kl-ref", help="the KL penalty's reference instead of --init")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=25, help="updates between checkpoints")
    p.add_argument("--port", type=int, default=8787, help="control page port (0 = any free port)")
    p.add_argument("--duty", type=float, default=1.0, help="start throttled: fraction of time spent working")
    p.add_argument("--gpu-temp-limit", type=int, default=0, help="auto-pause at this GPU temperature in C (0 = off)")
    return p


def mix_of(a):
    """--mix as {setup: workers}, in the order given."""
    mix = {}
    for item in a.mix:
        setup, _, count = item.partition("=")
        if setup not in PLAYLISTS or setup in mix or not count.isdigit() or int(count) < 1:
            raise SystemExit(f"--mix takes SETUP=WORKERS, each setup once, one of {', '.join(PLAYLISTS)} and at least "
                             f"1 worker: {item!r}")
        mix[setup] = int(count)
    return mix


def nations_of(a):
    """The --mix setups whose games every frozen evaluation also plays against the game's own nations (4N)."""
    return [s for s in (mix_of(a) if a.mix else ()) if s in WHOLE]


def parse_args(argv=None):
    a = parser().parse_args(argv)
    if a.mix:
        a.workers = sum(mix_of(a).values())  # the mix decides the workers
    return a


def given_flags(p, argv):
    """The flags given on the command line, with their parsed values (an explicit value equal to the default counts)."""
    unset = object()
    ns = argparse.Namespace(**{k: unset for k in vars(p.parse_args([]))})
    return {k: v for k, v in vars(p.parse_args(argv, namespace=ns)).items() if v is not unset}


def resume_args(p, argv, saved, may_change):
    """A resumed run's settings: the checkpoint's (`saved`), this sitting's SESSION flags and the given `may_change`
    flags. Any other flag would be ignored, so one given with a value other than the run's is refused."""
    given, a = given_flags(p, argv), p.parse_args(argv)
    refused = [k for k, v in given.items() if k not in (*SESSION, *may_change, "run", "resume") and saved.get(k) != v]
    if refused:
        flags = ", ".join("--" + k.replace("_", "-") for k in refused)
        raise SystemExit(f"--resume keeps the run's settings, so {flags} would be ignored. Drop them, or start a new "
                         f"run from its weights: --run <new name> --init checkpoints/{a.run}/latest.pt")
    return argparse.Namespace(**{**vars(p.parse_args([])), **saved, **{k: getattr(a, k) for k in SESSION},
                                 **{k: given[k] for k in may_change if k in given}, "resume": True})


def network_args(p, argv, a):
    """A new run's network settings. With --init they are the checkpoint's: its weights decide the network, and a
    different --core or --widths given alongside is refused. Without it, a Phase 3B network (100 troop shares, 15
    actions) from --core and --widths. A new run never overwrites another: a folder that holds a run's checkpoint is
    refused."""
    if a.init:
        spec = arch_of(torch.load(a.init, map_location="cpu", weights_only=False)["args"])
        given = given_flags(p, argv)
        clash = [k for k in ("core", "widths") if k in given and given[k] != spec[k]]
        if clash:
            raise SystemExit(f"--init {a.init} holds a {spec['arch']} network ({spec['core']} core, widths "
                             f"{spec['widths']}): drop {', '.join('--' + k for k in clash)}")
        vars(a).update(spec)
    else:
        a.arch = "v4"
    if (ROOT / "checkpoints" / a.run / "latest.pt").exists():
        raise SystemExit(f"checkpoints/{a.run} already holds a run: add --resume to continue it, or start a new one "
                         f"with another --run name (or delete the folder)")
    return a


def env_config(a):
    """The games' config: one for every worker, or with --mix a list with one per worker."""
    if a.mix:
        return [PLAYLISTS[s] for s, n in mix_of(a).items() for _ in range(n)]
    if a.setup == "ranked1v1":
        return RANKED_1V1
    return {"map": a.map, "compact": not a.normal_size, "nations": a.nations, "difficulty": a.difficulty,
            "bots": a.bots, "timerMinutes": a.timer, "seats": a.seats}


def run_info(a):
    """What the control page says about the run."""
    frozen = f"a league of frozen policies: PFSP picks every {ROTATE} updates, averages join every {SNAPSHOT}" \
        if a.league else f"frozen {a.opponent}"
    if a.mix:
        mix = mix_of(a)
        game = "a mix of setups: " + ", ".join(f"{s} ({n * a.games_per_worker} games)" for s, n in mix.items())
        rivals = "; ".join(f"{s}: " + ("the game's own nations" if s in WHOLE else frozen) for s in mix)
        plays = {"Learner seat": "; ".join(
            f"{s}: " + (f"all {PLAYLISTS[s]['seats']}" if s in WHOLE else f"one of {PLAYLISTS[s]['seats']}, in turn")
            for s in mix)}
    else:
        if a.setup == "ranked1v1":
            game, rivals = "ranked 1v1 (the official playlist)", frozen
        else:
            game = f"{a.map}, {'Normal' if a.normal_size else 'Compact'}, {a.timer} min"
            rivals = f"{a.nations} {a.difficulty} nations" + (f" and {a.bots} bots" if a.bots else "")
            if a.seats > 1:
                rivals += f"; frozen {a.opponent} in the other seats"
        seats = 2 if a.setup == "ranked1v1" else a.seats
        plays = {"Learner seat": "both, half the games each" if seats == 2 else f"all {seats}, equally often"} \
            if seats > 1 else {}
    ref = {"KL reference": a.kl_ref} if a.kl_ref else {}
    nations = nations_of(a)
    panel = {"Frozen evaluations": f"{a.eval_games} games against {len(a.panel)} policies every {a.eval_every} updates"
             + (f", and {a.eval_games_public} games of each of {', '.join(nations)} against the nations"
                if nations else "")} if a.panel else {}
    return {"Name": a.run, "Method": "PPO (reinforcement learning)", "Game": game, "Against": rivals, **plays,
            "Starts from": a.init or "random weights", "KL penalty to the start": a.kl_coef or "none", **ref, **panel,
            "Games at once": f"{a.workers * a.games_per_worker} ({a.workers} workers x {a.games_per_worker})",
            "Updates": f"{a.updates:,}"}


def seat_rows(seats, whole, device="cpu"):
    """Who plays which row, for games of `seats[e]` rows each, game after game (VecEnv's layout).
    - A game with whole[e] (FFA, Humans vs Nations: the opponents are the game's own nations) gives every row to the
      learner.
    - In the others the learner plays seat k % S of the k-th such game, so it plays each seat equally often, as in
      trainer.eval (the game is symmetric, so the per-seat win rates should match); the frozen opponents play the rest.
    Returns the learner's rows [L], the opponents' (a tensor [K] per seat but the learner's), the K frozen-opponent
    games (the league's slots), and the (game, seat) of each learner row."""
    seats = [int(n) for n in seats]
    first = np.cumsum(seats) - seats
    slots, owner = [], []
    for e, n in enumerate(seats):
        if whole[e]:
            owner += [(e, s) for s in range(n)]
        else:
            owner.append((e, len(slots) % n))
            slots.append(e)
    rows = lambda pairs: torch.tensor([int(first[e]) + s for e, s in pairs], dtype=torch.long, device=device)
    S = seats[slots[0]] if slots else 1  # ponytail: the K games share one seat count (ranked's 2), else a list per game
    return (rows(owner), [rows([(e, (k + s) % seats[e]) for k, e in enumerate(slots)]) for s in range(1, S)], slots,
            owner)


def cap_binds(players):
    """Where the transformer's ATTENDED slots (trainer/model.py) fall short: the seat's real players (those that exist
    and are self, friendly or enemy; tribes do not count) and row 0, the unowned land, outnumber them. players: [n, P,
    F], bridge/obs.ts's features (0 exists, 2 self, 3 friendly, 4 enemy)."""
    real = (players[:, 1:, 0] > 0) & (players[:, 1:, 2:5] > 0).any(-1)
    return real.sum(1) + 1 > ATTENDED


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


def average(paths):
    """The uniform average of checkpoints' weights, loaded one at a time (a PPO checkpoint holds ~200 MB)."""
    total = None
    for p in paths:
        model = torch.load(p, map_location="cpu", weights_only=False)["model"]
        total = {k: v.double() + (total[k] if total else 0) for k, v in model.items()}
    return {k: (v / len(paths)).to(model[k].dtype) for k, v in total.items()}


class Outcomes:
    """Rolling episode results, and how well the value head predicted each finished game's reward, by game phase:
    sign accuracy (trivially high while one outcome dominates, e.g. an agent that rarely wins) and the correlation
    between value and final reward (the gate's measure; rank-based rewards keep it defined without wins)."""

    def __init__(self, rows, window=200):
        self.window = window
        self.episodes = collections.deque(maxlen=window)
        self.by_setup = {}  # setup -> its own window of episodes (4N: a mixed run's setups finish games at other rates)
        self.pending = [[] for _ in range(rows)]
        self.samples = collections.deque(maxlen=50_000)  # (phase bucket, value, final reward)

    def record(self, progress, values):
        for row, (p, v) in enumerate(zip(progress, values)):
            self.pending[row].append((p, v))

    def finish(self, infos):
        """The learner's finished games: each info carries the learner's `row` and its game's `setup`."""
        for info in infos:
            row = info["row"]
            self.samples.extend((min(int(p * 3), 2), v, info["reward"]) for p, v in self.pending[row])
            self.pending[row] = []
            self.episodes.append(info)
            self.by_setup.setdefault(info["setup"], collections.deque(maxlen=self.window)).append(info)

    def state(self):
        """The finished games and value samples, kept in checkpoints so the curves carry on after a resume."""
        return {"episodes": list(self.episodes), "samples": list(self.samples),
                "by_setup": {s: list(eps) for s, eps in self.by_setup.items()}}

    def load(self, state):
        self.episodes.extend(state["episodes"])
        self.samples.extend(state["samples"])
        for s, eps in state.get("by_setup", {}).items():  # older checkpoints lack it
            self.by_setup.setdefault(s, collections.deque(maxlen=self.window)).extend(eps)

    def summary(self):
        if not self.episodes:
            return {}
        ep = list(self.episodes)
        out = {"episodes/win_rate": float(np.mean([e["rank"] == 1 for e in ep])),
               "episodes/mean_reward": float(np.mean([e["reward"] for e in ep])),
               "episodes/mean_rank": float(np.mean([e["rank"] for e in ep])),
               "episodes/ticks": float(np.mean([e["ticks"] for e in ep]))}
        if len(self.by_setup) > 1:  # a mixed run: each setup's win rate, reward and game length
            for s, eps in self.by_setup.items():
                out[f"setups/win_rate/{s}"] = float(np.mean([e["rank"] == 1 for e in eps]))
                out[f"setups/mean_reward/{s}"] = float(np.mean([e["reward"] for e in eps]))
                out[f"setups/ticks/{s}"] = float(np.mean([e["ticks"] for e in eps]))
        seats = sorted({e["seat"] for e in ep})
        if len(seats) > 1 and len(self.by_setup) < 2:  # the learner plays every seat: does it win from each?
            for seat in seats:
                out[f"episodes/win_rate_seat{seat}"] = float(np.mean([e["rank"] == 1 for e in ep if e["seat"] == seat]))
        s = np.array(self.samples, float).reshape(-1, 3)
        for bucket, name in enumerate(("early", "mid", "late")):
            v, r = s[s[:, 0] == bucket, 1], s[s[:, 0] == bucket, 2]
            if len(v) > 0:
                out[f"value/outcome_acc_{name}"] = float(np.mean((v > 0) == (r > 0)))
            if len(v) > 1 and v.std() > 0 and r.std() > 0:
                out[f"value/outcome_corr_{name}"] = float(np.corrcoef(v, r)[0, 1])
        return out


def pack_bits(m):
    """A bool tensor [..., n] (n a multiple of 8) as uint8 [..., n / 8]. The rollout keeps its cell masks this way: 12
    heat maps of 49,152 cells take 74 KB a row instead of 590 KB."""
    bits = torch.arange(8, device=m.device, dtype=torch.uint8)
    return (m.unflatten(-1, (-1, 8)).to(torch.uint8) << bits).sum(-1, dtype=torch.uint8)


def unpack_bits(b):
    """pack_bits undone."""
    bits = torch.arange(8, device=b.device, dtype=torch.uint8)
    return ((b.unsqueeze(-1) >> bits) & 1).bool().flatten(-2)


class HostRows:
    """The rollout's rows in pinned system RAM, row t * N + n: on the ranked grid their observations take ~4.3 GB that
    the card needs for learning (3B step 3). While the GPU learns from one minibatch, a worker thread gathers the next
    into one of two pinned slots and copies it to the GPU on a stream of its own."""

    def __init__(self, example, rows, size):
        """example: one step's rows {key: [N, ...]}; rows: T * N; size: a minibatch's rows."""
        self.n, self.size = len(next(iter(example.values()))), size
        self.buf = {k: torch.empty((rows,) + v.shape[1:], dtype=v.dtype, pin_memory=True) for k, v in example.items()}
        self.slots = [{k: torch.empty((size,) + v.shape[1:], dtype=v.dtype, pin_memory=True)
                       for k, v in example.items()} for _ in range(2)]
        self.side = torch.cuda.Stream()  # one for good: PyTorch caches freed GPU memory per stream
        self.pool = ThreadPoolExecutor(1)  # one for good too: a thread's first parallel op starts an OpenMP team that
        # outlives the thread, and a new thread per update leaked ~21 GiB over the first league run (PLAN 4A)

    def put(self, step, o):
        """One step's rows (GPU tensors), copied in without waiting."""
        for k, v in o.items():
            self.buf[k][step * self.n:(step + 1) * self.n].copy_(v, non_blocking=True)

    def batches(self, idx, device):
        """The minibatches with the rows idx[i] (CPU index tensors of at most `size` rows), as GPU tensors. One call
        takes every epoch's minibatches, so each epoch's first one is fetched ahead too."""
        main, side = torch.cuda.current_stream(), self.side
        copied = [torch.cuda.Event(), torch.cuda.Event()]  # the slot's last minibatch is on the GPU
        main.synchronize()  # the rollout's copies have landed

        def fetch(i):
            slot, m = self.slots[i % 2], len(idx[i])
            copied[i % 2].synchronize()
            for k, v in self.buf.items():
                torch.index_select(v, 0, idx[i], out=slot[k][:m])
            with torch.cuda.stream(side):
                o = {k: v[:m].to(device, non_blocking=True) for k, v in slot.items()}
            copied[i % 2].record(side)
            return o

        ahead = self.pool.submit(fetch, 0)
        for i in range(len(idx)):
            o = ahead.result()
            main.wait_stream(side)
            for v in o.values():
                v.record_stream(main)  # made on the side stream: keep it until the learning stream is done
            if i + 1 < len(idx):
                ahead = self.pool.submit(fetch, i + 1)
            yield o


def main(argv=None):
    a = parse_args(argv)
    ckpt_dir = ROOT / "checkpoints" / a.run
    state = None
    if a.resume:
        state = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        a = resume_args(parser(), argv, state["args"], MAY_CHANGE)
    else:
        a = network_args(parser(), argv, a)
    a.kl_coef = a.kl_coef or 0.0
    ranked = a.setup == "ranked1v1" or bool(a.mix)  # a mix's games share the ranked grid
    a.steps = a.steps or (48 if ranked else 128)
    a.minibatches = a.minibatches or (8 if ranked else 4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run_dir = ROOT / "runs" / a.run
    pool = [Path(p).as_posix() for g in a.league for p in sorted(glob.glob(g)) or [g]] if a.league else [a.opponent]
    missing = [p for p in [*pool, *(a.panel or []), a.kl_ref] if p and not Path(p).exists()]
    if missing:
        raise SystemExit(f"no such checkpoint: {', '.join(missing)}")
    if (a.league or a.panel) and SNAPSHOT % a.save_every:
        raise SystemExit(f"--save-every must divide {SNAPSHOT}: the league plays the checkpoints kept every {SNAPSHOT}")
    if a.eval_every < SNAPSHOT or a.eval_every % SNAPSHOT:
        raise SystemExit(f"--eval-every must be a multiple of {SNAPSHOT}: the weight averages that play are kept then")
    panel_log = run_dir / "panel.jsonl"  # the frozen evaluations, one per line
    if a.panel and not state and panel_log.exists():
        raise SystemExit(f"{panel_log} holds another run's evaluations: delete runs/{a.run} or pick another --run name")

    envs = VecEnv(env_config(a), a.workers, a.games_per_worker, seed=f"{a.run}-{a.seed}", device=device,
                  episodes=state["episodes"] if state else None, log_dir=run_dir / "workers")
    model = new_net(envs.dims, vars(a)).to(device)
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
    setups = [s for s, n in mix_of(a).items() for _ in range(n * a.games_per_worker)] if a.mix \
        else [a.setup] * envs.E  # the setup of each game
    learner, others, slots, owner = seat_rows(envs.seats, [s in WHOLE for s in setups], device)
    if others and not (a.opponent or a.league):
        raise SystemExit("2-seat setups need --opponent or --league")
    if a.league and not others:
        raise SystemExit("--league needs games with a frozen opponent: ranked 1v1, or 2 seats")
    if a.kl_coef > 0 and not (a.kl_ref or a.init):
        raise SystemExit("--kl-coef needs --init or --kl-ref")
    if a.init and not state:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
    frozen = {}

    def net(path):
        if path not in frozen:
            frozen[path] = load_net(path, envs.dims, device)
        return frozen[path]

    ref_path = a.kl_ref or a.init
    ref = net(ref_path) if a.kl_coef > 0 else None
    league = League(pool, len(slots)) if others else None  # a slot is a game with a frozen opponent
    if league and state and state.get("league"):
        league.load(state["league"])
    others_cpu = [r.cpu() for r in others]

    def opponents():
        """(network, rows) of every opponent playing now. A network no game uses leaves the GPU: the pool grows."""
        for p in [p for p in frozen if p not in league.slot and p != ref_path]:
            del frozen[p]
        return [(net(p), torch.cat([r[[e for e, q in enumerate(league.slot) if q == p]] for r in others_cpu]).to(device))
                for p in dict.fromkeys(league.slot)]

    teams = opponents() if league else []
    learner_np = learner.cpu().numpy()  # for the per-row arrays the workers return
    row_of = {game_seat: r for r, game_seat in enumerate(owner)}  # (game, seat) -> the learner's row
    slot_of = {e: k for k, e in enumerate(slots)}
    names = list(dict.fromkeys(setups))
    nations = nations_of(a)
    setup_row = torch.tensor([names.index(setups[e]) for e, _ in owner], device=device)  # each learner row's setup
    rows_of = np.bincount(setup_row.cpu().numpy(), minlength=len(names))  # learner rows per setup
    rows = lambda o, r: {k: v[r] for k, v in o.items()}
    control = Control(duty=a.duty, gpu_temp_limit=a.gpu_temp_limit)
    control.update(run=run_info(a))
    server = serve(control, a.port, log_dir=run_dir)
    control.note(f"control page on http://127.0.0.1:{server.server_address[1]}")
    writer = SummaryWriter(run_dir, purge_step=global_step + 1 if state else None)  # a resume hides a crash's tail
    commit = game_commit()

    def checkpoint():
        # Games in progress are dropped on quit; a resume starts each slot at its next episode, so no seed replays.
        kept = update % SNAPSHOT == 0 and update > 0
        averaged = ckpt_dir / f"avg{update:06d}.pt"
        if kept and a.league:
            league.add(averaged.as_posix())  # saved below
        s = dict(model=model.state_dict(), optimizer=opt.state_dict(), args=vars(a), update=update,
                 global_step=global_step, episodes=[e + 1 for e in envs.episodes], game_commit=commit,
                 outcomes=outcomes.state(), league=league.state() if league else None,
                 rng={"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all()})
        save_checkpoint(ckpt_dir / "latest.pt", **s)
        if kept:
            save_checkpoint(ckpt_dir / f"update{update:06d}.pt", **s)
            members = [ckpt_dir / f"update{k:06d}.pt" if k else Path(a.init)
                       for k in range(update, -1, -SNAPSHOT)[:AVERAGED] if k or a.init]
            save_checkpoint(averaged, model=average(members), args=vars(a), update=update, game_commit=commit,
                            averaged=[m.as_posix() for m in members])
        control.note(f"saved checkpoint at update {update}")

    evals = [json.loads(line) for line in panel_log.read_text(encoding="utf-8").splitlines()] \
        if a.panel and panel_log.exists() else []
    if evals:
        control.update(panel=panel_table(evals, a.panel, *ratings(evals, a.panel)))

    def evaluate():
        """This update's weight average (update 0: --init) plays the panel on fixed games while training waits (PLAN
        4A), and in a mixed run also the FFA and Humans vs Nations games against the nations (4N). False if a stop came
        first."""
        control.set_state("evaluating")
        control.note(f"frozen evaluation of update {update} against the panel"
                     + (f" and the nations ({', '.join(nations)})" if nations else ""))
        spec = (ckpt_dir / f"avg{update:06d}.pt").as_posix() if update else a.init
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        # the panel is a ranked 1v1 match, mixed run or not; then each nations setup, with the policy in all its seats
        games = [([spec, a.panel], PLAYLISTS["ranked1v1"] if a.mix else env_config(a), a.eval_games)]
        games += [([spec] * c["seats"], c, a.eval_games_public) for c in map(frozen_eval_config, nations)]
        played = []
        with torch.random.fork_rng(devices=[device]):  # play() fixes its own seed; training's carries on after
            for specs, config, n in games:
                r = play(specs, config, n, a.workers, a.games_per_worker, wait=control.wait)
                if r is None:
                    return False
                played.append(r)
        r, *vs_nations = played
        evals.append({"update": update, "global_step": global_step, "minutes": (time.perf_counter() - t0) / 60,
                      "vs": r["by_b"], "nations": dict(zip(nations, vs_nations))})
        with open(panel_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(evals[-1]) + "\n")
        fit, trend = ratings(evals, a.panel)
        for m, v in r["by_b"].items():
            writer.add_scalar(f"eval/vs_{name(m)}", v["win_rate"], global_step)
        for setup, v in evals[-1]["nations"].items():
            writer.add_scalar(f"eval/nations_{setup}", v["win_rate"], global_step)
        writer.add_scalar("eval/rating", fit[update][0], global_step)
        writer.add_scalar("eval/minutes", evals[-1]["minutes"], global_step)
        control.update(panel=panel_table(evals, a.panel, fit, trend))
        control.set_state("running")
        return True

    T, L = a.steps, len(learner)
    obs = envs.obs
    def stored(o):
        """A seat's rows as the rollout keeps them: the cell masks as bits and, with a KL penalty, the reference's
        outputs, computed once here instead of in every epoch of learning (PLAN 4A, "Speed work")."""
        kept = {**o, "cell_mask": pack_bits(o["cell_mask"])}
        if ref is not None:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                kept.update({"ref_" + k: v for k, v in ref(o).items() if k in REF_OUTPUTS})
        return kept

    host = HostRows(stored(rows(obs, learner)), T * L, T * L // a.minibatches)
    actions = torch.zeros(T, L, 4, dtype=torch.long, device=device)
    logps, values, rewards, dones = (torch.zeros(T, L, device=device) for _ in range(4))
    outcomes = Outcomes(L)
    if state and "outcomes" in state:  # older checkpoints lack it
        outcomes.load(state["outcomes"])
    # Per setup, on the GPU, read once per update: the action types taken, then the decisions where the attention cap
    # binds, then those of a dead seat (past the spawn phase, not alive)
    tally = torch.zeros(len(names), envs.T + 2, device=device)
    invalid_count, env_time = 0, 0.0

    def rollout():
        """Collects T steps from every game; False if a stop was requested meanwhile."""
        nonlocal obs, global_step, invalid_count, env_time, teams
        for step in range(T):
            if not control.wait():
                return False
            t0 = time.perf_counter()
            mine = rows(obs, learner)
            host.put(step, stored(mine))
            full = torch.zeros(envs.N, 4, dtype=torch.long, device=device)
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(mine)
                dist = ActionDist(out, mine)
                act = dist.sample()
                actions[step], logps[step], values[step] = act, dist.log_prob(act), out["value"].float()
                full[learner] = act
                dead = (mine["scalars"][:, 1] == 0) & (mine["scalars"][:, 4] == 0)
                tally.index_add_(0, setup_row, torch.cat([nn.functional.one_hot(act[:, 0], envs.T).float(),
                                                          cap_binds(mine["players"]).float()[:, None],
                                                          dead.float()[:, None]], 1))
                for opponent, r in teams:
                    theirs = rows(obs, r)
                    with torch.autocast("cuda", torch.bfloat16):
                        out_o = opponent(theirs)
                    full[r] = ActionDist(out_o, theirs).sample()
            outcomes.record(mine["scalars"][:, 0].tolist(), values[step].tolist())
            acts = full.cpu().numpy()  # waits for the GPU
            te = time.perf_counter()
            obs, rew, done, infos, invalid = envs.step(acts)
            env_time += time.perf_counter() - te
            rewards[step] = torch.from_numpy(rew[learner_np]).to(device)
            dones[step] = torch.from_numpy(done[learner_np]).to(device)
            ended = [{**i, "row": row_of[i["env"], i["seat"]], "setup": setups[i["env"]]}
                     for i in infos if (i["env"], i["seat"]) in row_of]
            outcomes.finish(ended)
            slotted = [{**i, "env": slot_of[i["env"]]} for i in ended if i["env"] in slot_of]  # the league's view
            if league and slotted:  # these slots' next games are against the current pick
                league.finished(slotted)
                teams = opponents()
            invalid_count += int(invalid[learner_np].sum())
            global_step += L
            control.throttle(time.perf_counter() - t0)
        return True

    def learn():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            last_value = model(rows(obs, learner))["value"].float()
        adv = gae(rewards, values, dones, last_value, a.gamma, a.gae_lambda)
        ret = adv + values
        b_act, b_logp, b_adv, b_ret = actions.flatten(0, 1), logps.flatten(), adv.flatten(), ret.flatten()
        stats = collections.defaultdict(list)
        idx = [mb for _ in range(a.epochs) for mb in torch.randperm(T * L).split(host.size)]  # every epoch's minibatches
        on_gpu = torch.cat(idx).pin_memory().to(device, non_blocking=True).split([len(mb) for mb in idx])
        for i, (mb, o) in enumerate(zip(on_gpu, host.batches(idx, device))):
            o["cell_mask"] = unpack_bits(o["cell_mask"])
            with torch.autocast("cuda", torch.bfloat16):
                out = model(o)
            dist = ActionDist(out, o)
            logp = dist.log_prob(b_act[mb])
            # The entropy enters the graph only with a weight: 12 heat maps of it. At 0 it is logged after the
            # backward pass, since torch's Categorical caches each head's probabilities, and caching them without a
            # graph before the KL would cut the KL's gradients.
            ent = dist.entropy() if a.ent_coef else None
            log_ratio = logp - b_logp[mb]
            ratio = log_ratio.exp()
            mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
            pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
            v_loss = 0.5 * (out["value"].float() - b_ret[mb]).pow(2).mean()
            loss = pg_loss + a.vf_coef * v_loss - (a.ent_coef * ent.mean() if ent is not None else 0)
            if ref is not None:
                out_r = {k: o["ref_" + k] for k in REF_OUTPUTS if "ref_" + k in o}  # computed in the rollout
                kl = dist.kl(ActionDist(out_r, o)).mean()  # exact: the one-sample estimate spiked into the hundreds
                loss = loss + a.kl_coef * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
            opt.step()
            # The minibatch's only wait for the GPU, after all its work is queued: a wait mid-minibatch (an .item(), a
            # copy from pageable memory) left the GPU idle while the CPU queued the rest (PLAN 4A, "Speed work").
            with torch.no_grad():
                if ref is not None:
                    stats["loss/kl_ref"].append(kl.item())
                stats["loss/policy"].append(pg_loss.item())
                stats["loss/value"].append(v_loss.item())
                if ent is not None or i == 0:  # only for the log at ent_coef 0: once per update
                    stats["loss/entropy"].append((dist.entropy() if ent is None else ent).mean().item())
                stats["loss/approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["loss/clip_frac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
        metrics = {k: float(np.mean(v)) for k, v in stats.items()}
        var = ret.var().item()
        metrics["value/explained_variance"] = float("nan") if var == 0 else 1 - (ret - values).var().item() / var
        return metrics

    control.set_state("running")
    try:
        while True:
            if a.panel and update % a.eval_every == 0 and (update or a.init) \
                    and update not in {e["update"] for e in evals}:
                if not evaluate():
                    break
            if update >= a.updates:
                break
            if a.league and update % ROTATE == 0 and update:
                league.pick()
                control.note(f"PFSP picked {name(league.current)} for the next games")
            t0 = time.perf_counter()
            if not rollout():
                break
            t1 = time.perf_counter()
            torch.cuda.empty_cache()  # the rollouts' cached blocks would otherwise sit under learning's peak
            metrics = learn()
            control.throttle(time.perf_counter() - t1)
            update += 1
            metrics.update(outcomes.summary())
            dt = time.perf_counter() - t0
            metrics["perf/decisions_per_s"] = T * L / dt
            metrics["perf/rollout_share"] = (t1 - t0) / dt
            metrics["perf/env_share"] = env_time / dt  # the workers' simulation and the pipes, within the rollouts
            # past the card's size when Windows pages GPU memory into system RAM (nvidia-smi stops at the card)
            metrics["perf/gpu_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            metrics["perf/gpu_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30  # in use; the rest is cache
            metrics["perf/private_gb"] = psutil.Process().memory_info().private / 2**30  # Windows adds its GPU memory
            metrics["actions/invalid_rate"] = invalid_count / (T * L)
            counts = tally.cpu().numpy()
            tally.zero_()
            for i, act_name in enumerate(ACT_NAMES):
                metrics[f"actions/{act_name}"] = float(counts[:, i].sum() / (T * L))
            for k, setup in enumerate(names):
                n = T * rows_of[k]  # this setup's decisions
                for i, act_name in enumerate(ACT_NAMES):
                    metrics[f"actions/{setup}/{act_name}"] = float(counts[k, i] / n)
                metrics[f"setups/cap_binds/{setup}"] = float(counts[k, envs.T] / n)
                metrics[f"setups/dead_share/{setup}"] = float(counts[k, envs.T + 1] / n)
            if a.league:
                metrics["league/pool"], metrics["league/opponents"] = len(league.pool), len(teams)
                control.update(league=league.table())
            invalid_count, env_time = 0, 0.0
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
