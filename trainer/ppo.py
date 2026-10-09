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
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trainer.control import Control, serve
from trainer.env import GAME_DIR, RANKED_1V1, ROOT, VecEnv
from trainer.model import ACT_NAMES, CORES, ActionDist, arch_of, load_net, new_net

SESSION = ("port", "duty", "gpu_temp_limit")  # settings of this sitting, not of the run: a resume takes them as given


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
    p.add_argument("--seats", type=int, default=1, help="agent seats in the onion setup")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--steps", type=int, default=None,
                   help="decisions per game per update (default: 48 for ranked1v1, which keeps the run under 8 GiB of "
                        "VRAM; 128 otherwise)")
    p.add_argument("--updates", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatches", type=int, default=None, help="default: 8 for ranked1v1, 4 otherwise")
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--widths", default="64,128,192", help="CNN stage widths")
    p.add_argument("--core", choices=CORES, default="moe", help="the core of a new network (not with --init)")
    p.add_argument("--init", help="start from this checkpoint's weights, e.g. checkpoints/bc/latest.pt")
    p.add_argument("--opponent", help="frozen checkpoint that plays every seat but the learner's (2-seat setups)")
    p.add_argument("--kl-coef", type=float, default=None,
                   help="KL penalty toward the --init policy (default 0; given with --resume, it replaces the run's)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=25, help="updates between checkpoints")
    p.add_argument("--port", type=int, default=8787, help="control page port (0 = any free port)")
    p.add_argument("--duty", type=float, default=1.0, help="start throttled: fraction of time spent working")
    p.add_argument("--gpu-temp-limit", type=int, default=0, help="auto-pause at this GPU temperature in C (0 = off)")
    return p


def parse_args(argv=None):
    return parser().parse_args(argv)


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
    if a.setup == "ranked1v1":
        return RANKED_1V1
    return {"map": a.map, "compact": not a.normal_size, "nations": a.nations, "difficulty": a.difficulty,
            "bots": a.bots, "timerMinutes": a.timer, "seats": a.seats}


def run_info(a):
    """What the control page says about the run."""
    if a.setup == "ranked1v1":
        game, rivals = "ranked 1v1 (the official playlist)", f"frozen {a.opponent}"
    else:
        game = f"{a.map}, {'Normal' if a.normal_size else 'Compact'}, {a.timer} min"
        rivals = f"{a.nations} {a.difficulty} nations" + (f" and {a.bots} bots" if a.bots else "")
        if a.seats > 1:
            rivals += f"; frozen {a.opponent} in the other seats"
    seats = 2 if a.setup == "ranked1v1" else a.seats
    plays = {"Learner seat": "both, half the games each" if seats == 2 else f"all {seats}, equally often"} \
        if seats > 1 else {}
    return {"Name": a.run, "Method": "PPO (reinforcement learning)", "Game": game, "Against": rivals, **plays,
            "Starts from": a.init or "random weights", "KL penalty to the start": a.kl_coef or "none",
            "Games at once": f"{a.workers * a.games_per_worker} ({a.workers} workers x {a.games_per_worker})",
            "Updates": f"{a.updates:,}"}


def seat_rows(E, S, device="cpu"):
    """Rows of the learner and of every other seat. The learner plays seat e % S of game slot e, so it plays each seat
    equally often, as in trainer.eval (the game is symmetric, so the per-seat win rates should match)."""
    seat = torch.arange(E, device=device) % S
    base = torch.arange(E, device=device) * S
    return base + seat, [base + (seat + s) % S for s in range(1, S)], seat


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

    def finish(self, infos):
        """The learner's finished games; its row is the game slot, whichever seat it plays there."""
        for info in infos:
            row = info["env"]
            self.samples.extend((min(int(p * 3), 2), v, info["reward"]) for p, v in self.pending[row])
            self.pending[row] = []
            self.episodes.append(info)

    def state(self):
        """The finished games and value samples, kept in checkpoints so the curves carry on after a resume."""
        return {"episodes": list(self.episodes), "samples": list(self.samples)}

    def load(self, state):
        self.episodes.extend(state["episodes"])
        self.samples.extend(state["samples"])

    def summary(self):
        if not self.episodes:
            return {}
        ep = list(self.episodes)
        out = {"episodes/win_rate": float(np.mean([e["rank"] == 1 for e in ep])),
               "episodes/mean_reward": float(np.mean([e["reward"] for e in ep])),
               "episodes/mean_rank": float(np.mean([e["rank"] for e in ep])),
               "episodes/ticks": float(np.mean([e["ticks"] for e in ep]))}
        seats = sorted({e["seat"] for e in ep})
        if len(seats) > 1:  # the learner plays every seat: does it win from each?
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


def main(argv=None):
    a = parse_args(argv)
    ckpt_dir = ROOT / "checkpoints" / a.run
    state = None
    if a.resume:
        state = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        a = resume_args(parser(), argv, state["args"], ("updates", "kl_coef"))
    else:
        a = network_args(parser(), argv, a)
    a.kl_coef = a.kl_coef or 0.0
    ranked = a.setup == "ranked1v1"
    a.steps = a.steps or (48 if ranked else 128)
    a.minibatches = a.minibatches or (8 if ranked else 4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run_dir = ROOT / "runs" / a.run

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
    S = envs.S
    if S > 1 and not a.opponent:
        raise SystemExit("2-seat setups need --opponent")
    if a.kl_coef > 0 and not a.init:
        raise SystemExit("--kl-coef needs --init")
    if a.init and not state:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
    frozen = {}

    def net(path):
        if path not in frozen:
            frozen[path] = load_net(path, envs.dims, device)
        return frozen[path]

    opponent = net(a.opponent) if S > 1 else None
    ref = net(a.init) if a.kl_coef > 0 else None
    learner, others, seat = seat_rows(envs.E, S, device)
    games, seat = np.arange(envs.E), seat.cpu().numpy()  # for the per-seat arrays the workers return
    rows = lambda o, r: {k: v[r] for k, v in o.items()}
    control = Control(duty=a.duty, gpu_temp_limit=a.gpu_temp_limit)
    control.update(run=run_info(a))
    server = serve(control, a.port, log_dir=run_dir)
    control.note(f"control page on http://127.0.0.1:{server.server_address[1]}")
    writer = SummaryWriter(run_dir, purge_step=global_step + 1 if state else None)  # a resume hides a crash's tail
    commit = game_commit()

    def checkpoint():
        # Games in progress are dropped on quit; a resume starts each slot at its next episode, so no seed replays.
        s = dict(model=model.state_dict(), optimizer=opt.state_dict(), args=vars(a), update=update,
                 global_step=global_step, episodes=[e + 1 for e in envs.episodes], game_commit=commit,
                 outcomes=outcomes.state(), rng={"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all()})
        save_checkpoint(ckpt_dir / "latest.pt", **s)
        if update % 250 == 0 and update > 0:
            save_checkpoint(ckpt_dir / f"update{update:06d}.pt", **s)
        control.note(f"saved checkpoint at update {update}")

    T, N = a.steps, envs.E
    obs = envs.obs
    def stored(o):
        """A seat's rows as the rollout keeps them: the cell masks as bits."""
        return {**o, "cell_mask": pack_bits(o["cell_mask"])}

    buf = {k: torch.zeros((T,) + tuple(v.shape), dtype=v.dtype, device=device)
           for k, v in stored(rows(obs, learner)).items()}
    actions = torch.zeros(T, N, 4, dtype=torch.long, device=device)
    logps, values, rewards, dones = (torch.zeros(T, N, device=device) for _ in range(4))
    outcomes = Outcomes(N)
    if state and "outcomes" in state:  # older checkpoints lack it
        outcomes.load(state["outcomes"])
    act_counts, invalid_count = np.zeros(envs.T), 0

    def rollout():
        """Collects T steps from every game; False if a stop was requested meanwhile."""
        nonlocal obs, global_step, invalid_count
        for step in range(T):
            if not control.wait():
                return False
            t0 = time.perf_counter()
            mine = rows(obs, learner)
            for k, v in stored(mine).items():
                buf[k][step] = v
            full = torch.zeros(envs.N, 4, dtype=torch.long, device=device)
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(mine)
                dist = ActionDist(out, mine)
                act = dist.sample()
                actions[step], logps[step], values[step] = act, dist.log_prob(act), out["value"].float()
                full[learner] = act
                for r in others:
                    theirs = rows(obs, r)
                    with torch.autocast("cuda", torch.bfloat16):
                        out_o = opponent(theirs)
                    full[r] = ActionDist(out_o, theirs).sample()
            outcomes.record(mine["scalars"][:, 0].tolist(), values[step].tolist())
            obs, rew, done, infos, invalid = envs.step(full.cpu().numpy())
            rewards[step] = torch.from_numpy(rew.reshape(envs.E, S)[games, seat]).to(device)
            dones[step] = torch.from_numpy(done.reshape(envs.E, S)[games, seat]).to(device)
            outcomes.finish([i for i in infos if i["seat"] == seat[i["env"]]])
            np.add.at(act_counts, act.cpu().numpy()[:, 0], 1)
            invalid_count += int(invalid.reshape(envs.E, S)[games, seat].sum())
            global_step += N
            control.throttle(time.perf_counter() - t0)
        return True

    def learn():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            last_value = model(rows(obs, learner))["value"].float()
        adv = gae(rewards, values, dones, last_value, a.gamma, a.gae_lambda)
        ret = adv + values
        flat = {k: v.flatten(0, 1) for k, v in buf.items()}
        b_act, b_logp, b_adv, b_ret = actions.flatten(0, 1), logps.flatten(), adv.flatten(), ret.flatten()
        stats = collections.defaultdict(list)
        for _ in range(a.epochs):
            for mb in torch.randperm(T * N, device=device).split(T * N // a.minibatches):
                o = {k: v[mb] for k, v in flat.items()}
                o["cell_mask"] = unpack_bits(o["cell_mask"])
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
                if ref is not None:
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        out_r = ref(o)
                    kl = dist.kl(ActionDist(out_r, o)).mean()  # exact: the one-sample estimate spiked into the hundreds
                    loss = loss + a.kl_coef * kl
                    stats["loss/kl_ref"].append(kl.item())
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
            dt = time.perf_counter() - t0
            metrics["perf/decisions_per_s"] = T * N / dt
            metrics["perf/rollout_share"] = (t1 - t0) / dt
            # past the card's size when Windows pages GPU memory into system RAM (nvidia-smi stops at the card)
            metrics["perf/gpu_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
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
