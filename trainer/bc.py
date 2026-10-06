"""Behaviour cloning on ranked 1v1 replays (PLAN Phases 2-3): the policy learns to issue the command a human issued in
the same situation, and the value head learns who won.

  Train:   .venv/Scripts/python -m trainer.bc --run bc3-moe --core moe   (one pass over data/ from trainer.crawl)
  Tune:    .venv/Scripts/python -m trainer.bc --run probe --init checkpoints/bc/update0060000.pt --updates 2000
  Resume:  .venv/Scripts/python -m trainer.bc --run bc3-moe --resume
  Score:   .venv/Scripts/python -m trainer.bc --score checkpoints/bc/update0060000.pt checkpoints/bc3-moe/latest.pt
  Control: http://127.0.0.1:8787 or python -m trainer.ctl pause|resume|save|stop|throttle|status
  Charts:  .venv/Scripts/tensorboard --logdir runs
"""
import argparse
import datetime
import json
import math
import random
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trainer import crawl
from trainer.control import Control, serve
from trainer.env import RANKED_1V1, ROOT, VecEnv
from trainer.eval import tag
from trainer.model import (ATTACK, BOAT, BUILD, CORES, SHARES, SIZE_OF, SPAWN, ActionDist, describe, load_net,
                           new_net)
from trainer.ppo import game_commit, network_args, resume_args, save_checkpoint

CELL_HEAD = {SPAWN: 0, BOAT: 1, BUILD: 2}  # the heat map each type's cell comes from (trainer/model.py)
PHASES = ("spawn", "early", "mid", "late")
SCORE_CUTOFF = 1790812800000  # 2026-10-01 00:00 UTC, in ms: the scoring set comes from the Phase 2 crawl's games
SCORE_LAYOUT = {"workers": 12, "games_per_worker": 6, "steps": 2000}  # fixed, so every scoring run sees the same rows
SHARE_SMOOTH = 0.1  # a troop-share label's weight on its two neighbouring percents (PLAN 3B: soft labels)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="bc", help="name for checkpoints/<run> and runs/<run>")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/latest.pt")
    p.add_argument("--init", help="start from this checkpoint's weights and network, with a fresh optimizer")
    p.add_argument("--score", nargs="+", metavar="CHECKPOINT",
                   help="score these checkpoints on the fixed held-out set (~10 min) instead of training")
    p.add_argument("--core", choices=CORES, default="moe", help="the core of a new network (not with --init)")
    p.add_argument("--twin", choices=CORES,
                   help="also train a second new network with this core on the same windows, row by row, from the same "
                        "weights outside the core (PLAN 3B); it saves to checkpoints/<run>-<twin>")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--games-per-worker", type=int, default=8)
    p.add_argument("--val-workers", type=int, default=2, help="0 = no validation")
    p.add_argument("--val-share", type=float, default=0.05, help="share of the games held out")
    p.add_argument("--val-every", type=int, default=250, help="updates between validations")
    p.add_argument("--val-steps", type=int, default=100, help="replay steps per validation")
    p.add_argument("--updates", type=int, default=None, help="optimizer steps (default: one pass over the training records)")
    p.add_argument("--accum", type=int, default=8, help="replay windows averaged per optimizer step")
    p.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=500, help="updates of linear warmup before the cosine decay")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.999, help="discount of the value target, as in PPO")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--min-elo", type=float, default=0.0, help="leave players rated below this out of the loss")
    p.add_argument("--widths", default="64,128,192", help="CNN stage widths")
    p.add_argument("--seed", type=int, default=1,
                   help="network init and record order: runs with one seed replay the same records in the same order")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=250, help="updates between checkpoints (every 10th is kept)")
    p.add_argument("--port", type=int, default=8787, help="control page port (0 = any free port)")
    p.add_argument("--duty", type=float, default=1.0, help="start throttled: fraction of time spent working")
    p.add_argument("--gpu-temp-limit", type=int, default=0, help="auto-pause at this GPU temperature in C (0 = off)")
    return p


def parse_args(argv=None):
    return parser().parse_args(argv)


def is_val(game_id, share):
    """Held-out games, picked by a stable hash of the game ID."""
    return zlib.crc32(game_id.encode()) % 10_000 < share * 10_000


def run_info(a, train, val):
    """What the control page says about the run."""
    network = describe(vars(a)) + (f"; twin: {describe({**vars(a), 'core': a.twin})}" if a.twin else "")
    return {"Name": a.run, "Method": "behaviour cloning (imitating humans)", "Network": network,
            "Starts from": a.init or "random weights",
            "Data": f"{len(train):,} ranked 1v1 replays, {len(val):,} held out",
            "Games at once": f"{a.workers * a.games_per_worker} ({a.workers} workers x {a.games_per_worker})",
            "Updates": f"{a.updates:,} of {a.accum} windows each"}


def label_legal(obs, labels):
    """Rows whose label can enter the loss: an action type (not a command outside the set) that the observation's
    masks allow, with a legal cell or target."""
    t, cell, target = labels[:, 0], labels[:, 1], labels[:, 2]
    rows = torch.arange(len(t), device=t.device)
    ok = (t >= 0) & obs["type_mask"][rows, t.clamp(min=0)]
    head = torch.full_like(t, -1)
    for typ, h in CELL_HEAD.items():
        head[t == typ] = h
    cell_ok = obs["cell_mask"][rows, head.clamp(min=0), cell.clamp(0, obs["cell_mask"].shape[-1] - 1)]
    ok &= torch.where(head >= 0, cell_ok, True)
    target_ok = obs["attack_mask"][rows, target.clamp(0, obs["attack_mask"].shape[-1] - 1)]
    return ok & torch.where(t == ATTACK, target_ok, True)


def value_targets(env, gamma, device):
    """Per row, the discounted outcome: +-1 x gamma^(windows left), the return PPO's value head predicts."""
    left = torch.from_numpy(np.repeat(env.left, env.S)).to(device).float()
    return torch.from_numpy(env.outcome.reshape(-1).copy()).to(device) * gamma ** left


def bc_loss(model, obs, labels, legal, target, vf_coef, smooth=0.0):
    """-log pi(label) summed over the rows that can enter the loss and divided by all rows, plus the value loss. With
    `smooth`, a troop-share label keeps 1 - smooth of its weight and spreads the rest onto the neighbouring percents.
    Every window has the same rows, so k accumulated windows give the gradient of one batch of all of them."""
    with torch.autocast("cuda", torch.bfloat16):
        out = model(obs)
    dist = ActionDist(out, obs)
    safe = torch.where(legal[:, None], labels, torch.zeros_like(labels))  # other rows: a no-op
    nll = -(dist.log_prob(safe, smooth=smooth) * legal).sum() / len(legal)
    value = out["value"].float()
    return nll + vf_coef * 0.5 * (value - target).pow(2).mean(), dist, value, out.get("gates")


class Meter:
    """Running sums kept on the GPU (no sync per step); means() reads and resets them."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, name, total, count):
        self.sums[name] = self.sums.get(name, 0) + total
        self.counts[name] = self.counts.get(name, 0) + count

    def means(self, prefix=""):
        out = {prefix + k: float(s) / float(self.counts[k]) for k, s in self.sums.items() if float(self.counts[k]) > 0}
        self.sums, self.counts = {}, {}
        return out


@torch.no_grad()
def measure(m, obs, labels, legal, dist, value, target, gates=None):
    """For one batch:
    - label statistics;
    - accuracy and NLL per head. bc/nll and the troop heads' bc/*_bin are like for like for every network: a
      100-share network's shares are summed into the five sizes' ranges (SIZE_OF). Its own 100-share NLL is
      bc/nll_shares, with bc/nll_*_share per head;
    - NLL per bucket (game phase and map size);
    - the router's mean weight per expert and bucket, for MoE networks."""
    t = labels[:, 0]
    safe = torch.where(legal[:, None], labels, torch.zeros_like(labels))  # rows outside the loss: a no-op
    _, cell, tgt, b = safe.unbind(1)
    n, nl = len(t), legal.sum()
    logp = dist.log_prob(safe, sizes=True)
    m.add("labels/out_of_set", (t == -1).sum(), n)
    m.add("labels/unmapped", (t == -2).sum(), n)
    m.add("labels/illegal", ((t >= 0) & ~legal).sum(), n)
    m.add("labels/noop", ((t == 0) & legal).sum(), nl)
    m.add("bc/nll", -(logp * legal).sum(), nl)
    if dist.shares:
        m.add("bc/nll_shares", -(dist.log_prob(safe) * legal).sum(), nl)
    m.add("bc/nll_type", -(dist.type.log_prob(torch.where(legal, t, 0)) * legal).sum(), nl)
    m.add("bc/type_acc", ((dist.type.logits.argmax(1) == t) & legal).sum(), nl)
    m.add("bc/act_rate_human", ((t > 0) & legal).sum(), nl)
    m.add("bc/act_rate_policy", ((1 - dist.type.probs[:, 0]) * legal).sum(), nl)
    lc = dist.cells.log_prob(cell[:, None].expand(-1, 3))  # [N, maps]
    rows = legal & (t == ATTACK)
    m.add("bc/attack_target_acc", ((dist.ptr.logits.argmax(1) == tgt) & rows).sum(), rows.sum())
    m.add("bc/nll_attack_target", -(dist.ptr.log_prob(tgt) * rows).sum(), rows.sum())
    size = SIZE_OF.to(b.device)[b.clamp(0, SHARES)]  # the label's percent as one of the five sizes
    ls = dist.size_log_probs(tgt, cell)  # [N, attack/boat, 5]
    lt = dist.troop_log_prob(tgt, cell, b) if dist.shares else None  # [N, attack/boat]
    for k, (typ, name) in enumerate(((ATTACK, "attack"), (BOAT, "boat"))):
        rows = legal & (t == typ)
        m.add(f"bc/{name}_bin_acc", ((ls[:, k].argmax(1) == size) & rows).sum(), rows.sum())
        m.add(f"bc/nll_{name}_bin", -(ls[:, k].gather(1, size[:, None])[:, 0] * rows).sum(), rows.sum())
        if lt is not None:
            m.add(f"bc/nll_{name}_share", -(lt[:, k] * rows).sum(), rows.sum())
    for typ, name in ((SPAWN, "spawn"), (BOAT, "boat"), (BUILD, "city")):
        rows = legal & (t == typ)
        hit = dist.cells.logits[:, CELL_HEAD[typ]].argmax(1) == cell
        m.add(f"bc/{name}_cell_acc", (hit & rows).sum(), rows.sum())
        m.add(f"bc/nll_{name}_cell", -(lc[:, CELL_HEAD[typ]] * rows).sum(), rows.sum())
    phase = torch.where(obs["scalars"][:, 1] > 0, 0, 1 + (obs["scalars"][:, 0] * 3).long().clamp(max=2))
    buckets = {name: phase == i for i, name in enumerate(PHASES)}
    normal = obs["scalars"][:, 14] > 0.5  # tiles per cell / 64: Normal maps 1, Compact 0.25
    buckets.update(compact=~normal, normal=normal)
    for name, inside in buckets.items():
        rows = legal & inside
        m.add(f"bc/nll_{name}", -(logp * rows).sum(), rows.sum())
        if gates is not None:
            for i in range(gates.shape[1]):
                m.add(f"router/{name}_{i}", (gates[:, i].float() * inside).sum(), inside.sum())
    m.add("bc/value_mse", (value - target).pow(2).sum(), n)


def lr_at(update, a):
    """The learning rate of an update: a linear warmup over a.warmup updates, then a cosine decay to 0 at a.updates."""
    if update < a.warmup:
        return a.lr * (update + 1) / a.warmup
    done = min(1.0, (update - a.warmup) / max(1, a.updates - a.warmup))
    return a.lr * 0.5 * (1 + math.cos(math.pi * done))


def one_pass(entries, a):
    """Updates that replay every record of `entries` about once: each update takes a.accum windows of 5 turns from every
    game slot."""
    windows = sum(e["turns"] for e in entries) / 5
    return max(1, round(windows / (a.workers * a.games_per_worker * a.accum)))


def backward_windows(models, windows, a, meters):
    """Accumulates each network's gradient over a.accum windows, each (obs, labels, legal rows, value targets): their
    mean loss gives the gradient of one batch of all their rows. Every network trains on every window (--twin)."""
    for obs, labels, legal, target in windows:
        for model, meter in zip(models, meters):
            loss, dist, value, gates = bc_loss(model, obs, labels, legal, target, a.vf_coef, SHARE_SMOOTH)
            (loss / a.accum).backward()
            measure(meter, obs, labels, legal, dist, value, target, gates)


def score_records():
    """The fixed scoring set: the held-out records (5%, by game ID) of games started before SCORE_CUTOFF, by game ID.
    Later crawls do not change it; a purge can drop a record (the result counts them)."""
    entries = [e for e in crawl.load_index() if e["start"] < SCORE_CUTOFF and is_val(e["id"], 0.05)]
    paths = [crawl.record_path(e["id"]) for e in sorted(entries, key=lambda e: e["id"])]
    return [p for p in paths if p.exists()]


def score(paths, records, workers, games_per_worker, steps, device="cuda"):
    """Every checkpoint's statistics (measure, plus the value/outcome correlation) on the same replayed rows. The rows
    depend only on the records, the layout and the seed "score", so every comparison is paired."""
    envs = VecEnv(RANKED_1V1, workers, games_per_worker, seed="score", device=device, records=records)
    try:
        nets = [load_net(p, envs.dims, device) for p in paths]
        meters, values, outcomes, games = [Meter() for _ in nets], [[] for _ in nets], [], set()
        with torch.no_grad():
            for _ in range(steps):
                obs, labels = envs.obs, envs.labels
                target = value_targets(envs, 0.999, device)
                legal = label_legal(obs, labels)
                for net, m, v in zip(nets, meters, values):
                    _, dist, value, gates = bc_loss(net, obs, labels, legal, target, 0.0)
                    measure(m, obs, labels, legal, dist, value, target, gates)
                    v.append(value.cpu().numpy())
                outcomes.append(envs.outcome.reshape(-1).copy())
                games.update(envs.record_of)
                envs.step()
    finally:
        envs.close()
    o = np.concatenate(outcomes)
    metrics = []
    for m, v in zip(meters, values):
        stats, v = m.means(), np.concatenate(v)
        stats["value/outcome_corr"] = float(np.corrcoef(v, o)[0, 1]) if v.std() > 0 and o.std() > 0 else float("nan")
        metrics.append(stats)
    return {"checkpoints": [str(p) for p in paths], "records": len(records), "games": len(games), "rows": len(o),
            "metrics": metrics}


def table(result):
    """The scores as text: one row per statistic and one column per checkpoint (its run and update), then the
    routers' mean expert weights per bucket."""
    names = [tag(p) for p in result["checkpoints"]]
    keys = sorted({k for s in result["metrics"] for k in s if not k.startswith("router/")})
    w = max(len(k) for k in keys)
    lines = [f"{result['games']} games, {result['rows']:,} rows ({result['records']} held-out records before "
             f"2026-10-01)", " " * w + "".join(f"{n:>18}" for n in names)]
    lines += [k.ljust(w) + "".join(f"{s.get(k, float('nan')):18.4f}" for s in result["metrics"]) for k in keys]
    for name, s in zip(names, result["metrics"]):
        for bucket in (*PHASES, "compact", "normal"):
            weights = [s[k] for k in sorted(s) if k.startswith(f"router/{bucket}_")]
            if weights:
                lines.append(f"router {name} {bucket:>7}: " + " ".join(f"{x:.2f}" for x in weights))
    return "\n".join(lines)


def score_main(paths):
    records = score_records()
    if not records:
        raise SystemExit("no held-out records of games before 2026-10-01 in data/: run trainer.crawl first")
    result = score(paths, records, **SCORE_LAYOUT)
    out = ROOT / "runs" / "score" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(table(result))
    print(f"saved {out}")


def main(argv=None):
    a = parse_args(argv)
    if a.score:
        return score_main(a.score)
    if a.twin and a.init:
        raise SystemExit("--twin trains two new networks from the same random weights: drop --init")
    ckpt_dir = ROOT / "checkpoints" / a.run
    state = twin_state = None
    if a.resume:
        state = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        if "accum" not in state["args"]:
            raise SystemExit(f"checkpoints/{a.run} predates the Phase 3 optimizer (accumulation, learning-rate "
                             f"schedule), so it cannot resume. Start a new run from it: --run <new name> --init "
                             f"checkpoints/{a.run}/latest.pt")
        a = resume_args(parser(), argv, state["args"], ("updates",))
        if a.twin:
            path = ROOT / "checkpoints" / f"{a.run}-{a.twin}" / "latest.pt"
            if not path.exists():
                raise SystemExit(f"{path} is missing: this run also trains a {a.twin} twin, and both resume together")
            twin_state = torch.load(path, map_location="cpu", weights_only=False)
            if twin_state["update"] != state["update"]:
                raise SystemExit(f"the twin's checkpoint is at update {twin_state['update']} and the run's at "
                                 f"{state['update']}: both must resume from the same update")
    else:
        a = network_args(parser(), argv, a)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run_dir = ROOT / "runs" / a.run

    entries = [e for e in crawl.load_index() if crawl.record_path(e["id"]).exists()]
    ratings = crawl.elo(entries)
    train_entries = [e for e in entries if not is_val(e["id"], a.val_share)]
    train = [crawl.record_path(e["id"]) for e in train_entries]
    val = [crawl.record_path(e["id"]) for e in entries if is_val(e["id"], a.val_share)]
    if len(train) < a.workers or len(val) < a.val_workers:
        raise SystemExit(f"{len(train)} training and {len(val)} held-out records: run trainer.crawl first")
    if a.updates is None:
        a.updates = one_pass(train_entries, a)
    # Runs with one --seed replay the same records in the same order (the dense/MoE A/B), and a resumed run draws a new
    # order: with the old one, every stop and resume would replay the same records.
    start = state["update"] if state else 0
    envs = VecEnv(RANKED_1V1, a.workers, a.games_per_worker, seed=f"bc-{a.seed}-u{start}", device=device,
                  records=train, log_dir=run_dir / "workers")
    val_envs = None
    if a.val_workers:
        val_envs = VecEnv(RANKED_1V1, a.val_workers, a.games_per_worker, seed="val", device=device, records=val,
                          log_dir=run_dir / "val_workers")
    torch.manual_seed(a.seed)  # again before each network: a twin starts from the same weights outside its core
    nets = [new_net(envs.dims, vars(a)).to(device).to(memory_format=torch.channels_last)]
    # A resume hides what a crashed sitting logged after its last checkpoint (TensorBoard's purge_step).
    purge = state["global_step"] + 1 if state else None
    net_args, dirs, writers = [vars(a)], [ckpt_dir], [SummaryWriter(run_dir, purge_step=purge)]
    if a.twin:
        torch.manual_seed(a.seed)
        twin_run = f"{a.run}-{a.twin}"
        net_args.append({**vars(a), "run": twin_run, "core": a.twin, "twin": None})
        nets.append(new_net(envs.dims, net_args[1]).to(device).to(memory_format=torch.channels_last))
        dirs.append(ROOT / "checkpoints" / twin_run)
        writers.append(SummaryWriter(ROOT / "runs" / twin_run, purge_step=purge))
    model, writer = nets[0], writers[0]
    opts = [torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5) for net in nets]
    meters = [Meter() for _ in nets]
    update = global_step = 0
    if state:
        for net, opt, s in zip(nets, opts, (state, twin_state)):
            net.load_state_dict(s["model"])
            opt.load_state_dict(s["optimizer"])
        update, global_step = state["update"], state["global_step"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    elif a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
    control = Control(duty=a.duty, gpu_temp_limit=a.gpu_temp_limit)
    control.update(run=run_info(a, train, val))
    server = serve(control, a.port, log_dir=run_dir)
    control.note(f"control page on http://127.0.0.1:{server.server_address[1]}")
    commit = game_commit()
    done = {"records": 0, "desync": 0, "errors": 0}

    def elo_rows(env):
        return torch.tensor([ratings.get(rec, {}).get(c, 1500.0) for rec, seats in zip(env.record_of, env.seats_of)
                             for c in seats], device=device)

    def windows():
        """The next a.accum windows; the workers replay each next window while the GPU trains on the current one."""
        nonlocal global_step
        for _ in range(a.accum):
            obs, labels = envs.obs, envs.labels
            target = value_targets(envs, a.gamma, device)
            legal = label_legal(obs, labels) & (elo_rows(envs) >= a.min_elo)
            envs.send()
            yield obs, labels, legal, target
            for i in envs.recv()[3]:
                done["records"] += 1
                done["desync"] += int(i["desync"])
                done["errors"] += int(i["error"] is not None)
            global_step += envs.N

    def checkpoint():
        """Every network's checkpoint, at the same update."""
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all()}
        for net, opt, args, d in zip(nets, opts, net_args, dirs):
            s = dict(kind="bc", model=net.state_dict(), optimizer=opt.state_dict(), args=args, update=update,
                     global_step=global_step, game_commit=commit, rng=rng)
            save_checkpoint(d / "latest.pt", **s)
            if update % (10 * a.save_every) == 0 and update > 0:
                save_checkpoint(d / f"update{update:07d}.pt", **s)
        control.note(f"saved checkpoint at update {update}")

    def validate():
        """Every network's held-out statistics, on the same windows."""
        ms, values, outcomes = [Meter() for _ in nets], [[] for _ in nets], []
        with torch.no_grad():
            for _ in range(a.val_steps):
                obs, labels = val_envs.obs, val_envs.labels
                target = value_targets(val_envs, a.gamma, device)
                legal = label_legal(obs, labels)
                for net, m, v in zip(nets, ms, values):
                    _, dist, value, gates = bc_loss(net, obs, labels, legal, target, a.vf_coef)
                    measure(m, obs, labels, legal, dist, value, target, gates)
                    v.append(value.cpu().numpy())
                outcomes.append(val_envs.outcome.reshape(-1).copy())
                val_envs.step()
        o, results = np.concatenate(outcomes), []
        for m, v in zip(ms, values):
            out, v = m.means("val/"), np.concatenate(v)
            if v.std() > 0 and o.std() > 0:
                out["val/value_outcome_corr"] = float(np.corrcoef(v, o)[0, 1])
            results.append(out)
        return results

    control.set_state("running")
    t_log = time.perf_counter()
    try:
        while update < a.updates:
            if not control.wait():
                break
            t0 = time.perf_counter()
            for opt in opts:
                for group in opt.param_groups:
                    group["lr"] = lr_at(update, a)
                opt.zero_grad(set_to_none=True)
            backward_windows(nets, windows(), a, meters)
            for net, opt in zip(nets, opts):
                nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
                opt.step()
            update += 1
            control.throttle(time.perf_counter() - t0)
            if update % a.log_every == 0:
                metrics = meters[0].means()
                metrics.update({"data/records_done": done["records"], "data/errors": done["errors"],
                                "data/desync_share": done["desync"] / max(done["records"], 1),
                                "train/lr": opts[0].param_groups[0]["lr"],
                                "perf/samples_per_s": a.log_every * a.accum * envs.N / (time.perf_counter() - t_log)})
                t_log = time.perf_counter()
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
                if a.twin:
                    twin = meters[1].means()
                    for k, v in twin.items():
                        writers[1].add_scalar(k, v, global_step)
                    if "bc/nll" in twin:
                        writer.add_scalar("twin/bc/nll", twin["bc/nll"], global_step)  # the control page's loss chart
                control.update(update=update, global_step=global_step, metrics=metrics)
            if val_envs is not None and update % a.val_every == 0:
                results = validate()
                for w, metrics in zip(writers, results):
                    for k, v in metrics.items():
                        w.add_scalar(k, v, global_step)
                if a.twin and "val/bc/nll" in results[1]:
                    writer.add_scalar("twin/val/bc/nll", results[1]["val/bc/nll"], global_step)
            if update % a.save_every == 0 or control.take_save():
                checkpoint()
    except KeyboardInterrupt:
        control.note("interrupted")
    finally:
        control.set_state("saving")
        checkpoint()
        envs.close()
        if val_envs is not None:
            val_envs.close()
        for w in writers:
            w.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
