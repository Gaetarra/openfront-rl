"""Behaviour cloning on ranked 1v1 replays (PLAN Phase 2): the policy learns to issue the command a human issued in
the same situation, and the value head learns who won.

  Train:   .venv/Scripts/python -m trainer.bc --run bc          (needs data/ from trainer.crawl)
  Resume:  .venv/Scripts/python -m trainer.bc --run bc --resume
  Control: http://127.0.0.1:8787 or python -m trainer.ctl pause|resume|save|stop|throttle|status
  Charts:  .venv/Scripts/tensorboard --logdir runs
"""
import argparse
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
from trainer.model import ATTACK, BOAT, BUILD, SPAWN, ActionDist, Policy
from trainer.ppo import game_commit, save_checkpoint

KEEP_ON_RESUME = ("resume", "port", "duty", "gpu_temp_limit", "updates")
CELL_HEAD = {SPAWN: 0, BOAT: 1, BUILD: 2}  # the heat map each type's cell comes from (trainer/model.py)
PHASES = ("spawn", "early", "mid", "late")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="bc", help="name for checkpoints/<run> and runs/<run>")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/latest.pt")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--games-per-worker", type=int, default=8)
    p.add_argument("--val-workers", type=int, default=2, help="0 = no validation")
    p.add_argument("--val-share", type=float, default=0.05, help="share of the games held out")
    p.add_argument("--val-every", type=int, default=500, help="updates between validations")
    p.add_argument("--val-steps", type=int, default=100, help="replay steps per validation")
    p.add_argument("--updates", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.999, help="discount of the value target, as in PPO")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--min-elo", type=float, default=0.0, help="leave players rated below this out of the loss")
    p.add_argument("--widths", default="64,128,192", help="CNN stage widths")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000, help="updates between checkpoints")
    p.add_argument("--port", type=int, default=8787, help="control page port (0 = any free port)")
    p.add_argument("--duty", type=float, default=1.0, help="start throttled: fraction of time spent working")
    p.add_argument("--gpu-temp-limit", type=int, default=0, help="auto-pause at this GPU temperature in C (0 = off)")
    return p.parse_args(argv)


def is_val(game_id, share):
    """Held-out games, picked by a stable hash of the game ID."""
    return zlib.crc32(game_id.encode()) % 10_000 < share * 10_000


def run_info(a, train, val):
    """What the control page says about the run."""
    return {"Name": a.run, "Method": "behaviour cloning (imitating humans)",
            "Data": f"{len(train):,} ranked 1v1 replays, {len(val):,} held out",
            "Games at once": f"{a.workers * a.games_per_worker} ({a.workers} workers x {a.games_per_worker})",
            "Updates": f"{a.updates:,}"}


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


def bc_loss(model, obs, labels, legal, target, vf_coef):
    """-log pi(label) over the rows that can enter the loss, plus the value loss."""
    with torch.autocast("cuda", torch.bfloat16):
        out = model(obs)
    dist = ActionDist(out, obs)
    logp = dist.log_prob(torch.where(legal[:, None], labels, torch.zeros_like(labels)))  # other rows: a no-op
    nll = -(logp * legal).sum() / legal.sum().clamp(min=1)
    value = out["value"].float()
    return nll + vf_coef * 0.5 * (value - target).pow(2).mean(), dist, logp, value


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
def measure(m, obs, labels, legal, dist, logp, value, target):
    """Label statistics, per-head accuracy and NLL per game phase of one batch."""
    t, cell, tgt, b = labels.unbind(1)
    n, nl = len(t), legal.sum()
    m.add("labels/out_of_set", (t == -1).sum(), n)
    m.add("labels/unmapped", (t == -2).sum(), n)
    m.add("labels/illegal", ((t >= 0) & ~legal).sum(), n)
    m.add("labels/noop", ((t == 0) & legal).sum(), nl)
    m.add("bc/nll", -(logp * legal).sum(), nl)
    m.add("bc/type_acc", ((dist.type.logits.argmax(1) == t) & legal).sum(), nl)
    m.add("bc/act_rate_human", ((t > 0) & legal).sum(), nl)
    m.add("bc/act_rate_policy", ((1 - dist.type.probs[:, 0]) * legal).sum(), nl)
    rows = legal & (t == ATTACK)
    m.add("bc/attack_target_acc", ((dist.ptr.logits.argmax(1) == tgt) & rows).sum(), rows.sum())
    m.add("bc/attack_bin_acc", ((dist.bins.logits[:, 0].argmax(1) == b) & rows).sum(), rows.sum())
    rows = legal & (t == BOAT)
    m.add("bc/boat_bin_acc", ((dist.bins.logits[:, 1].argmax(1) == b) & rows).sum(), rows.sum())
    for typ, name in ((SPAWN, "spawn"), (BOAT, "boat"), (BUILD, "city")):
        rows = legal & (t == typ)
        hit = dist.cells.logits[:, CELL_HEAD[typ]].argmax(1) == cell
        m.add(f"bc/{name}_cell_acc", (hit & rows).sum(), rows.sum())
    phase = torch.where(obs["scalars"][:, 1] > 0, 0, 1 + (obs["scalars"][:, 0] * 3).long().clamp(max=2))
    for i, name in enumerate(PHASES):
        rows = legal & (phase == i)
        m.add(f"bc/nll_{name}", -(logp * rows).sum(), rows.sum())
    m.add("bc/value_mse", (value - target).pow(2).sum(), n)


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

    entries = [e for e in crawl.load_index() if crawl.record_path(e["id"]).exists()]
    ratings = crawl.elo(entries)
    train = [crawl.record_path(e["id"]) for e in entries if not is_val(e["id"], a.val_share)]
    val = [crawl.record_path(e["id"]) for e in entries if is_val(e["id"], a.val_share)]
    if len(train) < a.workers or len(val) < a.val_workers:
        raise SystemExit(f"{len(train)} training and {len(val)} held-out records: run trainer.crawl first")
    # A resumed run draws a new record order: with the old one, every stop and resume would replay the same records.
    start = state["update"] if state else 0
    envs = VecEnv(RANKED_1V1, a.workers, a.games_per_worker, seed=f"{a.run}-{a.seed}-u{start}", device=device,
                  records=train, log_dir=run_dir / "workers")
    val_envs = None
    if a.val_workers:
        val_envs = VecEnv(RANKED_1V1, a.val_workers, a.games_per_worker, seed="val", device=device, records=val,
                          log_dir=run_dir / "val_workers")
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
    control.update(run=run_info(a, train, val))
    server = serve(control, a.port, log_dir=run_dir)
    control.note(f"control page on http://127.0.0.1:{server.server_address[1]}")
    writer = SummaryWriter(run_dir)
    commit = game_commit()
    meter, done = Meter(), {"records": 0, "desync": 0, "errors": 0}

    def elo_rows(env):
        return torch.tensor([ratings.get(rec, {}).get(c, 1500.0) for rec, seats in zip(env.record_of, env.seats_of)
                             for c in seats], device=device)

    def checkpoint():
        s = dict(kind="bc", model=model.state_dict(), optimizer=opt.state_dict(), args=vars(a), update=update,
                 global_step=global_step, game_commit=commit,
                 rng={"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all()})
        save_checkpoint(ckpt_dir / "latest.pt", **s)
        if update % (10 * a.save_every) == 0 and update > 0:
            save_checkpoint(ckpt_dir / f"update{update:07d}.pt", **s)
        control.note(f"saved checkpoint at update {update}")

    def validate():
        m, values, outcomes = Meter(), [], []
        with torch.no_grad():
            for _ in range(a.val_steps):
                obs, labels = val_envs.obs, val_envs.labels
                target = value_targets(val_envs, a.gamma, device)
                legal = label_legal(obs, labels)
                _, dist, logp, value = bc_loss(model, obs, labels, legal, target, a.vf_coef)
                measure(m, obs, labels, legal, dist, logp, value, target)
                values.append(value.cpu().numpy())
                outcomes.append(val_envs.outcome.reshape(-1).copy())
                val_envs.step()
        out = m.means("val/")
        v, o = np.concatenate(values), np.concatenate(outcomes)
        if v.std() > 0 and o.std() > 0:
            out["val/value_outcome_corr"] = float(np.corrcoef(v, o)[0, 1])
        return out

    control.set_state("running")
    t_log = time.perf_counter()
    try:
        while update < a.updates:
            if not control.wait():
                break
            t0 = time.perf_counter()
            obs, labels = envs.obs, envs.labels
            target = value_targets(envs, a.gamma, device)
            legal = label_legal(obs, labels) & (elo_rows(envs) >= a.min_elo)
            envs.send()  # the workers replay the next window while the GPU trains on this one
            loss, dist, logp, value = bc_loss(model, obs, labels, legal, target, a.vf_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
            opt.step()
            measure(meter, obs, labels, legal, dist, logp, value, target)
            infos = envs.recv()[3]
            for i in infos:
                done["records"] += 1
                done["desync"] += int(i["desync"])
                done["errors"] += int(i["error"] is not None)
            update += 1
            global_step += envs.N
            control.throttle(time.perf_counter() - t0)
            if update % a.log_every == 0:
                metrics = meter.means()
                metrics.update({"data/records_done": done["records"], "data/errors": done["errors"],
                                "data/desync_share": done["desync"] / max(done["records"], 1),
                                "perf/samples_per_s": a.log_every * envs.N / (time.perf_counter() - t_log)})
                t_log = time.perf_counter()
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
                control.update(update=update, global_step=global_step, metrics=metrics)
            if val_envs is not None and update % a.val_every == 0:
                for k, v in validate().items():
                    writer.add_scalar(k, v, global_step)
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
        writer.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
