"""Phase 1 diagnosis (docs/PLAN.md, "Phase 1 results"): random-legal play vs a scripted expansion policy at 2- and
5-minute timers, 1 agent vs 3 Easy nations on Compact Onion, 64 games per setting.

  .venv/Scripts/python scripts/diag_baselines.py      (repo root; starts 8 Node workers, needs the GPU)
"""
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
from trainer.env import VecEnv  # noqa: E402

ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "seats": 1}
SIZES = [10, 25, 50, 75, 100]


def act(obs, rng, policy, step):
    """policy: "random" (uniform over legal actions) or (troop share in percent, attack every n decisions)."""
    types, attack, cells = (obs[k].cpu().numpy() for k in ("type_mask", "attack_mask", "cell_mask"))
    rows = []
    for i in range(types.shape[0]):
        if policy == "random":
            t = rng.choice(np.flatnonzero(types[i]))
            m = {1: 0, 3: 1, 4: 2}.get(t)
            cell = rng.choice(np.flatnonzero(cells[i, m])) if m is not None and cells[i, m].any() else 0
            rows.append([t, cell, rng.choice(np.flatnonzero(attack[i])) if t == 2 else 0, int(rng.choice(SIZES))])
            continue
        share, every = policy
        if types[i, 1] and (step == 0 or not types[i, 0]):  # spawn at the start, again only when forced
            rows.append([1, rng.choice(np.flatnonzero(cells[i, 0])), 0, 0])
        elif types[i, 2] and attack[i, 0] and step % every == 0:  # attack unowned land
            rows.append([2, 0, 0, share])
        else:
            rows.append([0, 0, 0, 0])
    return np.array(rows, np.int64)


for timer in (2, 5):
    for policy in ("random", (10, 1), (25, 4)):
        env = VecEnv({**ONION, "timerMinutes": timer}, 8, 8, seed="diag")
        rng, obs, ranks, step, t0 = np.random.default_rng(0), env.obs, {}, 0, time.time()
        while len(ranks) < env.E:
            obs, _, _, infos, _ = env.step(act(obs, rng, policy, step))
            step += 1
            for info in infos:
                ranks.setdefault(info["env"], info["rank"])
        env.close()
        r = list(ranks.values())
        name = policy if policy == "random" else f"expand {policy[0]}% every {policy[1]}"
        print(f"timer {timer} min | {name:20s} | mean rank {np.mean(r):.2f} | ranks {dict(sorted(Counter(r).items()))}"
              f" | {time.time() - t0:.0f}s", flush=True)
