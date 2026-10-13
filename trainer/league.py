"""The 4A league (docs/PLAN.md, "League design"): the learner's opponents, picked by prioritized fictitious self-play
(PFSP), and a Bradley-Terry rating of its snapshots from their frozen games against the panel."""
from pathlib import Path

import numpy as np

ROTATE = 25  # updates between PFSP picks; a game in progress keeps its opponent
WINDOW = 50  # the learner's last training games against an opponent, which set its weight
ELO = 400 / np.log(10)  # one logit in Elo points
PRIOR = 0.01  # Bradley-Terry: a Gaussian prior of sd 10 logits on every rating keeps a perfect score finite


def name(path):
    """A checkpoint's short name: its run folder and file, e.g. ppo4a-s0/update001000."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


class League:
    """The pool of frozen opponents and each game slot's opponent. PFSP as AlphaStar's f_hard with p = 2: a member's
    weight is (1 - x)^2, x the learner's win rate in its last WINDOW training games against it, counted with one win
    and one loss more, so a new member starts at x = 1/2."""

    def __init__(self, pool, slots):
        self.pool = list(pool)
        self.results = {p: [] for p in self.pool}  # the learner's games against each member: 1 won, 0 lost
        self.pick()
        self.slot = [self.current] * slots

    def win_rate(self, p):
        r = self.results[p][-WINDOW:]
        return (sum(r) + 1) / (len(r) + 2)

    def weights(self):
        w = np.array([(1 - self.win_rate(p)) ** 2 for p in self.pool])
        return w / w.sum()

    def pick(self):
        """The opponent of the games that start from now on (numpy's global generator: checkpoints keep its state)."""
        self.current = self.pool[np.random.choice(len(self.pool), p=self.weights())]

    def add(self, path):
        if path not in self.results:
            self.pool.append(path)
            self.results[path] = []

    def finished(self, infos):
        """The learner's finished games: each counts against its slot's opponent, and the slot's next game is against
        the current pick."""
        for i in infos:
            self.results[self.slot[i["env"]]].append(int(i["rank"] == 1))
            self.slot[i["env"]] = self.current

    def state(self):
        return {"pool": self.pool, "results": self.results, "current": self.current}

    def load(self, s):
        """A resume drops the games in progress, so every slot starts against the saved pick."""
        self.pool, self.results, self.current = list(s["pool"]), s["results"], s["current"]
        self.slot = [self.current] * len(self.slot)

    def table(self):
        """The control page's rows: each member's games, the learner's win rate against it (as weighted) and its
        weight."""
        return [[name(p) + (" (now)" if p == self.current else ""),
                 f"{len(self.results[p])} games · learner {self.win_rate(p):.2f} · weight {w:.0%}"]
                for p, w in zip(self.pool, self.weights())]


def _fit(i, j, wins, games, k):
    """Ratings in logits of k players, player 0 at 0, from games of i against j (`wins` of `games`): Newton's method
    on the log-likelihood plus the prior."""
    r = np.zeros(k)
    for _ in range(100):
        p = 1 / (1 + np.exp(r[j] - r[i]))  # i beats j
        g = -PRIOR * r
        np.add.at(g, i, wins - games * p)
        np.add.at(g, j, games * p - wins)
        h = np.diag(np.full(k, PRIOR))  # minus the Hessian
        c = games * p * (1 - p)
        for a, b, s in ((i, i, c), (j, j, c), (i, j, -c), (j, i, -c)):
            np.add.at(h, (a, b), s)
        step = np.linalg.solve(h[1:, 1:], g[1:])
        r[1:] += step
        if np.abs(step).max() < 1e-8:
            break
    return r


def ratings(evals, panel, boot=200, last=5):
    """Bradley-Terry ratings in Elo points, the first panel member at 0, of the panel and every evaluated snapshot
    (by update), from the frozen evaluations (trainer.ppo's panel.jsonl records). Returns {player: (rating, bootstrap
    sd)} and the trend of the last `last` snapshots: its slope in points per 250 updates and the share of bootstrap
    fits in which it rises (None before 2 snapshots). The bootstrap redraws each pair's wins from its win rate."""
    at = {p: k for k, p in enumerate([*panel, *(e["update"] for e in evals)])}
    pairs = [(at[e["update"]], at[m], round(s["win_rate"] * s["episodes"]), s["episodes"])
             for e in evals for m, s in e["vs"].items() if s["episodes"]]
    i, j, w, n = (np.array(c) for c in zip(*pairs))
    point = ELO * _fit(i, j, w, n, len(at))
    rng = np.random.default_rng(0)
    draws = ELO * np.array([_fit(i, j, rng.binomial(n, w / n), n, len(at)) for _ in range(boot)])
    fit = {p: (float(point[k]), float(draws[:, k].std())) for p, k in at.items()}
    recent = [e["update"] for e in evals[-last:]]
    if len(recent) < 2:
        return fit, None
    cols, x = [at[u] for u in recent], np.array(recent) / 250
    slope = lambda r: np.polyfit(x, r[cols], 1)[0]
    return fit, (float(slope(point)), float(np.mean([slope(d) > 0 for d in draws])))


def panel_table(evals, panel, fit, trend):
    """The control page's rows: the latest snapshots' win rates against each panel member and their ratings, the
    panel's ratings, and the trend."""
    rows = []
    for e in evals[::-1][:8]:
        vs = " · ".join(f"{name(m)} {s['win_rate']:.2f}" for m, s in e["vs"].items())
        rows.append([f"update {e['update']:,} ({e['minutes']:.0f} min)",
                     "{} · rating {:+.0f} ± {:.0f}".format(vs, *fit[e["update"]])])
    rows += [[name(m), "0, the anchor" if k == 0 else "{:+.0f} ± {:.0f}".format(*fit[m])] for k, m in enumerate(panel)]
    if trend:
        rows.append([f"Rating trend, last {min(len(evals), 5)} snapshots",
                     f"{trend[0]:+.0f} per 250 updates, rising in {trend[1]:.0%} of the bootstrap"])
    return rows
