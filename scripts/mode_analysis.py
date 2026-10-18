"""The two mode analyses of PLAN "After 4N" item 2, on the human games in data/v0.35.0/ (v0.35.0-v0.35.2):

  .venv/Scripts/python scripts/mode_analysis.py play     how humans play each mode, per player-minute (5b; ~2 min)
  .venv/Scripts/python scripts/mode_analysis.py traces   replays a stable sample of each mode (scripts/mode_traces.ts;
                                                         ~2 h on 12 workers, resumable: detach it)
  .venv/Scripts/python scripts/mode_analysis.py win      what predicts winning, per mode and minute (5a, the traces)

The modes are the crawler's, with Humans vs Nations ("hvn") apart from the other public Team games. Results are printed
and saved to runs/modes/{play,win}.json.
"""
import argparse
import collections
import gzip
import json
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

from coverage import BUILDS, IN_SET  # scripts/coverage.py: the 15 actions

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "v0.35.0"
GAME_DIR = ROOT / "OpenFront_Source_Code" / "openfrontio"
OUT = ROOT / "runs" / "modes"
MODES = ["ranked1v1", "ranked2v2", "rankedffa", "clanwars", "ffa", "team", "hvn"]
TEAM_MODES = {"ranked2v2", "clanwars", "team", "hvn"}
MINUTE = 600  # ticks
MINUTES = [1, 2, 3, 5, 7, 10, 15, 20]  # after the spawn phase (half of the ranked 1v1 games end by 6.5)
AHEAD, LOSS_AT = 3, [2, 5, 10]  # land lost in the AHEAD minutes after each of LOSS_AT


def mode_of(entry, record):
    """The analysis mode of an index entry and its record: the crawler's, with Humans vs Nations apart."""
    hvn = record["info"]["config"].get("playerTeams") == "Humans Vs Nations"
    return "hvn" if hvn else entry["mode"]


def index():
    with open(STORE / "index.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(gid):
    return json.loads(gzip.decompress((STORE / "records" / f"{gid}.json.gz").read_bytes()))


def records():
    """(index entry, record) of every record in the store."""
    for e in index():
        yield e, load(e["id"])


def table(rows, head):
    """A markdown table."""
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


# ---- 5b: how humans play each mode ----------------------------------------------------------------------------------

KINDS = ["attack", "boat", "cancel_attack", "cancel_boat", "build:City", "build:Port", "build:Factory",
         "build:Defense Post", "build:SAM Launcher", "build:Missile Silo", "build:Warship", "build:Atom Bomb",
         "build:Hydrogen Bomb", "build:MIRV", "upgrade_structure", "move_warship", "allianceRequest", "allianceReject",
         "allianceExtension", "breakAlliance", "donate_troops", "donate_gold", "embargo", "embargo_all", "targetPlayer",
         "quick_chat", "emoji"]
GOLD = ["work", "war", "trade", "steal", "own trains", "others' trains", "donations"]  # StatsSchemas GOLD_INDEX_*


def kind_of(intent):
    t = intent["type"]
    return f"build:{intent.get('unit')}" if t == "build_unit" else t


def play_of(record):
    """Per active human (one who sent a command): (minutes played, command counts by kind, end stats). A player's time
    ends at its elimination, at a disconnect it did not come back from, or at the game's end."""
    info = record["info"]
    end = info["num_turns"]
    gone, counts = {}, collections.defaultdict(collections.Counter)
    for turn in record["turns"]:
        for i in turn.get("intents", []):
            if i["type"] == "mark_disconnected":
                gone[i["clientID"]] = turn["turnNumber"] if i.get("isDisconnected") else None
            else:
                counts[i["clientID"]][kind_of(i)] += 1
    out = []
    for p in info["players"]:
        c = p["clientID"]
        if not counts[c]:
            continue
        stats = p.get("stats") or {}
        stop = min(int(stats.get("killedAt") or end), gone.get(c) or end, end)
        out.append((stop / MINUTE, counts[c], stats))
    return out


def num(a, i):
    return int(a[i]) if a and len(a) > i else 0


def nukes(count):
    return sum(count[f"build:{u}"] for u in ("Atom Bomb", "Hydrogen Bomb", "MIRV"))


def play(_args):
    acc = {m: {"games": 0, "players": 0, "minutes": 0.0, "count": collections.Counter(), "users": collections.Counter(),
               "gold": np.zeros(len(GOLD)), "trans": np.zeros(5), "trade_captured": 0, "bombs": np.zeros(3)}
           for m in MODES}
    for e, r in records():
        a = acc[mode_of(e, r)]
        a["games"] += 1
        for minutes, counts, s in play_of(r):
            a["players"] += 1
            a["minutes"] += minutes
            a["count"].update(counts)
            a["users"].update(counts.keys())
            a["gold"] += [num(s.get("gold"), i) for i in range(len(GOLD))]
            a["trans"] += [num((s.get("boats") or {}).get("trans"), i) for i in range(5)]
            a["trade_captured"] += num((s.get("boats") or {}).get("trade"), 2)
            for kind, b in (s.get("bombs") or {}).items():
                if kind != "mirvw":  # a MIRV's warheads
                    a["bombs"] += [num(b, i) for i in range(3)]
    modes = [m for m in MODES if acc[m]["players"]]
    rows = [["games", *(acc[m]["games"] for m in modes)], ["active players", *(acc[m]["players"] for m in modes)],
            ["minutes per player", *(f"{acc[m]['minutes'] / acc[m]['players']:.1f}" for m in modes)]]
    for k in KINDS:
        rows.append([k, *(f"{acc[m]['count'][k] / acc[m]['minutes']:.3g} ({acc[m]['users'][k] / acc[m]['players']:.0%})"
                          for m in modes)])
    out_of_set = {}
    for m in modes:
        c = acc[m]["count"]
        n = sum(c.values())
        inside = sum(v for k, v in c.items() if k in IN_SET or (k.startswith("build:") and k[6:] in BUILDS))
        out_of_set[m] = 1 - inside / n
    rows.append(["outside the 15 actions", *(f"{out_of_set[m]:.2%}" for m in modes)])
    print("Commands per player-minute (players who used it at least once):\n" + table(rows, ["", *modes]))
    rows = [[f"gold: {g}", *(f"{acc[m]['gold'][i] / acc[m]['gold'].sum():.1%}" for m in modes)]
            for i, g in enumerate(GOLD)]
    rows += [["gold earned per player-minute", *(f"{acc[m]['gold'].sum() / acc[m]['minutes']:,.0f}" for m in modes)],
             ["boats: share arrived", *(f"{acc[m]['trans'][1] / max(acc[m]['trans'][0], 1):.0%}" for m in modes)],
             ["boats: share lost", *(f"{acc[m]['trans'][4] / max(acc[m]['trans'][0], 1):.0%}" for m in modes)],
             ["trade ships captured per player-hour",
              *(f"{60 * acc[m]['trade_captured'] / acc[m]['minutes']:.2f}" for m in modes)],
             # the game credits an interception to the SAM's owner, and counts a launch only at a player's land
             ["SAM intercepts per human nuke", *(f"{acc[m]['bombs'][2] / max(nukes(acc[m]['count']), 1):.0%}"
                                                 for m in modes)]]
    print("\nFrom the players' end-of-game stats:\n" + table(rows, ["", *modes]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "play.json").write_text(json.dumps(
        {m: {**{k: v for k, v in acc[m].items() if k in ("games", "players", "minutes", "trade_captured")},
             "count": dict(acc[m]["count"]), "users": dict(acc[m]["users"]), "gold": dict(zip(GOLD, acc[m]["gold"])),
             "trans": acc[m]["trans"].tolist(), "bombs": acc[m]["bombs"].tolist(), "out_of_set": out_of_set[m]}
         for m in modes}, indent=1, default=float))


# ---- the traces -----------------------------------------------------------------------------------------------------

def sample_key(gid):
    """A record's place in the stable sample: the lowest keys go first, so a larger sample keeps a smaller one."""
    return zlib.crc32(f"analysis:{gid}".encode())


def traced():
    """The traces written so far, one dict per record."""
    for part in sorted(OUT.glob("traces-*.jsonl")):
        for line in open(part, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def traces(args):
    by = collections.defaultdict(list)
    for e in index():
        by[mode_of(e, load(e["id"])) if e["mode"] == "team" else e["mode"]].append(e)
    done = {m.group(1) for part in OUT.glob("traces-*.jsonl") for line in open(part, encoding="utf-8")
            if (m := re.match(r'\{"id":"([^"]+)"', line))}
    todo = []
    for m in MODES:
        pick = sorted(by[m], key=lambda e: sample_key(e["id"]))[:args.per_mode]
        todo += [(m, e) for e in pick if e["id"] not in done]
        print(f"{m}: {len(pick)} sampled of {len(by[m])}, {sum(e['id'] in done for e in pick)} traced")
    todo.sort(key=lambda me: -me[1]["turns"] * len(me[1]["players"]) ** 0.5)  # long games first, spread over workers
    if not todo:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    procs = []
    for k in range(args.workers):
        mine = todo[k::args.workers]
        if not mine:
            break
        lst = OUT / f"list-{k}.txt"
        lst.write_text("".join(f"{m} {(STORE / 'records' / (e['id'] + '.json.gz')).as_posix()}\n" for m, e in mine))
        log = open(OUT / f"traces-{k}.log", "a")
        procs.append(subprocess.Popen([shutil.which("node"), "--import", "tsx", "../../scripts/mode_traces.ts",
                                       str(lst), str(OUT / f"traces-{k}.jsonl")], cwd=GAME_DIR, stdout=log, stderr=log))
    print(f"{len(todo)} records on {len(procs)} workers (progress: runs/modes/traces-*.log)", flush=True)
    codes = [p.wait() for p in procs]
    print("worker exits:", codes)
    sys.exit(max(codes))


# ---- 5a: what predicts winning --------------------------------------------------------------------------------------

UNITS = ["city", "port", "factory", "defense post", "SAM", "silo", "warship"]  # mode_traces.ts UNITS
RAW = ["tiles", "troops", "gold", "earned", *UNITS, "kills", "tribes", "allies", "attackers", "away"]
FULL = ["tiles", "troops", "gold", *UNITS, "kills", "tribes", "allies", "attackers", "away share"]
R = {f: i for i, f in enumerate(RAW)}


def sides_at(tr, by_team, minutes):
    """A trace's sides (players, or teams in team modes) at each of the minutes the game reached: their raw measures
    [sides, RAW], summed over members. Also the winner's index (None when the replay's winner is not the record's)."""
    members = tr["sides"]  # [smallID, type, clientID or nation name, team]
    keys = [m[3] if by_team else m[0] for m in members]
    names = list(dict.fromkeys(keys))
    of = np.array([names.index(k) for k in keys])
    at = {m[0]: i for i, m in enumerate(members)}
    t = np.array(tr["t"])
    out = {}
    for m in minutes:
        tick = tr["spawn"] + m * MINUTE
        if t[-1] < tick:
            break  # the game was over
        k = np.searchsorted(t, tick)
        raw = np.zeros((len(members), len(RAW)))
        for f in ("tiles", "troops", "gold", "allies", "attackers", "away"):
            raw[:, R[f]] = tr[f][k]
        raw[:, R["earned"]] = [np.nan if v is None else v for v in tr["earned"][k]]
        raw[:, R["city"]:R["city"] + len(UNITS)] = tr["units"][k]
        for when, a, _, kind in tr["kills"]:
            if when <= t[k] and a in at:
                raw[at[a], R["tribes" if kind == "BOT" else "kills"]] += 1
        side = np.zeros((len(names), len(RAW)))
        np.add.at(side, of, raw)  # a nation's missing gold earned stays in its own side
        out[m] = side
    w = tr["winner"]
    if by_team:
        key = w[1] if w[0] == "team" else None
    else:
        key = next((m[0] for m in members if w[0] == "player" and m[2] == w[1]), None)
    return out, names.index(key) if key is not None and key == tr["simWinner"] else None


def features(side):
    """The model's inputs of the alive sides at one sample: log shares of tiles, troops and gold, log(1 + x) of the
    structure levels, kills and tribes taken, alliances and attackers as counts, and the share of the army away."""
    def share(x):
        return np.log(x / x.sum())
    troops, away = side[:, R["troops"]], side[:, R["away"]]
    cols = {"tiles": share(side[:, R["tiles"]]), "troops": share(np.maximum(troops, 1)),
            "gold": share(side[:, R["gold"]] + 1), "earned": share(side[:, R["earned"]] + 1),
            **{u: np.log1p(side[:, R[u]]) for u in UNITS}, "kills": np.log1p(side[:, R["kills"]]),
            "tribes": np.log1p(side[:, R["tribes"]]), "allies": side[:, R["allies"]],
            "attackers": side[:, R["attackers"]], "away share": away / np.maximum(troops + away, 1)}
    return cols


def padded(games, names):
    """Stacks games' feature columns into X [games, sides, k] with a mask of real sides."""
    n = max(len(g[0]["tiles"]) for g in games)
    X = np.zeros((len(games), n, len(names)))
    mask = np.zeros((len(games), n), dtype=bool)
    for j, (cols, _, _) in enumerate(games):
        k = len(cols["tiles"])
        X[j, :k] = np.stack([cols[f] for f in names], axis=1)
        mask[j, :k] = True
    return X, mask, np.array([g[1] for g in games])


def probs(X, mask, b):
    z = np.where(mask, X @ b, -np.inf)
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fit(X, mask, w, l2=1.0):
    """Conditional logit, one winner per game: P(side i wins) = softmax over the game's sides of x_i . b. Newton's
    method with step halving and a ridge (a normal prior of SD 1 / sqrt(l2) on each coefficient, which keeps a
    separating input finite); returns b and its covariance."""
    k = X.shape[2]
    b, rows = np.zeros(k), np.arange(len(w))

    def ll(b):
        return np.log(np.maximum(probs(X, mask, b)[rows, w], 1e-300)).sum() - l2 / 2 * b @ b

    for _ in range(100):
        p = probs(X, mask, b)
        xbar = np.einsum("gs,gsk->gk", p, X)
        g = (X[rows, w] - xbar).sum(0) - l2 * b
        H = np.einsum("gs,gsk,gsl->kl", p, X, X) - xbar.T @ xbar + l2 * np.eye(k)
        step, before = np.linalg.solve(H, g), ll(b)
        for _ in range(20):
            if ll(b + step) >= before:
                break
            step /= 2
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b, np.linalg.inv(H)


def held_out(X, mask, w, folds):
    """Each game's probabilities from the model fit on the other folds."""
    p = np.zeros(mask.shape)
    for f in range(5):
        test = folds == f
        b, _ = fit(X[~test], mask[~test], w[~test])
        p[test] = probs(X[test], mask[test], b)
    return p


def scores(p, mask, w):
    """McFadden's pseudo-R2 against a uniform pick among the alive sides, the expected calibration error over 10 bins,
    and the favourite's mean probability against its win rate."""
    rows = np.arange(len(w))
    ll = np.log(np.maximum(p[rows, w], 1e-12)).sum()
    ll0 = -np.log(mask.sum(1)).sum()
    y = np.zeros(mask.shape)
    y[rows, w] = 1
    pp, yy = p[mask], y[mask]
    bins = np.minimum((pp * 10).astype(int), 9)
    ece = sum(abs(pp[bins == i].sum() - yy[bins == i].sum()) for i in range(10)) / len(pp)
    fav = p.argmax(1)
    return 1 - ll / ll0, ece, p[rows, fav].mean(), (fav == w).mean()


def rank_stats(X, mask, w):
    """Per game, with the measure alone: the winner's AUC against the other alive sides and whether it leads (ties
    split); means over games."""
    rows = np.arange(len(w))
    xw = X[rows, w][:, None]
    others = mask.copy()
    others[rows, w] = False
    auc = ((X < xw) + 0.5 * (X == xw)) * others
    auc = auc.sum(1) / np.maximum(others.sum(1), 1)
    lead = np.where(mask, X, -np.inf).max(1)
    ties = ((X == lead[:, None]) & mask).sum(1)
    top = (xw[:, 0] == lead) / ties
    return auc.mean(), auc.std() / np.sqrt(len(w)), top.mean()


def land_loss(tr, m, snaps):
    """Humans alive at minute m of a game still running AHEAD minutes later: their tile share, troops per tile, share of
    the army away, rivals attacking, alliances, and whether they lost 10% of their land or more by m + AHEAD."""
    if m + AHEAD not in snaps:
        return []
    now, later = snaps[m], snaps[m + AHEAD]
    alive = now[:, R["tiles"]] > 0
    total = now[alive, R["tiles"]].sum()
    rows = []
    for i in np.flatnonzero(alive & np.array([s[1] == "HUMAN" for s in tr["sides"]])):
        tiles, troops, away = now[i, R["tiles"]], now[i, R["troops"]], now[i, R["away"]]
        rows.append((tiles / total, troops / tiles, away / max(troops + away, 1), now[i, R["attackers"]],
                     now[i, R["allies"]], later[i, R["tiles"]] < 0.9 * tiles))
    return rows


def win(_args):
    cells = collections.defaultdict(list)  # (mode, minute) -> games: (features of the alive sides, winner, id)
    losses = collections.defaultdict(list)  # (mode, minute) -> land_loss rows
    at = sorted(set(MINUTES) | {m + AHEAD for m in LOSS_AT} | set(LOSS_AT))
    bad = collections.Counter()
    for tr in traced():
        mode = tr["mode"]
        if "error" in tr or tr["desync"] is not None:
            bad["error" if "error" in tr else "desync"] += 1
            continue
        snaps, w = sides_at(tr, mode in TEAM_MODES, at)
        if w is None:
            bad["winner mismatch"] += 1
            continue
        if not snaps:
            continue  # over within a minute of the spawn phase
        # who can win: every team, or in the FFA modes every human (no nation won a public FFA game of the store)
        can = np.ones(len(snaps[min(snaps)]), dtype=bool) if mode in TEAM_MODES else \
            np.array([s[1] == "HUMAN" for s in tr["sides"]])
        for m, side in snaps.items():
            if m not in MINUTES:
                continue
            alive = side[:, R["tiles"]] > 0
            cand = alive & can
            if cand[w] and cand.sum() >= 2:
                cols = {f: v[cand[alive]] for f, v in features(side[alive]).items()}  # shares among all alive sides
                cells[mode, m].append((cols, int(cand[:w].sum()), tr["id"]))
        players = snaps if mode not in TEAM_MODES else sides_at(tr, False, at)[0]
        for m in LOSS_AT:
            losses[mode, m] += land_loss(tr, m, players)
    print("unusable traces:", dict(bad) or "none")
    results = {}
    for mode in MODES:
        res = {}
        for m in MINUTES:
            games = cells[mode, m]
            if len(games) < 20:
                continue
            folds = np.array([zlib.crc32(g[2].encode()) % 5 for g in games])
            cell = res[m] = {"games": len(games), "sides": float(np.mean([len(g[0]["tiles"]) for g in games])),
                             "single": {}, "models": {}}
            for f in ["earned", *FULL]:
                ok = [g for g in games if np.isfinite(g[0][f]).all()]
                if len(ok) < 20:
                    continue
                X, mask, w = padded(ok, [f])
                if X[mask].std() == 0:
                    continue
                auc, auc_se, top = rank_stats(X[:, :, 0], mask, w)
                okf = np.array([zlib.crc32(g[2].encode()) % 5 for g in ok])
                r2, ece, fav_p, fav_w = scores(held_out(X, mask, w, okf), mask, w)
                cell["single"][f] = {"games": len(ok), "auc": auc, "auc_se": auc_se, "lead_wins": top, "r2": r2,
                                     "ece": ece, "fav_p": fav_p, "fav_wins": fav_w, "beta": fit(X, mask, w)[0][0]}
            X, mask, w = padded(games, FULL)
            sd = X[mask].std(0)
            keep = [f for f, v in zip(FULL, sd) if v > 0]
            Z = np.where(mask[:, :, None], (X - X[mask].mean(0)) / np.where(sd > 0, sd, 1), 0)
            for name, fs in [("land", ["tiles"]), ("land + kills", ["tiles", "kills"]),
                             ("land + troops", ["tiles", "troops"]), ("all", keep)]:
                fs = [f for f in fs if f in keep]
                cols = [FULL.index(f) for f in fs]
                r2, ece, fav_p, fav_w = scores(held_out(Z[:, :, cols], mask, w, folds), mask, w)
                b, cov = fit(Z[:, :, cols], mask, w)
                cell["models"][name] = {"r2": r2, "ece": ece, "fav_p": fav_p, "fav_wins": fav_w,
                                        "coef": {f: [b[i], np.sqrt(cov[i, i])] for i, f in enumerate(fs)}}
        for m in LOSS_AT:
            if len(losses[mode, m]) >= 20:
                res[f"lost@{m}"] = loss_table(np.array(losses[mode, m], dtype=float))
        if res:
            results[mode] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "win.json").write_text(json.dumps(results, indent=1, default=float))
    report(results)


def loss_table(rows):
    """P(losing 10% of the land or more in the next AHEAD minutes) by the user's FFA risks, with counts."""
    share, density, away, attackers, allies, lost = rows.T
    big = share >= np.quantile(share, 2 / 3)
    lo, hi = np.quantile(density[big], [1 / 3, 2 / 3]) if big.any() else (0, 0)
    groups = {"all": np.ones(len(rows), bool), "big, thin": big & (density <= lo), "big, thick": big & (density >= hi),
              "army away 0": away == 0, "army away 0-30%": (away > 0) & (away <= 0.3), "army away >30%": away > 0.3,
              "attacked by 0": attackers == 0, "attacked by 1": attackers == 1, "attacked by 2+": attackers >= 2,
              "no alliance": allies == 0, "allied": allies > 0}
    out = {g: [float(lost[s].mean()) if s.any() else None, int(s.sum())] for g, s in groups.items()}
    # all the risks at once: a logistic regression (a conditional logit of two options, lost or not) on standardized
    # inputs; its SEs ignore that players of one game are not independent
    F = np.column_stack([np.log(share), np.log(np.maximum(density, 1e-3)), away, attackers, allies])
    names = ["log tile share", "log troops per tile", "army away", "attackers", "alliances"]
    sd = F.std(0)
    names = [n for n, v in zip(names, sd) if v > 0]
    Z = (F[:, sd > 0] - F[:, sd > 0].mean(0)) / sd[sd > 0]
    X = np.zeros((len(rows), 2, len(names) + 1))
    X[:, 0, 0], X[:, 0, 1:] = 1, Z
    b, cov = fit(X, np.ones((len(rows), 2), dtype=bool), np.where(lost > 0, 0, 1))
    out["logit"] = {n: [b[i], np.sqrt(cov[i, i])] for i, n in enumerate(["intercept", *names])}
    return out


def report(results):
    for mode, res in results.items():
        print(f"\n## {mode}")
        rows = []
        for m, c in res.items():
            if not isinstance(m, int):
                continue
            s, mo = c["single"], c["models"]
            best = max(s, key=lambda f: s[f]["r2"])
            rows.append([m, c["games"], f"{c['sides']:.1f}", f"{s['tiles']['lead_wins']:.2f}",
                         f"{s['tiles']['auc']:.3f} ± {s['tiles']['auc_se']:.3f}", f"{s['tiles']['r2']:.3f}",
                         f"{s['tiles']['beta']:.2f}",
                         f"{best} {s[best]['r2']:.3f}", f"{mo['land + kills']['r2'] - mo['land']['r2']:+.3f}",
                         f"{mo['all']['r2']:.3f}", f"{mo['all']['fav_p']:.2f} / {mo['all']['fav_wins']:.2f}",
                         f"{mo['all']['ece']:.3f}"])
        print(table(rows, ["min", "games", "sides", "land leader wins", "land AUC", "land R2", "land β",
                           "best single (R2)", "kills add (R2)", "all R2", "favourite p / wins", "ECE"]))
        for m, c in res.items():
            if isinstance(m, int):
                print(f"minute {m}: " + ", ".join(f"{f} {v['r2']:.3f} (AUC {v['auc']:.2f})"
                                                  for f, v in c["single"].items()))
        coef = {m: c["models"]["all"]["coef"] for m, c in res.items() if isinstance(m, int)}
        if coef:
            fs = list(dict.fromkeys(f for c in coef.values() for f in c))
            print(table([[f, *(f"{c[f][0]:+.2f} ± {c[f][1]:.2f}" if f in c else "" for c in coef.values())]
                         for f in fs], ["all: coefficient per SD", *(f"min {m}" for m in coef)]))
        for key in [k for k in res if isinstance(k, str)]:
            groups = {g: v for g, v in res[key].items() if g != "logit"}
            print(key + ": " + ", ".join(f"{g} {v[0]:.2f} ({v[1]})" for g, v in groups.items() if v[0] is not None))
            logit = res[key]["logit"].items()
            print("  logit per SD: " + ", ".join(f"{n} {b:+.2f} ± {se:.2f}" for n, (b, se) in logit))


def selftest(_args):
    """Checks the conditional logit and the rank statistics on simulated games of 4 to 6 sides."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 6, 1))
    mask = np.ones((3000, 6), dtype=bool)
    mask[:, 4:] = rng.random((3000, 2)) < 0.5
    w = np.array([rng.choice(6, p=row) for row in probs(X, mask, np.array([1.5]))])
    b, cov = fit(X, mask, w)
    assert abs(b[0] - 1.5) < 4 * np.sqrt(cov[0, 0]), (b, cov)
    auc, _, top = rank_stats((np.arange(6) == w[:, None]).astype(float), mask, w)
    assert auc == 1 and top == 1, (auc, top)
    print(f"ok: b = {b[0]:.3f} ± {np.sqrt(cov[0, 0]):.3f} (true 1.5)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("play")
    t = sub.add_parser("traces")
    t.add_argument("--per-mode", type=int, default=600, help="records sampled per mode (default 600)")
    t.add_argument("--workers", type=int, default=12)
    sub.add_parser("win")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # the tables' ± and β, also into a file
    {"play": play, "traces": traces, "win": win, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
