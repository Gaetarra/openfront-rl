"""Vectorised OpenFront games for training.

Node workers (bridge/worker.ts, protocol described there) run K games each and send per-cell frame deltas; the
observation tensors are assembled on the GPU from the state kept here. In replay mode the workers replay game records
instead, and every step also brings each seat's label (the command its human player issued) for behaviour cloning.
"""
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = ROOT / "OpenFront_Source_Code" / "openfrontio"
WORKER_TS = ROOT / "bridge" / "worker.ts"

CELL = np.dtype([("idx", "<u4"), ("o1", "<u2"), ("o2", "<u2"), ("n1", "u1"), ("n2", "u1"), ("fo", "u1")])
UNIT = np.dtype([("type", "u1"), ("level", "u1"), ("owner", "<u2"), ("cell", "<u4"), ("troops", "<f4"),
                 ("ready", "u1"), ("flags", "u1"), ("zx", "<u2"), ("zy", "<u2"), ("zr", "<u2"), ("bound", "<u2")])  # bridge/frame.ts
assert CELL.itemsize == 11 and UNIT.itemsize == 22


def game_commit():
    return subprocess.run(["git", "-C", str(GAME_DIR), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()

# Spatial channels. They reach the model as uint8 in 64ths of a unit (quantize_; trainer/model.py divides by 64): tile
# shares stay exact, other values move by at most 1/128, and PPO stores its rollouts as 8 bits. Tile counts are divided
# by the tiles per cell.
LAND, SHORE, OCEAN, HEIGHT, VALID = 0, 1, 2, 3, 4
OWN = 5  # 5..8: land owned by self, friendly, enemy, bot (relation codes 1..4 from the worker)
UNOWNED, DENSITY, FALLOUT = 9, 10, 11
OWNER = 12  # 12..16: the cell's owners' player features OWNER_FEATURES, weighted by their shares of its land
OWNER_FEATURES = (8, 9, 10, 13, 14)  # log troops, troops / max, log gold, troops attacking the seat, sent at it
BUILDINGS = 17  # 17..48: BUILDING_TYPES x relation (self, friendly, enemy, bot), see VecEnv._add_units
BUILDING_TYPES = ("City", "Port", "Factory", "Missile Silo", "SAM Launcher", "Defense Post", "Transport", "Warship")
LEVEL = BUILDINGS + 4 * len(BUILDING_TYPES)  # 49: the cell's structure's level / 16; 0 while under construction
READY = LEVEL + 1  # 50: the missiles loaded in the cell's silo or SAM / 16
ZONES = READY + 1  # 51..56: ZONES + 2 * kind + side (0 ours: self and friends; 1 theirs: enemies and bots)
ZONE_KINDS = {"SAM Launcher": 0, "Defense Post": 1, "Atom Bomb": 2, "Hydrogen Bomb": 2, "MIRV Warhead": 2}
HEALTH = ZONES + 2 * 3  # a SAM's zone counts its loaded interceptors / 16, a post's or a nuke's 1
# 57..60: HEALTH + side, VETERAN + side, by the warship's owner's side as for ZONES: the summed health / max health / 4,
# and the summed veteran levels / 12
VETERAN = HEALTH + 2
PATROL = VETERAN + 2  # 61: the seat's warships whose patrol tile lies in the cell / 16
TRADE = PATROL + 1  # 62: trade ships that pay the seat or its friends / 16; 63: the ones the seat could capture / 16
N_CHANNELS = TRADE + 2
# The mode input (bridge/obs.ts): scalars MODE.. hold the lobby kind one-hot in KINDS order, then log2(non-bot players) / 8,
# log2(teams) / 6 and the nations' difficulty.
MODE = 26
KINDS = ("ranked1v1", "ranked2v2", "rankedffa", "clanwars", "ffa", "team", "hvn")
N_SCALARS = MODE + len(KINDS) + 3
DENSITY_FEATURE = 16  # player feature: log troops per tile
WORKER_V4_FEATURES = 17  # the worker's player features before step 6; the buildings follow them, then its later ones
# The action type of each heat map (bridge/actions.ts Act): spawn, boat, the nine builds (city, defense post, silo,
# factory, port, SAM, warship, atom bomb, hydrogen bomb), upgrade and the warship moves' source. Networks before 3B step
# 3 have the first three, v4 the first twelve. Warship moves also take a destination cell, mask DST after the heat maps.
MAP_TYPES = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19)
DST = len(MAP_TYPES)
MOVE = 19
WARSHIP = BUILDING_TYPES.index("Warship")
STRUCTURES = 6  # BUILDING_TYPES[:6] are structures, at most one per cell
UPGRADABLE = 5  # BUILDING_TYPES[:5] (city, port, factory, silo, SAM) can be upgraded
BUILD_OF = (4, 8, 7, 6, 9)  # their build actions: an upgrade needs the gold to build another one (canUpgradeUnit)


def quantize_(x):
    """x in 64ths as uint8 (x itself is overwritten on the way). Values above 255/64 (a structure above level 3,
    several warships in one cell) saturate."""
    return x.mul_(64).round_().clamp_(0, 255).to(torch.uint8)


def disk_cells(zx, zy, zr, side, H, W):
    """The cells of an H x W grid that disks cover, as (disk index, cell index) pairs. A cell is in when its centre lies
    within zr tiles of the tile (zx, zy), by Euclidean distance. All int64 tensors; side is the tiles per cell side of
    each disk's game. The arithmetic stays in integers (positions doubled), so the edge is exact."""
    if len(zr) == 0:
        return zr, zr
    R = int(((zr + side - 1) // side).max()) + 1  # cells from a centre's cell to the farthest one inside
    # ponytail: one window size for every disk (the largest); chunk the disks if their memory ever matters
    d = torch.arange(-R, R + 1, device=zr.device)
    s = side[:, None, None]
    gy = (zy // side)[:, None, None] + d[:, None]  # [n, 2R + 1, 1]
    gx = (zx // side)[:, None, None] + d  # [n, 1, 2R + 1]
    dy, dx = (2 * gy + 1) * s - 2 * zy[:, None, None], (2 * gx + 1) * s - 2 * zx[:, None, None]
    near = (dx * dx + dy * dy <= (2 * zr[:, None, None]) ** 2) & (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    n, i, j = near.nonzero(as_tuple=True)
    return n, gy[n, i, 0] * W + gx[n, 0, j]


# The Phase 1 gate setup, and the public playlists' games (the worker draws map, size, bots and timer from the official
# playlist, bridge/setups.ts; of the other fields only "seats" is used, plus the ones the FFA and HVN configs set).
ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "timerMinutes": 5, "seats": 1}
RANKED_1V1 = {"map": "Australia", "compact": False, "nations": 0, "difficulty": "Medium", "bots": 400,
              "timerMinutes": 15, "seats": 2, "playlist": "ranked1v1"}
# 4N: 4 learner seats against the map's nations (FFA) or 4 nations (Humans vs Nations), Medium or Hard drawn per game,
# on "training" maps, with a 30-minute timer. Mirrors bridge/testkit.ts FFA and HVN, except HvN's Medium games: the
# public game's are Hard, and the start won 24 of 28 against Medium nations and 0 of 28 against Hard (PLAN 4N).
FFA = {**RANKED_1V1, "playlist": "ffa", "seats": 4, "maps": "training", "difficulty": "Medium,Hard", "timerMinutes": 30}
HVN = {**FFA, "playlist": "hvn"}
PLAYLISTS = {"ranked1v1": RANKED_1V1, "ffa": FFA, "hvn": HVN}


def frozen_eval_config(setup):
    """A frozen evaluation's games: one FFA seat, or the Humans vs Nations team, against Hard nations; ranked 1v1 as
    trained."""
    if setup == "ffa":
        return {**PLAYLISTS["ffa"], "seats": 1, "difficulty": "Hard"}
    if setup == "hvn":
        return {**PLAYLISTS["hvn"], "difficulty": "Hard"}
    return dict(PLAYLISTS[setup])


def gate_config(setup, maps):
    """The gate's games: the public rules (no timer) on `maps` ("heldout" or "seen")."""
    return {**frozen_eval_config(setup), "timerMinutes": 0, "maps": maps}


class Worker:
    """One Node process running K games (bridge/worker.ts)."""

    def __init__(self, index, log_path=None):
        self.index = index
        self.log_path = log_path
        self.log = open(log_path, "wb") if log_path is not None else subprocess.DEVNULL
        self.proc = subprocess.Popen([shutil.which("node"), "--import", "tsx", str(WORKER_TS)], cwd=GAME_DIR,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        try:  # the PC is shared: training yields to interactive programs
            below = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10)
            psutil.Process(self.proc.pid).nice(below)
        except psutil.Error:
            pass

    def send(self, msg):
        data = json.dumps(msg).encode()
        try:
            self.proc.stdin.write(struct.pack("<I", len(data)) + data)
            self.proc.stdin.flush()
        except OSError as e:
            raise self._died() from e

    def recv(self):
        """Returns (header, [dict name -> bytes per game])."""
        head = self.proc.stdout.read(4)
        if len(head) < 4:
            raise self._died()
        (size,) = struct.unpack("<I", head)
        body = self.proc.stdout.read(size)
        if len(body) < size:
            raise self._died()
        (hlen,) = struct.unpack_from("<I", body)
        header = json.loads(body[4:4 + hlen])
        off, games = 4 + hlen, []
        for g in header["games"]:
            blobs = {}
            for name, n in g["blobs"]:
                blobs[name] = body[off:off + n]
                off += n
            games.append(blobs)
        return header, games

    def _died(self):
        where = f"see {self.log_path}" if self.log_path else "run with log_dir to keep its stderr"
        return RuntimeError(f"worker {self.index} exited (code {self.proc.poll()}); {where}")

    def close(self):
        try:
            self.send({"cmd": "close"})
            self.proc.wait(timeout=10)
        except (RuntimeError, OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except OSError:  # unflushed bytes to a dead worker
                pass
        if self.log is not subprocess.DEVNULL:
            self.log.close()


class VecEnv:
    """E = workers x games_per_worker games; N = their seats in all, one observation row each. Rows are game-major: game
    e owns rows first[e] .. first[e] + seats[e] - 1. `config`: one setup for every worker, or a list with one per
    worker (the workers then share the grid and player table that fit all of them). S is the seats per game when
    every game has the same number, else None. Replay mode: `records` is a list of paths shared out among the workers,
    or a list with one list of paths per worker; a replay imitates config["seats"] players a record; `slots` sets the
    player table's rows (default: the config's)."""

    def __init__(self, config, workers, games_per_worker, seed, device="cuda", episodes=None, log_dir=None,
                 full_frames=False, records=None, record=None, slots=None):
        self.device = torch.device(device)
        self.G = games_per_worker
        self.E = workers * games_per_worker
        self.full_frames = full_frames
        self.replay = records is not None
        per_worker = self.replay and bool(records) and all(isinstance(r, (list, tuple)) for r in records)
        if per_worker and len(records) != workers:
            raise ValueError(f"{len(records)} record lists for {workers} workers")
        if self.replay and len(records) < workers:
            raise ValueError(f"{len(records)} records cannot feed {workers} workers")
        configs = list(config) if isinstance(config, list) else [config] * workers
        if len(configs) != workers:
            raise ValueError(f"{len(configs)} configs for {workers} workers")
        if self.replay and any(c != configs[0] for c in configs):
            raise ValueError("replay mode takes a single config")
        mix = [c for i, c in enumerate(configs) if c not in configs[:i]]  # the distinct configs of the run
        self.episodes = list(episodes) if episodes is not None else [0] * self.E
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.workers = [Worker(w, Path(log_dir) / f"worker{w}.log" if log_dir is not None else None)
                        for w in range(workers)]
        for w, wk in enumerate(self.workers):
            msg = {"cmd": "init", "config": configs[w], "mix": mix, "games": self.G, "seed": seed, "worker": w}
            if slots is not None:
                msg["slots"] = slots
            if self.replay:
                mine = records[w] if per_worker else records[w::workers]
                msg["replay"] = {"records": [str(r) for r in mine]}
            else:
                msg["episodes"] = self.episodes[w * self.G:(w + 1) * self.G]
                if record is not None:  # {"dir", "gitCommit", "games"}: save each worker's first games as records
                    msg["record"] = record
            wk.send(msg)
        replies = [wk.recv() for wk in self.workers]
        d = replies[0][0]["dims"]
        if any(r[0]["dims"][k] != d[k] for r in replies for k in ("grid", "slots")):
            raise ValueError("the workers' grids and player tables differ")
        self.H, self.W = d["grid"]  # padded by the worker to multiples of 16 (the CNN downsamples by 16)
        self.P = d["slots"]
        self.F, self.K, self.T = d["playerFeatures"], d["scalars"], d["acts"]
        if self.K != N_SCALARS:
            raise ValueError(f"the workers send {self.K} scalars, the trainer expects {N_SCALARS} (bridge/obs.ts SCALARS)")
        self.seats = np.repeat([r[0]["dims"]["seats"] for r in replies], self.G)  # [E]
        self.first = np.cumsum(self.seats) - self.seats  # each game's first row
        self.N = int(self.seats.sum())
        self.S = int(self.seats[0]) if (self.seats == self.seats[0]).all() else None
        dev, hw = self.device, self.H * self.W
        self.row_env = torch.from_numpy(np.repeat(np.arange(self.E), self.seats)).to(dev)  # [N]
        self.dims = {**{k: v for k, v in d.items() if k != "seats"}, "channels": N_CHANNELS,
                     "playerFeatures": self.F + len(BUILDING_TYPES), "H": self.H, "W": self.W}
        self.building = torch.tensor([BUILDING_TYPES.index(t) if t in BUILDING_TYPES else -1 for t in d["unitTypes"]],
                                     device=dev)
        self.transport = d["unitTypes"].index("Transport")
        self.trade = d["unitTypes"].index("Trade Ship")
        self.zone_kind = torch.tensor([ZONE_KINDS.get(t, -1) for t in d["unitTypes"]], device=dev)
        self.terrain = torch.zeros(self.E, 5, hw, device=dev)  # land, shore, ocean, height, inside the map
        self.area = torch.full((self.E,), 64.0, device=dev)  # tiles per cell of each game
        self.o1 = torch.zeros(self.E * hw, dtype=torch.long, device=dev)
        self.o2 = torch.zeros_like(self.o1)
        self.n1 = torch.zeros(self.E * hw, device=dev)
        self.n2 = torch.zeros_like(self.n1)
        self.fo = torch.zeros_like(self.n1)
        # Replay mode: per game its record, players, their outcome (+1 winner, -1 otherwise) and windows left; per row
        # the label [type, cell, target, troops %] (type -1: a command outside the action set, -2: unmapped).
        self.labels = None
        self.outcome = np.zeros((self.E, self.seats.max()), np.float32)
        self.left = np.zeros(self.E, np.int64)
        self.record_of = [""] * self.E
        self.seats_of = [[] for _ in range(self.E)]
        self.obs = self._absorb(replies)[0]

    def send(self, actions=None):
        """Starts every worker on its next step (replay mode takes no actions); recv() collects it, so the caller can
        use the GPU meanwhile. actions: int array [N, 4] of [type, cell, target, troops %] with grid cell indexes, or [N, 8]
        for two actions per decision (the bandwidth probe, PLAN 3B); each worker gets the rows of its games."""
        if self.replay:
            for wk in self.workers:
                wk.send({"cmd": "step"})
            return
        a = np.split(np.asarray(actions, dtype=np.int64).reshape(self.N, -1), self.first[1:])  # one array per game;
        for w, wk in enumerate(self.workers):  # 4 per action, the bandwidth probe sends 8
            games = [g.tolist() for g in a[w * self.G:(w + 1) * self.G]]
            wk.send({"cmd": "step", "actions": games, "full": self.full_frames})

    def recv(self):
        out = self._absorb([wk.recv() for wk in self.workers])
        self.obs = out[0]
        return out

    def step(self, actions=None):
        self.send(actions)
        return self.recv()

    def close(self):
        for wk in self.workers:
            wk.close()

    def _absorb(self, replies):
        N, P, F, K, T = self.N, self.P, self.F, self.K, self.T
        rewards = np.zeros(N, np.float32)
        dones = np.zeros(N, np.float32)
        invalid = np.zeros(N, bool)
        players = np.empty((N, P, F), np.float32)
        rel = np.empty((N, P), np.uint8)
        scalars = np.empty((N, K), np.float32)
        types = np.empty((N, T), bool)
        diplomacy = np.empty((N, 4, P), bool)
        attack = np.empty((N, P), bool)
        retreat = np.empty((N, P), bool)
        labels = np.zeros((N, 4), np.int64)
        cells, units, infos, resets = [], [], [], []
        e = 0
        for header, games in replies:
            for g, b in zip(header["games"], games):
                n = int(self.seats[e])
                seats = slice(self.first[e], self.first[e] + n)  # this game's rows
                if g.get("done") and self.replay:
                    infos.append({"env": e, "record": g["finished"], "desync": g.get("desync", False),
                                  "error": g.get("error")})
                elif g.get("done"):
                    dones[seats] = 1
                    rewards[seats] = g["rewards"]
                    infos += [{"env": e, "seat": s, "reward": g["rewards"][s], "rank": g["ranks"][s],
                               "players": g["players"], "ticks": g["endTick"], "cut": g["cut"],
                               "recorded": g.get("recorded")} for s in range(n)]
                if self.replay:
                    labels[seats] = np.frombuffer(b["labels"], np.int32).reshape(n, 4)
                    self.outcome[e] = g["outcome"]
                    self.left[e] = g["left"]
                    self.record_of[e] = g["record"]
                    self.seats_of[e] = g["seats"]
                self.episodes[e] = g["episode"]
                if g.get("reset"):
                    resets.append(e)
                    terr = np.frombuffer(b["terrain"], np.uint8).reshape(5, -1)
                    self.terrain[e].zero_()
                    self.terrain[e][:, :terr.shape[1]] = torch.from_numpy(terr.astype(np.float32)).to(self.device)
                    self.area[e] = float(g["cellSize"]) ** 2
                invalid[seats] = g["invalid"]
                cells.append((e, np.frombuffer(b["cells"], CELL)))
                units.append((e, np.frombuffer(b["units"], UNIT)))
                players[seats] = np.frombuffer(b["players"], np.float32).reshape(n, P, F)
                rel[seats] = np.frombuffer(b["rel"], np.uint8).reshape(n, P)
                scalars[seats] = np.frombuffer(b["scalars"], np.float32).reshape(n, K)
                types[seats] = np.frombuffer(b["types"], np.uint8).reshape(n, T)
                diplomacy[seats] = np.frombuffer(b["diplomacy"], np.uint8).reshape(n, 4, P)
                attack[seats] = np.frombuffer(b["attack"], np.uint8).reshape(n, P)
                retreat[seats] = np.frombuffer(b["retreat"], np.uint8).reshape(n, P)
                e += 1
        if self.replay:
            self.labels = torch.from_numpy(labels).to(self.device)
        self._apply_cells(resets, cells)
        obs = self._observe(players, rel, scalars, types, attack, retreat, units, diplomacy)
        return obs, rewards, dones, infos, invalid

    def _apply_cells(self, resets, cells):
        hw, dev = self.H * self.W, self.device
        for e in resets:
            for t in (self.o1, self.o2, self.n1, self.n2, self.fo):
                t[e * hw:(e + 1) * hw] = 0
        rec = np.concatenate([c for _, c in cells])
        if len(rec) == 0:
            return
        env = np.concatenate([np.full(len(c), e, np.int64) for e, c in cells])
        idx = torch.from_numpy(env * hw + rec["idx"].astype(np.int64)).to(dev)
        for name, t in (("o1", self.o1), ("o2", self.o2), ("n1", self.n1), ("n2", self.n2), ("fo", self.fo)):
            t[idx] = torch.from_numpy(rec[name].astype(np.int64 if name[0] == "o" else np.float32)).to(dev)

    def _observe(self, players, rel, scalars, types, attack, retreat, units, diplomacy):
        """The model's inputs, one row per seat:
        - "spatial": uint8 [N, N_CHANNELS, H, W], in 64ths;
        - "players": [N, P, F + len(BUILDING_TYPES)], the worker's first WORKER_V4_FEATURES features, the player's
          buildings by type (log1p of their summed values, / 3), then the worker's later features, so the networks
          that read only a prefix keep their inputs;
        - "owners" (int16) and "owner_land" (uint8) [N, 2, H * W]: each cell's two largest owners and their shares of
          it in 64ths, for the model's per-player map features;
        - the scalars and the masks: the action types, "cell_mask" [N, DST + 1, H * W] (a heat map per MAP_TYPES entry,
          then the warship moves' destinations), the attack and retreat pointers, and "diplomacy_mask" [N, 4, P]: the
          targets of an alliance request, a renewal, a break and a donation (bridge/actions.ts DIPLOMACY)."""
        N, P, hw, dev, row_env = self.N, self.P, self.H * self.W, self.device, self.row_env
        area = self.area[row_env][:, None]  # [N, 1]
        rel_t = torch.from_numpy(rel.reshape(N, P).astype(np.int64)).to(dev)
        pl = torch.from_numpy(players.reshape(N, P, self.F)).to(dev)
        x = torch.zeros(N, N_CHANNELS, hw, device=dev)
        terr = self.terrain[row_env]
        valid = terr[:, 4] > 0
        x[:, LAND:OCEAN + 1] = terr[:, 0:3] / area[:, None]
        x[:, HEIGHT] = terr[:, 3] / 31
        x[:, VALID] = valid.float()
        flat = row_env[:, None] * hw + torch.arange(hw, device=dev)
        own = torch.zeros(N, 4, hw, device=dev)
        owners, shares = [], []
        for o, n in ((self.o1, self.n1), (self.o2, self.n2)):
            owner = o[flat]
            frac = n[flat] / area
            r = torch.gather(rel_t, 1, owner)
            held = frac * (r > 0)  # owner 0 (no owner) holds nothing
            own.scatter_add_(1, (r - 1).clamp(min=0).unsqueeze(1), held.unsqueeze(1))
            x[:, DENSITY] += frac * torch.gather(pl[..., DENSITY_FEATURE], 1, owner)
            for i, f in enumerate(OWNER_FEATURES):
                x[:, OWNER + i] += held * torch.gather(pl[..., f], 1, owner)
            owners.append(owner)
            shares.append(frac)
        x[:, OWN:OWN + 4] = own
        x[:, UNOWNED] = (x[:, LAND] - own.sum(1)).clamp(min=0)
        x[:, FALLOUT] = self.fo[flat] / area
        type_mask = torch.from_numpy(types.reshape(N, self.T)).to(dev)
        buildings, upgradable = self._add_units(x, units, rel_t, type_mask[:, list(BUILD_OF)] > 0)
        mine, shore = x[:, OWN] > 0, x[:, SHORE] > 0
        nukable = x[:, OWN + 1] < 0.5  # not mostly a friend's land; self-nukes stay legal (the user, 2026-10-08)
        cell_mask = torch.stack([
            x[:, UNOWNED] > 0, shore & (x[:, LAND] - x[:, OWN] > 1e-6),  # spawn, boat
            mine, mine, mine, mine, mine & shore, mine,  # city, defense post, silo, factory, port, SAM
            x[:, OCEAN] > 0, nukable, nukable, upgradable,  # warship, atom and hydrogen bombs, upgrade
            x[:, BUILDINGS + 4 * WARSHIP] > 0,  # warship moves: the seat's own warships
            x[:, OCEAN] > 0,  # and the water they go to
        ], 1) & valid[:, None]
        type_mask[:, list(MAP_TYPES)] &= cell_mask[:, :DST].any(-1)  # a cell action needs a legal cell on its heat map
        type_mask[:, MOVE] &= cell_mask[:, DST].any(-1)
        return {
            "spatial": quantize_(x).view(N, N_CHANNELS, self.H, self.W),
            "scalars": torch.from_numpy(scalars.reshape(N, self.K)).to(dev),
            "players": torch.cat([pl[..., :WORKER_V4_FEATURES], torch.log1p(buildings[row_env]) / 3,
                                  pl[..., WORKER_V4_FEATURES:]], 2),
            "owners": torch.stack(owners, 1).to(torch.int16),
            "owner_land": quantize_(torch.stack(shares, 1)),
            "type_mask": type_mask,
            "attack_mask": torch.from_numpy(attack.reshape(N, P)).to(dev),
            "retreat_mask": torch.from_numpy(retreat.reshape(N, P)).to(dev),
            "cell_mask": cell_mask,
            "diplomacy_mask": torch.from_numpy(diplomacy.reshape(N, 4, P)).to(dev),
        }

    def _add_units(self, x, units, rel_t, afford):
        """Paints each seat's units into x (the rows of the unit's own game), and returns every player's buildings by
        type [E, P, len(BUILDING_TYPES)].
        - By type and relation, a structure counts its level (under construction: 1), a transport its log troops / 10,
          a warship 1.
        - LEVEL and READY hold each structure's level (0 under construction) and its loaded missiles, / 16.
        - HEALTH and VETERAN hold the warships' summed health / 4 and veteran levels / 12 by their owner's side (ours:
          self and friends, theirs: enemies and bots); PATROL the seat's own warships by patrol tile / 16.
        - TRADE holds the trade ships that pay the seat (their owner or destination port is its own or a friend's), and
          TRADE + 1 the ones it could capture (an enemy's, bound for a port that is neither its nor a friend's, not safe
          from pirates), / 16.
        - ZONES hold each zone (bridge/frame.ts) by kind and by its owner's side. A SAM's zone counts its loaded
          interceptors / 16, a post's or a nuke's 1.
        Also returns the upgrade map [N, H·W]: each seat's finished buildings of the types it can afford (afford
        [N, UPGRADABLE]), the game's canUpgradeUnit."""
        buildings = torch.zeros(self.E, self.P, len(BUILDING_TYPES), device=self.device)
        upgradable = torch.zeros(self.N, self.H * self.W, dtype=torch.bool, device=self.device)
        rec = np.concatenate([u for _, u in units])
        if len(rec) == 0:
            return buildings, upgradable
        dev = self.device
        env = torch.from_numpy(np.concatenate([np.full(len(u), e, np.int64) for e, u in units])).to(dev)
        typ = torch.from_numpy(rec["type"].astype(np.int64)).to(dev)
        owner = torch.from_numpy(rec["owner"].astype(np.int64)).to(dev)
        cell = torch.from_numpy(rec["cell"].astype(np.int64)).to(dev)
        level = torch.from_numpy(rec["level"].astype(np.float32)).to(dev)
        troops = torch.from_numpy(rec["troops"].astype(np.float32)).to(dev)
        ready = torch.from_numpy(rec["ready"].astype(np.float32)).to(dev)
        zr = torch.from_numpy(rec["zr"].astype(np.int64)).to(dev)
        flags = torch.from_numpy(rec["flags"].astype(np.int64)).to(dev)
        bound = torch.from_numpy(rec["bound"].astype(np.int64)).to(dev)
        side = self.area[env].sqrt().round().long()  # tiles per cell side of each record's game
        patrol = (torch.from_numpy(rec["zy"].astype(np.int64)).to(dev) // side * self.W
                  + torch.from_numpy(rec["zx"].astype(np.int64)).to(dev) // side)  # a warship's patrol cell
        zk = self.zone_kind[typ]
        zone = ((zk >= 0) & (zr > 0)).nonzero().squeeze(1)  # the records with a zone
        zx, zy = (torch.from_numpy(rec[f].astype(np.int64)).to(dev)[zone] for f in ("zx", "zy"))
        disk, zcell = disk_cells(zx, zy, zr[zone], side[zone], self.H, self.W)
        zi = zone[disk]  # the record behind each covered cell
        zvalue = torch.where(zk[zi] == 0, ready[zi] / 16, 1.0)
        kind = self.building[typ]
        value = torch.where(typ == self.transport, torch.log1p(troops) / 10, level.clamp(min=1))
        keep = kind >= 0
        buildings.index_put_((env[keep], owner[keep], kind[keep]), value[keep], accumulate=True)
        first, seats = (torch.as_tensor(a, dtype=torch.long, device=dev)[env] for a in (self.first, self.seats))
        friend = lambda rel: (rel == 1) | (rel == 2)  # the seat or its friends
        one = x.new_tensor(1 / 16)
        for s in range(int(self.seats.max())):
            row = (first + s).clamp(max=self.N - 1)  # a game with fewer seats has no seat s: r = 0 drops its records
            r = torch.where(seats > s, rel_t[row, owner], 0)
            k = keep & (r > 0)
            x.index_put_((row[k], BUILDINGS + 4 * kind[k] + r[k] - 1, cell[k]), value[k], accumulate=True)
            st = k & (kind < STRUCTURES)
            x[:, LEVEL].index_put_((row[st], cell[st]), level[st] / 16, accumulate=True)
            x[:, READY].index_put_((row[st], cell[st]), ready[st] / 16, accumulate=True)
            rz = r[zi]
            on = rz > 0
            x.index_put_((row[zi][on], ZONES + 2 * zk[zi][on] + (rz[on] >= 3).long(), zcell[on]), zvalue[on],
                         accumulate=True)
            rb = torch.where(seats > s, rel_t[row, bound], 0)  # the relation of a trade ship's destination port's owner
            ship = (kind == WARSHIP) & (r > 0)
            x.index_put_((row[ship], HEALTH + (r[ship] >= 3).long(), cell[ship]), troops[ship] / 4, accumulate=True)
            x.index_put_((row[ship], VETERAN + (r[ship] >= 3).long(), cell[ship]), ready[ship] / 12, accumulate=True)
            own = ship & (r == 1)
            x[:, PATROL].index_put_((row[own], patrol[own]), one, accumulate=True)
            trade = typ == self.trade
            pays = trade & (friend(r) | friend(rb))
            capture = trade & (r == 3) & ~friend(rb) & (flags & 1 == 0)
            x[:, TRADE].index_put_((row[pays], cell[pays]), one, accumulate=True)
            x[:, TRADE + 1].index_put_((row[capture], cell[capture]), one, accumulate=True)
            up = keep & (r == 1) & (kind < UPGRADABLE) & (level > 0)  # own, finished, upgradable
            up &= afford[row, kind.clamp(0, UPGRADABLE - 1)]  # and the gold to build another of its type
            upgradable[row[up], cell[up]] = True
        return buildings, upgradable
