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
                 ("ready", "u1"), ("pad", "u1"), ("zx", "<u2"), ("zy", "<u2"), ("zr", "<u2")])  # bridge/frame.ts
assert CELL.itemsize == 11 and UNIT.itemsize == 20

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
N_CHANNELS = ZONES + 2 * 3  # a SAM's zone counts its loaded interceptors / 16, a post's or a nuke's 1
DENSITY_FEATURE = 16  # player feature: log troops per tile
# The action type of each heat map (bridge/actions.ts Act): spawn, boat, the nine builds (city, defense post, silo,
# factory, port, SAM, warship, atom bomb, hydrogen bomb) and upgrade. Networks before 3B step 3 have the first three.
MAP_TYPES = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
STRUCTURES = 6  # BUILDING_TYPES[:6] are structures, at most one per cell
UPGRADABLE = 5  # BUILDING_TYPES[:5] (city, port, factory, silo, SAM) can be upgraded
BUILD_OF = (4, 8, 7, 6, 9)  # their build actions: an upgrade needs the gold to build another one (canUpgradeUnit)


def pinned(a, device):
    """A numpy array copied to `device` without waiting: a copy from pageable memory waits for everything queued on
    the GPU, and the observations made ~70 such copies a step (PLAN 4A, "Speed work")."""
    return torch.from_numpy(a).pin_memory().to(device, non_blocking=True)


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


# The Phase 1 gate setup, and ranked 1v1 games (the worker draws map, size, bots and timer from the official ranked 1v1
# playlist, bridge/setups.ts; of the other fields only "seats" is used).
ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "timerMinutes": 5, "seats": 1}
RANKED_1V1 = {"map": "Australia", "compact": False, "nations": 0, "difficulty": "Medium", "bots": 400,
              "timerMinutes": 15, "seats": 2, "playlist": "ranked1v1"}


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
    """E = workers x games_per_worker games with S seats each; N = E x S observation rows (env-major)."""

    def __init__(self, config, workers, games_per_worker, seed, device="cuda", episodes=None, log_dir=None,
                 full_frames=False, records=None, record=None):
        self.device = torch.device(device)
        self.G = games_per_worker
        self.E = workers * games_per_worker
        self.full_frames = full_frames
        self.replay = records is not None
        if self.replay and len(records) < workers:
            raise ValueError(f"{len(records)} records cannot feed {workers} workers")
        self.episodes = list(episodes) if episodes is not None else [0] * self.E
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.workers = [Worker(w, Path(log_dir) / f"worker{w}.log" if log_dir is not None else None)
                        for w in range(workers)]
        for w, wk in enumerate(self.workers):
            msg = {"cmd": "init", "config": config, "games": self.G, "seed": seed, "worker": w}
            if self.replay:
                msg["replay"] = {"records": [str(r) for r in records[w::workers]]}
            else:
                msg["episodes"] = self.episodes[w * self.G:(w + 1) * self.G]
                if record is not None:  # {"dir", "gitCommit", "games"}: save each worker's first games as records
                    msg["record"] = record
            wk.send(msg)
        replies = [wk.recv() for wk in self.workers]
        d = replies[0][0]["dims"]
        self.H, self.W = d["grid"]  # padded by the worker to multiples of 16 (the CNN downsamples by 16)
        self.P, self.S = d["slots"], d["seats"]
        self.F, self.K, self.T = d["playerFeatures"], d["scalars"], d["acts"]
        self.N = self.E * self.S
        self.dims = {**d, "channels": N_CHANNELS, "playerFeatures": self.F + len(BUILDING_TYPES), "H": self.H,
                     "W": self.W}
        dev, hw = self.device, self.H * self.W
        self.building = np.array([BUILDING_TYPES.index(t) if t in BUILDING_TYPES else -1 for t in d["unitTypes"]],
                                 np.int64)
        self.transport = d["unitTypes"].index("Transport")
        self.zone_kind = np.array([ZONE_KINDS.get(t, -1) for t in d["unitTypes"]], np.int64)
        self.map_types = torch.tensor(MAP_TYPES, device=dev)  # a Python list as an index is copied to the GPU each time
        self.terrain = torch.zeros(self.E, 5, hw, device=dev)  # land, shore, ocean, height, inside the map
        self.area = torch.full((self.E,), 64.0, device=dev)  # tiles per cell of each game
        self.side = np.full(self.E, 8, np.int64)  # and tiles per cell side
        self.o1 = torch.zeros(self.E * hw, dtype=torch.long, device=dev)
        self.o2 = torch.zeros_like(self.o1)
        self.n1 = torch.zeros(self.E * hw, device=dev)
        self.n2 = torch.zeros_like(self.n1)
        self.fo = torch.zeros_like(self.n1)
        # Replay mode: per game its record, players, their outcome (+1 winner, -1 otherwise) and windows left; per row
        # the label [type, cell, target, troops %] (type -1: a command outside the action set, -2: unmapped).
        self.labels = None
        self.outcome = np.zeros((self.E, self.S), np.float32)
        self.left = np.zeros(self.E, np.int64)
        self.record_of = [""] * self.E
        self.seats_of = [[] for _ in range(self.E)]
        self.obs = self._absorb(replies)[0]

    def send(self, actions=None):
        """Starts every worker on its next step (replay mode takes no actions); recv() collects it, so the caller can
        use the GPU meanwhile. actions: int array [N, 4] of [type, cell, target, troops %] with grid cell indexes, or [N, 8]
        for two actions per decision (the bandwidth probe, PLAN 3B)."""
        if self.replay:
            for wk in self.workers:
                wk.send({"cmd": "step"})
            return
        a = np.asarray(actions, dtype=np.int64).reshape(self.E, self.S, -1)  # 4 per action; the bandwidth probe sends 8
        for w, wk in enumerate(self.workers):
            wk.send({"cmd": "step", "actions": a[w * self.G:(w + 1) * self.G].tolist(), "full": self.full_frames})

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
        E, S, P, F, K, T = self.E, self.S, self.P, self.F, self.K, self.T
        rewards = np.zeros((E, S), np.float32)
        dones = np.zeros((E, S), np.float32)
        invalid = np.zeros((E, S), bool)
        players = np.empty((E, S, P, F), np.float32)
        rel = np.empty((E, S, P), np.uint8)
        scalars = np.empty((E, S, K), np.float32)
        types = np.empty((E, S, T), bool)
        attack = np.empty((E, S, P), bool)
        retreat = np.empty((E, S, P), bool)
        labels = np.zeros((E, S, 4), np.int64)
        cells, units, infos, resets = [], [], [], []
        e = 0
        for header, games in replies:
            for g, b in zip(header["games"], games):
                if g.get("done") and self.replay:
                    infos.append({"env": e, "record": g["finished"], "desync": g.get("desync", False),
                                  "error": g.get("error")})
                elif g.get("done"):
                    dones[e] = 1
                    rewards[e] = g["rewards"]
                    infos += [{"env": e, "seat": s, "reward": g["rewards"][s], "rank": g["ranks"][s],
                               "players": g["players"], "ticks": g["endTick"], "recorded": g.get("recorded")}
                              for s in range(S)]
                if self.replay:
                    labels[e] = np.frombuffer(b["labels"], np.int32).reshape(S, 4)
                    self.outcome[e] = g["outcome"]
                    self.left[e] = g["left"]
                    self.record_of[e] = g["record"]
                    self.seats_of[e] = g["seats"]
                self.episodes[e] = g["episode"]
                if g.get("reset"):
                    resets.append(e)
                    terr = np.frombuffer(b["terrain"], np.uint8).reshape(5, -1)
                    self.terrain[e].zero_()
                    self.terrain[e][:, :terr.shape[1]] = pinned(terr.astype(np.float32), self.device)
                    self.area[e] = float(g["cellSize"]) ** 2
                    self.side[e] = g["cellSize"]
                invalid[e] = g["invalid"]
                cells.append((e, np.frombuffer(b["cells"], CELL)))
                units.append((e, np.frombuffer(b["units"], UNIT)))
                players[e] = np.frombuffer(b["players"], np.float32).reshape(S, P, F)
                rel[e] = np.frombuffer(b["rel"], np.uint8).reshape(S, P)
                scalars[e] = np.frombuffer(b["scalars"], np.float32).reshape(S, K)
                types[e] = np.frombuffer(b["types"], np.uint8).reshape(S, T)
                attack[e] = np.frombuffer(b["attack"], np.uint8).reshape(S, P)
                retreat[e] = np.frombuffer(b["retreat"], np.uint8).reshape(S, P)
                e += 1
        if self.replay:
            self.labels = pinned(labels.reshape(-1, 4), self.device)
        self._apply_cells(resets, cells)
        obs = self._observe(players, rel, scalars, types, attack, retreat, units)
        return obs, rewards.reshape(-1), dones.reshape(-1), infos, invalid.reshape(-1)

    def _apply_cells(self, resets, cells):
        hw, dev = self.H * self.W, self.device
        for e in resets:
            for t in (self.o1, self.o2, self.n1, self.n2, self.fo):
                t[e * hw:(e + 1) * hw] = 0
        rec = np.concatenate([c for _, c in cells])
        if len(rec) == 0:
            return
        env = np.concatenate([np.full(len(c), e, np.int64) for e, c in cells])
        idx = pinned(env * hw + rec["idx"].astype(np.int64), dev)
        for name, t in (("o1", self.o1), ("o2", self.o2), ("n1", self.n1), ("n2", self.n2), ("fo", self.fo)):
            t[idx] = pinned(rec[name].astype(np.int64 if name[0] == "o" else np.float32), dev)

    def _observe(self, players, rel, scalars, types, attack, retreat, units):
        """The model's inputs, one row per seat:
        - "spatial": uint8 [N, N_CHANNELS, H, W], in 64ths;
        - "players": [N, P, F + len(BUILDING_TYPES)], the worker's features, then the player's buildings by type
          (log1p of their summed values, / 3);
        - "owners" (int16) and "owner_land" (uint8) [N, 2, H * W]: each cell's two largest owners and their shares of
          it in 64ths, for the model's per-player map features;
        - the scalars and the masks: 15 action types, 12 heat maps (MAP_TYPES), the attack and retreat pointers."""
        N, S, P, hw, dev = self.N, self.S, self.P, self.H * self.W, self.device
        row_env = torch.arange(N, device=dev) // S
        area = self.area[row_env][:, None]  # [N, 1]
        rel = rel.reshape(N, P).astype(np.int64)
        rel_t = pinned(rel, dev)
        pl = pinned(players.reshape(N, P, self.F), dev)
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
        types = types.reshape(N, self.T)
        type_mask = pinned(types, dev)
        buildings, upgradable = self._add_units(x, units, rel, types[:, list(BUILD_OF)])
        mine, shore = x[:, OWN] > 0, x[:, SHORE] > 0
        nukable = x[:, OWN + 1] < 0.5  # not mostly a friend's land; self-nukes stay legal (the user, 2026-10-08)
        cell_mask = torch.stack([
            x[:, UNOWNED] > 0, shore & (x[:, LAND] - x[:, OWN] > 1e-6),  # spawn, boat
            mine, mine, mine, mine, mine & shore, mine,  # city, defense post, silo, factory, port, SAM
            x[:, OCEAN] > 0, nukable, nukable, upgradable,  # warship, atom and hydrogen bombs, upgrade
        ], 1) & valid[:, None]
        type_mask[:, self.map_types] &= cell_mask.any(-1)  # a cell action needs a legal cell on its heat map
        return {
            "spatial": quantize_(x).view(N, N_CHANNELS, self.H, self.W),
            "scalars": pinned(scalars.reshape(N, self.K), dev),
            "players": torch.cat([pl, torch.log1p(buildings[row_env]) / 3], 2),
            "owners": torch.stack(owners, 1).to(torch.int16),
            "owner_land": quantize_(torch.stack(shares, 1)),
            "type_mask": type_mask,
            "attack_mask": pinned(attack.reshape(N, P), dev),
            "retreat_mask": pinned(retreat.reshape(N, P), dev),
            "cell_mask": cell_mask,
        }

    def _add_units(self, x, units, rel, afford):
        """Paints each seat's units into x, and returns every player's buildings by type [E, P, len(BUILDING_TYPES)].
        - By type and relation, a structure counts its level (under construction: 1), a transport its log troops / 10,
          a warship 1.
        - LEVEL and READY hold each structure's level (0 under construction) and its loaded missiles, / 16.
        - ZONES hold each zone (bridge/frame.ts) by kind and by its owner's side. A SAM's zone counts its loaded
          interceptors / 16, a post's or a nuke's 1.
        Also returns the upgrade map [N, H·W]: each seat's finished buildings of the types it can afford (afford
        [N, UPGRADABLE]), the game's canUpgradeUnit.
        rel [N, P] and afford are numpy arrays. The records are picked on the CPU, and only what is painted goes to the
        GPU: picking them there (a boolean index, the zones' cells) waited for it (PLAN 4A, "Speed work")."""
        buildings = torch.zeros(self.E, self.P, len(BUILDING_TYPES), device=self.device)
        upgradable = torch.zeros(self.N, self.H * self.W, dtype=torch.bool, device=self.device)
        rec = np.concatenate([u for _, u in units])
        if len(rec) == 0:
            return buildings, upgradable
        gpu = lambda a: pinned(np.ascontiguousarray(a), self.device)
        env = np.concatenate([np.full(len(u), e, np.int64) for e, u in units])
        typ, owner, cell = (rec[f].astype(np.int64) for f in ("type", "owner", "cell"))
        level, ready = rec["level"].astype(np.float32), rec["ready"].astype(np.float32)
        kind, zk = self.building[typ], self.zone_kind[typ]
        # computed on the GPU as before, so the values stay the same to the bit
        value = torch.where(gpu(typ == self.transport), torch.log1p(gpu(rec["troops"].astype(np.float32))) / 10,
                            gpu(level).clamp(min=1))
        keep = np.flatnonzero(kind >= 0)
        buildings.index_put_((gpu(env[keep]), gpu(owner[keep]), gpu(kind[keep])), value[gpu(keep)], accumulate=True)
        zone = np.flatnonzero((zk >= 0) & (rec["zr"] > 0))  # the records with a zone
        z = lambda f: torch.from_numpy(rec[f][zone].astype(np.int64))
        disk, zcell = (t.numpy() for t in disk_cells(z("zx"), z("zy"), z("zr"), torch.from_numpy(self.side[env[zone]]),
                                                     self.H, self.W))
        zi = zone[disk]  # the record behind each covered cell
        zvalue = np.where(zk[zi] == 0, ready[zi] / 16, np.float32(1)).astype(np.float32)
        for s in range(self.S):
            row = env * self.S + s
            r = rel[row, owner]
            k = keep[r[keep] > 0]
            x.index_put_((gpu(row[k]), gpu(BUILDINGS + 4 * kind[k] + r[k] - 1), gpu(cell[k])), value[gpu(k)],
                         accumulate=True)
            st = k[kind[k] < STRUCTURES]
            at = gpu(row[st]), gpu(cell[st])
            x[:, LEVEL].index_put_(at, gpu(level[st] / 16), accumulate=True)
            x[:, READY].index_put_(at, gpu(ready[st] / 16), accumulate=True)
            on = np.flatnonzero(r[zi] > 0)
            x.index_put_((gpu(row[zi[on]]), gpu(ZONES + 2 * zk[zi[on]] + (r[zi[on]] >= 3)), gpu(zcell[on])),
                         gpu(zvalue[on]), accumulate=True)
            up = k[(r[k] == 1) & (kind[k] < UPGRADABLE) & (level[k] > 0)]  # own, finished, upgradable
            up = up[afford[row[up], kind[up]]]  # and the gold to build another of its type
            upgradable[gpu(row[up]), gpu(cell[up])] = True
        return buildings, upgradable
