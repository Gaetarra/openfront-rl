"""Vectorised OpenFront games for training.

Node workers (bridge/worker.ts, protocol described there) run K games each and send per-cell frame deltas; the
observation tensors are assembled on the GPU from the state kept here.
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
UNIT = np.dtype([("type", "u1"), ("level", "u1"), ("owner", "<u2"), ("cell", "<u4"), ("troops", "<f4")])
assert CELL.itemsize == 11 and UNIT.itemsize == 12

# Spatial channels (tile counts are divided by the tiles per cell)
LAND, SHORE, OCEAN, HEIGHT, VALID = 0, 1, 2, 3, 4
OWN = 5  # 5..8: land owned by self, friendly, enemy, bot (relation codes 1..4 from the worker)
UNOWNED, DENSITY, FALLOUT = 9, 10, 11
UNITS = 12  # 12..21: (city, port, other structure, transport, warship) x (own, other)
N_CHANNELS = 22
UNIT_GROUP = {"City": 0, "Port": 1, "Factory": 2, "Missile Silo": 2, "SAM Launcher": 2, "Defense Post": 2,
              "Transport": 3, "Warship": 4}
DENSITY_FEATURE = 16  # player feature: log troops per tile

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
                 full_frames=False):
        self.device = torch.device(device)
        self.G = games_per_worker
        self.E = workers * games_per_worker
        self.full_frames = full_frames
        self.episodes = list(episodes) if episodes is not None else [0] * self.E
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.workers = [Worker(w, Path(log_dir) / f"worker{w}.log" if log_dir is not None else None)
                        for w in range(workers)]
        for w, wk in enumerate(self.workers):
            wk.send({"cmd": "init", "config": config, "games": self.G, "seed": seed, "worker": w,
                     "episodes": self.episodes[w * self.G:(w + 1) * self.G]})
        replies = [wk.recv() for wk in self.workers]
        d = replies[0][0]["dims"]
        self.H, self.W = d["grid"]  # padded by the worker to multiples of 16 (the CNN downsamples by 16)
        self.P, self.S = d["slots"], d["seats"]
        self.F, self.K, self.T = d["playerFeatures"], d["scalars"], d["acts"]
        self.N = self.E * self.S
        self.dims = {**d, "channels": N_CHANNELS, "H": self.H, "W": self.W}
        dev, hw = self.device, self.H * self.W
        self.unit_group = torch.tensor([UNIT_GROUP.get(t, -1) for t in d["unitTypes"]], device=dev)
        self.transport = d["unitTypes"].index("Transport")
        self.terrain = torch.zeros(self.E, 5, hw, device=dev)  # land, shore, ocean, height, inside the map
        self.area = torch.full((self.E,), 64.0, device=dev)  # tiles per cell of each game
        self.o1 = torch.zeros(self.E * hw, dtype=torch.long, device=dev)
        self.o2 = torch.zeros_like(self.o1)
        self.n1 = torch.zeros(self.E * hw, device=dev)
        self.n2 = torch.zeros_like(self.n1)
        self.fo = torch.zeros_like(self.n1)
        self.obs = self._absorb(replies)[0]

    def step(self, actions):
        """actions: int array [N, 4] of [type, cell, target, bin]; cells are grid indexes (row x W + column)."""
        a = np.asarray(actions, dtype=np.int64).reshape(self.E, self.S, 4)
        for w, wk in enumerate(self.workers):
            wk.send({"cmd": "step", "actions": a[w * self.G:(w + 1) * self.G].tolist(), "full": self.full_frames})
        out = self._absorb([wk.recv() for wk in self.workers])
        self.obs = out[0]
        return out

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
        cells, units, infos, resets = [], [], [], []
        e = 0
        for header, games in replies:
            for g, b in zip(header["games"], games):
                if g.get("done"):
                    dones[e] = 1
                    rewards[e] = g["rewards"]
                    infos += [{"env": e, "seat": s, "reward": g["rewards"][s], "rank": g["ranks"][s],
                               "players": g["players"], "ticks": g["endTick"]} for s in range(S)]
                self.episodes[e] = g["episode"]
                if g.get("reset"):
                    resets.append(e)
                    terr = np.frombuffer(b["terrain"], np.uint8).reshape(5, -1)
                    self.terrain[e].zero_()
                    self.terrain[e][:, :terr.shape[1]] = torch.from_numpy(terr.astype(np.float32)).to(self.device)
                    self.area[e] = float(g["cellSize"]) ** 2
                invalid[e] = g["invalid"]
                cells.append((e, np.frombuffer(b["cells"], CELL)))
                units.append((e, np.frombuffer(b["units"], UNIT)))
                players[e] = np.frombuffer(b["players"], np.float32).reshape(S, P, F)
                rel[e] = np.frombuffer(b["rel"], np.uint8).reshape(S, P)
                scalars[e] = np.frombuffer(b["scalars"], np.float32).reshape(S, K)
                types[e] = np.frombuffer(b["types"], np.uint8).reshape(S, T)
                attack[e] = np.frombuffer(b["attack"], np.uint8).reshape(S, P)
                e += 1
        self._apply_cells(resets, cells)
        obs = self._observe(players, rel, scalars, types, attack, units)
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
        idx = torch.from_numpy(env * hw + rec["idx"].astype(np.int64)).to(dev)
        for name, t in (("o1", self.o1), ("o2", self.o2), ("n1", self.n1), ("n2", self.n2), ("fo", self.fo)):
            t[idx] = torch.from_numpy(rec[name].astype(np.int64 if name[0] == "o" else np.float32)).to(dev)

    def _observe(self, players, rel, scalars, types, attack, units):
        N, S, P, hw, dev = self.N, self.S, self.P, self.H * self.W, self.device
        row_env = torch.arange(N, device=dev) // S
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
        dens = pl[..., DENSITY_FEATURE]
        for o, n in ((self.o1, self.n1), (self.o2, self.n2)):
            owner = o[flat]
            frac = n[flat] / area
            r = torch.gather(rel_t, 1, owner)
            own.scatter_add_(1, (r - 1).clamp(min=0).unsqueeze(1), (frac * (r > 0)).unsqueeze(1))
            x[:, DENSITY] += frac * torch.gather(dens, 1, owner)
        x[:, OWN:OWN + 4] = own
        x[:, UNOWNED] = (x[:, LAND] - own.sum(1)).clamp(min=0)
        x[:, FALLOUT] = self.fo[flat] / area
        self._add_units(x, units, rel_t)
        cell_mask = torch.stack([x[:, UNOWNED] > 0,
                                 (x[:, SHORE] > 0) & (x[:, LAND] - x[:, OWN] > 1e-6),
                                 x[:, OWN] > 0], 1) & valid[:, None]
        return {
            "spatial": x.view(N, N_CHANNELS, self.H, self.W).to(torch.bfloat16),
            "scalars": torch.from_numpy(scalars.reshape(N, self.K)).to(dev),
            "players": pl,
            "type_mask": torch.from_numpy(types.reshape(N, self.T)).to(dev),
            "attack_mask": torch.from_numpy(attack.reshape(N, P)).to(dev),
            "cell_mask": cell_mask,
        }

    def _add_units(self, x, units, rel_t):
        rec = np.concatenate([u for _, u in units])
        if len(rec) == 0:
            return
        dev = self.device
        env = torch.from_numpy(np.concatenate([np.full(len(u), e, np.int64) for e, u in units])).to(dev)
        typ = torch.from_numpy(rec["type"].astype(np.int64)).to(dev)
        owner = torch.from_numpy(rec["owner"].astype(np.int64)).to(dev)
        cell = torch.from_numpy(rec["cell"].astype(np.int64)).to(dev)
        troops = torch.from_numpy(rec["troops"].astype(np.float32)).to(dev)
        group = self.unit_group[typ]
        value = torch.where(typ == self.transport, torch.log1p(troops) / 10, torch.ones_like(troops))
        for s in range(self.S):
            row = env * self.S + s
            r = rel_t[row, owner]
            keep = (group >= 0) & (r > 0)
            ch = UNITS + 2 * group + (r != 1).long()
            x.index_put_((row[keep], ch[keep], cell[keep]), value[keep], accumulate=True)
