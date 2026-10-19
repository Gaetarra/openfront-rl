import collections
import gzip
import json
import unittest
import unittest.mock

import numpy as np
import torch

from trainer import env as env_module
from trainer.env import (BUILD_OF, BUILDING_TYPES, BUILDINGS, DST, FFA, HEALTH, HVN, LAND, LEVEL, MAP_TYPES, MOVE,
                         N_CHANNELS, N_SCALARS, OWN, OWNER, OWNER_FEATURES, PATROL, RANKED_1V1, ROOT, TRADE, UNIT,
                         UPGRADABLE, VALID, VETERAN, WORKER_V4_FEATURES, ZONE_KINDS, VecEnv, disk_cells,
                         frozen_eval_config, gate_config, quantize_)
from scripts.coverage import label_kinds
from trainer.eval import Scripted
from trainer.model import ACT_NAMES, DIPLOMACY, DONATE, MAP_OF, REQUEST, ActionDist, RandomPolicy, v1_spatial

ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "timerMinutes": 5, "seats": 1}
FIXTURE = ROOT / "bridge" / "testdata" / "ranked1v1.json.gz"
UNIT_GROUP_V1 = {"City": 0, "Port": 1, "Factory": 2, "Missile Silo": 2, "SAM Launcher": 2, "Defense Post": 2,
                 "Transport": 3, "Warship": 4}


def random_legal(obs, rng):
    """Uniformly random legal actions from the masks, as padded-grid [type, cell, target, bin] rows: a diplomatic
    action aims at a player of its row of diplomacy_mask (a donation sends 1-100% of the troops), a warship move
    goes from a cell of its source map to a cell of the destination map, alone or with its fleet."""
    rows = []
    types, attack, cells, diplomacy = (obs[k].cpu().numpy() for k in ("type_mask", "attack_mask", "cell_mask",
                                                                      "diplomacy_mask"))
    for i in range(types.shape[0]):
        t = rng.choice(np.flatnonzero(types[i]))
        m = int(MAP_OF[t])
        cell = rng.choice(np.flatnonzero(cells[i, m])) if m >= 0 and cells[i, m].any() else 0
        target, troops = 0, rng.integers(5)
        if t == 2:
            target = rng.choice(np.flatnonzero(attack[i]))
        elif t in DIPLOMACY:
            target = rng.choice(np.flatnonzero(diplomacy[i, t - DIPLOMACY[0]]))
            troops = rng.integers(1, 101) if t == DONATE else 0
        elif t == MOVE:
            target, troops = rng.choice(np.flatnonzero(cells[i, DST])), rng.integers(2)
        rows.append([t, cell, target, troops])
    return np.array(rows, np.int64)


def watch_observations(env, keep):
    """Wraps env._observe so keep(args, obs) sees every observation with its inputs, while the env's cell state still
    matches them."""
    observe = env._observe

    def spy(*args):
        obs = observe(*args)
        keep(args, obs)
        return obs

    env._observe = spy


def phase2_spatial(env, players, rel, units):
    """The 22 channels Phase 2's VecEnv._observe built (copied from it at commit b6c05ea), to check v1_spatial."""
    N, S, P, hw, dev = env.N, env.S, env.P, env.H * env.W, env.device
    row_env = torch.arange(N, device=dev) // S
    area = env.area[row_env][:, None]
    rel_t = torch.from_numpy(rel.reshape(N, P).astype(np.int64)).to(dev)
    pl = torch.from_numpy(players.reshape(N, P, env.F)).to(dev)
    x = torch.zeros(N, 22, hw, device=dev)
    terr = env.terrain[row_env]
    x[:, 0:3] = terr[:, 0:3] / area[:, None]
    x[:, 3] = terr[:, 3] / 31
    x[:, 4] = (terr[:, 4] > 0).float()
    flat = row_env[:, None] * hw + torch.arange(hw, device=dev)
    own = torch.zeros(N, 4, hw, device=dev)
    for o, n in ((env.o1, env.n1), (env.o2, env.n2)):
        owner = o[flat]
        frac = n[flat] / area
        r = torch.gather(rel_t, 1, owner)
        own.scatter_add_(1, (r - 1).clamp(min=0).unsqueeze(1), (frac * (r > 0)).unsqueeze(1))
        x[:, 10] += frac * torch.gather(pl[..., 16], 1, owner)
    x[:, 5:9] = own
    x[:, 9] = (x[:, 0] - own.sum(1)).clamp(min=0)
    x[:, 11] = env.fo[flat] / area
    rec = np.concatenate([u for _, u in units])
    if len(rec):
        group_of = torch.tensor([UNIT_GROUP_V1.get(t, -1) for t in env.dims["unitTypes"]], device=dev)
        transport = env.dims["unitTypes"].index("Transport")
        e = torch.from_numpy(np.concatenate([np.full(len(u), i, np.int64) for i, u in units])).to(dev)
        typ = torch.from_numpy(rec["type"].astype(np.int64)).to(dev)
        owner = torch.from_numpy(rec["owner"].astype(np.int64)).to(dev)
        cell = torch.from_numpy(rec["cell"].astype(np.int64)).to(dev)
        troops = torch.from_numpy(rec["troops"].astype(np.float32)).to(dev)
        group = group_of[typ]
        value = torch.where(typ == transport, torch.log1p(troops) / 10, torch.ones_like(troops))
        for s in range(S):
            row = e * S + s
            r = rel_t[row, owner]
            keep = (group >= 0) & (r > 0)
            ch = 12 + 2 * group + (r != 1).long()
            x.index_put_((row[keep], ch[keep], cell[keep]), value[keep], accumulate=True)
    return x.view(N, 22, env.H, env.W)


def new_channels(units, rel, names, side, H, W):
    """One seat's channels from LEVEL on, from the unit records by their definitions in trainer/env.py."""
    want = np.zeros((N_CHANNELS - LEVEL, H * W))
    cy, cx = np.divmod(np.arange(H * W), W)
    ours = lambda r: r in (1, 2)  # self or a friend
    for u in units:
        name, r = names[u["type"]], int(rel[u["owner"]])
        if name == "Trade Ship":  # paying the seat or its friends; capturable: an enemy's, not safe, bound for a stranger
            bound = int(rel[u["bound"]])
            if ours(r) or ours(bound):
                want[TRADE - LEVEL, u["cell"]] += 1 / 16
            elif r == 3 and not ours(bound) and not u["flags"] & 1:
                want[TRADE + 1 - LEVEL, u["cell"]] += 1 / 16
        if r == 0:
            continue
        if name in BUILDING_TYPES[:6]:  # structures, one per cell at most
            want[0, u["cell"]] += u["level"] / 16
            want[1, u["cell"]] += u["ready"] / 16
        if name == "Warship":
            theirs = int(r >= 3)
            want[HEALTH + theirs - LEVEL, u["cell"]] += u["troops"] / 4
            want[VETERAN + theirs - LEVEL, u["cell"]] += u["ready"] / 12
            if r == 1:
                want[PATROL - LEVEL, (int(u["zy"]) // side) * W + int(u["zx"]) // side] += 1 / 16
        kind = ZONE_KINDS.get(name)
        if kind is not None and u["zr"] > 0:
            x, y, radius = int(u["zx"]), int(u["zy"]), int(u["zr"])
            near = ((2 * cx + 1) * side - 2 * x) ** 2 + ((2 * cy + 1) * side - 2 * y) ** 2 <= (2 * radius) ** 2
            want[2 + 2 * kind + int(r >= 3), near] += u["ready"] / 16 if kind == 0 else 1
    return want


class VecEnvTest(unittest.TestCase):
    def test_observations_during_random_play(self):
        env = VecEnv(ONION, workers=1, games_per_worker=2, seed="t")
        try:
            obs = env.obs
            self.assertEqual(obs["spatial"].shape, (2, N_CHANNELS, 64, 64))
            self.assertEqual(obs["spatial"].dtype, torch.uint8)
            self.assertEqual(obs["players"].shape, (2, 5, 31))
            self.assertEqual(env.dims["scalars"], N_SCALARS)
            self.assertEqual(obs["scalars"].shape, (2, N_SCALARS))
            self.assertEqual((obs["owners"].shape, obs["owners"].dtype), ((2, 2, 64 * 64), torch.int16))
            self.assertEqual(obs["owner_land"].dtype, torch.uint8)
            self.assertEqual(obs["cell_mask"].shape, (2, DST + 1, 64 * 64))
            self.assertTrue(obs["cell_mask"][:, 0].any(), "spawn cells exist at the start")
            rng = np.random.default_rng(0)
            for _ in range(80):
                obs, rew, done, infos, invalid = env.step(random_legal(obs, rng))
                x = obs["spatial"].float() / 64
                self.assertTrue(torch.isfinite(x).all())
                self.assertLessEqual(float(x[:, LAND:OWN + 4].max()), 1.0 + 1e-3)
            self.assertGreater(float(obs["spatial"][:, OWN].float().sum()), 0, "the seats own land")
        finally:
            env.close()

    def test_a_cell_action_is_legal_only_with_a_legal_cell(self):
        env = VecEnv(ONION, workers=1, games_per_worker=2, seed="masks")
        try:
            policy = Scripted()
            for _ in range(80):  # the spawn phase and some 25 s after it
                obs = env.obs
                self.assertEqual(obs["type_mask"].shape[1], len(ACT_NAMES))
                self.assertEqual(obs["cell_mask"].shape[1], DST + 1)
                self.assertEqual(obs["retreat_mask"].shape, obs["attack_mask"].shape)
                self.assertFalse((obs["type_mask"][:, list(MAP_TYPES)] & ~obs["cell_mask"][:, :DST].any(-1)).any())
                self.assertFalse((obs["type_mask"][:, MOVE] & ~obs["cell_mask"][:, DST].any(-1)).any())
                env.step(policy(obs).cpu().numpy())
            own = obs["spatial"][:, OWN].flatten(1) > 0
            self.assertTrue((obs["cell_mask"][:, 9] & own).any(), "the atom map keeps own cells: self-nukes stay legal")
        finally:
            env.close()

    def test_the_observation_carries_v5_masks_and_inputs(self):
        env = VecEnv(ONION, workers=1, games_per_worker=1, seed="v5")
        try:
            obs, rng = env.obs, np.random.default_rng(2)
            for _ in range(3):
                obs = env.step(random_legal(obs, rng))[0]
            self.assertEqual(obs["spatial"].shape, (1, 64, env.H, env.W))
            self.assertEqual(obs["players"].shape, (1, env.P, 31))
            self.assertEqual(obs["cell_mask"].shape, (1, 14, env.H * env.W))
            self.assertEqual(obs["diplomacy_mask"].shape, (1, 4, env.P))
            self.assertEqual(obs["diplomacy_mask"].dtype, torch.bool)
            self.assertEqual(obs["type_mask"].shape, (1, 20))
        finally:
            env.close()

    def test_random_v5_play_drops_no_diplomatic_action(self):
        """The spec's smoke games: a random v5 policy plays 300 decisions on each setup through real workers."""
        for config in (RANKED_1V1, FFA, HVN):
            env = VecEnv(config, workers=1, games_per_worker=1, seed="v5-random")
            try:
                policy, sent, dropped = RandomPolicy(env.dims).cuda(), collections.Counter(), collections.Counter()
                for _ in range(300):
                    obs = env.obs
                    act = ActionDist(policy(obs), obs).sample()
                    invalid = env.step(act.cpu().numpy())[4]
                    for t, bad in zip(act[:, 0].tolist(), invalid):
                        sent[t] += 1
                        dropped[t] += int(bad)
                for t in DIPLOMACY:
                    self.assertEqual(dropped[t], 0, (config["playlist"], t))
                self.assertGreater(sent[REQUEST], 0, config["playlist"])
                if config is HVN:
                    self.assertGreater(sent[DONATE], 0)
                self.assertLessEqual(dropped[MOVE], 0.1 * max(sent[MOVE], 1))
            finally:
                env.close()

    def test_deltas_match_full_frames(self):
        a = VecEnv(ONION, 1, 2, seed="same")
        b = VecEnv(ONION, 1, 2, seed="same", full_frames=True)
        try:
            rng = np.random.default_rng(1)
            for _ in range(80):
                act = random_legal(a.obs, rng)
                a.step(act)
                b.step(act)
            for k in ("o1", "o2", "n1", "n2", "fo"):
                self.assertTrue(torch.equal(getattr(a, k), getattr(b, k)), k)
            self.assertTrue(torch.equal(a.obs["spatial"], b.obs["spatial"]))
        finally:
            a.close()
            b.close()

    def test_game_end_and_restart(self):
        env = VecEnv({**ONION, "timerMinutes": 1}, 1, 1, seed="end")
        try:
            noop = np.zeros((1, 4), np.int64)
            for _ in range(60):
                obs, rew, done, infos, invalid = env.step(noop)
                if done[0]:
                    break
            self.assertEqual(done[0], 1)
            self.assertEqual(rew[0], -1)
            self.assertEqual(infos[0]["rank"], 4)
            self.assertIs(infos[0]["cut"], False)
            self.assertEqual(env.episodes, [1])
            self.assertTrue(obs["cell_mask"][0, 0].any(), "the new game starts in its spawn phase")
        finally:
            env.close()

    def test_dead_worker_raises_with_log_path(self):
        env = VecEnv(ONION, 1, 1, seed="dead")
        try:
            env.workers[0].proc.kill()
            env.workers[0].proc.wait()
            with self.assertRaisesRegex(RuntimeError, "worker 0"):
                env.step(np.zeros((1, 4), np.int64))
        finally:
            env.close()

    def test_ranked_games_share_one_padded_grid(self):
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=2, seed="grid")
        try:
            obs = env.obs
            self.assertEqual(obs["spatial"].shape, (4, N_CHANNELS, 192, 256))
            self.assertEqual(obs["players"].shape[1], 403)
            valid = obs["spatial"][:, VALID].float().flatten(1) > 0
            self.assertTrue(set(valid.sum(1).tolist()) <= {47000, 37500, 31250})  # the ranked maps' cells
            self.assertFalse((obs["cell_mask"] & ~valid[:, None]).any(), "no legal cell outside the map")
            self.assertTrue(set(env.area.tolist()) <= {16.0, 64.0})
            env.step(np.zeros((4, 4), np.int64))
        finally:
            env.close()

    def test_replay_stream_labels_every_seat(self):
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=1, seed="rep",
                     records=[ROOT / "bridge" / "testdata" / "ranked1v1.json.gz"])
        try:
            seen = collections.Counter()
            for _ in range(200):
                self.assertEqual(tuple(env.labels.shape), (2, 4))
                seen.update(env.labels[:, 0].tolist())
                env.step()
            self.assertGreater(seen[0], seen[2], "no-ops outnumber attacks")
            self.assertEqual(sorted(env.outcome[0].tolist()), [-1.0, 1.0])
            self.assertEqual(len(env.seats_of[0]), 2)
        finally:
            env.close()

    def test_the_coverage_script_counts_the_commands_the_extractor_labels(self):
        kinds = label_kinds(json.loads(gzip.decompress(FIXTURE.read_bytes())))
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=1, seed="coverage", records=[FIXTURE])
        try:
            passes, commands, outside = 0, 0, 0
            while passes < 2:  # a slot's first record starts at a random window, the next one at window 0
                if passes == 1:
                    t = env.labels[:, 0]
                    commands += int((t != 0).sum())
                    outside += int((t == -1).sum())
                passes += sum("record" in i for i in env.step()[3])
        finally:
            env.close()
        self.assertEqual((commands, outside), (len(kinds), sum(k != "in" for k in kinds)))
        self.assertEqual(outside, 2, "the fixture's emoji and quick chat")

    def test_a_mixed_runs_ffa_game_matches_the_same_game_played_alone(self):
        ffa1, hvn1 = {**FFA, "timerMinutes": 1}, {**HVN, "timerMinutes": 1}
        mixed, alone = VecEnv([RANKED_1V1, ffa1, hvn1], 3, 1, seed="mix"), VecEnv(ffa1, 2, 1, seed="mix")
        try:
            self.assertEqual(mixed.seats.tolist(), [2, 4, 4])
            self.assertEqual(mixed.first.tolist(), [0, 2, 6])
            self.assertEqual(mixed.row_env.tolist(), [0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
            self.assertEqual(mixed.N, 10)
            self.assertIsNone(mixed.S)
            self.assertEqual(alone.S, 4)
            self.assertEqual(mixed.obs["spatial"].shape, (10, N_CHANNELS, 192, 256))
            self.assertEqual(mixed.obs["players"].shape[1], 477)
            rows = lambda obs, lo, hi: {k: v[lo:hi] for k, v in obs.items()}
            in_mixed, in_alone = Scripted(seed=0), Scripted(seed=0)
            for step in range(400):  # the 1-minute FFA game restarts twice; no-op seats end their game at once
                a, b = np.zeros((10, 4), np.int64), np.zeros((8, 4), np.int64)
                a[2:6] = in_mixed(rows(mixed.obs, 2, 6)).cpu().numpy()
                b[4:8] = in_alone(rows(alone.obs, 4, 8)).cpu().numpy()
                mixed.step(a)
                alone.step(b)
                if (step + 1) % 50 == 0:
                    for k, v in rows(mixed.obs, 2, 6).items():
                        self.assertTrue(torch.equal(v, alone.obs[k][4:8]), f"{k} at step {step + 1}")
            self.assertGreaterEqual(mixed.episodes[1], 2, "the FFA game restarted")
            self.assertEqual(mixed.episodes[1], alone.episodes[1])
            self.assertTrue(all(e >= 1 for e in mixed.episodes), "every game restarted, at its own time")
        finally:
            mixed.close()
            alone.close()

    def test_warship_and_trade_channels_follow_their_definitions(self):
        env = VecEnv.__new__(VecEnv)  # one game of 2 seats on a 4 x 4 grid, as in test_units_paint_every_seat...
        env.device = torch.device("cpu")
        env.E, env.N, env.P, env.H, env.W = 1, 2, 4, 4, 4
        env.seats, env.first = np.array([2]), np.array([0])
        env.building = torch.tensor([BUILDING_TYPES.index("Warship"), -1])
        env.zone_kind, env.transport, env.trade = torch.tensor([-1, -1]), -1, 1
        env.area = torch.full((1,), 64.0)
        rel = torch.tensor([[0, 1, 3, 3], [0, 3, 1, 3]])  # row 0 is player 1, row 1 is player 2; player 3 an enemy of both
        units = np.zeros(3, UNIT)
        units["type"], units["owner"], units["cell"] = (0, 1, 1), (1, 2, 2), (5, 6, 7)
        units["troops"][0], units["ready"][0], units["zx"][0], units["zy"][0] = 0.5, 2, 11, 19  # patrol tile in cell 9
        units["bound"][1:] = 3, 1  # one ship bound for player 3's port, one for player 1's
        x = torch.zeros(2, N_CHANNELS, 16)
        env._add_units(x, [(0, units)], rel, torch.ones(2, UPGRADABLE, dtype=torch.bool))
        near = lambda got, want: self.assertAlmostEqual(got.item(), want, places=6)
        near(x[0, HEALTH, 5], 0.5 / 4)
        near(x[0, VETERAN, 5], 2 / 12)
        near(x[0, PATROL, 9], 1 / 16)
        near(x[0, TRADE + 1, 6], 1 / 16)  # an enemy's ship bound for an enemy: capturable
        near(x[0, TRADE, 7], 1 / 16)  # an enemy's ship bound for row 0's port: it pays row 0
        near(x[1, HEALTH + 1, 5], 0.5 / 4)  # the same warship is the enemy's for row 1
        near(x[1, PATROL].sum(), 0)  # and not its patrol
        near(x[1, TRADE, 6] + x[1, TRADE, 7], 2 / 16)  # row 1's own ships

    def test_units_paint_every_seat_of_their_own_game(self):
        env = VecEnv.__new__(VecEnv)  # the attributes _add_units reads
        env.device = torch.device("cpu")
        env.E, env.N, env.P, env.H, env.W = 2, 6, 8, 4, 4
        env.seats, env.first = np.array([2, 4]), np.array([0, 2])
        env.building, env.zone_kind, env.transport = torch.tensor([0, -1]), torch.tensor([-1, -1]), 1
        env.trade = -1  # no trade ships here
        env.area = torch.full((2,), 64.0)
        rel = torch.zeros(6, 8, dtype=torch.long)
        rel[4, 5], rel[[2, 3, 5], 5] = 1, 3  # player 5: Self for row 4, Enemy for the rest of game 1
        rel[2:, 7] = 3  # player 7 plays in game 0: game 1's rows must not paint its city
        units = np.zeros(2, UNIT)
        units["level"], units["owner"], units["cell"] = 1, (5, 7), (9, 3)
        x = torch.zeros(6, N_CHANNELS, 16)
        afford = torch.ones(6, UPGRADABLE, dtype=torch.bool)
        buildings, upgradable = env._add_units(x, [(1, units[:1]), (0, units[1:])], rel, afford)
        want = torch.zeros_like(x)
        for row, r in ((2, 3), (3, 3), (4, 1), (5, 3)):
            want[row, BUILDINGS + 4 * 0 + r - 1, 9] = 1  # a city (kind 0) by the row's own relation
            want[row, LEVEL, 9] = 1 / 16
        self.assertTrue(torch.equal(x, want))
        self.assertEqual(upgradable.nonzero().tolist(), [[4, 9]])
        self.assertEqual(buildings[:, 5, 0].tolist(), [0, 1])
        self.assertEqual(buildings[:, 7, 0].tolist(), [1, 0])


class ConfigTest(unittest.TestCase):
    def test_frozen_evaluations_and_gates_derive_from_the_playlists(self):
        self.assertEqual(frozen_eval_config("ranked1v1"), RANKED_1V1)
        self.assertEqual(HVN["difficulty"], "Medium,Hard")
        hvn = frozen_eval_config("hvn")
        self.assertEqual(hvn, {**HVN, "difficulty": "Hard"}, "trained on Medium or Hard, judged on Hard")
        ffa = frozen_eval_config("ffa")
        self.assertEqual((ffa["seats"], ffa["difficulty"], ffa["timerMinutes"]), (1, "Hard", 30))
        self.assertEqual(gate_config("ffa", "heldout"), {**ffa, "timerMinutes": 0, "maps": "heldout"})
        self.assertIsNot(frozen_eval_config("hvn"), env_module.PLAYLISTS["hvn"], "a copy, not the playlist itself")
        with unittest.mock.patch.dict(env_module.PLAYLISTS, {"hvn": {**HVN, "seats": 2}}):
            self.assertEqual(gate_config("hvn", "seen")["seats"], 2)


class QuantizeTest(unittest.TestCase):
    def test_values_are_rounded_to_64ths_and_saturate(self):
        q = quantize_(torch.tensor([0.0, 0.004, 0.0079, 0.5, 1.0, 3.99, 5.0]))
        self.assertEqual(q.dtype, torch.uint8)
        self.assertEqual(q.tolist(), [0, 0, 1, 32, 64, 255, 255])  # 5.0 saturates: it does not wrap to 64


class DiskTest(unittest.TestCase):
    def test_a_disk_covers_the_cells_whose_centres_lie_within_its_radius(self):
        H, W = 6, 8
        long = lambda *v: torch.tensor(v, dtype=torch.long, device="cuda")
        disks = [(4, 4, 8, 8), (0, 0, 4, 4)]  # x, y, radius, tiles per cell side: a Normal map's, a Compact corner's
        n, cell = disk_cells(*(long(*v) for v in zip(*disks)), H, W)
        want = {(k, c) for k, (x, y, r, s) in enumerate(disks) for c in range(H * W)
                if ((c % W + 0.5) * s - x) ** 2 + ((c // W + 0.5) * s - y) ** 2 <= r * r}
        self.assertEqual({(int(a), int(c)) for a, c in zip(n, cell)}, want)
        self.assertIn((0, 1), want, "a centre exactly 8 tiles away is inside")
        self.assertEqual(len(disk_cells(*(long() for _ in range(4)), H, W)[0]), 0, "no disks, no cells")


class ChannelTest(unittest.TestCase):
    """One replay of the fixture record, checked against the channel definitions and against the Phase 2 code."""

    @classmethod
    def setUpClass(cls):
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=1, seed="channels", records=[FIXTURE])
        cls.seen, cls.legacy_error = None, 0.0
        cls.upgrade = np.zeros(3, int)  # wrong cells, open cells, own buildings left shut
        cls.ships = np.zeros(N_CHANNELS - HEALTH, int)  # observations with each warship and trade channel set
        cls.new_error, cls.zones = 0, np.zeros(3, int)  # worst error from LEVEL on; observations with each zone kind
        structures = [env.dims["unitTypes"].index(t) for t in BUILDING_TYPES[:6]]

        def keep(args, obs):
            players, rel, _, types, _, _, units, _ = args
            old = phase2_spatial(env, players, rel, units).to(torch.bfloat16).float()
            new = v1_spatial(obs["spatial"]).to(torch.bfloat16).float() / 64  # what PolicyV1 reads
            cls.legacy_error = max(cls.legacy_error, float((new - old).abs().max()))
            rec = np.concatenate([u for _, u in units])
            for seat in range(2):  # the upgrade map: own, finished buildings whose type the seat can afford
                want, shut = np.zeros(env.H * env.W, bool), np.zeros(env.H * env.W, bool)
                for u in rec:
                    name = env.dims["unitTypes"][u["type"]]
                    if name in BUILDING_TYPES[:UPGRADABLE] and rel[seat][u["owner"]] == 1:
                        ok = u["level"] > 0 and types[seat][BUILD_OF[BUILDING_TYPES.index(name)]] > 0
                        (want if ok else shut)[u["cell"]] = True
                got = obs["cell_mask"][seat, MAP_TYPES.index(13)].cpu().numpy()
                cls.upgrade += [int((got != want).sum()), int(want.sum()), int((shut & ~want).sum())]
            n = int(np.isin(rec["type"], structures).sum())
            if cls.seen is None or n >= cls.seen["structures"]:  # keep the observation with the most structures
                cls.seen = {"structures": n, "players": players.copy(), "rel": rel.copy(), "units": rec.copy(),
                            "spatial": obs["spatial"].cpu().numpy(),
                            "features": obs["players"][:, :, WORKER_V4_FEATURES:WORKER_V4_FEATURES + len(BUILDING_TYPES)].cpu().numpy(),
                            "cells": [t[:env.H * env.W].cpu().numpy() for t in (env.o1, env.n1, env.o2, env.n2)],
                            "area": float(env.area[0])}
            side = round(float(env.area[0]) ** 0.5)
            for seat in range(2):
                want = new_channels(rec, rel[seat], env.dims["unitTypes"], side, env.H, env.W)
                got = obs["spatial"][seat, LEVEL:].reshape(N_CHANNELS - LEVEL, -1).cpu().numpy().astype(np.int64)
                cls.new_error = max(cls.new_error, int(np.abs(got - np.clip(np.round(want * 64), 0, 255)).max()))
                cls.zones += want[2:8].reshape(3, 2, -1).any((1, 2))
                cls.ships += want[HEALTH - LEVEL:].any(1)

        try:
            watch_observations(env, keep)
            for _ in range(560):  # past the end of the record (552 windows), whatever window it started at
                env.step()
            cls.names, cls.hw, cls.P = env.dims["unitTypes"], env.H * env.W, env.P
        finally:
            env.close()

    def test_old_networks_get_the_phase2_inputs(self):
        self.assertLessEqual(self.legacy_error, 1 / 32)  # 1/64 rounding of up to 3 summed channels, then bf16

    def test_the_upgrade_map_opens_exactly_the_upgradable_buildings(self):
        wrong, opened, shut = self.upgrade
        self.assertEqual(wrong, 0)
        self.assertGreater(opened, 0, "a building the seat could upgrade was seen")
        self.assertGreater(shut, 0, "an own building it could not upgrade (unfinished or unaffordable) was seen")

    def test_owner_and_building_channels_follow_their_definitions(self):
        s = self.seen
        self.assertGreaterEqual(s["structures"], 5, "the record's structures were seen")
        o1, n1, o2, n2 = s["cells"]
        for seat in range(2):
            players, rel = s["players"][seat], s["rel"][seat]
            want = np.zeros((N_CHANNELS, self.hw))
            for o, n in ((o1, n1), (o2, n2)):
                share = n / s["area"] * (rel[o] > 0)
                for i, f in enumerate(OWNER_FEATURES):
                    want[OWNER + i] += share * players[o, f]
            levels = np.zeros((self.P, len(BUILDING_TYPES)))
            for u in s["units"]:
                name = self.names[u["type"]]
                if name in BUILDING_TYPES:
                    k = BUILDING_TYPES.index(name)
                    value = np.log1p(u["troops"]) / 10 if name == "Transport" else max(int(u["level"]), 1)
                    want[BUILDINGS + 4 * k + int(rel[u["owner"]]) - 1, u["cell"]] += value
                    levels[u["owner"], k] += value
            got = s["spatial"][seat].reshape(N_CHANNELS, -1)[OWNER:LEVEL].astype(np.int64)
            self.assertLessEqual(np.abs(got - np.clip(np.round(want[OWNER:LEVEL] * 64), 0, 255)).max(), 1)
            np.testing.assert_allclose(s["features"][seat], np.log1p(levels) / 3, atol=1e-5)

    def test_level_missile_and_zone_channels_follow_their_definitions(self):
        self.assertEqual(self.new_error, 0)  # 1/16 steps and whole counts are exact in 64ths
        self.assertGreater(self.zones[1], 0, "a defense post's zone was seen")
        self.assertGreater(self.zones[2], 0, "a nuke in flight was seen")
        self.assertGreater(self.ships[TRADE - HEALTH], 0, "a trade ship that pays a seat was seen")


class TeamChannelTest(unittest.TestCase):
    TEAM = ROOT / "bridge" / "testdata" / "team.json.gz"

    def test_warship_and_trade_channels_follow_their_definitions_on_real_ships(self):
        """The ranked fixture has no warships: the Team record has them from window ~80 on, and trade ships from the
        start. Checks a window in 20 of the first 200, for all 18 seats."""
        config = {"map": "Between Two Seas", "compact": True, "nations": 0, "bots": 100, "difficulty": "Medium",
                  "timerMinutes": 0, "seats": 18}
        env = VecEnv(config, workers=1, games_per_worker=1, seed="team", records=[self.TEAM])
        error, ships, windows = 0, np.zeros(N_CHANNELS - HEALTH, int), []

        def keep(args, obs):
            nonlocal error, ships
            players, rel, _, types, _, _, units, _ = args
            windows.append(None)
            if len(windows) % 20:
                return
            rec = np.concatenate([u for _, u in units])
            side = round(float(env.area[0]) ** 0.5)
            for seat in range(env.S):
                want = new_channels(rec, rel[seat], env.dims["unitTypes"], side, env.H, env.W)
                got = obs["spatial"][seat, LEVEL:].reshape(N_CHANNELS - LEVEL, -1).cpu().numpy().astype(np.int64)
                error = max(error, int(np.abs(got - np.clip(np.round(want * 64), 0, 255)).max()))
                ships += want[HEALTH - LEVEL:].any(1)

        try:
            watch_observations(env, keep)
            while len(windows) < 200:
                env.step()
        finally:
            env.close()
        self.assertEqual(error, 0)
        # no warship has a veteran level yet in these windows, so the veteran channels stay 0 here (the synthetic units of
        # frame.test.ts cover them)
        for channel, what in ((HEALTH, "an own warship"), (HEALTH + 1, "a hostile warship"), (PATROL, "an own patrol point"),
                              (TRADE, "a trade ship that pays a seat"), (TRADE + 1, "a capturable trade ship")):
            self.assertGreater(ships[channel - HEALTH], 0, f"{what} was seen: {ships}")


if __name__ == "__main__":
    unittest.main()
