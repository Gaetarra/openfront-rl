import argparse
import contextlib
import gzip
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trainer import bc, crawl
from trainer.bc import (Meter, backward_windows, bc_loss, family_of, is_val, label_legal, lr_at, main, measure, mix_of,
                        one_pass, score, slots_of, table, usable, val_records, worker_records)
from trainer.env import DST, MOVE, RANKED_1V1, ROOT, VecEnv, map_table
from trainer.model import (ACT_NAMES, ATTACK, BREAK, BUILD, DIPLOMACY, DONATE, MAP_OF, RENEW, REQUEST, RETREAT, SHARES,
                           NEW_TYPE_BIAS, V4_ACTS, ActionDist, Policy, PolicyV1, new_net)
from trainer.test_model import DIMS, DIMS_V5, N as ROWS, exact_fp32, fake_obs, randomized

FIXTURE = ROOT / "bridge" / "testdata" / "ranked1v1.json.gz"
TEAM = ROOT / "bridge" / "testdata" / "team.json.gz"
FIXTURE_SIZE =json.loads(gzip.decompress(FIXTURE.read_bytes()))["info"]["config"]["gameMapSize"].lower()  # its bucket


class LabelTest(unittest.TestCase):
    def test_label_legal(self):
        obs = {"type_mask": torch.ones(4, 5, dtype=torch.bool),
               "cell_mask": torch.zeros(4, 3, 16, dtype=torch.bool),
               "attack_mask": torch.zeros(4, 5, dtype=torch.bool), "retreat_mask": torch.zeros(4, 5, dtype=torch.bool)}
        obs["cell_mask"][3, 0, 5] = True
        obs["attack_mask"][:, 2] = True
        labels = torch.tensor([[-1, 0, 0, 0], [2, 0, 2, 1], [2, 0, 3, 1], [1, 5, 0, 0]])
        self.assertEqual(label_legal(obs, labels).tolist(), [False, True, False, True])
        obs["type_mask"][1, 2] = False
        self.assertFalse(bool(label_legal(obs, labels)[1]), "a type the mask forbids")

    def test_validation_split_is_stable(self):
        ids = [f"g{i:04d}" for i in range(2000)]
        val = [i for i in ids if is_val(i, 0.05)]
        self.assertTrue(60 < len(val) < 140)
        self.assertEqual(val, [i for i in ids if is_val(i, 0.05)])

    def test_the_control_page_says_what_is_training(self):
        a = bc.parse_args(["--workers", "14", "--games-per-worker", "6", "--updates", "10"])
        a.arch = "v3"
        info = bc.run_info(a, [entry("a"), entry("b"), entry("c", "team", teams=6)], [entry("v")])
        self.assertEqual(info["Method"], "behaviour cloning (imitating humans)")
        self.assertEqual(info["Network"], "player transformer, Soft-MoE core (8 experts), 100 troop shares")
        self.assertEqual(info["Starts from"], "random weights")
        self.assertEqual(info["Data"], "3 replays (ranked1v1 2, team 1), 1 held out")
        self.assertEqual(info["Games at once"], "84 (14 workers x 6)")
        self.assertEqual(info["Updates"], "10 of 8 windows each")

    def test_meter_averages_and_resets(self):
        m = Meter()
        m.add("a", torch.tensor(3.0), 2)
        m.add("a", torch.tensor(1.0), 2)
        self.assertEqual(m.means(), {"a": 1.0})
        self.assertEqual(m.means(), {})


def entry(game_id, mode="ranked1v1", **fields):
    """An index entry as the crawler writes it (trainer/crawl.py entry_of), on a map that fits the grid."""
    return {"id": game_id, "start": 0, "commit": None, "map": "Australia", "size": "Normal", "turns": 1,
            "players": [{"clientID": "a", "publicID": None}, {"clientID": "b", "publicID": None}], "winner": "a",
            "mode": mode, "teams": None, "modifiers": [], "disabled": [], "bots": 400, "nations": "disabled", **fields}


def paths(*ids):
    return [crawl.record_path(i, crawl.store()) for i in ids]


def paths_in(root, *ids):
    return [crawl.record_path(i, root) for i in ids]


class RecordsTest(unittest.TestCase):
    def test_the_family_of_an_entry(self):
        self.assertEqual(family_of(entry("a", "ranked2v2")), "ranked2v2")
        self.assertEqual(family_of(entry("a", "clanwars")), "team")
        self.assertEqual(family_of(entry("a", "team", teams="Humans Vs Nations")), "hvn")
        self.assertEqual(family_of(entry("a", "team", teams=6)), "team")

    def test_usable_records_fit_the_grid_and_set_no_modifier(self):
        t = map_table()
        ok = lambda **fields: usable(entry("a", **fields), t["maps"], t["grid"])
        self.assertTrue(ok(modifiers=[]))
        self.assertTrue(ok(modifiers=["isCompact"]))
        self.assertFalse(ok(modifiers=["isCompact", "isWaterNukes"]))
        self.assertFalse(ok(modifiers=["isWaterNukes"]))
        self.assertFalse(ok(disabled=["SAM Launcher"]))
        self.assertFalse(ok(map="Brain Rot"), "294 x 375 cells do not fit the 192 x 256 grid")
        self.assertFalse(ok(map="A map the game does not have"))

    def test_an_index_from_the_older_crawler_asks_for_a_reindex(self):
        t = map_table()
        old = {k: v for k, v in entry("a").items() if k != "teams"}
        with self.assertRaisesRegex(SystemExit, "--reindex"):
            usable(old, t["maps"], t["grid"])

    def test_mix_of_reads_families_in_the_order_given(self):
        a = bc.parse_args(["--workers", "9", "--mix", "team=2", "ranked1v1=3"])
        self.assertEqual(list(mix_of(a).items()), [("team", 2), ("ranked1v1", 3)])
        self.assertEqual(a.workers, 5, "the mix decides the workers")
        self.assertEqual(mix_of(bc.parse_args(["--workers", "9"])), {"ranked1v1": 9})
        for bad in (["nope=1"], ["team=1", "team=2"], ["team=0"], ["team"]):
            with self.assertRaisesRegex(SystemExit, "--mix"):
                mix_of(bc.parse_args(["--mix", *bad]))

    def test_each_family_is_dealt_to_its_workers(self):
        entries = [entry("c"), entry("a"), entry("t", "team", teams=6), entry("b")]
        self.assertEqual(worker_records(entries, {"ranked1v1": 2, "team": 1}),
                         [paths("a", "c"), paths("b"), paths("t")])

    def test_a_family_short_of_records_stops_the_run(self):
        with self.assertRaisesRegex(SystemExit, "hvn.*1|1.*hvn"):
            worker_records([entry("h", "team", teams="Humans Vs Nations"), entry("a")], {"hvn": 2})

    def test_validation_leaves_out_a_family_without_held_out_records(self):
        held = [entry("b"), entry("a")]
        self.assertEqual(val_records(held, {"ranked1v1": 2, "team": 1}), ([paths("a", "b")], ["team"]))

    def test_the_player_table_fits_the_largest_game(self):
        t = map_table()["maps"]
        players = lambda n: [{"clientID": str(i), "publicID": None} for i in range(n)]
        small = entry("a", players=players(2), bots=400, nations="disabled")
        big = entry("b", players=players(50), bots=400, nations="default", map="Alps")
        self.assertLess(t["Alps"]["nations"], 50)
        self.assertEqual(slots_of([small, big], t), 1 + 50 + 50 + 400, "as many nations as humans in a Humans vs Nations game")
        big["map"] = "World"
        self.assertEqual(slots_of([small, big], t), 1 + 50 + t["World"]["nations"] + 400)
        self.assertEqual(slots_of([entry("c", nations=8, bots=0)], t), 1 + 2 + 8)


class ScheduleTest(unittest.TestCase):
    def test_warmup_then_cosine_to_zero(self):
        a = argparse.Namespace(lr=1e-3, warmup=10, updates=110)
        lrs = [lr_at(u, a) for u in range(111)]
        self.assertAlmostEqual(lrs[0], 1e-4)
        self.assertAlmostEqual(lrs[9], 1e-3)
        self.assertAlmostEqual(lrs[10], 1e-3)
        self.assertAlmostEqual(lrs[60], 5e-4)
        self.assertAlmostEqual(lrs[110], 0.0)
        self.assertTrue(all(x >= y for x, y in zip(lrs[10:], lrs[11:])))

    def test_one_pass_counts_the_windows_of_every_record(self):
        a = argparse.Namespace(workers=2, games_per_worker=3, accum=4)
        self.assertEqual(one_pass([{"turns": 5 * 120}, {"turns": 5 * 360}], a), 20)  # 480 windows / 24 per update


class AccumulationTest(unittest.TestCase):
    def test_accumulated_windows_give_the_gradient_of_one_batch(self):
        # In exact fp32 (no bf16 autocast, no TF32): rounding would move the max-pool's argmax between batch sizes and
        # bury the comparison in noise (15% on a small gradient in bf16, 4e-6 in exact fp32).
        exact_fp32(self)
        torch.manual_seed(0)
        model = randomized(Policy(DIMS, core="moe")).cuda().to(memory_format=torch.channels_last)
        windows = []
        for i, n_legal in enumerate((6, 2, 4)):  # unequal legal rows: a mean of window means would differ
            labels = torch.zeros(ROWS, 4, dtype=torch.long, device="cuda")
            legal = torch.arange(ROWS, device="cuda") < n_legal
            windows.append((fake_obs(seed=i), labels, legal, torch.full((ROWS,), 0.5, device="cuda")))
        with mock.patch.object(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()):
            model.zero_grad()
            backward_windows([model], windows, argparse.Namespace(accum=3, vf_coef=0.5), [Meter()])
            accumulated = [p.grad.clone() for p in model.parameters()]
            model.zero_grad()
            obs = {k: torch.cat([w[0][k] for w in windows]) for k in windows[0][0]}
            bc_loss(model, obs, *(torch.cat([w[i] for w in windows]) for i in (1, 2, 3)), 0.5)[0].backward()
        for p, g in zip(model.parameters(), accumulated):
            self.assertLessEqual(float((p.grad - g).norm()), 1e-4 * float(p.grad.norm()) + 1e-8)


class CloneTest(unittest.TestCase):
    def test_overfitting_a_few_windows_lowers_their_loss(self):
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=1, seed="bc", records=[FIXTURE])
        try:
            batches = []
            for _ in range(40):
                batches.append(({k: v.clone() for k, v in env.obs.items()}, env.labels.clone()))
                env.step()
            dims = env.dims
        finally:
            env.close()
        obs = {k: torch.cat([b[0][k] for b in batches]) for k in batches[0][0]}
        labels = torch.cat([b[1] for b in batches])
        legal = label_legal(obs, labels)
        self.assertGreater(int(legal.sum()), 40)
        torch.manual_seed(0)
        model = Policy(dims, core="moe", shares=True).cuda().to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        target = torch.zeros(len(labels), device="cuda")
        losses = []
        for _ in range(40):
            loss, *_ = bc_loss(model, obs, labels, legal, target, vf_coef=0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] - 0.3)  # at least the no-op prior is learnt


class TeamTest(unittest.TestCase):
    def test_a_v5_network_learns_from_the_team_record(self):
        config = {"map": "Between Two Seas", "compact": True, "nations": 0, "bots": 100, "difficulty": "Medium",
                  "timerMinutes": 0, "seats": 18}
        env = VecEnv(config, workers=1, games_per_worker=1, seed="team", records=[TEAM])
        try:
            windows = []
            for _ in range(400):
                if (env.labels[:, 0] >= REQUEST).any():
                    windows.append(({k: v.clone() for k, v in env.obs.items()}, env.labels.clone()))
                    if len(windows) == 8:
                        break
                env.step()
            dims = env.dims
        finally:
            env.close()
        obs = {k: torch.cat([w[0][k] for w in windows]) for k in windows[0][0]}
        labels = torch.cat([w[1] for w in windows])
        legal = label_legal(obs, labels)
        self.assertTrue(bool((legal & (labels[:, 0] >= REQUEST)).any()), "a new-type label was seen and is legal")
        torch.manual_seed(0)
        model = Policy(dims, v5=True).cuda().to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        target = torch.zeros(len(labels), device="cuda")
        losses = []
        for _ in range(21):
            loss, *_ = bc_loss(model, obs, labels, legal, target, vf_coef=0.0)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertTrue(math.isfinite(losses[0]))
        self.assertLess(losses[-1], losses[0])
        # a v4 network cannot express the new types: it counts their labels in labels/beyond_net and scores the rest
        v4 = Policy({**dims, "playerFeatures": DIMS["playerFeatures"]}, shares=True, full=True).cuda().eval()
        m, zero = Meter(), torch.zeros(len(labels), device="cuda")
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = v4(obs)
        measure(m, obs, labels, legal, ActionDist(out, obs), zero, zero)
        s = m.means()
        self.assertGreater(s["labels/beyond_net"], 0)
        self.assertTrue(math.isfinite(s["bc/nll"]) and "bc/nll_donate_share" not in s)


class RandomPolicy5(torch.nn.Module):
    """Uniform outputs of a network with the 5 types and 3 heat maps of before 3B step 3."""

    def forward(self, obs):
        n, dev = len(obs["type_mask"]), obs["type_mask"].device
        return {"type": torch.zeros(n, 5, device=dev), "bins": torch.zeros(n, 2, 5, device=dev),
                "maps": torch.zeros(n, 3, obs["cell_mask"].shape[-1], device=dev),
                "ptr": torch.zeros(obs["attack_mask"].shape, device=dev),
                "value": torch.zeros(n, device=dev)}


class MeasureTest(unittest.TestCase):
    def test_label_shares_keep_their_denominators(self):
        obs = fake_obs()
        obs["type_mask"][:] = True
        obs["cell_mask"][:] = True
        labels = torch.tensor([[0, 0, 0, 0]] * 2 + [[4, 7, 0, 0]] + [[-1, 0, 0, 0]] * 3, device="cuda")
        legal = label_legal(obs, labels)
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, legal, ActionDist(RandomPolicy5()(obs), obs), zero, zero)
        s = m.means()
        self.assertAlmostEqual(s["labels/noop"], 2 / 3)  # of the legal rows, as before the label mix
        self.assertAlmostEqual(s["labels/build_city"], 1 / 6)  # of all rows
        self.assertAlmostEqual(s["labels/out_of_set_commands"], 3 / 4)  # of the commands: Gate 3B's measure

    def test_new_labels_need_their_heat_map_cell_or_an_attack_to_cancel(self):
        obs = fake_obs()
        obs["type_mask"][:] = True
        obs["cell_mask"][:] = False
        obs["cell_mask"][:, 4, 7] = True  # the silo map allows cell 7 only
        obs["retreat_mask"][:, 3] = True  # the seat attacks player 3
        labels = torch.tensor([[6, 7, 0, 0], [6, 8, 0, 0], [RETREAT, 0, 3, 0], [RETREAT, 0, 2, 0], [13, 7, 0, 0],
                               [-1, 0, 0, 0]], device="cuda")
        self.assertEqual(label_legal(obs, labels).tolist(), [True, False, True, False, False, False])

    def test_new_labels_need_their_target_or_destination(self):
        n, p, hw = 10, 5, 16
        obs = {"type_mask": torch.ones(n, len(ACT_NAMES), dtype=torch.bool, device="cuda"),
               "cell_mask": torch.zeros(n, DST + 1, hw, dtype=torch.bool, device="cuda"),
               "attack_mask": torch.zeros(n, p, dtype=torch.bool, device="cuda"),
               "retreat_mask": torch.zeros(n, p, dtype=torch.bool, device="cuda"),
               "diplomacy_mask": torch.zeros(n, len(DIPLOMACY), p, dtype=torch.bool, device="cuda")}
        obs["diplomacy_mask"][:, 0, 2] = True  # a request, a donation and a break can go to player 2
        obs["diplomacy_mask"][:, BREAK - REQUEST, 2] = True
        obs["diplomacy_mask"][:, DONATE - REQUEST, 2] = True
        obs["cell_mask"][:, MAP_OF[MOVE], 5] = True  # a warship stands in cell 5
        obs["cell_mask"][:, DST, 9] = True  # and cell 9 is water
        labels = torch.tensor([[REQUEST, 0, 2, 0], [REQUEST, 0, 3, 0],  # a legal row, a masked-out one
                               [RENEW, 0, 2, 0],  # renewals are not allowed for player 2: the mask of its own type
                               [BREAK, 0, 2, 0],
                               [DONATE, 0, 2, 40], [DONATE, 0, 2, 0],  # a share of 40%, a share of 0
                               [MOVE, 5, 9, 1], [MOVE, 5, 8, 1], [MOVE, 5, 9, 2], [MOVE, 6, 9, 0]], device="cuda")
        self.assertEqual(label_legal(obs, labels).tolist(),
                         [True, False, False, True, True, False, True, False, False, False])

    def test_a_v5_networks_diplomacy_and_warship_heads_are_scored_per_head(self):
        obs = fake_obs()
        obs["type_mask"][:], obs["cell_mask"][:] = True, True
        obs["diplomacy_mask"][:] = False
        obs["diplomacy_mask"][:, :, 2] = True  # player 2 is the only recipient
        hw, players = obs["cell_mask"].shape[-1], obs["attack_mask"].shape[-1]
        out = {"type": torch.zeros(ROWS, len(ACT_NAMES), device="cuda"), "maps": torch.zeros(ROWS, DST, hw, device="cuda"),
               "ptr": torch.zeros(ROWS, players, device="cuda"), "retreat": torch.zeros(ROWS, players, device="cuda"),
               "shares": torch.zeros(ROWS, 3, players, SHARES, device="cuda"),
               "diplomacy": torch.zeros(ROWS, len(DIPLOMACY), players, device="cuda"),
               "move": lambda src: (torch.zeros(ROWS, hw, device="cuda"),
                                    torch.tensor([[0.0, 5.0]] * ROWS, device="cuda"))}  # the fleet is the likelier
        labels = torch.tensor([[DONATE, 0, 2, 40]] * 2 + [[REQUEST, 0, 2, 0]] + [[MOVE, 7, 9, 0]] * 2 + [[MOVE, 7, 9, 1]],
                              device="cuda")
        legal = label_legal(obs, labels)
        self.assertTrue(bool(legal.all()))
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, legal, ActionDist(out, obs), zero, zero)
        s = m.means()
        self.assertEqual((s["bc/donate_troops_target_acc"], s["bc/nll_donate_troops_target"]), (1.0, 0.0))
        self.assertEqual(s["bc/alliance_request_target_acc"], 1.0)
        self.assertAlmostEqual(s["bc/nll_donate_share"], math.log(SHARES), places=4)
        self.assertAlmostEqual(s["bc/nll_warship_dst"], math.log(hw), places=4)
        self.assertAlmostEqual(s["bc/warship_fleet_acc"], 1 / 3)
        self.assertNotIn("bc/nll_renew_alliance_target", s)  # no such rows

    def test_a_network_learns_and_is_scored_only_on_the_rows_its_actions_express(self):
        obs = fake_obs()
        obs["type_mask"][:] = True
        obs["cell_mask"][:] = True
        old = RandomPolicy5()(obs)
        labels = torch.tensor([[6, 7, 0, 0]] * 3 + [[4, 7, 0, 0]] * 3, device="cuda")  # 3 silos, 3 cities
        legal = label_legal(obs, labels)
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, legal, ActionDist(old, obs), zero, zero)
        s = m.means()
        self.assertAlmostEqual(s["labels/beyond_net"], 0.5)
        self.assertAlmostEqual(s["labels/build_silo"], 0.5)
        self.assertAlmostEqual(s["bc/nll_type"], math.log(5), places=4)  # only the 3 city rows, over 5 types

    def test_a_share_networks_troops_score_like_for_like_and_on_its_own_shares(self):
        obs = fake_obs()  # rows 3-5: no-op, attack (only on player 2) and city are legal
        hw, players = obs["cell_mask"].shape[-1], obs["attack_mask"].shape[-1]
        out = {"type": torch.zeros(ROWS, 5, device="cuda"), "maps": torch.zeros(ROWS, 3, hw, device="cuda"),
               "ptr": torch.zeros(ROWS, players, device="cuda"),
               "shares": torch.zeros(ROWS, 2, players, SHARES, device="cuda")}
        labels = torch.tensor([[ATTACK, 0, 2, 20]] * ROWS, device="cuda")
        legal = label_legal(obs, labels)
        self.assertEqual(legal.tolist(), [False] * 3 + [True] * 3)
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, legal, ActionDist(out, obs), zero, zero)
        s = m.means()
        self.assertAlmostEqual(s["bc/nll_attack_bin"], -math.log(0.20), places=4)  # 20% lies in 25%'s range, 16-35%
        self.assertAlmostEqual(s["bc/nll_attack_share"], math.log(SHARES), places=4)
        self.assertAlmostEqual(s["bc/nll"] - s["bc/nll_type"], -math.log(0.20), places=4)  # one legal target adds 0
        self.assertAlmostEqual(s["bc/nll_shares"] - s["bc/nll_type"], math.log(SHARES), places=4)

    def test_cell_heads_score_how_close_the_bots_cell_lands_to_the_humans(self):
        obs = fake_obs()
        obs["scalars"][:, 14] = 1.0  # a Normal map: cells of 8 x 8 tiles
        obs["type_mask"][:], obs["cell_mask"][:] = True, True
        hw, w = obs["cell_mask"].shape[-1], obs["spatial"].shape[-1]
        human = 10 * w + 10  # the cell where the human built a city
        maps = torch.full((ROWS, 3, hw), -30.0, device="cuda")
        maps[:3, 2, human + 1] = 30.0  # rows 0-2 build one cell to the right: 8 tiles away
        maps[3:, 2, human + 3 * w] = 30.0  # rows 3-5 three cells below: 24 tiles away
        out = {"type": torch.zeros(ROWS, 5, device="cuda"), "maps": maps,
               "ptr": torch.zeros(ROWS, obs["attack_mask"].shape[-1], device="cuda"),
               "bins": torch.zeros(ROWS, 2, 5, device="cuda")}
        labels = torch.tensor([[BUILD, human, 0, 0]] * ROWS, device="cuda")
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, label_legal(obs, labels), ActionDist(out, obs), zero, zero)
        s = m.means()
        self.assertAlmostEqual(s["bc/city_cell_near"], 0.5, places=4)  # within 16 tiles: rows 0-2 only
        self.assertAlmostEqual(s["bc/city_cell_dist"], (8 + 24) / 2, places=3)

    def test_closeness_has_a_uniform_pick_as_its_baseline(self):
        obs = fake_obs()
        obs["scalars"][:, 14] = 1.0  # a Normal map: cells of 8 x 8 tiles
        obs["type_mask"][:] = True
        w = obs["spatial"].shape[-1]
        human = 10 * w + 10
        obs["cell_mask"][:] = False
        obs["cell_mask"][:, 2, [human, human + 1, human + 3 * w]] = True  # 0, 8 and 24 tiles from the human's city
        labels = torch.tensor([[BUILD, human, 0, 0]] * ROWS, device="cuda")
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, label_legal(obs, labels), ActionDist(RandomPolicy5()(obs), obs), zero, zero)
        s = m.means()
        self.assertAlmostEqual(s["bc/city_cell_near_uniform"], 2 / 3, places=5)  # 2 of the 3 within 16 tiles
        self.assertAlmostEqual(s["bc/city_cell_dist_uniform"], 32 / 3, places=4)  # (0 + 8 + 24) / 3


class ScoreTest(unittest.TestCase):
    def test_scores_are_repeatable_and_per_head(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, args in (("old", {"widths": "64,128,192"}),
                               ("moe", {"arch": "v2", "core": "moe", "widths": "64,128,192"}),
                               ("shares", {"arch": "v3", "core": "moe", "widths": "64,128,192"}),
                               ("full", {"arch": "v4", "core": "moe", "widths": "64,128,192"})):
                torch.manual_seed(0)
                path = Path(d) / name / "update0000001.pt"
                path.parent.mkdir()
                torch.save({"model": new_net(DIMS, args).state_dict(), "args": {**args, "run": name}, "update": 1},
                           path)
                paths.append(path)
            first = score(paths, [FIXTURE], workers=1, games_per_worker=1, steps=30)
            second = score(paths, [FIXTURE], workers=1, games_per_worker=1, steps=30)
            text = table(first)
        self.assertEqual((first["rows"], first["games"], first["records"]), (60, 1, 1))
        for a, b in zip(first["metrics"], second["metrics"]):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assertTrue(math.isclose(a[k], b[k], abs_tol=1e-4) or (math.isnan(a[k]) and math.isnan(b[k])), k)
        old, moe, shares, full = first["metrics"]
        self.assertIn("labels/build_silo", full)
        for k in ("bc/nll", "bc/nll_type", f"bc/nll_{FIXTURE_SIZE}", "bc/type_acc", "bc/value_mse", "value/outcome_corr"):
            self.assertIn(k, old)
        self.assertFalse(any(k.startswith("router/") for k in old), "the Phase 2 network has no router")
        self.assertAlmostEqual(sum(moe[f"router/{FIXTURE_SIZE}_{i}"] for i in range(8)), 1.0, places=4)
        self.assertIn("bc/nll_shares", shares)
        self.assertNotIn("bc/nll_shares", moe)
        self.assertIn("old 1", text)
        self.assertIn("bc/nll_type", text)


class MainTest(unittest.TestCase):
    RUN = "unittest-bc"

    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        self.store = self.data / "v0.35.0"  # crawl.store() under the patched crawl.DATA
        (self.store / "records").mkdir(parents=True)
        shutil.copy(FIXTURE, self.store / "records" / "fixture.json.gz")
        self.entries = [entry("fixture", bots=400)]
        self.write_index()
        self.common = ["--run", self.RUN, "--workers", "1", "--games-per-worker", "1", "--val-workers", "0",
                       "--val-share", "0", "--port", "0", "--log-every", "1", "--save-every", "2", "--accum", "2",
                       "--warmup", "1"]
        self.seeds, self.calls = [], []

    def write_index(self):
        (self.store / "index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.entries))

    def add_record(self, source, game_id, mode, **fields):
        shutil.copy(source, self.store / "records" / f"{game_id}.json.gz")
        self.entries.append(entry(game_id, mode, **fields))
        self.write_index()

    def tearDown(self):
        for d in ("checkpoints", "runs"):
            for run in (self.RUN, f"{self.RUN}-moe-swiglu"):
                shutil.rmtree(ROOT / d / run, ignore_errors=True)
        shutil.rmtree(self.data, ignore_errors=True)

    def run_main(self, argv):
        real = bc.VecEnv

        def spy(*args, **kwargs):
            self.seeds.append(kwargs["seed"])
            self.calls.append(kwargs)
            return real(*args, **kwargs)

        with mock.patch.object(crawl, "DATA", self.data), mock.patch.object(bc, "VecEnv", spy):
            main(argv)

    def latest(self, run=None):
        return torch.load(ROOT / "checkpoints" / (run or self.RUN) / "latest.pt", weights_only=False)

    def test_train_save_and_resume(self):
        self.run_main(self.common + ["--updates", "3"])
        first = self.latest()
        self.assertEqual((first["kind"], first["update"]), ("bc", 3))
        self.assertEqual((first["args"]["arch"], first["args"]["core"]), ("v5", "moe"))
        self.assertEqual(first["global_step"], 3 * 2 * 2)  # 3 updates of 2 windows of 2 rows
        self.assertEqual(self.seeds[0], "bc-1-u0", "runs with one seed replay the same records in the same order")
        self.run_main(self.common + ["--updates", "4", "--resume"])
        self.assertEqual(self.latest()["update"], 4)
        self.assertTrue((ROOT / "runs" / self.RUN).is_dir())
        self.assertNotEqual(self.seeds[0], self.seeds[1], "a resumed run replays the records in a new order")

    def test_each_family_trains_on_its_own_records(self):
        info = json.loads(gzip.decompress(TEAM.read_bytes()))["info"]
        self.add_record(TEAM, "team", "team", map="Between Two Seas", teams=6, bots=info["config"]["bots"],
                        players=[{"clientID": p["clientID"], "publicID": None} for p in info["players"]])
        self.run_main(self.common + ["--mix", "ranked1v1=1", "team=1", "--updates", "1"])
        train = self.calls[0]
        self.assertEqual(train["records"], [paths_in(self.store, "fixture"), paths_in(self.store, "team")])
        self.assertGreaterEqual(train["slots"], 1 + 18 + info["config"]["bots"])
        self.assertEqual(self.latest()["global_step"], 1 * 2 * 4)  # 1 update of 2 windows, 2 workers x 1 game x 2 players
        self.assertEqual(self.latest()["args"]["mix"], ["ranked1v1=1", "team=1"])
        self.run_main(["--run", self.RUN, "--resume", "--updates", "2"])
        self.assertEqual(self.latest()["update"], 2, "a resume keeps the mix")
        self.assertEqual(self.calls[-1]["records"], train["records"])

    def test_unusable_records_stay_out_of_the_run(self):
        self.add_record(FIXTURE, "modified", "ranked1v1", modifiers=["isWaterNukes"])
        self.add_record(FIXTURE, "huge", "ranked1v1", map="Brain Rot")
        self.run_main(self.common + ["--updates", "1"])
        self.assertEqual(self.calls[0]["records"], [paths_in(self.store, "fixture")])

    def test_validation_has_a_worker_per_family_and_runs_once_before_the_first_update(self):
        # 0.05 of the games are held out: "sample" is, "train" and "ttrain" are not (zlib.crc32 of the ID)
        self.entries = [entry("train")]
        for game_id in ("train", "sample"):
            shutil.copy(FIXTURE, self.store / "records" / f"{game_id}.json.gz")
        self.entries.append(entry("sample"))
        self.add_record(TEAM, "ttrain", "team", teams=6, map="Between Two Seas")
        self.run_main(self.common + ["--val-share", "0.05", "--val-workers", "1", "--val-steps", "2", "--mix",
                                     "ranked1v1=1", "team=1", "--updates", "1"])
        val = self.calls[1]
        self.assertEqual((val["seed"], val["records"]), ("val", [paths_in(self.store, "sample")]),
                         "no held-out team record: that family is left out")
        curves = EventAccumulator(str(ROOT / "runs" / self.RUN))
        curves.Reload()
        self.assertEqual([e.step for e in curves.Scalars("val/bc/nll")], [0], "the baseline, before any update")

    def test_fine_tuning_keeps_the_checkpoints_network(self):
        init = ROOT / "checkpoints" / self.RUN / "init.pt"
        init.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": PolicyV1(DIMS).state_dict(), "args": {"widths": "64,128,192"}}, init)
        self.run_main(self.common + ["--init", str(init), "--updates", "1"])
        self.assertEqual(self.latest()["args"]["arch"], "v1")
        with self.assertRaisesRegex(SystemExit, "drop --core"):
            self.run_main(self.common + ["--init", str(init), "--core", "moe", "--updates", "1"])

    def test_a_resume_refuses_flags_it_would_ignore(self):
        self.run_main(self.common + ["--updates", "1"])
        with self.assertRaisesRegex(SystemExit, "--lr would be ignored"):
            self.run_main(self.common + ["--updates", "2", "--lr", "1e-3", "--resume"])

    def test_a_new_run_refuses_a_folder_that_holds_a_run(self):
        self.run_main(self.common + ["--updates", "1"])
        with self.assertRaisesRegex(SystemExit, "add --resume"):
            self.run_main(self.common + ["--updates", "1"])

    def test_a_resume_hides_what_the_crashed_run_logged_after_its_checkpoint(self):
        self.run_main(self.common + ["--updates", "3"])  # logs updates 1-3 at steps 4, 8 and 12
        path = ROOT / "checkpoints" / self.RUN / "latest.pt"
        state = torch.load(path, weights_only=False)
        state.update(update=1, global_step=4)  # as if it crashed after logging update 3, its last checkpoint at 1
        torch.save(state, path)
        self.run_main(self.common + ["--updates", "3", "--resume"])
        curves = EventAccumulator(str(ROOT / "runs" / self.RUN))
        curves.Reload()
        self.assertEqual([e.step for e in curves.Scalars("bc/nll")], [4, 8, 12], "the curve never steps back")

    def test_a_twin_starts_from_the_same_weights_outside_its_core(self):
        self.run_main(self.common + ["--updates", "0", "--twin", "moe-swiglu"])
        main, twin = self.latest(), self.latest(f"{self.RUN}-moe-swiglu")
        self.assertEqual((main["args"]["core"], twin["args"]["core"]), ("moe", "moe-swiglu"))
        self.assertEqual((twin["args"]["run"], twin["args"]["twin"]), (f"{self.RUN}-moe-swiglu", None))
        for k, v in main["model"].items():
            if not k.startswith("core."):
                self.assertTrue(torch.equal(v, twin["model"][k]), k)

    def test_a_twin_trains_on_the_same_windows_and_resumes_with_the_run(self):
        self.run_main(self.common + ["--updates", "2", "--twin", "moe-swiglu"])
        main, twin = self.latest(), self.latest(f"{self.RUN}-moe-swiglu")
        self.assertEqual((twin["update"], twin["global_step"]), (2, main["global_step"]))
        self.assertTrue((ROOT / "runs" / f"{self.RUN}-moe-swiglu").is_dir())
        self.run_main(self.common + ["--updates", "3", "--resume"])
        self.assertEqual((self.latest()["update"], self.latest(f"{self.RUN}-moe-swiglu")["update"]), (3, 3))
        (ROOT / "checkpoints" / f"{self.RUN}-moe-swiglu" / "latest.pt").unlink()
        with self.assertRaisesRegex(SystemExit, "is missing"):
            self.run_main(self.common + ["--updates", "4", "--resume"])
        with self.assertRaisesRegex(SystemExit, "drop --init"):
            self.run_main(self.common + ["--init", str(ROOT / "checkpoints" / self.RUN / "latest.pt"), "--twin", "moe",
                                         "--updates", "1"])

    def test_a_grown_run_starts_from_the_v4_checkpoint_and_resumes_without_growing(self):
        init = ROOT / "checkpoints" / self.RUN / "init.pt"
        init.parent.mkdir(parents=True, exist_ok=True)
        args = {"arch": "v4", "core": "moe", "widths": "64,128,192"}
        v4 = randomized(Policy(DIMS, shares=True, full=True))
        torch.save({"model": v4.state_dict(), "args": args}, init)
        self.run_main(self.common + ["--init", str(init), "--grow", "--updates", "0"])
        grown = self.latest()
        self.assertEqual((grown["args"]["arch"], grown["args"]["core"], grown["args"]["grow"]), ("v5", "moe", True))
        self.assertTrue((grown["model"]["type_head.bias"][V4_ACTS:] == NEW_TYPE_BIAS).all())
        self.assertTrue(torch.equal(grown["model"]["trunk.0.weight"].cpu(), v4.state_dict()["trunk.0.weight"]))
        self.run_main(self.common + ["--updates", "1", "--resume"])
        self.assertEqual(self.latest()["update"], 1)
        with self.assertRaisesRegex(SystemExit, "grows the --init"):
            self.run_main(self.common + ["--grow", "--updates", "1"])
        v5 = {**args, "arch": "v5"}
        torch.save({"model": new_net(DIMS_V5, v5).state_dict(), "args": v5}, init)
        with self.assertRaisesRegex(SystemExit, "arch v4"):
            self.run_main(self.common + ["--run", f"{self.RUN}-v5", "--init", str(init), "--grow", "--updates", "1"])

    def test_old_bc_checkpoints_cannot_resume(self):
        path = ROOT / "checkpoints" / self.RUN / "latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"args": {"run": self.RUN, "widths": "64,128,192"}, "update": 5}, path)  # no "accum"
        with self.assertRaisesRegex(SystemExit, "predates.*--init"):
            self.run_main(["--run", self.RUN, "--resume"])

if __name__ == "__main__":
    unittest.main()
