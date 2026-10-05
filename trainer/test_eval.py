import json
import tempfile
import unittest
from pathlib import Path

import torch

from trainer.control import REPLAYS
from trainer.env import ONION
from trainer.eval import Cooldown, label, name_players, play, replay_dir, seats_for, summarize, tag, two_bin_sample
from trainer.model import ActionDist, new_net
from trainer.test_model import DIMS

ONION2 = {**ONION, "seats": 2, "nations": 0, "timerMinutes": 1}


class EvalTest(unittest.TestCase):
    def test_summarize(self):
        s = summarize([{"reward": 1, "rank": 1}, {"reward": -1, "rank": 4}])
        self.assertEqual(s["episodes"], 2)
        self.assertEqual(s["win_rate"], 0.5)
        self.assertEqual(s["mean_reward"], 0.0)
        self.assertEqual(s["mean_rank"], 2.5)

    def test_summarize_splits_the_win_rate_by_seat(self):
        s = summarize([{"reward": 1, "rank": 1, "seat": 0}, {"reward": 1, "rank": 1, "seat": 1},
                       {"reward": -1, "rank": 2, "seat": 1}])
        self.assertEqual((s["win_rate_seat0"], s["win_rate_seat1"]), (1.0, 0.5))

    def test_policies_swap_seats_between_slots(self):
        self.assertEqual(seats_for(4, 2), [[0, 1], [1, 0], [0, 1], [1, 0]])
        self.assertEqual(seats_for(2, 1), [[0], [0]])

    def test_recorded_players_are_named_after_their_policies(self):
        with tempfile.TemporaryDirectory() as d:
            ppo, clone, rec = Path(d) / "ppo.pt", Path(d) / "bc.pt", Path(d) / "game.json"
            torch.save({"args": {"kl_coef": 0.2}, "update": 5000}, ppo)
            torch.save({"args": {}, "update": 60000}, clone)
            self.assertEqual([label(s) for s in (str(ppo), str(clone), "random", "scripted")],
                             ["PPO 5000", "Clone 60000", "Random", "Scripted"])
            rec.write_text(json.dumps({"info": {"players": [{"clientID": "AGENT000", "username": "agent0"},
                                                            {"clientID": "AGENT001", "username": "agent1"}]}}))
            name_players(rec, {0: "Clone 60000", 1: "PPO 5000"})
            players = json.loads(rec.read_text())["info"]["players"]
            self.assertEqual([p["username"] for p in players], ["Clone 60000", "PPO 5000"])

    def test_recordings_go_to_a_folder_per_matchup(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, old = Path(d) / "bc" / "update0060000.pt", Path(d) / "x.pt"
            ckpt.parent.mkdir()
            torch.save({"args": {"run": "bc"}, "update": 60000}, ckpt)
            torch.save({"args": {}, "update": 7}, old)
            self.assertEqual(tag(str(ckpt)), "bc 60000")
            self.assertEqual(tag(str(old)), f"{Path(d).name} 7")  # no run in its settings: its folder
            self.assertEqual(replay_dir([str(ckpt), "random"]), REPLAYS / "bc 60000 vs random")

    def test_old_and_new_networks_play_each_other(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, args in (("new", {"arch": "v2", "core": "moe", "widths": "64,128,192"}),
                               ("old", {"widths": "64,128,192"})):
                path = Path(d) / f"{name}.pt"
                torch.save({"model": new_net(DIMS, args).state_dict(), "args": args, "update": 1}, path)
                paths.append(str(path))
            r = play(paths, ONION2, episodes=2, workers=1, games_per_worker=2, seed="mix")
        self.assertEqual(r["episodes"], 2)

    def test_one_seat_on_fixed_seeds_is_reproducible(self):
        run = lambda: play(["random"], {**ONION, "timerMinutes": 1}, episodes=2, workers=1, games_per_worker=2,
                           seed="eval-test")
        first, second = run(), run()
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(first, second)

    def test_policy_a_can_take_a_troop_and_a_gold_action_per_decision(self):
        r = play(["random", "random"], ONION2, episodes=2, workers=1, games_per_worker=2, seed="bins", a_two_bins=True)
        self.assertEqual(r["episodes"], 2)
        self.assertIn("parallel_share", r["meters"])

    def test_a_cooldown_keeps_policy_a_from_acting_twice_in_a_row(self):
        r = play(["random", "random"], ONION2, episodes=2, workers=1, games_per_worker=2, seed="cool", a_cooldown=1)
        self.assertEqual(r["meters"]["follow_share"], 0.0)
        self.assertGreater(r["meters"]["blocked_share"], 0.0)

    def test_scripted_against_random(self):
        tmp = tempfile.mkdtemp()
        r = play(["scripted", "random"], ONION2, episodes=4, workers=1, games_per_worker=4, seed="eval-2",
                 record={"dir": tmp, "gitCommit": "DEV", "games": 1})
        self.assertEqual(r["episodes"], 4)
        self.assertTrue(0.0 <= r["win_rate"] <= 1.0)
        self.assertEqual(len(r["recorded"]), 1)
        saved = json.loads((Path(tmp) / f"{r['recorded'][0]}.json").read_text())
        self.assertEqual(saved["info"]["gameID"], r["recorded"][0])


def type_mask(spawn_phase=False):
    """One row's legal types (no-op, spawn, attack, boat, city): the spawn phase allows only no-op and spawn."""
    return torch.tensor([[True, True, False, False, False] if spawn_phase else [True, False, True, True, True]])


def always(row):
    return lambda obs: torch.tensor([row])


class CooldownTest(unittest.TestCase):
    """The bandwidth test (PLAN 3B): policy A's next decisions after an action become no-ops, and meters count how it
    acts."""

    def test_the_decision_after_an_action_becomes_a_no_op(self):
        policy = Cooldown(always([2, 0, 1, 0]), 1)
        self.assertEqual([int(policy({"type_mask": type_mask()})[0, 0]) for _ in range(4)], [2, 0, 2, 0])
        m = policy.meters()
        self.assertEqual((m["act_share"], m["follow_share"], m["blocked_share"]), (0.5, 0.0, 0.5))

    def test_the_spawn_phase_is_left_alone(self):
        policy = Cooldown(always([1, 7, 0, 0]), 1)
        self.assertEqual([int(policy({"type_mask": type_mask(True)})[0, 0]) for _ in range(3)], [1, 1, 1])

    def test_without_a_cooldown_the_meters_only_count(self):
        policy = Cooldown(always([2, 0, 1, 0]), 0)
        self.assertEqual([int(policy({"type_mask": type_mask()})[0, 0]) for _ in range(3)], [2, 2, 2])
        m = policy.meters()
        self.assertEqual((m["act_share"], m["blocked_share"]), (1.0, 0.0))
        self.assertAlmostEqual(m["follow_share"], 2 / 3)

    def test_a_decision_with_two_actions_counts_as_parallel_and_the_cooldown_drops_both(self):
        policy = Cooldown(always([2, 0, 1, 0, 4, 3, 0, 0]), 1)
        rows = [policy({"type_mask": type_mask()})[0].tolist() for _ in range(2)]
        self.assertEqual(rows, [[2, 0, 1, 0, 4, 3, 0, 0], [0] * 8])
        m = policy.meters()
        self.assertEqual((m["act_share"], m["parallel_share"]), (0.5, 0.5))

    def test_both_wanted_counts_decisions_with_weight_on_a_troop_and_a_gold_action(self):
        def policy(obs):
            policy.type_probs = torch.tensor([[0.2, 0.0, 0.3, 0.0, 0.5], [0.5, 0.0, 0.4, 0.0, 0.1]])
            return torch.zeros(2, 4, dtype=torch.long)

        meter = Cooldown(policy, 0)
        meter({"type_mask": type_mask().expand(2, -1)})
        self.assertEqual(meter.meters()["both_share"], 0.5)


def dist(n, type_logits, legal_types):
    """n rows of one policy with 4 cells and 3 player rows: the city map allows only cell 3, the pointer only player 2,
    and the attack troop bin is almost surely bin 1."""
    out = {"type": torch.tensor([type_logits] * n, dtype=torch.float), "maps": torch.zeros(n, 3, 4),
           "ptr": torch.zeros(n, 3), "bins": torch.zeros(n, 2, 5)}
    out["bins"][:, 0, 1] = 30.0
    obs = {"type_mask": torch.tensor([legal_types] * n), "cell_mask": torch.ones(n, 3, 4, dtype=torch.bool),
           "attack_mask": torch.tensor([[False, False, True]] * n)}
    obs["cell_mask"][:, 2, :3] = False
    return ActionDist(out, obs), obs


class TwoBinTest(unittest.TestCase):
    """The upward probe of the bandwidth test (PLAN 3B): one troop and one gold action per decision."""

    def test_each_kind_of_action_keeps_its_own_probability_and_both_can_happen_together(self):
        torch.manual_seed(0)
        d, obs = dist(4000, [-30.0, 0.0, 0.0, -30.0, 0.0], [True, False, True, True, True])  # attack 1/2, city 1/2
        a = two_bin_sample(d, obs)
        attack, city = a[:, 0] == 2, a[:, 4] == 4
        self.assertAlmostEqual(attack.float().mean().item(), 0.5, delta=0.03)
        self.assertAlmostEqual(city.float().mean().item(), 0.5, delta=0.03)
        self.assertAlmostEqual((attack & city).float().mean().item(), 0.25, delta=0.03)
        self.assertEqual(a[attack][:, 1:4].unique(dim=0).tolist(), [[0, 2, 1]])  # target 2, bin 1
        self.assertEqual(a[city][:, 5:8].unique(dim=0).tolist(), [[3, 0, 0]])  # the city map's only legal cell
        self.assertEqual(a[~attack][:, :4].abs().sum().item(), 0)
        self.assertEqual(a[~city][:, 4:].abs().sum().item(), 0)

    def test_the_spawn_phase_keeps_one_ordinary_action(self):
        torch.manual_seed(0)
        d, obs = dist(100, [-30.0, 0.0, 0.0, 0.0, 0.0], [True, True, False, False, False])
        a = two_bin_sample(d, obs)
        self.assertTrue((a[:, 0] == 1).all())
        self.assertEqual(a[:, 4:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
