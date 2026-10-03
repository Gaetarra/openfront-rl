import json
import tempfile
import unittest
from pathlib import Path

import torch

from trainer.control import REPLAYS
from trainer.env import ONION
from trainer.eval import label, name_players, play, replay_dir, seats_for, summarize, tag

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

    def test_one_seat_on_fixed_seeds_is_reproducible(self):
        run = lambda: play(["random"], {**ONION, "timerMinutes": 1}, episodes=2, workers=1, games_per_worker=2,
                           seed="eval-test")
        first, second = run(), run()
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(first, second)

    def test_scripted_against_random(self):
        tmp = tempfile.mkdtemp()
        r = play(["scripted", "random"], ONION2, episodes=4, workers=1, games_per_worker=4, seed="eval-2",
                 record={"dir": tmp, "gitCommit": "DEV", "games": 1})
        self.assertEqual(r["episodes"], 4)
        self.assertTrue(0.0 <= r["win_rate"] <= 1.0)
        self.assertEqual(len(r["recorded"]), 1)
        saved = json.loads((Path(tmp) / f"{r['recorded'][0]}.json").read_text())
        self.assertEqual(saved["info"]["gameID"], r["recorded"][0])


if __name__ == "__main__":
    unittest.main()
