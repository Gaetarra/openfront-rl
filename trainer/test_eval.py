import json
import tempfile
import unittest
from pathlib import Path

from trainer.env import ONION
from trainer.eval import play, seats_for, summarize

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
