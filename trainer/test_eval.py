import unittest

from trainer.eval import evaluate, summarize
from trainer.model import RandomPolicy

ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "timerMinutes": 1, "seats": 1}


class EvalTest(unittest.TestCase):
    def test_summarize(self):
        s = summarize([{"reward": 1, "rank": 1}, {"reward": -1, "rank": 4}])
        self.assertEqual(s["episodes"], 2)
        self.assertEqual(s["win_rate"], 0.5)
        self.assertEqual(s["mean_reward"], 0.0)
        self.assertEqual(s["mean_rank"], 2.5)

    def test_random_policy_on_fixed_seeds_is_reproducible(self):
        run = lambda: evaluate(RandomPolicy, ONION, episodes=2, workers=1, games_per_worker=2, seed="eval-test")
        first, second = run(), run()
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
