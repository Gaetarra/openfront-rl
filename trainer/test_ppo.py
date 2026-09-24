import shutil
import unittest

import torch

from trainer.env import ROOT
from trainer.ppo import Outcomes, gae, main


class OutcomesTest(unittest.TestCase):
    def test_value_outcome_correlation_by_phase(self):
        o = Outcomes(rows=4)
        rewards = [1.0, -1.0, -2 / 3, -5 / 6]
        o.record([0.1] * 4, [0.3, 0.3, -0.2, -0.2])  # early: only loosely related to the outcome
        o.record([0.9] * 4, [0.9 * r for r in rewards])  # late: tracks it
        o.finish([{"env": i, "seat": 0, "reward": r, "rank": 1, "ticks": 1} for i, r in enumerate(rewards)], seats=1)
        s = o.summary()
        self.assertAlmostEqual(s["value/outcome_corr_late"], 1.0, places=5)
        self.assertLess(s["value/outcome_corr_early"], 0.9)
        self.assertNotIn("value/outcome_corr_mid", s)

    def test_no_correlation_when_every_game_ends_the_same(self):
        o = Outcomes(rows=2)
        o.record([0.9, 0.9], [-0.5, -0.8])
        o.finish([{"env": i, "seat": 0, "reward": -1.0, "rank": 4, "ticks": 1} for i in range(2)], seats=1)
        s = o.summary()
        self.assertEqual(s["value/outcome_acc_late"], 1.0)  # trivially right: why the gate uses the correlation
        self.assertNotIn("value/outcome_corr_late", s)


class GaeTest(unittest.TestCase):
    def test_hand_computed(self):
        r = torch.tensor([[0.0], [0.0], [1.0]])
        v = torch.tensor([[0.5], [0.6], [0.7]])
        d = torch.tensor([[0.0], [0.0], [1.0]])
        adv = gae(r, v, d, torch.tensor([9.0]), gamma=0.9, lam=0.8)
        # t2: 1 - 0.7 = 0.3; t1: 0.9*0.7 - 0.6 + 0.72*0.3 = 0.246; t0: 0.9*0.6 - 0.5 + 0.72*0.246 = 0.21712
        self.assertTrue(torch.allclose(adv[:, 0], torch.tensor([0.21712, 0.246, 0.3])))

    def test_episode_boundary_cuts_bootstrap(self):
        r = torch.tensor([[1.0], [0.0]])
        v = torch.tensor([[0.2], [0.4]])
        d = torch.tensor([[1.0], [0.0]])
        adv = gae(r, v, d, torch.tensor([0.5]), gamma=1.0, lam=1.0)
        self.assertAlmostEqual(float(adv[0, 0]), 0.8, places=6)  # the next episode's values do not leak back
        self.assertAlmostEqual(float(adv[1, 0]), 0.1, places=6)


class SmokeTest(unittest.TestCase):
    RUN = "unittest-smoke"

    def tearDown(self):
        for d in ("checkpoints", "runs"):
            shutil.rmtree(ROOT / d / self.RUN, ignore_errors=True)

    def test_train_save_and_resume(self):
        common = ["--run", self.RUN, "--workers", "1", "--games-per-worker", "2", "--steps", "16",
                  "--minibatches", "2", "--epochs", "1", "--port", "0", "--timer", "1"]
        main(common + ["--updates", "2"])
        first = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
        self.assertEqual(first["update"], 2)
        self.assertEqual(first["global_step"], 2 * 16 * 2)
        # 32 decisions do not finish a game: the unfinished games are skipped on resume, so no seed is played twice
        self.assertEqual(first["episodes"], [1, 1])
        main(common + ["--updates", "3", "--resume"])
        second = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
        self.assertEqual(second["update"], 3)
        self.assertEqual(second["global_step"], 3 * 16 * 2)
        self.assertTrue(all(b >= a for a, b in zip(first["episodes"], second["episodes"])))
        self.assertTrue((ROOT / "runs" / self.RUN).is_dir())


if __name__ == "__main__":
    unittest.main()
