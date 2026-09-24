import unittest

import numpy as np
import torch

from trainer.env import LAND, N_CHANNELS, OWN, VecEnv

ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "timerMinutes": 5, "seats": 1}


def random_legal(obs, rng):
    """Uniformly random legal actions from the masks, as padded-grid [type, cell, target, bin] rows."""
    rows = []
    types, attack, cells = (obs[k].cpu().numpy() for k in ("type_mask", "attack_mask", "cell_mask"))
    for i in range(types.shape[0]):
        t = rng.choice(np.flatnonzero(types[i]))
        m = {1: 0, 3: 1, 4: 2}.get(t)
        cell = rng.choice(np.flatnonzero(cells[i, m])) if m is not None and cells[i, m].any() else 0
        target = rng.choice(np.flatnonzero(attack[i])) if t == 2 else 0
        rows.append([t, cell, target, rng.integers(5)])
    return np.array(rows, np.int64)


class VecEnvTest(unittest.TestCase):
    def test_observations_during_random_play(self):
        env = VecEnv(ONION, workers=1, games_per_worker=2, seed="t")
        try:
            obs = env.obs
            self.assertEqual(obs["spatial"].shape, (2, N_CHANNELS, 64, 64))
            self.assertEqual(obs["spatial"].dtype, torch.bfloat16)
            self.assertEqual(obs["players"].shape, (2, 5, 17))
            self.assertEqual(obs["cell_mask"].shape, (2, 3, 64 * 64))
            self.assertTrue(obs["cell_mask"][:, 0].any(), "spawn cells exist at the start")
            rng = np.random.default_rng(0)
            for _ in range(80):
                obs, rew, done, infos, invalid = env.step(random_legal(obs, rng))
                x = obs["spatial"].float()
                self.assertTrue(torch.isfinite(x).all())
                self.assertLessEqual(float(x[:, LAND:OWN + 4].max()), 1.0 + 1e-3)
            self.assertGreater(float(obs["spatial"][:, OWN].float().sum()), 0, "the seats own land")
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


if __name__ == "__main__":
    unittest.main()
