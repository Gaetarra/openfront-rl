import shutil
import unittest

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trainer.env import ROOT
from trainer.model import PolicyV1
from trainer.ppo import (HostRows, Outcomes, gae, main, pack_bits, parse_args, parser, resume_args, run_info,
                         seat_rows, unpack_bits)


class RunInfoTest(unittest.TestCase):
    def test_the_control_page_says_what_is_training(self):
        ranked = run_info(parse_args(["--run", "ppo-bc", "--setup", "ranked1v1", "--init", "bc.pt", "--opponent",
                                      "bc.pt", "--kl-coef", "0.05", "--workers", "16", "--games-per-worker", "2"]))
        self.assertEqual(ranked["Method"], "PPO (reinforcement learning)")
        self.assertEqual(ranked["Game"], "ranked 1v1 (the official playlist)")
        self.assertEqual((ranked["Against"], ranked["Starts from"]), ("frozen bc.pt", "bc.pt"))
        self.assertEqual(ranked["KL penalty to the start"], 0.05)
        self.assertEqual(ranked["Games at once"], "32 (16 workers x 2)")
        self.assertEqual(ranked["Learner seat"], "both, half the games each")
        onion = run_info(parse_args([]))
        self.assertEqual(onion["Game"], "Onion, Compact, 5 min")
        self.assertEqual(onion["Against"], "3 Easy nations")
        self.assertEqual((onion["Starts from"], onion["KL penalty to the start"]), ("random weights", "none"))
        self.assertNotIn("Learner seat", onion, "a single agent seat")


class ResumeArgsTest(unittest.TestCase):
    SAVED = vars(parse_args(["--run", "r", "--setup", "ranked1v1", "--lr", "1e-4", "--kl-coef", "0.2"]))

    def resume(self, *flags):
        return resume_args(parser(), ["--run", "r", "--resume", *flags], self.SAVED, ("updates", "kl_coef"))

    def test_a_resume_keeps_the_runs_settings(self):
        a = self.resume("--port", "0", "--updates", "9")
        self.assertEqual((a.setup, a.lr, a.kl_coef, a.updates, a.port, a.resume), ("ranked1v1", 1e-4, 0.2, 9, 0, True))

    def test_a_given_kl_coef_replaces_the_runs(self):
        self.assertEqual(self.resume("--kl-coef", "0.05").kl_coef, 0.05)

    def test_flags_a_resume_would_ignore_are_refused(self):
        with self.assertRaisesRegex(SystemExit, r"--lr would be ignored.*--init checkpoints/r/latest\.pt"):
            self.resume("--lr", "2.5e-4")  # the default value, but not the run's
        self.resume("--lr", "1e-4", "--setup", "ranked1v1")  # the run's own values pass


class SeatRowsTest(unittest.TestCase):
    def test_the_learner_plays_every_seat_equally(self):
        learner, others, seat = seat_rows(4, 2)
        self.assertEqual(learner.tolist(), [0, 3, 4, 7])  # game slot e, seat e % 2: row e * 2 + seat
        self.assertEqual([o.tolist() for o in others], [[1, 2, 5, 6]])
        self.assertEqual(seat.tolist(), [0, 1, 0, 1])
        per_seat = np.arange(8).reshape(4, 2)  # what the workers return per game and seat: rewards, dones
        self.assertEqual(per_seat[np.arange(4), seat.numpy()].tolist(), learner.tolist(), "the learner's own values")
        learner, others, seat = seat_rows(3, 1)
        self.assertEqual((learner.tolist(), others, seat.tolist()), ([0, 1, 2], [], [0, 0, 0]))


class OutcomesTest(unittest.TestCase):
    def test_results_survive_a_restart(self):
        o = Outcomes(rows=2)
        o.record([0.9, 0.9], [0.8, -0.8])
        o.finish([{"env": 0, "seat": 0, "reward": 1.0, "rank": 1, "ticks": 10},
                  {"env": 1, "seat": 0, "reward": -1.0, "rank": 2, "ticks": 20}])
        resumed = Outcomes(rows=2)
        resumed.load(o.state())
        self.assertEqual(resumed.summary(), o.summary(), "the win rate and value curves carry on after a resume")
        self.assertEqual(resumed.summary()["episodes/win_rate"], 0.5)

    def test_win_rate_per_seat(self):
        o = Outcomes(rows=2)
        o.finish([{"env": 0, "seat": 0, "reward": 1.0, "rank": 1, "ticks": 10},
                  {"env": 1, "seat": 1, "reward": -1.0, "rank": 2, "ticks": 20}])
        s = o.summary()
        self.assertEqual((s["episodes/win_rate_seat0"], s["episodes/win_rate_seat1"]), (1.0, 0.0))

    def test_value_outcome_correlation_by_phase(self):
        o = Outcomes(rows=4)
        rewards = [1.0, -1.0, -2 / 3, -5 / 6]
        o.record([0.1] * 4, [0.3, 0.3, -0.2, -0.2])  # early: only loosely related to the outcome
        o.record([0.9] * 4, [0.9 * r for r in rewards])  # late: tracks it
        o.finish([{"env": i, "seat": 0, "reward": r, "rank": 1, "ticks": 1} for i, r in enumerate(rewards)])
        s = o.summary()
        self.assertAlmostEqual(s["value/outcome_corr_late"], 1.0, places=5)
        self.assertLess(s["value/outcome_corr_early"], 0.9)
        self.assertNotIn("value/outcome_corr_mid", s)

    def test_no_correlation_when_every_game_ends_the_same(self):
        o = Outcomes(rows=2)
        o.record([0.9, 0.9], [-0.5, -0.8])
        o.finish([{"env": i, "seat": 0, "reward": -1.0, "rank": 4, "ticks": 1} for i in range(2)])
        s = o.summary()
        self.assertEqual(s["value/outcome_acc_late"], 1.0)  # trivially right: why the gate uses the correlation
        self.assertNotIn("value/outcome_corr_late", s)


class PackTest(unittest.TestCase):
    def test_cell_masks_come_back_exactly_from_their_bits(self):
        m = torch.rand(3, 12, 64, device="cuda") < 0.3
        b = pack_bits(m)
        self.assertEqual((b.dtype, tuple(b.shape)), (torch.uint8, (3, 12, 8)))
        self.assertTrue(torch.equal(unpack_bits(b), m))


class AllocatorTest(unittest.TestCase):
    def test_freed_gpu_memory_serves_a_larger_tensor(self):
        """The trainers run with expandable allocator segments. PyTorch's default kept a freed block reserved beside a
        larger new one: v4's PPO peak was 10.5 GB reserved by default and 9.9 GB with them (3B step 3)."""
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved()
        x = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
        del x
        y = torch.empty(300 * 2**20, dtype=torch.uint8, device="cuda")
        self.assertLess(torch.cuda.memory_reserved() - before, 400 * 2**20)
        del y


class HostRowsTest(unittest.TestCase):
    def test_minibatches_come_back_from_system_ram_as_the_rollout_wrote_them(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        T, N = 5, 4
        steps = [{"spatial": torch.randint(0, 65, (N, 3, 8, 8), device="cuda", generator=g).to(torch.uint8),
                  "scalars": torch.rand(N, 6, device="cuda", generator=g)} for _ in range(T)]
        host = HostRows(steps[0], T * N, 3)
        for t, o in enumerate(steps):
            host.put(t, o)
        flat = {k: torch.cat([o[k] for o in steps]) for k in steps[0]}
        idx = [mb for _ in range(2) for mb in torch.randperm(T * N).split(3)]  # two epochs, as PPO passes them
        batches = list(zip(idx, host.batches(idx, "cuda")))
        self.assertEqual(len(batches), 14)  # per epoch 6 minibatches of 3 and one of 2: each staging slot is reused
        for idx, o in batches:
            for k in flat:
                self.assertTrue(torch.equal(o[k], flat[k][idx.cuda()]), k)

    def test_every_epoch_reuses_the_gpu_memory_of_the_last(self):
        """PyTorch caches freed GPU memory per stream: with a new copy stream each epoch, every epoch's minibatches stayed
        reserved (2.5-3.7 GB over the in-use peak in 3B's final speed check)."""
        rows = {"x": torch.zeros(4, 2**20, dtype=torch.uint8, device="cuda")}  # 1 MiB a row
        host = HostRows(rows, 40, 8)
        for t in range(10):
            host.put(t, rows)
        reserved = []
        for _ in range(4):
            for o in host.batches(torch.randperm(40).split(8), "cuda"):
                pass
            torch.cuda.synchronize()
            reserved.append(torch.cuda.memory_reserved())
        self.assertEqual(reserved[-1], reserved[0])


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
        self.assertEqual((first["args"]["arch"], first["args"]["core"]), ("v4", "moe"), "a new run records its network")
        self.assertEqual(first["global_step"], 2 * 16 * 2)
        # 32 decisions do not finish a game: the unfinished games are skipped on resume, so no seed is played twice
        self.assertEqual(first["episodes"], [1, 1])
        self.assertEqual(first["outcomes"], {"episodes": [], "samples": []}, "no game finished yet")
        main(common + ["--updates", "3", "--resume"])
        second = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
        self.assertEqual(second["update"], 3)
        self.assertEqual(second["global_step"], 3 * 16 * 2)
        self.assertTrue(all(b >= a for a, b in zip(first["episodes"], second["episodes"])))
        self.assertTrue((ROOT / "runs" / self.RUN).is_dir())

    def test_two_seats_against_a_frozen_opponent_with_kl(self):
        """At learning rate 0 the learner stays the KL reference, so the KL must stay ~0: the reference's outputs, kept
        from the rollout, belong to the same rows. Random heads make each row's policy its own (a new network's zero
        heads are uniform on every row)."""
        init = ROOT / "checkpoints" / self.RUN / "init.pt"
        init.parent.mkdir(parents=True, exist_ok=True)
        dims = {"channels": 22, "playerFeatures": 17, "scalars": 16, "acts": 5}
        torch.manual_seed(0)
        net = PolicyV1(dims)
        for layer in (net.type_head, net.bin_head, net.shuffle, net.skip, net.query):
            torch.nn.init.normal_(layer.weight, std=1.0)
        torch.save({"model": net.state_dict(), "args": {"widths": "64,128,192"}}, init)
        main(["--run", self.RUN, "--workers", "1", "--games-per-worker", "2", "--steps", "32", "--minibatches", "2",
              "--epochs", "2", "--port", "0", "--timer", "1", "--nations", "0", "--seats", "2", "--lr", "0",
              "--init", str(init), "--opponent", str(init), "--kl-coef", "0.1", "--updates", "2"])
        state = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
        self.assertEqual(state["update"], 2)
        self.assertEqual(state["global_step"], 2 * 32 * 2, "only the learner's decisions count")
        self.assertEqual(state["args"]["kl_coef"], 0.1)
        logs = EventAccumulator(str(ROOT / "runs" / self.RUN))
        logs.Reload()
        kl = [e.value for e in logs.Scalars("loss/kl_ref")]
        self.assertEqual(len(kl), 2)
        self.assertLess(max(kl), 1e-4)  # 2e-7 here; reference rows shifted by one gave 4e-3 and 1.3e-2


if __name__ == "__main__":
    unittest.main()
