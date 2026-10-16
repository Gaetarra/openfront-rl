import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import psutil
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trainer.env import FFA, HVN, ONION, ROOT
from trainer.model import PolicyV1
from trainer.ppo import (MAY_CHANGE, HostRows, Outcomes, cap_binds, gae, main, mix_of, pack_bits, parse_args, parser,
                         resume_args, run_info, seat_rows, unpack_bits)


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
        league = run_info(parse_args(["--setup", "ranked1v1", "--league", "a.pt", "b.pt", "--panel", "a.pt", "c.pt",
                                      "--kl-ref", "c.pt"]))
        self.assertTrue(league["Against"].startswith("a league of frozen policies"))
        self.assertEqual(league["KL reference"], "c.pt")
        self.assertEqual(league["Frozen evaluations"], "128 games against 2 policies every 250 updates")

    def test_a_mix_lists_each_setups_games_and_learner_seats(self):
        mixed = run_info(parse_args(["--mix", "ranked1v1=2", "ffa=1", "hvn=1", "--games-per-worker", "2",
                                     "--league", "a.pt", "b.pt"]))
        self.assertEqual(mixed["Game"], "a mix of setups: ranked1v1 (4 games), ffa (2 games), hvn (2 games)")
        self.assertEqual(mixed["Games at once"], "8 (4 workers x 2)", "the mix decides the workers")
        self.assertEqual(mixed["Learner seat"], "ranked1v1: one of 2, in turn; ffa: all 4; hvn: all 4")
        self.assertTrue(mixed["Against"].startswith("ranked1v1: a league of frozen policies"))
        self.assertTrue(mixed["Against"].endswith("ffa: the game's own nations; hvn: the game's own nations"))
        league = run_info(parse_args(["--mix", "ranked1v1=1", "ffa=1", "hvn=1", "--league", "a.pt", "--panel", "a.pt", "b.pt",
                                      "--eval-every", "500", "--eval-games-public", "16"]))
        self.assertEqual(league["Frozen evaluations"], "128 games against 2 policies every 500 updates, and 16 games of "
                                                       "each of ffa, hvn against the nations")

    def test_a_mix_names_playlists_and_counts_of_at_least_one_worker(self):
        self.assertEqual(mix_of(parse_args(["--mix", "ranked1v1=2", "ffa=3"])), {"ranked1v1": 2, "ffa": 3})
        for bad in (["onion=1"], ["ffa=0"], ["ffa"], ["ffa=x"], ["ffa=1", "ffa=2"]):
            with self.assertRaises(SystemExit, msg=bad):
                parse_args(["--mix", *bad])


class ResumeArgsTest(unittest.TestCase):
    SAVED = vars(parse_args(["--run", "r", "--setup", "ranked1v1", "--lr", "1e-4", "--kl-coef", "0.2"]))

    def resume(self, *flags):
        return resume_args(parser(), ["--run", "r", "--resume", *flags], self.SAVED, MAY_CHANGE)

    def test_a_resume_keeps_the_runs_settings(self):
        a = self.resume("--port", "0", "--updates", "9")
        self.assertEqual((a.setup, a.lr, a.kl_coef, a.updates, a.port, a.resume), ("ranked1v1", 1e-4, 0.2, 9, 0, True))

    def test_a_given_kl_coef_replaces_the_runs(self):
        self.assertEqual(self.resume("--kl-coef", "0.05").kl_coef, 0.05)

    def test_the_evaluation_settings_may_change(self):
        a = self.resume("--eval-every", "500", "--eval-games-public", "8")
        self.assertEqual((a.eval_every, a.eval_games_public), (500, 8))
        self.assertEqual((self.resume().eval_every, self.resume().eval_games_public), (250, 32), "older runs get defaults")

    def test_flags_a_resume_would_ignore_are_refused(self):
        with self.assertRaisesRegex(SystemExit, r"--lr would be ignored.*--init checkpoints/r/latest\.pt"):
            self.resume("--lr", "2.5e-4")  # the default value, but not the run's
        self.resume("--lr", "1e-4", "--setup", "ranked1v1")  # the run's own values pass


class SeatRowsTest(unittest.TestCase):
    def test_the_learner_plays_every_seat_equally(self):
        learner, others, slots, owner = seat_rows([2] * 4, [False] * 4)
        self.assertEqual(learner.tolist(), [0, 3, 4, 7])  # game slot e, seat e % 2: row e * 2 + seat
        self.assertEqual([o.tolist() for o in others], [[1, 2, 5, 6]])
        self.assertEqual((slots, owner), ([0, 1, 2, 3], [(0, 0), (1, 1), (2, 0), (3, 1)]))
        per_seat = np.arange(8).reshape(4, 2)  # what the workers return per game and seat: rewards, dones
        self.assertEqual([per_seat[e, s] for e, s in owner], learner.tolist(), "the learner's own values")
        learner, others, slots, owner = seat_rows([1, 1, 1], [False] * 3)
        self.assertEqual((learner.tolist(), others, slots, owner), ([0, 1, 2], [], [0, 1, 2], [(0, 0), (1, 0), (2, 0)]))

    def test_the_learner_plays_one_seat_of_ranked_games_and_every_seat_of_ffa_and_hvn(self):
        learner, others, slots, owner = seat_rows([2, 2, 4, 4], [False, False, True, True])
        self.assertEqual(learner.tolist(), [0, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual([o.tolist() for o in others], [[1, 2]])
        self.assertEqual(slots, [0, 1])
        self.assertEqual(owner[:3], [(0, 0), (1, 1), (2, 0)])
        uniform = seat_rows([2] * 4, [False] * 4)  # today's league: seat e % 2
        self.assertEqual(uniform[0].tolist(), [0, 3, 4, 7])
        learner, others, slots, owner = seat_rows([4, 4], [True, True])
        self.assertEqual((learner.tolist(), others, slots), (list(range(8)), [], []), "no frozen opponent at all")


class CapBindsTest(unittest.TestCase):
    def test_the_attention_cap_binds_above_63_real_players(self):
        players = torch.zeros(3, 80, 17)
        players[:, 0, 0] = 1  # row 0, the unowned land, exists
        players[:, 1:64, 0] = 1  # 63 players exist: ...
        players[:, 1, 2] = 1  # ... the seat itself,
        players[:, 2:30, 3] = 1  # friends,
        players[:, 30:64, 4] = 1  # and enemies
        players[1, 64, 0], players[1, 64, 4] = 1, 1  # row 1 has a 64th
        players[2, 64:74, 0], players[2, 64:74, 5] = 1, 1  # row 2 has 10 bots instead: they do not count
        self.assertEqual(cap_binds(players).tolist(), [False, True, False], "63 players and row 0 fit the 64 slots")


class OutcomesTest(unittest.TestCase):
    def test_results_survive_a_restart(self):
        o = Outcomes(rows=2)
        o.record([0.9, 0.9], [0.8, -0.8])
        o.finish([{"row": 0, "setup": "s", "env": 0, "seat": 0, "reward": 1.0, "rank": 1, "ticks": 10},
                  {"row": 1, "setup": "s", "env": 1, "seat": 0, "reward": -1.0, "rank": 2, "ticks": 20}])
        resumed = Outcomes(rows=2)
        resumed.load(o.state())
        self.assertEqual(resumed.summary(), o.summary(), "the win rate and value curves carry on after a resume")
        self.assertEqual(resumed.summary()["episodes/win_rate"], 0.5)

    def test_win_rate_per_seat(self):
        o = Outcomes(rows=2)
        o.finish([{"row": 0, "setup": "s", "env": 0, "seat": 0, "reward": 1.0, "rank": 1, "ticks": 10},
                  {"row": 1, "setup": "s", "env": 1, "seat": 1, "reward": -1.0, "rank": 2, "ticks": 20}])
        s = o.summary()
        self.assertFalse([k for k in s if k.startswith("setups/")], "one setup: no per-setup tags")
        self.assertEqual((s["episodes/win_rate_seat0"], s["episodes/win_rate_seat1"]), (1.0, 0.0))

    def test_value_outcome_correlation_by_phase(self):
        o = Outcomes(rows=4)
        rewards = [1.0, -1.0, -2 / 3, -5 / 6]
        o.record([0.1] * 4, [0.3, 0.3, -0.2, -0.2])  # early: only loosely related to the outcome
        o.record([0.9] * 4, [0.9 * r for r in rewards])  # late: tracks it
        o.finish([{"row": i, "setup": "s", "env": i, "seat": 0, "reward": r, "rank": 1, "ticks": 1}
                  for i, r in enumerate(rewards)])
        s = o.summary()
        self.assertAlmostEqual(s["value/outcome_corr_late"], 1.0, places=5)
        self.assertLess(s["value/outcome_corr_early"], 0.9)
        self.assertNotIn("value/outcome_corr_mid", s)

    def test_no_correlation_when_every_game_ends_the_same(self):
        o = Outcomes(rows=2)
        o.record([0.9, 0.9], [-0.5, -0.8])
        o.finish([{"row": i, "setup": "s", "env": i, "seat": 0, "reward": -1.0, "rank": 4, "ticks": 1}
                  for i in range(2)])
        s = o.summary()
        self.assertEqual(s["value/outcome_acc_late"], 1.0)  # trivially right: why the gate uses the correlation
        self.assertNotIn("value/outcome_corr_late", s)

    def test_each_setup_keeps_its_own_window_and_it_survives_a_restart(self):
        def game(row, setup, rank, ticks):
            return {"row": row, "setup": setup, "env": row, "seat": 0, "reward": 1.0 if rank == 1 else -1.0,
                    "rank": rank, "ticks": ticks}
        o = Outcomes(rows=2, window=2)
        o.finish([game(0, "ranked1v1", 1, 100)])
        o.finish([game(1, "ffa", 3, 200), game(1, "ffa", 4, 300), game(1, "ffa", 1, 400)])
        s = o.summary()
        self.assertEqual((s["setups/win_rate/ranked1v1"], s["setups/ticks/ranked1v1"]), (1.0, 100.0))
        self.assertEqual((s["setups/win_rate/ffa"], s["setups/mean_reward/ffa"], s["setups/ticks/ffa"]),
                         (0.5, 0.0, 350.0), "the last 2 of the setup's games; the other setup's never pushed them out")
        self.assertEqual(s["episodes/win_rate"], 0.5, "the overall window still holds the last 2 games of any setup")
        self.assertNotIn("episodes/win_rate_seat0", s, "seats mean different things in different setups")
        self.assertEqual(sorted(o.state()["by_setup"]), ["ffa", "ranked1v1"])
        resumed = Outcomes(rows=2, window=2)
        resumed.load(o.state())
        self.assertEqual(resumed.summary(), s)

    def test_a_mixed_run_logs_each_setups_share_of_cut_games(self):
        """4N: a game the env's backstop ended is decided by tiles, so a setup whose win rate rests on them shows it.
        Games from before the flag (older checkpoints) count as not cut."""
        def game(row, setup, **extra):
            return {"row": row, "setup": setup, "env": row, "seat": 0, "reward": 1.0, "rank": 1, "ticks": 10, **extra}
        o = Outcomes(rows=2)
        o.finish([game(0, "ffa", cut=False)] + [game(0, "ffa", cut=True)] * 3)
        self.assertFalse([k for k in o.summary() if "cut_share" in k], "one setup: no per-setup tags")
        o.finish([game(1, "ranked1v1")])
        s = o.summary()
        self.assertEqual((s["setups/cut_share/ffa"], s["setups/cut_share/ranked1v1"]), (0.75, 0.0))


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

    def test_every_learning_pass_reuses_the_thread_of_the_last(self):
        """A thread's first parallel op starts an OpenMP team that outlives the thread: with a new thread per pass, the
        first league run leaked 13 threads and ~5.4 MiB an update, ~21 GiB in all (PLAN 4A, "First league run")."""
        rows = {"x": torch.zeros(4, 2**20, dtype=torch.uint8, device="cuda")}  # 1 MiB rows: gathered in parallel
        host = HostRows(rows, 40, 8)
        for t in range(10):
            host.put(t, rows)
        threads = []
        for _ in range(4):
            for o in host.batches(torch.randperm(40).split(8), "cuda"):
                pass
            threads.append(psutil.Process().num_threads())
        self.assertEqual(threads[-1], threads[0])


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

    def test_evaluations_come_at_a_multiple_of_the_snapshots(self):
        for bad in ("0", "100", "300"):
            with self.assertRaisesRegex(SystemExit, "--eval-every must be a multiple of 250"):
                main(["--run", self.RUN, "--eval-every", bad])

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
        self.assertEqual(first["outcomes"], {"episodes": [], "samples": [], "by_setup": {}}, "no game finished yet")
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

    def test_a_league_with_frozen_evaluations(self):
        """PLAN 4A: the opponents come from a pool the learner's weight averages join, and every average plays the
        panel, here with a snapshot every 2 updates and a pick every update. Games of 1 minute end within 3 updates of
        48 steps, so slots change opponents mid-rollout."""
        ckpt, run = ROOT / "checkpoints" / self.RUN, ROOT / "runs" / self.RUN
        ckpt.mkdir(parents=True, exist_ok=True)
        pool = []
        for k in range(2):
            torch.manual_seed(k)
            net = PolicyV1({"channels": 22, "playerFeatures": 17, "scalars": 16, "acts": 5})
            pool.append((ckpt / f"pool{k}.pt").as_posix())
            torch.save({"model": net.state_dict(), "args": {"widths": "64,128,192"}, "update": 0}, pool[-1])
        common = ["--run", self.RUN, "--workers", "1", "--games-per-worker", "2", "--steps", "48", "--minibatches",
                  "2", "--epochs", "1", "--port", "0", "--timer", "1", "--nations", "0", "--seats", "2"]
        with mock.patch("trainer.ppo.SNAPSHOT", 2), mock.patch("trainer.ppo.ROTATE", 1):
            main(common + ["--init", pool[0], "--league", *pool, "--panel", *pool, "--eval-games", "2",
                           "--save-every", "1", "--updates", "4"])
            main(common + ["--resume", "--updates", "6", "--eval-games", "4"])
        state = torch.load(ckpt / "latest.pt", weights_only=False)
        league = state["league"]
        self.assertEqual([Path(p).name for p in league["pool"]],
                         ["pool0.pt", "pool1.pt", "avg000002.pt", "avg000004.pt", "avg000006.pt"])
        avg = torch.load(ckpt / "avg000006.pt", weights_only=False)
        members = [(ckpt / f"update00000{k}.pt").as_posix() for k in (6, 4, 2)] + [pool[0]]
        self.assertEqual(avg["averaged"], members, "the last 5 kept checkpoints at most, --init as update 0")
        models = [torch.load(m, map_location="cpu", weights_only=False)["model"] for m in members]
        for k, v in avg["model"].items():
            torch.testing.assert_close(v, (sum(m[k].double() for m in models) / len(models)).to(v.dtype))
        finished = len(state["outcomes"]["episodes"])
        self.assertGreater(finished, 0)
        self.assertEqual(sum(map(len, league["results"].values())), finished, "every finished game counts for PFSP")
        evals = [json.loads(line) for line in (run / "panel.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["update"] for e in evals], [0, 2, 4, 6], "--init's, then every snapshot's, once")
        self.assertEqual([sum(v["episodes"] for v in e["vs"].values()) for e in evals], [2, 2, 2, 4])
        self.assertEqual([list(e["vs"]) for e in evals], [pool] * 4)
        logs = EventAccumulator(str(run))
        logs.Reload()
        self.assertEqual(len(logs.Scalars("eval/rating")), 4)
        self.assertEqual(len(logs.Scalars(f"eval/vs_{self.RUN}/pool1")), 4)

    def test_a_mixed_run_learns_from_every_ffa_and_hvn_seat(self):
        """4N: a ranked game gives the learner one seat (the league's pick plays the other), an FFA or Humans vs Nations
        game all four. The patched ranked entry is a 2-seat Onion game: the rows follow the setup's name, not its
        config. It is listed second, so it is game 1 but the league's slot 0. Games of 1 minute end at tick ~800 when
        someone wins, but a Humans vs Nations game the nations lead runs to the backstop at tick 1400, 280 steps: at 96
        steps an update, every setup finishes a game."""
        init = ROOT / "checkpoints" / self.RUN / "init.pt"
        init.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        net = PolicyV1({"channels": 22, "playerFeatures": 17, "scalars": 16, "acts": 5})
        torch.save({"model": net.state_dict(), "args": {"widths": "64,128,192"}}, init)
        playlists = {"ranked1v1": {**ONION, "seats": 2, "nations": 0, "timerMinutes": 1},
                     "ffa": {**FFA, "timerMinutes": 1}, "hvn": {**HVN, "timerMinutes": 1}}
        with mock.patch.dict("trainer.env.PLAYLISTS", playlists), mock.patch("trainer.ppo.SNAPSHOT", 2):
            main(["--run", self.RUN, "--mix", "ffa=1", "ranked1v1=1", "hvn=1", "--games-per-worker", "1",
                  "--steps", "96", "--minibatches", "2", "--epochs", "1", "--port", "0", "--init", str(init),
                  "--league", str(init), "--panel", str(init), "--eval-every", "4", "--eval-games", "2",
                  "--eval-games-public", "1", "--save-every", "1", "--updates", "4"])
            main(["--run", self.RUN, "--resume", "--port", "0", "--updates", "5", "--eval-every", "6",
                  "--eval-games-public", "2"])
        state = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
        self.assertEqual(state["update"], 5)
        self.assertEqual(state["global_step"], 5 * 96 * 9, "1 + 4 + 4 learner rows")
        self.assertEqual((state["args"]["mix"], state["args"]["workers"]), (["ffa=1", "ranked1v1=1", "hvn=1"], 3))
        self.assertEqual((state["args"]["eval_every"], state["args"]["eval_games_public"]), (6, 2),
                         "a resume may change both")
        log = (ROOT / "runs" / self.RUN / "panel.jsonl").read_text(encoding="utf-8")
        evals = [json.loads(line) for line in log.splitlines()]
        self.assertEqual([e["update"] for e in evals], [0, 4], "every --eval-every updates, --init's first")
        for e in evals:
            self.assertEqual(sorted(e["nations"]), ["ffa", "hvn"], "frozen games against the nations, per setup")
            self.assertEqual([e["nations"][s]["episodes"] for s in ("ffa", "hvn")], [3, 3])
            self.assertTrue(all(0 <= n["cut_share"] <= 1 for n in e["nations"].values()))
        logs = EventAccumulator(str(ROOT / "runs" / self.RUN))
        logs.Reload()
        tags = logs.Tags()["scalars"]
        self.assertEqual([len(logs.Scalars(f"eval/nations_{s}")) for s in ("ffa", "hvn")], [2, 2])
        for tag in ("setups/win_rate/ranked1v1", "setups/win_rate/ffa", "setups/win_rate/hvn", "setups/cap_binds/ffa",
                    "setups/dead_share/hvn", "setups/cut_share/hvn", "actions/hvn/attack", "actions/attack"):
            self.assertIn(tag, tags)
        by_setup = state["outcomes"]["by_setup"]
        for setup, seats in (("ffa", 4), ("hvn", 4), ("ranked1v1", 1)):
            eps = by_setup[setup]
            self.assertTrue(eps and len(eps) % seats == 0, f"{setup}: whole games, one episode per learner seat")
            for i in range(0, len(eps), seats):  # a finished game's seats come together
                game = eps[i:i + seats]
                self.assertEqual(len({(e["env"], e["ticks"]) for e in game}), 1, setup)
                if seats == 4:
                    self.assertEqual([e["seat"] for e in game], [0, 1, 2, 3], setup)
                if setup == "hvn":
                    self.assertEqual(len({e["reward"] for e in game}), 1, "a Humans vs Nations team shares one result")
        self.assertEqual(sum(map(len, state["league"]["results"].values())), len(by_setup["ranked1v1"]),
                         "every finished ranked game counts for PFSP, though its game number is not its league slot")


if __name__ == "__main__":
    unittest.main()
