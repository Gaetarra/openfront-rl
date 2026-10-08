import argparse
import contextlib
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
from trainer.bc import (Meter, backward_windows, bc_loss, is_val, label_legal, lr_at, main, measure, one_pass, score,
                        table)
from trainer.env import RANKED_1V1, ROOT, VecEnv
from trainer.model import ATTACK, BUILD, SHARES, ActionDist, Policy, PolicyV1, new_net
from trainer.test_model import DIMS, N as ROWS, exact_fp32, fake_obs, randomized

FIXTURE = ROOT / "bridge" / "testdata" / "ranked1v1.json.gz"


class LabelTest(unittest.TestCase):
    def test_label_legal(self):
        obs = {"type_mask": torch.ones(4, 5, dtype=torch.bool),
               "cell_mask": torch.zeros(4, 3, 16, dtype=torch.bool),
               "attack_mask": torch.zeros(4, 5, dtype=torch.bool)}
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
        info = bc.run_info(a, ["r"] * 3, ["v"])
        self.assertEqual(info["Method"], "behaviour cloning (imitating humans)")
        self.assertEqual(info["Network"], "player transformer, Soft-MoE core (8 experts), 100 troop shares")
        self.assertEqual(info["Starts from"], "random weights")
        self.assertEqual(info["Data"], "3 ranked 1v1 replays, 1 held out")
        self.assertEqual(info["Games at once"], "84 (14 workers x 6)")
        self.assertEqual(info["Updates"], "10 of 8 windows each")

    def test_meter_averages_and_resets(self):
        m = Meter()
        m.add("a", torch.tensor(3.0), 2)
        m.add("a", torch.tensor(1.0), 2)
        self.assertEqual(m.means(), {"a": 1.0})
        self.assertEqual(m.means(), {})


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


class MeasureTest(unittest.TestCase):
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


class ScoreTest(unittest.TestCase):
    def test_scores_are_repeatable_and_per_head(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, args in (("old", {"widths": "64,128,192"}),
                               ("moe", {"arch": "v2", "core": "moe", "widths": "64,128,192"}),
                               ("shares", {"arch": "v3", "core": "moe", "widths": "64,128,192"})):
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
        old, moe, shares = first["metrics"]
        for k in ("bc/nll", "bc/nll_type", "bc/nll_compact", "bc/type_acc", "bc/value_mse", "value/outcome_corr"):
            self.assertIn(k, old)
        self.assertFalse(any(k.startswith("router/") for k in old), "the Phase 2 network has no router")
        self.assertAlmostEqual(sum(moe[f"router/compact_{i}"] for i in range(8)), 1.0, places=4)
        self.assertIn("bc/nll_shares", shares)
        self.assertNotIn("bc/nll_shares", moe)
        self.assertIn("old 1", text)
        self.assertIn("bc/nll_type", text)


class MainTest(unittest.TestCase):
    RUN = "unittest-bc"

    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        (self.data / "records").mkdir()
        shutil.copy(FIXTURE, self.data / "records" / "fixture.json.gz")
        entry = {"id": "fixture", "start": 0, "commit": None, "map": "x", "size": "Compact", "turns": 1,
                 "players": [{"clientID": "a", "publicID": None}, {"clientID": "b", "publicID": None}], "winner": "a"}
        (self.data / "index.jsonl").write_text(json.dumps(entry) + "\n")
        self.common = ["--run", self.RUN, "--workers", "1", "--games-per-worker", "1", "--val-workers", "0",
                       "--val-share", "0", "--port", "0", "--log-every", "1", "--save-every", "2", "--accum", "2",
                       "--warmup", "1"]
        self.seeds = []

    def tearDown(self):
        for d in ("checkpoints", "runs"):
            for run in (self.RUN, f"{self.RUN}-moe-swiglu"):
                shutil.rmtree(ROOT / d / run, ignore_errors=True)
        shutil.rmtree(self.data, ignore_errors=True)

    def run_main(self, argv):
        real = bc.VecEnv

        def spy(*args, **kwargs):
            self.seeds.append(kwargs["seed"])
            return real(*args, **kwargs)

        with mock.patch.multiple(crawl, DATA=self.data, RECORDS=self.data / "records",
                                 INDEX=self.data / "index.jsonl"), mock.patch.object(bc, "VecEnv", spy):
            main(argv)

    def latest(self, run=None):
        return torch.load(ROOT / "checkpoints" / (run or self.RUN) / "latest.pt", weights_only=False)

    def test_train_save_and_resume(self):
        self.run_main(self.common + ["--updates", "3"])
        first = self.latest()
        self.assertEqual((first["kind"], first["update"]), ("bc", 3))
        self.assertEqual((first["args"]["arch"], first["args"]["core"]), ("v4", "moe"))
        self.assertEqual(first["global_step"], 3 * 2 * 2)  # 3 updates of 2 windows of 2 rows
        self.assertEqual(self.seeds[0], "bc-1-u0", "runs with one seed replay the same records in the same order")
        self.run_main(self.common + ["--updates", "4", "--resume"])
        self.assertEqual(self.latest()["update"], 4)
        self.assertTrue((ROOT / "runs" / self.RUN).is_dir())
        self.assertNotEqual(self.seeds[0], self.seeds[1], "a resumed run replays the records in a new order")

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

    def test_old_bc_checkpoints_cannot_resume(self):
        path = ROOT / "checkpoints" / self.RUN / "latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"args": {"run": self.RUN, "widths": "64,128,192"}, "update": 5}, path)  # no "accum"
        with self.assertRaisesRegex(SystemExit, "predates.*--init"):
            self.run_main(["--run", self.RUN, "--resume"])

if __name__ == "__main__":
    unittest.main()
