import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from trainer import bc, crawl
from trainer.bc import Meter, bc_loss, is_val, label_legal, main
from trainer.env import RANKED_1V1, ROOT, VecEnv
from trainer.model import Policy, PolicyV1
from trainer.test_model import DIMS

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
        a = bc.parse_args(["--workers", "14", "--games-per-worker", "6"])
        a.arch = "v2"
        info = bc.run_info(a, ["r"] * 3, ["v"])
        self.assertEqual(info["Method"], "behaviour cloning (imitating humans)")
        self.assertEqual(info["Network"], "player transformer, Soft-MoE core (8 experts)")
        self.assertEqual(info["Starts from"], "random weights")
        self.assertEqual(info["Data"], "3 ranked 1v1 replays, 1 held out")
        self.assertEqual(info["Games at once"], "84 (14 workers x 6)")

    def test_meter_averages_and_resets(self):
        m = Meter()
        m.add("a", torch.tensor(3.0), 2)
        m.add("a", torch.tensor(1.0), 2)
        self.assertEqual(m.means(), {"a": 1.0})
        self.assertEqual(m.means(), {})


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
        model = Policy(dims, core="moe").cuda().to(memory_format=torch.channels_last)
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
                       "--val-share", "0", "--port", "0", "--log-every", "1", "--save-every", "2"]
        self.seeds = []

    def tearDown(self):
        for d in ("checkpoints", "runs"):
            shutil.rmtree(ROOT / d / self.RUN, ignore_errors=True)
        shutil.rmtree(self.data, ignore_errors=True)

    def run_main(self, argv):
        real = bc.VecEnv

        def spy(*args, **kwargs):
            self.seeds.append(kwargs["seed"])
            return real(*args, **kwargs)

        with mock.patch.multiple(crawl, DATA=self.data, RECORDS=self.data / "records",
                                 INDEX=self.data / "index.jsonl"), mock.patch.object(bc, "VecEnv", spy):
            main(argv)

    def latest(self):
        return torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)

    def test_train_save_and_resume(self):
        self.run_main(self.common + ["--updates", "3"])
        first = self.latest()
        self.assertEqual((first["kind"], first["update"]), ("bc", 3))
        self.assertEqual((first["args"]["arch"], first["args"]["core"]), ("v2", "moe"))
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

if __name__ == "__main__":
    unittest.main()
