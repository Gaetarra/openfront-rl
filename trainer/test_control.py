import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from trainer.control import REPLAYS, Control, cpu_temp, serve


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.c = Control()
        self.server = serve(self.c, 0, sampler=False)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, header=True):
        req = urllib.request.Request(self.base + path, data=b"", method="POST",
                                     headers={"X-Control": "1"} if header else {})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)

    def test_buttons_need_the_control_header(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post("/pause", header=False)
        self.assertEqual(e.exception.code, 403)
        self.assertFalse(self.c.paused)

    def test_pause_resume_throttle_save_stop(self):
        self.post("/pause")
        self.assertTrue(self.c.paused)
        self.post("/resume")
        self.assertFalse(self.c.paused)
        self.post("/throttle/half")
        self.assertEqual(self.c.duty, 0.5)
        self.post("/throttle/0.3")
        self.assertEqual(self.c.duty, 0.3)
        self.post("/save")
        self.assertTrue(self.c.take_save())
        self.assertFalse(self.c.take_save())
        self.post("/stop")
        self.assertFalse(self.c.wait())

    def test_bad_commands_are_rejected(self):
        for path in ("/throttle/fast", "/throttle/0", "/explode"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.post(path)
            self.assertEqual(e.exception.code, 400, path)

    def test_status_and_page(self):
        self.c.update(update=3, metrics={"episodes/win_rate": 0.5})
        with urllib.request.urlopen(self.base + "/status", timeout=5) as r:
            s = json.load(r)
        self.assertEqual(s["update"], 3)
        self.assertEqual(s["metrics"]["episodes/win_rate"], 0.5)
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:
            self.assertIn("<title>Training control</title>", r.read().decode())

    def test_history_follows_the_run_log(self):
        log_dir = Path(tempfile.mkdtemp()) / "run"  # the trainers start the server before their writer creates it
        server = serve(Control(), 0, sampler=False, log_dir=log_dir)

        def history():
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/history", timeout=10) as r:
                return r.read().decode()

        try:
            self.assertEqual(json.loads(history()), {})
            with SummaryWriter(log_dir) as w:
                w.add_scalar("episodes/win_rate", 0.25, 100)
                w.flush()
                self.assertEqual(json.loads(history()), {"episodes/win_rate": [[100, 0.25]]})
                w.add_scalar("episodes/win_rate", 0.5, 200)
                w.add_scalar("loss/kl_ref", float("nan"), 200)
                w.flush()
                body = history()
            self.assertEqual(json.loads(body)["episodes/win_rate"], [[100, 0.25], [200, 0.5]])
            self.assertNotIn("NaN", body, "the page's JSON.parse rejects NaN")
        finally:
            server.shutdown()
            server.server_close()
        with urllib.request.urlopen(self.base + "/history", timeout=5) as r:
            self.assertEqual(json.load(r), {}, "no run log, no curves")

    def test_wait_blocks_while_paused(self):
        self.c.command("pause")
        result = []
        t = threading.Thread(target=lambda: result.append(self.c.wait()))
        t.start()
        time.sleep(0.5)
        self.assertTrue(t.is_alive())
        self.c.command("resume")
        t.join(2)
        self.assertEqual(result, [True])

    def test_stop_while_paused_releases_wait(self):
        self.c.command("pause")
        result = []
        t = threading.Thread(target=lambda: result.append(self.c.wait()))
        t.start()
        self.c.command("stop")
        t.join(2)
        self.assertEqual(result, [False])

    def test_cpu_temp_without_monitor_is_none(self):
        self.assertIsNone(cpu_temp("http://127.0.0.1:9/data.json"))

    def test_saved_games_are_served_with_cors_and_commands_are_not(self):
        REPLAYS.mkdir(exist_ok=True)
        path = REPLAYS / "Test0001.json"
        path.write_text('{"info": {}}')
        try:
            with urllib.request.urlopen(f"{self.base}/game/Test0001", timeout=5) as r:
                self.assertEqual(json.load(r), {"info": {}})
                self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(f"{self.base}/game/Nope0001", timeout=5)
            self.assertEqual(e.exception.code, 404)
            preflight = urllib.request.Request(f"{self.base}/game/Test0001", method="OPTIONS")
            with urllib.request.urlopen(preflight, timeout=5) as r:
                self.assertEqual(r.status, 204)
                self.assertEqual(r.headers["Access-Control-Allow-Methods"], "GET")
            with self.assertRaises(urllib.error.HTTPError):  # no preflight approval for the buttons
                urllib.request.urlopen(urllib.request.Request(f"{self.base}/pause", method="OPTIONS"), timeout=5)
            with self.assertRaises(urllib.error.HTTPError):  # only game IDs, never paths
                urllib.request.urlopen(f"{self.base}/game/..%2F..%2FREADME", timeout=5)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
