import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from trainer.control import Control, cpu_temp, serve


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


if __name__ == "__main__":
    unittest.main()
