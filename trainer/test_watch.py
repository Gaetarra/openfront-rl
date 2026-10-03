import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import psutil

from trainer import watch


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ClientTest(unittest.TestCase):
    def test_the_client_is_started_once_and_stopped_with_its_children(self):
        # Like npm run dev: a parent process whose child serves the page.
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        child = f"import subprocess, sys; subprocess.run([sys.executable, '-m', 'http.server', '{port}', '--bind', '127.0.0.1'])"
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "client.log"
            proc = watch.start_client(log, cmd=[sys.executable, "-c", child], url=url, timeout=60)
            self.assertTrue(watch.answers(url))
            self.assertIsNone(watch.start_client(log, cmd=[sys.executable, "-c", "pass"], url=url))  # already running
            children = psutil.Process(proc.pid).children(recursive=True)
            self.assertTrue(children)
            watch.stop(proc)
            self.assertFalse(watch.answers(url))
            self.assertFalse(any(c.is_running() for c in children))



class DirTest(unittest.TestCase):
    def test_a_missing_folder_stops_before_anything_starts(self):
        with mock.patch.object(watch, "start_client") as start, mock.patch.object(watch, "serve") as serve:
            with self.assertRaisesRegex(SystemExit, "no folder .*no such matchup"):
                watch.main(["--dir", "no such matchup"])
        start.assert_not_called()
        serve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
