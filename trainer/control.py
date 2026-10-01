"""Training control for a shared PC: a local HTTP API (127.0.0.1 only) with a status page, and a system sampler.

The CLI is `python -m trainer.ctl`. Pausing leaves the Node workers idle and frees the GPU's compute."""
import collections
import json
import re
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil
import torch

REPLAYS = Path(__file__).resolve().parents[1] / "replays"  # saved games (trainer.eval --record), served at /game/<id>
GAME_PATH = re.compile(r"^/game/([A-Za-z0-9]{8,10})$")

THROTTLE = {"full": 1.0, "half": 0.5, "low": 0.2}


def parse_duty(arg):
    if arg in THROTTLE:
        return THROTTLE[arg]
    try:
        duty = float(arg)
    except (TypeError, ValueError):
        duty = 0.0
    if not 0 < duty <= 1:
        raise ValueError(f"throttle takes full, half, low or a number in (0, 1], got {arg!r}")
    return duty


class Control:
    def __init__(self, duty=1.0, gpu_temp_limit=0):
        self._lock = threading.Lock()
        self.paused = self.auto_paused = self.stop_requested = self._save = False
        self.duty = duty
        self.gpu_temp_limit = gpu_temp_limit
        self.state = "starting"
        self.info = {}
        self.system = {}
        self.log = collections.deque(maxlen=30)
        self.started = time.time()

    def command(self, name, arg=None):
        with self._lock:
            if name == "pause":
                self.paused, self.auto_paused = True, False
            elif name == "resume":
                self.paused = self.auto_paused = False
            elif name == "save":
                self._save = True
            elif name == "stop":
                self.stop_requested = True
            elif name == "throttle":
                self.duty = parse_duty(arg)
            else:
                raise ValueError(f"unknown command {name!r}")
        self.note(name if arg is None else f"{name} {arg}")
        return name

    def wait(self):
        """Called between environment steps: blocks while paused; returns False once a stop was requested."""
        if self.paused and not self.stop_requested:
            self.state = "paused"
            torch.cuda.empty_cache()
            while self.paused and not self.stop_requested:
                time.sleep(0.2)
            self.state = "running"
        return not self.stop_requested

    def throttle(self, busy_s):
        """Sleeps so that work takes `duty` of the wall time."""
        if self.duty < 1:
            time.sleep(busy_s * (1 / self.duty - 1))

    def take_save(self):
        with self._lock:
            requested, self._save = self._save, False
        return requested

    def set_state(self, state):
        self.state = state

    def update(self, **info):
        self.info.update(info)

    def note(self, msg):
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")

    def status(self):
        return {"state": self.state, "paused": self.paused, "auto_paused": self.auto_paused, "duty": self.duty,
                "gpu_temp_limit": self.gpu_temp_limit, "uptime_s": time.time() - self.started, **self.info,
                "system": self.system, "log": list(self.log)}

    def start_sampler(self, period=5.0):
        def loop():
            while True:
                self.system = sample_system()
                temp = (self.system.get("gpu") or {}).get("temp")
                if self.gpu_temp_limit and temp is not None:
                    if temp >= self.gpu_temp_limit and not self.paused:
                        self.paused = self.auto_paused = True
                        self.note(f"auto-pause: GPU at {temp:.0f} °C")
                    elif self.auto_paused and temp <= self.gpu_temp_limit - 8:
                        self.paused = self.auto_paused = False
                        self.note(f"auto-resume: GPU at {temp:.0f} °C")
                time.sleep(period)

        threading.Thread(target=loop, daemon=True).start()


def gpu_status():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,"
                              "power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5).stdout
        util, used, total, temp, power = (float(v) for v in out.strip().splitlines()[0].split(","))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return {"util": util, "mem_used_gb": used / 1024, "mem_total_gb": total / 1024, "temp": temp, "power_w": power}


def cpu_temp(url="http://127.0.0.1:8085/data.json"):
    """CPU package temperature from LibreHardwareMonitor's web server (Options > Remote Web Server), else None:
    Windows does not expose CPU temperatures reliably."""
    try:
        with urllib.request.urlopen(url, timeout=0.5) as r:
            stack = [json.load(r)]
    except (OSError, ValueError):
        return None
    while stack:
        node = stack.pop()
        if node.get("Text", "").startswith("CPU Package") and "°C" in node.get("Value", ""):
            try:
                return float(node["Value"].split()[0].replace(",", "."))
            except ValueError:
                return None
        stack.extend(node.get("Children", []))
    return None


def sample_system():
    me = psutil.Process()
    rss = 0
    for p in [me, *me.children(recursive=True)]:
        try:
            rss += p.memory_info().rss
        except psutil.Error:
            pass
    vm = psutil.virtual_memory()
    return {"cpu_percent": psutil.cpu_percent(), "ram_used_gb": vm.used / 2**30, "ram_total_gb": vm.total / 2**30,
            "training_rss_gb": rss / 2**30, "gpu": gpu_status(), "cpu_temp": cpu_temp()}


def serve(control, port, sampler=True):
    """Starts the control server on 127.0.0.1:port (0 = any free port) in a daemon thread."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json", cors=False):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if cors:
                self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            game = GAME_PATH.match(self.path)
            if game:
                # For the game's own client: in dev mode it loads archived games from localhost:8787/game/<id>.
                path = REPLAYS / f"{game[1]}.json"
                if not path.exists():
                    return self._send(404, "{}", cors=True)
                return self._send(200, path.read_text(encoding="utf-8"), cors=True)
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/status":
                self._send(200, json.dumps(control.status()))
            else:
                self._send(404, "{}")

        def do_OPTIONS(self):
            # CORS preflight: other pages may read saved games and nothing else (the buttons stay same-page, do_POST).
            if not GAME_PATH.match(self.path):
                return self._send(404, "{}")
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_POST(self):
            # Browsers cannot add a custom header to a cross-site request without a CORS preflight, which this server
            # never approves: other web pages open in the browser cannot press these buttons.
            if self.headers.get("X-Control") != "1":
                return self._send(403, json.dumps({"error": "missing X-Control header"}))
            parts = self.path.strip("/").split("/")
            try:
                done = control.command(parts[0], parts[1] if len(parts) > 1 else None)
            except ValueError as e:
                return self._send(400, json.dumps({"error": str(e)}))
            self._send(200, json.dumps({"ok": done}))

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if sampler:
        control.start_sampler()
    return server


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Training control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg: #f6f7f9; --card: #fff; --text: #1d2330; --muted: #677084; --accent: #2f6fe4; --hot: #c2410c;
  --line: #e3e6eb; }
@media (prefers-color-scheme: dark) { :root { --bg: #14171c; --card: #1d2128; --text: #e6e9ef; --muted: #98a1b3;
  --accent: #6b9bff; --hot: #fb923c; --line: #2c323c; } }
body { margin: 0; font: 15px/1.45 system-ui, sans-serif; background: var(--bg); color: var(--text); }
main { max-width: 980px; margin: 0 auto; padding: 16px; }
header { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; justify-content: space-between; }
h1 { font-size: 20px; margin: 8px 0; }
#state { font-weight: 600; }
.buttons { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }
button { font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid var(--line); background: var(--card);
  color: var(--text); cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 0 0 8px; }
table { width: 100%; border-collapse: collapse; }
td { padding: 3px 0; }
td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.hot { color: var(--hot); font-weight: 600; }
#log { font: 12px/1.5 ui-monospace, monospace; color: var(--muted); white-space: pre-wrap; }
p.hint { color: var(--muted); font-size: 13px; }
</style></head><body><main>
<header><h1>Training control</h1><div>State: <span id="state">…</span></div></header>
<div class="buttons">
  <button class="primary" onclick="send('resume')">Resume</button>
  <button onclick="send('pause')">Pause</button>
  <button onclick="send('save')">Save now</button>
  <button onclick="if (confirm('Save and stop training?')) send('stop')">Save &amp; quit</button>
  <button onclick="send('throttle/full')">Full speed</button>
  <button onclick="send('throttle/half')">Half speed</button>
  <button onclick="send('throttle/low')">Low speed</button>
</div>
<div class="grid">
  <section><h2>Training</h2><table id="training"></table></section>
  <section><h2>Computer</h2><table id="system"></table></section>
</div>
<section style="margin-top:12px"><h2>Events</h2><div id="log"></div></section>
<p class="hint">Charts: run <code>.venv/Scripts/tensorboard --logdir runs</code> and open
<a href="http://localhost:6006">localhost:6006</a>.</p>
</main>
<script>
const fmt = (v, d = 2) => v === null || v === undefined ? "–" : typeof v === "number" ? v.toFixed(d) : v;
function rows(id, items) {
  document.getElementById(id).innerHTML = items.map(([k, v, hot]) =>
    `<tr><td>${k}</td><td class="${hot ? "hot" : ""}">${v}</td></tr>`).join("");
}
async function send(cmd) {
  await fetch("/" + cmd, { method: "POST", headers: { "X-Control": "1" } });
  refresh();
}
async function refresh() {
  try {
    const s = await (await fetch("/status")).json();
    const m = s.metrics || {}, sys = s.system || {}, g = sys.gpu || {};
    document.getElementById("state").textContent = s.state + (s.auto_paused ? " (GPU too hot)" : "") +
      (s.duty < 1 ? ` · throttled to ${Math.round(s.duty * 100)}%` : "");
    rows("training", [
      ["Update", s.update ?? "–"], ["Decisions", s.global_step ?? "–"],
      ["Decisions / s", fmt(m["perf/decisions_per_s"], 0)],
      ["Win rate (last 200 games)", fmt(m["episodes/win_rate"])], ["Mean reward", fmt(m["episodes/mean_reward"])],
      ["Mean rank", fmt(m["episodes/mean_rank"])], ["Value vs outcome correlation, late game", fmt(m["value/outcome_corr_late"])],
      ["Entropy", fmt(m["loss/entropy"], 3)], ["Invalid actions", fmt(m["actions/invalid_rate"], 3)]]);
    rows("system", [
      ["CPU load", fmt(sys.cpu_percent, 0) + " %"],
      ["CPU temperature", sys.cpu_temp == null ? "needs LibreHardwareMonitor" : fmt(sys.cpu_temp, 0) + " °C",
       sys.cpu_temp >= 90],
      ["RAM", `${fmt(sys.ram_used_gb, 1)} / ${fmt(sys.ram_total_gb, 0)} GB`],
      ["Training processes", fmt(sys.training_rss_gb, 1) + " GB"],
      ["GPU load", fmt(g.util, 0) + " %"], ["GPU temperature", fmt(g.temp, 0) + " °C", g.temp >= 83],
      ["GPU memory", `${fmt(g.mem_used_gb, 1)} / ${fmt(g.mem_total_gb, 0)} GB`], ["GPU power", fmt(g.power_w, 0) + " W"]]);
    document.getElementById("log").textContent = (s.log || []).slice().reverse().join("\\n");
  } catch (e) {
    document.getElementById("state").textContent = "not responding (training stopped?)";
  }
}
refresh();
setInterval(refresh, 2000);
</script></body></html>
"""
