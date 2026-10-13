"""Training control for a shared PC: a local HTTP API (127.0.0.1 only) with a status page, and a system sampler.

The CLI is `python -m trainer.ctl`. Pausing leaves the Node workers idle and frees the GPU's compute."""
import collections
import json
import math
import re
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil
import torch
from tensorboard.backend.event_processing.directory_watcher import DirectoryDeletedError
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPLAYS = Path(__file__).resolve().parents[1] / "replays"  # saved games: trainer.eval keeps one folder per matchup
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
            state, self.state = self.state, "paused"  # running, or evaluating (trainer.ppo's panel)
            torch.cuda.empty_cache()
            while self.paused and not self.stop_requested:
                time.sleep(0.2)
            self.state = state
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


def serve(control, port, sampler=True, log_dir=None, replays=REPLAYS):
    """Starts the control server on 127.0.0.1:port (0 = any free port) in a daemon thread. With the run's TensorBoard
    `log_dir`, the page charts its curves. Saved games are served from `replays`."""
    # At most 1,000 points per curve, sampled the way TensorBoard samples them.
    curves = EventAccumulator(str(log_dir), size_guidance={"scalars": 1000}) if log_dir else None
    lock = threading.Lock()

    def history():
        if curves is None:
            return {}
        with lock:  # requests run in parallel threads
            try:
                curves.Reload()  # reads only what the trainer appended since the last request
            except DirectoryDeletedError:  # the trainer has not created its log yet
                return {}
            return {tag: [[e.step, e.value] for e in curves.Scalars(tag) if math.isfinite(e.value)]
                    for tag in curves.Tags()["scalars"]}

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
                path = replays / f"{game[1]}.json"
                if not path.exists():
                    return self._send(404, "{}", cors=True)
                return self._send(200, path.read_text(encoding="utf-8"), cors=True)
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/status":
                self._send(200, json.dumps(control.status()))
            elif self.path == "/history":
                self._send(200, json.dumps(history()))
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
  --line: #e3e6eb; --second: #d97706; --third: #15803d; --fourth: #7c3aed; }
@media (prefers-color-scheme: dark) { :root { --bg: #14171c; --card: #1d2128; --text: #e6e9ef; --muted: #98a1b3;
  --accent: #6b9bff; --hot: #fb923c; --line: #2c323c; --second: #f0a742; --third: #4ade80; --fourth: #a78bfa; } }
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
.chart-bar { display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: baseline; justify-content: space-between; }
.chart-bar label { font-size: 13px; color: var(--muted); display: inline-flex; gap: 6px; align-items: center; }
.charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin-top: 6px; }
.chart-head { font-size: 13px; font-weight: 600; display: flex; justify-content: space-between; gap: 8px;
  align-items: baseline; }
.goal { font-size: 11px; font-weight: 500; color: var(--muted); white-space: nowrap; }
.chart-note { margin: 6px 0 0; font-size: 12px; line-height: 1.4; color: var(--muted); }
.chart-last { font: 12px/1.5 ui-monospace, monospace; display: flex; flex-wrap: wrap; gap: 0 10px; min-height: 18px; }
.chart-box { position: relative; height: 160px; }
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
  <section><h2>Run</h2><table id="run"></table></section>
  <section><h2>Training</h2><table id="training"></table></section>
  <section><h2>Computer</h2><table id="system"></table></section>
  <section hidden><h2>League</h2><table id="league"></table></section>
  <section hidden><h2>Frozen evaluations</h2><table id="panel"></table></section>
</div>
<section style="margin-top:12px">
  <div class="chart-bar"><h2>Curves</h2>
    <label>Smoothing <input type="range" id="smoothing" min="0" max="0.99" step="0.01" value="0.8">
    <output id="smoothing-out">0.80</output></label></div>
  <p class="hint" style="margin: 0">Bold lines are smoothed, faint ones are the raw values. X axis: training updates.
  Hover a chart to read its values.</p>
  <div id="charts" class="charts"></div>
</section>
<section style="margin-top:12px"><h2>Events</h2><div id="log"></div></section>
<p class="hint">The curves refresh every 30 s. Every metric: run <code>.venv/Scripts/tensorboard --logdir runs</code>
and open <a href="http://localhost:6006">localhost:6006</a>.</p>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
  integrity="sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6djDQZjh3Y8SZ1d+6I+1zze6Z7kHXO7q3UyZAWw=="
  crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script>
const fmt = (v, d = 2) => v === null || v === undefined ? "–" : typeof v === "number" ? v.toFixed(d) : v;
function rows(id, items) {
  const table = document.getElementById(id);
  table.innerHTML = items.map(([k, v, hot]) =>
    `<tr><td>${k}</td><td class="${hot ? "hot" : ""}">${v}</td></tr>`).join("");
  table.closest("section").hidden = !items.length;
}
async function send(cmd) {
  await fetch("/" + cmd, { method: "POST", headers: { "X-Control": "1" } });
  refresh();
}
async function refresh() {
  try {
    const s = await (await fetch("/status")).json();
    if (s.update) perUpdate = s.global_step / s.update;
    rows("run", Object.entries(s.run || {}));
    rows("league", s.league || []);
    rows("panel", s.panel || []);
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
// A chart shows when the run logs its tags (prefix: every tag that starts with it, named by the rest). goal: which
// way is good; ref: [height of a dashed line, its name].
const CHARTS = [
  { title: "Frozen evaluations", prefix: "eval/vs_", goal: "higher is better", ref: [0.8, "Gate 4A bar, 3B clone"],
    note: "Every 250 updates the snapshot plays fixed games against each panel policy while training waits: its win " +
      "rate against each. Dashed: Gate 4A's bar against the 3B clone (0.70 against the Phase 2 PPO bot)." },
  { title: "Rating", lines: [["eval/rating", "rating"]], goal: "still rising",
    note: "Each snapshot's Bradley-Terry rating in Elo points, from every frozen evaluation so far, with the first " +
      "panel policy (the 3B clone) at 0. Gate 4A wants it still rising over the last 5 snapshots." },
  { title: "Win rate", lines: [["episodes/win_rate", "all"], ["episodes/win_rate_seat0", "seat 0"],
    ["episodes/win_rate_seat1", "seat 1"]], goal: "higher is better", ref: [0.545, "Gate B bar"],
    note: "Share of the last 200 games the learner won, overall and from each seat it plays. Against a copy of " +
      "itself, 0.5 is a draw. Dashed: the Gate B bar, for games against the frozen clone." },
  { title: "KL to the initial policy", lines: [["loss/kl_ref", "KL"]], goal: "small and level",
    note: "How far the policy has moved from where it started, per decision. A lower --kl-coef lets it move further: " +
      "expect a step up that levels off. If it keeps climbing while the win rate stays flat or falls, resume with a " +
      "higher --kl-coef." },
  { title: "Entropy", lines: [["loss/entropy", "entropy"]], goal: "a slow fall is normal",
    note: "How random the policy's choices are. A steady rise means it is playing more randomly." },
  { title: "Value vs outcome correlation", lines: [["value/outcome_corr_early", "early"],
    ["value/outcome_corr_mid", "mid"], ["value/outcome_corr_late", "late"]], goal: "higher is better",
    note: "How well the value head predicts the winner, by game phase (1 = perfect). It should be highest late in " +
      "the game." },
  { title: "Commands", lines: [["actions/attack", "attack"], ["actions/boat", "boat"], ["actions/build_city", "city"],
    ["actions/spawn", "spawn"]], goal: "watch for sudden swings",
    note: "Share of decisions where the learner attacks, launches a boat, builds a city or picks a spawn point. The " +
      "rest are waits." },
  { title: "Decisions / s", lines: [["perf/decisions_per_s", "decisions/s"]], goal: "higher is better",
    note: "Training speed. It drops when the PC is busy or the run is throttled." },
  { title: "Trainer memory", lines: [["perf/private_gb", "GiB"]], goal: "level",
    note: "The trainer's private memory in GiB; Windows counts its GPU memory in it. A steady climb is a leak: once " +
      "the PC's RAM is full, training slows." },
  { title: "Loss", lines: [["bc/nll", "training"], ["val/bc/nll", "held-out"], ["twin/bc/nll", "twin, training"],
    ["twin/val/bc/nll", "twin, held-out"]], goal: "lower is better",
    note: "How unlikely the clone finds the command the human actually gave. Trust the held-out line: those games " +
      "are never trained on." },
  { title: "Value vs outcome correlation", lines: [["val/value_outcome_corr", "held-out"]], goal: "higher is better",
    ref: [0.5, "Gate A bar"],
    note: "How well the value head predicts the winner of games it never trained on. Dashed: the Gate A bar." },
  { title: "Attack target accuracy", lines: [["bc/attack_target_acc", "training"],
    ["val/bc/attack_target_acc", "held-out"]], goal: "higher is better",
    note: "How often the clone's likeliest attack target is the one the human chose." },
  { title: "Windows with a command", lines: [["bc/act_rate_human", "humans"], ["bc/act_rate_policy", "clone"]],
    goal: "the lines should match", note: "How often humans give a command, and how often the clone would." },
  { title: "Rows / s", lines: [["perf/samples_per_s", "rows/s"]], goal: "higher is better",
    note: "Training speed: replay rows learned per second." },
];
const smoothing = document.getElementById("smoothing");
let curves = {}, perUpdate = 1;  // TensorBoard steps per update, from /status
function points(pts, w) {  // TensorBoard's smoothing: a debiased moving average
  let last = 0, n = 0;
  return pts.map(([x, y]) => { last = last * w + (1 - w) * y; n += 1; return {x: x / perUpdate, y: last / (1 - w ** n)}; });
}
function nearestPerLine(chart, e) {  // the point nearest the pointer on every line: lines are logged at different steps
  const x = chart.scales.x.getValueForPixel(e.x), items = [];
  chart.data.datasets.forEach((ds, i) => {
    if (ds.faint || !chart.isDatasetVisible(i) || !ds.data.length) return;
    let lo = 0, hi = ds.data.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (ds.data[mid].x < x) lo = mid + 1; else hi = mid; }
    if (lo > 0 && x - ds.data[lo - 1].x < ds.data[lo].x - x) lo -= 1;
    items.push({ element: chart.getDatasetMeta(i).data[lo], datasetIndex: i, index: lo });
  });
  return items;
}
function drawCharts() {
  const box = document.getElementById("charts");
  if (!window.Chart) { box.textContent = "The charts need Chart.js from cdnjs.cloudflare.com, which did not load."; return; }
  Chart.Interaction.modes.nearestPerLine = nearestPerLine;
  const css = getComputedStyle(document.documentElement), color = name => css.getPropertyValue(name).trim();
  const colors = ["--accent", "--second", "--third", "--fourth"].map(color), muted = color("--muted");
  const grid = color("--line"), w = Number(smoothing.value);
  for (const spec of CHARTS) {
    const lines = spec.prefix ? Object.keys(curves).filter(t => t.startsWith(spec.prefix)).sort()
      .map(t => [t, t.slice(spec.prefix.length)]) : spec.lines;
    const shown = lines.filter(([tag]) => (curves[tag] || []).length), ref = spec.ref;
    if (!shown.length) continue;
    const datasets = shown.flatMap(([tag, label], i) => [
      { label, faint: true, data: points(curves[tag], 0), borderColor: colors[i] + "40", borderWidth: 1 },
      { label, data: points(curves[tag], w), borderColor: colors[i], borderWidth: 1.75 }]);
    if (ref) {
      const xs = datasets.flatMap(d => [d.data[0].x, d.data[d.data.length - 1].x]);
      datasets.push({ label: ref[1], faint: true, borderColor: muted, borderDash: [4, 4], borderWidth: 1,
        data: [{ x: Math.min(...xs), y: ref[0] }, { x: Math.max(...xs), y: ref[0] }] });
    }
    let chart = spec.chart;
    if (chart) {
      chart.data.datasets = datasets;
      chart.update("none");
    } else {
      const card = document.createElement("div");
      card.innerHTML = '<div class="chart-head"><span></span><span class="goal"></span></div>' +
        '<div class="chart-last"></div><div class="chart-box"><canvas></canvas></div><p class="chart-note"></p>';
      card.querySelector(".chart-head span").textContent = spec.title;
      card.querySelector(".goal").textContent = spec.goal;
      card.querySelector(".chart-note").textContent = spec.note;
      box.append(card);
      const font = { size: 10 };
      chart = spec.chart = new Chart(card.querySelector("canvas"), { type: "line", data: { datasets }, options: {
        animation: false, parsing: false, normalized: true, responsive: true, maintainAspectRatio: false,
        interaction: { mode: "nearestPerLine", intersect: false }, elements: { point: { radius: 0, hoverRadius: 3 } },
        plugins: { legend: { display: false },
          tooltip: { callbacks: { title: items => items.length ? `update ${Math.round(items[0].parsed.x)}` : "" } } },
        scales: { x: { type: "linear", ticks: { color: muted, font, maxTicksLimit: 6 }, grid: { color: grid } },
                  y: { ticks: { color: muted, font, maxTicksLimit: 5 }, grid: { color: grid } } } } });
      chart.last = card.querySelector(".chart-last");
    }
    chart.last.innerHTML = datasets.filter(d => !d.faint).map(d => {
      const y = d.data[d.data.length - 1].y;
      return `<span style="color:${d.borderColor}">${d.label} ${fmt(y, Math.abs(y) >= 100 ? 0 : 3)}</span>`;
    }).join("");
  }
}
async function loadCurves() {
  try {
    curves = await (await fetch("/history")).json();
    drawCharts();
  } catch (e) {}
}
smoothing.addEventListener("input", () => {
  document.getElementById("smoothing-out").textContent = Number(smoothing.value).toFixed(2);
  drawCharts();
});
refresh().then(loadCurves);
setInterval(refresh, 2000);
setInterval(loadCurves, 30000);
</script></body></html>
"""
