"""Watch saved games (replays/<id>.json) in the game's own client: starts the game's dev client and serves the games.

  .venv/Scripts/python -m trainer.watch --dir "bc 60000 vs random"   then open http://localhost:9000/game/<id>; Ctrl+C stops both
  .venv/Scripts/python -m trainer.watch                    (the games saved directly in replays/, e.g. by --fetch)
  .venv/Scripts/python -m trainer.watch --fetch <gameID>   (first download a public game, e.g. to compare with humans)

On localhost the game's dev client (npm run dev in the game folder, http://localhost:9000) loads archived games from
http://localhost:8787/game/<id> and skips the version check (src/client/ApiBase.ts, JoinLobbyModal.ts), so the games
are served on port 8787. Without that server the client cannot find the game and falls back to its lobby page. A
running trainer's control server also serves them on 8787; then only the client is started.
"""
import argparse
import json
import shutil
import subprocess
import time
import urllib.request

import psutil

from trainer import crawl
from trainer.control import REPLAYS, Control, serve
from trainer.env import GAME_DIR, ROOT

CLIENT = "http://localhost:9000"


def answers(url):
    try:
        urllib.request.urlopen(url, timeout=2).close()
        return True
    except OSError:
        return False


def start_client(log_path, cmd=None, url=CLIENT, timeout=120):
    """Starts the game's dev client (npm run dev in the game folder) and returns its process once `url` answers, or None
    if a client already answers there (started by hand, or left over)."""
    if answers(url):
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        # Its own process group: Ctrl+C in this console would otherwise make npm's cmd.exe ask "Terminate batch job?";
        # stop() ends it instead.
        proc = subprocess.Popen(cmd or [shutil.which("npm"), "run", "dev"], cwd=GAME_DIR, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    deadline = time.monotonic() + timeout
    while not answers(url):
        if proc.poll() is not None or time.monotonic() > deadline:
            stop(proc)
            raise SystemExit(f"the game client did not start; see {log_path}")
        time.sleep(1)
    return proc


def stop(proc):
    """Stops a process and everything it started: npm runs the client and the game server as child processes, which
    otherwise outlive it."""
    try:
        procs = [psutil.Process(proc.pid)]
        procs += procs[0].children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=10)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fetch", help="download this public game into replays/ first")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--dir", help='serve this folder of replays/ (one per matchup, e.g. "bc 60000 vs random")')
    a = p.parse_args(argv)
    REPLAYS.mkdir(exist_ok=True)
    if a.fetch:
        record = crawl.Api().get(f"/game/{a.fetch}")
        if record is None:
            raise SystemExit(f"game {a.fetch} not found")
        (REPLAYS / f"{a.fetch}.json").write_text(json.dumps(crawl.blank(record)), encoding="utf-8")
    folder = REPLAYS / a.dir if a.dir else REPLAYS
    if not folder.is_dir():
        saved = sorted(d.name for d in REPLAYS.iterdir() if d.is_dir())
        raise SystemExit(f"no folder {folder}. Matchups saved so far: {', '.join(saved) or 'none'}")
    try:
        server = serve(Control(), a.port, sampler=False, replays=folder)
        print(f"serving {folder} on http://127.0.0.1:{a.port}")
    except OSError:
        server = None
        print(f"port {a.port} is taken: a running trainer's control server serves {REPLAYS} (not --dir); stop it to "
              f"use --dir")
    log = ROOT / "runs" / "watch-client.log"
    print(f"starting the game client (npm run dev, log: {log}); the first start takes up to a minute ...")
    client = None
    try:
        client = start_client(log)
        print(f"the game client is ready on {CLIENT}. Open a game (newest last), and press Ctrl+C here to stop:")
        for f in sorted(folder.glob("*.json"), key=lambda f: f.stat().st_mtime)[-10:]:
            print(f"  {CLIENT}/game/{f.stem}")
        while True:
            time.sleep(1)  # short naps: a stop request (Ctrl+C) is seen within a second however it arrives
    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            stop(client)
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
