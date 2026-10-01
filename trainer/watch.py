"""Serves saved games (replays/<id>.json) to the game's own client, to watch them in the real game view.

  .venv/Scripts/python -m trainer.watch                    (a running trainer's control server serves them too)
  .venv/Scripts/python -m trainer.watch --fetch <gameID>   (first download a public game, e.g. to compare with humans)
  then in OpenFront_Source_Code/openfrontio: npm run dev, and open http://localhost:9000/game/<id>

On localhost the game's dev client loads archived games from http://localhost:8787/game/<id> and skips the version
check (src/client/ApiBase.ts, JoinLobbyModal.ts), so this serves on port 8787.
"""
import argparse
import json
import time

from trainer import crawl
from trainer.control import REPLAYS, Control, serve


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fetch", help="download this public game into replays/ first")
    p.add_argument("--port", type=int, default=8787)
    a = p.parse_args(argv)
    REPLAYS.mkdir(exist_ok=True)
    if a.fetch:
        record = crawl.Api().get(f"/game/{a.fetch}")
        if record is None:
            raise SystemExit(f"game {a.fetch} not found")
        (REPLAYS / f"{a.fetch}.json").write_text(json.dumps(crawl.blank(record)), encoding="utf-8")
    try:
        server = serve(Control(), a.port, sampler=False)
    except OSError:
        raise SystemExit(f"port {a.port} is taken: a running trainer's control server already serves replays/")
    print(f"serving {REPLAYS} on http://127.0.0.1:{a.port}; start the game's dev client (npm run dev), then open:")
    for f in sorted(REPLAYS.glob("*.json"), key=lambda f: f.stat().st_mtime)[-10:]:
        print(f"  http://localhost:9000/game/{f.stem}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
