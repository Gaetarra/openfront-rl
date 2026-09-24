"""Control a running trainer: python -m trainer.ctl status|pause|resume|save|stop|throttle <full|half|low|0.3>"""
import argparse
import json
import urllib.error
import urllib.request


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["status", "pause", "resume", "save", "stop", "throttle"])
    p.add_argument("value", nargs="?", help="throttle level: full, half, low or a fraction in (0, 1]")
    p.add_argument("--port", type=int, default=8787)
    a = p.parse_args(argv)
    if a.command == "throttle" and a.value is None:
        p.error("throttle needs a level")
    base = f"http://127.0.0.1:{a.port}"
    if a.command == "status":
        req = urllib.request.Request(base + "/status")
    else:
        path = f"/{a.command}" + (f"/{a.value}" if a.value else "")
        req = urllib.request.Request(base + path, data=b"", method="POST", headers={"X-Control": "1"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            print(json.dumps(json.load(r), indent=2))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"refused: {e.read().decode()}")
    except urllib.error.URLError as e:
        raise SystemExit(f"no trainer is answering on port {a.port} ({e.reason})")


if __name__ == "__main__":
    main()
