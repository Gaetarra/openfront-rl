"""Downloads ranked 1v1 game records from the public OpenFront API for behaviour cloning (PLAN Phase 2).

  .venv/Scripts/python -m trainer.crawl                 every ranked 1v1 game since v0.34.17 went live (hours: detach)
  .venv/Scripts/python -m trainer.crawl --limit 50      a small sample
  .venv/Scripts/python -m trainer.crawl --purge         only remove deleted players' records (run daily while data is kept)

The owner cleared using the API for training (2026-09-30). The crawler stays polite: at most one request per second, a
User-Agent that names the project, every record cached and never fetched twice. Usernames are blanked in the cache and
the records of deleted players are removed (docs/API.md "Recently Deleted Players").
"""
import argparse
import datetime as dt
import gzip
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from trainer.env import ROOT

API = "https://api.openfront.io/public"
USER_AGENT = "openfront-rl/0.2 (+https://github.com/Gaetarra/openfront-rl; behaviour-cloning data, 1 req/s)"
DATA = ROOT / "data"
RECORDS = DATA / "records"
INDEX = DATA / "index.jsonl"
SKIPPED = DATA / "skipped.txt"  # IDs of void records, so they are not fetched again
PURGED_AT = DATA / "purged_at.txt"  # time of the last deleted-players sync
WINDOW = dt.timedelta(days=2)  # the games listing takes at most 2-day windows
# v0.34.17 reached the servers between 14:23 and 18:00 UTC that day; v0.34.17-v0.34.21 share one simulation.
SINCE = "2026-09-23T18:00:00Z"


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


class Api:
    """GET requests, at most one per `interval` seconds, retried with backoff on HTTP 429, 5xx, network errors and
    broken bodies."""

    def __init__(self, interval=1.0, opener=urllib.request.urlopen, sleep=time.sleep, clock=time.monotonic):
        self.interval, self.opener, self.sleep, self.clock = interval, opener, sleep, clock
        self.last = None

    def get(self, path, **params):
        """The decoded JSON, or None for 404."""
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{API}{path}" + (f"?{query}" if query else "")
        for attempt in range(6):
            if self.last is not None and self.clock() - self.last < self.interval:
                self.sleep(self.interval - (self.clock() - self.last))
            self.last = self.clock()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                with self.opener(req, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code != 429 and e.code < 500:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, ValueError):
                pass  # network errors, and bodies cut off mid-read (IncompleteRead) or garbled (JSONDecodeError)
            self.sleep(min(300, 5 * 2 ** attempt))
        raise RuntimeError(f"giving up on {url}")


def list_games(api, start, end, ranked="1v1"):
    """Metadata of the public ranked games that started between start and end, oldest window first."""
    t = start
    while t < end:
        stop = min(t + WINDOW, end)
        offset = 0
        while True:
            page = api.get("/games", start=iso(t), end=iso(stop), type="Public", rankedType=ranked, limit=1000,
                           offset=offset) or []
            yield from page
            if len(page) < 1000:
                break
            offset += len(page)
        t = stop


def is_void(record):
    """A record behaviour cloning cannot use: not exactly 2 players, no winner, or a player who never spawned."""
    info = record["info"]
    if len(info["players"]) != 2 or not info.get("winner"):
        return True
    spawned = {i["clientID"] for t in record["turns"] for i in t["intents"] if i["type"] == "spawn"}
    return any(p["clientID"] not in spawned for p in info["players"])


def blank(record, public_ids=False):
    """Replaces the usernames (not simulation inputs: the fixture still replays in sync, bridge/replay.test.ts) and,
    for data committed to the repo, the account IDs."""
    for i, p in enumerate(record["info"]["players"]):
        p["username"] = f"Player{i + 1}"
        if public_ids:
            p["publicID"] = None
    return record


def entry_of(record):
    info = record["info"]
    winner = info["winner"]
    return {"id": info["gameID"], "start": info["start"], "commit": record.get("gitCommit"),
            "map": info["config"]["gameMap"], "size": info["config"]["gameMapSize"], "turns": info["num_turns"],
            "players": [{"clientID": p["clientID"], "publicID": p.get("publicID")} for p in info["players"]],
            "winner": winner[1] if winner and winner[0] == "player" else None}


def record_path(game_id):
    return RECORDS / f"{game_id}.json.gz"


def save(record):
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = record_path(record["info"]["gameID"])
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(record, f, separators=(",", ":"))
    tmp.replace(path)


def load_index():
    if not INDEX.exists():
        return []
    with open(INDEX, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_index(entry):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def purge(api, now=None):
    """Removes the records and index lines of players deleted since the last sync; returns how many records went. The
    API lists deletions for 7 days, so after a longer gap every indexed player is looked up instead (slow, rare)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    last = parse_time(PURGED_AT.read_text().strip()) if PURGED_AT.exists() else None
    entries = load_index()
    if last is not None and now - last > dt.timedelta(days=6, hours=23):
        known = {p["publicID"] for e in entries for p in e["players"] if p["publicID"]}
        deleted = {pid for pid in known if api.get(f"/player/{pid}") is None}
    else:
        rows = api.get("/players/recently-deleted", since=iso(last) if last else None) or []
        deleted = {r["publicId"] for r in rows}
    keep = [e for e in entries if not any(p["publicID"] in deleted for p in e["players"])]
    if len(keep) < len(entries):
        for e in entries:
            if e not in keep:
                record_path(e["id"]).unlink(missing_ok=True)
        tmp = INDEX.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for e in keep), encoding="utf-8")
        tmp.replace(INDEX)
    DATA.mkdir(parents=True, exist_ok=True)
    PURGED_AT.write_text(iso(now))
    return len(entries) - len(keep)


def elo(entries, k=32.0, base=1500.0):
    """Each player's Elo before each game, from the games in start order: {game ID: {clientID: rating}}."""
    rating, out = {}, {}
    for e in sorted(entries, key=lambda e: e["start"]):
        a, b = e["players"]
        ra, rb = rating.get(a["publicID"], base), rating.get(b["publicID"], base)
        out[e["id"]] = {a["clientID"]: ra, b["clientID"]: rb}
        if e["winner"] is None or not a["publicID"] or not b["publicID"]:
            continue
        delta = k * ((1.0 if e["winner"] == a["clientID"] else 0.0) - 1.0 / (1.0 + 10 ** ((rb - ra) / 400)))
        rating[a["publicID"]], rating[b["publicID"]] = ra + delta, rb - delta
    return out


def crawl(api, since, until, limit=None, log=print):
    """Saves the usable ranked 1v1 records that started between since and until; returns (saved, void)."""
    purge(api)
    done = {e["id"] for e in load_index()}
    if SKIPPED.exists():
        done |= set(SKIPPED.read_text().split())
    saved = void = 0
    for meta in list_games(api, since, until):
        gid = meta["game"]
        if gid in done:
            continue
        done.add(gid)
        if record_path(gid).exists():  # saved, but the index line was lost (killed in between)
            with gzip.open(record_path(gid), "rt", encoding="utf-8") as f:
                record = json.load(f)
        else:
            record = api.get(f"/game/{gid}")
            if record is None:
                continue
        if is_void(record):
            DATA.mkdir(parents=True, exist_ok=True)
            with open(SKIPPED, "a") as f:
                f.write(gid + "\n")
            void += 1
            continue
        save(blank(record))
        append_index(entry_of(record))
        saved += 1
        if saved % 100 == 0:
            log(f"{saved} saved, {void} void, up to {meta.get('start')}", flush=True)
        if limit is not None and saved >= limit:
            break
    return saved, void


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default=SINCE, help="first game start, ISO 8601")
    p.add_argument("--until", help="last game start (default: now)")
    p.add_argument("--limit", type=int, help="stop after saving this many records")
    p.add_argument("--purge", action="store_true", help="only remove deleted players' records")
    a = p.parse_args(argv)
    api = Api()
    if a.purge:
        print(f"removed {purge(api)} records")
        return
    until = parse_time(a.until) if a.until else dt.datetime.now(dt.timezone.utc)
    saved, void = crawl(api, parse_time(a.since), until, a.limit)
    print(f"done: {saved} saved, {void} void; {len(load_index())} records in {INDEX}")


if __name__ == "__main__":
    main()
