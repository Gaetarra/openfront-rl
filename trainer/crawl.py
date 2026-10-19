"""Downloads game records from the public OpenFront API for behaviour cloning and the mode analyses (PLAN Phase 2,
Phase 4 "After 4N").

  .venv/Scripts/python -m trainer.crawl                 every ranked 1v1 game of the pin's simulation (hours: detach)
  .venv/Scripts/python -m trainer.crawl --modes ranked1v1 rankedffa ffa=0.25
                                                        several modes; MODE=SHARE keeps that share of a mode's games
  .venv/Scripts/python -m trainer.crawl --limit 50      a small sample of each mode
  .venv/Scripts/python -m trainer.crawl --purge         only remove deleted players' records (run daily while data is kept)
  .venv/Scripts/python -m trainer.crawl --reindex       refill the index fields of the entries already there (crawl stopped)

Records desync at a pin with another simulation (PLAN decision 14), so each simulation's records have their own store:
data/ holds v0.34.17-v0.34.22's (the old pin's, pins/v0.34.23), and data/<the simulation's first tag>/ each later
one's. The crawler fills the store of the pin's simulation and skips records of other game commits.

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
import zlib

from trainer.env import ROOT

API = "https://api.openfront.io/public"
USER_AGENT = "openfront-rl/0.2 (+https://github.com/Gaetarra/openfront-rl; behaviour-cloning data, 1 req/s)"
DATA = ROOT / "data"  # the old pin's store; the later simulations' stores are its subfolders
PURGED_AT = DATA / "purged_at.txt"  # time of the last deleted-players sync, for every store
WINDOW = dt.timedelta(days=2)  # the games listing takes at most 2-day windows
# The pin's simulation. v0.35.0-v0.35.2 share it: their engine packages differ only in a map's tribe themes and an API
# schema, and a v0.35.0 record replays in sync at v0.35.2. v0.35.0 replaced v0.34.25 on the servers on 2026-10-16
# between 17:46 and 17:49 UTC.
SIM = "v0.35.0"  # its first tag, which names its store
SIM_COMMITS = {
    "dfd420fd7836ae023504a9440d9d68897745b7b4",  # v0.35.0
    "538d205f1279d5037e28fb933e631c612550f5ff",  # v0.35.1
    "08b63c054897c030e6516e0731ab7cc62b2998b7",  # v0.35.2
}
SINCE = "2026-10-16T17:45:00Z"
# The games listing's filters of each mode (docs/API.md). The names follow the trainer's setups: "ffa" and "team" are
# the public games, and Humans vs Nations is among the Team ones.
MODES = {
    "ranked1v1": {"rankedType": "1v1"},
    "ranked2v2": {"rankedType": "2v2"},
    "rankedffa": {"rankedType": "ffa"},
    "clanwars": {"rankedType": "clanwars"},
    "ffa": {"rankedType": "unranked", "mode": "Free For All"},
    "team": {"rankedType": "unranked", "mode": "Team"},
}


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


def list_games(api, start, end, filters):
    """Metadata of the public games that started between start and end and match the listing filters (a MODES value),
    oldest window first."""
    t = start
    while t < end:
        stop = min(t + WINDOW, end)
        offset = 0
        while True:
            page = api.get("/games", start=iso(t), end=iso(stop), type="Public", limit=1000, offset=offset,
                           **filters) or []
            yield from page
            if len(page) < 1000:
                break
            offset += len(page)
        t = stop


def is_void(record, mode="ranked1v1"):
    """A record the trainer cannot use: no winner, and in ranked 1v1 also not exactly 2 players or a player who never
    spawned."""
    info = record["info"]
    if not info.get("winner"):
        return True
    if mode != "ranked1v1":
        return False
    if len(info["players"]) != 2:
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


# What behaviour cloning filters records on, from the lobby's config (step 7): `teams` is playerTeams (a team count or a
# name such as "Duos" or "Humans Vs Nations"), `modifiers` the sorted names of the public game modifiers that are on,
# `bots` and `nations` "default", "disabled" or a number.
INDEX_FIELDS = ("teams", "modifiers", "disabled", "bots", "nations")


def entry_of(record, mode="ranked1v1"):
    info = record["info"]
    winner = info["winner"]
    config = info["config"]
    return {"id": info["gameID"], "start": info["start"], "commit": record.get("gitCommit"),
            "map": info["config"]["gameMap"], "size": info["config"]["gameMapSize"], "turns": info["num_turns"],
            "players": [{"clientID": p["clientID"], "publicID": p.get("publicID")} for p in info["players"]],
            "winner": winner[1] if winner and winner[0] == "player" else None, "mode": mode,
            "teams": config.get("playerTeams"),
            "modifiers": sorted(k for k, v in (config.get("publicGameModifiers") or {}).items() if v),
            "disabled": config.get("disabledUnits") or [], "bots": config.get("bots"),
            "nations": config.get("nations")}


def store(sim=SIM):
    """The store of a simulation's records: data/<its first tag>/, laid out like data/."""
    return DATA / sim


def stores():
    """Every store: the old pin's (data/), then each later simulation's."""
    later = sorted(d for d in DATA.iterdir() if (d / "index.jsonl").is_file()) if DATA.is_dir() else []
    return [DATA, *later]


def record_path(game_id, root=None):
    """A record's file in the store `root` (default: the old pin's, data/)."""
    return (root or DATA) / "records" / f"{game_id}.json.gz"


def save(record, root=None):
    path = record_path(record["info"]["gameID"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(record, f, separators=(",", ":"))
    tmp.replace(path)


def load_index(root=None):
    """A store's index (default: the old pin's): one entry per saved record, in crawl order."""
    index = (root or DATA) / "index.jsonl"
    if not index.exists():
        return []
    with open(index, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_record(game_id, root=None):
    with gzip.open(record_path(game_id, root), "rt", encoding="utf-8") as f:
        return json.load(f)


def append_index(entry, root=None):
    root = root or DATA
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def purge(api, now=None):
    """Removes the records and index lines of players deleted since the last sync, from every store; returns how many
    records went. The API lists deletions for 7 days, so after a longer gap every indexed player is looked up instead
    (slow, rare)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    last = parse_time(PURGED_AT.read_text().strip()) if PURGED_AT.exists() else None
    indexes = {root: load_index(root) for root in stores()}
    if last is not None and now - last > dt.timedelta(days=6, hours=23):
        known = {p["publicID"] for entries in indexes.values() for e in entries for p in e["players"] if p["publicID"]}
        deleted = {pid for pid in known if api.get(f"/player/{pid}") is None}
    else:
        rows = api.get("/players/recently-deleted", since=iso(last) if last else None) or []
        deleted = {r["publicId"] for r in rows}
    removed = 0
    for root, entries in indexes.items():
        gone = {e["id"] for e in entries if any(p["publicID"] in deleted for p in e["players"])}
        if not gone:
            continue
        for gid in gone:
            record_path(gid, root).unlink(missing_ok=True)
        index = root / "index.jsonl"
        tmp = index.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for e in entries if e["id"] not in gone), encoding="utf-8")
        tmp.replace(index)
        removed += len(gone)
    DATA.mkdir(parents=True, exist_ok=True)
    PURGED_AT.write_text(iso(now))
    return removed


def reindex(root=None):
    """Refills the INDEX_FIELDS of every entry of a store's index (default: the old pin's) from its record, for the
    entries written before the fields existed; entries without a record file stay as they are. Returns how many
    entries changed. A crawl appends to the index while it runs, so the file is replaced only if its bytes are what
    this read: otherwise nothing is replaced and the rerun waits for the crawl."""
    index = (root or DATA) / "index.jsonl"
    before = index.read_bytes()
    entries = [json.loads(line) for line in before.decode("utf-8").splitlines() if line.strip()]
    changed = 0
    for e in entries:
        if not record_path(e["id"], root).exists():
            continue
        fresh = entry_of(load_record(e["id"], root), e["mode"])
        update = {k: fresh[k] for k in INDEX_FIELDS}
        if any(e.get(k) != v for k, v in update.items()) or any(k not in e for k in update):
            e.update(update)
            changed += 1
    if index.read_bytes() != before:
        raise SystemExit(f"{index} changed while reindexing: a crawl is writing it. Rerun when the crawl is done")
    tmp = index.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    tmp.replace(index)
    return changed


def sampled(game_id, share):
    """Whether a mode's share keeps a game: a stable hash of its ID, salted apart from BC's held-out split."""
    return share >= 1 or zlib.crc32(f"crawl:{game_id}".encode()) % 10_000 < share * 10_000


def elo(entries, k=32.0, base=1500.0):
    """Each player's Elo before each game, from the ranked 1v1 games in start order: {game ID: {clientID: rating}}."""
    rating, out = {}, {}
    for e in sorted((e for e in entries if e.get("mode", "ranked1v1") == "ranked1v1"), key=lambda e: e["start"]):
        a, b = e["players"]
        ra, rb = rating.get(a["publicID"], base), rating.get(b["publicID"], base)
        out[e["id"]] = {a["clientID"]: ra, b["clientID"]: rb}
        if e["winner"] is None or not a["publicID"] or not b["publicID"]:
            continue
        delta = k * ((1.0 if e["winner"] == a["clientID"] else 0.0) - 1.0 / (1.0 + 10 ** ((rb - ra) / 400)))
        rating[a["publicID"]], rating[b["publicID"]] = ra + delta, rb - delta
    return out


def crawl(api, since, until, modes=(("ranked1v1", 1.0),), root=None, limit=None, log=print):
    """Saves the usable records of each (mode, share) that started between since and until into the store `root` (the
    pin's simulation's by default), at most `limit` per mode. Records of other game commits are skipped and remembered,
    as void ones are. Returns (saved, void, other versions)."""
    root = root or store()
    purge(api)
    done = {e["id"] for e in load_index(root)}
    skipped = root / "skipped.txt"  # IDs of void and other-version records, so they are not fetched again
    if skipped.exists():
        done |= set(skipped.read_text().split())
    saved = void = other = 0
    for mode, share in modes:
        kept = 0
        for meta in list_games(api, since, until, MODES[mode]):
            gid = meta["game"]
            if gid in done or not sampled(gid, share):
                continue
            done.add(gid)
            if record_path(gid, root).exists():  # saved, but the index line was lost (killed in between)
                record = load_record(gid, root)
            else:
                record = api.get(f"/game/{gid}")
                if record is None:
                    continue
            ours = record.get("gitCommit") in SIM_COMMITS
            if not ours or is_void(record, mode):
                root.mkdir(parents=True, exist_ok=True)
                with open(skipped, "a") as f:
                    f.write(gid + "\n")
                void += ours
                other += not ours
                continue
            save(blank(record), root)
            append_index(entry_of(record, mode), root)
            saved += 1
            kept += 1
            if saved % 100 == 0:
                log(f"{saved} saved, {void} void, {other} of other versions; {mode} up to {meta.get('start')}",
                    flush=True)
            if limit is not None and kept >= limit:
                break
    return saved, void, other


def mode_share(text):
    """A --modes value: MODE or MODE=SHARE."""
    mode, _, share = text.partition("=")
    if mode not in MODES:
        raise argparse.ArgumentTypeError(f"unknown mode {mode!r}, not one of {', '.join(MODES)}")
    try:
        value = float(share) if share else 1.0
    except ValueError:
        value = 0.0
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"{text}: a share is a number in (0, 1]")
    return mode, value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", type=mode_share, default=[("ranked1v1", 1.0)], metavar="MODE[=SHARE]",
                   help=f"what to crawl, from {', '.join(MODES)}; SHARE keeps that share of the mode's games (default 1)")
    p.add_argument("--since", default=SINCE, help="first game start, ISO 8601 (default: the pin's simulation went live)")
    p.add_argument("--until", help="last game start (default: now)")
    p.add_argument("--limit", type=int, help="stop each mode after saving this many records")
    p.add_argument("--purge", action="store_true", help="only remove deleted players' records")
    p.add_argument("--reindex", action="store_true",
                   help="only refill the index fields of the entries already there (not while a crawl runs)")
    a = p.parse_args(argv)
    if a.reindex:
        print(f"reindexed {reindex(store())} entries in {store()}")
        return
    api = Api()
    if a.purge:
        print(f"removed {purge(api)} records")
        return
    until = parse_time(a.until) if a.until else dt.datetime.now(dt.timezone.utc)
    saved, void, other = crawl(api, parse_time(a.since), until, a.modes, limit=a.limit)
    print(f"done: {saved} saved, {void} void, {other} of other versions; {len(load_index(store()))} records in "
          f"{store()}")


if __name__ == "__main__":
    main()
