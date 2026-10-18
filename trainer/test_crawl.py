import datetime as dt
import gzip
import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from trainer import crawl


def record(game_id, a="A1", b="B1", winner="ca", spawned=("ca", "cb"), start=0, commit="x" * 40):
    turns = [{"turnNumber": 5, "intents": [{"type": "spawn", "tile": 1, "clientID": c} for c in spawned]}]
    return {"gitCommit": commit, "turns": turns,
            "info": {"gameID": game_id, "start": start, "num_turns": 100,
                     "winner": ["player", winner] if winner else None,
                     "config": {"gameMap": "Iceland", "gameMapSize": "Normal"},
                     "players": [{"clientID": "ca", "username": "alice", "publicID": a},
                                 {"clientID": "cb", "username": "bob", "publicID": b}]}}


class FakeApi:
    def __init__(self, games=(), records=None, deleted=()):
        self.games, self.records, self.deleted, self.calls = list(games), records or {}, list(deleted), []

    def get(self, path, **params):
        self.calls.append((path, params))
        if path == "/games":
            return self.games if params.get("offset", 0) == 0 else []
        if path == "/players/recently-deleted":
            return [{"publicId": p, "deletedAt": "2026-10-01T00:00:00Z"} for p in self.deleted]
        if path.startswith("/game/"):
            return json.loads(json.dumps(self.records.get(path[len("/game/"):])))  # a fresh copy, like a response
        raise AssertionError(path)


class CrawlTest(unittest.TestCase):
    def setUp(self):
        data = Path(tempfile.mkdtemp()) / "data"
        self.patch = mock.patch.multiple(crawl, DATA=data, PURGED_AT=data / "purged_at.txt", SIM_COMMITS={"x" * 40})
        self.patch.start()
        self.t0 = dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)

    def tearDown(self):
        self.patch.stop()

    def test_void_records(self):
        self.assertFalse(crawl.is_void(record("g1")))
        self.assertTrue(crawl.is_void(record("g1", winner=None)))
        self.assertTrue(crawl.is_void(record("g1", spawned=("ca",))))
        self.assertFalse(crawl.is_void(record("g1", spawned=("ca",)), "ffa"), "only ranked 1v1 needs every player")
        self.assertTrue(crawl.is_void(record("g1", winner=None), "team"))

    def test_blank_keeps_simulation_inputs(self):
        r = crawl.blank(record("g1"))
        self.assertEqual([p["username"] for p in r["info"]["players"]], ["Player1", "Player2"])
        self.assertEqual(r["info"]["players"][0]["publicID"], "A1")
        self.assertIsNone(crawl.blank(record("g1"), public_ids=True)["info"]["players"][0]["publicID"])

    def test_crawl_saves_usable_records_once_and_remembers_void_ones(self):
        api = FakeApi(games=[{"game": "good", "start": "t"}, {"game": "void", "start": "t"}],
                      records={"good": record("good"), "void": record("void", winner=None)})
        hour = dt.timedelta(hours=1)
        self.assertEqual(crawl.crawl(api, self.t0, self.t0 + hour), (1, 1, 0))
        with gzip.open(crawl.record_path("good", crawl.store()), "rt", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["info"]["players"][0]["username"], "Player1")
        self.assertEqual([e["id"] for e in crawl.load_index(crawl.store())], ["good"])
        self.assertEqual(crawl.load_index(crawl.store())[0]["winner"], "ca")
        self.assertEqual(crawl.load_index(), [], "the old pin's store stays as it was")
        fetches = sum(1 for p, _ in api.calls if p.startswith("/game/"))
        self.assertEqual(crawl.crawl(api, self.t0, self.t0 + hour), (0, 0, 0))
        self.assertEqual(sum(1 for p, _ in api.calls if p.startswith("/game/")), fetches, "nothing fetched twice")

    def test_records_of_other_versions_are_skipped_once(self):
        api = FakeApi(games=[{"game": "old", "start": "t"}], records={"old": record("old", commit="y" * 40)})
        hour = dt.timedelta(hours=1)
        self.assertEqual(crawl.crawl(api, self.t0, self.t0 + hour), (0, 0, 1))
        self.assertEqual(crawl.crawl(api, self.t0, self.t0 + hour), (0, 0, 0))
        self.assertEqual(sum(1 for p, _ in api.calls if p.startswith("/game/")), 1)

    def test_modes_filter_the_listing_and_label_their_entries(self):
        api = FakeApi(games=[{"game": "f1", "start": "t"}], records={"f1": record("f1", spawned=("ca",))})
        crawl.crawl(api, self.t0, self.t0 + dt.timedelta(hours=1), modes=[("ffa", 1.0)])
        listing = [params for path, params in api.calls if path == "/games"]
        self.assertEqual((listing[0]["rankedType"], listing[0]["mode"], listing[0]["type"]),
                         ("unranked", "Free For All", "Public"))
        self.assertEqual([(e["id"], e["mode"]) for e in crawl.load_index(crawl.store())], [("f1", "ffa")])

    def test_a_share_keeps_a_stable_part_of_the_games(self):
        ids = [f"g{i:04d}" for i in range(4000)]
        kept = [i for i in ids if crawl.sampled(i, 0.25)]
        self.assertAlmostEqual(len(kept) / len(ids), 0.25, delta=0.03)
        self.assertEqual(kept, [i for i in ids if crawl.sampled(i, 0.25)])
        self.assertTrue(all(crawl.sampled(i, 1.0) for i in ids[:10]))
        self.assertEqual(crawl.mode_share("team=0.5"), ("team", 0.5))
        for bad in ("chess", "ffa=0", "ffa=1.5", "ffa=x"):
            with self.assertRaises(Exception, msg=bad):
                crawl.mode_share(bad)

    def test_purge_removes_deleted_players_records(self):
        api = FakeApi(games=[{"game": "g1", "start": "t"}, {"game": "g2", "start": "t"}],
                      records={"g1": record("g1", b="GONE"), "g2": record("g2")})
        crawl.crawl(api, self.t0, self.t0 + dt.timedelta(hours=1))
        old = record("o1", a="GONE")  # a record in the old pin's store
        crawl.save(old)
        crawl.append_index(crawl.entry_of(old))
        self.assertEqual(crawl.purge(FakeApi(deleted=["GONE"])), 2, "every store is purged")
        self.assertEqual([e["id"] for e in crawl.load_index(crawl.store())], ["g2"])
        self.assertFalse(crawl.record_path("g1", crawl.store()).exists())
        self.assertEqual(crawl.load_index(), [])
        self.assertFalse(crawl.record_path("o1").exists())
        self.assertTrue(crawl.PURGED_AT.exists())

    def test_elo_is_the_rating_before_each_game(self):
        entries = [crawl.entry_of(record(f"g{i}", start=i)) for i in range(3)]  # alice (ca) wins all three
        r = crawl.elo(entries)
        self.assertEqual(r["g0"], {"ca": 1500.0, "cb": 1500.0})
        self.assertGreater(r["g1"]["ca"], 1500.0)
        self.assertAlmostEqual(r["g1"]["ca"] + r["g1"]["cb"], 3000.0)
        self.assertGreater(r["g2"]["ca"], r["g1"]["ca"])

    def test_api_spaces_requests_and_retries_rate_limits(self):
        clock, slept, calls = [0.0], [], []

        def sleep(s):
            slept.append(s)
            clock[0] += s

        def opener(req, timeout):
            calls.append(req)
            if len(calls) == 2:
                raise urllib.error.HTTPError(req.full_url, 429, "slow down", {}, None)
            return io.BytesIO(b'{"ok": true}')

        api = crawl.Api(interval=1.0, opener=opener, sleep=sleep, clock=lambda: clock[0])
        self.assertEqual(api.get("/games", limit=1), {"ok": True})
        self.assertEqual(api.get("/game/x"), {"ok": True})  # a 429 first, then retried
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0].get_header("User-agent").startswith("openfront-rl/"))
        self.assertGreaterEqual(sum(slept), 1.0 + 5.0)  # the spacing, then the backoff

    def test_api_retries_a_response_cut_off_mid_read(self):
        calls = []

        class CutOff(io.BytesIO):
            def read(self, *args):
                raise http.client.IncompleteRead(b'{"ok"')

        def opener(req, timeout):
            calls.append(req)
            return CutOff() if len(calls) == 1 else io.BytesIO(b'{"ok": true}')

        api = crawl.Api(interval=0.0, opener=opener, sleep=lambda s: None)
        self.assertEqual(api.get("/game/x"), {"ok": True})  # a long crawl must not stop at one dropped connection
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
