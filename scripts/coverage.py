"""Gate 3B's coverage (docs/PLAN.md, 3B): the share of human commands outside the actions (15 until step 6, 20 with its
diplomacy and warship moves), over every held-out record of the scoring set. It replays the extractor's label stream
(bridge/replay.ts RecordGame.labels: windows of STEP_TICKS turns, at most MAX_PENDING commands queued per seat, the
server's disconnect notices skipped) without the simulation, so it reads all the records in seconds (test:
trainer/test_env.py checks it against the extractor's labels):

  .venv/Scripts/python scripts/coverage.py
"""
import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP_TICKS, MAX_PENDING = 5, 3  # bridge/game.ts, bridge/replay.ts
IN_SET = {"spawn", "attack", "boat", "upgrade_structure", "cancel_attack", "allianceRequest", "allianceExtension",
          "breakAlliance", "donate_troops", "move_warship"}  # the 20 actions' intents, besides the builds
BUILDS = {"City", "Defense Post", "Missile Silo", "Factory", "Port", "SAM Launcher", "Warship", "Atom Bomb",
          "Hydrogen Bomb"}  # bridge/actions.ts BUILDS


def label_kinds(record):
    """The kind of each command the extractor labels in a record, in order: "in" for the 20 actions, otherwise the
    command's type (build_unit:<unit> for builds outside the set)."""
    seat = {p["clientID"]: s for s, p in enumerate(record["info"]["players"])}
    windows = collections.defaultdict(list)
    for turn in record["turns"]:
        windows[turn["turnNumber"] // STEP_TICKS] += turn.get("intents", [])
    pending, kinds = [[] for _ in seat], []
    for w in range(max(windows, default=-1) + 1):
        for i in windows[w]:
            s = seat.get(i.get("clientID"))
            if s is not None and i["type"] != "mark_disconnected" and len(pending[s]) < MAX_PENDING:
                pending[s].append(i)
        for q in pending:
            if q:
                i = q.pop(0)
                t = i["type"]
                build = t == "build_unit"
                kinds.append("in" if t in IN_SET or (build and i.get("unit") in BUILDS)
                             else f"build_unit:{i.get('unit')}" if build else t)
    return kinds


def main():
    sys.path.insert(0, str(ROOT))
    from trainer import crawl
    from trainer.bc import is_val

    # Gate 3B's set, as score_records() was then: the old pin's held-out ranked 1v1 records of games before 2026-10-01
    entries = [e for e in crawl.load_index() if e["start"] < 1790812800000 and is_val(e["id"], 0.05)]
    paths = [p for p in (crawl.record_path(e["id"]) for e in sorted(entries, key=lambda e: e["id"])) if p.exists()]
    count = collections.Counter()
    for p in paths:
        count.update(label_kinds(json.loads(gzip.decompress(p.read_bytes()))))
    n = sum(count.values())
    print(f"{len(paths)} held-out records, {n:,} commands: {(n - count['in']) / n:.2%} outside the 20 actions "
          "(Gate 3B: < 0.5%)")
    for kind, k in count.most_common():
        if kind != "in":
            print(f"  {kind:28} {k:6} {k / n:.3%}")


if __name__ == "__main__":
    main()
