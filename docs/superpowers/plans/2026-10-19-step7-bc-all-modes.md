# Step 7: behaviour cloning on all modes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Behaviour-clone one arch v5 network on the human records of every mode, with the mode as an input, starting
from the 3B clone grown into v5.

**Architecture:** The bridge does three things:
- it adds 10 mode scalars to every observation;
- it replays each record through its two best-placed players (the best human of each of the two best-placed sides);
- it counts team wins in the value targets and keeps disconnected players' rows out of the loss.

The trainer does four:
- it picks usable records from `data/v0.35.0/` through new index fields;
- it gives each replay worker the records of one family (`--mix`);
- it grows `bc3b-v4` into v5 (`--grow`);
- it scores every statistic per lobby kind on a new fixed set.

**Tech Stack:** TypeScript bridge (node:test, tsx), Python 3.11 trainer (PyTorch, unittest), the pinned game v0.35.2.

**Spec:** `docs/superpowers/specs/2026-10-19-bc-all-modes-design.md` (approved by the user on 2026-10-19).

## Global Constraints

- **Commits:** they are authored by the repo owner. Never add `Co-Authored-By: Claude` or any other Claude attribution
  line to a commit message.
- **The game:** never edit anything inside `OpenFront_Source_Code/openfrontio/`. Reading it is fine.
- **Rule values:** read them from the game's Config at runtime, never hard-coded.
- **Determinism:** code that only builds observations may use `Math.*`; nothing whose output reaches the simulation
  may.
- **Older networks:** v1–v4 keep reading only their own inputs. `V4_CHANNELS`, `V4_FEATURES`, `V4_SCALARS` (26),
  `V4_ACTS` and `V4_MAPS` stay as they are.
- **Branch:** work on `phase3`, one commit per task, nothing pushed.
- **The user's files:** never touch or commit the untracked `0.34.24 to 0.35.3 changelog.txt` or `scripts/*dashboard*`.
- **The running crawl:** the top-up crawl (`runs/crawl035b.cmd`) may be appending to `data/v0.35.0/index.jsonl` while
  you work. No task writes that store, and tests use temporary stores.
- **Long runs:** the crawl, the smoke run, the night and the gate are the user's to start. "After the code" gives their
  commands.
- **Commands:**
  - bridge tests, from `OpenFront_Source_Code/openfrontio/`: `node --import tsx --test "../../bridge/*.test.ts"`, or one
    file with `node --import tsx --test ../../bridge/replay.test.ts`;
  - typecheck: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`;
  - trainer, from the repo root: `.venv/Scripts/python -m unittest discover -s trainer -t .` (needs the GPU and starts
    Node workers, ~9 min), or one module with `.venv/Scripts/python -m unittest trainer.test_bc -v`.

## Review Focus

1. **The reindex racing the running crawl.** `trainer.crawl --reindex` run while the top-up crawl appends to the index
   must not lose the crawl's lines. It refuses to replace an index that changed while it ran (Task 1 test).
2. **An index written by the older crawler.** Entries without the new fields stop `trainer.bc` with "run python -m
   trainer.crawl --reindex", never a KeyError (Task 5 test).
3. **A family the filters leave short.**
   - A `--mix` family with fewer usable training records than workers stops the run, naming the family and its count.
   - Validation and the score leave out a family without held-out records, and say so (Tasks 5 and 6 tests).
4. **Records the player choice can't use.** A record with fewer than two eligible humans, or a picked player missing
   from the game (kicked by team assignment), is rejected with its reason, and the worker loads the next record
   (Task 3 tests).
5. **A grown run resumed after a crash.** `--resume` of a `--grow` run loads its v5 checkpoint and never grows again
   (Task 4 test).

---

### Task 1: The index records what BC filters on

**Files:**
- Modify: `trainer/crawl.py` (`entry_of`, `elo`, `main`; new `reindex`)
- Test: `trainer/test_crawl.py`

**Interfaces:**
- Produces:
  - `INDEX_FIELDS = ("teams", "modifiers", "disabled", "bots", "nations")`, which `entry_of(record, mode)` sets;
  - `reindex(root=None) -> int`;
  - `python -m trainer.crawl --reindex`;
  - `elo(entries)`, which now rates ranked 1v1 entries only.
- The fields' values, from the record's `info.config`:
  - `teams`: its `playerTeams`, or None (for example 6, "Duos" or "Humans Vs Nations");
  - `modifiers`: the sorted keys of `publicGameModifiers` whose value is truthy (for example
    `["isCompact", "isWaterNukes"]`), and `[]` without it;
  - `disabled`: its `disabledUnits`, or `[]`;
  - `bots` and `nations`: its `bots` and `nations` ("default", "disabled" or a number).

- [ ] **Step 1: Write the failing tests** in `trainer/test_crawl.py`

```python
TEAM = ROOT / "bridge" / "testdata" / "team.json.gz"

def test_an_entry_holds_what_bc_filters_on(self):
    record = json.loads(gzip.decompress(TEAM.read_bytes()))
    e = crawl.entry_of(record, "team")
    self.assertEqual((e["teams"], e["modifiers"], e["disabled"], e["bots"], e["nations"]),
                     (6, ["isCompact", "isWaterNukes"], [], record["info"]["config"]["bots"], "disabled"))

def test_reindex_fills_old_entries_and_keeps_the_rest(self):
    # temp store: the Team fixture under records/, its index line without the INDEX_FIELDS
    # reindex(root) == 1; the line now has the five fields; every other key is unchanged; a second reindex(root) == 0

def test_reindex_refuses_an_index_that_changed_while_it_ran(self):
    # patch the record reading so that the first read appends a line to the index (a crawl at work)
    # reindex(root) raises SystemExit mentioning the crawl; the index still holds the appended line

def test_elo_rates_ranked_1v1_entries_only(self):
    # two ranked1v1 entries and one "team" entry with 18 players: no exception, and the team entry's id is not rated
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_crawl -v`
Expected: the four new tests FAIL (KeyError `teams`, no `reindex`, ValueError in `elo`).

- [ ] **Step 3: Implement `reindex(root=None) -> int` and the rest in `trainer/crawl.py`**
  - **What `reindex` does:**
    - it reads the index and refills the INDEX_FIELDS of each entry from its record (`entry_of(record, e["mode"])`);
    - it leaves entries without a record file as they are;
    - it returns how many entries changed.
  - **How it writes:** it replaces the file atomically, with a tmp file and replace as `purge` does.
  - **The race check:** before it replaces the file, it checks that the index's bytes are unchanged since it read
    them. If they changed, it raises SystemExit("<index> changed while reindexing: a crawl is writing it. Rerun when
    the crawl is done") and replaces nothing.
  - **The CLI:** `--reindex` prints `reindexed {n} entries in {store()}`.
  - **`elo`:** it skips entries whose `mode` (default "ranked1v1") is not ranked 1v1.

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/Scripts/python -m unittest trainer.test_crawl -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/crawl.py trainer/test_crawl.py
git commit -m "The index records what behaviour cloning filters on (step 7 Task 1): each entry keeps its lobby's playerTeams, the modifiers it set, its disabled units, its bots and its nations; trainer.crawl --reindex fills them in for the entries already there and refuses to replace an index that a crawl wrote meanwhile; Elo rates ranked 1v1 entries only"
```

### Task 2: The mode input

**Files:**
- Modify: `bridge/obs.ts`, `trainer/env.py`, `trainer/test_model.py` (`DIMS_V5`, `fake_obs`), `trainer/test_ppo.py`
  (its v5 dims)
- Test: `bridge/obs.test.ts`, `trainer/test_env.py`

**Interfaces:**
- **Produces, in `bridge/obs.ts`:**
  - `export const KINDS = ["ranked1v1", "ranked2v2", "rankedffa", "clanwars", "ffa", "team", "hvn"] as const`;
  - `export const MODE = 26`;
  - `SCALARS = 36`;
  - `export function lobbyKind(gc: GameConfig): number`, an index into KINDS.
- **Produces, in `trainer/env.py`:**
  - `MODE = 26`;
  - `KINDS`, in the same order;
  - `N_SCALARS = MODE + len(KINDS) + 3`, which is 36.
  - VecEnv raises ValueError when the worker's `dims["scalars"]` differs from N_SCALARS.
- **What `encodeSeat` writes** (its docstring lists them):
  - **26–32, the lobby kind, one-hot.** `lobbyKind` reads, in this order:
    - `rankedType`: 1v1 → ranked1v1, 2v2 → ranked2v2, ffa → rankedffa, clanwars → clanwars;
    - otherwise `gameMode` FFA → ffa, including private and singleplayer games;
    - otherwise `playerTeams` "Humans Vs Nations" → hvn;
    - any other Team game → team.
  - **33:** `min(log2(non-bot players) / 8, 1)`, counting humans and nations, alive or not.
  - **34:** `min(log2(teams) / 6, 1)` over the distinct teams of the non-bot players, and 0 when none has a team.
  - **35:** the nations' difficulty from the config's `difficulty` (Easy 0.25, Medium 0.5, Hard 0.75, Impossible 1),
    and 0 when the game has no nation.

- [ ] **Step 1: Write the failing tests** in `bridge/obs.test.ts`

```ts
test("the lobby kind comes from the config's ranked type, mode and teams", () => {
  // rankedType 1v1, 2v2, ffa, clanwars; unranked FFA; private FFA; Team with playerTeams 6; Humans Vs Nations
  // -> 0, 1, 2, 3, 4, 4, 5, 6
});
test("the mode scalars of ranked 1v1, FFA and Humans vs Nations training games", async () => {
  // EnvGame.create(RANKED, ...): scalars 26..35 are [1, 0, 0, 0, 0, 0, 0, 1 / 8, 0, 0] for both seats
  // EnvGame.create(HVN, ...): one-hot hvn; [33] = log2(4 seats + 4 nations) / 8 = 0.375; [34] = 1 / 6; [35] = 0.75
  // EnvGame.create(FFA, ...): one-hot ffa; [34] = 0; [35] = {Medium: 0.5, Hard: 0.75} of its config's difficulty
});
test("a record and a training game of the same kind give the same mode scalars", async () => {
  // a RecordGame of bridge/testdata/ranked1v1.json.gz and EnvGame.create(RANKED, ...): scalars 26..35 equal, every seat
});
```

  - In `trainer/test_env.py`, the test that opens a VecEnv on random play also asserts
    `env.dims["scalars"] == N_SCALARS`.

- [ ] **Step 2: Run them to see them fail**

Run: `node --import tsx --test ../../bridge/obs.test.ts` (game dir)
Expected: FAIL (`lobbyKind` missing, scalar 26 is 0).

- [ ] **Step 3: Implement it in `bridge/obs.ts` and `trainer/env.py`, then update the fixtures that describe the v5
  inputs**
  - `DIMS_V5["scalars"]` in `trainer/test_model.py`, and the v5 dims in `trainer/test_ppo.py` (near lines 343 and 385),
    become `N_SCALARS`.
  - `fake_obs` appends the 10 mode scalars after the 26. They are drawn after the old fields, so the older networks'
    tests keep their values, and they hold a one-hot kind `row % len(KINDS)` with three values in [0, 1].

- [ ] **Step 4: Run everything to see it pass**

Run: the typecheck, the bridge tests and the trainer tests.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bridge/obs.ts bridge/obs.test.ts trainer/env.py trainer/test_env.py trainer/test_model.py trainer/test_ppo.py
git commit -m "Observations carry the mode (step 7 Task 2, PLAN decision 5a): scalars 26-35 hold the lobby kind one-hot (ranked 1v1, ranked 2v2, Ranked FFA, Clan Wars, FFA, Team, Humans vs Nations) from the config's ranked type, mode and teams, then the non-bot players, the teams and the nations' difficulty, the same for a replayed record and a training game; v5 reads 36 scalars, and v2-v4 keep their 26"
```

### Task 3: Replay picks two players per game

**Files:**
- Modify: `bridge/game.ts` (`fromStart`), `bridge/replay.ts`, `bridge/worker.ts`, `trainer/env.py` (VecEnv)
- Test: `bridge/replay.test.ts`, `bridge/worker.test.ts`, `trainer/test_env.py`

**Interfaces:**
- **Produces:**
  - **`EnvGame.fromStart(start, stride, slots, latencySeed, choose?: (game: Game) => string[])`:** the seats are the
    players with `choose(game)`'s clientIDs, in that order (`game.playerByClientID`). Without it, every player of the
    start info, as today.
  - **`export function pickSeats(info: GameRecord["info"], game: { playerByClientID(id: string): { team(): string |
    null } | null }, count: number): string[]`.**
  - **`RecordGame.create(record, stride, slots, seats = 0)`:**
    - 0 means every player, as today (the scripts and the label tests);
    - otherwise it takes the players `pickSeats(info, game, seats)` returns, and fails when it throws.
  - **`RecordGame.outcome`:** +1 for a seat on the winning side, −1 otherwise.
  - **`RecordGame.labels()`:** a seat whose player is `isDisconnected()` at the window's start labels
    `[UNMAPPED, 0, 0, 0]`.
  - **The worker:**
    - its replay init takes `slots?: number`, the player table's rows (default: the config's);
    - replay picks `config.seats` players per record, and the old player-count check goes;
    - the reply header's `seats` are the picked players' clientIDs.
  - **`VecEnv(..., records, slots=None)`:**
    - `records` is either a list of paths shared out among the workers, as today, or a list holding one list of paths
      per worker;
    - `slots` goes into every worker's init.
- **`pickSeats`, the algorithm the tests pin:**

```
if count == the number of players: every player's clientID, in record order (ranked 1v1, as today)
side(id) = game.playerByClientID(id).team() ?? id
the winning side: winner ["player", id] -> side(id); ["team", name, ...] -> name; ["nation", ...] or none -> none
eligible: players present in the game (playerByClientID not null) whose stats show peak tiles (stats.tiles[0]) > 0
order the eligible: on the winning side first; alive at the end (no stats.killedAt) before dead; later killedAt
  first; more peak tiles first; then record order
pick: walk the order taking each player whose side has none picked yet, until count; then fill from the order
throw Error(`only ${n} eligible players`) when fewer than count are eligible
```

- [ ] **Step 1: Write the failing tests** in `bridge/replay.test.ts`

The expected IDs come from the fixtures' stats: in `team.json.gz`, Red won, and 6jEni9S5 (Red, 148,272 peak tiles,
alive) and EgHrXRwr (Yellow, 84,289, alive) placed best.

```ts
const ids = (rg: RecordGame) => rg.env.seats.map((p) => p.clientID());
test("two players a game: the best of the winning team and the best of the other teams", async () => {
  const rg = await RecordGame.create(readRecord(TEAM), 256, 160, 2);
  assert.deepEqual(ids(rg), ["6jEni9S5", "EgHrXRwr"]);
  assert.deepEqual(rg.outcome, [1, -1]);
});
test("the winning side comes first even when another side placed better", async () => {
  const record = readRecord(TEAM);
  record.info.winner = ["team", "Orange"];
  const rg = await RecordGame.create(record, 256, 160, 2);
  assert.deepEqual(ids(rg), ["QoxpLfYC", "6jEni9S5"]);
  assert.deepEqual(rg.outcome, [1, -1]);
});
test("a nation's win: both players by placement, both -1", async () => {
  // TEAM with winner ["nation", "Somewhere"] -> ["6jEni9S5", "EgHrXRwr"], outcome [-1, -1]
});
test("ranked 1v1 keeps both players in record order", async () => {
  // FIXTURE, seats 2 -> ["TP8tutXW", "crPmaJR7"], outcome [-1, 1] (crPmaJR7 won)
});
test("a team's win gives +1 to every player of the winning team", async () => {
  // TEAM, seats 0 (all 18): outcome[i] === 1 exactly for 7cQT5VuR, vsvieAD6 and 6jEni9S5
});
test("a disconnected player's window labels UNMAPPED", async () => {
  // FIXTURE, seats 0: TP8tutXW (seat 0) is disconnected at window 516 only; there labels()[0] is [UNMAPPED, 0, 0, 0]
});
test("a record without two eligible players is rejected", async () => {
  // TEAM with every player's stats deleted: RecordGame.create(record, 256, 160, 2) rejects /eligible/
});
test("a picked player missing from the game is skipped", async () => {
  // game = (await RecordGame.create(readRecord(TEAM), 256, 160)).env.game, seen through
  // { playerByClientID: (id) => (id === "6jEni9S5" ? null : game.playerByClientID(id)) }
  // pickSeats(info, that, 2) -> ["vsvieAD6", "EgHrXRwr"]
});
```

  - In `bridge/worker.test.ts`: a replay init with `slots: 512`, config seats 2 and the Team fixture replies with
    `dims.slots` 512, header `seats` ["6jEni9S5", "EgHrXRwr"] and `outcome` [1, -1].
  - In `trainer/test_env.py`:

```python
def test_replay_workers_take_their_own_records_and_two_players_each(self):
    env = VecEnv(RANKED_1V1, workers=2, games_per_worker=1, seed="mix", records=[[FIXTURE], [TEAM]], slots=512)
    # env.P == 512; env.seats_of == [["TP8tutXW", "crPmaJR7"], ["6jEni9S5", "EgHrXRwr"]]
    # env.outcome.tolist() == [[-1, 1], [1, -1]]; env.labels.shape == (4, 4)
```

- [ ] **Step 2: Run them to see them fail**

Run: `node --import tsx --test ../../bridge/replay.test.ts ../../bridge/worker.test.ts` (game dir)
Expected: FAIL (`pickSeats` missing; the seats are all players).

- [ ] **Step 3: Implement it**
  - **`RecordGame`:**
    - `seatOf` maps the picked seats only;
    - every turn's intents still execute in `advance()`, but only the picked seats are labelled.
  - **`replayPart`'s header:** it sends `rg.env.seats.map((p) => p.clientID())`.

- [ ] **Step 4: Run everything to see it pass**

Run: the typecheck, the bridge tests and the trainer tests.
Expected: PASS. The existing replay, coverage and Team tests are unchanged: they replay every player.

- [ ] **Step 5: Commit**

```bash
git add bridge/game.ts bridge/replay.ts bridge/worker.ts bridge/replay.test.ts bridge/worker.test.ts trainer/env.py trainer/test_env.py
git commit -m "Replays imitate two players a game and count team wins (step 7 Task 3): RecordGame picks the best-placed human of each of the two best-placed sides from the record's stats (the winning side first, then alive before dead, later deaths and more peak tiles; ranked 1v1 keeps both players), a seat on the winning team is +1, a disconnected player's window labels UNMAPPED, and a record without two eligible players is skipped; VecEnv takes one record list per worker and the player table's size"
```

### Task 4: Grow the 3B clone into v5

**Files:**
- Modify: `trainer/model.py` (new `grow`, `NEW_TYPE_BIAS`), `trainer/bc.py` (`--grow`)
- Test: `trainer/test_model.py`, `trainer/test_bc.py`

**Interfaces:**
- **Consumes:** `N_SCALARS`, through `DIMS_V5` (Task 2).
- **Produces:**
  - `NEW_TYPE_BIAS = -8.0`;
  - `grow(v4_state: dict, net: Policy) -> Policy`, which fills a fresh v5 `net` in place from a v4 `state_dict` and
    returns it;
  - `trainer.bc --init <arch v4 checkpoint> --grow`.
- **The rules,** which keep every v4 tensor:
  - **Same shape in v5:** copied.
  - **`stem.weight` and `skip.weight`:** v4's 57 input channels copied, the new ones zero. `skip`'s 13th output, the
    warship-source map, keeps v5's zero init.
  - **`embed.weight`:** columns 0–24 come from v4's 0–24, columns 25–30 are zero, and columns 31 onward come from v4's
    25 onward. The land features follow the player features.
  - **`scalars.0.weight`:** v4's 26 columns, and the 10 mode columns zero.
  - **`shuffle` (weight and bias):** v4's first 192 output channels (12 maps × 16). The rest keep v5's zero init.
  - **`type_head`:** v4's 15 rows. The 5 new rows get zero weights and a bias of NEW_TYPE_BIAS.
  - **`share_h` and `share_t`:** v4's first 2 × DS rows (attack, boat). The donation's rows keep v5's init.
  - **`share_out`:** `share_out.0` and `share_out.1` are copied, and `share_out.2` keeps v5's init.
  - **v5-only modules** (`diplomacy_query`, `dst_film`, `dst_shuffle`, `dst_skip`, `fleet_head`): they keep v5's init.
  - **Refusals:** ValueError when `net` is not a v5 Policy, or the state is not a v4 network's (its `type_head.weight`
    must have V4_ACTS rows).
- **`--grow`:**
  - With `--init` of an arch v4 checkpoint, the run's network is v5 with the checkpoint's core and widths, and its
    weights are `grow(checkpoint["model"], net)`.
  - It exits without `--init` ("--grow grows the --init checkpoint") and with an `--init` of another arch ("--grow takes
    an arch v4 checkpoint").
  - The run's args keep `grow: True` and `arch: "v5"`, so `--resume` loads the v5 checkpoint and never grows again.

- [ ] **Step 1: Write the failing tests**

```python
# trainer/test_model.py
class GrowTest(unittest.TestCase):
    def test_a_grown_network_starts_as_the_v4_network(self):
        exact_fp32(self)
        obs = fake_obs()
        torch.manual_seed(0)
        v4 = randomized(Policy(DIMS, core="moe-swiglu", shares=True, full=True)).cuda().eval()
        v5 = grow(v4.state_dict(), Policy(DIMS_V5, core="moe-swiglu", v5=True).cuda().eval())
        with torch.no_grad():
            a, b = v4(obs), v5(obs)
        close = lambda x, y: torch.allclose(x, y, atol=1e-5)
        self.assertTrue(close(a["type"], b["type"][:, :V4_ACTS]))
        self.assertTrue((b["type"][:, V4_ACTS:] == NEW_TYPE_BIAS).all())
        self.assertTrue(close(a["maps"], b["maps"][:, :V4_MAPS]))
        for k in ("ptr", "retreat", "value", "gates"):
            self.assertTrue(close(a[k], b[k]), k)
        self.assertTrue(close(a["shares"], b["shares"][:, :2]))

    def test_grow_takes_only_a_v4_state_into_a_v5_network(self):
        # a v3 network's state -> ValueError; a v4 state into a v4 network -> ValueError

# trainer/test_bc.py, MainTest
    def test_a_grown_run_starts_from_the_v4_checkpoint_and_resumes_without_growing(self):
        # init.pt: a v4 network (core "moe"), args {"arch": "v4", "core": "moe", "widths": "64,128,192"}
        # --init init.pt --grow --updates 0 -> latest.pt: args arch "v5", core "moe", grow True;
        #   model["type_head.bias"][15:] all -8; model["trunk.0.weight"] equal to init.pt's
        # --resume --updates 1 -> update 1
        # --grow without --init -> SystemExit "grows the --init"; --grow with a v5 --init -> SystemExit "arch v4"
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model.GrowTest -v`
Expected: FAIL (`grow` missing).

- [ ] **Step 3: Implement `grow` in `trainer/model.py` and `--grow` in `trainer/bc.py`**
  - `grow` works under `torch.no_grad()`, so the zeroed slices stay exactly zero.
  - In `bc.main`, the `--grow` arch switch comes after `network_args`, and the grow replaces the `--init` branch's
    `load_state_dict`.

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/Scripts/python -m unittest trainer.test_model trainer.test_bc -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/model.py trainer/bc.py trainer/test_model.py trainer/test_bc.py
git commit -m "The 3B clone grows into arch v5 (step 7 Task 4): grow() keeps every trained v4 weight, gives the new inputs zero weights (7 channels, the 6 diplomacy features between v4's features and the land features, the mode scalars) and the 5 new types a bias of -8, so the grown network starts as the v4 one; trainer.bc --init <v4 checkpoint> --grow trains it, and a resume loads the v5 checkpoint"
```

### Task 5: BC trains on every family's records

**Files:**
- Create: `bridge/maptable.ts`
- Modify: `trainer/env.py` (`map_table`), `trainer/bc.py`
- Test: `trainer/test_env.py`, `trainer/test_bc.py`

**Interfaces:**
- **Consumes:** `INDEX_FIELDS` (Task 1), and `VecEnv(records=<one list per worker>, slots=...)` (Task 3).
- **Produces:**
  - **`bridge/maptable.ts`** prints
    `{"grid": [rows, cols], "maps": {"<map name>": {"h": rows, "w": cols, "nations": n}}}`:
    - `grid` is the grid `shapeOf` gives the specialists' setups (`ranked1v1`, `ffa`, `hvn`);
    - each map has its `mapGrid` at Normal size and its `nationCount`.
    - The game's own logs go to stderr.
  - **`trainer/env.py`:** `map_table() -> dict`, cached. It runs the script once with node from GAME_DIR.
  - **`trainer/bc.py`:**
    - `FAMILIES = ("ranked1v1", "ranked2v2", "rankedffa", "ffa", "team", "hvn")`.
    - `family_of(entry) -> str`: the crawl mode, except that "clanwars" becomes "team", and a "team" entry whose `teams`
      is "Humans Vs Nations" becomes "hvn".
    - `usable(entry, maps, grid) -> bool`:
      - the map's `h` and `w` fit `grid`;
      - `modifiers` holds nothing but "isCompact";
      - `disabled` is empty.
      - It raises SystemExit("data/v0.35.0's index predates step 7 ... run python -m trainer.crawl --reindex") for an
        entry without the INDEX_FIELDS.
    - `mix_of(a) -> dict[str, int]`:
      - `--mix FAMILY=WORKERS ...`, in the order given;
      - without `--mix`, `{"ranked1v1": a.workers}`;
      - SystemExit for an unknown or repeated family, or a count under 1.
    - `worker_records(entries, mix) -> list[list[Path]]`:
      - for each family in mix order, its entries' record paths in the store (`crawl.record_path(id, crawl.store())`),
        sorted by game ID and dealt round-robin to its workers;
      - SystemExit naming the family and its count when it has fewer records than workers.
    - `val_records(entries, mix) -> tuple[list[list[Path]], list[str]]`:
      - one list per family of the mix that has held-out entries, in mix order (one validation worker each);
      - also the families left out because they have none.
    - `slots_of(entries, maps) -> int`: 1 + the most humans + nations + bots of any entry. Nations count as:
      - 0 for "disabled";
      - the number, when it is one;
      - otherwise max(the map's nation count, the humans), since a Humans vs Nations game has as many nations as humans.
- **`parse_args`:** with `--mix`, `--workers` becomes its sum, as in PPO. `--mix` is a run setting, so a resume keeps it.
- **`main`:**
  - **The store** is `crawl.store()` (`data/v0.35.0`). The old pin's `data/` is no longer read here: its records desync
    at v0.35, and the pin keeps its own code.
  - **Entries:** indexed, with a record file, and `usable`. They split into training and held-out by `is_val`, as today.
  - **Training:** `VecEnv(RANKED_1V1, sum(mix), G, seed, records=worker_records(train, mix),
    slots=slots_of(every usable entry, maps))`.
    - RANKED_1V1's 2 seats are the 2 players picked per record, and its grid is the specialists'.
    - `main` checks that `(envs.H, envs.W) == tuple(grid)`.
  - **Validation** (unless `--val-workers 0`; the flag's help says so):
    - one worker per family that has held-out records (`val_records`), 2 games each, with the same slots;
    - a family without held-out records is left out, with a control-page note;
    - a run that starts at update 0 validates once before its first update. That gives the forgetting trigger its
      baseline: the grown start's own validation NLL.
  - **`run_info`'s "Data"** gives the records per family. `elo_rows` is unchanged; it uses Task 1's Elo.

- [ ] **Step 1: Write the failing tests**

```python
# trainer/test_env.py
def test_the_map_table_has_the_specialists_grid(self):
    t = map_table()
    # t["grid"] == [192, 256]; "Australia" fits it; "Brain Rot" (294 x 375) does not; t["maps"]["World"]["nations"] > 0

# trainer/test_bc.py
class RecordsTest(unittest.TestCase):
    def test_the_family_of_an_entry(self):
        # ranked2v2 -> ranked2v2; clanwars -> team; team with teams "Humans Vs Nations" -> hvn; team with teams 6 -> team
    def test_usable_records_fit_the_grid_and_set_no_modifier(self):
        # Australia with [] -> True; with ["isCompact"] -> True; with ["isWaterNukes"] -> False;
        # disabled ["SAM Launcher"] -> False; map "Brain Rot" -> False
    def test_an_index_from_the_older_crawler_asks_for_a_reindex(self):
        # an entry without "teams" -> SystemExit mentioning "--reindex"
    def test_each_family_is_dealt_to_its_workers(self):
        # ranked1v1 entries a, b, c and team entry t; mix {"ranked1v1": 2, "team": 1} -> [[a, c], [b], [t]] (paths by id)
    def test_a_family_short_of_records_stops_the_run(self):
        # mix {"hvn": 2} with one hvn entry -> SystemExit mentioning "hvn" and "1"
    def test_validation_leaves_out_a_family_without_held_out_records(self):
        # held-out entries only for ranked1v1; mix {"ranked1v1": 2, "team": 1} -> ([[the ranked1v1 paths]], ["team"])
    def test_the_player_table_fits_the_largest_game(self):
        # (2 humans, 400 bots, nations "disabled") and (50 humans, nations "default" on a 20-nation map, 400 bots)
        # -> 1 + 50 + 50 + 400

# MainTest: setUp writes the store self.data / "v0.35.0": the 1v1 fixture as a ranked1v1 entry on "Australia", with
# all INDEX_FIELDS (teams None, modifiers [], disabled [], bots 400, nations "disabled")
    def test_each_family_trains_on_its_own_records(self):
        # adds the Team fixture as a "team" entry: map "Between Two Seas", teams 6, modifiers [], disabled [], its bots,
        #   nations "disabled"
        # --mix ranked1v1=1 team=1 --updates 1 -> the training VecEnv got records [[fixture], [team]] and
        #   slots >= 1 + 18 + its bots; latest.pt global_step == 1 * 2 * 4 (1 update of 2 windows, 2 workers x 1 game
        #   x 2 players)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_bc trainer.test_env -v`
Expected: the new tests FAIL. The old MainTests pass once their setUp writes the new store.

- [ ] **Step 3: Implement it**

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/Scripts/python -m unittest trainer.test_bc trainer.test_env -v`, then the whole trainer suite.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bridge/maptable.ts trainer/env.py trainer/bc.py trainer/test_env.py trainer/test_bc.py
git commit -m "Behaviour cloning trains on every mode's records (step 7 Task 5): trainer.bc reads data/v0.35.0, keeps the records whose map fits the specialists' grid (bridge/maptable.ts) and that set no modifier but Compact and disable no unit, and with --mix FAMILY=WORKERS gives each replay worker one family's records (ranked 1v1, ranked 2v2, Ranked FFA, public FFA, Team with Clan Wars, Humans vs Nations); validation runs one worker per family, and the player table fits the run's largest game"
```

### Task 6: Scores per lobby kind, and the fixed v0.35 scoring set

**Files:**
- Modify: `trainer/bc.py` (`measure`, `score_records`, `score`, `table`, `score_main`)
- Test: `trainer/test_bc.py`

**Interfaces:**
- **Consumes:** `MODE` and `KINDS` (Task 2); `family_of`, `usable`, `slots_of` and `map_table` (Task 5).
- **Produces, in `measure()`:**
  - `labels/illegal/<type>`: for every type in `ACT_NAMES[1:]`, the share of the rows labelled with it that fall
    outside the masks.
  - **Per lobby kind** (the rows whose one-hot `obs["scalars"][:, MODE:MODE + len(KINDS)]` is that kind):
    - `bc/nll_<kind>`, over this network's rows;
    - `bc/nll_common_<kind>`, over the legal rows of v4's 15 types, for every network;
    - `router/<kind>_<i>`.
  - `bc/nll_common` over all kinds.
  - **Uniform baselines:**
    - `bc/nll_<map>_cell_uniform`: the log of the legal cells of that map, over the rows of its type;
    - `bc/nll_<type>_target_uniform`, for the four diplomatic types: the log of the legal recipients;
    - `bc/nll_warship_dst_uniform`: the log of the legal destinations.
- **Produces, for the score:**
  - `SCORE_CUTOFF = 1792411200000` (2026-10-19 12:00 UTC);
  - `SCORE_LAYOUT = {"workers_per_family": 2, "games_per_worker": 6, "steps": 1500}`.
  - `score_records(entries, maps, grid) -> dict[str, list[Path]]`:
    - for each family, in FAMILIES order, the held-out (`is_val(id, 0.05)`) usable entries of games started before
      SCORE_CUTOFF that have a record, sorted by ID;
    - families without any are left out.
  - `score(paths, records, games_per_worker, steps, slots, device="cuda")`:
    - `records` holds one list per worker, and the workers are `len(records)`;
    - `score_main` adds `"families"`, {family: records}, to the result.
  - `table(result)` also prints the router lines per kind.
  - `score_main`:
    - it puts each family on min(2, its records) workers and calls `score` with `slots_of` of the scored entries;
    - it prints the families left out;
    - it keeps saving to `runs/score/`.

- [ ] **Step 1: Write the failing tests** in `trainer/test_bc.py`

```python
def test_illegal_labels_are_counted_per_type(self):
    # two attack labels, one aimed at a target outside its attack mask -> labels/illegal/attack == 0.5
def test_scores_per_lobby_kind_on_every_networks_rows(self):
    # fake_obs-style rows with kind one-hots row % 7; a v4 and a v5 network: both report bc/nll_common_<kind> for the
    # kinds present; only the v5 one has rows of the new types in bc/nll_<kind>
def test_a_uniform_pick_scores_its_baseline(self):
    # RandomPolicy (zero logits) on a v5 batch: bc/nll_<map>_cell == bc/nll_<map>_cell_uniform for each map with rows,
    # bc/nll_<type>_target == bc/nll_<type>_target_uniform, bc/nll_warship_dst == bc/nll_warship_dst_uniform
def test_the_scoring_set_ignores_later_and_training_games(self):
    # held-out and before the cutoff -> in; held-out after the cutoff -> out; training before it -> out;
    # held-out but unusable -> out; a family with no held-out entry is not a key of the result
# ScoreTest: score(paths, [[FIXTURE]], games_per_worker=1, steps=30, slots=403), with the same assertions as today
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement it**

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/bc.py trainer/test_bc.py
git commit -m "Behaviour cloning scores each mode apart on a fixed v0.35 set (step 7 Task 6): every statistic per lobby kind, the NLL on the rows of v4's 15 types for like-for-like comparisons with the 3B clone, labels outside the masks per action type, and uniform baselines for the cell, target and destination heads; --score scores the held-out records of games before 2026-10-19 12:00 UTC, two workers per family"
```

### Task 7: The eval reports the action mix per type

**Files:**
- Modify: `trainer/eval.py` (`play`)
- Test: `trainer/test_eval.py`

**Interfaces:**
- **Produces:** `play()`'s meters gain two dicts, over the types policy A took (the first action of a decision):
  - `"types"`: {type name: its share of A's decisions that act};
  - `"invalid_by_type"`: {type name: the share of that type's actions the bridge refused}.

- [ ] **Step 1: Write the failing test**

```python
def test_the_meters_give_the_action_mix_and_the_refusals_per_type(self):
    # a short random vs random match: sum(meters["types"].values()) == 1 (1e-9); the keys are in ACT_NAMES[1:];
    # every invalid_by_type value is in [0, 1]
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python -m unittest trainer.test_eval -v`
Expected: FAIL (KeyError `types`).

- [ ] **Step 3: Implement it**

- [ ] **Step 4: Run it to see it pass**

Run: `.venv/Scripts/python -m unittest trainer.test_eval -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/eval.py trainer/test_eval.py
git commit -m "Matches report policy A's action mix and the bridge's refusals per action type (step 7 Task 7), for Gate 7's per-mode action report"
```

### Task 8: Docs and launchers

**Files:**
- Modify: `CLAUDE.md`, `docs/PLAN.md`, `docs/superpowers/specs/2026-10-19-bc-all-modes-design.md`
- Create (git-ignored, not committed): `runs/bc7.cmd`, `runs/gate7.cmd`

- [ ] **Step 1: Update `CLAUDE.md`**
  - **Training commands:**
    - the BC night's command, with `--init checkpoints/bc3b-v4/latest.pt --grow --lr 1e-4 --mix ranked1v1=3
      ranked2v2=2 rankedffa=1 ffa=4 team=3 hvn=1 --games-per-worker 6`;
    - `python -m trainer.crawl --reindex`.
  - **The "Observations" bullets:** the mode scalars 26–35.
  - **"Networks":** v5 reads 36 scalars, and older networks read 26.
  - **The store bullets:** `trainer.bc` reads `data/v0.35.0` by family, and `--score` scores the v0.35 set. The old set
    is scored only from the pin.
  - **The replay facts:**
    - two players a game;
    - team wins count;
    - disconnected players' windows are out of the loss;
    - a nations' win is −1 for both players.
- [ ] **Step 2: Update `docs/PLAN.md`**
  - **"After 4N" item 4:**
    - the design and the plan, and what was built (the commits);
    - the operations that remain;
    - the expected and, once counted, the actual records per family.
  - **The routing test moves to Phase 6's model growth,** in 3C's section and in Phase 6's list, with its reason: the
    grown clone inherits its core.
- [ ] **Step 3: Update the design's status line:** approved on 2026-10-19, the plan's path, and the commits it was
  built in.
- [ ] **Step 4: Write `runs/bc7.cmd`**
  - It uses `runs/crawl035b.cmd`'s layout: a console title, notes, attempts logged to `runs/bc7-runs.log`, and up to 6
    attempts 60 s apart.
  - Each attempt runs `.venv\Scripts\python.exe -u -m trainer.bc --run bc7 --resume --updates 24000` into
    `runs\bc7.log`.
  - `checkpoints/bc7` exists by then (see "After the code").
- [ ] **Step 5: Write `runs/gate7.cmd`.** It runs these into `runs\gate7-*.log`:
  - `trainer.bc --score checkpoints/bc3b-v4/latest.pt checkpoints/bc7/latest.pt`;
  - `trainer.eval --a checkpoints/bc7/latest.pt --b checkpoints/bc3b-v4/latest.pt --episodes 256 --workers 16
    --games-per-worker 2`;
  - for each clone, `trainer.eval --setup ffa|hvn --difficulty Medium|Hard --a <clone> --maps training --timer 30
    --episodes 56 --workers 14 --games-per-worker 2`.
- [ ] **Step 6: Commit the docs** (not `runs/`)

```bash
git add CLAUDE.md docs/PLAN.md docs/superpowers/specs/2026-10-19-bc-all-modes-design.md
git commit -m "Step 7 docs (step 7 Task 8): how behaviour cloning on all modes runs (families, the grown start, the mode input, the replay's two players a game), the routing test moved to Phase 6's model growth, and the operations before and after the night"
```

---

## After the code: the runs (each needs the user's go)

1. **The crawl ends** (`runs\crawl035b-done.txt`). Then run `.venv/Scripts/python -m trainer.crawl --reindex`.
2. **Counts:** the usable training and held-out records per family. They go into PLAN's passes table, and the mix
   changes if a family is far from the design's expectation.
3. **Smoke run** (~10 min, PC idle):

   ```
   .venv/Scripts/python -m trainer.bc --run bc7-smoke --init checkpoints/bc3b-v4/latest.pt --grow --lr 1e-4 --mix ranked1v1=3 ranked2v2=2 rankedffa=1 ffa=4 team=3 hvn=1 --games-per-worker 6 --updates 60 --val-every 30
   ```

   - Check for finite losses, the `perf/samples_per_s`, and the peak GPU memory (Task Manager's dedicated and shared
     memory).
   - Then delete `checkpoints/bc7-smoke` and `runs/bc7-smoke`.
4. **The grown start and its score** (~20 min):
   - Create it (it also validates the start):

     ```
     .venv/Scripts/python -m trainer.bc --run bc7 --init checkpoints/bc3b-v4/latest.pt --grow --lr 1e-4 --mix ranked1v1=3 ranked2v2=2 rankedffa=1 ffa=4 team=3 hvn=1 --games-per-worker 6 --updates 0
     ```

   - Score it: `.venv/Scripts/python -m trainer.bc --score checkpoints/bc3b-v4/latest.pt checkpoints/bc7/latest.pt`.
   - **Pass:** the two clones' `bc/nll_common_ranked1v1` agree to ~1e-3.
   - **Into PLAN:** each new type's `labels/illegal/<type>` (step 6's leftover check).
5. **The night** (~15 h). Run `explorer.exe runs\bc7.cmd`, with Windows Update paused. It resumes `bc7`, so it keeps
   the run's learning rate and mix.
   - **The forgetting trigger** (the design's risk): in the first 2,500 updates, `val/bc/nll_common_ranked1v1` must
     stay within +0.02 of its update-0 value.
   - If it doesn't, stop the run and restart it from the grown start at `--lr 5e-5`, or with more ranked 1v1 workers.
6. **Gate 7** (~3 h). Run `explorer.exe runs\gate7.cmd`, then put the results into PLAN against the design's bars.

## Execution notes

- Dispatches can batch tasks that share files or are small:
  - Tasks 1 and 2;
  - Task 3;
  - Task 4;
  - Tasks 5 and 6 (both in `bc.py`);
  - Tasks 7 and 8.
- **Task reviews:** Tasks 3 and 5 move the data (labels, value targets and record selection), so they get task
  reviews.
- **The final review** reads the whole branch against the design.
