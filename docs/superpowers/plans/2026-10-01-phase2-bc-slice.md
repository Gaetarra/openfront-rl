# Phase 2 — Behaviour-Cloning Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A policy cloned from ranked 1v1 human replays, on the 5 Phase 1 actions, clearly beats random play and a
scripted expander in ranked 1v1. PPO started from it, with a KL penalty toward it, beats a frozen copy of it. The
evaluation games can be watched in the game's own client.

**Architecture:**
- **Data:** a crawler (`trainer/crawl.py`) downloads ranked 1v1 records from the public API into a gzip cache with an
  index. Usernames are blanked and deleted players purged.
- **Replay extraction:** a new worker mode replays records (`bridge/replay.ts`) through the same game builder, cell
  frame and observation code as training games. Every 5 ticks it sends each human seat's observation with a label: the
  command that player issued in the next 5 turns, mapped to the 5 actions.
- **Training:** `trainer/env.py` builds the observations on the GPU exactly as in training. `trainer/bc.py` minimises
  −log π(label) under the masks, plus a value loss on the game outcome, while the workers already replay the next
  window.
- **One grid:** every game, env or replay, now sits on one padded cell grid per trainer, so the 4 ranked maps and
  both map sizes mix in one batch.
- **Ranked setup:** ranked 1v1 training games use a seeded copy of the official ranked 1v1 playlist.
- **Evaluation and RL:** `trainer/eval.py` plays two policies against each other. `trainer/ppo.py` trains one seat
  against a frozen opponent seat. Evaluation games can be saved as game records and served to the game's dev client.

**Tech Stack:** as Phase 1. Node 24 with the game's `tsx`/TypeScript and `node:test`. Python 3.11 with torch 2.14 cu130,
numpy, psutil, tensorboard, `unittest`, and stdlib `urllib`/`gzip`/`zlib`.

**Spec:** `docs/PLAN.md`:
- "Phase 2 — Behaviour-cloning slice on ranked 1v1"
- decisions 9–12
- the coverage note under "Action space (v1)"

## Global Constraints

- **Code location:** never edit anything inside `OpenFront_Source_Code/openfrontio/`. Our code lives in `bridge/`,
  `trainer/` and `scripts/`.
- **Commits:** plain messages, with no `Co-Authored-By` or any other Claude attribution line.
- **Rule values** come from the game's `Config` at runtime.
  - The ranked 1v1 playlist is copied from `get1v1Config` in `src/server/MapPlaylist.ts` (v0.34.21), with a comment.
    That function draws with `Math.random`, so it cannot be called.
- **Determinism:** anything that reaches the simulation uses seeded `PseudoRandom`.
  - A game's RNG is seeded from its game ID, as the game's client does, so recorded games replay in the official
    client.
  - Other draws (playlist, latency, record order) use their own seeds.
- **API use** (cleared by the owner on 2026-09-30):
  - at most 1 request per second, with a User-Agent that names the project;
  - every record cached and never fetched twice;
  - deleted players purged at every crawl (`docs/API.md` "Recently Deleted Players");
  - usernames blanked in the cache.
- **Long runs:** the full crawl, BC training and PPO start detached (see `CLAUDE.md`), and only after the user's OK.
- **Timing:** decisions every Δ = 5 ticks, and labels use the same windows.
- **Dependencies:** no new npm or Python packages.
- **GPU memory:** a ranked grid is 192 × 256 cells, so one observation row is about 2.2 MB (22 bf16 channels). Keep
  rollout buffers at or below about 2k rows.
- **Where commands run:** TS commands from `OpenFront_Source_Code/openfrontio/`. Python commands from the repo root
  with `.venv/Scripts/python`.

## Review Focus

- **Replay parity:** an observation built from a record equals the one built from the same env game. Test: a recorded
  env game replayed through `RecordGame` gives the same blobs and hashes (Task 8).
- **Label alignment:**
  - The commands executed in turns [5w, 5w + 5) label the observation taken after turn 5w − 1.
  - Extra commands queue, at most 3, into later windows.
  - Commands outside the action set are excluded, never turned into no-ops.
  - Labels illegal under the masks are excluded and counted.
  - Tests in Tasks 4 and 5.
- **Bad records:**
  - A record that desyncs is cut at its first mismatching hash.
  - A record that fails to load or crashes the simulation is skipped.
  - Neither stops the stream (Task 4).
- **Padded grid:** cells outside a game's map are never legal actions, labels or observations (Tasks 2 and 5).
- **Deleted players:** their records leave the cache and the index at the next crawl (Task 3).

---

## File structure

| File | Responsibility |
|---|---|
| `bridge/setups.ts` | New. Game IDs from seeds, the ranked 1v1 playlist, and the grid and player-table size per config |
| `bridge/maps.ts` | Adds `mapGrid`: a map's cell grid, read from its manifest without building the map |
| `bridge/frame.ts` | Puts the cell frame on the trainer's padded grid (`stride`); adds a fifth terrain plane, inside-map |
| `bridge/game.ts` | Builds `EnvGame` from a `GameStartInfo`, the way the client does, for env games and records alike; can record a game (Task 8) |
| `bridge/obs.ts` | `encodeSeat` takes the timer instead of the whole config |
| `bridge/replay.ts` | New. `RecordGame`: replays a record window by window, checks hashes, labels each seat |
| `bridge/worker.ts` | Adds per-game `cellSize`, replay mode and recording |
| `trainer/env.py` | Padded grid from the worker, per-game cell area and valid cells, replay mode, `send`/`recv` split, setups |
| `trainer/crawl.py` | New. Crawler, index, Elo, purge of deleted players |
| `trainer/bc.py` | New. Behaviour cloning training with control, checkpoints and validation |
| `trainer/eval.py` | Matches between two policies on fixed seeds; scripted expander; recording |
| `trainer/ppo.py` | Ranked setup, `--init`, frozen `--opponent`, KL penalty to the initial policy |
| `trainer/control.py` | `GET /game/<id>` serves saved records, with CORS, to the game's dev client |
| `trainer/watch.py` | New. Serves `replays/` when no training is running |
| `bridge/testdata/ranked1v1.json.gz` | A short real ranked 1v1 record, usernames and account IDs blanked, for tests |

**Protocol changes** (both sides depend on these):

- **init**
  - Env mode is as in Phase 1, plus an optional `"record": {"dir", "gitCommit", "games"}` (Task 8).
  - Replay mode: `{"cmd":"init","config":EnvConfig,"replay":{"records":[path,...]},"games":K,"seed":str,"worker":int}`.
    The config only fixes the grid and the player table.
- **dims** (init reply)
  - `grid: [H, W]` is the padded grid (multiples of 16) that fits every game the config can produce.
  - `slots` is fixed per config.
  - `cellSize` moves to the per-game headers.
- **Per-game header**
  - `cellSize` on reset.
  - Replay mode adds `record` (game ID), `seats` (clientIDs), `outcome` (±1 per seat), `left` (windows left), and on
    the finished game `finished`, `desync` and `error`.
  - Recording adds `recorded` (game ID) on the finished game.
- **Blobs**
  - `terrain` has 5 planes: land, shore, ocean, height, inside-map.
  - Cell indexes in `cells`, `units` and actions are grid indexes, y × W + x.
  - Replay mode adds `labels`: i32 [seats × 4] of type, cell, target, bin. Type −1 means a command outside the action
    set, −2 a command that could not be mapped.

---

### Task 1: Game setups — game IDs, ranked 1v1 playlist, grids

**Files:**
- Create: `bridge/setups.ts`, `bridge/setups.test.ts`
- Modify: `bridge/maps.ts` (add `mapGrid`), `bridge/game.ts` (`EnvConfig.playlist`), `bridge/testkit.ts` (`RANKED`)

**Interfaces:**
- Produces:
  - `gameIdOf(seed): string`
  - `RANKED_1V1_MAPS`
  - `ranked1v1Config(random): GameConfig`
  - `gameConfigOf(cfg, random): GameConfig`
  - `gridFor(cfg): [number, number]`
  - `slotsFor(cfg): number`
  - `mapGrid(map, size): {h, w, cs}`
  - `EnvConfig.playlist?: "ranked1v1"`

- [ ] **Step 1: Write the failing test** — `bridge/setups.test.ts`

```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { GameMapSize, GameMapType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GAME_ID_REGEX, GameConfigSchema } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { gameIdOf, gridFor, ranked1v1Config, slotsFor } from "./setups";
import { ONION, RANKED } from "./testkit";

test("game IDs are valid, stable and differ between seeds", () => {
  const id = gameIdOf("run-w0-g0-e0");
  assert.match(id, GAME_ID_REGEX);
  assert.equal(gameIdOf("run-w0-g0-e0"), id);
  assert.notEqual(gameIdOf("run-w0-g0-e1"), id);
});

test("the ranked 1v1 playlist has the official weights and valid configs", () => {
  const random = new PseudoRandom(7);
  const n = 2000;
  const maps = new Map<string, number>();
  let compact = 0;
  for (let i = 0; i < n; i++) {
    const c = ranked1v1Config(random);
    GameConfigSchema.parse(c);
    maps.set(c.gameMap, (maps.get(c.gameMap) ?? 0) + 1);
    const small = c.gameMapSize === GameMapSize.Compact;
    if (small) compact++;
    assert.equal(c.bots, small ? 100 : 400);
    assert.equal(c.maxTimerValue, small ? 10 : 15);
  }
  assert.ok(Math.abs(maps.get(GameMapType.Australia)! / n - 0.4) < 0.04);
  for (const m of [GameMapType.Iceland, GameMapType.Asia, GameMapType.EuropeClassic]) {
    assert.ok(Math.abs(maps.get(m)! / n - 0.2) < 0.03, m);
  }
  assert.ok(Math.abs(compact / n - 0.2) < 0.03);
});

test("the grid fits every map a config can produce, padded to multiples of 16", () => {
  assert.deepEqual(gridFor(ONION), [64, 64]);
  assert.deepEqual(gridFor(RANKED), [192, 256]); // Australia/Iceland 188 rows, all four 250 columns
  assert.equal(slotsFor(ONION), 5);
  assert.equal(slotsFor(RANKED), 403);
});
```

In `bridge/testkit.ts` add, after `ONION`:

```ts
/** Ranked 1v1 training games: map, size, bots and timer come from the playlist (bridge/setups.ts). */
export const RANKED: EnvConfig = { ...ONION, seats: 2, playlist: "ranked1v1" };
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test ../../bridge/setups.test.ts`
Expected: FAIL, `Cannot find module '.../bridge/setups'`.

- [ ] **Step 3: Implement**

In `bridge/game.ts`, add to `EnvConfig`, after `seats`:

```ts
  /** Draw each game's map, size, bots and timer from this playlist (bridge/setups.ts); then only `seats` is used. */
  playlist?: "ranked1v1";
```

In `bridge/maps.ts`, add after `loadFreshTerrain`:

```ts
/**
 * A map's cell grid without building the map: Normal maps use 8x8-tile cells on the full-size map, Compact maps 4x4 on
 * the half-size map, so both sizes of a map share one grid.
 */
export function mapGrid(map: GameMapType, size: GameMapSize): { h: number; w: number; cs: number } {
  const manifest = JSON.parse(mapFiles(map).manifest) as MapManifest;
  const normal = size === GameMapSize.Normal;
  const m = normal ? manifest.map : manifest.map4x;
  const cs = normal ? 8 : 4;
  return { h: Math.ceil(m.height / cs), w: Math.ceil(m.width / cs), cs };
}
```

`bridge/setups.ts`:

```ts
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  RankedType,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GameConfig } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import type { EnvConfig } from "./game";
import { mapGrid } from "./maps";

const ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/**
 * The game ID of a seed: 8 alphanumerics, as GAME_ID_REGEX requires. A game seeds its RNG from its ID (as the game's
 * client does when it replays a record), so equal seeds give equal games.
 */
export function gameIdOf(seed: string): string {
  const random = new PseudoRandom(simpleHash(seed));
  let id = "";
  for (let i = 0; i < 8; i++) id += ALNUM[random.nextInt(0, ALNUM.length)];
  return id;
}

/** Ranked 1v1 maps with the official weights: Australia 40%, the others 20% each. */
export const RANKED_1V1_MAPS: GameMapType[] = [
  GameMapType.Australia,
  GameMapType.Australia,
  GameMapType.Iceland,
  GameMapType.Asia,
  GameMapType.EuropeClassic,
];

/**
 * A ranked 1v1 game as the official servers create it: a copy of get1v1Config in src/server/MapPlaylist.ts (v0.34.21),
 * which draws with Math.random. Compare it with the game's on every upgrade.
 */
export function ranked1v1Config(random: PseudoRandom): GameConfig {
  const compact = random.next() < 0.2;
  return {
    donateGold: false,
    donateTroops: false,
    gameMap: RANKED_1V1_MAPS[random.nextInt(0, RANKED_1V1_MAPS.length)],
    maxPlayers: 2,
    gameType: GameType.Public,
    gameMapSize: compact ? GameMapSize.Compact : GameMapSize.Normal,
    difficulty: Difficulty.Medium,
    rankedType: RankedType.OneVOne,
    infiniteGold: false,
    infiniteTroops: false,
    maxTimerValue: compact ? 10 : 15,
    instantBuild: false,
    randomSpawn: false,
    nations: "disabled",
    gameMode: GameMode.FFA,
    bots: compact ? 100 : 400,
    spawnImmunityDuration: 30 * 10,
    disabledUnits: [],
  };
}

/** The game config of one training game. `random` is only used for playlist draws. */
export function gameConfigOf(cfg: EnvConfig, random: PseudoRandom): GameConfig {
  if (cfg.playlist === "ranked1v1") {
    if (cfg.seats !== 2) throw new Error("the ranked1v1 playlist needs 2 seats");
    return ranked1v1Config(random);
  }
  return {
    gameMap: cfg.map as GameMapType,
    gameMapSize: cfg.compact ? GameMapSize.Compact : GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Private,
    difficulty: cfg.difficulty as Difficulty,
    nations: cfg.nations > 0 ? cfg.nations : "disabled",
    bots: cfg.bots,
    donateGold: false,
    donateTroops: false,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
    maxTimerValue: cfg.timerMinutes,
  };
}

/** Every map and size a config can produce. */
function mapsOf(cfg: EnvConfig): [GameMapType, GameMapSize][] {
  if (cfg.playlist === "ranked1v1") {
    return [...new Set(RANKED_1V1_MAPS)].flatMap((m): [GameMapType, GameMapSize][] => [
      [m, GameMapSize.Normal],
      [m, GameMapSize.Compact],
    ]);
  }
  return [[cfg.map as GameMapType, cfg.compact ? GameMapSize.Compact : GameMapSize.Normal]];
}

/** The trainer's cell grid [rows, columns]: fits every map the config can produce, padded to multiples of 16 (the CNN
 *  downsamples by 16). */
export function gridFor(cfg: EnvConfig): [number, number] {
  let h = 0;
  let w = 0;
  for (const [map, size] of mapsOf(cfg)) {
    const g = mapGrid(map, size);
    h = Math.max(h, g.h);
    w = Math.max(w, g.w);
  }
  return [Math.ceil(h / 16) * 16, Math.ceil(w / 16) * 16];
}

/** Player-table rows: row 0 (unowned land), the seats, the nations and the most bots any game of the config has. */
export function slotsFor(cfg: EnvConfig): number {
  if (cfg.playlist === "ranked1v1") return 1 + 2 + 400;
  return 1 + cfg.seats + cfg.nations + cfg.bots;
}
```

- [ ] **Step 4: Run the test and the typecheck**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test ../../bridge/setups.test.ts`
Expected: PASS (3 tests).
Run: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add bridge/setups.ts bridge/setups.test.ts bridge/maps.ts bridge/game.ts bridge/testkit.ts
git commit -m "Game setups: seeded ranked 1v1 playlist, game IDs and grids"
```

---

### Task 2: One padded grid for every game

All games of a trainer share one cell grid, so ranked maps of different sizes, and both map sizes, mix in one batch.
The worker indexes cells as y × W + x on that grid. The game is built from a `GameStartInfo` the way the game's
client builds it, which the replay extractor (Task 4) reuses.

**Files:**
- Modify: `bridge/frame.ts`, `bridge/game.ts`, `bridge/obs.ts`, `bridge/worker.ts`, `bridge/game.test.ts`,
  `bridge/obs.test.ts`, `trainer/env.py`, `trainer/test_env.py`

**Interfaces:**
- Consumes: Task 1 (`gameConfigOf`, `gameIdOf`, `gridFor`, `slotsFor`).
- Produces:
  - `CellFrame(map, cs, stride?)` with `stride`, `n = ch × stride`; `tilesOf` of a padding cell is `[]`.
  - `terrainCells` returns 5 planes.
  - `EnvGame.fromStart(start, stride, slots, latencySeed)`; `EnvGame.start`, `.timerMinutes`, `.hashTick`.
  - `encodeSeat(game, seat, timerMinutes, cellSize, ...)`.
  - Python: `ONION`, `RANKED_1V1`; `VecEnv.area` [E] (tiles per cell); the VALID channel per game.

- [ ] **Step 1: Write the failing tests**

Append to `bridge/game.test.ts` (add `landTile` and `RANKED` to the testkit import):

```ts
test("a frame on a wider grid indexes cells by grid row and leaves the padding empty", async () => {
  const env = await EnvGame.create(ONION, "pad");
  script(env, 600);
  const plain = new CellFrame(env.game.map(), env.frame.cs);
  const padded = new CellFrame(env.game.map(), env.frame.cs, 80);
  plain.encode(true);
  padded.encode(true);
  assert.equal(padded.n, plain.ch * 80);
  for (let y = 0; y < plain.ch; y++) {
    for (let x = 0; x < 80; x++) {
      const c = y * 80 + x;
      if (x < plain.cw) {
        assert.equal(padded.o1[c], plain.o1[y * plain.cw + x]);
      } else {
        assert.equal(padded.o1[c], 0);
        assert.deepEqual(padded.tilesOf(c), []);
      }
    }
  }
  const tile = landTile(env);
  assert.ok(padded.tilesOf(padded.cellOf(tile)).includes(tile));
});

test("ranked games draw from the playlist and share the ranked grid", async () => {
  const env = await EnvGame.create(RANKED, "ranked-a");
  assert.equal(env.seats.length, 2);
  assert.equal(env.frame.stride, 256);
  assert.equal(env.slots, 403);
  assert.equal(env.start.config.rankedType, "1v1");
  assert.match(env.start.gameID, /^[A-Za-z0-9]{8}$/);
});
```

Append to `trainer/test_env.py` (import `VALID` and `RANKED_1V1` from `trainer.env`):

```python
    def test_ranked_games_share_one_padded_grid(self):
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=2, seed="grid")
        try:
            obs = env.obs
            self.assertEqual(obs["spatial"].shape, (4, N_CHANNELS, 192, 256))
            self.assertEqual(obs["players"].shape[1], 403)
            valid = obs["spatial"][:, VALID].float().flatten(1) > 0
            self.assertTrue(set(valid.sum(1).tolist()) <= {47000, 37500, 31250})  # the ranked maps' cells
            self.assertFalse((obs["cell_mask"] & ~valid[:, None]).any(), "no legal cell outside the map")
            self.assertTrue(set(env.area.tolist()) <= {16.0, 64.0})
            env.step(np.zeros((4, 4), np.int64))
        finally:
            env.close()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test ../../bridge/game.test.ts`
Expected: FAIL (`CellFrame` has no stride parameter; `RANKED` games are not created from the playlist).
Run: `.venv/Scripts/python -m unittest trainer.test_env -v`
Expected: FAIL, `ImportError: cannot import name 'VALID'` or `'RANKED_1V1'`.

- [ ] **Step 3: Put the frame on the padded grid** — `bridge/frame.ts`

Replace the class fields, constructor, `cellOf`, `tilesOf`, `center` and the first lines of `compute`:

```ts
export class CellFrame {
  /** Cells per row of this game's map. */
  readonly cw: number;
  readonly ch: number;
  /** Cells per grid row: the trainer's grid width, shared by every game of a worker (padding cells hold no tiles). */
  readonly stride: number;
  /** Grid cells up to the map's last row: ch x stride. */
  readonly n: number;
  // ... (o1, o2, n1, n2, fo and the private fields stay as they are)

  constructor(
    map: GameMap,
    readonly cs: number,
    stride?: number,
  ) {
    this.w = map.width();
    this.h = map.height();
    this.cw = Math.ceil(this.w / cs);
    this.ch = Math.ceil(this.h / cs);
    this.stride = stride ?? this.cw;
    if (this.stride < this.cw) throw new Error(`grid width ${this.stride} < map width ${this.cw} cells`);
    this.n = this.ch * this.stride;
    // ... (the array allocations stay as they are, sized by this.n)
  }

  cellOf(tile: TileRef): number {
    const x = tile % this.w;
    const y = (tile - x) / this.w;
    return Math.floor(y / this.cs) * this.stride + Math.floor(x / this.cs);
  }

  /** The cell's tiles, row by row; none for a padding cell. */
  tilesOf(cell: number): TileRef[] {
    const cx = cell % this.stride;
    if (cx >= this.cw) return [];
    const x0 = cx * this.cs;
    const y0 = Math.floor(cell / this.stride) * this.cs;
    const tiles: TileRef[] = [];
    for (let y = y0; y < Math.min(y0 + this.cs, this.h); y++) {
      for (let x = x0; x < Math.min(x0 + this.cs, this.w); x++) tiles.push(y * this.w + x);
    }
    return tiles;
  }

  /** Tile coordinates of the cell's centre. */
  center(cell: number): [number, number] {
    return [(cell % this.stride) * this.cs + this.cs / 2, Math.floor(cell / this.stride) * this.cs + this.cs / 2];
  }
```

In `compute(c)`, use the stride:

```ts
    const x0 = (c % this.stride) * this.cs;
    const y0 = Math.floor(c / this.stride) * this.cs;
```

A padding cell has `x0 >= w`, so its tile loop is empty and it stays all zero.

Replace `terrainCells`:

```ts
/** Static terrain per cell, 5 planes of frame.n bytes: land, shore (land next to water) and ocean tiles, the mean height
 *  (magnitude 0-31) of the land, and 1 for cells inside the map (0 = grid padding). */
export function terrainCells(map: GameMap, frame: CellFrame): Uint8Array {
  const out = new Uint8Array(5 * frame.n);
  for (let c = 0; c < frame.n; c++) {
    const tiles = frame.tilesOf(c);
    let land = 0;
    let shore = 0;
    let ocean = 0;
    let height = 0;
    for (const t of tiles) {
      if (map.isLand(t) && !map.isImpassable(t)) {
        land++;
        height += map.magnitude(t);
        if (map.isShore(t)) shore++;
      } else if (map.isOcean(t)) {
        ocean++;
      }
    }
    out[c] = land;
    out[frame.n + c] = shore;
    out[2 * frame.n + c] = ocean;
    out[3 * frame.n + c] = land > 0 ? Math.round(height / land) : 0;
    out[4 * frame.n + c] = tiles.length > 0 ? 1 : 0;
  }
  return out;
}
```

- [ ] **Step 4: Build games from their start info** — `bridge/game.ts`

Replace the imports and everything from `export class EnvGame {` down to (not including) `/** Executes one turn`:

```ts
import { Config } from "../OpenFront_Source_Code/openfrontio/src/core/configuration/Config";
import { Executor } from "../OpenFront_Source_Code/openfrontio/src/core/execution/ExecutionManager";
import {
  Game,
  GameMapSize,
  Player,
  PlayerInfo,
  PlayerType,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { createGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameImpl";
import { GameUpdateType, HashUpdate } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameUpdates";
import { createNationsForGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/NationCreation";
import { GameRunner } from "../OpenFront_Source_Code/openfrontio/src/core/GameRunner";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GameStartInfo, StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { attackMask, decodeAction, N_ACTS, typeMask } from "./actions";
import { CellFrame, encodeUnits, terrainCells } from "./frame";
import { LatencyQueue } from "./latency";
import { loadFreshTerrain, resetGameStatics } from "./maps";
import { encodeSeat, PLAYER_FEATURES, SCALARS } from "./obs";
import { rewardOf, seatStanding } from "./ranking";
import { gameConfigOf, gameIdOf, gridFor, slotsFor } from "./setups";
```

(`EnvConfig`, `STEP_TICKS`, `seatClientID`, `Blob` and `bytes` stay as they are.)

```ts
export class EnvGame {
  done = false;
  rewards: number[] = [];
  ranks: number[] = [];
  /** Non-bot players in the game: the n of the reward formula. */
  nPlayers = 0;
  /** Per seat: its last action could not be applied and was dropped. */
  invalid: boolean[];
  private turn = 0;
  private masks: { types: Uint8Array; attack: Uint8Array }[] | null = null;

  private constructor(
    /** How the game was created: a record of it replays from this. */
    readonly start: GameStartInfo,
    readonly game: Game,
    readonly seats: Player[],
    readonly frame: CellFrame,
    /** Rows of the player table: smallID 0 (unowned land) up to the config's largest player count. */
    readonly slots: number,
    private readonly runner: GameRunner,
    private readonly sim: { hash: number; hashTick: number; fatal: string | null },
    private readonly latency: LatencyQueue,
  ) {
    this.invalid = seats.map(() => false);
  }

  /** Latest state hash reported by the game (every 10 ticks). */
  get hash(): number {
    return this.sim.hash;
  }

  /** Tick of the latest hash. */
  get hashTick(): number {
    return this.sim.hashTick;
  }

  get timerMinutes(): number {
    return this.start.config.maxTimerValue ?? 60;
  }

  /** A training game: its game ID and every random draw derive from the seed. */
  static async create(cfg: EnvConfig, seed: string): Promise<EnvGame> {
    const config = gameConfigOf(cfg, new PseudoRandom(simpleHash(`${seed}:setup`)));
    const players = Array.from({ length: cfg.seats }, (_, i) => ({
      clientID: seatClientID(i),
      username: `agent${i}`,
      clanTag: null,
    }));
    const start: GameStartInfo = { gameID: gameIdOf(seed), lobbyCreatedAt: 0, config, players };
    return EnvGame.fromStart(start, gridFor(cfg)[1], slotsFor(cfg), `${seed}:latency`);
  }

  /**
   * Builds a game from its start info the way the game's client does (tests/replay/ReplayGame.ts): the RNG is seeded
   * from the game ID and the seats are the start info's players, in order. `stride`: the trainer's grid width in cells.
   */
  static async fromStart(start: GameStartInfo, stride: number, slots: number, latencySeed: string): Promise<EnvGame> {
    resetGameStatics();
    const config = start.config;
    const terrain = await loadFreshTerrain(config.gameMap, config.gameMapSize);
    const random = new PseudoRandom(simpleHash(start.gameID));
    const humans = start.players.map(
      (p) =>
        new PlayerInfo(
          p.username,
          PlayerType.Human,
          p.clientID,
          random.nextID(),
          p.isLobbyCreator ?? false,
          p.clanTag,
          p.friends ?? [],
          p.teamIndex ?? null,
        ),
    );
    const nations = createNationsForGame(start, terrain.nations, terrain.additionalNations, humans.length, random);
    const game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      new Config(config, null, false),
      terrain.teamGameSpawnAreas,
    );
    const frame = new CellFrame(game.map(), config.gameMapSize === GameMapSize.Compact ? 4 : 8, stride);
    const sim = { hash: 0, hashTick: -1, fatal: null as string | null };
    const executor = new Executor(game, start.gameID, undefined, start.tribes?.map((t) => t.name));
    const runner = new GameRunner(game, executor, (gu) => {
      if ("errMsg" in gu) {
        sim.fatal = gu.errMsg;
        return;
      }
      frame.mark(gu.packedTileUpdates);
      for (const h of gu.updates[GameUpdateType.Hash] as HashUpdate[]) {
        sim.hash = h.hash;
        sim.hashTick = h.tick;
      }
    });
    runner.init();
    const seats = humans.map((h) => game.player(h.id));
    const latency = new LatencyQueue(new PseudoRandom(simpleHash(latencySeed)));
    return new EnvGame(start, game, seats, frame, slots, runner, sim, latency);
  }
```

In `observe`, pass `this.timerMinutes` instead of `this.cfg`. In `checkDone`, use `this.timerMinutes` instead of
`this.cfg.timerMinutes`.

- [ ] **Step 5: The observation takes the timer** — `bridge/obs.ts`

Remove `import type { EnvConfig } from "./game";`. Change the parameter `cfg: EnvConfig` to `timerMinutes: number`, and
its two uses `cfg.timerMinutes` to `timerMinutes`. In `bridge/obs.test.ts`, call
`encodeSeat(env.game, seat, env.timerMinutes, env.frame.cs, attack, players, rel, scalars)`.

- [ ] **Step 6: Per-game cell size in the worker** — `bridge/worker.ts`

Import `gridFor` and `slotsFor` from `./setups`, and replace `dims`:

```ts
function dims(cfg: EnvConfig): Record<string, unknown> {
  return {
    grid: gridFor(cfg),
    slots: slotsFor(cfg),
    seats: cfg.seats,
    playerFeatures: PLAYER_FEATURES,
    scalars: SCALARS,
    acts: N_ACTS,
    troopBins: TROOP_BINS,
    unitTypes: UNIT_TYPES,
  };
}
```

In `handle`, return `encode(parts, dims(msg.config))`. In `fresh`, add the cell size to the reset header:
`{ reset: true, cellSize: games[g].frame.cs, ...extra }`.

- [ ] **Step 7: The trainer takes grid indexes** — `trainer/env.py`

After `DENSITY_FEATURE`, add the setups:

```python
# The Phase 1 gate setup, and ranked 1v1 games (the worker draws map, size, bots and timer from the official ranked 1v1
# playlist, bridge/setups.ts; of the other fields only "seats" is used).
ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "timerMinutes": 5, "seats": 1}
RANKED_1V1 = {"map": "Australia", "compact": False, "nations": 0, "difficulty": "Medium", "bots": 400,
              "timerMinutes": 15, "seats": 2, "playlist": "ranked1v1"}
```

In `VecEnv.__init__`, replace everything from `d = replies[0][0]["dims"]` to `self.fo = torch.zeros_like(self.n1)`:

```python
        d = replies[0][0]["dims"]
        self.H, self.W = d["grid"]  # padded by the worker to multiples of 16 (the CNN downsamples by 16)
        self.P, self.S = d["slots"], d["seats"]
        self.F, self.K, self.T = d["playerFeatures"], d["scalars"], d["acts"]
        self.N = self.E * self.S
        self.dims = {**d, "channels": N_CHANNELS, "H": self.H, "W": self.W}
        dev, hw = self.device, self.H * self.W
        self.unit_group = torch.tensor([UNIT_GROUP.get(t, -1) for t in d["unitTypes"]], device=dev)
        self.transport = d["unitTypes"].index("Transport")
        self.terrain = torch.zeros(self.E, 5, hw, device=dev)  # land, shore, ocean, height, inside the map
        self.area = torch.full((self.E,), 64.0, device=dev)  # tiles per cell of each game
        self.o1 = torch.zeros(self.E * hw, dtype=torch.long, device=dev)
        self.o2 = torch.zeros_like(self.o1)
        self.n1 = torch.zeros(self.E * hw, device=dev)
        self.n2 = torch.zeros_like(self.n1)
        self.fo = torch.zeros_like(self.n1)
```

In `step`, drop the padding conversion; cells are grid indexes:

```python
    def step(self, actions):
        """actions: int array [N, 4] of [type, cell, target, bin]; cells are grid indexes (row x W + column)."""
        a = np.asarray(actions, dtype=np.int64).reshape(self.E, self.S, 4)
```

In `_absorb`, replace the reset block:

```python
                if g.get("reset"):
                    resets.append(e)
                    terr = np.frombuffer(b["terrain"], np.uint8).reshape(5, -1)
                    self.terrain[e].zero_()
                    self.terrain[e][:, :terr.shape[1]] = torch.from_numpy(terr.astype(np.float32)).to(self.device)
                    self.area[e] = float(g["cellSize"]) ** 2
```

In `_apply_cells`, the index is the grid index:

```python
        idx = torch.from_numpy(env * hw + rec["idx"].astype(np.int64)).to(dev)
```

In `_add_units`: `cell = torch.from_numpy(rec["cell"].astype(np.int64)).to(dev)`.

In `_observe`, use each game's cell area and valid cells:

```python
        N, S, P, hw, dev = self.N, self.S, self.P, self.H * self.W, self.device
        row_env = torch.arange(N, device=dev) // S
        area = self.area[row_env][:, None]  # [N, 1]
        rel_t = torch.from_numpy(rel.reshape(N, P).astype(np.int64)).to(dev)
        pl = torch.from_numpy(players.reshape(N, P, self.F)).to(dev)
        x = torch.zeros(N, N_CHANNELS, hw, device=dev)
        terr = self.terrain[row_env]
        valid = terr[:, 4] > 0
        x[:, LAND:OCEAN + 1] = terr[:, 0:3] / area[:, None]
        x[:, HEIGHT] = terr[:, 3] / 31
        x[:, VALID] = valid.float()
```

and further down use `frac = n[flat] / area`, `x[:, FALLOUT] = self.fo[flat] / area`, and end the cell mask with
`& valid[:, None]` instead of `& self.valid`.

- [ ] **Step 8: Run all tests and the typecheck**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test --test-timeout=120000 "../../bridge/*.test.ts" && node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: all bridge tests pass (the worker test still sees grid [64, 64] and 64 × 64 cell records on Onion); tsc
exit 0.
Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK.
Run: `.venv/Scripts/python scripts/diag_baselines.py` for about a minute, then Ctrl+C.
Expected: its first line is `timer 2 min | random | mean rank 3.78 | ...`. The numbers can differ slightly from
PLAN's (game IDs changed the seeds); what matters is that it runs.

- [ ] **Step 9: Commit**

```bash
git add bridge/frame.ts bridge/game.ts bridge/obs.ts bridge/worker.ts bridge/game.test.ts bridge/obs.test.ts trainer/env.py trainer/test_env.py
git commit -m "One padded cell grid per trainer; games built from their start info"
```

---
### Task 3: Crawler, index, Elo and purge

**Files:**
- Create: `trainer/crawl.py`, `trainer/test_crawl.py`, `bridge/testdata/ranked1v1.json.gz` (fixture)
- Modify: `.gitignore` (add `replays/`)

**Interfaces:**
- Produces:
  - `Api(interval, opener, sleep, clock).get(path, **params)`; returns `None` on 404.
  - `list_games(api, start, end, ranked="1v1")`
  - `is_void(record)`, `blank(record, public_ids=False)`, `entry_of(record)`
  - `record_path(game_id)`, `save(record)`, `load_index()`, `append_index(entry)`
  - `purge(api, now=None)`, `elo(entries) -> {game ID: {clientID: elo before the game}}`
  - `crawl(api, since, until, limit=None)`
  - Module paths `DATA`, `RECORDS`, `INDEX`, `SKIPPED`, `PURGED_AT`; tests patch them.
- Data layout (git-ignored `data/`):
  - `records/<id>.json.gz`: the full record, usernames blanked.
  - `index.jsonl`: one line per usable record:
    `{"id", "start", "commit", "map", "size", "turns", "players": [{"clientID", "publicID"}], "winner"}`.
  - `skipped.txt`: IDs of void records, never fetched again.
  - `purged_at.txt`: time of the last deleted-players sync.

- [ ] **Step 1: Write the failing test** — `trainer/test_crawl.py`

```python
import datetime as dt
import gzip
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from trainer import crawl


def record(game_id, a="A1", b="B1", winner="ca", spawned=("ca", "cb"), start=0):
    turns = [{"turnNumber": 5, "intents": [{"type": "spawn", "tile": 1, "clientID": c} for c in spawned]}]
    return {"gitCommit": "x" * 40, "turns": turns,
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
        self.patch = mock.patch.multiple(crawl, DATA=data, RECORDS=data / "records", INDEX=data / "index.jsonl",
                                         SKIPPED=data / "skipped.txt", PURGED_AT=data / "purged_at.txt")
        self.patch.start()
        self.t0 = dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)

    def tearDown(self):
        self.patch.stop()

    def test_void_records(self):
        self.assertFalse(crawl.is_void(record("g1")))
        self.assertTrue(crawl.is_void(record("g1", winner=None)))
        self.assertTrue(crawl.is_void(record("g1", spawned=("ca",))))

    def test_blank_keeps_simulation_inputs(self):
        r = crawl.blank(record("g1"))
        self.assertEqual([p["username"] for p in r["info"]["players"]], ["Player1", "Player2"])
        self.assertEqual(r["info"]["players"][0]["publicID"], "A1")
        self.assertIsNone(crawl.blank(record("g1"), public_ids=True)["info"]["players"][0]["publicID"])

    def test_crawl_saves_usable_records_once_and_remembers_void_ones(self):
        api = FakeApi(games=[{"game": "good", "start": "t"}, {"game": "void", "start": "t"}],
                      records={"good": record("good"), "void": record("void", winner=None)})
        hour = dt.timedelta(hours=1)
        self.assertEqual(crawl.crawl(api, self.t0, self.t0 + hour), (1, 1))
        with gzip.open(crawl.record_path("good"), "rt", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["info"]["players"][0]["username"], "Player1")
        self.assertEqual([e["id"] for e in crawl.load_index()], ["good"])
        self.assertEqual(crawl.load_index()[0]["winner"], "ca")
        fetches = sum(1 for p, _ in api.calls if p.startswith("/game/"))
        self.assertEqual(crawl.crawl(api, self.t0, self.t0 + hour), (0, 0))
        self.assertEqual(sum(1 for p, _ in api.calls if p.startswith("/game/")), fetches, "nothing fetched twice")

    def test_purge_removes_deleted_players_records(self):
        api = FakeApi(games=[{"game": "g1", "start": "t"}, {"game": "g2", "start": "t"}],
                      records={"g1": record("g1", b="GONE"), "g2": record("g2")})
        crawl.crawl(api, self.t0, self.t0 + dt.timedelta(hours=1))
        self.assertEqual(crawl.purge(FakeApi(deleted=["GONE"])), 1)
        self.assertEqual([e["id"] for e in crawl.load_index()], ["g2"])
        self.assertFalse(crawl.record_path("g1").exists())
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_crawl -v`
Expected: FAIL, `ImportError: cannot import name 'crawl'`.

- [ ] **Step 3: Implement** — `trainer/crawl.py`

```python
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
    """GET requests, at most one per `interval` seconds, retried with backoff on HTTP 429, 5xx and network errors."""

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
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_crawl -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Save the test fixture**

The fixture is a real, short ranked 1v1 record at the pinned version. It is Compact so the tests stay fast.

```bash
.venv/Scripts/python - <<'EOF'
import datetime as dt, gzip, json
from trainer import crawl
api = crawl.Api()
now = dt.datetime.now(dt.timezone.utc)
for meta in crawl.list_games(api, now - dt.timedelta(hours=12), now):
    secs = (crawl.parse_time(meta["end"]) - crawl.parse_time(meta["start"])).total_seconds()
    if not 150 <= secs <= 300:
        continue
    rec = api.get(f"/game/{meta['game']}")
    if rec and rec["info"]["config"]["gameMapSize"] == "Compact" and not crawl.is_void(rec):
        with gzip.open("bridge/testdata/ranked1v1.json.gz", "wt", encoding="utf-8") as f:
            json.dump(crawl.blank(rec, public_ids=True), f, separators=(",", ":"))
        print("fixture:", meta["game"], rec["gitCommit"], secs, "s")
        break
EOF
```

Expected: one line, `fixture: <id> d7075bfa... <secs> s`. The file is under 200 KB. Put its game ID in the comment at
the top of `bridge/replay.test.ts` (Task 4).

In `.gitignore`, add `replays/` to the "Training outputs" block. That folder holds the evaluation games saved for
watching (Task 8).

- [ ] **Step 6: Commit**

```bash
git add trainer/crawl.py trainer/test_crawl.py bridge/testdata/ranked1v1.json.gz .gitignore
git commit -m "Crawler for ranked 1v1 records: polite API client, index, Elo and deleted-player purge"
```

- [ ] **Step 7: Ask the user before the full crawl.** At 1 request per second it takes hours (about 2.4k games a
  day since 23 Sep). With their OK, start it detached so it runs while Tasks 4–8 are built. It logs to
  `data/crawl.log`, using the launcher described in `CLAUDE.md`:

```bash
.venv/Scripts/python -c "import subprocess, sys, pathlib; pathlib.Path('data').mkdir(exist_ok=True); log = open('data/crawl.log', 'ab'); subprocess.Popen([sys.executable, '-m', 'trainer.crawl'], stdout=log, stderr=log, creationflags=0x00000200 | 0x08000000 | 0x01000000)"
```

---

### Task 4: Replay extractor and replay mode

**Files:**
- Create: `bridge/replay.ts`, `bridge/replay.test.ts`
- Modify: `bridge/worker.ts`, `bridge/worker.test.ts`, `trainer/env.py`, `trainer/test_env.py`

**Interfaces:**
- Consumes: `EnvGame.fromStart`, `.start`, `.hash`, `.hashTick`, `.observe`, `.terrain`, `.tickWith` (Task 2); the
  fixture (Task 3).
- Produces:
  - `readRecord(file)`, `binOf(troops, have)`, `labelOf(env, seat, intent)`
  - `OUT_OF_SET = -1`, `UNMAPPED = -2`, `MAX_PENDING = 3`
  - `RecordGame.create(record, stride, slots)`, with `.labels()`, `.advance()`, `.done`, `.desync`, `.dropped`,
    `.left`, `.outcome`, `.env`, `.record`
  - Worker replay mode.
  - Python: `VecEnv(..., records=[paths])`, `.send(actions=None)`, `.recv()`, `.labels` ([N, 4] long on the
    device), `.outcome` ([E, S]), `.left` ([E]), `.record_of`, `.seats_of`.

- [ ] **Step 1: Write the failing tests** — `bridge/replay.test.ts`

```ts
// ranked1v1.json.gz: ranked 1v1 game <ID from Task 3>, Compact, v0.34.21; usernames and account IDs blanked.
import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GameRecord, StampedIntent, Turn } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { decompressGameRecord } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { Act } from "./actions";
import { seatClientID } from "./game";
import { binOf, OUT_OF_SET, readRecord, RecordGame } from "./replay";
import { gameConfigOf } from "./setups";
import { ONION } from "./testkit";

const FIXTURE = path.join(path.dirname(fileURLToPath(import.meta.url)), "testdata/ranked1v1.json.gz");

test("troop shares map to the nearest bin in log scale", () => {
  assert.equal(binOf(10, 100), 0);
  assert.equal(binOf(30, 100), 1); // 0.3: nearer 0.25 than 0.5
  assert.equal(binOf(40, 100), 2); // 0.4: nearer 0.5 than 0.25 in log scale
  assert.equal(binOf(100, 100), 4);
  assert.equal(binOf(500, 100), 4);
  assert.equal(binOf(1, 100), 0);
});

test("a blanked ranked record replays in sync and labels every window", async () => {
  const rg = await RecordGame.create(readRecord(FIXTURE), 256, 403);
  assert.equal(rg.env.seats.length, 2);
  assert.deepEqual([...rg.outcome].sort(), [-1, 1]);
  const counts = new Map<number, number>();
  while (!rg.done) {
    rg.env.observe(rg.window === 0);
    for (const [type] of rg.labels()) counts.set(type, (counts.get(type) ?? 0) + 1);
    rg.advance();
  }
  assert.equal(rg.desync, false, "blanked usernames keep the replay in sync");
  assert.ok((counts.get(Act.Spawn) ?? 0) >= 2, "both players spawned");
  assert.ok((counts.get(Act.Attack) ?? 0) > 20);
  assert.ok(counts.get(Act.Noop)! > counts.get(Act.Attack)!, "most windows have no command");
});

function synthetic(turns: Turn[], numTurns: number): GameRecord {
  const players = [0, 1].map((i) => ({
    clientID: seatClientID(i),
    username: `agent${i}`,
    clanTag: null,
    persistentID: null,
    stats: {},
  }));
  const config = gameConfigOf({ ...ONION, seats: 2, nations: 0 }, new PseudoRandom(1));
  const info = {
    gameID: "Synth001",
    lobbyCreatedAt: 0,
    config,
    players,
    start: 0,
    end: 0,
    duration: 0,
    num_turns: numTurns,
    winner: ["player", seatClientID(0)],
    lobbyFillTime: 0,
  };
  return decompressGameRecord({ info, version: "v0.0.2", gitCommit: "DEV", turns } as unknown as GameRecord);
}

test("extra commands queue into later windows; unexpressible ones stay out of the loss", async () => {
  const a = seatClientID(0);
  const b = seatClientID(1);
  const attack = (troops: number): StampedIntent => ({ type: "attack", targetID: null, troops, clientID: a });
  const silo: StampedIntent = { type: "build_unit", unit: UnitType.MissileSilo, tile: 0, clientID: b };
  const record = synthetic(
    [
      { turnNumber: 300, intents: [attack(100), attack(50), silo] },
      { turnNumber: 400, intents: [attack(1), attack(2), attack(3), attack(4), attack(5)] },
    ],
    420,
  );
  const rg = await RecordGame.create(record, 64, 3);
  const types: number[][] = [];
  while (!rg.done) {
    types.push(rg.labels().map((l) => l[0]));
    rg.advance();
  }
  assert.deepEqual(types[59], [Act.Noop, Act.Noop]);
  assert.deepEqual(types[60], [Act.Attack, OUT_OF_SET]);
  assert.deepEqual(types[61], [Act.Attack, Act.Noop]); // the second attack of turn 300, one window later
  assert.deepEqual(types[62], [Act.Noop, Act.Noop]);
  assert.deepEqual(types.slice(80, 84).map((t) => t[0]), [Act.Attack, Act.Attack, Act.Attack, Act.Noop]);
  assert.equal(rg.dropped, 2);
});
```

Append to `bridge/worker.test.ts`:

```ts
test("replay mode streams observations with labels and moves on to the next record", async () => {
  const fixture = path.join(path.dirname(WORKER), "testdata/ranked1v1.json.gz");
  const c = new Client();
  const init = await c.request({
    cmd: "init",
    config: { ...ONION, seats: 2, playlist: "ranked1v1" },
    replay: { records: [fixture] },
    games: 1,
    seed: "rep",
    worker: 0,
  });
  assert.deepEqual(init.header.dims.grid, [192, 256]);
  const g0 = init.header.games[0];
  assert.equal(g0.reset, true);
  assert.equal(g0.seats.length, 2);
  assert.deepEqual([...g0.outcome].sort(), [-1, 1]);
  assert.deepEqual(g0.blobs.at(-1), ["labels", 2 * 4 * 4]);
  let finished: any = null;
  for (let step = 0; step < 2000 && finished === null; step++) {
    const g = (await c.request({ cmd: "step" })).header.games[0];
    if (g.done) finished = g;
  }
  c.close();
  assert.ok(finished !== null, "the record ended and the next one started");
  assert.equal(finished.reset, true);
  assert.equal(finished.desync, false);
  assert.equal(finished.finished, g0.record);
});
```

Append to `trainer/test_env.py` (add `import collections`; import `ROOT` and `RANKED_1V1` from `trainer.env`):

```python
    def test_replay_stream_labels_every_seat(self):
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=1, seed="rep",
                     records=[ROOT / "bridge" / "testdata" / "ranked1v1.json.gz"])
        try:
            seen = collections.Counter()
            for _ in range(200):
                self.assertEqual(tuple(env.labels.shape), (2, 4))
                seen.update(env.labels[:, 0].tolist())
                env.step()
            self.assertGreater(seen[0], seen[2], "no-ops outnumber attacks")
            self.assertEqual(sorted(env.outcome[0].tolist()), [-1.0, 1.0])
            self.assertEqual(len(env.seats_of[0]), 2)
        finally:
            env.close()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test ../../bridge/replay.test.ts`
Expected: FAIL, `Cannot find module '.../bridge/replay'`.

- [ ] **Step 3: Implement the extractor** — `bridge/replay.ts`

```ts
import fs from "node:fs";
import zlib from "node:zlib";
import { Player, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { GameRecord, StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { decompressGameRecord, toWireGameStartInfo } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { Act, TROOP_BINS } from "./actions";
import { EnvGame, STEP_TICKS } from "./game";

/** Label types of windows that stay out of the loss (PLAN Phase 2, b). */
export const OUT_OF_SET = -1; // a command the 5 actions cannot express, e.g. building a silo
export const UNMAPPED = -2; // a command of an expressible type whose label could not be built
/** Commands a seat may have queued; more are dropped and counted (humans rarely issue 2 in one window). */
export const MAX_PENDING = 3;

/** A record from a .json or .json.gz file, with every turn present (records store only non-empty turns). */
export function readRecord(file: string): GameRecord {
  const raw = fs.readFileSync(file);
  const text = (file.endsWith(".gz") ? zlib.gunzipSync(raw) : raw).toString("utf8");
  return decompressGameRecord(JSON.parse(text) as GameRecord);
}

/** The troop bin nearest, in log scale, to the share of its troops a player sent. */
export function binOf(troops: number, have: number): number {
  const share = Math.min(Math.max(troops / Math.max(have, 1), TROOP_BINS[0]), 1);
  let best = 0;
  for (let i = 1; i < TROOP_BINS.length; i++) {
    if (Math.abs(Math.log(share / TROOP_BINS[i])) < Math.abs(Math.log(share / TROOP_BINS[best]))) best = i;
  }
  return best;
}

/** A human command as the [type, cell, target, bin] action the agent would take for it, from the current state. */
export function labelOf(env: EnvGame, seat: Player, intent: StampedIntent): number[] {
  const { game, frame } = env;
  switch (intent.type) {
    case "spawn":
      return [Act.Spawn, frame.cellOf(intent.tile), 0, 0];
    case "attack": {
      if (intent.troops === null) return [UNMAPPED, 0, 0, 0];
      if (intent.targetID !== null && !game.hasPlayer(intent.targetID)) return [UNMAPPED, 0, 0, 0];
      const target = intent.targetID === null ? 0 : game.player(intent.targetID).smallID();
      return [Act.Attack, 0, target, binOf(intent.troops, seat.troops())];
    }
    case "boat":
      return [Act.Boat, frame.cellOf(intent.dst), 0, binOf(intent.troops, seat.troops())];
    case "build_unit":
      return intent.unit === UnitType.City ? [Act.BuildCity, frame.cellOf(intent.tile), 0, 0] : [OUT_OF_SET, 0, 0, 0];
    default:
      return [OUT_OF_SET, 0, 0, 0];
  }
}

/**
 * Replays a game record window by window (STEP_TICKS turns) for behaviour cloning. The seats are the record's players
 * and the observations come from the same EnvGame code as training games. A seat's label for a window is the oldest of
 * its queued commands; the commands of the turns about to execute are queued first, so a command labels the
 * observation its player saw just before it executed.
 */
export class RecordGame {
  window = 0;
  done = false;
  /** A recorded hash differed from the replay's: the rest of the record is unusable. */
  desync = false;
  /** Commands dropped because the seat already had MAX_PENDING queued. */
  dropped = 0;
  private readonly pending: StampedIntent[][];
  private readonly seatOf: Map<string, number>;

  private constructor(
    readonly env: EnvGame,
    readonly record: GameRecord,
    /** Per seat: +1 for the winner, -1 otherwise. */
    readonly outcome: number[],
  ) {
    this.pending = env.seats.map(() => []);
    this.seatOf = new Map(env.start.players.map((p, i) => [p.clientID, i]));
  }

  static async create(record: GameRecord, stride: number, slots: number): Promise<RecordGame> {
    const info = record.info;
    const start = toWireGameStartInfo({
      gameID: info.gameID,
      lobbyCreatedAt: info.lobbyCreatedAt,
      config: info.config,
      players: info.players,
      tribes: info.tribes,
    });
    const env = await EnvGame.fromStart(start, stride, slots, info.gameID);
    const winner = info.winner?.[0] === "player" ? info.winner[1] : null;
    return new RecordGame(env, record, info.players.map((p) => (p.clientID === winner ? 1 : -1)));
  }

  /** Windows until the record ends. */
  get left(): number {
    return Math.max(0, Math.ceil(this.record.turns.length / STEP_TICKS) - this.window);
  }

  /** One label row per seat for the current observation. Call it once per window, before advance(). */
  labels(): number[][] {
    const t0 = this.window * STEP_TICKS;
    for (const turn of this.record.turns.slice(t0, t0 + STEP_TICKS)) {
      for (const intent of turn.intents) {
        const s = this.seatOf.get(intent.clientID);
        if (s === undefined || intent.type === "mark_disconnected") continue;
        if (this.pending[s].length >= MAX_PENDING) this.dropped++;
        else this.pending[s].push(intent);
      }
    }
    return this.env.seats.map((seat, s) => {
      const intent = this.pending[s].shift();
      return intent === undefined ? [Act.Noop, 0, 0, 0] : labelOf(this.env, seat, intent);
    });
  }

  /** Executes the window's turns and checks every recorded hash. */
  advance(): void {
    const t0 = this.window * STEP_TICKS;
    for (const turn of this.record.turns.slice(t0, t0 + STEP_TICKS)) {
      this.env.tickWith(turn.intents);
      if (turn.hash != null && this.env.hashTick === turn.turnNumber && this.env.hash !== turn.hash) {
        this.desync = true;
        break;
      }
    }
    this.window++;
    this.done = this.desync || this.window * STEP_TICKS >= this.record.turns.length;
  }
}
```

- [ ] **Step 4: Replay mode in the worker** — `bridge/worker.ts`

Replace the header comment's `In:` lines and add one paragraph:

```ts
 * In:  u32 length + JSON: {cmd:"init", config, games, seed, worker, episodes?} starts K training games;
 *      {cmd:"init", config, replay:{records}, games, seed, worker} replays game records instead (behaviour cloning:
 *      every reply carries the seats' observations and labels, bridge/replay.ts); {cmd:"step", actions?, full?};
 *      {cmd:"close"}. "init" may be sent again to restart.
 ...
 * Replay mode cycles through the worker's records in a seeded order; the first record of each slot starts at a random
 * window, so the slots sit at different game phases.
```

Add the imports:

```ts
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { readRecord, RecordGame } from "./replay";
```

Add `replay?: { records: string[] };` to `Init`, and make the step actions optional: `actions?: number[][][];`. After
`seedOf`, add the replay state and helpers:

```ts
let records: RecordGame[] = [];
let order: number[] = [];
let next = 0;
let pick = new PseudoRandom(0);

/** The next record of this worker's list that loads; `skip`: start it at a random window. */
async function nextRecord(skip: boolean): Promise<RecordGame> {
  const files = init!.replay!.records;
  for (let tries = 0; tries < files.length; tries++) {
    if (next >= order.length) {
      order = pick.shuffleArray(order);
      next = 0;
    }
    const file = files[order[next++]];
    try {
      const rg = await RecordGame.create(readRecord(file), gridFor(init!.config)[1], slotsFor(init!.config));
      if (rg.env.seats.length !== init!.config.seats) throw new Error(`${rg.env.seats.length} players`);
      const skipped = skip ? pick.nextInt(0, Math.max(1, Math.floor(rg.left / 2))) : 0;
      for (let i = 0; i < skipped && !rg.done; i++) rg.advance();
      if (!rg.done) return rg;
    } catch (err) {
      console.error(`skipping ${file}: ${err}`);
    }
  }
  throw new Error("none of this worker's records can be replayed");
}

function replayPart(rg: RecordGame, reset: boolean, extra: Record<string, unknown>): Part {
  const labels = new Int32Array(rg.labels().flat());
  const blobs: Blob[] = [
    ...(reset ? [["terrain", rg.env.terrain()] as Blob] : []),
    ...rg.env.observe(reset),
    ["labels", new Uint8Array(labels.buffer)],
  ];
  const header = {
    tick: rg.env.game.ticks(),
    episode: 0,
    done: false,
    reset,
    hash: rg.env.hash,
    invalid: rg.env.seats.map(() => false),
    record: rg.record.info.gameID,
    seats: rg.env.start.players.map((p) => p.clientID),
    outcome: rg.outcome,
    left: rg.left,
    ...(reset ? { cellSize: rg.env.frame.cs } : {}),
    ...extra,
  };
  return { header, blobs };
}

async function replayStep(): Promise<Part[]> {
  const parts: Part[] = [];
  for (let g = 0; g < records.length; g++) {
    const rg = records[g];
    let error: string | undefined;
    try {
      rg.advance();
    } catch (err) {
      error = String(err);
      console.error(`${rg.record.info.gameID}: ${error}`);
      rg.done = true;
    }
    if (!rg.done) {
      parts.push(replayPart(rg, false, {}));
      continue;
    }
    const end = { done: true, finished: rg.record.info.gameID, desync: rg.desync, ...(error ? { error } : {}) };
    records[g] = await nextRecord(false);
    parts.push(replayPart(records[g], true, end));
  }
  return parts;
}
```

In `handle`, start the init branch with the replay case, and route replay steps before the training-game step:

```ts
  if (msg.cmd === "init") {
    init = msg;
    if (msg.replay !== undefined) {
      if (msg.replay.records.length === 0) throw new Error("replay mode needs records");
      pick = new PseudoRandom(simpleHash(`${msg.seed}-w${msg.worker}`));
      order = pick.shuffleArray(msg.replay.records.map((_, i) => i));
      next = 0;
      records = [];
      for (let g = 0; g < msg.games; g++) records.push(await nextRecord(true));
      return encode(records.map((rg) => replayPart(rg, true, {})), dims(msg.config));
    }
    // ... the training-game init, unchanged
  }
  if (init === null) throw new Error("step before init");
  if (init.replay !== undefined) return encode(await replayStep(), null);
  // ... the training-game step, unchanged except env.step(msg.actions?.[g] ?? [])
```

- [ ] **Step 5: Replay mode in the trainer** — `trainer/env.py`

Module docstring:

```python
"""Vectorised OpenFront games for training.

Node workers (bridge/worker.ts, protocol described there) run K games each and send per-cell frame deltas; the
observation tensors are assembled on the GPU from the state kept here. In replay mode the workers replay game records
instead, and every step also brings each seat's label (the command its human player issued) for behaviour cloning.
"""
```

`VecEnv.__init__` gains `records=None` and sends replay or training inits:

```python
    def __init__(self, config, workers, games_per_worker, seed, device="cuda", episodes=None, log_dir=None,
                 full_frames=False, records=None):
        self.device = torch.device(device)
        self.G = games_per_worker
        self.E = workers * games_per_worker
        self.full_frames = full_frames
        self.replay = records is not None
        if self.replay and len(records) < workers:
            raise ValueError(f"{len(records)} records cannot feed {workers} workers")
        self.episodes = list(episodes) if episodes is not None else [0] * self.E
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.workers = [Worker(w, Path(log_dir) / f"worker{w}.log" if log_dir is not None else None)
                        for w in range(workers)]
        for w, wk in enumerate(self.workers):
            msg = {"cmd": "init", "config": config, "games": self.G, "seed": seed, "worker": w}
            if self.replay:
                msg["replay"] = {"records": [str(r) for r in records[w::workers]]}
            else:
                msg["episodes"] = self.episodes[w * self.G:(w + 1) * self.G]
            wk.send(msg)
```

After `self.fo = torch.zeros_like(self.n1)`, and before `self.obs = self._absorb(replies)[0]`:

```python
        # Replay mode: per game its record, players, their outcome (+1 winner, -1 otherwise) and windows left; per row
        # the label [type, cell, target, bin] (type -1: a command outside the action set, -2: unmapped).
        self.labels = None
        self.outcome = np.zeros((self.E, self.S), np.float32)
        self.left = np.zeros(self.E, np.int64)
        self.record_of = [""] * self.E
        self.seats_of = [[] for _ in range(self.E)]
```

Replace `step` with `send`, `recv` and `step`:

```python
    def send(self, actions=None):
        """Starts every worker on its next step (replay mode takes no actions); recv() collects it, so the caller can
        use the GPU meanwhile. actions: int array [N, 4] of [type, cell, target, bin] with grid cell indexes."""
        if self.replay:
            for wk in self.workers:
                wk.send({"cmd": "step"})
            return
        a = np.asarray(actions, dtype=np.int64).reshape(self.E, self.S, 4)
        for w, wk in enumerate(self.workers):
            wk.send({"cmd": "step", "actions": a[w * self.G:(w + 1) * self.G].tolist(), "full": self.full_frames})

    def recv(self):
        out = self._absorb([wk.recv() for wk in self.workers])
        self.obs = out[0]
        return out

    def step(self, actions=None):
        self.send(actions)
        return self.recv()
```

In `_absorb`, allocate `labels = np.zeros((E, S, 4), np.int64)` next to the other arrays, and replace the
`if g.get("done"):` block with:

```python
                if g.get("done") and self.replay:
                    infos.append({"env": e, "record": g["finished"], "desync": g.get("desync", False),
                                  "error": g.get("error")})
                elif g.get("done"):
                    dones[e] = 1
                    rewards[e] = g["rewards"]
                    infos += [{"env": e, "seat": s, "reward": g["rewards"][s], "rank": g["ranks"][s],
                               "players": g["players"], "ticks": g["endTick"]} for s in range(S)]
                if self.replay:
                    labels[e] = np.frombuffer(b["labels"], np.int32).reshape(S, 4)
                    self.outcome[e] = g["outcome"]
                    self.left[e] = g["left"]
                    self.record_of[e] = g["record"]
                    self.seats_of[e] = g["seats"]
```

and before `self._apply_cells(resets, cells)`:

```python
        if self.replay:
            self.labels = torch.from_numpy(labels.reshape(-1, 4)).to(self.device)
```

- [ ] **Step 6: Run all tests and the typecheck**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test --test-timeout=300000 "../../bridge/*.test.ts" && node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: all pass (the replay tests take about 2–10 s each on the fixture); tsc exit 0.
Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK.

- [ ] **Step 7: Check the extraction on real data**

Once `data/records/` holds records (the crawl is running or done), replay 20 of them with a scratch script. Put it at
`bridge/check_replays.ts` and don't commit it.

```ts
// Run from the game dir: node --import tsx ../../bridge/check_replays.ts
import fs from "node:fs";
import { readRecord, RecordGame } from "./replay";

const dir = "../../data/records";
let desync = 0;
let windows = 0;
const t0 = performance.now();
const types = new Map<number, number>();
for (const f of fs.readdirSync(dir).slice(0, 20)) {
  const rg = await RecordGame.create(readRecord(`${dir}/${f}`), 256, 403);
  while (!rg.done) {
    for (const [t] of rg.labels()) types.set(t, (types.get(t) ?? 0) + 1);
    rg.env.observe(false);
    rg.advance();
    windows++;
  }
  if (rg.desync) desync++;
}
console.error({ desync, windows, msPerWindow: (performance.now() - t0) / windows, types: Object.fromEntries(types) });
```

Expected: `desync: 0`, since records from v0.34.17 on share the pinned simulation, and a few ms per window. Record the
desync rate, ms per window and label mix in the task report; they size the BC run in Task 9. Delete the script.

- [ ] **Step 8: Commit**

```bash
git add bridge/replay.ts bridge/replay.test.ts bridge/worker.ts bridge/worker.test.ts trainer/env.py trainer/test_env.py
git commit -m "Replay extractor: labelled observations from game records, streamed by the workers"
```

---

### Task 5: Behaviour cloning trainer

**Files:**
- Create: `trainer/bc.py`, `trainer/test_bc.py`

**Interfaces:**
- Consumes: the `VecEnv` replay mode (Task 4), `crawl.load_index/record_path/elo` (Task 3), `Policy`, `ActionDist`
  (Phase 1), `Control`/`serve`, `save_checkpoint`, `game_commit` (Phase 1).
- Produces:
  - `is_val(game_id, share)`, `label_legal(obs, labels)`, `value_targets(env, gamma, device)`
  - `bc_loss(model, obs, labels, legal, target, vf_coef) -> (loss, dist, logp, value)`
  - `Meter`, `measure(...)`, `main(argv)`
  - Checkpoint keys as in PPO, plus `kind="bc"`. `args.widths` lets `trainer.eval` and `trainer.ppo --init` load it.

- [ ] **Step 1: Write the failing test** — `trainer/test_bc.py`

```python
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from trainer import crawl
from trainer.bc import Meter, bc_loss, is_val, label_legal, main
from trainer.env import RANKED_1V1, ROOT, VecEnv
from trainer.model import Policy

FIXTURE = ROOT / "bridge" / "testdata" / "ranked1v1.json.gz"


class LabelTest(unittest.TestCase):
    def test_label_legal(self):
        obs = {"type_mask": torch.ones(4, 5, dtype=torch.bool),
               "cell_mask": torch.zeros(4, 3, 16, dtype=torch.bool),
               "attack_mask": torch.zeros(4, 5, dtype=torch.bool)}
        obs["cell_mask"][3, 0, 5] = True
        obs["attack_mask"][:, 2] = True
        labels = torch.tensor([[-1, 0, 0, 0], [2, 0, 2, 1], [2, 0, 3, 1], [1, 5, 0, 0]])
        self.assertEqual(label_legal(obs, labels).tolist(), [False, True, False, True])
        obs["type_mask"][1, 2] = False
        self.assertFalse(bool(label_legal(obs, labels)[1]), "a type the mask forbids")

    def test_validation_split_is_stable(self):
        ids = [f"g{i:04d}" for i in range(2000)]
        val = [i for i in ids if is_val(i, 0.05)]
        self.assertTrue(60 < len(val) < 140)
        self.assertEqual(val, [i for i in ids if is_val(i, 0.05)])

    def test_meter_averages_and_resets(self):
        m = Meter()
        m.add("a", torch.tensor(3.0), 2)
        m.add("a", torch.tensor(1.0), 2)
        self.assertEqual(m.means(), {"a": 1.0})
        self.assertEqual(m.means(), {})


class CloneTest(unittest.TestCase):
    def test_overfitting_a_few_windows_lowers_their_loss(self):
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=1, seed="bc", records=[FIXTURE])
        try:
            batches = []
            for _ in range(40):
                batches.append(({k: v.clone() for k, v in env.obs.items()}, env.labels.clone()))
                env.step()
            dims = env.dims
        finally:
            env.close()
        obs = {k: torch.cat([b[0][k] for b in batches]) for k in batches[0][0]}
        labels = torch.cat([b[1] for b in batches])
        legal = label_legal(obs, labels)
        self.assertGreater(int(legal.sum()), 40)
        torch.manual_seed(0)
        model = Policy(dims).cuda().to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        target = torch.zeros(len(labels), device="cuda")
        losses = []
        for _ in range(40):
            loss, *_ = bc_loss(model, obs, labels, legal, target, vf_coef=0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] - 0.3)  # at least the no-op prior is learnt


class MainTest(unittest.TestCase):
    RUN = "unittest-bc"

    def tearDown(self):
        for d in ("checkpoints", "runs"):
            shutil.rmtree(ROOT / d / self.RUN, ignore_errors=True)

    def test_train_save_and_resume(self):
        data = Path(tempfile.mkdtemp())
        (data / "records").mkdir()
        shutil.copy(FIXTURE, data / "records" / "fixture.json.gz")
        entry = {"id": "fixture", "start": 0, "commit": None, "map": "x", "size": "Compact", "turns": 1,
                 "players": [{"clientID": "a", "publicID": None}, {"clientID": "b", "publicID": None}], "winner": "a"}
        (data / "index.jsonl").write_text(json.dumps(entry) + "\n")
        common = ["--run", self.RUN, "--workers", "1", "--games-per-worker", "1", "--val-workers", "0",
                  "--val-share", "0", "--port", "0", "--log-every", "1", "--save-every", "2"]
        with mock.patch.multiple(crawl, DATA=data, RECORDS=data / "records", INDEX=data / "index.jsonl"):
            main(common + ["--updates", "3"])
            first = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
            self.assertEqual((first["kind"], first["update"]), ("bc", 3))
            main(common + ["--updates", "4", "--resume"])
        self.assertEqual(torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)["update"], 4)
        self.assertTrue((ROOT / "runs" / self.RUN).is_dir())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.bc'`.

- [ ] **Step 3: Implement** — `trainer/bc.py`

```python
"""Behaviour cloning on ranked 1v1 replays (PLAN Phase 2): the policy learns to issue the command a human issued in
the same situation, and the value head learns who won.

  Train:   .venv/Scripts/python -m trainer.bc --run bc          (needs data/ from trainer.crawl)
  Resume:  .venv/Scripts/python -m trainer.bc --run bc --resume
  Control: http://127.0.0.1:8787 or python -m trainer.ctl pause|resume|save|stop|throttle|status
  Charts:  .venv/Scripts/tensorboard --logdir runs
"""
import argparse
import random
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trainer import crawl
from trainer.control import Control, serve
from trainer.env import RANKED_1V1, ROOT, VecEnv
from trainer.model import ATTACK, BOAT, BUILD, SPAWN, ActionDist, Policy
from trainer.ppo import game_commit, save_checkpoint

KEEP_ON_RESUME = ("resume", "port", "duty", "gpu_temp_limit", "updates")
CELL_HEAD = {SPAWN: 0, BOAT: 1, BUILD: 2}  # the heat map each type's cell comes from (trainer/model.py)
PHASES = ("spawn", "early", "mid", "late")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="bc", help="name for checkpoints/<run> and runs/<run>")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/latest.pt")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--games-per-worker", type=int, default=8)
    p.add_argument("--val-workers", type=int, default=2, help="0 = no validation")
    p.add_argument("--val-share", type=float, default=0.05, help="share of the games held out")
    p.add_argument("--val-every", type=int, default=500, help="updates between validations")
    p.add_argument("--val-steps", type=int, default=100, help="replay steps per validation")
    p.add_argument("--updates", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.999, help="discount of the value target, as in PPO")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--min-elo", type=float, default=0.0, help="leave players rated below this out of the loss")
    p.add_argument("--widths", default="64,128,192", help="CNN stage widths")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000, help="updates between checkpoints")
    p.add_argument("--port", type=int, default=8787, help="control page port (0 = any free port)")
    p.add_argument("--duty", type=float, default=1.0, help="start throttled: fraction of time spent working")
    p.add_argument("--gpu-temp-limit", type=int, default=0, help="auto-pause at this GPU temperature in C (0 = off)")
    return p.parse_args(argv)


def is_val(game_id, share):
    """Held-out games, picked by a stable hash of the game ID."""
    return zlib.crc32(game_id.encode()) % 10_000 < share * 10_000


def label_legal(obs, labels):
    """Rows whose label can enter the loss: an action type (not a command outside the set) that the observation's
    masks allow, with a legal cell or target."""
    t, cell, target = labels[:, 0], labels[:, 1], labels[:, 2]
    rows = torch.arange(len(t), device=t.device)
    ok = (t >= 0) & obs["type_mask"][rows, t.clamp(min=0)]
    head = torch.full_like(t, -1)
    for typ, h in CELL_HEAD.items():
        head[t == typ] = h
    cell_ok = obs["cell_mask"][rows, head.clamp(min=0), cell.clamp(0, obs["cell_mask"].shape[-1] - 1)]
    ok &= torch.where(head >= 0, cell_ok, True)
    target_ok = obs["attack_mask"][rows, target.clamp(0, obs["attack_mask"].shape[-1] - 1)]
    return ok & torch.where(t == ATTACK, target_ok, True)


def value_targets(env, gamma, device):
    """Per row, the discounted outcome: +-1 x gamma^(windows left), the return PPO's value head predicts."""
    left = torch.from_numpy(np.repeat(env.left, env.S)).to(device).float()
    return torch.from_numpy(env.outcome.reshape(-1).copy()).to(device) * gamma ** left


def bc_loss(model, obs, labels, legal, target, vf_coef):
    """-log pi(label) over the rows that can enter the loss, plus the value loss."""
    with torch.autocast("cuda", torch.bfloat16):
        out = model(obs)
    dist = ActionDist(out, obs)
    logp = dist.log_prob(torch.where(legal[:, None], labels, torch.zeros_like(labels)))  # other rows: a no-op
    nll = -(logp * legal).sum() / legal.sum().clamp(min=1)
    value = out["value"].float()
    return nll + vf_coef * 0.5 * (value - target).pow(2).mean(), dist, logp, value


class Meter:
    """Running sums kept on the GPU (no sync per step); means() reads and resets them."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, name, total, count):
        self.sums[name] = self.sums.get(name, 0) + total
        self.counts[name] = self.counts.get(name, 0) + count

    def means(self, prefix=""):
        out = {prefix + k: float(s) / float(self.counts[k]) for k, s in self.sums.items() if float(self.counts[k]) > 0}
        self.sums, self.counts = {}, {}
        return out


@torch.no_grad()
def measure(m, obs, labels, legal, dist, logp, value, target):
    """Label statistics, per-head accuracy and NLL per game phase of one batch."""
    t, cell, tgt, b = labels.unbind(1)
    n, nl = len(t), legal.sum()
    m.add("labels/out_of_set", (t == -1).sum(), n)
    m.add("labels/unmapped", (t == -2).sum(), n)
    m.add("labels/illegal", ((t >= 0) & ~legal).sum(), n)
    m.add("labels/noop", ((t == 0) & legal).sum(), nl)
    m.add("bc/nll", -(logp * legal).sum(), nl)
    m.add("bc/type_acc", ((dist.type.logits.argmax(1) == t) & legal).sum(), nl)
    m.add("bc/act_rate_human", ((t > 0) & legal).sum(), nl)
    m.add("bc/act_rate_policy", ((1 - dist.type.probs[:, 0]) * legal).sum(), nl)
    rows = legal & (t == ATTACK)
    m.add("bc/attack_target_acc", ((dist.ptr.logits.argmax(1) == tgt) & rows).sum(), rows.sum())
    m.add("bc/attack_bin_acc", ((dist.bins.logits[:, 0].argmax(1) == b) & rows).sum(), rows.sum())
    rows = legal & (t == BOAT)
    m.add("bc/boat_bin_acc", ((dist.bins.logits[:, 1].argmax(1) == b) & rows).sum(), rows.sum())
    for typ, name in ((SPAWN, "spawn"), (BOAT, "boat"), (BUILD, "city")):
        rows = legal & (t == typ)
        hit = dist.cells.logits[:, CELL_HEAD[typ]].argmax(1) == cell
        m.add(f"bc/{name}_cell_acc", (hit & rows).sum(), rows.sum())
    phase = torch.where(obs["scalars"][:, 1] > 0, 0, 1 + (obs["scalars"][:, 0] * 3).long().clamp(max=2))
    for i, name in enumerate(PHASES):
        rows = legal & (phase == i)
        m.add(f"bc/nll_{name}", -(logp * rows).sum(), rows.sum())
    m.add("bc/value_mse", (value - target).pow(2).sum(), n)


def main(argv=None):
    a = parse_args(argv)
    ckpt_dir = ROOT / "checkpoints" / a.run
    state = None
    if a.resume:
        state = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        a = argparse.Namespace(**{**state["args"], **{k: getattr(a, k) for k in KEEP_ON_RESUME}})
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run_dir = ROOT / "runs" / a.run

    entries = [e for e in crawl.load_index() if crawl.record_path(e["id"]).exists()]
    ratings = crawl.elo(entries)
    train = [crawl.record_path(e["id"]) for e in entries if not is_val(e["id"], a.val_share)]
    val = [crawl.record_path(e["id"]) for e in entries if is_val(e["id"], a.val_share)]
    if len(train) < a.workers or len(val) < a.val_workers:
        raise SystemExit(f"{len(train)} training and {len(val)} held-out records: run trainer.crawl first")
    envs = VecEnv(RANKED_1V1, a.workers, a.games_per_worker, seed=f"{a.run}-{a.seed}", device=device, records=train,
                  log_dir=run_dir / "workers")
    val_envs = None
    if a.val_workers:
        val_envs = VecEnv(RANKED_1V1, a.val_workers, a.games_per_worker, seed="val", device=device, records=val,
                          log_dir=run_dir / "val_workers")
    model = Policy(envs.dims, tuple(int(w) for w in a.widths.split(","))).to(device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    update = global_step = 0
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        update, global_step = state["update"], state["global_step"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    control = Control(duty=a.duty, gpu_temp_limit=a.gpu_temp_limit)
    server = serve(control, a.port)
    control.note(f"control page on http://127.0.0.1:{server.server_address[1]}")
    writer = SummaryWriter(run_dir)
    commit = game_commit()
    meter, done = Meter(), {"records": 0, "desync": 0, "errors": 0}

    def elo_rows(env):
        return torch.tensor([ratings.get(rec, {}).get(c, 1500.0) for rec, seats in zip(env.record_of, env.seats_of)
                             for c in seats], device=device)

    def checkpoint():
        s = dict(kind="bc", model=model.state_dict(), optimizer=opt.state_dict(), args=vars(a), update=update,
                 global_step=global_step, game_commit=commit,
                 rng={"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all()})
        save_checkpoint(ckpt_dir / "latest.pt", **s)
        if update % (10 * a.save_every) == 0 and update > 0:
            save_checkpoint(ckpt_dir / f"update{update:07d}.pt", **s)
        control.note(f"saved checkpoint at update {update}")

    def validate():
        m, values, outcomes = Meter(), [], []
        with torch.no_grad():
            for _ in range(a.val_steps):
                obs, labels = val_envs.obs, val_envs.labels
                target = value_targets(val_envs, a.gamma, device)
                legal = label_legal(obs, labels)
                _, dist, logp, value = bc_loss(model, obs, labels, legal, target, a.vf_coef)
                measure(m, obs, labels, legal, dist, logp, value, target)
                values.append(value.cpu().numpy())
                outcomes.append(val_envs.outcome.reshape(-1).copy())
                val_envs.step()
        out = m.means("val/")
        v, o = np.concatenate(values), np.concatenate(outcomes)
        if v.std() > 0 and o.std() > 0:
            out["val/value_outcome_corr"] = float(np.corrcoef(v, o)[0, 1])
        return out

    control.set_state("running")
    t_log = time.perf_counter()
    try:
        while update < a.updates:
            if not control.wait():
                break
            t0 = time.perf_counter()
            obs, labels = envs.obs, envs.labels
            target = value_targets(envs, a.gamma, device)
            legal = label_legal(obs, labels) & (elo_rows(envs) >= a.min_elo)
            envs.send()  # the workers replay the next window while the GPU trains on this one
            loss, dist, logp, value = bc_loss(model, obs, labels, legal, target, a.vf_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
            opt.step()
            measure(meter, obs, labels, legal, dist, logp, value, target)
            infos = envs.recv()[3]
            for i in infos:
                done["records"] += 1
                done["desync"] += int(i["desync"])
                done["errors"] += int(i["error"] is not None)
            update += 1
            global_step += envs.N
            control.throttle(time.perf_counter() - t0)
            if update % a.log_every == 0:
                metrics = meter.means()
                metrics.update({"data/records_done": done["records"], "data/errors": done["errors"],
                                "data/desync_share": done["desync"] / max(done["records"], 1),
                                "perf/samples_per_s": a.log_every * envs.N / (time.perf_counter() - t_log)})
                t_log = time.perf_counter()
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
                control.update(update=update, global_step=global_step, metrics=metrics)
            if val_envs is not None and update % a.val_every == 0:
                for k, v in validate().items():
                    writer.add_scalar(k, v, global_step)
            if update % a.save_every == 0 or control.take_save():
                checkpoint()
    except KeyboardInterrupt:
        control.note("interrupted")
    finally:
        control.set_state("saving")
        checkpoint()
        envs.close()
        if val_envs is not None:
            val_envs.close()
        writer.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v`
Expected: PASS (5 tests).
Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK.

- [ ] **Step 5: Profile a short run on the crawled data** (the crawl need not be finished):
  `.venv/Scripts/python -m trainer.bc --run bc-probe --updates 300 --val-every 150 --val-steps 20`, then read
  `perf/samples_per_s`, `labels/*` and the GPU utilisation on the control page. Expected: over 1k samples/s; a
  `labels/illegal` share below 10%; `labels/out_of_set` around 5%. Record them in the task report.
  - If `labels/illegal` is higher, report which types fail before going on (the approximate masks may need a fix).
  - If throughput is far lower, profile before the long run (Task 9).
  - Delete `checkpoints/bc-probe` and `runs/bc-probe` afterwards.

- [ ] **Step 6: Commit**

```bash
git add trainer/bc.py trainer/test_bc.py
git commit -m "Behaviour cloning trainer on ranked 1v1 replays"
```

---

### Task 6: Matches between policies and the scripted expander

**Files:**
- Modify: `trainer/eval.py`, `trainer/test_eval.py`, `trainer/model.py` (`load_net`)

**Interfaces:**
- Consumes: `ONION`, `RANKED_1V1` (Task 2).
- Produces:
  - `load_net(path, dims, device)` in `trainer/model.py`: a frozen policy from a checkpoint.
  - `Scripted(bin, every, seed)`
  - `load_policy(spec, dims, device)`
  - `seats_for(E, S)`
  - `play(specs, config, episodes, workers, games_per_worker, device, seed) -> summary of policy 0`
  - CLI `--a`, `--b`, `--setup`.

- [ ] **Step 1: Write the failing test** — replace `trainer/test_eval.py`

```python
import unittest

from trainer.env import ONION
from trainer.eval import play, seats_for, summarize

ONION2 = {**ONION, "seats": 2, "nations": 0, "timerMinutes": 1}


class EvalTest(unittest.TestCase):
    def test_summarize(self):
        s = summarize([{"reward": 1, "rank": 1}, {"reward": -1, "rank": 4}])
        self.assertEqual(s["episodes"], 2)
        self.assertEqual(s["win_rate"], 0.5)
        self.assertEqual(s["mean_reward"], 0.0)
        self.assertEqual(s["mean_rank"], 2.5)

    def test_policies_swap_seats_between_slots(self):
        self.assertEqual(seats_for(4, 2), [[0, 1], [1, 0], [0, 1], [1, 0]])
        self.assertEqual(seats_for(2, 1), [[0], [0]])

    def test_one_seat_on_fixed_seeds_is_reproducible(self):
        run = lambda: play(["random"], {**ONION, "timerMinutes": 1}, episodes=2, workers=1, games_per_worker=2,
                           seed="eval-test")
        first, second = run(), run()
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(first, second)

    def test_scripted_against_random(self):
        r = play(["scripted", "random"], ONION2, episodes=4, workers=1, games_per_worker=4, seed="eval-2")
        self.assertEqual(r["episodes"], 4)
        self.assertTrue(0.0 <= r["win_rate"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_eval -v`
Expected: FAIL, `ImportError: cannot import name 'play'`.

- [ ] **Step 3: Implement**

Append to `trainer/model.py`:

```python
def load_net(path, dims, device="cuda"):
    """A frozen policy from a trainer.bc or trainer.ppo checkpoint, in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = Policy(dims, tuple(int(w) for w in ckpt["args"]["widths"].split(",")))
    net.load_state_dict(ckpt["model"])
    return net.to(device).to(memory_format=torch.channels_last).eval().requires_grad_(False)
```

Replace `trainer/eval.py`:

```python
"""Plays policies against each other on fixed seeds (the same games every time, so results compare directly).

  .venv/Scripts/python -m trainer.eval --a checkpoints/bc/latest.pt --b scripted      (ranked 1v1, 256 games)
  .venv/Scripts/python -m trainer.eval --a checkpoints/ppo/latest.pt --b checkpoints/bc/latest.pt
  .venv/Scripts/python -m trainer.eval --setup onion --a random                        (Phase 1: one seat vs 3 nations)

A policy is "random", "scripted" or a checkpoint path. The results are the first policy's; in 2-seat setups the two
policies swap seats from one game slot to the next.
"""
import argparse
import json

import numpy as np
import torch

from trainer.env import ONION, RANKED_1V1, VecEnv
from trainer.model import ActionDist, RandomPolicy, load_net

SETUPS = {"onion": ONION, "ranked1v1": RANKED_1V1}


def summarize(results):
    n = len(results)
    reward = np.array([r["reward"] for r in results], float)
    win = np.array([r["rank"] == 1 for r in results], float)
    rank = np.array([r["rank"] for r in results], float)
    se = lambda v: float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"episodes": n, "win_rate": float(win.mean()), "win_rate_se": se(win), "mean_reward": float(reward.mean()),
            "mean_reward_se": se(reward), "mean_rank": float(rank.mean())}


class Scripted:
    """The scripted expander (PLAN Phase 1 diagnosis): spawns on a random legal cell at the first decision (and when
    forced), then every `every` decisions attacks unowned land, or else a random attackable bot, with troop bin
    `bin`; otherwise waits."""

    def __init__(self, bin=1, every=4, seed=0):
        self.bin, self.every, self.rng, self.step = bin, every, np.random.default_rng(seed), 0

    def __call__(self, obs):
        types, attack, cells = (obs[k].cpu().numpy() for k in ("type_mask", "attack_mask", "cell_mask"))
        bots = obs["players"][..., 5].cpu().numpy() > 0.5  # player feature 5: bot (bridge/obs.ts)
        rows = []
        for i in range(types.shape[0]):
            act = [0, 0, 0, 0]
            if types[i, 1] and (self.step == 0 or not types[i, 0]) and cells[i, 0].any():
                act = [1, int(self.rng.choice(np.flatnonzero(cells[i, 0]))), 0, 0]
            elif types[i, 2] and self.step % self.every == 0:
                targets = [0] if attack[i, 0] else np.flatnonzero(attack[i] & bots[i])
                if len(targets):
                    act = [2, 0, int(self.rng.choice(targets)), self.bin]
            rows.append(act)
        self.step += 1
        return torch.tensor(rows, dtype=torch.long, device=obs["type_mask"].device)


def load_policy(spec, dims, device="cuda"):
    """obs -> actions [n, 4] for "random", "scripted" or a checkpoint path (sampling, as in training)."""
    if spec == "scripted":
        return Scripted()
    model = RandomPolicy(dims).to(device) if spec == "random" else load_net(spec, dims, device)

    def act(obs):
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                out = model(obs)
            return ActionDist(out, obs).sample()

    return act


def seats_for(E, S):
    """seats[e][p]: the seat policy p plays in game slot e (the policies rotate through the seats)."""
    return [[(e + p) % S for p in range(S)] for e in range(E)]


def play(specs, config, episodes, workers, games_per_worker, device="cuda", seed="eval"):
    """Plays the first ceil(episodes / games) episodes of every game slot; returns the summary of policy 0. Sampling
    uses a fixed torch seed, so a given set of policies always gives the same result."""
    torch.manual_seed(0)
    envs = VecEnv(config, workers, games_per_worker, seed=seed, device=device)
    try:
        S, E = envs.S, envs.E
        if len(specs) != S:
            raise ValueError(f"{S} seats need {S} policies, got {len(specs)}")
        policies = [load_policy(s, envs.dims, device) for s in specs]
        seats = seats_for(E, S)
        rows = [torch.tensor([e * S + seats[e][p] for e in range(E)], device=device) for p in range(S)]
        per_slot = -(-episodes // E)
        counts, results, obs = np.zeros(E, int), [], envs.obs
        while counts.min() < per_slot:
            act = torch.zeros(envs.N, 4, dtype=torch.long, device=device)
            for p, policy in enumerate(policies):
                act[rows[p]] = policy({k: v[rows[p]] for k, v in obs.items()})
            obs, _, _, infos, _ = envs.step(act.cpu().numpy())
            for info in infos:
                e = info["env"]
                if info["seat"] == seats[e][0] and counts[e] < per_slot:
                    counts[e] += 1
                    results.append(info)
    finally:
        envs.close()
    return summarize(results)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="the scored policy: random, scripted or a checkpoint path")
    p.add_argument("--b", help="its opponent in 2-seat setups")
    p.add_argument("--setup", choices=sorted(SETUPS), default="ranked1v1")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--seed", default="eval")
    a = p.parse_args(argv)
    config = SETUPS[a.setup]
    if config["seats"] == 2 and a.b is None:
        p.error("--b is required in 2-seat setups")
    result = play([a.a, a.b][:config["seats"]], config, a.episodes, a.workers, a.games_per_worker, seed=a.seed)
    print(json.dumps({"a": a.a, "b": a.b, "setup": a.setup, **result}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_eval -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Measure the baselines in ranked 1v1** (about 10–20 minutes)

Run both, and record the results in the task report (the gate in Task 9 compares BC against these):
- `.venv/Scripts/python -m trainer.eval --a scripted --b random --episodes 64`
- `.venv/Scripts/python -m trainer.eval --a random --b random --episodes 64`

Expected:
- random against random: a win rate near 0.5, a sanity check of the seat swap;
- scripted against random: clearly above 0.5.

If the expander does not beat random in ranked 1v1, say so in the report: the BC gate then rests on beating random and
the per-head accuracies.

- [ ] **Step 6: Commit**

```bash
git add trainer/eval.py trainer/test_eval.py trainer/model.py
git commit -m "Matches between two policies on fixed seeds, with the scripted expander as a baseline"
```

---

### Task 7: PPO from the cloned policy against a frozen opponent

**Files:**
- Modify: `trainer/ppo.py`, `trainer/test_ppo.py`, `CLAUDE.md` (training commands)

**Interfaces:**
- Consumes: `load_net` (Task 6), `RANKED_1V1` (Task 2), BC checkpoints (Task 5).
- Produces:
  - New flags: `--setup onion|ranked1v1`, `--seats`, `--init`, `--opponent`, `--kl-coef`.
  - `--steps` and `--minibatches` default per setup (ranked: 64 and 8, so the rollout buffer stays at about 2k rows).
  - The learner plays seat 0 of every game and the frozen opponent the other seats. Only the learner's rows fill the
    buffer, the metrics and `global_step`.
  - With `--kl-coef`, the loss adds the k3 estimate of KL(π‖π_init) per minibatch (`loss/kl_ref`). Given with
    `--resume`, it replaces the run's value, so the penalty can be retuned without losing the run.

- [ ] **Step 1: Write the failing test** — add to `SmokeTest` in `trainer/test_ppo.py`
  (`from trainer.model import Policy`):

```python
    def test_two_seats_against_a_frozen_opponent_with_kl(self):
        init = ROOT / "checkpoints" / self.RUN / "init.pt"
        init.parent.mkdir(parents=True, exist_ok=True)
        dims = {"channels": 22, "playerFeatures": 17, "scalars": 16, "acts": 5, "troopBins": [0.1, 0.25, 0.5, 0.75, 1.0]}
        torch.save({"model": Policy(dims).state_dict(), "args": {"widths": "64,128,192"}}, init)
        main(["--run", self.RUN, "--workers", "1", "--games-per-worker", "2", "--steps", "8", "--minibatches", "2",
              "--epochs", "1", "--port", "0", "--timer", "1", "--nations", "0", "--seats", "2",
              "--init", str(init), "--opponent", str(init), "--kl-coef", "0.1", "--updates", "2"])
        state = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
        self.assertEqual(state["update"], 2)
        self.assertEqual(state["global_step"], 2 * 8 * 2, "only the learner's decisions count")
        self.assertEqual(state["args"]["kl_coef"], 0.1)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_ppo -v`
Expected: FAIL, `error: unrecognized arguments: --seats 2 --init ...`.

- [ ] **Step 3: Implement** — `trainer/ppo.py`

Imports: `from trainer.env import GAME_DIR, RANKED_1V1, ROOT, VecEnv` and
`from trainer.model import ACT_NAMES, ActionDist, Policy, load_net`.

In `parse_args`, change `--steps` and `--minibatches` to `default=None` (their help says "default: 64 / 8 for
ranked1v1, 128 / 4 otherwise"), and add:

```python
    p.add_argument("--setup", choices=("onion", "ranked1v1"), default="onion",
                   help="ranked1v1: the official ranked 1v1 playlist (the map, size, nation, bot and timer flags are ignored)")
    p.add_argument("--seats", type=int, default=1, help="agent seats in the onion setup")
    p.add_argument("--init", help="start from this checkpoint's weights, e.g. checkpoints/bc/latest.pt")
    p.add_argument("--opponent", help="frozen checkpoint that plays every seat but the learner's (2-seat setups)")
    p.add_argument("--kl-coef", type=float, default=None,
                   help="KL penalty toward the --init policy (default 0; given with --resume, it replaces the run's)")
```

`env_config`:

```python
def env_config(a):
    if a.setup == "ranked1v1":
        return RANKED_1V1
    return {"map": a.map, "compact": not a.normal_size, "nations": a.nations, "difficulty": a.difficulty,
            "bots": a.bots, "timerMinutes": a.timer, "seats": a.seats}
```

In `main`, resolve the resumed and per-setup arguments. Older checkpoints lack the new flags, and a `--kl-coef` given
with `--resume` retunes the penalty of a running experiment:

```python
    if a.resume:
        given = a
        state = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        a = argparse.Namespace(**{**vars(parse_args([])), **state["args"],
                                  **{k: getattr(given, k) for k in KEEP_ON_RESUME}})
        if given.kl_coef is not None:
            a.kl_coef = given.kl_coef
    a.kl_coef = a.kl_coef or 0.0
    ranked = a.setup == "ranked1v1"
    a.steps = a.steps or (64 if ranked else 128)
    a.minibatches = a.minibatches or (8 if ranked else 4)
```

After creating `envs` and `model`, load the initial weights, the opponent and the KL reference:

```python
    S = envs.S
    if S > 1 and not a.opponent:
        raise SystemExit("2-seat setups need --opponent")
    if a.kl_coef > 0 and not a.init:
        raise SystemExit("--kl-coef needs --init")
    if a.init and not state:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
    frozen = {}

    def net(path):
        if path not in frozen:
            frozen[path] = load_net(path, envs.dims, device)
        return frozen[path]

    opponent = net(a.opponent) if S > 1 else None
    ref = net(a.init) if a.kl_coef > 0 else None
    learner = torch.arange(envs.E, device=device) * S  # the learner plays seat 0 of every game
    others = [learner + s for s in range(1, S)]
    rows = lambda o, r: {k: v[r] for k, v in o.items()}
```

Size the buffer by the learner's rows (`T, N = a.steps, envs.E`) and build it from `rows(obs, learner)`. Add a buffer
for the reference log-probabilities:
`logps, values, rewards, dones, ref_logps = (torch.zeros(T, N, device=device) for _ in range(5))`.

Replace the body of `rollout`'s loop:

```python
            t0 = time.perf_counter()
            mine = rows(obs, learner)
            for k, v in mine.items():
                buf[k][step] = v
            full = torch.zeros(envs.N, 4, dtype=torch.long, device=device)
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(mine)
                dist = ActionDist(out, mine)
                act = dist.sample()
                actions[step], logps[step], values[step] = act, dist.log_prob(act), out["value"].float()
                full[learner] = act
                for r in others:
                    theirs = rows(obs, r)
                    with torch.autocast("cuda", torch.bfloat16):
                        out_o = opponent(theirs)
                    full[r] = ActionDist(out_o, theirs).sample()
                if ref is not None:
                    with torch.autocast("cuda", torch.bfloat16):
                        out_r = ref(mine)
                    ref_logps[step] = ActionDist(out_r, mine).log_prob(act)
            outcomes.record(mine["scalars"][:, 0].tolist(), values[step].tolist())
            obs, rew, done, infos, invalid = envs.step(full.cpu().numpy())
            rewards[step] = torch.from_numpy(rew.reshape(envs.E, S)[:, 0].copy()).to(device)
            dones[step] = torch.from_numpy(done.reshape(envs.E, S)[:, 0].copy()).to(device)
            outcomes.finish([i for i in infos if i["seat"] == 0], seats=1)
            np.add.at(act_counts, act.cpu().numpy()[:, 0], 1)
            invalid_count += int(invalid.reshape(envs.E, S)[:, 0].sum())
            global_step += N
            control.throttle(time.perf_counter() - t0)
```

In `learn`, bootstrap from the learner's rows (`model(rows(obs, learner))`), flatten `b_ref = ref_logps.flatten()`,
and after `loss = pg_loss - ...` add:

```python
                if ref is not None:
                    d = (b_ref[mb] - logp).clamp(-20, 20)  # log pi_ref(a) / pi(a) of the taken action
                    kl = (d.exp() - 1 - d).mean()  # k3 estimator of KL(pi || pi_ref): unbiased and >= 0
                    loss = loss + a.kl_coef * kl
                    stats["loss/kl_ref"].append(kl.item())
```

Update the module docstring's usage lines:

```python
"""PPO training (CleanRL-style) with pause/save/resume, for a PC that is also used daily.

  Train:   .venv/Scripts/python -m trainer.ppo --run onion
  Ranked:  .venv/Scripts/python -m trainer.ppo --run ppo-bc --setup ranked1v1 --init checkpoints/bc/latest.pt
               --opponent checkpoints/bc/latest.pt --kl-coef 0.05
  Resume:  .venv/Scripts/python -m trainer.ppo --run onion --resume
  Control: http://127.0.0.1:8787 or python -m trainer.ctl pause|resume|save|stop|throttle|status
  Charts:  .venv/Scripts/tensorboard --logdir runs
"""
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_ppo -v`
Expected: PASS. The Phase 1 smoke test is unchanged: onion, 1 seat, steps 16.
Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK.

- [ ] **Step 5: Check memory and speed on ranked maps** with the untrained model
  (`.venv/Scripts/python -m trainer.ppo --run ppo-probe --setup ranked1v1 --opponent <any checkpoint> --updates 3`).
  Read the VRAM and `perf/decisions_per_s` from the control page.
  - Expected: under 10 GB of VRAM.
  - If it is higher, lower `--steps` or `--games-per-worker` and record the numbers.
  - Delete `checkpoints/ppo-probe` and `runs/ppo-probe` afterwards.

- [ ] **Step 6: Commit**

```bash
git add trainer/ppo.py trainer/test_ppo.py
git commit -m "PPO from a checkpoint against a frozen opponent, with a KL penalty to the initial policy"
```

---

### Task 8: Watching games in the game's own client

On localhost, the game's dev client loads an archived game from `http://localhost:8787/game/<id>`
(`src/client/ApiBase.ts`). Its `gitCommit` is `"DEV"`, so it skips the version check (`src/client/JoinLobbyModal.ts`,
`checkArchivedGame`). Port 8787 is our control port, so the control server serves saved games at `/game/<id>`, and
`trainer.watch` does the same when no training runs. The task starts with a short spike on a real public record,
before any recording code is written.

**Files:**
- Create: `trainer/watch.py`
- Modify: `trainer/control.py`, `trainer/test_control.py`, `bridge/game.ts`, `bridge/worker.ts`,
  `bridge/replay.test.ts`, `trainer/env.py`, `trainer/eval.py`, `trainer/test_eval.py`

**Interfaces:**
- Produces:
  - `REPLAYS` (`replays/` at the repo root) and `GET /game/<id>` with CORS, whose preflight is approved only for these
    paths.
  - `python -m trainer.watch [--fetch <id>]`
  - `EnvGame.record()`, `.recording`, `.toRecord(gitCommit): GameRecord`
  - Worker init option `record: {dir, gitCommit, games}` (games recorded per worker); the finished game's header
    carries `recorded`.
  - `VecEnv(..., record=...)`; the infos carry `recorded`.
  - `play(..., record=...)` returns `recorded` IDs; `trainer.eval --record N`.

- [ ] **Step 1: Write the failing control test** — add to `ControlTest` in `trainer/test_control.py`
  (import `REPLAYS` from `trainer.control`):

```python
    def test_saved_games_are_served_with_cors_and_commands_are_not(self):
        REPLAYS.mkdir(exist_ok=True)
        path = REPLAYS / "Test0001.json"
        path.write_text('{"info": {}}')
        try:
            with urllib.request.urlopen(f"{self.base}/game/Test0001", timeout=5) as r:
                self.assertEqual(json.load(r), {"info": {}})
                self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(f"{self.base}/game/Nope0001", timeout=5)
            self.assertEqual(e.exception.code, 404)
            preflight = urllib.request.Request(f"{self.base}/game/Test0001", method="OPTIONS")
            with urllib.request.urlopen(preflight, timeout=5) as r:
                self.assertEqual(r.status, 204)
                self.assertEqual(r.headers["Access-Control-Allow-Methods"], "GET")
            with self.assertRaises(urllib.error.HTTPError):  # no preflight approval for the buttons
                urllib.request.urlopen(urllib.request.Request(f"{self.base}/pause", method="OPTIONS"), timeout=5)
            with self.assertRaises(urllib.error.HTTPError):  # only game IDs, never paths
                urllib.request.urlopen(f"{self.base}/game/..%2F..%2FREADME", timeout=5)
        finally:
            path.unlink(missing_ok=True)
```

Run: `.venv/Scripts/python -m unittest trainer.test_control -v`
Expected: FAIL, `ImportError: cannot import name 'REPLAYS'`.

- [ ] **Step 2: Serve saved games** — `trainer/control.py`

Add `import re` and `from pathlib import Path`, and below the imports:

```python
REPLAYS = Path(__file__).resolve().parents[1] / "replays"  # saved games (trainer.eval --record), served at /game/<id>
GAME_PATH = re.compile(r"^/game/([A-Za-z0-9]{8,10})$")
```

In `serve`'s handler, give `_send` a `cors=False` parameter that adds `Access-Control-Allow-Origin: *` before
`end_headers()`. Start `do_GET` with:

```python
            game = GAME_PATH.match(self.path)
            if game:
                # For the game's own client: in dev mode it loads archived games from localhost:8787/game/<id>.
                path = REPLAYS / f"{game[1]}.json"
                if not path.exists():
                    return self._send(404, "{}", cors=True)
                return self._send(200, path.read_text(encoding="utf-8"), cors=True)
```

and add:

```python
        def do_OPTIONS(self):
            # CORS preflight: other pages may read saved games and nothing else (the buttons stay same-page, do_POST).
            if not GAME_PATH.match(self.path):
                return self._send(404, "{}")
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
```

Run: `.venv/Scripts/python -m unittest trainer.test_control -v`
Expected: PASS.

- [ ] **Step 3: The watch command** — `trainer/watch.py`

```python
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
```

- [ ] **Step 4: Spike — watch a public record in the dev client** (time box: half a day)

1. `.venv/Scripts/python -m trainer.watch --fetch <the fixture's game ID from bridge/replay.test.ts>`
2. In another terminal: `cd OpenFront_Source_Code/openfrontio && npm run dev`. The `run-openfront` skill in the game
   repo describes starting and driving the dev client.
3. Open `http://localhost:9000/game/<id>` in the built-in browser.

Expected: the replay plays: the map, both players' territory and the bots.
- If it does not, read the browser console and the request log.
  - If the dev client fetches other `localhost:8787` paths and stops without them, add the smallest stub for each one
    to `do_GET` and try again.
  - If the join flow cannot reach the replay at all, stop the task and report to the user. Watching is then deferred,
    and every other task is independent of it.
- Record what was needed in the task report.

- [ ] **Step 5: Write the failing recording test** — append to `bridge/replay.test.ts`

Add to the imports `GameRecordSchema` (Schemas), `EnvGame` (./game), `landTile` (./testkit) and `Act` (already
imported).

```ts
test("a recorded game passes the record schema and replays to the same state", async () => {
  const env = await EnvGame.create({ ...ONION, seats: 2, nations: 0, timerMinutes: 1 }, "rec");
  env.record();
  const cell = env.frame.cellOf(landTile(env));
  for (let step = 0; !env.done && step < 1000; step++) {
    env.observe(step === 0);
    const mine = step === 2 ? [Act.Spawn, cell, 0, 0] : step > 45 && step % 10 === 0 ? [Act.Attack, 0, 0, 2] : [0, 0, 0, 0];
    env.step([mine, [Act.Noop, 0, 0, 0]]);
  }
  assert.ok(env.done);
  const record = env.toRecord("DEV");
  GameRecordSchema.parse(record); // what the game's client checks before replaying
  const rg = await RecordGame.create(decompressGameRecord(structuredClone(record)), env.frame.stride, env.slots);
  while (!rg.done) {
    rg.labels();
    rg.advance();
  }
  assert.equal(rg.desync, false);
  assert.equal(rg.env.hash, env.hash);
  assert.deepEqual(rg.env.observe(true), env.observe(true), "a replay sees what the agent saw");
});
```

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test ../../bridge/replay.test.ts`
Expected: FAIL, `env.record is not a function`.

- [ ] **Step 6: Record games** — `bridge/game.ts`

Imports: add `GameRecord`, `PlayerRecord`, `Turn` and `Winner` to the `Schemas` import, and `createPartialGameRecord` to
the `Util` import. Add to `EnvGame`:

```ts
  /** Executed turns while the game is recorded (record()). */
  private turns: Turn[] | null = null;

  /** Records every executed turn from now on (call it before the first step), for toRecord(). */
  record(): void {
    this.turns = [];
  }

  get recording(): boolean {
    return this.turns !== null;
  }

  /** The recorded game as a game record (GameRecordSchema), which the game's own client can replay. */
  toRecord(gitCommit: string): GameRecord {
    if (this.turns === null) throw new Error("this game was not recorded");
    const w = this.game.getWinner();
    const winner: Winner =
      w === null
        ? undefined
        : typeof w === "string"
          ? ["team", w]
          : w.clientID() !== null
            ? ["player", w.clientID()!]
            : ["nation", w.name()];
    const players: PlayerRecord[] = this.start.players.map((p) => ({ ...p, persistentID: null, stats: {} }));
    const end = this.game.ticks() * 100; // ms of game time: 10 ticks a second
    const partial = createPartialGameRecord(this.start.gameID, this.start.config, players, this.turns, 0, end, winner);
    return { ...partial, gitCommit } as GameRecord;
  }
```

and log the turns in `tickWith`:

```ts
  tickWith(intents: StampedIntent[]): void {
    const turnNumber = this.turn++;
    this.runner.addTurn({ turnNumber, intents });
    if (!this.runner.executeNextTick()) throw new Error(`tick ${turnNumber} failed: ${this.sim.fatal}`);
    if (this.turns !== null) {
      this.turns.push(this.sim.hashTick === turnNumber ? { turnNumber, intents, hash: this.sim.hash } : { turnNumber, intents });
    }
  }
```

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test ../../bridge/replay.test.ts && node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: PASS, tsc exit 0.

- [ ] **Step 7: Record from the worker, the env and eval**

`bridge/worker.ts`:
- Import `fs` and `path`.
- Add `record?: { dir: string; gitCommit: string; games: number };` to `Init`.
- Keep a counter `let recordStarted = 0;`, reset to 0 in the training-game init, and create training games through:

```ts
/** A new training game; the first `record.games` games of this worker are recorded. */
async function newGame(g: number): Promise<EnvGame> {
  const env = await EnvGame.create(init!.config, seedOf(g));
  if (init!.record !== undefined && recordStarted < init!.record.games) {
    env.record();
    recordStarted++;
  }
  return env;
}
```

In the training-game step, when a game ends, save it before replacing it:

```ts
    let recorded: string | undefined;
    if (env.recording) {
      fs.mkdirSync(init.record!.dir, { recursive: true });
      recorded = env.start.gameID;
      fs.writeFileSync(path.join(init.record!.dir, `${recorded}.json`), JSON.stringify(env.toRecord(init.record!.gitCommit)));
    }
```

and add `...(recorded ? { recorded } : {})` to the finished game's `end` header.

`trainer/env.py`:
- `VecEnv.__init__` takes `record=None`, and in training mode sets `msg["record"] = record` when it is given.
- In `_absorb`, the finished game's infos gain `"recorded": g.get("recorded")`.

`trainer/eval.py`:
- `play(..., record=None)` passes `record` to `VecEnv` and returns
  `{**summarize(results), "recorded": sorted({i["recorded"] for i in seen if i.get("recorded")})}`, where `seen` holds
  every info (keep one list next to `results`).
- `main` gains `--record N` (games saved per worker, default 0). With N > 0 it passes
  `{"dir": str(REPLAYS), "gitCommit": game_commit(), "games": N}` (`from trainer.control import REPLAYS`,
  `from trainer.ppo import game_commit`) and prints the watch URLs:

```python
    for game_id in result["recorded"]:
        print(f"watch: python -m trainer.watch (or a running trainer), then http://localhost:9000/game/{game_id}")
```

In `trainer/test_eval.py`, extend `test_scripted_against_random` to record one game into a temporary directory:

```python
    def test_scripted_against_random(self):
        tmp = tempfile.mkdtemp()
        r = play(["scripted", "random"], ONION2, episodes=4, workers=1, games_per_worker=4, seed="eval-2",
                 record={"dir": tmp, "gitCommit": "DEV", "games": 1})
        self.assertEqual(r["episodes"], 4)
        self.assertTrue(0.0 <= r["win_rate"] <= 1.0)
        self.assertEqual(len(r["recorded"]), 1)
        saved = json.loads((Path(tmp) / f"{r['recorded'][0]}.json").read_text())
        self.assertEqual(saved["info"]["gameID"], r["recorded"][0])
```

(with `import json`, `import tempfile` and `from pathlib import Path`).

- [ ] **Step 8: Run all tests**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test --test-timeout=300000 "../../bridge/*.test.ts" && node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: all pass; tsc exit 0.
Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK.

- [ ] **Step 9: Watch one of our own games**

1. `.venv/Scripts/python -m trainer.eval --a scripted --b random --episodes 8 --record 1`
2. `.venv/Scripts/python -m trainer.watch`
3. Open one printed URL in the dev client (Step 4).

Expected: the game plays with the agents named agent0 and agent1. Take one screenshot for the task report.

- [ ] **Step 10: Commit**

```bash
git add trainer/watch.py trainer/control.py trainer/test_control.py bridge/game.ts bridge/worker.ts bridge/replay.test.ts trainer/env.py trainer/eval.py trainer/test_eval.py
git commit -m "Record evaluation games and serve them to the game's own client"
```

---

### Task 9: Documentation, training runs and the gate

**Files:**
- Modify: `CLAUDE.md`, `docs/PLAN.md` (Phase 2 results), `README.md` (status)

- [ ] **Step 1: Update the commands in `CLAUDE.md`**

Replace the eval line and add the new commands in "Training commands (repo root)":

```bash
.venv/Scripts/python -m trainer.crawl                          # download ranked 1v1 records into data/ (hours: detach)
.venv/Scripts/python -m trainer.crawl --purge                  # remove deleted players' records (daily while data is kept)
.venv/Scripts/python -m trainer.bc --run bc                     # behaviour cloning on the records; same control page
.venv/Scripts/python -m trainer.eval --a checkpoints/bc/latest.pt --b scripted   # ranked 1v1 match, 256 games
.venv/Scripts/python -m trainer.eval --setup onion --a random   # Phase 1 setup: one seat vs 3 Easy nations
.venv/Scripts/python -m trainer.ppo --run ppo-bc --setup ranked1v1 --init checkpoints/bc/latest.pt --opponent checkpoints/bc/latest.pt --kl-coef 0.05
.venv/Scripts/python -m trainer.eval --a checkpoints/bc/latest.pt --b random --episodes 8 --record 1   # save games to replays/
.venv/Scripts/python -m trainer.watch                          # serve replays/ to the game's dev client (npm run dev)
```

Under "Simulation facts the bridge relies on", add:

```markdown
- A game's RNG is seeded from its game ID (as in the game's client), so training games get a valid 8-character ID from
  their seed (`bridge/setups.ts: gameIdOf`) and `EnvGame.fromStart` builds env games and replayed records alike. A
  recorded game (`EnvGame.record/toRecord`) replays in the game's own client.
- All games of a trainer share one padded cell grid (`gridFor`: 192 x 256 for ranked 1v1). Cell indexes everywhere
  are grid indexes y x W + x; padding cells hold no tiles.
- Records store only non-empty turns: expand them with `decompressGameRecord` (`bridge/replay.ts: readRecord`).
```

- [ ] **Step 2: BC run (ask the user first)**

Size `--updates` for about one pass over the training records, from Task 4's windows per record. That is about
records × windows per record ÷ games in flight: roughly 200k updates for 17k records, which is the default. Estimate
the hours from Task 5's samples/s and tell the user before starting. With their OK, start it detached:

```bash
.venv/Scripts/python -c "import subprocess, sys; log = open('runs/bc.log', 'ab'); subprocess.Popen([sys.executable, '-m', 'trainer.bc', '--run', 'bc'], stdout=log, stderr=log, creationflags=0x00000200 | 0x08000000 | 0x01000000)"
```

While it runs, read `val/nll_*`, `val/type_acc`, `val/*_acc`, `val/value_outcome_corr` and `labels/*` in TensorBoard.
Stop early (`python -m trainer.ctl stop`) if `val/nll` has risen for 3 validations in a row; the latest checkpoint
before the rise is the one to use.

- [ ] **Step 3: Gate A — the clone beats the baselines in ranked 1v1**

```bash
.venv/Scripts/python -m trainer.eval --a checkpoints/bc/latest.pt --b random --episodes 256
.venv/Scripts/python -m trainer.eval --a checkpoints/bc/latest.pt --b scripted --episodes 256
```

Gate:
- a win rate of at least 0.90 against random and at least 0.75 against the scripted expander;
- each with a standard error of at most 0.03 (256 games);
- `val/value_outcome_corr` of at least 0.5.

If either is missed, report the per-head accuracies and label statistics, and ask the user before going on. Record and
watch 2 games of BC against the scripted expander (`--record 1`, `trainer.watch`), and describe what the clone does in
the report.

- [ ] **Step 4: PPO from the clone (ask the user first)**

Estimate the hours from Task 7's decisions/s and propose `--updates`. With the user's OK, start it detached (log
`runs/ppo-bc.log`):

```bash
.venv/Scripts/python -m trainer.ppo --run ppo-bc --setup ranked1v1 --init checkpoints/bc/latest.pt --opponent checkpoints/bc/latest.pt --kl-coef 0.05 --updates <N>
```

Watch `episodes/win_rate` (against the frozen clone, starting near 0.5), `loss/kl_ref`, `value/outcome_corr_*` and
`actions/*`.
- If `loss/kl_ref` grows without the win rate moving, stop the run (`python -m trainer.ctl stop`) and resume it
  with a higher `--kl-coef`. On `--resume`, the new value replaces the run's.
- If the win rate climbs fast and the KL stays small, lower it the same way.

- [ ] **Step 5: Gate B — PPO beats the frozen clone**

```bash
.venv/Scripts/python -m trainer.eval --a checkpoints/ppo-bc/latest.pt --b checkpoints/bc/latest.pt --episodes 512
```

Gate: a win rate above 0.5 + 2 × SE, about 0.545 with 512 games. Record 2 games and watch one.

- [ ] **Step 6: Record the results**

- In `docs/PLAN.md`, add "Phase 2 results" under Phase 2. Include:
  - data: records, void share, desync share, label mix, out-of-set and illegal shares;
  - speed: BC samples/s and PPO decisions/s;
  - the BC curves: val NLL and accuracy per head and per phase;
  - Gate A and Gate B with their numbers;
  - what watching the games showed;
  - open issues.
- Update the README status line and the memory note.

```bash
git add CLAUDE.md docs/PLAN.md README.md
git commit -m "Phase 2 results: behaviour cloning on ranked 1v1 and PPO from the clone"
```

---

## Verification (whole phase)

- **Bridge:** `node --import tsx --test --test-timeout=300000 "../../bridge/*.test.ts"` and `tsc`, from the game dir:
  - setups (IDs, playlist weights, grids);
  - the padded frame;
  - ranked env games;
  - replay of the fixture in sync;
  - label queue and out-of-set;
  - worker replay mode;
  - a recorded game that replays to identical blobs;
  - every Phase 1 test.
- **Trainer:** `.venv/Scripts/python -m unittest discover -s trainer -t .`:
  - crawler (offline), padded ranked grid, replay stream;
  - BC (label legality, overfit, train/save/resume);
  - matches (seat swap, recording), control (saved games, CORS);
  - PPO against a frozen opponent with KL;
  - every Phase 1 test.
- **Data checks:** 20 crawled records replay with 0 desyncs (Task 4, Step 7); the BC probe shows under 10% illegal labels
  (Task 5, Step 5).
- **Gates:** A (BC against random and scripted, value correlation) and B (PPO against frozen BC) in ranked 1v1, 256 and
  512 games, with standard errors.
- **Watching:** a public record and one of our games play in the game's dev client.
