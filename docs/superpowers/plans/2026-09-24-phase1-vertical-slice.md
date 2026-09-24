# Phase 1 — Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One agent learns, with PPO and a terminal-only reward, to beat 3 Easy nations on the Compact Onion map clearly
better than a random-legal agent, while the training can be paused, saved, resumed and throttled from a local web page
or a CLI.

**Architecture:** Node workers (`bridge/`, TypeScript, the unmodified game imported by relative path) each run K games
and exchange length-prefixed binary messages with one Python trainer over stdin/stdout. Workers send per-cell frame
deltas, unit lists, a per-seat player table, scalars and masks. The trainer (`trainer/`, PyTorch) builds the observation
tensors on the GPU, samples actions from a small ResNet policy with spatial heat-map heads and trains it with PPO. A
local HTTP control server (127.0.0.1) serves a status page and takes pause/resume/save/stop/throttle commands.

**Tech Stack:** Node 24 with the game's own `tsx` 4.22 and TypeScript 6.0 (from the submodule's `node_modules`),
`node:test`; Python 3.11 with torch 2.14 cu130, numpy, psutil, tensorboard, stdlib `unittest`, `http.server`.

**Spec:** `docs/PLAN.md`, sections "Key decisions", "Architecture", "Action space (v1)", "Phase 0 results" and
"Phase 1 — Vertical slice".

## Global Constraints

- Never edit anything inside `OpenFront_Source_Code/openfrontio/`; our code lives in `bridge/`, `trainer/`, `scripts/`.
- Commits are plain messages with no `Co-Authored-By` or any other Claude attribution line.
- Rule values (costs, limits, spawn-phase length, boat limit) come from the game's `Config` at runtime, never hard-coded.
- Anything that feeds the simulation must be deterministic: seeded `PseudoRandom`, never `Math.random`.
- Decisions every Δ = 5 ticks; agent intents delayed L ∈ {1, 2, 3} turns with p = .6/.3/.1; intents due in the same
  turn are shuffled (PLAN decisions 11, 12).
- Reward is terminal only: win = +1, otherwise −1 + 0.5·(N − rank)/(N − 1) (PLAN decision 9).
- No new npm packages. Python adds nothing beyond `requirements.txt` (torch, numpy, psutil, tensorboard).
- Windows 11: spawn Node through `shutil.which("node")`; the control server binds to 127.0.0.1 only.
- TS commands run from `OpenFront_Source_Code/openfrontio/` (the game's `tsx` and `tsconfig` paths must resolve);
  Python commands run from the repo root with `.venv/Scripts/python`.

## Review Focus

- A Node worker crashes or is killed mid-training → the trainer raises an error naming the worker's log file, saves a
  checkpoint and exits; it never hangs. (Test in Task 7; save-on-exit in Task 10.)
- Pause, then resume, or quit while paused → training continues without a lost or duplicated update, and quitting
  while paused still saves. (Tests in Tasks 9 and 10.)
- Resume after quit → the update counter, RNG states and every game slot's episode counter continue, so no game seed
  is played twice. (Test in Task 10.)
- A seat that never spawns, or dies early → the episode ends at once with the last rank and reward −1, and the slot
  restarts with the next seed. (Tests in Tasks 5, 6 and 7.)
- Actions that the approximate cell masks allow but the game rejects (boat to an unreachable shore, city on a blocked
  tile, malformed numbers) → dropped as a no-op and counted in `actions/invalid_rate`, never a crash. (Test in Task 4.)

---

## File structure

| File | Responsibility |
|---|---|
| `package.json` | `"type": "module"` so our `.ts` files are ES modules (top-level `await`, `import.meta.url`) |
| `tsconfig.json` | Typecheck of `bridge/` against the game's compiler settings and types |
| `bridge/maps.ts` | Fresh (uncached) terrain per game; reset of the game's cross-game static fields |
| `bridge/frame.ts` | Incremental per-cell ownership frame, static terrain per cell, unit list encoding |
| `bridge/latency.ts` | Seeded intent latency queue |
| `bridge/ranking.ts` | Seat rank and terminal reward |
| `bridge/game.ts` | `EnvGame`: builds a game from an `EnvConfig` + seed, steps it, observes it, detects the end |
| `bridge/actions.ts` | Action types, masks, decoding `[type, cell, target, bin]` into game intents |
| `bridge/obs.ts` | Per-seat player table, relations and scalars |
| `bridge/worker.ts` | stdin/stdout protocol: init, step (with auto-reset), close |
| `bridge/testkit.ts` | Shared test helpers (Onion config, scripted spawn/attack) |
| `bridge/*.test.ts` | `node:test` tests |
| `trainer/env.py` | Worker processes + `VecEnv` (GPU frame state, observation builder) |
| `trainer/model.py` | `Policy` network, `RandomPolicy`, `ActionDist` (masked sampling, log-prob, entropy) |
| `trainer/control.py` | Control state, HTTP API + status page, system sampler (CPU/RAM/GPU/temperatures) |
| `trainer/ctl.py` | CLI for the control API |
| `trainer/ppo.py` | PPO training loop, metrics, checkpoints, resume |
| `trainer/eval.py` | Fixed-seed evaluation (random baseline vs checkpoint) |
| `trainer/test_*.py` | `unittest` tests |

Protocol (worker ⇄ trainer), fixed here because both sides depend on it:

- Trainer → worker: `u32 length` + UTF-8 JSON, one of
  `{"cmd":"init","config":EnvConfig,"games":K,"seed":str,"worker":int,"episodes":[int]*K}`,
  `{"cmd":"step","actions":[[[type,cell,target,bin] per seat] per game],"full":bool}`, `{"cmd":"close"}`.
- Worker → trainer: `u32 size` + `u32 header length` + header JSON + blobs. Header:
  `{"dims":{...} (init only), "games":[{"tick","episode","done","reset","hash","invalid":[bool],"rewards"?,"ranks"?,
  "players"?,"endTick"?,"blobs":[[name, byteLength], ...]}]}`; blobs follow in the listed order, game by game.
- Blobs per game: `terrain` (reset only; u8 [4 × cells]: land, shore, ocean tiles, mean land height 0–31), `cells`
  (11-byte records: u32 cell, u16 owner1, u16 owner2, u8 tiles1, u8 tiles2, u8 fallout; all cells on reset or `full`,
  else only changed ones), `units` (12-byte records: u8 type index, u8 level, u16 owner, u32 cell, f32 troops),
  `players` (f32 [seats × slots × 17]), `rel` (u8 [seats × slots]), `scalars` (f32 [seats × 16]),
  `types` (u8 [seats × 5]), `attack` (u8 [seats × slots]).
- Game seeds: `f"{seed}-w{worker}-g{game}-e{episode}"`. A finished game is replaced by the next episode in the same
  reply (`done` and `reset` both true, `episode` = the new game's counter).

---

### Task 1: Bridge tooling and fresh maps

**Files:**
- Create: `package.json`, `tsconfig.json`, `bridge/maps.ts`, `bridge/maps.test.ts`
- Modify: `CLAUDE.md` (commands, simulation facts), `docs/PLAN.md:299` (test command)

**Interfaces:**
- Produces: `GAME_ROOT: string`; `loadFreshTerrain(map: GameMapType, size: GameMapSize): Promise<TerrainMapData>`;
  `resetGameStatics(): void`

- [ ] **Step 1: Create the module and typecheck config**

`package.json`:
```json
{
  "private": true,
  "type": "module"
}
```

`tsconfig.json` (the game's `.d.ts` files and `Main.ts` carry global declarations that imported client code needs):
```json
{
  "extends": "./OpenFront_Source_Code/openfrontio/tsconfig.json",
  "compilerOptions": {
    "noEmit": true,
    "types": ["node"],
    "typeRoots": ["./OpenFront_Source_Code/openfrontio/node_modules/@types"]
  },
  "include": [
    "bridge/**/*.ts",
    "OpenFront_Source_Code/openfrontio/src/**/*.d.ts",
    "OpenFront_Source_Code/openfrontio/src/client/Main.ts"
  ]
}
```

- [ ] **Step 2: Write the failing test** — `bridge/maps.test.ts`

```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { TransportShipExecution } from "../OpenFront_Source_Code/openfrontio/src/core/execution/TransportShipExecution";
import { GameMapSize, GameMapType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { loadFreshTerrain, resetGameStatics } from "./maps";

test("each call returns new map objects with compact coordinates halved once", async () => {
  const a = await loadFreshTerrain(GameMapType.Onion, GameMapSize.Compact);
  const b = await loadFreshTerrain(GameMapType.Onion, GameMapSize.Compact);
  assert.notEqual(a.gameMap, b.gameMap);
  assert.notEqual(a.gameMap.tileStateBuffer(), b.gameMap.tileStateBuffer());
  assert.equal(a.gameMap.width(), 256);
  assert.equal(a.miniGameMap.width(), 128);
  assert.deepEqual(a.nations.map((n) => n.coordinates), b.nations.map((n) => n.coordinates));
  const normal = await loadFreshTerrain(GameMapType.Onion, GameMapSize.Normal);
  const [x, y] = normal.nations[0].coordinates!;
  assert.deepEqual(a.nations[0].coordinates, [Math.floor(x / 2), Math.floor(y / 2)]);
});

test("resetGameStatics zeroes the ship stagger counters", () => {
  (TransportShipExecution as unknown as { _staggerCounter: number })._staggerCounter = 7;
  resetGameStatics();
  assert.equal((TransportShipExecution as unknown as { _staggerCounter: number })._staggerCounter, 0);
});
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test ../../bridge/maps.test.ts`
Expected: FAIL, `Cannot find module '.../bridge/maps'`.

- [ ] **Step 4: Implement** — `bridge/maps.ts`

```ts
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { NationMIRVBehavior } from "../OpenFront_Source_Code/openfrontio/src/core/execution/nation/NationMIRVBehavior";
import { TradeShipExecution } from "../OpenFront_Source_Code/openfrontio/src/core/execution/TradeShipExecution";
import { TransportShipExecution } from "../OpenFront_Source_Code/openfrontio/src/core/execution/TransportShipExecution";
import {
  GameMapSize,
  GameMapType,
  TeamGameSpawnAreas,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import {
  genTerrainFromBin,
  MapManifest,
  TerrainMapData,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/TerrainMapLoader";

export const GAME_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../OpenFront_Source_Code/openfrontio",
);

type MapFiles = { manifest: string; map: Uint8Array; map4x: Uint8Array; map16x: Uint8Array };
const files = new Map<GameMapType, MapFiles>();

function mapFiles(map: GameMapType): MapFiles {
  let f = files.get(map);
  if (f === undefined) {
    const key = Object.keys(GameMapType).find((k) => GameMapType[k as keyof typeof GameMapType] === map);
    if (key === undefined) throw new Error(`unknown map: ${map}`);
    const dir = path.join(GAME_ROOT, "resources/maps", key.toLowerCase());
    const read = (name: string) => new Uint8Array(fs.readFileSync(path.join(dir, name)));
    f = {
      manifest: fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
      map: read("map.bin"),
      map4x: read("map4x.bin"),
      map16x: read("map16x.bin"),
    };
    files.set(map, f);
  }
  return f;
}

const half = (c: [number, number] | undefined): [number, number] | undefined =>
  c === undefined ? undefined : [Math.floor(c[0] / 2), Math.floor(c[1] / 2)];

/**
 * Same result as the game's loadTerrainMap, but every call returns new map objects: loadTerrainMap caches one mutable
 * map per map and size, so a second game in the same process would start on the first game's territory.
 */
export async function loadFreshTerrain(map: GameMapType, size: GameMapSize): Promise<TerrainMapData> {
  const f = mapFiles(map);
  const manifest = JSON.parse(f.manifest) as MapManifest;
  const normal = size === GameMapSize.Normal;
  // slice(): a map writes to its terrain bytes (water nukes), so every game gets its own copy.
  const gameMap = await genTerrainFromBin(normal ? manifest.map : manifest.map4x, (normal ? f.map : f.map4x).slice());
  const miniGameMap = await genTerrainFromBin(
    normal ? manifest.map4x : manifest.map16x,
    (normal ? f.map4x : f.map16x).slice(),
  );
  let teamGameSpawnAreas = manifest.teamGameSpawnAreas;
  if (!normal) {
    for (const n of manifest.nations) n.coordinates = half(n.coordinates);
    for (const n of manifest.additionalNations ?? []) n.coordinates = half(n.coordinates);
    if (teamGameSpawnAreas !== undefined) {
      const scaled: TeamGameSpawnAreas = {};
      for (const [key, areas] of Object.entries(teamGameSpawnAreas)) {
        scaled[key] = areas.map((a) => ({
          x: Math.floor(a.x / 2),
          y: Math.floor(a.y / 2),
          width: Math.max(1, Math.floor(a.width / 2)),
          height: Math.max(1, Math.floor(a.height / 2)),
        }));
      }
      teamGameSpawnAreas = scaled;
    }
  }
  return {
    nations: manifest.nations,
    additionalNations: manifest.additionalNations ?? [],
    gameMap,
    miniGameMap,
    teamGameSpawnAreas,
  };
}

/**
 * Static fields in the game code that outlive a game. The real client runs one game per worker, so they start fresh;
 * here several games share a process, so they are reset before each game or a game's outcome would depend on the
 * games that ran before it. On every game upgrade, grep src/core for `private static` fields that are not readonly.
 */
export function resetGameStatics(): void {
  (TransportShipExecution as unknown as { _staggerCounter: number })._staggerCounter = 0;
  (TradeShipExecution as unknown as { _staggerCounter: number })._staggerCounter = 0;
  (NationMIRVBehavior as unknown as { recentMirvTargets: Map<string, number> }).recentMirvTargets.clear();
}
```

- [ ] **Step 5: Run the test and the typecheck**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test ../../bridge/maps.test.ts`
Expected: PASS (2 tests).
Run: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: no output, exit 0.

- [ ] **Step 6: Update the docs**

In `CLAUDE.md`, add after the "Game commands" block:

````markdown
## Bridge commands (run inside `OpenFront_Source_Code/openfrontio/`, which provides `tsx` and the tsconfig paths)

```bash
node --import tsx --test "../../bridge/*.test.ts"              # all bridge tests (node:test)
node --import tsx --test ../../bridge/game.test.ts             # one test file
node node_modules/typescript/bin/tsc -p ../../tsconfig.json    # typecheck bridge/ (tsx does not typecheck)
```
````

Replace the last bullet of "Simulation facts the bridge relies on" with:

```markdown
- v0.34.x has no game snapshots. `loadTerrainMap` caches one mutable map per map and size per process, so the bridge
  builds a fresh map for every game (`bridge/maps.ts: loadFreshTerrain`) and resets the game's cross-game static
  fields before each game (`resetGameStatics`); several games can then share a worker process. On every upgrade, grep
  `src/core` for non-readonly `private static` fields and add new ones to `resetGameStatics`.
- Code that only builds observations may use `Math.*`; code whose output reaches the simulation (intents, latency)
  must use seeded `PseudoRandom`.
```

In `docs/PLAN.md` "Verification", replace `` `npx vitest bridge --run` `` with
`` `node --import tsx --test "../../bridge/*.test.ts"` (node:test, run from the game dir) ``.

- [ ] **Step 7: Commit**

```bash
git add package.json tsconfig.json bridge/maps.ts bridge/maps.test.ts CLAUDE.md docs/PLAN.md
git commit -m "Bridge tooling and fresh per-game maps"
```

---

### Task 2: Latency queue and ranking

**Files:**
- Create: `bridge/latency.ts`, `bridge/latency.test.ts`, `bridge/ranking.ts`, `bridge/ranking.test.ts`

**Interfaces:**
- Produces: `class LatencyQueue { constructor(random: PseudoRandom); push(intent: StampedIntent, nextTurn: number): void;
  take(turn: number): StampedIntent[] }`; `interface Standing { won; spawned; deathPosition?; tiles }`;
  `rankOf(me: Standing, others: number[], n: number): number`; `rewardOf(rank: number, n: number): number`;
  `seatStanding(game: Game, seat: Player): { rank: number; n: number }`

- [ ] **Step 1: Write the failing tests**

`bridge/latency.test.ts`:
```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { LatencyQueue } from "./latency";

const intent = (tile: number): StampedIntent => ({ type: "spawn", tile, clientID: "AGENT000" });

test("delays are 1, 2 or 3 turns with p = .6/.3/.1", () => {
  const q = new LatencyQueue(new PseudoRandom(42));
  const counts = [0, 0, 0];
  for (let i = 0; i < 10000; i++) {
    q.push(intent(i), 100);
    for (let d = 0; d < 3; d++) if (q.take(100 + d).length > 0) counts[d]++;
  }
  assert.deepEqual(counts.map((c) => Math.round(c / 1000)), [6, 3, 1]);
});

test("same seed, same delays", () => {
  const run = () => {
    const q = new LatencyQueue(new PseudoRandom(7));
    const due: number[] = [];
    for (let i = 0; i < 50; i++) {
      q.push(intent(i), 0);
      for (let t = 0; t < 3; t++) if (q.take(t).length > 0) due.push(t);
    }
    return due;
  };
  assert.deepEqual(run(), run());
});

test("intents due in the same turn come out in varying order", () => {
  const q = new LatencyQueue(new PseudoRandom(3));
  const firsts = new Set<number>();
  for (let i = 0; i < 200; i++) {
    const turn = i * 10;
    q.push(intent(1), turn);
    q.push(intent(2), turn);
    const out = [...q.take(turn), ...q.take(turn + 1), ...q.take(turn + 2)];
    const first = out[0];
    if (out.length === 2 && first.type === "spawn") firsts.add(first.tile);
  }
  assert.deepEqual([...firsts].sort(), [1, 2]);
});
```

`bridge/ranking.test.ts`:
```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { rankOf, rewardOf } from "./ranking";

const alive = { won: false, spawned: true, tiles: 500 };

test("rank: winner first, dead keep their placement, alive by tiles, never spawned last", () => {
  assert.equal(rankOf({ ...alive, won: true }, [900], 4), 1);
  assert.equal(rankOf({ ...alive, tiles: 0, deathPosition: 3 }, [900, 100], 4), 3);
  assert.equal(rankOf(alive, [900, 100, 700], 4), 3);
  assert.equal(rankOf(alive, [100, 200], 4), 1);
  assert.equal(rankOf({ ...alive, spawned: false, tiles: 0 }, [900], 4), 4);
});

test("reward: +1 for a win, -1 + 0.5 (n - rank) / (n - 1) otherwise", () => {
  assert.equal(rewardOf(1, 4), 1);
  assert.equal(rewardOf(4, 4), -1);
  assert.equal(rewardOf(2, 4), -1 + 0.5 * (2 / 3));
  assert.equal(rewardOf(2, 2), -1);
  assert.equal(rewardOf(1, 1), 1);
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `node --import tsx --test ../../bridge/latency.test.ts ../../bridge/ranking.test.ts`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement**

`bridge/latency.ts`:
```ts
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";

/**
 * Delays agent intents like a real client's network latency: L = 1, 2 or 3 turns with p = .6/.3/.1 (PLAN decision
 * 12). Intents that fall due in the same turn are shuffled so no seat always acts first.
 */
export class LatencyQueue {
  private pending: { due: number; intent: StampedIntent }[] = [];

  constructor(private readonly random: PseudoRandom) {}

  /** Queues an intent decided before turn `nextTurn` executes. */
  push(intent: StampedIntent, nextTurn: number): void {
    const r = this.random.next();
    this.pending.push({ due: nextTurn + (r < 0.6 ? 0 : r < 0.9 ? 1 : 2), intent });
  }

  /** The intents to execute in `turn`, in shuffled order. */
  take(turn: number): StampedIntent[] {
    const due = this.pending.filter((p) => p.due <= turn).map((p) => p.intent);
    if (due.length === 0) return due;
    this.pending = this.pending.filter((p) => p.due > turn);
    return due.length > 1 ? this.random.shuffleArray(due) : due;
  }
}
```

`bridge/ranking.ts`:
```ts
import { Game, Player, PlayerType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";

export interface Standing {
  /** The game's winner is this player or its team. */
  won: boolean;
  spawned: boolean;
  /** Placement the game recorded when the player died (non-bot players still standing + 1). */
  deathPosition?: number;
  tiles: number;
}

/**
 * 1 = best. The winner is first; dead players keep the placement recorded at their death; players alive at the end
 * are ordered by tiles; a player who never spawned is last. `others`: tiles of the other alive non-bot players.
 */
export function rankOf(me: Standing, others: number[], n: number): number {
  if (me.won) return 1;
  if (!me.spawned) return n;
  if (me.deathPosition !== undefined) return me.deathPosition;
  return 1 + others.filter((t) => t > me.tiles).length;
}

/** Terminal reward (PLAN decision 9): +1 for a win, otherwise -1 + 0.5 (n - rank) / (n - 1). */
export function rewardOf(rank: number, n: number): number {
  if (rank === 1) return 1;
  return -1 + (0.5 * (n - rank)) / (n - 1);
}

/** The seat's rank among the game's non-bot players (humans and nations). */
export function seatStanding(game: Game, seat: Player): { rank: number; n: number } {
  const nonBot = game.allPlayers().filter((p) => p.type() !== PlayerType.Bot);
  const winner = game.getWinner();
  const won = winner !== null && (winner === seat || (typeof winner === "string" && seat.team() === winner));
  const others = nonBot.filter((p) => p !== seat && p.isAlive()).map((p) => p.numTilesOwned());
  const standing: Standing = {
    won,
    spawned: seat.hasSpawned(),
    deathPosition: seat.isAlive() ? undefined : (game.stats().getPlayerStats(seat)?.deathPosition ?? undefined),
    tiles: seat.numTilesOwned(),
  };
  return { rank: rankOf(standing, others, nonBot.length), n: nonBot.length };
}
```

- [ ] **Step 4: Run the tests and the typecheck**

Run: `node --import tsx --test ../../bridge/latency.test.ts ../../bridge/ranking.test.ts`
Expected: PASS (5 tests).
Run: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json` → exit 0.

- [ ] **Step 5: Commit**

```bash
git add bridge/latency.ts bridge/latency.test.ts bridge/ranking.ts bridge/ranking.test.ts
git commit -m "Bridge latency queue and terminal ranking"
```

---

### Task 3: Game creation, cell frame and determinism

**Files:**
- Create: `bridge/frame.ts`, `bridge/game.ts`, `bridge/testkit.ts`, `bridge/game.test.ts`

**Interfaces:**
- Consumes: `loadFreshTerrain`, `resetGameStatics` (Task 1)
- Produces: `CELL_BYTES = 11`, `UNIT_BYTES = 12`, `UNIT_TYPES: UnitType[]`,
  `class CellFrame { cs; cw; ch; n; o1; o2; n1; n2; fo; cellOf(tile); tilesOf(cell): TileRef[];
  center(cell): [number, number]; mark(packed: Uint32Array); encode(all: boolean): Uint8Array }`,
  `terrainCells(map: GameMap, frame: CellFrame): Uint8Array`, `encodeUnits(game: Game, frame: CellFrame): Uint8Array`;
  `interface EnvConfig { map; compact; nations; difficulty; bots; timerMinutes; seats }`, `STEP_TICKS = 5`,
  `seatClientID(seat: number): string`, `class EnvGame { static create(cfg, seed): Promise<EnvGame>; cfg; game; seats;
  frame; slots; hash; tickWith(intents: StampedIntent[]): void }`; testkit `ONION`, `landTile(env)`, `spawnCell(env)`,
  `script(env, ticks, onStep?)`

- [ ] **Step 1: Write the test kit and the failing tests**

`bridge/testkit.ts`:
```ts
import { TileRef } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameMap";
import { StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { EnvConfig, EnvGame, seatClientID } from "./game";

/** The Phase 1 gate setup: one agent against 3 Easy nations on Compact Onion, 5-minute timer. */
export const ONION: EnvConfig = {
  map: "Onion",
  compact: true,
  nations: 3,
  difficulty: "Easy",
  bots: 0,
  timerMinutes: 5,
  seats: 1,
};

/** A fixed land tile in the Onion map's lower right, for scripted spawns. */
export function landTile(env: EnvGame): TileRef {
  const g = env.game;
  let found = -1;
  g.forEachTile((t) => {
    if (found < 0 && g.isLand(t) && !g.isImpassable(t) && g.x(t) >= 100 && g.y(t) >= 100) found = t;
  });
  if (found < 0) throw new Error("no land tile found");
  return found;
}

export const spawnCell = (env: EnvGame): number => env.frame.cellOf(landTile(env));

/** Seat 0 spawns at tick 20 and attacks unowned land every 50 ticks after the spawn phase. */
export function script(env: EnvGame, ticks: number, onStep?: () => void): void {
  const tile = landTile(env);
  const clientID = seatClientID(0);
  for (let t = 0; t < ticks; t++) {
    const intents: StampedIntent[] =
      t === 20
        ? [{ type: "spawn", tile, clientID }]
        : t > 250 && t % 50 === 0
          ? [{ type: "attack", targetID: null, troops: 1000, clientID }]
          : [];
    env.tickWith(intents);
    if (t % 5 === 4) onStep?.();
  }
}
```

`bridge/game.test.ts`:
```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { CellFrame } from "./frame";
import { EnvGame } from "./game";
import { ONION, script } from "./testkit";

test("same seed gives the same game, also after other games in the process", async () => {
  const a = await EnvGame.create(ONION, "det-a");
  script(a, 1500);
  const b = await EnvGame.create(ONION, "det-b");
  script(b, 1500);
  const again = await EnvGame.create(ONION, "det-a");
  script(again, 1500);
  assert.notEqual(a.hash, 0);
  assert.equal(again.hash, a.hash);
  assert.notEqual(b.hash, a.hash);
});

test("the scripted seat spawns and expands", async () => {
  const env = await EnvGame.create(ONION, "expand");
  script(env, 1500);
  assert.ok(env.seats[0].numTilesOwned() > 100, `tiles ${env.seats[0].numTilesOwned()}`);
  assert.equal(env.slots, 5);
});

test("incrementally updated cells equal a full recompute", async () => {
  const env = await EnvGame.create(ONION, "frame");
  script(env, 1500, () => env.frame.encode(false));
  env.frame.encode(false);
  const full = new CellFrame(env.game.map(), env.frame.cs);
  full.encode(true);
  for (const k of ["o1", "o2", "n1", "n2", "fo"] as const) assert.deepEqual(env.frame[k], full[k], k);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --import tsx --test ../../bridge/game.test.ts`
Expected: FAIL, `Cannot find module '.../bridge/game'`.

- [ ] **Step 3: Implement** — `bridge/frame.ts`

```ts
import { Game, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { GameMap, TileRef } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameMap";

/** Cell record: u32 cell, u16 owner 1, u16 owner 2, u8 tiles of owner 1, u8 tiles of owner 2, u8 fallout tiles. */
export const CELL_BYTES = 11;
/** Unit record: u8 type index (UNIT_TYPES), u8 level (0 = under construction), u16 owner, u32 cell, f32 troops. */
export const UNIT_BYTES = 12;
/** Unit types sent to the trainer, by index. Shells, SAM missiles and trains are not observed. */
export const UNIT_TYPES: UnitType[] = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
  UnitType.MissileSilo,
  UnitType.SAMLauncher,
  UnitType.DefensePost,
  UnitType.TransportShip,
  UnitType.Warship,
  UnitType.TradeShip,
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
  UnitType.MIRVWarhead,
];
const UNIT_INDEX = new Map(UNIT_TYPES.map((t, i) => [t, i]));

/**
 * The map on a grid of cs x cs tiles: per cell its two largest owners with their tile counts, and the fallout count.
 * Only cells touched by the game's tile updates are recomputed (a full rebuild costs over half of the simulation
 * time, see PLAN.md Phase 0 results).
 */
export class CellFrame {
  readonly cw: number;
  readonly ch: number;
  readonly n: number;
  readonly o1: Uint16Array;
  readonly o2: Uint16Array;
  readonly n1: Uint8Array;
  readonly n2: Uint8Array;
  readonly fo: Uint8Array;
  private readonly w: number;
  private readonly h: number;
  private readonly state: Uint16Array;
  private readonly dirty: Uint8Array;
  private readonly dirtyList: Int32Array;
  private nDirty = 0;
  private readonly cnt = new Uint16Array(4096);
  private readonly touched: number[] = [];
  private readonly out: Uint8Array;
  private readonly dv: DataView;

  constructor(
    map: GameMap,
    readonly cs: number,
  ) {
    this.w = map.width();
    this.h = map.height();
    this.cw = Math.ceil(this.w / cs);
    this.ch = Math.ceil(this.h / cs);
    this.n = this.cw * this.ch;
    this.o1 = new Uint16Array(this.n);
    this.o2 = new Uint16Array(this.n);
    this.n1 = new Uint8Array(this.n);
    this.n2 = new Uint8Array(this.n);
    this.fo = new Uint8Array(this.n);
    this.state = map.tileStateBuffer();
    this.dirty = new Uint8Array(this.n);
    this.dirtyList = new Int32Array(this.n);
    this.out = new Uint8Array(this.n * CELL_BYTES);
    this.dv = new DataView(this.out.buffer);
  }

  cellOf(tile: TileRef): number {
    const x = tile % this.w;
    const y = (tile - x) / this.w;
    return Math.floor(y / this.cs) * this.cw + Math.floor(x / this.cs);
  }

  /** The cell's tiles, row by row. */
  tilesOf(cell: number): TileRef[] {
    const x0 = (cell % this.cw) * this.cs;
    const y0 = Math.floor(cell / this.cw) * this.cs;
    const tiles: TileRef[] = [];
    for (let y = y0; y < Math.min(y0 + this.cs, this.h); y++) {
      for (let x = x0; x < Math.min(x0 + this.cs, this.w); x++) tiles.push(y * this.w + x);
    }
    return tiles;
  }

  /** Tile coordinates of the cell's centre. */
  center(cell: number): [number, number] {
    return [(cell % this.cw) * this.cs + this.cs / 2, Math.floor(cell / this.cw) * this.cs + this.cs / 2];
  }

  /** Marks the cells touched by one tick's packedTileUpdates ([tileRef, state] pairs). */
  mark(packed: Uint32Array): void {
    for (let i = 0; i < packed.length; i += 2) {
      const c = this.cellOf(packed[i]);
      if (this.dirty[c] === 0) {
        this.dirty[c] = 1;
        this.dirtyList[this.nDirty++] = c;
      }
    }
  }

  /**
   * Recomputes the cells marked since the last call (or all cells) and returns CELL_BYTES records: every cell when
   * `all`, otherwise only the cells that changed.
   */
  encode(all: boolean): Uint8Array {
    let off = 0;
    const emit = (c: number) => {
      this.dv.setUint32(off, c, true);
      this.dv.setUint16(off + 4, this.o1[c], true);
      this.dv.setUint16(off + 6, this.o2[c], true);
      this.out[off + 8] = this.n1[c];
      this.out[off + 9] = this.n2[c];
      this.out[off + 10] = this.fo[c];
      off += CELL_BYTES;
    };
    if (all) {
      for (let c = 0; c < this.n; c++) {
        this.compute(c);
        emit(c);
      }
    } else {
      for (let k = 0; k < this.nDirty; k++) {
        const c = this.dirtyList[k];
        if (this.compute(c)) emit(c);
      }
    }
    for (let k = 0; k < this.nDirty; k++) this.dirty[this.dirtyList[k]] = 0;
    this.nDirty = 0;
    return this.out.slice(0, off);
  }

  /** Recomputes one cell from the tile states; returns whether it changed. Ties keep the first owner seen. */
  private compute(c: number): boolean {
    const x0 = (c % this.cw) * this.cs;
    const y0 = Math.floor(c / this.cw) * this.cs;
    const x1 = Math.min(x0 + this.cs, this.w);
    const y1 = Math.min(y0 + this.cs, this.h);
    let fallout = 0;
    for (let y = y0; y < y1; y++) {
      for (let i = y * this.w + x0, end = y * this.w + x1; i < end; i++) {
        const v = this.state[i];
        if (v & 0x2000) fallout++;
        const id = v & 0xfff;
        if (id !== 0 && this.cnt[id]++ === 0) this.touched.push(id);
      }
    }
    let o1 = 0;
    let o2 = 0;
    let n1 = 0;
    let n2 = 0;
    for (const id of this.touched) {
      const k = this.cnt[id];
      if (k > n1) {
        o2 = o1;
        n2 = n1;
        o1 = id;
        n1 = k;
      } else if (k > n2) {
        o2 = id;
        n2 = k;
      }
      this.cnt[id] = 0;
    }
    this.touched.length = 0;
    const changed =
      o1 !== this.o1[c] || o2 !== this.o2[c] || n1 !== this.n1[c] || n2 !== this.n2[c] || fallout !== this.fo[c];
    this.o1[c] = o1;
    this.o2[c] = o2;
    this.n1[c] = n1;
    this.n2[c] = n2;
    this.fo[c] = fallout;
    return changed;
  }
}

/** Static terrain per cell, 4 planes of frame.n bytes: land, shore (land next to water) and ocean tiles, and the mean
 *  height (magnitude 0-31) of the land. */
export function terrainCells(map: GameMap, frame: CellFrame): Uint8Array {
  const out = new Uint8Array(4 * frame.n);
  for (let c = 0; c < frame.n; c++) {
    let land = 0;
    let shore = 0;
    let ocean = 0;
    let height = 0;
    for (const t of frame.tilesOf(c)) {
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
  }
  return out;
}

/** All observed units as UNIT_BYTES records. */
export function encodeUnits(game: Game, frame: CellFrame): Uint8Array {
  const units = game.units();
  const out = new Uint8Array(units.length * UNIT_BYTES);
  const dv = new DataView(out.buffer);
  let off = 0;
  for (const u of units) {
    const type = UNIT_INDEX.get(u.type());
    if (type === undefined) continue;
    out[off] = type;
    out[off + 1] = u.isUnderConstruction() ? 0 : Math.min(u.level(), 255);
    dv.setUint16(off + 2, u.owner().smallID(), true);
    dv.setUint32(off + 4, frame.cellOf(u.tile()), true);
    dv.setFloat32(off + 8, u.troops(), true);
    off += UNIT_BYTES;
  }
  return out.subarray(0, off);
}
```

- [ ] **Step 4: Implement** — `bridge/game.ts` (creation and ticking; Task 5 adds decisions and observations)

```ts
import { Config } from "../OpenFront_Source_Code/openfrontio/src/core/configuration/Config";
import { Executor } from "../OpenFront_Source_Code/openfrontio/src/core/execution/ExecutionManager";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerType,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { createGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameImpl";
import { GameUpdateType, HashUpdate } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameUpdates";
import { createNationsForGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/NationCreation";
import { GameRunner } from "../OpenFront_Source_Code/openfrontio/src/core/GameRunner";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GameConfig, GameStartInfo, StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { CellFrame } from "./frame";
import { loadFreshTerrain, resetGameStatics } from "./maps";

/** A training game setup, sent by the trainer as JSON. FFA only in Phase 1. */
export interface EnvConfig {
  map: string; // a GameMapType value, e.g. "Onion"
  compact: boolean;
  nations: number; // 0 = no nations
  difficulty: string; // a Difficulty value, e.g. "Easy"
  bots: number;
  timerMinutes: number;
  seats: number; // agent-controlled players
}

/** Ticks per agent decision (PLAN decision 11). */
export const STEP_TICKS = 5;

export const seatClientID = (seat: number): string => `AGENT${seat.toString().padStart(3, "0")}`;

export class EnvGame {
  private turn = 0;

  private constructor(
    readonly cfg: EnvConfig,
    readonly game: Game,
    readonly seats: Player[],
    readonly frame: CellFrame,
    /** Rows of the player table: smallID 0 (unowned land) up to seats + nations + bots. */
    readonly slots: number,
    private readonly runner: GameRunner,
    private readonly sim: { hash: number; fatal: string | null },
  ) {}

  /** Latest state hash reported by the game (every 10 ticks). */
  get hash(): number {
    return this.sim.hash;
  }

  static async create(cfg: EnvConfig, seed: string): Promise<EnvGame> {
    resetGameStatics();
    const config: GameConfig = {
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
    const terrain = await loadFreshTerrain(config.gameMap, config.gameMapSize);
    const gameStart: GameStartInfo = { gameID: seed, lobbyCreatedAt: 0, config, players: [] };
    const random = new PseudoRandom(simpleHash(seed));
    const humans = Array.from(
      { length: cfg.seats },
      (_, i) => new PlayerInfo(`agent${i}`, PlayerType.Human, seatClientID(i), random.nextID()),
    );
    const nations = createNationsForGame(gameStart, terrain.nations, terrain.additionalNations, humans.length, random);
    const game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      new Config(config, null, false),
      terrain.teamGameSpawnAreas,
    );
    const frame = new CellFrame(game.map(), cfg.compact ? 4 : 8);
    const sim = { hash: 0, fatal: null as string | null };
    const runner = new GameRunner(game, new Executor(game, seed, undefined), (gu) => {
      if ("errMsg" in gu) {
        sim.fatal = gu.errMsg;
        return;
      }
      frame.mark(gu.packedTileUpdates);
      for (const h of gu.updates[GameUpdateType.Hash] as HashUpdate[]) sim.hash = h.hash;
    });
    runner.init();
    const seats = humans.map((h) => game.player(h.id));
    return new EnvGame(cfg, game, seats, frame, 1 + cfg.seats + nations.length + cfg.bots, runner, sim);
  }

  /** Executes one turn (one tick) with the given intents. */
  tickWith(intents: StampedIntent[]): void {
    this.runner.addTurn({ turnNumber: this.turn++, intents });
    if (!this.runner.executeNextTick()) throw new Error(`tick ${this.turn - 1} failed: ${this.sim.fatal}`);
  }
}
```

- [ ] **Step 5: Run the tests and the typecheck**

Run: `node --import tsx --test ../../bridge/game.test.ts`
Expected: PASS (3 tests, ~2 s).
Run: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json` → exit 0.

- [ ] **Step 6: Commit**

```bash
git add bridge/frame.ts bridge/game.ts bridge/testkit.ts bridge/game.test.ts
git commit -m "Bridge game creation, incremental cell frame and determinism test"
```

---

### Task 4: Actions, masks and the per-seat observation

**Files:**
- Create: `bridge/actions.ts`, `bridge/actions.test.ts`, `bridge/obs.ts`, `bridge/obs.test.ts`

**Interfaces:**
- Consumes: `EnvGame`, `CellFrame`, testkit (Task 3)
- Produces: `enum Act { Noop, Spawn, Attack, Boat, BuildCity }`, `N_ACTS = 5`, `TROOP_BINS = [0.1, 0.25, 0.5, 0.75, 1]`,
  `attackMask(game, seat, slots): Uint8Array`, `typeMask(game, seat, attack): Uint8Array`,
  `decodeAction(game, seat, frame, types, attack, action: number[]): { intent: Intent | null; valid: boolean }`;
  `PLAYER_FEATURES = 17`, `SCALARS = 16`, `enum Rel { None, Self, Friendly, Enemy, Bot }`,
  `encodeSeat(game, seat, cfg, cellSize, attack, players, rel, scalars): void`.
  Player feature 16 = log troop density (the trainer's DENSITY channel reads it); scalar 0 = game progress.

- [ ] **Step 1: Write the failing tests**

`bridge/actions.test.ts`:
```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { Act, attackMask, decodeAction, typeMask } from "./actions";
import { EnvGame } from "./game";
import { landTile, ONION, script, spawnCell } from "./testkit";

const ticks = (env: EnvGame, n: number) => {
  for (let i = 0; i < n; i++) env.tickWith([]);
};

test("spawn is open early, forced late and closed at the very end of the spawn phase", async () => {
  const env = await EnvGame.create(ONION, "masks");
  const seat = env.seats[0];
  const types = () => [...typeMask(env.game, seat, attackMask(env.game, seat, env.slots))];
  ticks(env, 100);
  assert.deepEqual(types(), [1, 1, 0, 0, 0]);
  ticks(env, 85); // tick 185
  assert.deepEqual(types(), [0, 1, 0, 0, 0]);
  ticks(env, 12); // tick 197
  assert.deepEqual(types(), [1, 0, 0, 0, 0]);
});

test("spawn decodes to unowned land inside the chosen cell", async () => {
  const env = await EnvGame.create(ONION, "spawn");
  const seat = env.seats[0];
  const cell = spawnCell(env);
  const types = typeMask(env.game, seat, attackMask(env.game, seat, env.slots));
  const { intent, valid } = decodeAction(env.game, seat, env.frame, types, new Uint8Array(env.slots), [Act.Spawn, cell, 0, 0]);
  assert.ok(valid && intent !== null && intent.type === "spawn");
  if (intent?.type !== "spawn") return;
  assert.equal(env.frame.cellOf(intent.tile), cell);
  assert.ok(env.game.isLand(intent.tile) && !env.game.hasOwner(intent.tile));
});

test("after spawning, unowned land is attackable and the troop bin sets the troops", async () => {
  const env = await EnvGame.create(ONION, "attack");
  script(env, 260);
  const seat = env.seats[0];
  const attack = attackMask(env.game, seat, env.slots);
  assert.equal(attack[0], 1);
  const types = typeMask(env.game, seat, attack);
  assert.equal(types[Act.Attack], 1);
  const { intent, valid } = decodeAction(env.game, seat, env.frame, types, attack, [Act.Attack, 0, 0, 2]);
  assert.ok(valid);
  assert.deepEqual(intent, { type: "attack", targetID: null, troops: 0.5 * seat.troops() });
  const far = [...attack.keys()].find((s) => s > 1 && attack[s] === 0);
  if (far !== undefined) {
    assert.equal(decodeAction(env.game, seat, env.frame, types, attack, [Act.Attack, 0, far, 0]).valid, false);
  }
});

test("build city: valid on own land with gold, dropped elsewhere", async () => {
  const env = await EnvGame.create(ONION, "city");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(10_000_000n);
  const attack = attackMask(env.game, seat, env.slots);
  const types = typeMask(env.game, seat, attack);
  assert.equal(types[Act.BuildCity], 1);
  const own = env.frame.cellOf(landTile(env));
  const ok = decodeAction(env.game, seat, env.frame, types, attack, [Act.BuildCity, own, 0, 0]);
  assert.equal(ok.valid, true);
  assert.equal(ok.intent?.type === "build_unit" && ok.intent.unit, UnitType.City);
  assert.equal(decodeAction(env.game, seat, env.frame, types, attack, [Act.BuildCity, 0, 0, 0]).valid, false);
});

test("malformed or unusable actions are dropped, never thrown", async () => {
  const env = await EnvGame.create(ONION, "robust");
  script(env, 260);
  const seat = env.seats[0];
  const attack = attackMask(env.game, seat, env.slots);
  const types = typeMask(env.game, seat, attack);
  const bad = [[9, 0, 0, 0], [Act.Attack, 0, 99, 0], [Act.Attack, 0, 0, 7], [Act.Spawn, 0, 0, 0], [Act.BuildCity, -1, 0, 0], [Act.BuildCity, 1e9, 0, 0], []];
  for (const a of bad) assert.equal(decodeAction(env.game, seat, env.frame, types, attack, a).valid, false, `${a}`);
  for (let c = 0; c < env.frame.n; c += 7) decodeAction(env.game, seat, env.frame, types, attack, [Act.Boat, c, 0, 1]);
});
```

`bridge/obs.test.ts`:
```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { attackMask } from "./actions";
import { EnvGame } from "./game";
import { encodeSeat, PLAYER_FEATURES, Rel, SCALARS } from "./obs";
import { ONION, script } from "./testkit";

test("seat view: self, enemies, unowned land row and bounded scalars", async () => {
  const env = await EnvGame.create(ONION, "obs");
  script(env, 260);
  const seat = env.seats[0];
  const P = env.slots;
  const players = new Float32Array(P * PLAYER_FEATURES);
  const rel = new Uint8Array(P);
  const scalars = new Float32Array(SCALARS);
  const attack = attackMask(env.game, seat, P);
  encodeSeat(env.game, seat, env.cfg, env.frame.cs, attack, players, rel, scalars);
  const me = seat.smallID();
  assert.equal(rel[me], Rel.Self);
  assert.deepEqual([...rel].filter((r) => r === Rel.Enemy).length, 3);
  assert.equal(players[me * PLAYER_FEATURES + 2], 1);
  assert.ok(players[7] > 0 && players[7] <= 1, "unowned land share");
  assert.equal(players[12], attack[0]);
  assert.equal(scalars[3], 1, "spawned");
  for (const v of [...players, ...scalars]) assert.ok(Number.isFinite(v) && v >= 0 && v <= 2, `${v}`);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --import tsx --test ../../bridge/actions.test.ts ../../bridge/obs.test.ts`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement** — `bridge/actions.ts`

```ts
import { Game, Player, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { TileRef } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameMap";
import { canBuildTransportShip } from "../OpenFront_Source_Code/openfrontio/src/core/game/TransportShipUtils";
import { Intent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { CellFrame } from "./frame";

/** Phase 1 action types (PLAN "Phase 1": no-op, spawn, attack, boat, build city). */
export enum Act {
  Noop = 0,
  Spawn = 1,
  Attack = 2,
  Boat = 3,
  BuildCity = 4,
}
export const N_ACTS = 5;
/** Troops sent by attack and boat actions, as fractions of the seat's current troops. */
export const TROOP_BINS = [0.1, 0.25, 0.5, 0.75, 1.0];
/** A spawn intent must still execute inside the spawn phase after up to 3 turns of latency. */
const SPAWN_MARGIN = 4;
/** Decisions in the last ticks of the spawn phase must spawn if the seat has not spawned yet. */
const FORCE_SPAWN = 15;

/** Which table rows the seat can attack over land now: row = smallID, row 0 = unowned land. */
export function attackMask(game: Game, seat: Player, slots: number): Uint8Array {
  const mask = new Uint8Array(slots);
  if (game.inSpawnPhase() || !seat.isAlive()) return mask;
  const map = game.map();
  const me = seat.smallID();
  const nbuf = [0, 0, 0, 0];
  for (const border of seat.borderTiles()) {
    const k = map.neighbors4(border, nbuf);
    for (let i = 0; i < k; i++) {
      const t = nbuf[i];
      if (!map.isLand(t) || map.isImpassable(t)) continue;
      const owner = map.ownerID(t);
      if (owner === me || owner >= slots || mask[owner] === 1) continue;
      if (owner === 0) {
        if (!map.hasFallout(t)) mask[0] = 1;
        continue;
      }
      const other = game.playerBySmallID(owner) as Player;
      if (!seat.isFriendly(other) && seat.canAttackPlayer(other)) mask[owner] = 1;
    }
  }
  return mask;
}

/** Which action types are legal now (1 = allowed). */
export function typeMask(game: Game, seat: Player, attack: Uint8Array): Uint8Array {
  const m = new Uint8Array(N_ACTS);
  if (game.inSpawnPhase()) {
    const end = game.config().numSpawnPhaseTurns();
    const open = game.ticks() <= end - SPAWN_MARGIN;
    const forced = open && !seat.hasSpawned() && game.ticks() >= end - FORCE_SPAWN;
    m[Act.Noop] = forced ? 0 : 1;
    m[Act.Spawn] = open ? 1 : 0;
    return m;
  }
  m[Act.Noop] = 1;
  if (!seat.isAlive()) return m;
  m[Act.Attack] = attack.includes(1) ? 1 : 0;
  m[Act.Boat] = seat.unitCount(UnitType.TransportShip) < game.config().boatMaxNumber() ? 1 : 0;
  m[Act.BuildCity] = seat.gold() >= game.unitInfo(UnitType.City).cost(game, seat) ? 1 : 0;
  return m;
}

export interface Decoded {
  intent: Intent | null;
  /** False when the action could not be applied and was dropped (the trainer's cell masks are approximate). */
  valid: boolean;
}
const DROP: Decoded = { intent: null, valid: false };

/** Turns a sampled [type, cell, target, bin] into a game intent (null intent = no-op). */
export function decodeAction(
  game: Game,
  seat: Player,
  frame: CellFrame,
  types: Uint8Array,
  attack: Uint8Array,
  action: number[],
): Decoded {
  const [type, cell, target, bin] = action;
  if (!Number.isInteger(type) || types[type] !== 1) return DROP;
  if (type === Act.Noop) return { intent: null, valid: true };
  if (type === Act.Attack) {
    if (!Number.isInteger(target) || attack[target] !== 1 || !(bin in TROOP_BINS)) return DROP;
    const targetID = target === 0 ? null : (game.playerBySmallID(target) as Player).id();
    return { intent: { type: "attack", targetID, troops: TROOP_BINS[bin] * seat.troops() }, valid: true };
  }
  if (!Number.isInteger(cell) || cell < 0 || cell >= frame.n) return DROP;
  if (type === Act.Spawn) {
    const tile = spawnTile(game, frame, cell);
    return tile === null ? DROP : { intent: { type: "spawn", tile }, valid: true };
  }
  if (type === Act.Boat) {
    if (!(bin in TROOP_BINS)) return DROP;
    for (const tile of frame.tilesOf(cell)) {
      if (!game.isLand(tile) || game.isImpassable(tile) || !game.isShore(tile)) continue;
      if (game.ownerID(tile) === seat.smallID()) continue;
      if (canBuildTransportShip(game, seat, tile) !== false) {
        return { intent: { type: "boat", dst: tile, troops: TROOP_BINS[bin] * seat.troops() }, valid: true };
      }
    }
    return DROP;
  }
  if (type === Act.BuildCity) {
    for (const tile of frame.tilesOf(cell)) {
      if (game.ownerID(tile) === seat.smallID() && seat.canBuild(UnitType.City, tile) !== false) {
        return { intent: { type: "build_unit", unit: UnitType.City, tile }, valid: true };
      }
    }
  }
  return DROP;
}

/** The unowned land tile of the cell closest to its centre. */
function spawnTile(game: Game, frame: CellFrame, cell: number): TileRef | null {
  const [cx, cy] = frame.center(cell);
  let best: TileRef | null = null;
  let bestDist = Infinity;
  for (const t of frame.tilesOf(cell)) {
    if (!game.isLand(t) || game.isImpassable(t) || game.hasOwner(t)) continue;
    const dx = game.x(t) - cx;
    const dy = game.y(t) - cy;
    if (dx * dx + dy * dy < bestDist) {
      bestDist = dx * dx + dy * dy;
      best = t;
    }
  }
  return best;
}
```

- [ ] **Step 4: Implement** — `bridge/obs.ts`

```ts
import { Game, Player, PlayerType, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import type { EnvConfig } from "./game";

export const PLAYER_FEATURES = 17;
export const SCALARS = 16;

/** How a player-table row relates to the seat; the trainer maps cells and units through it. */
export enum Rel {
  None = 0,
  Self = 1,
  Friendly = 2,
  Enemy = 3,
  Bot = 4,
}

const logScale = (v: number, div: number) => Math.log1p(Math.max(0, v)) / div;

/**
 * Fills the seat's view of the player table (row = smallID, row 0 = unowned land) and its scalars. The arrays must be
 * zeroed: players [slots * PLAYER_FEATURES], rel [slots], scalars [SCALARS].
 *
 * Player features: 0 exists, 1 alive, 2 self, 3 friendly, 4 enemy, 5 bot, 6 nation, 7 tiles / land, 8 log troops,
 * 9 troops / max troops, 10 log gold, 11 traitor, 12 attackable by the seat, 13 log troops attacking the seat,
 * 14 log troops the seat sends at it, 15 spawned, 16 log troops per tile.
 * Scalars: 0 game progress, 1 spawn phase, 2 spawn-phase progress, 3 spawned, 4 alive, 5 log troops, 6 troops / max,
 * 7 log gold, 8 gold / city cost, 9 tiles / land, 10 boats / max boats, 11 alive players share, 12 time left,
 * 13 spawn immunity, 14 tiles per cell / 64, 15 share of players ahead in tiles.
 */
export function encodeSeat(
  game: Game,
  seat: Player,
  cfg: EnvConfig,
  cellSize: number,
  attack: Uint8Array,
  players: Float32Array,
  rel: Uint8Array,
  scalars: Float32Array,
): void {
  const F = PLAYER_FEATURES;
  const slots = rel.length;
  const land = game.numLandTiles();
  const config = game.config();
  const incoming = new Float64Array(slots);
  const outgoing = new Float64Array(slots);
  for (const a of seat.incomingAttacks()) incoming[a.attacker().smallID()] += a.troops();
  for (const a of seat.outgoingAttacks()) outgoing[a.target().smallID()] += a.troops();

  let owned = 0;
  for (const p of game.allPlayers()) {
    const s = p.smallID();
    if (s >= slots) throw new Error(`player smallID ${s} does not fit ${slots} table rows`);
    const tiles = p.numTilesOwned();
    const troops = p.troops();
    owned += tiles;
    const self = p === seat;
    const bot = p.type() === PlayerType.Bot;
    const friendly = !self && seat.isFriendly(p);
    rel[s] = self ? Rel.Self : bot ? Rel.Bot : friendly ? Rel.Friendly : Rel.Enemy;
    const o = s * F;
    players[o] = 1;
    players[o + 1] = p.isAlive() ? 1 : 0;
    players[o + 2] = self ? 1 : 0;
    players[o + 3] = friendly ? 1 : 0;
    players[o + 4] = !self && !friendly && !bot ? 1 : 0;
    players[o + 5] = bot ? 1 : 0;
    players[o + 6] = p.type() === PlayerType.Nation ? 1 : 0;
    players[o + 7] = tiles / land;
    players[o + 8] = logScale(troops, 15);
    players[o + 9] = Math.min(troops / Math.max(config.maxTroops(p), 1), 2) / 2;
    players[o + 10] = logScale(Number(p.gold()), 20);
    players[o + 11] = p.isTraitor() ? 1 : 0;
    players[o + 12] = attack[s];
    players[o + 13] = logScale(incoming[s], 15);
    players[o + 14] = logScale(outgoing[s], 15);
    players[o + 15] = p.hasSpawned() ? 1 : 0;
    players[o + 16] = logScale(troops / Math.max(tiles, 1), 5);
  }
  rel[0] = Rel.None;
  players[0] = 1;
  players[1] = 1;
  players[7] = Math.max(0, land - owned) / land;
  players[12] = attack[0];
  players[14] = logScale(outgoing[0], 15);

  const spawnTurns = config.numSpawnPhaseTurns();
  const nonBot = game.allPlayers().filter((p) => p.type() !== PlayerType.Bot);
  const alive = nonBot.filter((p) => p.isAlive());
  const myTiles = seat.numTilesOwned();
  const troops = seat.troops();
  const gold = Number(seat.gold());
  const cityCost = Number(game.unitInfo(UnitType.City).cost(game, seat));
  const ahead = alive.filter((p) => p !== seat && p.numTilesOwned() > myTiles).length;
  scalars[0] = Math.min(game.ticks() / (spawnTurns + cfg.timerMinutes * 600), 1);
  scalars[1] = game.inSpawnPhase() ? 1 : 0;
  scalars[2] = Math.min(game.ticks() / spawnTurns, 1);
  scalars[3] = seat.hasSpawned() ? 1 : 0;
  scalars[4] = seat.isAlive() ? 1 : 0;
  scalars[5] = logScale(troops, 15);
  scalars[6] = Math.min(troops / Math.max(config.maxTroops(seat), 1), 2) / 2;
  scalars[7] = logScale(gold, 20);
  scalars[8] = Math.min(gold / Math.max(cityCost, 1), 4) / 4;
  scalars[9] = myTiles / land;
  scalars[10] = seat.unitCount(UnitType.TransportShip) / Math.max(config.boatMaxNumber(), 1);
  scalars[11] = alive.length / Math.max(nonBot.length, 1);
  scalars[12] = Math.min(1, Math.max(0, 1 - game.elapsedGameSeconds() / (cfg.timerMinutes * 60)));
  scalars[13] = game.isSpawnImmunityActive() ? 1 : 0;
  scalars[14] = (cellSize * cellSize) / 64;
  scalars[15] = seat.isAlive() ? ahead / Math.max(nonBot.length - 1, 1) : 1;
}
```

- [ ] **Step 5: Run the tests and the typecheck**

Run: `node --import tsx --test ../../bridge/actions.test.ts ../../bridge/obs.test.ts`
Expected: PASS (6 tests). If `build city: valid` fails because the tile next to the spawn is too close to the map edge
for a city, pick `landTile`'s cell anyway (it is inside the seat's spawn area) and re-check `seat.canBuild` directly to
see the game's reason before changing the test.
Run: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json` → exit 0.

- [ ] **Step 6: Commit**

```bash
git add bridge/actions.ts bridge/actions.test.ts bridge/obs.ts bridge/obs.test.ts
git commit -m "Bridge action decoding, masks and per-seat observation"
```

---

### Task 5: EnvGame decisions, observations and game end

**Files:**
- Modify: `bridge/game.ts` (replace with the full version below)
- Modify: `bridge/game.test.ts` (append tests)

**Interfaces:**
- Consumes: Tasks 2–4
- Produces: `type Blob = [name: string, bytes: Uint8Array]`; `EnvGame` gains `done: boolean`, `rewards: number[]`,
  `ranks: number[]`, `nPlayers: number`, `invalid: boolean[]`, `step(actions: number[][]): void`,
  `observe(all: boolean): Blob[]` (names in order: cells, units, players, rel, scalars, types, attack),
  `terrain(): Uint8Array`

- [ ] **Step 1: Append the failing tests** to `bridge/game.test.ts`

```ts
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { Act } from "./actions";
import { rewardOf } from "./ranking";
import { spawnCell } from "./testkit";

const blob = (blobs: [string, Uint8Array][], name: string) => blobs.find(([n]) => n === name)![1];

test("random legal play reaches the end with a valid terminal reward", async () => {
  const env = await EnvGame.create({ ...ONION, timerMinutes: 1 }, "random-play");
  const rnd = new PseudoRandom(5);
  const pick = (mask: Uint8Array) => {
    const legal = [...mask.keys()].filter((i) => mask[i] === 1);
    return legal[rnd.nextInt(0, legal.length)];
  };
  let steps = 0;
  while (!env.done && steps < 400) {
    const blobs = env.observe(steps === 0);
    const type = pick(blob(blobs, "types"));
    const cell = type === Act.Spawn ? spawnCell(env) : rnd.nextInt(0, env.frame.n);
    const target = type === Act.Attack ? pick(blob(blobs, "attack")) : 0;
    env.step([[type, cell, target, rnd.nextInt(0, 5)]]);
    steps++;
  }
  assert.ok(env.done, "game ended");
  const [rank] = env.ranks;
  assert.ok(rank >= 1 && rank <= 4);
  assert.equal(env.nPlayers, 4);
  assert.equal(env.rewards[0], rewardOf(rank, 4));
});

test("a seat that never spawns ends the game last as soon as the spawn phase is over", async () => {
  const env = await EnvGame.create(ONION, "never-spawn");
  let steps = 0;
  while (!env.done && steps < 100) {
    env.observe(false);
    env.step([[Act.Noop, 0, 0, 0]]);
    steps++;
  }
  assert.ok(env.done && steps <= 41, `steps ${steps}`);
  assert.deepEqual(env.ranks, [4]);
  assert.deepEqual(env.rewards, [-1]);
});

test("observe returns the blobs in protocol order with consistent sizes", async () => {
  const env = await EnvGame.create(ONION, "blobs");
  const blobs = env.observe(true);
  assert.deepEqual(blobs.map(([n]) => n), ["cells", "units", "players", "rel", "scalars", "types", "attack"]);
  assert.equal(blob(blobs, "cells").byteLength, env.frame.n * 11);
  assert.equal(blob(blobs, "players").byteLength, env.slots * 17 * 4);
  assert.equal(env.terrain().byteLength, 4 * env.frame.n);
  env.step([[Act.Noop, 0, 0, 0]]);
  assert.ok(env.observe(false)[0][1].byteLength < env.frame.n * 11);
});
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `node --import tsx --test ../../bridge/game.test.ts`
Expected: the 3 new tests FAIL (`env.observe is not a function`), the 3 old ones pass.

- [ ] **Step 3: Replace `bridge/game.ts`** with the full version

```ts
import { Config } from "../OpenFront_Source_Code/openfrontio/src/core/configuration/Config";
import { Executor } from "../OpenFront_Source_Code/openfrontio/src/core/execution/ExecutionManager";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerType,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { createGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameImpl";
import { GameUpdateType, HashUpdate } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameUpdates";
import { createNationsForGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/NationCreation";
import { GameRunner } from "../OpenFront_Source_Code/openfrontio/src/core/GameRunner";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GameConfig, GameStartInfo, StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { attackMask, decodeAction, N_ACTS, typeMask } from "./actions";
import { CellFrame, encodeUnits, terrainCells } from "./frame";
import { LatencyQueue } from "./latency";
import { loadFreshTerrain, resetGameStatics } from "./maps";
import { encodeSeat, PLAYER_FEATURES, SCALARS } from "./obs";
import { rewardOf, seatStanding } from "./ranking";

/** A training game setup, sent by the trainer as JSON. FFA only in Phase 1. */
export interface EnvConfig {
  map: string; // a GameMapType value, e.g. "Onion"
  compact: boolean;
  nations: number; // 0 = no nations
  difficulty: string; // a Difficulty value, e.g. "Easy"
  bots: number;
  timerMinutes: number;
  seats: number; // agent-controlled players
}

/** Ticks per agent decision (PLAN decision 11). */
export const STEP_TICKS = 5;

export const seatClientID = (seat: number): string => `AGENT${seat.toString().padStart(3, "0")}`;

export type Blob = [name: string, bytes: Uint8Array];
const bytes = (a: Float32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

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
    readonly cfg: EnvConfig,
    readonly game: Game,
    readonly seats: Player[],
    readonly frame: CellFrame,
    /** Rows of the player table: smallID 0 (unowned land) up to seats + nations + bots. */
    readonly slots: number,
    private readonly runner: GameRunner,
    private readonly sim: { hash: number; fatal: string | null },
    private readonly latency: LatencyQueue,
  ) {
    this.invalid = seats.map(() => false);
  }

  /** Latest state hash reported by the game (every 10 ticks). */
  get hash(): number {
    return this.sim.hash;
  }

  static async create(cfg: EnvConfig, seed: string): Promise<EnvGame> {
    resetGameStatics();
    const config: GameConfig = {
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
    const terrain = await loadFreshTerrain(config.gameMap, config.gameMapSize);
    const gameStart: GameStartInfo = { gameID: seed, lobbyCreatedAt: 0, config, players: [] };
    const random = new PseudoRandom(simpleHash(seed));
    const humans = Array.from(
      { length: cfg.seats },
      (_, i) => new PlayerInfo(`agent${i}`, PlayerType.Human, seatClientID(i), random.nextID()),
    );
    const nations = createNationsForGame(gameStart, terrain.nations, terrain.additionalNations, humans.length, random);
    const game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      new Config(config, null, false),
      terrain.teamGameSpawnAreas,
    );
    const frame = new CellFrame(game.map(), cfg.compact ? 4 : 8);
    const sim = { hash: 0, fatal: null as string | null };
    const runner = new GameRunner(game, new Executor(game, seed, undefined), (gu) => {
      if ("errMsg" in gu) {
        sim.fatal = gu.errMsg;
        return;
      }
      frame.mark(gu.packedTileUpdates);
      for (const h of gu.updates[GameUpdateType.Hash] as HashUpdate[]) sim.hash = h.hash;
    });
    runner.init();
    const seats = humans.map((h) => game.player(h.id));
    const latency = new LatencyQueue(new PseudoRandom(simpleHash(`${seed}:latency`)));
    return new EnvGame(cfg, game, seats, frame, 1 + cfg.seats + nations.length + cfg.bots, runner, sim, latency);
  }

  /** Executes one turn (one tick) with the given intents. */
  tickWith(intents: StampedIntent[]): void {
    this.runner.addTurn({ turnNumber: this.turn++, intents });
    if (!this.runner.executeNextTick()) throw new Error(`tick ${this.turn - 1} failed: ${this.sim.fatal}`);
  }

  /** Applies one decision per seat ([type, cell, target, bin]) through the latency queue and advances STEP_TICKS. */
  step(actions: number[][]): void {
    if (this.done) throw new Error("step() after the game ended");
    const masks = this.masks ?? this.currentMasks();
    this.masks = null;
    this.invalid = this.seats.map((seat, i) => {
      const { intent, valid } = decodeAction(
        this.game,
        seat,
        this.frame,
        masks[i].types,
        masks[i].attack,
        actions[i] ?? [],
      );
      if (intent !== null) this.latency.push({ ...intent, clientID: seatClientID(i) } as StampedIntent, this.turn);
      return !valid;
    });
    for (let k = 0; k < STEP_TICKS; k++) this.tickWith(this.latency.take(this.turn));
    this.checkDone();
  }

  /** The observation, as blobs in protocol order. `all`: every cell instead of only the changed ones. */
  observe(all: boolean): Blob[] {
    const S = this.seats.length;
    const P = this.slots;
    const players = new Float32Array(S * P * PLAYER_FEATURES);
    const rel = new Uint8Array(S * P);
    const scalars = new Float32Array(S * SCALARS);
    const types = new Uint8Array(S * N_ACTS);
    const attack = new Uint8Array(S * P);
    this.masks = this.currentMasks();
    this.seats.forEach((seat, i) => {
      const m = this.masks![i];
      types.set(m.types, i * N_ACTS);
      attack.set(m.attack, i * P);
      encodeSeat(
        this.game,
        seat,
        this.cfg,
        this.frame.cs,
        m.attack,
        players.subarray(i * P * PLAYER_FEATURES, (i + 1) * P * PLAYER_FEATURES),
        rel.subarray(i * P, (i + 1) * P),
        scalars.subarray(i * SCALARS, (i + 1) * SCALARS),
      );
    });
    return [
      ["cells", this.frame.encode(all)],
      ["units", encodeUnits(this.game, this.frame)],
      ["players", bytes(players)],
      ["rel", rel],
      ["scalars", bytes(scalars)],
      ["types", types],
      ["attack", attack],
    ];
  }

  terrain(): Uint8Array {
    return terrainCells(this.game.map(), this.frame);
  }

  private currentMasks(): { types: Uint8Array; attack: Uint8Array }[] {
    return this.seats.map((seat) => {
      const attack = attackMask(this.game, seat, this.slots);
      return { types: typeMask(this.game, seat, attack), attack };
    });
  }

  /** The game ends at a winner, when every seat is dead or never spawned, or 1 min after the timer as a backstop. */
  private checkDone(): void {
    const g = this.game;
    const out = (s: Player) => !g.inSpawnPhase() && (!s.hasSpawned() || !s.isAlive());
    const backstop = g.ticks() >= g.config().numSpawnPhaseTurns() + (this.cfg.timerMinutes + 1) * 600;
    if (g.getWinner() === null && !this.seats.every(out) && !backstop) return;
    this.done = true;
    const standings = this.seats.map((s) => seatStanding(g, s));
    this.nPlayers = standings[0].n;
    this.ranks = standings.map((s) => s.rank);
    this.rewards = standings.map((s) => rewardOf(s.rank, s.n));
  }
}
```

- [ ] **Step 4: Run all bridge tests and the typecheck**

Run: `node --import tsx --test "../../bridge/*.test.ts"`
Expected: PASS (all tests in maps, latency, ranking, game, actions, obs).
Run: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json` → exit 0.

- [ ] **Step 5: Commit**

```bash
git add bridge/game.ts bridge/game.test.ts
git commit -m "EnvGame decisions, observations and game end"
```

---

### Task 6: Worker process and protocol

**Files:**
- Create: `bridge/worker.ts`, `bridge/worker.test.ts`

**Interfaces:**
- Consumes: `EnvGame`, `N_ACTS`, `TROOP_BINS`, `UNIT_TYPES`, `PLAYER_FEATURES`, `SCALARS`
- Produces: the protocol in "File structure" above. `dims = { grid: [ch, cw], cellSize, slots, seats,
  playerFeatures, scalars, acts, troopBins, unitTypes }`.

- [ ] **Step 1: Write the failing test** — `bridge/worker.test.ts`

```ts
import assert from "node:assert/strict";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Act } from "./actions";
import { EnvGame } from "./game";
import { GAME_ROOT } from "./maps";
import { ONION, spawnCell } from "./testkit";

const WORKER = path.join(path.dirname(fileURLToPath(import.meta.url)), "worker.ts");
type Reply = { header: any; body: Buffer };

class Client {
  private buf = Buffer.alloc(0);
  private waiting: ((m: Buffer) => void)[] = [];
  readonly proc: ChildProcessWithoutNullStreams = spawn(process.execPath, ["--import", "tsx", WORKER], {
    cwd: GAME_ROOT,
  });

  constructor() {
    this.proc.stdout.on("data", (chunk: Buffer) => {
      this.buf = Buffer.concat([this.buf, chunk]);
      this.drain();
    });
  }

  private drain(): void {
    while (this.waiting.length > 0 && this.buf.length >= 4 && this.buf.length >= 4 + this.buf.readUInt32LE(0)) {
      const n = this.buf.readUInt32LE(0);
      const msg = this.buf.subarray(4, 4 + n);
      this.buf = this.buf.subarray(4 + n);
      this.waiting.shift()!(msg);
    }
  }

  request(msg: object): Promise<Reply> {
    const data = Buffer.from(JSON.stringify(msg));
    const len = Buffer.alloc(4);
    len.writeUInt32LE(data.length);
    this.proc.stdin.write(Buffer.concat([len, data]));
    return new Promise((resolve) => {
      this.waiting.push((m) => {
        const h = m.readUInt32LE(0);
        resolve({ header: JSON.parse(m.subarray(4, 4 + h).toString("utf8")), body: m.subarray(4 + h) });
      });
      this.drain();
    });
  }

  close(): void {
    const data = Buffer.from(JSON.stringify({ cmd: "close" }));
    const len = Buffer.alloc(4);
    len.writeUInt32LE(data.length);
    this.proc.stdin.end(Buffer.concat([len, data]));
  }
}

test("init returns dims and a full first frame per game", async () => {
  const c = new Client();
  const { header, body } = await c.request({ cmd: "init", config: ONION, games: 2, seed: "t", worker: 0 });
  c.close();
  assert.deepEqual(header.dims.grid, [64, 64]);
  assert.equal(header.dims.slots, 5);
  assert.equal(header.games.length, 2);
  const g = header.games[0];
  assert.equal(g.reset, true);
  assert.deepEqual(g.blobs.map((b: [string, number]) => b[0]), ["terrain", "cells", "units", "players", "rel", "scalars", "types", "attack"]);
  assert.equal(g.blobs[1][1], 64 * 64 * 11);
  const total = header.games.flatMap((x: any) => x.blobs).reduce((s: number, b: [string, number]) => s + b[1], 0);
  assert.equal(body.length, total);
});

test("same seed and actions give the same hash in a fresh worker and after other games", async () => {
  const cell = spawnCell(await EnvGame.create(ONION, "cell"));
  const play = async (c: Client, seed: string): Promise<number> => {
    await c.request({ cmd: "init", config: ONION, games: 1, seed, worker: 0 });
    let reply: Reply | null = null;
    for (let step = 0; step < 200; step++) {
      const action = step === 2 ? [Act.Spawn, cell, 0, 0] : step > 45 && step % 10 === 0 ? [Act.Attack, 0, 0, 2] : [Act.Noop, 0, 0, 0];
      reply = await c.request({ cmd: "step", actions: [[action]] });
    }
    return reply!.header.games[0].hash;
  };
  const fresh = new Client();
  const first = await play(fresh, "det");
  fresh.close();
  const reused = new Client();
  await play(reused, "other");
  const second = await play(reused, "det");
  reused.close();
  assert.notEqual(first, 0);
  assert.equal(second, first);
});

test("a finished game is replaced by the next episode in the same reply", async () => {
  const c = new Client();
  await c.request({ cmd: "init", config: { ...ONION, timerMinutes: 1 }, games: 1, seed: "end", worker: 3 });
  let g: any = null;
  for (let step = 0; step < 60; step++) {
    g = (await c.request({ cmd: "step", actions: [[[Act.Noop, 0, 0, 0]]] })).header.games[0];
    if (g.done) break;
  }
  c.close();
  assert.equal(g.done, true);
  assert.equal(g.reset, true);
  assert.equal(g.episode, 1);
  assert.deepEqual(g.ranks, [4]);
  assert.deepEqual(g.rewards, [-1]);
  assert.equal(g.blobs[0][0], "terrain");
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `node --import tsx --test ../../bridge/worker.test.ts`
Expected: FAIL (the worker file does not exist; the child exits and the first request never resolves → the test
times out or errors).

- [ ] **Step 3: Implement** — `bridge/worker.ts`

```ts
/**
 * Env worker: runs K training games for the Python trainer (trainer/env.py) over stdin/stdout.
 *
 * In:  u32 length + JSON: {cmd:"init", config, games, seed, worker, episodes?} | {cmd:"step", actions, full?} |
 *      {cmd:"close"}. "init" may be sent again to restart all games.
 * Out: u32 size + u32 header length + header JSON + blobs. The header lists each game's blobs as [name, byteLength]
 *      in the order they follow. A finished game is replaced at once by the next episode: its reply has done and
 *      reset set, the finished game's rewards and ranks, and the new game's terrain and full frame.
 * Game seeds are `${seed}-w${worker}-g${game}-e${episode}`.
 */
import { N_ACTS, TROOP_BINS } from "./actions";
import { UNIT_TYPES } from "./frame";
import { Blob, EnvConfig, EnvGame } from "./game";
import { PLAYER_FEATURES, SCALARS } from "./obs";

// stdout carries replies only; the game logs a lot through console.log/info/warn.
console.log = console.info = console.warn = console.debug = () => {};

interface Init {
  cmd: "init";
  config: EnvConfig;
  games: number;
  seed: string;
  worker: number;
  episodes?: number[];
}
interface Step {
  cmd: "step";
  actions: number[][][];
  full?: boolean;
}
interface Part {
  header: Record<string, unknown>;
  blobs: Blob[];
}

let init: Init | null = null;
let games: EnvGame[] = [];
let episodes: number[] = [];

const seedOf = (g: number): string => `${init!.seed}-w${init!.worker}-g${g}-e${episodes[g]}`;

function part(env: EnvGame, g: number, blobs: Blob[], extra: Record<string, unknown>): Part {
  return {
    header: { tick: env.game.ticks(), episode: episodes[g], done: false, reset: false, hash: env.hash, invalid: env.invalid, ...extra },
    blobs,
  };
}

const fresh = (g: number, extra: Record<string, unknown>): Part =>
  part(games[g], g, [["terrain", games[g].terrain()], ...games[g].observe(true)], { reset: true, ...extra });

function dims(env: EnvGame): Record<string, unknown> {
  return {
    grid: [env.frame.ch, env.frame.cw],
    cellSize: env.frame.cs,
    slots: env.slots,
    seats: env.seats.length,
    playerFeatures: PLAYER_FEATURES,
    scalars: SCALARS,
    acts: N_ACTS,
    troopBins: TROOP_BINS,
    unitTypes: UNIT_TYPES,
  };
}

function encode(parts: Part[], dimsInfo: Record<string, unknown> | null): Buffer {
  const header = Buffer.from(
    JSON.stringify({
      ...(dimsInfo === null ? {} : { dims: dimsInfo }),
      games: parts.map((p) => ({ ...p.header, blobs: p.blobs.map(([name, b]) => [name, b.byteLength]) })),
    }),
    "utf8",
  );
  const size = 4 + header.length + parts.reduce((s, p) => s + p.blobs.reduce((t, [, b]) => t + b.byteLength, 0), 0);
  const out = Buffer.allocUnsafe(4 + size);
  out.writeUInt32LE(size, 0);
  out.writeUInt32LE(header.length, 4);
  header.copy(out, 8);
  let off = 8 + header.length;
  for (const p of parts) {
    for (const [, b] of p.blobs) {
      out.set(b, off);
      off += b.byteLength;
    }
  }
  return out;
}

async function handle(msg: Init | Step): Promise<Buffer> {
  if (msg.cmd === "init") {
    init = msg;
    episodes = msg.episodes ?? new Array<number>(msg.games).fill(0);
    games = [];
    const parts: Part[] = [];
    for (let g = 0; g < msg.games; g++) {
      games.push(await EnvGame.create(msg.config, seedOf(g)));
      parts.push(fresh(g, {}));
    }
    return encode(parts, dims(games[0]));
  }
  if (init === null) throw new Error("step before init");
  const parts: Part[] = [];
  for (let g = 0; g < games.length; g++) {
    const env = games[g];
    env.step(msg.actions[g] ?? []);
    if (!env.done) {
      parts.push(part(env, g, env.observe(msg.full === true), {}));
      continue;
    }
    const end = {
      done: true,
      rewards: env.rewards,
      ranks: env.ranks,
      players: env.nPlayers,
      endTick: env.game.ticks(),
      hash: env.hash,
      invalid: env.invalid,
    };
    episodes[g]++;
    games[g] = await EnvGame.create(init.config, seedOf(g));
    parts.push(fresh(g, end));
  }
  return encode(parts, null);
}

async function main(): Promise<void> {
  let buf = Buffer.alloc(0);
  for await (const chunk of process.stdin) {
    buf = Buffer.concat([buf, chunk as Buffer]);
    while (buf.length >= 4 && buf.length >= 4 + buf.readUInt32LE(0)) {
      const n = buf.readUInt32LE(0);
      const msg = JSON.parse(buf.subarray(4, 4 + n).toString("utf8")) as Init | Step | { cmd: "close" };
      buf = buf.subarray(4 + n);
      if (msg.cmd === "close") return;
      process.stdout.write(await handle(msg));
    }
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
```

- [ ] **Step 4: Run the tests and the typecheck**

Run: `node --import tsx --test ../../bridge/worker.test.ts`
Expected: PASS (3 tests).
Run: `node --import tsx --test "../../bridge/*.test.ts"` and `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: all pass, exit 0.

- [ ] **Step 5: Commit**

```bash
git add bridge/worker.ts bridge/worker.test.ts
git commit -m "Bridge worker process and protocol"
```

---

### Task 7: Python vectorised environment

**Files:**
- Create: `trainer/__init__.py` (empty), `trainer/env.py`, `trainer/test_env.py`

**Interfaces:**
- Consumes: the protocol (Task 6)
- Produces: `ROOT`, `GAME_DIR`, `WORKER_TS`; channel constants `LAND, SHORE, OCEAN, HEIGHT, VALID, OWN, UNOWNED, DENSITY,
  FALLOUT, UNITS, N_CHANNELS = 22`; `class Worker(log_path=None)`; `class VecEnv(config: dict, workers: int,
  games_per_worker: int, seed: str, device="cuda", episodes: list[int] | None = None, log_dir: Path | None = None,
  full_frames=False)` with attributes `E, S, N, G, P, F, K, T, H0, W0, H, W, cs, dims, obs, episodes, o1, o2, n1, n2,
  fo`, methods `step(actions: np.ndarray [N, 4]) -> (obs, rewards [N], dones [N], infos, invalid [N])`, `close()`.
  `obs` = dict: `spatial` bf16 [N, 22, H, W], `scalars` f32 [N, 16], `players` f32 [N, P, 17], `type_mask` bool [N, 5],
  `attack_mask` bool [N, P], `cell_mask` bool [N, 3, H·W] (spawn, boat, build city). `dims` = worker dims +
  `channels`, `H`, `W`. `infos`: one dict per finished seat-episode `{env, seat, reward, rank, players, ticks}`.
  Actions use padded cell indices (row · W + col); `step` converts them for the worker.

- [ ] **Step 1: Write the failing test** — `trainer/test_env.py`

```python
import unittest

import numpy as np
import torch

from trainer.env import LAND, N_CHANNELS, OWN, VecEnv

ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "timerMinutes": 5, "seats": 1}


def random_legal(obs, rng):
    """Uniformly random legal actions from the masks, as padded-grid [type, cell, target, bin] rows."""
    rows = []
    types, attack, cells = (obs[k].cpu().numpy() for k in ("type_mask", "attack_mask", "cell_mask"))
    for i in range(types.shape[0]):
        t = rng.choice(np.flatnonzero(types[i]))
        m = {1: 0, 3: 1, 4: 2}.get(t)
        cell = rng.choice(np.flatnonzero(cells[i, m])) if m is not None and cells[i, m].any() else 0
        target = rng.choice(np.flatnonzero(attack[i])) if t == 2 else 0
        rows.append([t, cell, target, rng.integers(5)])
    return np.array(rows, np.int64)


class VecEnvTest(unittest.TestCase):
    def test_observations_during_random_play(self):
        env = VecEnv(ONION, workers=1, games_per_worker=2, seed="t")
        try:
            obs = env.obs
            self.assertEqual(obs["spatial"].shape, (2, N_CHANNELS, 64, 64))
            self.assertEqual(obs["spatial"].dtype, torch.bfloat16)
            self.assertEqual(obs["players"].shape, (2, 5, 17))
            self.assertEqual(obs["cell_mask"].shape, (2, 3, 64 * 64))
            self.assertTrue(obs["cell_mask"][:, 0].any(), "spawn cells exist at the start")
            rng = np.random.default_rng(0)
            for _ in range(80):
                obs, rew, done, infos, invalid = env.step(random_legal(obs, rng))
                x = obs["spatial"].float()
                self.assertTrue(torch.isfinite(x).all())
                self.assertLessEqual(float(x[:, LAND:OWN + 4].max()), 1.0 + 1e-3)
            self.assertGreater(float(obs["spatial"][:, OWN].float().sum()), 0, "the seats own land")
        finally:
            env.close()

    def test_deltas_match_full_frames(self):
        a = VecEnv(ONION, 1, 2, seed="same")
        b = VecEnv(ONION, 1, 2, seed="same", full_frames=True)
        try:
            rng = np.random.default_rng(1)
            for _ in range(80):
                act = random_legal(a.obs, rng)
                a.step(act)
                b.step(act)
            for k in ("o1", "o2", "n1", "n2", "fo"):
                self.assertTrue(torch.equal(getattr(a, k), getattr(b, k)), k)
            self.assertTrue(torch.equal(a.obs["spatial"], b.obs["spatial"]))
        finally:
            a.close()
            b.close()

    def test_game_end_and_restart(self):
        env = VecEnv({**ONION, "timerMinutes": 1}, 1, 1, seed="end")
        try:
            noop = np.zeros((1, 4), np.int64)
            for _ in range(60):
                obs, rew, done, infos, invalid = env.step(noop)
                if done[0]:
                    break
            self.assertEqual(done[0], 1)
            self.assertEqual(rew[0], -1)
            self.assertEqual(infos[0]["rank"], 4)
            self.assertEqual(env.episodes, [1])
            self.assertTrue(obs["cell_mask"][0, 0].any(), "the new game starts in its spawn phase")
        finally:
            env.close()

    def test_dead_worker_raises_with_log_path(self):
        env = VecEnv(ONION, 1, 1, seed="dead")
        try:
            env.workers[0].proc.kill()
            env.workers[0].proc.wait()
            with self.assertRaisesRegex(RuntimeError, "worker 0"):
                env.step(np.zeros((1, 4), np.int64))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_env -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.env'`.

- [ ] **Step 3: Implement** — `trainer/__init__.py` (empty file) and `trainer/env.py`

```python
"""Vectorised OpenFront games for training.

Node workers (bridge/worker.ts, protocol described there) run K games each and send per-cell frame deltas; the
observation tensors are assembled on the GPU from the state kept here.
"""
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = ROOT / "OpenFront_Source_Code" / "openfrontio"
WORKER_TS = ROOT / "bridge" / "worker.ts"

CELL = np.dtype([("idx", "<u4"), ("o1", "<u2"), ("o2", "<u2"), ("n1", "u1"), ("n2", "u1"), ("fo", "u1")])
UNIT = np.dtype([("type", "u1"), ("level", "u1"), ("owner", "<u2"), ("cell", "<u4"), ("troops", "<f4")])
assert CELL.itemsize == 11 and UNIT.itemsize == 12

# Spatial channels (tile counts are divided by the tiles per cell)
LAND, SHORE, OCEAN, HEIGHT, VALID = 0, 1, 2, 3, 4
OWN = 5  # 5..8: land owned by self, friendly, enemy, bot (relation codes 1..4 from the worker)
UNOWNED, DENSITY, FALLOUT = 9, 10, 11
UNITS = 12  # 12..21: (city, port, other structure, transport, warship) x (own, other)
N_CHANNELS = 22
UNIT_GROUP = {"City": 0, "Port": 1, "Factory": 2, "Missile Silo": 2, "SAM Launcher": 2, "Defense Post": 2,
              "Transport": 3, "Warship": 4}
DENSITY_FEATURE = 16  # player feature: log troops per tile


class Worker:
    """One Node process running K games (bridge/worker.ts)."""

    def __init__(self, index, log_path=None):
        self.index = index
        self.log_path = log_path
        self.log = open(log_path, "wb") if log_path is not None else subprocess.DEVNULL
        self.proc = subprocess.Popen([shutil.which("node"), "--import", "tsx", str(WORKER_TS)], cwd=GAME_DIR,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        try:  # the PC is shared: training yields to interactive programs
            below = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10)
            psutil.Process(self.proc.pid).nice(below)
        except psutil.Error:
            pass

    def send(self, msg):
        data = json.dumps(msg).encode()
        try:
            self.proc.stdin.write(struct.pack("<I", len(data)) + data)
            self.proc.stdin.flush()
        except OSError as e:
            raise self._died() from e

    def recv(self):
        """Returns (header, [dict name -> bytes per game])."""
        head = self.proc.stdout.read(4)
        if len(head) < 4:
            raise self._died()
        (size,) = struct.unpack("<I", head)
        body = self.proc.stdout.read(size)
        if len(body) < size:
            raise self._died()
        (hlen,) = struct.unpack_from("<I", body)
        header = json.loads(body[4:4 + hlen])
        off, games = 4 + hlen, []
        for g in header["games"]:
            blobs = {}
            for name, n in g["blobs"]:
                blobs[name] = body[off:off + n]
                off += n
            games.append(blobs)
        return header, games

    def _died(self):
        where = f"see {self.log_path}" if self.log_path else "run with log_dir to keep its stderr"
        return RuntimeError(f"worker {self.index} exited (code {self.proc.poll()}); {where}")

    def close(self):
        try:
            self.send({"cmd": "close"})
            self.proc.wait(timeout=10)
        except (RuntimeError, OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        if self.log is not subprocess.DEVNULL:
            self.log.close()


class VecEnv:
    """E = workers x games_per_worker games with S seats each; N = E x S observation rows (env-major)."""

    def __init__(self, config, workers, games_per_worker, seed, device="cuda", episodes=None, log_dir=None,
                 full_frames=False):
        self.device = torch.device(device)
        self.G = games_per_worker
        self.E = workers * games_per_worker
        self.full_frames = full_frames
        self.episodes = list(episodes) if episodes is not None else [0] * self.E
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.workers = [Worker(w, Path(log_dir) / f"worker{w}.log" if log_dir is not None else None)
                        for w in range(workers)]
        for w, wk in enumerate(self.workers):
            wk.send({"cmd": "init", "config": config, "games": self.G, "seed": seed, "worker": w,
                     "episodes": self.episodes[w * self.G:(w + 1) * self.G]})
        replies = [wk.recv() for wk in self.workers]
        d = replies[0][0]["dims"]
        self.H0, self.W0 = d["grid"]
        self.cs, self.P, self.S = d["cellSize"], d["slots"], d["seats"]
        self.F, self.K, self.T = d["playerFeatures"], d["scalars"], d["acts"]
        self.H, self.W = -(-self.H0 // 16) * 16, -(-self.W0 // 16) * 16  # the CNN downsamples by 16
        self.N = self.E * self.S
        self.dims = {**d, "channels": N_CHANNELS, "H": self.H, "W": self.W}
        dev, hw = self.device, self.H * self.W
        cells = np.arange(self.H0 * self.W0)
        self.pad = torch.from_numpy((cells // self.W0) * self.W + cells % self.W0).to(dev)
        self.valid = torch.zeros(hw, dtype=torch.bool, device=dev)
        self.valid[self.pad] = True
        self.unit_group = torch.tensor([UNIT_GROUP.get(t, -1) for t in d["unitTypes"]], device=dev)
        self.transport = d["unitTypes"].index("Transport")
        self.terrain = torch.zeros(self.E, 4, hw, device=dev)
        self.o1 = torch.zeros(self.E * hw, dtype=torch.long, device=dev)
        self.o2 = torch.zeros_like(self.o1)
        self.n1 = torch.zeros(self.E * hw, device=dev)
        self.n2 = torch.zeros_like(self.n1)
        self.fo = torch.zeros_like(self.n1)
        self.obs = self._absorb(replies)[0]

    def step(self, actions):
        """actions: int array [N, 4] of [type, cell, target, bin] with padded cell indices."""
        a = np.asarray(actions, dtype=np.int64).reshape(self.E, self.S, 4).copy()
        a[..., 1] = (a[..., 1] // self.W) * self.W0 + a[..., 1] % self.W
        for w, wk in enumerate(self.workers):
            wk.send({"cmd": "step", "actions": a[w * self.G:(w + 1) * self.G].tolist(), "full": self.full_frames})
        out = self._absorb([wk.recv() for wk in self.workers])
        self.obs = out[0]
        return out

    def close(self):
        for wk in self.workers:
            wk.close()

    def _absorb(self, replies):
        E, S, P, F, K, T = self.E, self.S, self.P, self.F, self.K, self.T
        rewards = np.zeros((E, S), np.float32)
        dones = np.zeros((E, S), np.float32)
        invalid = np.zeros((E, S), bool)
        players = np.empty((E, S, P, F), np.float32)
        rel = np.empty((E, S, P), np.uint8)
        scalars = np.empty((E, S, K), np.float32)
        types = np.empty((E, S, T), bool)
        attack = np.empty((E, S, P), bool)
        cells, units, infos, resets = [], [], [], []
        e = 0
        for header, games in replies:
            for g, b in zip(header["games"], games):
                if g.get("done"):
                    dones[e] = 1
                    rewards[e] = g["rewards"]
                    infos += [{"env": e, "seat": s, "reward": g["rewards"][s], "rank": g["ranks"][s],
                               "players": g["players"], "ticks": g["endTick"]} for s in range(S)]
                self.episodes[e] = g["episode"]
                if g.get("reset"):
                    resets.append(e)
                    terr = torch.from_numpy(np.frombuffer(b["terrain"], np.uint8).reshape(4, -1).astype(np.float32))
                    self.terrain[e].zero_()
                    self.terrain[e][:, self.pad] = terr.to(self.device)
                invalid[e] = g["invalid"]
                cells.append((e, np.frombuffer(b["cells"], CELL)))
                units.append((e, np.frombuffer(b["units"], UNIT)))
                players[e] = np.frombuffer(b["players"], np.float32).reshape(S, P, F)
                rel[e] = np.frombuffer(b["rel"], np.uint8).reshape(S, P)
                scalars[e] = np.frombuffer(b["scalars"], np.float32).reshape(S, K)
                types[e] = np.frombuffer(b["types"], np.uint8).reshape(S, T)
                attack[e] = np.frombuffer(b["attack"], np.uint8).reshape(S, P)
                e += 1
        self._apply_cells(resets, cells)
        obs = self._observe(players, rel, scalars, types, attack, units)
        return obs, rewards.reshape(-1), dones.reshape(-1), infos, invalid.reshape(-1)

    def _apply_cells(self, resets, cells):
        hw, dev = self.H * self.W, self.device
        for e in resets:
            for t in (self.o1, self.o2, self.n1, self.n2, self.fo):
                t[e * hw:(e + 1) * hw] = 0
        rec = np.concatenate([c for _, c in cells])
        if len(rec) == 0:
            return
        env = np.concatenate([np.full(len(c), e, np.int64) for e, c in cells])
        idx = torch.from_numpy(env).to(dev) * hw + self.pad[torch.from_numpy(rec["idx"].astype(np.int64)).to(dev)]
        for name, t in (("o1", self.o1), ("o2", self.o2), ("n1", self.n1), ("n2", self.n2), ("fo", self.fo)):
            t[idx] = torch.from_numpy(rec[name].astype(np.int64 if name[0] == "o" else np.float32)).to(dev)

    def _observe(self, players, rel, scalars, types, attack, units):
        N, S, P, hw, dev = self.N, self.S, self.P, self.H * self.W, self.device
        area = float(self.cs * self.cs)
        row_env = torch.arange(N, device=dev) // S
        rel_t = torch.from_numpy(rel.reshape(N, P).astype(np.int64)).to(dev)
        pl = torch.from_numpy(players.reshape(N, P, self.F)).to(dev)
        x = torch.zeros(N, N_CHANNELS, hw, device=dev)
        terr = self.terrain[row_env]
        x[:, LAND:OCEAN + 1] = terr[:, 0:3] / area
        x[:, HEIGHT] = terr[:, 3] / 31
        x[:, VALID] = self.valid.float()
        flat = row_env[:, None] * hw + torch.arange(hw, device=dev)
        own = torch.zeros(N, 4, hw, device=dev)
        dens = pl[..., DENSITY_FEATURE]
        for o, n in ((self.o1, self.n1), (self.o2, self.n2)):
            owner = o[flat]
            frac = n[flat] / area
            r = torch.gather(rel_t, 1, owner)
            own.scatter_add_(1, (r - 1).clamp(min=0).unsqueeze(1), (frac * (r > 0)).unsqueeze(1))
            x[:, DENSITY] += frac * torch.gather(dens, 1, owner)
        x[:, OWN:OWN + 4] = own
        x[:, UNOWNED] = (x[:, LAND] - own.sum(1)).clamp(min=0)
        x[:, FALLOUT] = self.fo[flat] / area
        self._add_units(x, units, rel_t)
        cell_mask = torch.stack([x[:, UNOWNED] > 0,
                                 (x[:, SHORE] > 0) & (x[:, LAND] - x[:, OWN] > 1e-6),
                                 x[:, OWN] > 0], 1) & self.valid
        return {
            "spatial": x.view(N, N_CHANNELS, self.H, self.W).to(torch.bfloat16),
            "scalars": torch.from_numpy(scalars.reshape(N, self.K)).to(dev),
            "players": pl,
            "type_mask": torch.from_numpy(types.reshape(N, self.T)).to(dev),
            "attack_mask": torch.from_numpy(attack.reshape(N, P)).to(dev),
            "cell_mask": cell_mask,
        }

    def _add_units(self, x, units, rel_t):
        rec = np.concatenate([u for _, u in units])
        if len(rec) == 0:
            return
        dev = self.device
        env = torch.from_numpy(np.concatenate([np.full(len(u), e, np.int64) for e, u in units])).to(dev)
        typ = torch.from_numpy(rec["type"].astype(np.int64)).to(dev)
        owner = torch.from_numpy(rec["owner"].astype(np.int64)).to(dev)
        cell = self.pad[torch.from_numpy(rec["cell"].astype(np.int64)).to(dev)]
        troops = torch.from_numpy(rec["troops"].astype(np.float32)).to(dev)
        group = self.unit_group[typ]
        value = torch.where(typ == self.transport, torch.log1p(troops) / 10, torch.ones_like(troops))
        for s in range(self.S):
            row = env * self.S + s
            r = rel_t[row, owner]
            keep = (group >= 0) & (r > 0)
            ch = UNITS + 2 * group + (r != 1).long()
            x.index_put_((row[keep], ch[keep], cell[keep]), value[keep], accumulate=True)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_env -v`
Expected: PASS (4 tests, ~30 s: each test starts Node workers).

- [ ] **Step 5: Commit**

```bash
git add trainer/__init__.py trainer/env.py trainer/test_env.py
git commit -m "Trainer vectorised environment with GPU observation builder"
```

---

### Task 8: Policy network and action distribution

**Files:**
- Create: `trainer/model.py`, `trainer/test_model.py`

**Interfaces:**
- Consumes: `VecEnv.dims` keys `channels, playerFeatures, scalars, acts, troopBins` (Task 7)
- Produces: `ACT_NAMES = ["noop", "spawn", "attack", "boat", "build_city"]`; `SPAWN, ATTACK, BOAT, BUILD = 1, 2, 3, 4`;
  `class Policy(dims, widths=(64, 128, 192))` with `forward(obs) -> {"type" [N,T], "maps" [N,3,H·W], "bins" [N,2,B],
  "ptr" [N,P], "value" [N]}`; `class RandomPolicy(dims)` (same outputs, all zero); `class ActionDist(out, obs)` with
  `sample() -> LongTensor [N, 4]`, `log_prob(actions) -> [N]`, `entropy() -> [N]` (each head normalised by the log of
  its size).

- [ ] **Step 1: Write the failing test** — `trainer/test_model.py`

```python
import math
import unittest

import torch

from trainer.model import ATTACK, BUILD, SPAWN, ActionDist, Policy, RandomPolicy

DIMS = {"channels": 22, "playerFeatures": 17, "scalars": 16, "acts": 5, "troopBins": [0.1, 0.25, 0.5, 0.75, 1.0]}
N, P, H, W = 6, 5, 32, 32


def fake_obs(device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(0)
    cell_mask = torch.rand(N, 3, H * W, generator=g) < 0.1
    cell_mask[0] = False  # a row without any legal cell must not produce NaNs
    type_mask = torch.tensor([[1, 1, 0, 0, 0]] * 3 + [[1, 0, 1, 0, 1]] * 3, dtype=torch.bool)
    attack_mask = torch.zeros(N, P, dtype=torch.bool)
    attack_mask[3:, 2] = True
    obs = {
        "spatial": torch.rand(N, 22, H, W, generator=g).to(torch.bfloat16),
        "scalars": torch.rand(N, 16, generator=g),
        "players": torch.rand(N, P, 17, generator=g),
        "type_mask": type_mask, "attack_mask": attack_mask, "cell_mask": cell_mask,
    }
    return {k: v.to(device) for k, v in obs.items()}


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.obs = fake_obs()
        self.model = Policy(DIMS).cuda()

    def test_output_shapes(self):
        with torch.autocast("cuda", torch.bfloat16):
            out = self.model(self.obs)
        self.assertEqual(out["type"].shape, (N, 5))
        self.assertEqual(out["maps"].shape, (N, 3, H * W))
        self.assertEqual(out["bins"].shape, (N, 2, 5))
        self.assertEqual(out["ptr"].shape, (N, P))
        self.assertEqual(out["value"].shape, (N,))

    def test_samples_respect_masks_and_log_probs_are_finite(self):
        out = self.model(self.obs)
        for _ in range(20):
            dist = ActionDist(out, self.obs)
            a = dist.sample()
            t = a[:, 0]
            self.assertTrue(self.obs["type_mask"].gather(1, t[:, None]).all())
            for i in range(N):
                if t[i] == SPAWN and self.obs["cell_mask"][i, 0].any():
                    self.assertTrue(self.obs["cell_mask"][i, 0, a[i, 1]])
                if t[i] == ATTACK:
                    self.assertEqual(int(a[i, 2]), 2)
                if t[i] == BUILD and self.obs["cell_mask"][i, 2].any():
                    self.assertTrue(self.obs["cell_mask"][i, 2, a[i, 1]])
            lp = dist.log_prob(a)
            self.assertTrue(torch.isfinite(lp).all())
            self.assertTrue(torch.isfinite(dist.entropy()).all())

    def test_gradients_are_finite(self):
        out = self.model(self.obs)
        dist = ActionDist(out, self.obs)
        a = dist.sample()
        (-(dist.log_prob(a).mean()) - dist.entropy().mean() + out["value"].pow(2).mean()).backward()
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_random_policy_is_uniform_over_legal_types(self):
        dist = ActionDist(RandomPolicy(DIMS)(self.obs), self.obs)
        probs = dist.type.probs
        self.assertAlmostEqual(float(probs[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(probs[3, 4]), 1 / 3, places=5)
        noop = torch.zeros(N, 4, dtype=torch.long, device="cuda")
        self.assertAlmostEqual(float(dist.log_prob(noop)[0]), math.log(0.5), places=5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.model'`.

- [ ] **Step 3: Implement** — `trainer/model.py`

```python
"""Policy network (PLAN "Architecture"): ResNet map encoder, U-Net-lite heat maps, player and scalar encoders, heads."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ACT_NAMES = ["noop", "spawn", "attack", "boat", "build_city"]
SPAWN, ATTACK, BOAT, BUILD = 1, 2, 3, 4
N_MAPS = 3  # heat maps: spawn, boat landing, city
MASK_VALUE = -1e8  # finite, so masked entries never make NaN gradients


class Res(nn.Module):
    """IMPALA-style residual block: two 3x3 convs, no norm, zero-init residual branch."""

    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class Policy(nn.Module):
    def __init__(self, dims, widths=(64, 128, 192)):
        super().__init__()
        c_in, n_feat, n_scal = dims["channels"], dims["playerFeatures"], dims["scalars"]
        self.n_bins = len(dims["troopBins"])
        c1, c2, c3 = widths
        self.stem = nn.Conv2d(c_in, c1, 4, stride=4)
        self.s1 = nn.Sequential(Res(c1), Res(c1))
        self.d2, self.s2 = nn.Conv2d(c1, c2, 2, stride=2), nn.Sequential(Res(c2), Res(c2))
        self.d3, self.s3 = nn.Conv2d(c2, c3, 2, stride=2), nn.Sequential(Res(c3), Res(c3))
        self.players = nn.Sequential(nn.Linear(n_feat, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.scalars = nn.Sequential(nn.Linear(n_scal, 128), nn.ReLU())
        self.core = nn.Sequential(nn.Linear(2 * c3 + 256, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.film = nn.Linear(512, 2 * c1)
        self.u3, self.u2 = nn.Conv2d(c3, c2, 1), nn.Conv2d(c2, c1, 1)
        self.shuffle = nn.Conv2d(c1, N_MAPS * 16, 1)
        self.skip = nn.Conv2d(c_in, N_MAPS, 1)
        self.type_head = nn.Linear(512, dims["acts"])
        self.bin_head = nn.Linear(512, 2 * self.n_bins)
        self.query, self.key = nn.Linear(512, 128), nn.Linear(128, 128)
        self.value_head = nn.Linear(512, 1)
        # Policy outputs start at zero: a uniform policy over legal actions.
        for layer in (self.type_head, self.bin_head, self.shuffle, self.skip, self.query):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, obs):
        x = obs["spatial"].contiguous(memory_format=torch.channels_last)
        f1 = self.s1(self.stem(x))
        f2 = self.s2(self.d2(F.relu(f1)))
        f3 = self.s3(self.d3(F.relu(f2)))
        pooled = torch.cat([f3.mean((2, 3)), f3.amax((2, 3))], 1)
        emb = self.players(obs["players"])
        exists = obs["players"][..., :1]
        p_pool = (emb * exists).sum(1) / exists.sum(1).clamp(min=1)
        h = self.core(torch.cat([pooled, p_pool, self.scalars(obs["scalars"])], 1))
        g = f2 + F.interpolate(self.u3(F.relu(f3)), scale_factor=2)
        g = f1 + F.interpolate(self.u2(F.relu(g)), scale_factor=2)
        gamma, beta = self.film(h).chunk(2, 1)
        g = g * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        maps = F.pixel_shuffle(self.shuffle(F.relu(g)), 4) + self.skip(x)
        ptr = torch.einsum("nd,npd->np", self.query(h), self.key(emb)) / math.sqrt(128)
        return {"type": self.type_head(h), "maps": maps.flatten(2), "bins": self.bin_head(h).view(-1, 2, self.n_bins),
                "ptr": ptr, "value": self.value_head(h).squeeze(1)}


class RandomPolicy(nn.Module):
    """Uniform over legal actions: zero logits everywhere, the masks do the rest (the Phase 1 baseline)."""

    def __init__(self, dims):
        super().__init__()
        self.n_bins = len(dims["troopBins"])

    def forward(self, obs):
        n, dev = obs["type_mask"].shape[0], obs["type_mask"].device
        return {"type": torch.zeros(obs["type_mask"].shape, device=dev),
                "maps": torch.zeros(obs["cell_mask"].shape, device=dev),
                "bins": torch.zeros(n, 2, self.n_bins, device=dev),
                "ptr": torch.zeros(obs["attack_mask"].shape, device=dev),
                "value": torch.zeros(n, device=dev)}


def _masked(logits, mask):
    mask = mask.clone()
    mask[..., 0] |= ~mask.any(-1)  # rows without a legal entry get a dummy one; their head is never used
    return Categorical(logits=torch.where(mask, logits, MASK_VALUE))


class ActionDist:
    """The joint action [type, cell, target, bin]: the type picks which argument heads are used.

    spawn: spawn map cell; attack: player pointer + attack troops; boat: boat map cell + boat troops;
    build city: city map cell."""

    def __init__(self, out, obs):
        self.type = _masked(out["type"].float(), obs["type_mask"])
        self.cells = _masked(out["maps"].float(), obs["cell_mask"])  # batch [N, 3]
        self.ptr = _masked(out["ptr"].float(), obs["attack_mask"])
        self.bins = Categorical(logits=out["bins"].float())  # batch [N, 2]: attack, boat

    def sample(self):
        t, c, p, b = self.type.sample(), self.cells.sample(), self.ptr.sample(), self.bins.sample()
        zero = torch.zeros_like(t)
        cell = torch.where(t == SPAWN, c[:, 0], torch.where(t == BOAT, c[:, 1], torch.where(t == BUILD, c[:, 2], zero)))
        target = torch.where(t == ATTACK, p, zero)
        troops = torch.where(t == ATTACK, b[:, 0], torch.where(t == BOAT, b[:, 1], zero))
        return torch.stack([t, cell, target, troops], 1)

    def log_prob(self, actions):
        t, cell, target, troops = actions.unbind(1)
        lc = self.cells.log_prob(cell[:, None].expand(-1, N_MAPS))
        lb = self.bins.log_prob(troops[:, None].expand(-1, 2))
        zero = torch.zeros_like(lc[:, 0])
        return (self.type.log_prob(t)
                + torch.where(t == SPAWN, lc[:, 0], zero)
                + torch.where(t == ATTACK, self.ptr.log_prob(target) + lb[:, 0], zero)
                + torch.where(t == BOAT, lc[:, 1] + lb[:, 1], zero)
                + torch.where(t == BUILD, lc[:, 2], zero))

    def entropy(self):
        """Exact joint entropy, H(type) + sum_k p(k) H(args | k), with every head divided by the log of its size so
        the 4k-cell maps do not drown out the type head."""
        hc = self.cells.entropy() / math.log(self.cells.logits.shape[-1])
        hb = self.bins.entropy() / math.log(self.bins.logits.shape[-1])
        hp = self.ptr.entropy() / math.log(max(self.ptr.logits.shape[-1], 2))
        pt = self.type.probs
        return (self.type.entropy() / math.log(pt.shape[-1]) + pt[:, SPAWN] * hc[:, 0]
                + pt[:, ATTACK] * (hp + hb[:, 0]) + pt[:, BOAT] * (hc[:, 1] + hb[:, 1]) + pt[:, BUILD] * hc[:, 2])
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add trainer/model.py trainer/test_model.py
git commit -m "Policy network and masked joint action distribution"
```

---

### Task 9: Training control (API, page, CLI)

**Files:**
- Create: `trainer/control.py`, `trainer/ctl.py`, `trainer/test_control.py`

**Interfaces:**
- Produces: `THROTTLE = {"full": 1.0, "half": 0.5, "low": 0.2}`; `class Control(duty=1.0, gpu_temp_limit=0)` with
  `command(name, arg=None)`, `wait() -> bool` (blocks while paused; False once stop was requested), `throttle(busy_s)`,
  `take_save() -> bool`, `update(**info)`, `note(msg)`, `status() -> dict`, `set_state(s)`, `start_sampler(period)`;
  `serve(control, port, sampler=True) -> ThreadingHTTPServer`; `sample_system()`, `gpu_status()`,
  `cpu_temp(url=...)`. HTTP: `GET /` page, `GET /status` JSON, `POST /pause|/resume|/save|/stop|/throttle/<v>` with
  header `X-Control: 1`.

- [ ] **Step 1: Write the failing test** — `trainer/test_control.py`

```python
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from trainer.control import Control, cpu_temp, serve


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.c = Control()
        self.server = serve(self.c, 0, sampler=False)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, header=True):
        req = urllib.request.Request(self.base + path, data=b"", method="POST",
                                     headers={"X-Control": "1"} if header else {})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)

    def test_buttons_need_the_control_header(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post("/pause", header=False)
        self.assertEqual(e.exception.code, 403)
        self.assertFalse(self.c.paused)

    def test_pause_resume_throttle_save_stop(self):
        self.post("/pause")
        self.assertTrue(self.c.paused)
        self.post("/resume")
        self.assertFalse(self.c.paused)
        self.post("/throttle/half")
        self.assertEqual(self.c.duty, 0.5)
        self.post("/throttle/0.3")
        self.assertEqual(self.c.duty, 0.3)
        self.post("/save")
        self.assertTrue(self.c.take_save())
        self.assertFalse(self.c.take_save())
        self.post("/stop")
        self.assertFalse(self.c.wait())

    def test_bad_commands_are_rejected(self):
        for path in ("/throttle/fast", "/throttle/0", "/explode"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.post(path)
            self.assertEqual(e.exception.code, 400, path)

    def test_status_and_page(self):
        self.c.update(update=3, metrics={"episodes/win_rate": 0.5})
        with urllib.request.urlopen(self.base + "/status", timeout=5) as r:
            s = json.load(r)
        self.assertEqual(s["update"], 3)
        self.assertEqual(s["metrics"]["episodes/win_rate"], 0.5)
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:
            self.assertIn("<title>Training control</title>", r.read().decode())

    def test_wait_blocks_while_paused(self):
        self.c.command("pause")
        result = []
        t = threading.Thread(target=lambda: result.append(self.c.wait()))
        t.start()
        time.sleep(0.5)
        self.assertTrue(t.is_alive())
        self.c.command("resume")
        t.join(2)
        self.assertEqual(result, [True])

    def test_stop_while_paused_releases_wait(self):
        self.c.command("pause")
        result = []
        t = threading.Thread(target=lambda: result.append(self.c.wait()))
        t.start()
        self.c.command("stop")
        t.join(2)
        self.assertEqual(result, [False])

    def test_cpu_temp_without_monitor_is_none(self):
        self.assertIsNone(cpu_temp("http://127.0.0.1:9/data.json"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_control -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.control'`.

- [ ] **Step 3: Implement** — `trainer/control.py`

```python
"""Training control for a shared PC: a local HTTP API (127.0.0.1 only) with a status page, and a system sampler.

The CLI is `python -m trainer.ctl`. Pausing leaves the Node workers idle and frees the GPU's compute."""
import collections
import json
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil
import torch

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
            self.state = "paused"
            torch.cuda.empty_cache()
            while self.paused and not self.stop_requested:
                time.sleep(0.2)
            self.state = "running"
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


def serve(control, port, sampler=True):
    """Starts the control server on 127.0.0.1:port (0 = any free port) in a daemon thread."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/status":
                self._send(200, json.dumps(control.status()))
            else:
                self._send(404, "{}")

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
  --line: #e3e6eb; }
@media (prefers-color-scheme: dark) { :root { --bg: #14171c; --card: #1d2128; --text: #e6e9ef; --muted: #98a1b3;
  --accent: #6b9bff; --hot: #fb923c; --line: #2c323c; } }
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
  <section><h2>Training</h2><table id="training"></table></section>
  <section><h2>Computer</h2><table id="system"></table></section>
</div>
<section style="margin-top:12px"><h2>Events</h2><div id="log"></div></section>
<p class="hint">Charts: run <code>.venv/Scripts/tensorboard --logdir runs</code> and open
<a href="http://localhost:6006">localhost:6006</a>.</p>
</main>
<script>
const fmt = (v, d = 2) => v === null || v === undefined ? "–" : typeof v === "number" ? v.toFixed(d) : v;
function rows(id, items) {
  document.getElementById(id).innerHTML = items.map(([k, v, hot]) =>
    `<tr><td>${k}</td><td class="${hot ? "hot" : ""}">${v}</td></tr>`).join("");
}
async function send(cmd) {
  await fetch("/" + cmd, { method: "POST", headers: { "X-Control": "1" } });
  refresh();
}
async function refresh() {
  try {
    const s = await (await fetch("/status")).json();
    const m = s.metrics || {}, sys = s.system || {}, g = sys.gpu || {};
    document.getElementById("state").textContent = s.state + (s.auto_paused ? " (GPU too hot)" : "") +
      (s.duty < 1 ? ` · throttled to ${Math.round(s.duty * 100)}%` : "");
    rows("training", [
      ["Update", s.update ?? "–"], ["Decisions", s.global_step ?? "–"],
      ["Decisions / s", fmt(m["perf/decisions_per_s"], 0)],
      ["Win rate (last 200 games)", fmt(m["episodes/win_rate"])], ["Mean reward", fmt(m["episodes/mean_reward"])],
      ["Mean rank", fmt(m["episodes/mean_rank"])], ["Value accuracy, late game", fmt(m["value/outcome_acc_late"])],
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
refresh();
setInterval(refresh, 2000);
</script></body></html>
"""
```

- [ ] **Step 4: Implement** — `trainer/ctl.py`

```python
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
```

- [ ] **Step 5: Run the tests and try the CLI against nothing**

Run: `.venv/Scripts/python -m unittest trainer.test_control -v`
Expected: PASS (7 tests).
Run: `.venv/Scripts/python -m trainer.ctl status`
Expected: exits with `no trainer is answering on port 8787 (...)`.

- [ ] **Step 6: Commit**

```bash
git add trainer/control.py trainer/ctl.py trainer/test_control.py
git commit -m "Training control API, status page and CLI"
```

---

### Task 10: PPO training loop with checkpoints and resume

**Files:**
- Create: `trainer/ppo.py`, `trainer/test_ppo.py`

**Interfaces:**
- Consumes: `VecEnv` (Task 7), `Policy`, `ActionDist`, `ACT_NAMES` (Task 8), `Control`, `serve` (Task 9)
- Produces: `parse_args(argv)`, `env_config(args) -> dict`, `gae(rewards, values, dones, last_value, gamma, lam)`,
  `save_checkpoint(path, **state)`, `class Outcomes`, `main(argv=None)`. Checkpoint keys: `model, optimizer, args,
  update, global_step, episodes, game_commit, rng`. Files: `checkpoints/<run>/latest.pt` (+ `update000250.pt` …),
  TensorBoard in `runs/<run>`, worker logs in `runs/<run>/workers/`.

- [ ] **Step 1: Write the failing test** — `trainer/test_ppo.py`

```python
import shutil
import unittest

import torch

from trainer.env import ROOT
from trainer.ppo import gae, main


class GaeTest(unittest.TestCase):
    def test_hand_computed(self):
        r = torch.tensor([[0.0], [0.0], [1.0]])
        v = torch.tensor([[0.5], [0.6], [0.7]])
        d = torch.tensor([[0.0], [0.0], [1.0]])
        adv = gae(r, v, d, torch.tensor([9.0]), gamma=0.9, lam=0.8)
        # t2: 1 - 0.7 = 0.3; t1: 0.9*0.7 - 0.6 + 0.72*0.3 = 0.246; t0: 0.9*0.6 - 0.5 + 0.72*0.246 = 0.21712
        self.assertTrue(torch.allclose(adv[:, 0], torch.tensor([0.21712, 0.246, 0.3])))

    def test_episode_boundary_cuts_bootstrap(self):
        r = torch.tensor([[1.0], [0.0]])
        v = torch.tensor([[0.2], [0.4]])
        d = torch.tensor([[1.0], [0.0]])
        adv = gae(r, v, d, torch.tensor([0.5]), gamma=1.0, lam=1.0)
        self.assertAlmostEqual(float(adv[0, 0]), 0.8, places=6)  # the next episode's values do not leak back
        self.assertAlmostEqual(float(adv[1, 0]), 0.1, places=6)


class SmokeTest(unittest.TestCase):
    RUN = "unittest-smoke"

    def tearDown(self):
        for d in ("checkpoints", "runs"):
            shutil.rmtree(ROOT / d / self.RUN, ignore_errors=True)

    def test_train_save_and_resume(self):
        common = ["--run", self.RUN, "--workers", "1", "--games-per-worker", "2", "--steps", "16",
                  "--minibatches", "2", "--epochs", "1", "--port", "0", "--timer", "1"]
        main(common + ["--updates", "2"])
        first = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
        self.assertEqual(first["update"], 2)
        self.assertEqual(first["global_step"], 2 * 16 * 2)
        main(common + ["--updates", "3", "--resume"])
        second = torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)
        self.assertEqual(second["update"], 3)
        self.assertEqual(second["global_step"], 3 * 16 * 2)
        self.assertTrue(all(b >= a for a, b in zip(first["episodes"], second["episodes"])))
        self.assertTrue((ROOT / "runs" / self.RUN).is_dir())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_ppo -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.ppo'`.

- [ ] **Step 3: Implement** — `trainer/ppo.py`

```python
"""PPO training (CleanRL-style) with pause/save/resume, for a PC that is also used daily.

  Train:   .venv/Scripts/python -m trainer.ppo --run onion
  Resume:  .venv/Scripts/python -m trainer.ppo --run onion --resume
  Control: http://127.0.0.1:8787 or python -m trainer.ctl pause|resume|save|stop|throttle|status
  Charts:  .venv/Scripts/tensorboard --logdir runs
"""
import argparse
import collections
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trainer.control import Control, serve
from trainer.env import GAME_DIR, ROOT, VecEnv
from trainer.model import ACT_NAMES, ActionDist, Policy

KEEP_ON_RESUME = ("resume", "port", "duty", "gpu_temp_limit", "updates")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="onion", help="name for checkpoints/<run> and runs/<run>")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/latest.pt")
    p.add_argument("--map", default="Onion")
    p.add_argument("--normal-size", action="store_true", help="Normal map size instead of Compact")
    p.add_argument("--nations", type=int, default=3)
    p.add_argument("--difficulty", default="Easy")
    p.add_argument("--bots", type=int, default=0)
    p.add_argument("--timer", type=int, default=5, help="game timer in minutes")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--steps", type=int, default=128, help="decisions per game per update")
    p.add_argument("--updates", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--widths", default="64,128,192", help="CNN stage widths")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=25, help="updates between checkpoints")
    p.add_argument("--port", type=int, default=8787, help="control page port (0 = any free port)")
    p.add_argument("--duty", type=float, default=1.0, help="start throttled: fraction of time spent working")
    p.add_argument("--gpu-temp-limit", type=int, default=0, help="auto-pause at this GPU temperature in C (0 = off)")
    return p.parse_args(argv)


def env_config(a):
    return {"map": a.map, "compact": not a.normal_size, "nations": a.nations, "difficulty": a.difficulty,
            "bots": a.bots, "timerMinutes": a.timer, "seats": 1}


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation. rewards, values, dones: [T, N]; dones[t] = the episode ended after step t
    (the next observation already belongs to a new game). last_value: [N], the value after the last step."""
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * live - values[t]
        running = delta + gamma * lam * live * running
        adv[t] = running
    return adv


def save_checkpoint(path, **state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash during the save never corrupts the previous checkpoint


def game_commit():
    return subprocess.run(["git", "-C", str(GAME_DIR), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


class Outcomes:
    """Rolling episode results, and how often the value head's sign predicted the outcome, by game phase."""

    def __init__(self, rows, window=200):
        self.episodes = collections.deque(maxlen=window)
        self.pending = [[] for _ in range(rows)]
        self.hits = collections.deque(maxlen=50_000)

    def record(self, progress, values):
        for row, (p, v) in enumerate(zip(progress, values)):
            self.pending[row].append((p, v))

    def finish(self, infos, seats):
        for info in infos:
            row = info["env"] * seats + info["seat"]
            won = info["reward"] > 0
            self.hits.extend((min(int(p * 3), 2), (v > 0) == won) for p, v in self.pending[row])
            self.pending[row] = []
            self.episodes.append(info)

    def summary(self):
        if not self.episodes:
            return {}
        ep = list(self.episodes)
        out = {"episodes/win_rate": float(np.mean([e["rank"] == 1 for e in ep])),
               "episodes/mean_reward": float(np.mean([e["reward"] for e in ep])),
               "episodes/mean_rank": float(np.mean([e["rank"] for e in ep])),
               "episodes/ticks": float(np.mean([e["ticks"] for e in ep]))}
        for bucket, name in enumerate(("early", "mid", "late")):
            hits = [ok for b, ok in self.hits if b == bucket]
            if hits:
                out[f"value/outcome_acc_{name}"] = float(np.mean(hits))
        return out


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

    envs = VecEnv(env_config(a), a.workers, a.games_per_worker, seed=f"{a.run}-{a.seed}", device=device,
                  episodes=state["episodes"] if state else None, log_dir=run_dir / "workers")
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

    def checkpoint():
        s = dict(model=model.state_dict(), optimizer=opt.state_dict(), args=vars(a), update=update,
                 global_step=global_step, episodes=list(envs.episodes), game_commit=commit,
                 rng={"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all()})
        save_checkpoint(ckpt_dir / "latest.pt", **s)
        if update % 250 == 0 and update > 0:
            save_checkpoint(ckpt_dir / f"update{update:06d}.pt", **s)
        control.note(f"saved checkpoint at update {update}")

    T, N = a.steps, envs.N
    obs = envs.obs
    buf = {k: torch.zeros((T,) + tuple(v.shape), dtype=v.dtype, device=device) for k, v in obs.items()}
    actions = torch.zeros(T, N, 4, dtype=torch.long, device=device)
    logps, values, rewards, dones = (torch.zeros(T, N, device=device) for _ in range(4))
    outcomes = Outcomes(N)
    act_counts, invalid_count = np.zeros(envs.T), 0

    def rollout():
        """Collects T steps from every game; False if a stop was requested meanwhile."""
        nonlocal obs, global_step, invalid_count
        for step in range(T):
            if not control.wait():
                return False
            t0 = time.perf_counter()
            for k, v in obs.items():
                buf[k][step] = v
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(obs)
                dist = ActionDist(out, obs)
                act = dist.sample()
                actions[step], logps[step], values[step] = act, dist.log_prob(act), out["value"].float()
            act_np = act.cpu().numpy()
            outcomes.record(obs["scalars"][:, 0].tolist(), values[step].tolist())
            obs, rew, done, infos, invalid = envs.step(act_np)
            rewards[step] = torch.from_numpy(rew).to(device)
            dones[step] = torch.from_numpy(done).to(device)
            outcomes.finish(infos, envs.S)
            np.add.at(act_counts, act_np[:, 0], 1)
            invalid_count += int(invalid.sum())
            global_step += N
            control.throttle(time.perf_counter() - t0)
        return True

    def learn():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            last_value = model(obs)["value"].float()
        adv = gae(rewards, values, dones, last_value, a.gamma, a.gae_lambda)
        ret = adv + values
        flat = {k: v.flatten(0, 1) for k, v in buf.items()}
        b_act, b_logp, b_adv, b_ret = actions.flatten(0, 1), logps.flatten(), adv.flatten(), ret.flatten()
        stats = collections.defaultdict(list)
        for _ in range(a.epochs):
            for mb in torch.randperm(T * N, device=device).split(T * N // a.minibatches):
                o = {k: v[mb] for k, v in flat.items()}
                with torch.autocast("cuda", torch.bfloat16):
                    out = model(o)
                dist = ActionDist(out, o)
                logp, ent = dist.log_prob(b_act[mb]), dist.entropy()
                log_ratio = logp - b_logp[mb]
                ratio = log_ratio.exp()
                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                v_loss = 0.5 * (out["value"].float() - b_ret[mb]).pow(2).mean()
                loss = pg_loss - a.ent_coef * ent.mean() + a.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    stats["loss/policy"].append(pg_loss.item())
                    stats["loss/value"].append(v_loss.item())
                    stats["loss/entropy"].append(ent.mean().item())
                    stats["loss/approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["loss/clip_frac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
        metrics = {k: float(np.mean(v)) for k, v in stats.items()}
        var = ret.var().item()
        metrics["value/explained_variance"] = float("nan") if var == 0 else 1 - (ret - values).var().item() / var
        return metrics

    control.set_state("running")
    try:
        while update < a.updates:
            t0 = time.perf_counter()
            if not rollout():
                break
            t1 = time.perf_counter()
            metrics = learn()
            control.throttle(time.perf_counter() - t1)
            update += 1
            metrics.update(outcomes.summary())
            metrics["perf/decisions_per_s"] = T * N / (time.perf_counter() - t0)
            metrics["actions/invalid_rate"] = invalid_count / (T * N)
            for i, name in enumerate(ACT_NAMES):
                metrics[f"actions/{name}"] = float(act_counts[i] / (T * N))
            act_counts[:], invalid_count = 0, 0
            for k, v in metrics.items():
                writer.add_scalar(k, v, global_step)
            control.update(update=update, global_step=global_step, metrics=metrics)
            if update % a.save_every == 0 or control.take_save():
                checkpoint()
    except KeyboardInterrupt:
        control.note("interrupted")
    finally:
        control.set_state("saving")
        checkpoint()
        envs.close()
        writer.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_ppo -v`
Expected: PASS (3 tests; the smoke test takes ~20–40 s).

- [ ] **Step 5: Try the controls on a short live run**

Run in one terminal: `.venv/Scripts/python -m trainer.ppo --run try --workers 2 --updates 20 --timer 1`
In another: `.venv/Scripts/python -m trainer.ctl status`, `... ctl pause`, open http://127.0.0.1:8787 (state shows
"paused", GPU load drops), `... ctl resume`, `... ctl throttle half` (decisions/s roughly halves), `... ctl stop`.
Expected: the trainer exits after "saved checkpoint"; `checkpoints/try/latest.pt` exists. Then run
`.venv/Scripts/python -m trainer.ppo --run try --resume --updates 25` and confirm it continues from the saved update.
Delete `checkpoints/try` and `runs/try` afterwards.

- [ ] **Step 6: Commit**

```bash
git add trainer/ppo.py trainer/test_ppo.py
git commit -m "PPO training loop with control, checkpoints and resume"
```

---

### Task 11: Fixed-seed evaluation

**Files:**
- Create: `trainer/eval.py`, `trainer/test_eval.py`

**Interfaces:**
- Consumes: `VecEnv`, `Policy`, `RandomPolicy`, `ActionDist`, `env_config`, `parse_args` (Tasks 7, 8, 10)
- Produces: `evaluate(make_policy, config, episodes, workers, games_per_worker, device="cuda", seed="eval") -> dict`,
  `summarize(results) -> dict` with `episodes, win_rate, win_rate_se, mean_reward, mean_reward_se, mean_rank`.

- [ ] **Step 1: Write the failing test** — `trainer/test_eval.py`

```python
import unittest

from trainer.eval import evaluate, summarize
from trainer.model import RandomPolicy

ONION = {"map": "Onion", "compact": True, "nations": 3, "difficulty": "Easy", "bots": 0, "timerMinutes": 1, "seats": 1}


class EvalTest(unittest.TestCase):
    def test_summarize(self):
        s = summarize([{"reward": 1, "rank": 1}, {"reward": -1, "rank": 4}])
        self.assertEqual(s["episodes"], 2)
        self.assertEqual(s["win_rate"], 0.5)
        self.assertEqual(s["mean_reward"], 0.0)
        self.assertEqual(s["mean_rank"], 2.5)

    def test_random_policy_on_fixed_seeds_is_reproducible(self):
        run = lambda: evaluate(RandomPolicy, ONION, episodes=2, workers=1, games_per_worker=2, seed="eval-test")
        first, second = run(), run()
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_eval -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'trainer.eval'`.

- [ ] **Step 3: Implement** — `trainer/eval.py`

```python
"""Evaluate a policy on a fixed set of games (the same seeds for every policy, so results compare directly).

  .venv/Scripts/python -m trainer.eval --policy random --like checkpoints/onion/latest.pt
  .venv/Scripts/python -m trainer.eval --policy checkpoints/onion/latest.pt
"""
import argparse
import json

import numpy as np
import torch

from trainer.env import VecEnv
from trainer.model import ActionDist, Policy, RandomPolicy
from trainer.ppo import env_config, parse_args


def summarize(results):
    n = len(results)
    reward = np.array([r["reward"] for r in results], float)
    win = np.array([r["rank"] == 1 for r in results], float)
    rank = np.array([r["rank"] for r in results], float)
    se = lambda v: float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"episodes": n, "win_rate": float(win.mean()), "win_rate_se": se(win), "mean_reward": float(reward.mean()),
            "mean_reward_se": se(reward), "mean_rank": float(rank.mean())}


def evaluate(make_policy, config, episodes, workers, games_per_worker, device="cuda", seed="eval"):
    """Plays the first ceil(episodes / games) episodes of every game slot; `make_policy(dims)` builds the policy.
    Sampling uses a fixed torch seed, so a given policy always gives the same result."""
    torch.manual_seed(0)
    envs = VecEnv(config, workers, games_per_worker, seed=seed, device=device)
    try:
        policy = make_policy(envs.dims).to(device).eval()
        per_slot = -(-episodes // envs.E)
        counts, results, obs = np.zeros(envs.E, int), [], envs.obs
        while counts.min() < per_slot:
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    out = policy(obs)
                act = ActionDist(out, obs).sample()
            obs, _, _, infos, _ = envs.step(act.cpu().numpy())
            for info in infos:
                if counts[info["env"]] < per_slot:
                    counts[info["env"]] += 1
                    results.append(info)
    finally:
        envs.close()
    return summarize(results)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="'random' or a checkpoint path")
    p.add_argument("--like", help="for --policy random: take the game setup from this checkpoint")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--seed", default="eval")
    a = p.parse_args(argv)
    ckpt_path = a.like if a.policy == "random" else a.policy
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path else None
    train_args = argparse.Namespace(**ckpt["args"]) if ckpt else parse_args([])
    if a.policy == "random":
        make = RandomPolicy
    else:
        def make(dims):
            model = Policy(dims, tuple(int(w) for w in train_args.widths.split(",")))
            model.load_state_dict(ckpt["model"])
            return model.to(memory_format=torch.channels_last)
    result = evaluate(make, env_config(train_args), a.episodes, a.workers, a.games_per_worker, seed=a.seed)
    print(json.dumps({"policy": a.policy, "update": ckpt["update"] if ckpt and a.policy != "random" else None,
                      **result}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_eval -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add trainer/eval.py trainer/test_eval.py
git commit -m "Fixed-seed evaluation for policies and the random baseline"
```

---

### Task 12: Gate run and documentation

**Files:**
- Modify: `docs/PLAN.md` (Phase 1 results), `CLAUDE.md` (training commands), `README.md` (status)

- [ ] **Step 1: Run every test**

Run: `cd OpenFront_Source_Code/openfrontio && node --import tsx --test "../../bridge/*.test.ts" && node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Run: `.venv/Scripts/python -m unittest discover -s trainer -t . -v`
Expected: everything passes.

- [ ] **Step 2: Measure the random baseline**

Run: `.venv/Scripts/python -m trainer.eval --policy random --episodes 256`
Record the JSON (win rate, mean reward, mean rank with standard errors).

- [ ] **Step 3: Train (ask the user first: this keeps the GPU busy for hours; they can pause or throttle it)**

Run in the background: `.venv/Scripts/python -m trainer.ppo --run onion-gate --gpu-temp-limit 83`
Watch http://127.0.0.1:8787 and `.venv/Scripts/tensorboard --logdir runs`. Check early (first 10 minutes) that
`actions/invalid_rate` stays below ~0.2, `loss/approx_kl` below ~0.05 and the entropy decreases slowly; if the win
rate is flat after 1 hour, look at `value/explained_variance` and the action distribution before changing anything.
Stop with `python -m trainer.ctl stop` once the win rate has plateaued or after ~3 hours.

- [ ] **Step 4: Evaluate the checkpoint against the baseline on the same seeds**

Run: `.venv/Scripts/python -m trainer.eval --policy checkpoints/onion-gate/latest.pt --episodes 256`
Gate (PLAN Phase 1): PPO's win rate and mean reward exceed the random baseline's by more than 3 combined standard
errors, and `value/outcome_corr_late` ≥ 0.5 with mid ≥ early (the value head predicts outcomes, better as games
progress). The correlation between value and final reward replaced the sign accuracy `value/outcome_acc_*` (still
logged): the random baseline never wins, so an agent that rarely wins scores ~95% accuracy by always predicting a loss.

- [ ] **Step 5: Record the results and update the docs**

`docs/PLAN.md`, after the Phase 1 section, add "**Phase 1 results (<date>)**" with: throughput (decisions/s, workers ×
games), random vs PPO table (win rate, mean reward, mean rank ± SE), training time to the plateau, value accuracy by
phase, invalid-action rate, and the decisions taken while implementing (node:test instead of vitest; fresh maps +
`resetGameStatics` allow K games per worker; approximate GPU cell masks with exact checks in the worker; per-head
entropy normalisation; control API with the `X-Control` header).

`CLAUDE.md`, add a section:

````markdown
## Training commands (repo root)

```bash
.venv/Scripts/python -m unittest discover -s trainer -t .      # trainer tests (start real Node workers)
.venv/Scripts/python -m trainer.ppo --run onion                 # train; control page http://127.0.0.1:8787
.venv/Scripts/python -m trainer.ppo --run onion --resume        # continue from checkpoints/onion/latest.pt
.venv/Scripts/python -m trainer.ctl pause|resume|save|stop|throttle half|status
.venv/Scripts/python -m trainer.eval --policy checkpoints/onion/latest.pt   # fixed-seed evaluation
.venv/Scripts/tensorboard --logdir runs                         # charts
```
````

`README.md`: status → "Phase 1 (vertical slice) done; Phase 2 (full environment) next." plus one line on how to start
training and open the control page.

- [ ] **Step 6: Commit and push**

```bash
git add docs/PLAN.md CLAUDE.md README.md
git commit -m "Phase 1 gate results"
git push
```
