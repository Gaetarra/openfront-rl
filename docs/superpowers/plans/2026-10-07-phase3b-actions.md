# Phase 3B, Step 3 (Part 1): Every Human Action (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** StockShip can give every command ranked 1v1 players give, except diplomacy, chat, warship moves and MIRVs:
- 15 action types, with one heat map per cell action;
- a retreat pointer;
- a BC label for every such command.
Older networks keep playing with their five actions.

**Architecture:**
- **Bridge.**
  - 10 new action types: build defense post, silo, factory, port, SAM and warship; atom and hydrogen bombs; upgrade;
    retreat.
  - Type masks (gold, a ready silo, a port, spawn immunity, an upgradable structure, an attack to cancel) and a
    retreat mask.
  - Each action decodes through the game's own `canBuild` / `canUpgradeUnit`.
  - The extractor labels every command.
- **Trainer.**
  - The environment builds 12 cell masks on the GPU and the retreat mask. A cell type stays legal only while its heat
    map has a legal cell.
  - New networks (arch v4): 15 types, 12 heat maps and a retreat pointer, plus the 100 troop shares.
  - `ActionDist` maps each type to its heat map (`MAP_OF`) and reads only the masks of the actions its network knows,
    so the v1–v3 networks (5 types, 3 maps) play unchanged: the wrapper of PLAN 3B.
- **BC and evaluation.**
  - A network learns and is scored only on the rows its actions can express.
  - The bandwidth probe's gold bin gains the 10 gold actions; its troop bin gains retreat.

**Tech Stack:** TypeScript bridge (node:test via tsx), Python 3 + torch (unittest).

**Spec:** `docs/PLAN.md`, 3B, "Actions", "Masks" and "Extractor"; decision 3 (heat maps). The observations
("Observations": nukes in flight, coverage, silo readiness, unit costs) are Part 2, a separate task list. The BC refresh
(3B step 4) comes after both.

## Game facts this plan relies on (checked in v0.34.23)

- **Intents** (`src/core/Schemas.ts`):
  - `build_unit {unit, tile, rocketDirectionUp?, amount?}`;
  - `upgrade_structure {unit, unitId, amount?}`;
  - `cancel_attack {attackID}` (`RetreatExecution`).
- **`PlayerImpl.canBuild(type, tile)`** checks the unit isn't disabled, the gold covers its cost and the player is
  alive; then, by type:
  - structures (city, defense post, silo, factory, SAM): the target tile must be the player's own. It returns the own
    tile nearest it within 15 tiles that keeps the 15-tile spacing;
  - port: an own shore tile within `radiusPortSpawn` of an own target tile;
  - warship: the target must be water, and it returns the nearest active, finished own port in the same water body;
  - atom and hydrogen bombs (`nukeSpawn`): refused during spawn immunity, on impassable tiles and on teammates' land.
    It returns the nearest ready silo: active, not cooling down, finished. A nuke has no range limit.
- **Upgrades:** `findUnitToUpgrade` takes the own unit of that type nearest the tile within 15 tiles.
  `canUpgradeUnit(u)` needs an upgradable type, gold for that type, a finished unit and ownership. Upgradable: city,
  port, factory, silo and SAM; defense posts and warships are not.
- **Attacks:** `Player.outgoingAttacks()` → `Attack {id(), target(): Player | TerraNullius, troops(), isActive(),
  retreating()}`.
- **Construction** takes 2 s (city, factory), 5 s (port, defense post), 10 s (silo) or 30 s (SAM).
- **Test record:** `bridge/testdata/ranked1v1.json.gz` holds cities, factories, a port, silos, atom bombs and defense
  posts, but no SAM, warship, hydrogen bomb, upgrade or cancel. Those labels are tested on a live scripted game.

## Choices to confirm (not fixed by PLAN)

1. Retreat gets its own pointer (a second query over the player tokens), not the attack pointer: "which attack to
   cancel" is a different question from "whom to attack".
2. Nuke heat maps allow every map cell that is less than half own or friendly land. That rules out self-nukes, which
   the game allows.
3. An upgrade raises the own upgradable structure in the chosen cell. A cell holds at most one structure, because of
   the 15-tile spacing.
4. A cell action stays legal only while its heat map has a legal cell. That cuts invalid actions (Gate 3B: < 1%).
5. New networks are arch v4: 15 actions and 100 troop shares. v1–v3 keep 5 actions and 3 heat maps.
6. A network is scored on the rows its actions can express. Networks with different action sets therefore aren't
   comparable on `bc/nll`; the match decides, as Gate 3B does.

## Global Constraints

- No edits inside `OpenFront_Source_Code/openfrontio/`, and no new dependencies.
- Game rules come from the game at runtime (`canBuild`, `canUpgradeUnit`, `unitInfo(...).cost` and `.upgradable`,
  `isUnitDisabled`); none are copied into our code.
- Code whose output reaches the simulation stays deterministic: sorts by distance are stable, with no randomness.
- Commits are the repo owner's: never a Claude attribution line. Nothing is pushed.
- Ask the user before runs longer than ~10 minutes. Tests are fine, even while a night runs.
- Files stay LF; edit with the Edit tool or a byte-preserving script.
- Test commands:
  - trainer: `.venv/Scripts/python -m unittest discover -s trainer -t .` (repo root);
  - bridge: `node --import tsx --test "../../bridge/*.test.ts"` and
    `node node_modules/typescript/bin/tsc -p ../../tsconfig.json` (from `OpenFront_Source_Code/openfrontio/`).
- Baseline: 105 trainer + 35 bridge tests green at 4487e7d.

## Review Focus

1. **Old networks in the new game.** The v1–v3 clones are the opponents of every 3B match. They must sample only their
   5 types and 3 maps. Tests: `test_old_networks_see_only_their_own_actions` and
   `test_old_and_new_networks_play_each_other` (v4 against v1).
2. **Nukes and warships that the game refuses** (spawn immunity, no ready silo, no port in that water body) are masked
   or dropped as counted no-ops, never thrown. Test: bridge "nukes need a ready silo and no spawn immunity; warships a
   finished port".
3. **An upgrade of a structure still under construction, or one the seat can't afford**, is refused through
   `canUpgradeUnit`. Test: bridge "an upgrade raises the seat's structure in the chosen cell".
4. **Stale targets:** an attack that ended between the observation and the decision (latency up to 3 turns), or an
   upgrade label whose unit is gone. Decoding drops it; the extractor marks it unmapped. Tests: bridge "a retreat
   cancels the seat's largest attack on that target" and "every command maps to an action, except diplomacy, chat and
   MIRVs".
5. **Masks agree across the bridge and the GPU:** a cell type is legal only while its heat map has a legal cell. Test:
   `test_a_cell_action_is_legal_only_with_a_legal_cell`.

## Files

| File | Change |
|---|---|
| `bridge/actions.ts` | 15 action types, `BUILDS`, `retreatMask`, type masks, decoding |
| `bridge/game.ts` | the retreat mask in the masks and the observation (blob "retreat") |
| `bridge/replay.ts` | `labelOf` maps every command |
| `bridge/worker.ts` | protocol comment |
| `bridge/{actions,replay,game,worker}.test.ts` | new tests; blob lists gain "retreat" |
| `trainer/env.py` | `MAP_TYPES`; the retreat mask; 12 cell masks; a cell type needs a legal cell |
| `trainer/model.py` | act names; `MAP_OF`; arch v4 (`full=True`); `ActionDist` per type; old networks slice the masks |
| `trainer/bc.py` | label legality per type; rows a network can express; scores per heat map and for retreat |
| `trainer/eval.py` | the probe's bins and the meters cover the new types |
| `trainer/ppo.py` | new runs are v4 |
| `trainer/test_{env,model,bc,eval}.py` | as each task says |
| `CLAUDE.md`, `docs/PLAN.md` | Task 7 |

---
### Task 1: Bridge: 10 new action types, their masks and decoding

**Files:**
- Modify: `bridge/actions.ts`, `bridge/game.ts`
- Test: `bridge/actions.test.ts`, `bridge/game.test.ts` (blob list), `bridge/worker.test.ts` (blob list)

**Interfaces:**
- Produces:
  - `Act.BuildDefensePost = 5` … `Act.Retreat = 14`, `N_ACTS = 15`;
  - `BUILDS: ReadonlyMap<Act, UnitType>`;
  - `retreatMask(seat, slots): Uint8Array`;
  - `typeMask(game, seat, attack, retreat?)`;
  - `decodeAction(game, seat, frame, types, attack, action, retreat?)`;
  - observation blob `"retreat"` (u8 [seats × slots], after `"attack"`).
- The trailing `retreat` defaults to "no attack to cancel", so callers that track no attacks keep working.

- [ ] **Step 1: Write the failing tests**

In `bridge/actions.test.ts`:
- Extend the imports:
  - `import { Player, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";`
  - `import { TileRef } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameMap";`
  - `import { Intent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";`
  - `import { Act, attackMask, decodeAction, retreatMask, typeMask } from "./actions";`
  - `import { EnvGame, seatClientID } from "./game";`
- Add these helpers below `ticks`:

```ts
const ownTile = (seat: Player): TileRef => {
  for (const t of seat.tiles()) return t;
  throw new Error("the seat owns no land");
};
const masksOf = (env: EnvGame, seat: Player) => {
  const attack = attackMask(env.game, seat, env.slots);
  const retreat = retreatMask(seat, env.slots);
  return { types: typeMask(env.game, seat, attack, retreat), attack, retreat };
};
const decode = (env: EnvGame, seat: Player, action: number[]) => {
  const m = masksOf(env, seat);
  return decodeAction(env.game, seat, env.frame, m.types, m.attack, action, m.retreat);
};
const run = (env: EnvGame, intent: Intent) => env.tickWith([{ ...intent, clientID: seatClientID(0) }]);
```

Add the tests:

```ts
test("every structure decodes on the seat's land once it has the gold", async () => {
  const env = await EnvGame.create(ONION, "builds");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(1_000_000_000n);
  const m = masksOf(env, seat);
  for (const act of [Act.BuildCity, Act.BuildDefensePost, Act.BuildSilo, Act.BuildFactory, Act.BuildSAM]) {
    assert.equal(m.types[act], 1, Act[act]);
  }
  assert.equal(m.types[Act.BuildWarship], 0, "no port yet");
  assert.equal(m.types[Act.AtomBomb], 0, "no silo yet");
  assert.equal(m.types[Act.Upgrade], 0, "nothing to upgrade yet");
  const { intent, valid } = decode(env, seat, [Act.BuildSilo, env.frame.cellOf(ownTile(seat)), 0, 0]);
  assert.ok(valid);
  assert.equal(intent?.type === "build_unit" && intent.unit, UnitType.MissileSilo);
});

test("nukes need a ready silo and no spawn immunity; warships a finished port", async () => {
  const env = await EnvGame.create(ONION, "nukes");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(1_000_000_000n);
  run(env, decode(env, seat, [Act.BuildSilo, env.frame.cellOf(ownTile(seat)), 0, 0]).intent!);
  assert.equal(masksOf(env, seat).types[Act.AtomBomb], 0, "the silo is under construction");
  ticks(env, 110);
  for (let i = 0; i < 6000 && env.game.isSpawnImmunityActive(); i++) env.tickWith([]);
  const m = masksOf(env, seat);
  assert.deepEqual([m.types[Act.AtomBomb], m.types[Act.HydrogenBomb]], [1, 1]);
  const enemy = env.game.players().find((p) => p !== seat && p.isAlive() && p.numTilesOwned() > 0)!;
  const { intent, valid } = decode(env, seat, [Act.AtomBomb, env.frame.cellOf(ownTile(enemy)), 0, 0]);
  assert.ok(valid);
  assert.equal(intent?.type === "build_unit" && intent.unit, UnitType.AtomBomb);
  assert.equal(m.types[Act.BuildWarship], 0, "no port");
  const shore = [...seat.tiles()].find((t) => env.game.isShore(t));
  assert.ok(shore !== undefined, "the scripted seat reaches a coast on this seed");
  run(env, decode(env, seat, [Act.BuildPort, env.frame.cellOf(shore), 0, 0]).intent!);
  ticks(env, 60);
  assert.equal(masksOf(env, seat).types[Act.BuildWarship], 1);
  const port = seat.units(UnitType.Port)[0];
  const water = env.game.neighbors(port.tile()).find((t) => env.game.isWater(t))!;
  const ship = decode(env, seat, [Act.BuildWarship, env.frame.cellOf(water), 0, 0]);
  assert.ok(ship.valid);
  assert.equal(ship.intent?.type === "build_unit" && ship.intent.unit, UnitType.Warship);
});

test("an upgrade raises the seat's structure in the chosen cell", async () => {
  const env = await EnvGame.create(ONION, "upgrade");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(1_000_000_000n);
  run(env, decode(env, seat, [Act.BuildCity, env.frame.cellOf(ownTile(seat)), 0, 0]).intent!);
  assert.equal(masksOf(env, seat).types[Act.Upgrade], 0, "a city under construction cannot be upgraded");
  ticks(env, 30);
  assert.equal(masksOf(env, seat).types[Act.Upgrade], 1);
  const city = seat.units(UnitType.City)[0];
  const { intent, valid } = decode(env, seat, [Act.Upgrade, env.frame.cellOf(city.tile()), 0, 0]);
  assert.ok(valid);
  assert.deepEqual(intent, { type: "upgrade_structure", unit: UnitType.City, unitId: city.id() });
  assert.equal(decode(env, seat, [Act.Upgrade, 0, 0, 0]).valid, false, "no structure of the seat in cell 0");
});

test("a retreat cancels the seat's largest attack on that target", async () => {
  const env = await EnvGame.create(ONION, "retreat");
  script(env, 260);
  const seat = env.seats[0];
  run(env, { type: "attack", targetID: null, troops: seat.troops() / 2 });
  env.tickWith([]);
  const m = masksOf(env, seat);
  assert.deepEqual([m.retreat[0], m.types[Act.Retreat]], [1, 1]);
  const biggest = seat.outgoingAttacks().filter((a) => !a.target().isPlayer()).sort((x, y) => y.troops() - x.troops())[0];
  assert.deepEqual(decode(env, seat, [Act.Retreat, 0, 0, 0]).intent, { type: "cancel_attack", attackID: biggest.id() });
  assert.equal(decode(env, seat, [Act.Retreat, 0, 3, 0]).valid, false, "no attack on row 3");
});
```

- In "malformed or unusable actions are dropped, never thrown", the `bad` list gains
  `[Act.Upgrade, -1, 0, 0], [Act.Retreat, 0, 99, 0], [Act.AtomBomb, 1e9, 0, 0], [Act.BuildPort, 0.5, 0, 0]`.
- Every existing `decodeAction(...)` call keeps working, since `retreat` is optional.
- `bridge/game.test.ts` line 104 and `bridge/worker.test.ts` line 68: the blob lists gain `"retreat"` after
  `"attack"`.
- If the seat has no shore with seed "nukes", try "nukes-2", "nukes-3", … and keep the first seed where it has one
  (ledger ruling).

- [ ] **Step 2: Run them to see them fail**

Run (game dir): `node --import tsx --test "../../bridge/*.test.ts"`
Expected: FAIL. `actions.test.ts` does not load (`retreatMask` is not exported), and both blob lists differ.

- [ ] **Step 3: Implement**

`bridge/actions.ts`:
- Replace the `Act` enum, `N_ACTS`, `typeMask` and `decodeAction`; add `BUILDS`, `retreatMask` and the helpers.
- The `Game` import gains `Attack`, `TerraNullius` and `Unit`.

```ts
/** Action types (PLAN 3B "Actions"): Phase 1's five, then every other command ranked 1v1 players give, except
 * diplomacy, chat, warship moves and MIRVs. */
export enum Act {
  Noop = 0,
  Spawn = 1,
  Attack = 2,
  Boat = 3,
  BuildCity = 4,
  BuildDefensePost = 5,
  BuildSilo = 6,
  BuildFactory = 7,
  BuildPort = 8,
  BuildSAM = 9,
  BuildWarship = 10,
  AtomBomb = 11,
  HydrogenBomb = 12,
  Upgrade = 13,
  Retreat = 14,
}
export const N_ACTS = 15;
/** The unit each build action places: structures on own land, a port on own shore, a warship on water, a nuke on any
 * tile the game lets it hit. */
export const BUILDS: ReadonlyMap<Act, UnitType> = new Map([
  [Act.BuildCity, UnitType.City],
  [Act.BuildDefensePost, UnitType.DefensePost],
  [Act.BuildSilo, UnitType.MissileSilo],
  [Act.BuildFactory, UnitType.Factory],
  [Act.BuildPort, UnitType.Port],
  [Act.BuildSAM, UnitType.SAMLauncher],
  [Act.BuildWarship, UnitType.Warship],
  [Act.AtomBomb, UnitType.AtomBomb],
  [Act.HydrogenBomb, UnitType.HydrogenBomb],
]);

/** Which table rows the seat is attacking over land now, for a retreat: row = smallID, row 0 = unowned land. */
export function retreatMask(seat: Player, slots: number): Uint8Array {
  const mask = new Uint8Array(slots);
  for (const a of seat.outgoingAttacks()) {
    const row = rowOf(a.target());
    if (a.isActive() && !a.retreating() && row < slots) mask[row] = 1;
  }
  return mask;
}

/** Which action types are legal now (1 = allowed). `retreat`: retreatMask, or nothing to cancel. */
export function typeMask(game: Game, seat: Player, attack: Uint8Array, retreat = new Uint8Array(0)): Uint8Array {
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
  for (const [act, unit] of BUILDS) {
    m[act] = !game.config().isUnitDisabled(unit) && seat.gold() >= game.unitInfo(unit).cost(game, seat) ? 1 : 0;
  }
  if (!seat.units(UnitType.Port).some(finished)) m[Act.BuildWarship] = 0;
  if (game.isSpawnImmunityActive() || !seat.units(UnitType.MissileSilo).some((s) => finished(s) && !s.isInCooldown())) {
    m[Act.AtomBomb] = m[Act.HydrogenBomb] = 0;
  }
  m[Act.Upgrade] = seat.units().some((u) => u.isActive() && seat.canUpgradeUnit(u)) ? 1 : 0;
  m[Act.Retreat] = retreat.includes(1) ? 1 : 0;
  return m;
}

const finished = (u: Unit) => u.isActive() && !u.isUnderConstruction();
const rowOf = (target: Player | TerraNullius) => (target.isPlayer() ? target.smallID() : 0);

/** Turns a sampled [type, cell, target, share] into a game intent (null intent = no-op). share: the percent of its
 * current troops an attack or boat sends. `retreat`: retreatMask, or nothing to cancel. */
export function decodeAction(
  game: Game,
  seat: Player,
  frame: CellFrame,
  types: Uint8Array,
  attack: Uint8Array,
  action: number[],
  retreat = new Uint8Array(0),
): Decoded {
  const [type, cell, target, share] = action;
  if (!Number.isInteger(type) || types[type] !== 1) return DROP;
  if (type === Act.Noop) return { intent: null, valid: true };
  if (type === Act.Attack) {
    if (!Number.isInteger(target) || attack[target] !== 1 || !isShare(share)) return DROP;
    const targetID = target === 0 ? null : (game.playerBySmallID(target) as Player).id();
    return { intent: { type: "attack", targetID, troops: (share / 100) * seat.troops() }, valid: true };
  }
  if (type === Act.Retreat) {
    if (!Number.isInteger(target) || retreat[target] !== 1) return DROP;
    const largest = seat
      .outgoingAttacks()
      .filter((a) => a.isActive() && !a.retreating() && rowOf(a.target()) === target)
      .reduce<Attack | null>((best, a) => (best === null || a.troops() > best.troops() ? a : best), null);
    return largest === null ? DROP : { intent: { type: "cancel_attack", attackID: largest.id() }, valid: true };
  }
  if (!Number.isInteger(cell) || cell < 0 || cell >= frame.n) return DROP;
  if (type === Act.Spawn) {
    const tile = byCentre(game, frame, cell).find((t) => game.isLand(t) && !game.isImpassable(t) && !game.hasOwner(t));
    return tile === undefined ? DROP : { intent: { type: "spawn", tile }, valid: true };
  }
  if (type === Act.Boat) {
    if (!isShare(share)) return DROP;
    for (const tile of frame.tilesOf(cell)) {
      if (!game.isLand(tile) || game.isImpassable(tile) || !game.isShore(tile)) continue;
      if (game.ownerID(tile) === seat.smallID()) continue;
      if (canBuildTransportShip(game, seat, tile) !== false) {
        return { intent: { type: "boat", dst: tile, troops: (share / 100) * seat.troops() }, valid: true };
      }
    }
    return DROP;
  }
  if (type === Act.Upgrade) {
    // A cell holds at most one structure (the 15-tile spacing), so the seat's upgradable one in it is the choice.
    const unit = seat.units().find((u) => frame.cellOf(u.tile()) === cell && seat.canUpgradeUnit(u));
    if (unit === undefined) return DROP;
    return { intent: { type: "upgrade_structure", unit: unit.type(), unitId: unit.id() }, valid: true };
  }
  const unit = BUILDS.get(type);
  if (unit === undefined) return DROP;
  const tile = byCentre(game, frame, cell).find((t) => seat.canBuild(unit, t) !== false);
  return tile === undefined ? DROP : { intent: { type: "build_unit", unit, tile }, valid: true };
}

/** The cell's tiles, nearest its centre first (a stable sort: ties keep the frame's order). */
function byCentre(game: Game, frame: CellFrame, cell: number): TileRef[] {
  const [cx, cy] = frame.center(cell);
  const d = (t: TileRef) => (game.x(t) - cx) ** 2 + (game.y(t) - cy) ** 2;
  return [...frame.tilesOf(cell)].sort((a, b) => d(a) - d(b));
}
```

- Delete `spawnTile`: `byCentre(...).find(...)` gives the same tile, the unowned land tile nearest the centre, with
  ties in frame order as before.
- Build City now goes through the same `byCentre` + `canBuild` loop as every other build: the centre-most tile that
  can build. The old code took the first own tile in frame order, so a city can land a few tiles from where it used
  to.

`bridge/game.ts`:
- Import `retreatMask`.
- `masks` becomes `{ types: Uint8Array; attack: Uint8Array; retreat: Uint8Array }[] | null`.
- `currentMasks` becomes:

```ts
  private currentMasks(): { types: Uint8Array; attack: Uint8Array; retreat: Uint8Array }[] {
    return this.seats.map((seat) => {
      const attack = attackMask(this.game, seat, this.slots);
      const retreat = retreatMask(seat, this.slots);
      return { types: typeMask(this.game, seat, attack, retreat), attack, retreat };
    });
  }
```

- In `observe`:
  - add `const retreat = new Uint8Array(S * P);` next to `attack`;
  - `retreat.set(m.retreat, i * P);` in the loop;
  - add `["retreat", retreat],` after `["attack", attack],`.
- In `step`, the `decodeAction(...)` call passes `flat.slice(k, k + 4), masks[i].retreat`.

- [ ] **Step 4: Run the bridge tests and the typecheck**

Run (game dir): `node --import tsx --test "../../bridge/*.test.ts"`, then
`node node_modules/typescript/bin/tsc -p ../../tsconfig.json`.
Expected: 39 tests pass (35 + 4 new), and tsc prints nothing.

- [ ] **Step 5: Commit**

```bash
git add bridge
git commit -m "Bridge: every gold action and retreat (PLAN 3B): build defense post, silo, factory, port, SAM and warship, atom and hydrogen bombs, upgrade and retreat, with type masks (gold, a ready silo outside spawn immunity, a finished port, an upgradable structure, an attack to cancel), a retreat mask, and decoding through the game's own canBuild and canUpgradeUnit"
```

---

### Task 2: Bridge: the extractor maps every command

**Files:**
- Modify: `bridge/replay.ts`
- Test: `bridge/replay.test.ts`

**Interfaces:**
- Consumes (Task 1): `Act`, `BUILDS`.
- Produces: `labelOf` returns a label for every command in the action set:
  - build_unit of a `BUILDS` unit → `[act, cell, 0, 0]`;
  - upgrade_structure → `[Act.Upgrade, cell of the unit, 0, 0]`;
  - cancel_attack → `[Act.Retreat, 0, row of its target, 0]`.
  A command naming a unit or attack that no longer exists → `UNMAPPED`. Anything else → `OUT_OF_SET`.

- [ ] **Step 1: Write the failing test**

`bridge/replay.test.ts`:
- Imports gain `UnitType` (from `Game`), `Act` and `BUILDS` (from `./actions`), `labelOf` and `UNMAPPED` (from
  `./replay`), and `ONION` and `script` (from `./testkit`).
- Add:

```ts
test("every command maps to an action, except diplomacy, chat and MIRVs", async () => {
  const env = await EnvGame.create(ONION, "labels");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(1_000_000_000n);
  const clientID = seatClientID(0);
  let home = 0;
  for (const t of seat.tiles()) {
    home = t;
    break;
  }
  const build = (unit: UnitType): StampedIntent => ({ type: "build_unit", unit, tile: home, clientID });
  for (const [act, unit] of BUILDS) {
    assert.deepEqual(labelOf(env, seat, build(unit)), [act, env.frame.cellOf(home), 0, 0], unit);
  }
  assert.deepEqual(labelOf(env, seat, build(UnitType.MIRV)), [OUT_OF_SET, 0, 0, 0]);
  env.tickWith([build(UnitType.City)]);
  const city = seat.units(UnitType.City)[0];
  const upgrade = (unitId: number): StampedIntent => ({ type: "upgrade_structure", unit: UnitType.City, unitId, clientID });
  assert.deepEqual(labelOf(env, seat, upgrade(city.id())), [Act.Upgrade, env.frame.cellOf(city.tile()), 0, 0]);
  assert.deepEqual(labelOf(env, seat, upgrade(999_999)), [UNMAPPED, 0, 0, 0], "a unit that does not exist");
  env.tickWith([{ type: "attack", targetID: null, troops: seat.troops() / 2, clientID }]);
  const attack = seat.outgoingAttacks().find((a) => !a.target().isPlayer())!;
  const cancel = (attackID: string): StampedIntent => ({ type: "cancel_attack", attackID, clientID });
  assert.deepEqual(labelOf(env, seat, cancel(attack.id())), [Act.Retreat, 0, 0, 0]);
  assert.deepEqual(labelOf(env, seat, cancel("gone")), [UNMAPPED, 0, 0, 0], "an attack that already ended");
  const enemy = env.game.players().find((p) => p !== seat)!;
  const ally: StampedIntent = { type: "allianceRequest", recipient: enemy.id(), clientID };
  assert.deepEqual(labelOf(env, seat, ally), [OUT_OF_SET, 0, 0, 0]);
});
```

- [ ] **Step 2: Run it to see it fail**

Run (game dir): `node --import tsx --test ../../bridge/replay.test.ts`
Expected: FAIL. The first build (City) passes, then the defense post returns `[OUT_OF_SET, 0, 0, 0]`.

- [ ] **Step 3: Implement**

`bridge/replay.ts`:
- The import becomes `import { Act, BUILDS } from "./actions";`.
- The `OUT_OF_SET` comment becomes `// a command the actions cannot express: diplomacy, chat, warship moves, MIRVs`.
- Add above `labelOf`:

```ts
/** The action of each unit a build_unit command can place (BUILDS reversed). */
const BUILD_ACT = new Map([...BUILDS].map(([act, unit]) => [unit, act]));
```

In `labelOf`, the `build_unit` case and the new cases become:

```ts
    case "build_unit": {
      const act = BUILD_ACT.get(intent.unit);
      return act === undefined ? [OUT_OF_SET, 0, 0, 0] : [act, frame.cellOf(intent.tile), 0, 0];
    }
    case "upgrade_structure": {
      const unit = game.unit(intent.unitId);
      return unit === undefined ? [UNMAPPED, 0, 0, 0] : [Act.Upgrade, frame.cellOf(unit.tile()), 0, 0];
    }
    case "cancel_attack": {
      const attack = seat.outgoingAttacks().find((a) => a.id() === intent.attackID);
      if (attack === undefined) return [UNMAPPED, 0, 0, 0];
      const target = attack.target();
      return [Act.Retreat, 0, target.isPlayer() ? target.smallID() : 0, 0];
    }
```

- A nuke's `amount` and `rocketDirectionUp`, and an upgrade's `amount`, are dropped: the label is one missile on the
  default arc, or one level (PLAN, "Amounts and the arc are deferred").
- The existing test "extra commands queue into later windows; unexpressible ones stay out of the loss" uses a silo
  as its out-of-set command. A silo is now an action, so that command becomes `{ type: "build_unit", unit:
  UnitType.MIRV, tile: 0, clientID: b }` (still out of the set), and the test's assertions stay as they are.

- [ ] **Step 4: Run the bridge tests and the typecheck**

Run (game dir): `node --import tsx --test "../../bridge/*.test.ts"` and the tsc command.
Expected: 40 tests pass; tsc prints nothing.

- [ ] **Step 5: Commit**

```bash
git add bridge
git commit -m "Extractor: every command ranked players give maps to an action label (builds of every structure, warships, atom and hydrogen bombs, upgrades by the unit's cell, retreats by the attack's target); diplomacy, chat, warship moves and MIRVs stay outside the set, and commands naming a unit or attack that no longer exists are unmapped"
```

---
### Task 3: Environment: 12 cell masks, the retreat mask, and cell actions that need a legal cell

**Files:**
- Modify: `trainer/env.py`
- Test: `trainer/test_env.py`

**Interfaces:**
- Consumes (Task 1): the 15-type `"types"` blob and the `"retreat"` blob.
- Produces:
  - `MAP_TYPES = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)`, the action type of each heat map;
  - obs `"type_mask"` [N, 15], `"cell_mask"` [N, 12, H·W] and `"retreat_mask"` [N, P] (bool).

- [ ] **Step 1: Write the failing test**

`trainer/test_env.py`:
- Import `MAP_TYPES` from `trainer.env` and `Scripted` from `trainer.eval`.
- Add to `VecEnvTest`:

```python
    def test_a_cell_action_is_legal_only_with_a_legal_cell(self):
        env = VecEnv(ONION, workers=1, games_per_worker=2, seed="masks")
        try:
            policy = Scripted()
            for _ in range(80):  # the spawn phase and some 25 s after it
                obs = env.obs
                self.assertEqual(obs["type_mask"].shape[1], 15)
                self.assertEqual(obs["cell_mask"].shape[1], len(MAP_TYPES))
                self.assertEqual(obs["retreat_mask"].shape, obs["attack_mask"].shape)
                self.assertFalse((obs["type_mask"][:, list(MAP_TYPES)] & ~obs["cell_mask"].any(-1)).any())
                env.step(policy(obs))
        finally:
            env.close()
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python -m unittest trainer.test_env -k legal_cell 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'MAP_TYPES'`.

- [ ] **Step 3: Implement**

`trainer/env.py`: add below `DENSITY_FEATURE`:

```python
# The action type of each heat map (bridge/actions.ts Act): spawn, boat, the nine builds (city, defense post, silo,
# factory, port, SAM, warship, atom bomb, hydrogen bomb) and upgrade. Networks before 3B step 3 have the first three.
MAP_TYPES = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
UPGRADABLE = 5  # BUILDING_TYPES[:5] (city, port, factory, silo, SAM) can be upgraded; the worker checks it exactly
```

In `_absorb`:
- add `retreat = np.empty((E, S, P), bool)` next to `attack`;
- read `retreat[e] = np.frombuffer(b["retreat"], np.uint8).reshape(S, P)` after `attack[e] = ...`;
- the call becomes `self._observe(players, rel, scalars, types, attack, retreat, units)`.

In `_observe`:
- the signature becomes `def _observe(self, players, rel, scalars, types, attack, retreat, units):`;
- the docstring's last line becomes `- the scalars and the masks: 15 action types, 12 heat maps (MAP_TYPES), the attack
  and retreat pointers.`;
- replace the `cell_mask = ...` statement with:

```python
        mine, shore = x[:, OWN] > 0, x[:, SHORE] > 0
        upgradable = (x[:, BUILDINGS:BUILDINGS + 4 * UPGRADABLE].unflatten(1, (UPGRADABLE, 4))[:, :, 0] > 0).any(1)
        nukable = x[:, OWN] + x[:, OWN + 1] < 0.5  # less than half own or friendly land: no self-nukes
        cell_mask = torch.stack([x[:, UNOWNED] > 0, shore & (x[:, LAND] - x[:, OWN] > 1e-6),  # spawn, boat
                                 mine, mine, mine, mine, mine & shore, mine,  # city, defense post, silo, factory, port, SAM
                                 x[:, OCEAN] > 0, nukable, nukable, upgradable], 1) & valid[:, None]  # warship, nukes, upgrade
        type_mask = torch.from_numpy(types.reshape(N, self.T)).to(dev)
        type_mask[:, list(MAP_TYPES)] &= cell_mask.any(-1)  # a cell action needs a legal cell on its heat map
```

- in the returned dict, `"type_mask"` becomes `type_mask`, and `"retreat_mask": torch.from_numpy(retreat.reshape(N, P)).to(dev),` follows `"attack_mask"`.

- [ ] **Step 4: Run the env tests**

Run: `.venv/Scripts/python -m unittest trainer.test_env 2>&1 | tail -3`
Expected: OK. The networks' tests still run on fake observations; Task 4 fits them to the new masks.

- [ ] **Step 5: Commit**

```bash
git add trainer/env.py trainer/test_env.py
git commit -m "Environment: 12 cell masks (spawn, boat, city, defense post, silo, factory, port, SAM, warship, atom and hydrogen bombs, upgrade), the retreat mask, and a cell action stays legal only while its heat map has a legal cell"
```

---

### Task 4: Network v4 and the action distribution per type

**Files:**
- Modify: `trainer/model.py`, `trainer/ppo.py` (`network_args`)
- Test: `trainer/test_model.py`, `trainer/test_bc.py` and `trainer/test_ppo.py` (new runs are v4)

**Interfaces:**
- Consumes (Task 3): `MAP_TYPES`, obs `"retreat_mask"`.
- Produces:
  - `ACT_NAMES` (15 names), `RETREAT = 14`, `OLD_ACTS, OLD_MAPS = 5, 3`;
  - `MAP_OF` (long [15]: each type's heat map, −1 without one) and `MAP_NAMES` (12);
  - `Policy(..., full=False)`; with `full=True` the network has 15 types, 12 maps and `out["retreat"]` [N, P];
  - arch `"v4"` (shares + full);
  - `ActionDist.complete(t) -> [N, 4]`, and `ActionDist.retreat` (None for networks without the pointer).

- [ ] **Step 1: Write the failing tests**

`trainer/test_model.py`:
- Imports gain `MAP_OF` and `RETREAT` (from `trainer.model`) and `MAP_TYPES` (from `trainer.env`).
- `fake_obs` gets the full masks. Keep the random draws of the old fields in their old order, so every existing
  expectation holds:

```python
    cell_mask = torch.rand(N, 3, H * W, generator=g) < 0.1
    cell_mask[0] = False  # a row without any legal cell must not produce NaNs
    type_mask = torch.tensor([[1, 1] + [0] * 13] * 3 + [[1, 0, 1, 0, 1] + [0] * 10] * 3, dtype=torch.bool)
```

  and, after `owners` and before building `obs`:

```python
    cell_mask = torch.cat([cell_mask, torch.rand(N, 9, H * W, generator=g) < 0.1], 1)  # the 9 later heat maps
    cell_mask[0] = False
```

  with `"retreat_mask": torch.zeros(N, P, dtype=torch.bool)` added to `obs`.
- `nets()` gains `"v4": Policy(DIMS, shares=True, full=True)`.
- In `test_output_shapes`, the type, map and troop checks become:

```python
                full = name == "v4"
                self.assertEqual(out["type"].shape, (N, 15 if full else 5))
                self.assertEqual(out["maps"].shape, (N, 12 if full else 3, H * W))
                if name in ("v3", "v4"):
                    self.assertEqual(out["shares"].shape, (N, 2, P, SHARES))
                else:
                    self.assertEqual(out["bins"].shape, (N, 2, 5))
                self.assertEqual("retreat" in out, full)
```

- `brute_kl` enumerates every type the network has:

```python
def brute_kl(p_out, q_out, obs, troops):
    """Per row, the sum over every joint action of p log(p / q); `troops` are the percents a troop head can send."""
    hw, players = obs["cell_mask"].shape[-1], obs["attack_mask"].shape[-1]
    acts = [[0, 0, 0, 0]]
    for t in range(1, p_out["type"].shape[-1]):
        if t == ATTACK:
            acts += [[t, 0, p, s] for p in range(players) for s in troops]
        elif t == BOAT:
            acts += [[t, c, 0, s] for c in range(hw) for s in troops]
        elif t == RETREAT:
            acts += [[t, 0, p, 0] for p in range(players)]
        else:
            acts += [[t, c, 0, 0] for c in range(hw)]
    acts = torch.tensor(acts)
    rows = []
    for i in range(len(obs["type_mask"])):
        one = lambda d: {k: v[i:i + 1].expand(len(acts), *v.shape[1:]) for k, v in d.items()}
        lp = ActionDist(one(p_out), one(obs)).log_prob(acts)
        lq = ActionDist(one(q_out), one(obs)).log_prob(acts)
        rows.append(float((lp.exp() * (lp - lq)).sum()))
    return rows


def full_obs(seed=0):
    """small_obs with the masks of a v4 network: 15 types, 12 heat maps and the retreat pointer."""
    g = torch.Generator().manual_seed(seed + 100)
    obs = small_obs(seed)
    obs.update(type_mask=torch.ones(3, 15, dtype=torch.bool), cell_mask=torch.rand(3, 12, 16, generator=g) < 0.6,
               retreat_mask=torch.tensor([[True, False, True, False]] * 3))
    return obs


def full_out(seed=0):
    """Random outputs of a v4 network for full_obs."""
    g = torch.Generator().manual_seed(seed + 100)
    out = share_out(seed)
    out.update(type=torch.randn(3, 15, generator=g), maps=torch.randn(3, 12, 16, generator=g) * 3,
               retreat=torch.randn(3, 4, generator=g))
    return out
```

Add to `ShareDistTest`:

```python
    def test_kl_covers_every_action_of_a_v4_network(self):
        obs, p_out, q_out = full_obs(), full_out(1), full_out(2)
        kl = ActionDist(p_out, obs).kl(ActionDist(q_out, obs))
        for i, want in enumerate(brute_kl(p_out, q_out, obs, range(1, SHARES + 1))):
            self.assertAlmostEqual(float(kl[i]), want, delta=1e-4 * max(1.0, want))

    def test_each_type_takes_its_own_heat_map_and_retreat_its_pointer(self):
        obs, out = full_obs(), full_out()
        dist = ActionDist(out, obs)
        for t in range(15):
            a = dist.complete(torch.full((3,), t))
            m = int(MAP_OF[t])
            for i in range(3):
                if m >= 0:
                    self.assertTrue(obs["cell_mask"][i, m, a[i, 1]], (t, i))
                if t == RETREAT:
                    self.assertTrue(obs["retreat_mask"][i, a[i, 2]])
            self.assertTrue(torch.isfinite(dist.log_prob(a)).all(), t)
        self.assertTrue(torch.isfinite(dist.entropy()).all())

    def test_old_networks_see_only_their_own_actions(self):
        obs, old = full_obs(), share_out()  # a 5-type, 3-map network in a game with 15 types
        dist = ActionDist(old, obs)
        self.assertEqual((dist.type.logits.shape[-1], dist.cells.logits.shape[1], dist.retreat), (5, 3, None))
        for _ in range(20):
            self.assertTrue((dist.sample()[:, 0] < 5).all())
```

In `LoadTest`, the cases gain `({"arch": "v4", "core": "moe", "widths": "64,128,192"}, Policy)`, and the check
`self.assertEqual(getattr(net, "full", False), args.get("arch") == "v4")` is added.

`trainer/test_bc.py` `MainTest.test_train_save_and_resume` and `trainer/test_ppo.py` `SmokeTest`: a new run records
`("v4", "moe")`. `trainer/test_eval.py` `test_old_and_new_networks_play_each_other`: the "new" network becomes
`{"arch": "v4", "core": "moe", "widths": "64,128,192"}`, so a 15-action network plays a Phase 2 one through the bridge.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'MAP_OF'`.

- [ ] **Step 3: Implement**

`trainer/model.py`:
- The import becomes `from trainer.env import BUILDING_TYPES, BUILDINGS, MAP_TYPES, OWNER`.
- The constants:

```python
ACT_NAMES = ["noop", "spawn", "attack", "boat", "build_city", "build_defense_post", "build_silo", "build_factory",
             "build_port", "build_sam", "build_warship", "atom_bomb", "hydrogen_bomb", "upgrade", "retreat"]
SPAWN, ATTACK, BOAT, BUILD, RETREAT = 1, 2, 3, 4, 14
OLD_ACTS, OLD_MAPS = 5, 3  # the action types and heat maps of the networks before 3B step 3 (v1-v3)
MAP_NAMES = ("spawn", "boat", "city", "defense_post", "silo", "factory", "port", "sam", "warship", "atom", "hydrogen",
             "upgrade")  # one per MAP_TYPES entry
MAP_OF = torch.full((len(ACT_NAMES),), -1, dtype=torch.long)
MAP_OF[list(MAP_TYPES)] = torch.arange(len(MAP_TYPES))  # each type's heat map; -1 for types without a cell
MAP_TYPES_T = torch.tensor(MAP_TYPES)
```

- Delete `N_MAPS`.
- `PolicyV1` uses `OLD_MAPS` for `shuffle` and `skip`, and `nn.Linear(512, OLD_ACTS)` for `type_head`.
- `Policy.__init__(self, dims, widths=(64, 128, 192), core="moe", shares=False, full=False)`:
  - `self.full, self.n_maps = full, len(MAP_TYPES) if full else OLD_MAPS` is set before the convolutions;
  - `self.shuffle = nn.Conv2d(c1, self.n_maps * 16, 1)` and `self.skip = nn.Conv2d(c_in, self.n_maps, 1)`;
  - `self.type_head = nn.Linear(512, len(ACT_NAMES) if full else OLD_ACTS)`;
  - after `self.query, self.key = ...`:
    `self.retreat_query = nn.Linear(512, D) if full else None  # the retreat pointer: which target's attack to cancel`;
  - the zero-init tuple becomes
    `(self.type_head, *troop_heads, self.shuffle, self.skip, self.query, *([self.retreat_query] if full else []))`.
- The docstring gains: `- full=True (arch v4, 3B step 3): 15 action types, 12 heat maps (MAP_TYPES) and the retreat
  pointer; otherwise the 5 types and 3 heat maps of the networks before.`
- In `forward`, the pointer lines become:

```python
        keys = self.key(tokens)
        ptr = torch.einsum("nd,npd->np", self.query(h), keys) / math.sqrt(D)
        out.update(type=self.type_head(h), maps=maps.flatten(2), ptr=ptr, value=self.value_head(h).squeeze(1))
        if self.full:
            out["retreat"] = torch.einsum("nd,npd->np", self.retreat_query(h), keys) / math.sqrt(D)
```

(The troop block that follows, shares or bins, stays as it is.)

`RandomPolicy.forward` gains `"retreat": torch.zeros(obs["retreat_mask"].shape, device=dev),`.

The `ActionDist` docstring and the methods that change. `__init__` reads the network's own masks:

```python
    def __init__(self, out, obs):
        n_types, n_maps = out["type"].shape[-1], out["maps"].shape[1]
        self.type = _masked(out["type"].float(), obs["type_mask"][:, :n_types])
        self.cells = _masked(out["maps"].float(), obs["cell_mask"][:, :n_maps])  # batch [N, maps]
        self.ptr = _masked(out["ptr"].float(), obs["attack_mask"])
        self.retreat = _masked(out["retreat"].float(), obs["retreat_mask"]) if "retreat" in out else None
        self.map_types = MAP_TYPES_T[:n_maps].to(out["type"].device)  # the action type of each heat map
        self.shares = "shares" in out
        if self.shares:
            self.troops = Categorical(logits=out["shares"].float())  # batch [N, 2, P]: attack, boat x token
            self.owner = obs["owners"][:, 0].long()  # [N, cells]: each cell's main owner
        else:
            self.troops = Categorical(logits=out["bins"].float()[:, :, None])  # batch [N, 2, 1]
```

The class docstring's first paragraph becomes:

```
    """The joint action [type, cell, target, troops]: the type picks which argument heads are used.

    Every type in MAP_TYPES takes a cell on its own heat map (MAP_OF): spawn, boat, the nine builds and upgrade. attack:
    the player pointer + troops; boat: troops too; retreat: the retreat pointer (whose attack to cancel). A network
    reads the masks of the actions it has: the networks before 3B step 3 have 5 types, 3 heat maps and no retreat
    pointer, so they play on in a game with 15 types.
```

(The `troops:` lines that follow stay as they are.)

`sample`, `complete`, `log_prob`, `entropy` and `kl`:

```python
    def complete(self, t):
        """Actions [N, 4] of types t, their arguments sampled from this policy."""
        c, p = self.cells.sample(), self.ptr.sample()
        r = self.retreat.sample() if self.retreat is not None else torch.zeros_like(p)
        m = MAP_OF.to(t.device)[t]
        zero = torch.zeros_like(t)
        cell = torch.where(m >= 0, c.gather(1, m.clamp(min=0)[:, None])[:, 0], zero)
        target = torch.where(t == ATTACK, p, torch.where(t == RETREAT, r, zero))
        pct = self.troop_sample(p, c[:, 1])
        troops = torch.where(t == ATTACK, pct[:, 0], torch.where(t == BOAT, pct[:, 1], zero))
        return torch.stack([t, cell, target, troops], 1)

    def sample(self):
        return self.complete(self.type.sample())

    def log_prob(self, actions, smooth=0.0, sizes=False):
        """log pi(actions) [N]; `smooth` and `sizes` as in troop_log_prob. Every type must be one this network has."""
        t, cell, target, troops = actions.unbind(1)
        m = MAP_OF.to(t.device)[t]
        lc = self.cells.log_prob(cell[:, None].expand(-1, self.cells.logits.shape[1]))  # [N, maps]
        lt = self.troop_log_prob(target, cell, troops, smooth, sizes)
        zero = torch.zeros_like(lc[:, 0])
        lr = self.retreat.log_prob(target) if self.retreat is not None else zero
        return (self.type.log_prob(t)
                + torch.where(m >= 0, lc.gather(1, m.clamp(min=0)[:, None])[:, 0], zero)
                + torch.where(t == ATTACK, self.ptr.log_prob(target) + lt[:, 0], zero)
                + torch.where(t == BOAT, lt[:, 1], zero)
                + torch.where(t == RETREAT, lr, zero))

    def entropy(self):
        (docstring unchanged)
        hc = self.cells.entropy() / math.log(self.cells.logits.shape[-1])  # [N, maps]
        hp = self.ptr.entropy() / math.log(max(self.ptr.logits.shape[-1], 2))
        ht = (self._weights().detach() * self.troops.entropy()).sum(2) / math.log(self.troops.logits.shape[-1])
        pt = self.type.probs.detach()
        h = (self.type.entropy() / math.log(pt.shape[-1]) + (pt[:, self.map_types] * hc).sum(1)
             + pt[:, ATTACK] * (hp + ht[:, 0]) + pt[:, BOAT] * ht[:, 1])
        if self.retreat is not None:
            h = h + pt[:, RETREAT] * self.retreat.entropy() / math.log(max(self.retreat.logits.shape[-1], 2))
        return h

    def kl(self, ref):
        (docstring unchanged)
        if self.shares != ref.shares or self.type.logits.shape != ref.type.logits.shape:
            raise ValueError("the KL needs two networks with the same troop head and actions")
        kc = _kl(self.cells, ref.cells)  # [N, maps]
        kt = (self._weights() * _kl(self.troops, ref.troops)).sum(2)  # [N, 2]
        pt = self.type.probs
        k = (_kl(self.type, ref.type) + (pt[:, self.map_types] * kc).sum(1)
             + pt[:, ATTACK] * (_kl(self.ptr, ref.ptr) + kt[:, 0]) + pt[:, BOAT] * kt[:, 1])
        if self.retreat is not None:
            k = k + pt[:, RETREAT] * _kl(self.retreat, ref.retreat)
        return k
```

- For the old networks the sums are the old terms: `map_types` = (spawn, boat, city). The random draws come in the
  old order (type, cells, pointer, troops), so their games replay as before.
- `arch_of`'s docstring adds "v4 (Phase 3B step 3, 100 troop shares and 15 actions)". `new_net` builds
  `Policy(dims, widths, spec["core"], shares=spec["arch"] in ("v3", "v4"), full=spec["arch"] == "v4")`.
- `describe` reads:

```python
    troops = "100 troop shares" if spec["arch"] in ("v3", "v4") else "5 troop sizes"
    actions = ", 15 actions" if spec["arch"] == "v4" else ""
    return f"player transformer, {CORE_NAMES[spec['core']]}, {troops}{actions}"
```

`trainer/ppo.py` `network_args`: `a.arch = "v4"`. The docstring reads "a Phase 3B network (100 troop shares, 15
actions)".

- [ ] **Step 4: Run every trainer test**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . 2>&1 | tail -4`
Expected: OK. If the PPO smoke test or eval tests index `ACT_NAMES` with a 5-type `bincount`, give them
`minlength=len(ACT_NAMES)` (ledger ruling).

- [ ] **Step 5: Commit**

```bash
git add trainer
git commit -m "Network v4 (PLAN 3B): 15 action types, 12 heat maps and a retreat pointer next to the 100 troop shares; ActionDist maps each type to its heat map, samples, scores and takes exact entropies and KLs over every action, and reads only the masks of the actions a network has, so the v1-v3 networks play on unchanged; new runs are v4"
```

---

### Task 5: BC: labels and scores for every action

**Files:**
- Modify: `trainer/bc.py`
- Test: `trainer/test_bc.py`

**Interfaces:**
- Consumes (Task 4): `MAP_OF`, `MAP_NAMES`, `RETREAT`, `ACT_NAMES`, `dist.retreat`; (Task 3) obs `"retreat_mask"`.
- Produces:
  - `label_legal` checks each type's heat map cell and the retreat pointer;
  - a network learns and is scored only on the rows its types cover;
  - new metrics: `labels/<type>` (the label mix), `labels/beyond_net`, `bc/<map>_cell_*` for all 12 maps, and
    `bc/retreat_target_acc` and `bc/nll_retreat_target`.

- [ ] **Step 1: Write the failing tests**

`trainer/test_bc.py`: the imports from `trainer.model` gain `ACT_NAMES` and `RETREAT`. Add to `MeasureTest`:

```python
    def test_new_labels_need_their_heat_map_cell_or_an_attack_to_cancel(self):
        obs = fake_obs()
        obs["type_mask"][:] = True
        obs["cell_mask"][:] = False
        obs["cell_mask"][:, 4, 7] = True  # the silo map allows cell 7 only
        obs["retreat_mask"][:, 3] = True  # the seat attacks player 3
        labels = torch.tensor([[6, 7, 0, 0], [6, 8, 0, 0], [RETREAT, 0, 3, 0], [RETREAT, 0, 2, 0], [13, 7, 0, 0],
                               [-1, 0, 0, 0]], device="cuda")
        self.assertEqual(label_legal(obs, labels).tolist(), [True, False, True, False, False, False])

    def test_a_network_learns_and_is_scored_only_on_the_rows_its_actions_express(self):
        obs = fake_obs()
        obs["type_mask"][:] = True
        obs["cell_mask"][:] = True
        old = RandomPolicy5()(obs)
        labels = torch.tensor([[6, 7, 0, 0]] * 3 + [[4, 7, 0, 0]] * 3, device="cuda")  # 3 silos, 3 cities
        legal = label_legal(obs, labels)
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, legal, ActionDist(old, obs), zero, zero)
        s = m.means()
        self.assertAlmostEqual(s["labels/beyond_net"], 0.5)
        self.assertAlmostEqual(s["labels/build_silo"], 0.5)
        self.assertAlmostEqual(s["bc/nll_type"], math.log(5), places=4)  # only the 3 city rows, over 5 types
```

with this helper above the class (a 5-type network's uniform outputs, as `RandomPolicy` gives a v4 network's):

```python
class RandomPolicy5(torch.nn.Module):
    """Uniform outputs of a network with the 5 types and 3 heat maps of before 3B step 3."""

    def forward(self, obs):
        n, dev = len(obs["type_mask"]), obs["type_mask"].device
        return {"type": torch.zeros(n, 5, device=dev), "maps": torch.zeros(n, 3, obs["cell_mask"].shape[-1], device=dev),
                "ptr": torch.zeros(obs["attack_mask"].shape, device=dev), "bins": torch.zeros(n, 2, 5, device=dev),
                "value": torch.zeros(n, device=dev)}
```

`ScoreTest` gains the checkpoint `("full", {"arch": "v4", "core": "moe", "widths": "64,128,192"})`. The unpacking
becomes `old, moe, shares, full = first["metrics"]`, plus `self.assertIn("labels/build_silo", full)`.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -k MeasureTest 2>&1 | tail -4`
Expected: FAIL. `label_legal` ignores the silo map and the retreat mask, and `labels/beyond_net` is missing.

- [ ] **Step 3: Implement**

`trainer/bc.py`:
- The model import gains `ACT_NAMES, MAP_NAMES, MAP_OF, RETREAT`.
- Delete `CELL_HEAD`.
- `label_legal` becomes:

```python
def label_legal(obs, labels):
    """Rows whose label can enter the loss: an action type (not a command outside the set) that the observation's
    masks allow, with a legal cell on its heat map, a legal attack target or an attack to cancel."""
    t, cell, target = labels[:, 0], labels[:, 1], labels[:, 2]
    rows = torch.arange(len(t), device=t.device)
    ok = (t >= 0) & obs["type_mask"][rows, t.clamp(min=0)]
    m = MAP_OF.to(t.device)[t.clamp(min=0)]
    cell_ok = obs["cell_mask"][rows, m.clamp(min=0), cell.clamp(0, obs["cell_mask"].shape[-1] - 1)]
    ok &= torch.where(m >= 0, cell_ok, True)
    target = target.clamp(0, obs["attack_mask"].shape[-1] - 1)
    ok &= torch.where(t == ATTACK, obs["attack_mask"][rows, target], True)
    return ok & torch.where(t == RETREAT, obs["retreat_mask"][rows, target], True)
```

- In `bc_loss`, after `dist = ActionDist(out, obs)`:
  `legal = legal & (labels[:, 0] < dist.type.logits.shape[-1])  # rows this network's actions can express`.
- In `measure`:
  - the label statistics gain `for k, name in enumerate(ACT_NAMES): m.add(f"labels/{name}", (t == k).sum(), n)`;
  - after `labels/illegal`, add:

```python
    n_types = dist.type.logits.shape[-1]
    m.add("labels/beyond_net", (legal & (t >= n_types)).sum(), n)
    legal = legal & (t < n_types)  # rows this network's actions can express
```

    and `n, nl = len(t), legal.sum()` moves below it;
  - `lc = dist.cells.log_prob(cell[:, None].expand(-1, dist.cells.logits.shape[1]))`;
  - the cell loop runs over the network's heat maps:

```python
    for k in range(dist.cells.logits.shape[1]):
        name, rows = MAP_NAMES[k], legal & (t == MAP_TYPES[k])
        hit = dist.cells.logits[:, k].argmax(1) == cell
        m.add(f"bc/{name}_cell_acc", (hit & rows).sum(), rows.sum())
        m.add(f"bc/nll_{name}_cell", -(lc[:, k] * rows).sum(), rows.sum())
        p = dist.cells.probs[:, k]  # [N, cells]
        m.add(f"bc/{name}_cell_near", ((p * (tiles <= NEAR_TILES)).sum(1) * rows).sum(), rows.sum())
        m.add(f"bc/{name}_cell_dist", ((p * tiles).sum(1) * rows).sum(), rows.sum())
    if dist.retreat is not None:
        rows = legal & (t == RETREAT)
        m.add("bc/retreat_target_acc", ((dist.retreat.logits.argmax(1) == tgt) & rows).sum(), rows.sum())
        m.add("bc/nll_retreat_target", -(dist.retreat.log_prob(tgt) * rows).sum(), rows.sum())
```

    (`MAP_TYPES` comes from `trainer.env`.) The first three maps keep their metric names (spawn, boat, city), so
    older score files stay comparable;
  - the docstring gains: `- rows a network cannot express (a type it lacks) are left out of its scores and counted
    (labels/beyond_net), so networks with different action sets are not comparable on bc/nll;`.

- [ ] **Step 4: Run every trainer test**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . 2>&1 | tail -4`
Expected: OK. `LabelTest.test_label_legal` builds its own masks: if it fails because of the 15-type or 12-map
layout, extend its masks the same way as `fake_obs` (ledger ruling).

- [ ] **Step 5: Commit**

```bash
git add trainer/bc.py trainer/test_bc.py
git commit -m "BC for every action: labels are legal with their heat map's cell or an attack to cancel, a network learns and is scored only on the rows its types cover (older networks skip the new types), and the scores add the label mix per type, the 12 heat maps and the retreat pointer"
```

---

### Task 6: The bandwidth probe and the meters cover the new types

**Files:**
- Modify: `trainer/eval.py`
- Test: `trainer/test_eval.py`

**Interfaces:**
- Consumes (Task 4): `ActionDist.complete`, `RETREAT`.
- Produces:
  - `TROOP_TYPES = [ATTACK, BOAT, RETREAT]`;
  - `GOLD_TYPES = list(range(BUILD, RETREAT))` (city … upgrade);
  - `two_bin_sample` draws one troop and one gold action through `complete`.

- [ ] **Step 1: Write the failing test**

Add to `TwoBinTest`:

```python
    def test_the_gold_bin_takes_every_gold_action(self):
        torch.manual_seed(0)
        n = 2000
        out = {"type": torch.full((n, 15), -30.0), "maps": torch.zeros(n, 12, 4), "ptr": torch.zeros(n, 3),
               "retreat": torch.zeros(n, 3), "bins": torch.zeros(n, 2, 5)}
        out["type"][:, [0, 6]] = 0.0  # no-op or a silo, half each
        obs = {"type_mask": torch.ones(n, 15, dtype=torch.bool), "cell_mask": torch.ones(n, 12, 4, dtype=torch.bool),
               "attack_mask": torch.ones(n, 3, dtype=torch.bool), "retreat_mask": torch.ones(n, 3, dtype=torch.bool)}
        obs["type_mask"][:, SPAWN] = False
        a = two_bin_sample(ActionDist(out, obs), obs)
        self.assertAlmostEqual((a[:, 4] == 6).float().mean().item(), 0.5, delta=0.04)
        self.assertEqual(a[:, 0].abs().sum().item(), 0, "no troop action has any weight")
```

(`SPAWN` joins the model import.)

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python -m unittest trainer.test_eval -k gold_bin 2>&1 | tail -3`
Expected: FAIL. Today's gold bin knows only BUILD, so the silo never appears in the second slot.

- [ ] **Step 3: Implement**

`trainer/eval.py`:
- The import gains `RETREAT`.
- The constants:

```python
TROOP_TYPES = [ATTACK, BOAT, RETREAT]  # the two kinds of action the bins would split (PLAN decision 11)
GOLD_TYPES = list(range(BUILD, RETREAT))  # the nine builds and upgrade
```

- In `Cooldown`, where `probs[:, TROOP_TYPES]` and `probs[:, GOLD_TYPES]` appear, use
  `[k for k in TROOP_TYPES if k < probs.shape[1]]` and the same for `GOLD_TYPES` (old networks have 5 types).
- `two_bin_sample` becomes:

```python
def two_bin_sample(dist, obs):
    """The upward probe of the bandwidth test (PLAN 3B): per decision a troop action (attack, boat, retreat or no-op)
    and then a gold action (a build, a nuke, an upgrade or no-op), [n, 8]. Each kind keeps the policy's own
    probability, as bins trained by imitation would, so the bot acts as often as before and only acting in parallel
    is new. In the spawn phase the first action is an ordinary sample and the second a no-op."""
    p = dist.type.probs
    u = torch.rand(len(p), 2, device=p.device)
    first, second = dist.complete(pick(p, TROOP_TYPES, u[:, 0])), dist.complete(pick(p, GOLD_TYPES, u[:, 1]))
    single = torch.cat([dist.sample(), torch.zeros_like(first)], 1)
    return torch.where(obs["type_mask"][:, SPAWN, None], single, torch.cat([first, second], 1))


def pick(probs, kinds, u):
    """Per row, one of `kinds` with the policy's own probability of it (u uniform in [0, 1)), else no-op (0)."""
    kinds = [k for k in kinds if k < probs.shape[1]]
    i = (u[:, None] >= probs[:, kinds].cumsum(1)).sum(1)
    return torch.tensor(kinds + [0], device=probs.device)[i]
```

- [ ] **Step 4: Run every trainer test**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . 2>&1 | tail -4`
Expected: OK. The existing two-bin tests (5 types) keep their numbers.

- [ ] **Step 5: Commit**

```bash
git add trainer/eval.py trainer/test_eval.py
git commit -m "Bandwidth probe and meters over every action: the troop bin is attack, boat and retreat, the gold bin the nine builds and upgrade, and both draw their arguments through ActionDist.complete"
```

---

### Task 7: Docs, and the coverage check

**Files:**
- Modify: `CLAUDE.md`, `docs/PLAN.md`

- [ ] **Step 1: CLAUDE.md**

The "Networks:" bullet reads:

```
- Networks: new runs record theirs in the checkpoint (`args["arch"]`, `"core"`): v1 is Phase 2, v2 is Phase 3A (5 troop
  sizes), v3 is Phase 3B (100 troop shares), v4 adds every action (15 types, 12 heat maps, a retreat pointer);
  `--init` keeps the checkpoint's. A checkpoint without `"arch"` (Phase 2) loads as `PolicyV1`, which rebuilds its 22
  input channels from today's 49 (`v1_spatial`), so the Phase 2 bots still play. v1–v3 see only their 5 types and 3
  heat maps of the 15 (`ActionDist`), and BC scores each network on the rows its types can express.
```

A new bullet after the troops bullet:

```
- Actions (bridge/actions.ts `Act`): no-op, spawn, attack, boat, build city / defense post / silo / factory / port /
  SAM / warship, atom bomb, hydrogen bomb, upgrade (the seat's structure in the chosen cell) and retreat (cancels
  the largest attack on the chosen target).
  - The bridge masks types by gold, a ready silo outside spawn immunity, a finished port, an upgradable structure and
    an attack to cancel; the GPU masks the cells.
  - The game's `canBuild` / `canUpgradeUnit` decide in the worker, and a failed action is a counted no-op.
  - Outside the set: diplomacy, chat, warship moves and MIRVs.
```

- [ ] **Step 2: docs/PLAN.md**

In 3B, under "Actions", add:

```
  - **Code, part 1 (2026-10-07)**, `docs/superpowers/plans/2026-10-07-phase3b-actions.md`:
    - 15 types, 12 heat maps (spawn, boat, nine builds, upgrade) and a retreat pointer (arch v4);
    - masks in the bridge (types) and on the GPU (cells); a cell type needs a legal cell;
    - the extractor labels every command except diplomacy, chat, warship moves and MIRVs;
    - the v1–v3 networks keep their 5 types and 3 maps;
    - choices: retreat has its own pointer; nukes may target cells under half own or friendly land; an upgrade raises
      the seat's structure in the chosen cell.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/PLAN.md
git commit -m "Docs for 3B step 3, part 1: every action (15 types, 12 heat maps, a retreat pointer), masks, the extractor and how older networks play on"
```

- [ ] **Step 4: Coverage check (~10 min with 12 workers and the GPU; ask the user first, and not during a night)**

Run: `.venv/Scripts/python -m trainer.bc --score checkpoints/bc3b/latest.pt`
Read:
- `labels/out_of_set`: Gate 3B wants < 0.5%, against ~6% with the five actions;
- `labels/unmapped`;
- the label mix (`labels/<type>`).
Write the numbers into PLAN 3B. If out-of-set is ≥ 0.5%, list the commands behind it before Part 2.

---

## After this plan

Part 2, the observations, comes next as its own task list:
- nukes in flight and their targets;
- SAM and defense-post coverage (ready interceptors by relation);
- silo readiness;
- unit costs as scalars;
- the speed check on arch v4.
Then the BC refresh (3B step 4) and Gate 3B.
