import assert from "node:assert/strict";
import { test } from "node:test";
import { TileRef } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameMap";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { Intent } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { Player } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/Game";
import { Act, attackMask, decodeAction, N_ACTS, seatMasks, typeMask } from "./actions";
import { EnvGame, seatClientID } from "./game";
import { ONION, script, spawnCell } from "./testkit";

const ticks = (env: EnvGame, n: number) => {
  for (let i = 0; i < n; i++) env.tickWith([]);
};
const ownTile = (seat: Player): TileRef => {
  for (const t of seat.tiles()) return t;
  throw new Error("the seat owns no land");
};
const masksOf = (env: EnvGame, seat: Player) => seatMasks(env.game, seat, env.slots);
const decode = (env: EnvGame, seat: Player, action: number[]) =>
  decodeAction(env.game, seat, env.frame, masksOf(env, seat), action);
const run = (env: EnvGame, intent: Intent) => env.tickWith([{ ...intent, clientID: seatClientID(0) }]);

test("spawn is open early, forced late and closed at the very end of the spawn phase", async () => {
  const env = await EnvGame.create(ONION, "masks");
  const seat = env.seats[0];
  const types = () => [...typeMask(env.game, seat, attackMask(env.game, seat, env.slots))];
  const only = (noop: number, spawn: number) => [noop, spawn, ...new Array(N_ACTS - 2).fill(0)];
  ticks(env, 100);
  assert.deepEqual(types(), only(1, 1));
  ticks(env, 85); // tick 185
  assert.deepEqual(types(), only(0, 1));
  ticks(env, 12); // tick 197
  assert.deepEqual(types(), only(1, 0));
});

test("spawn decodes to unowned land inside the chosen cell", async () => {
  const env = await EnvGame.create(ONION, "spawn");
  const seat = env.seats[0];
  const cell = spawnCell(env);
  const { intent, valid } = decode(env, seat, [Act.Spawn, cell, 0, 0]);
  assert.ok(valid && intent !== null && intent.type === "spawn");
  if (intent?.type !== "spawn") return;
  assert.equal(env.frame.cellOf(intent.tile), cell);
  assert.ok(env.game.isLand(intent.tile) && !env.game.hasOwner(intent.tile));
});

test("after spawning, unowned land is attackable and the troop share sets the troops", async () => {
  const env = await EnvGame.create(ONION, "attack");
  script(env, 260);
  const seat = env.seats[0];
  const m = masksOf(env, seat);
  const attack = m.attack;
  assert.equal(attack[0], 1);
  assert.equal(m.types[Act.Attack], 1);
  const { intent, valid } = decodeAction(env.game, seat, env.frame, m, [Act.Attack, 0, 0, 37]);
  assert.ok(valid);
  assert.deepEqual(intent, { type: "attack", targetID: null, troops: 0.37 * seat.troops() });
  const far = [...attack.keys()].find((s) => s > 1 && attack[s] === 0);
  if (far !== undefined) {
    assert.equal(decodeAction(env.game, seat, env.frame, m, [Act.Attack, 0, far, 10]).valid, false);
  }
});

test("the five sizes of earlier networks send what they sent before", async () => {
  const env = await EnvGame.create(ONION, "sizes");
  script(env, 260);
  const seat = env.seats[0];
  const m = masksOf(env, seat);
  for (const [share, fraction] of [[10, 0.1], [25, 0.25], [50, 0.5], [75, 0.75], [100, 1.0]] as const) {
    const { intent } = decodeAction(env.game, seat, env.frame, m, [Act.Attack, 0, 0, share]);
    assert.deepEqual(intent, { type: "attack", targetID: null, troops: fraction * seat.troops() }, `${share}%`);
  }
});

test("build city: valid on own land with gold, dropped elsewhere", async () => {
  const env = await EnvGame.create(ONION, "city");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(10_000_000n);
  const m = masksOf(env, seat);
  assert.equal(m.types[Act.BuildCity], 1);
  const own = env.frame.cellOf([...seat.tiles()][0]);
  const ok = decodeAction(env.game, seat, env.frame, m, [Act.BuildCity, own, 0, 0]);
  assert.equal(ok.valid, true);
  assert.equal(ok.intent?.type === "build_unit" && ok.intent.unit, UnitType.City);
  assert.equal(decodeAction(env.game, seat, env.frame, m, [Act.BuildCity, 0, 0, 0]).valid, false);
});

test("malformed or unusable actions are dropped, never thrown", async () => {
  const env = await EnvGame.create(ONION, "robust");
  script(env, 260);
  const seat = env.seats[0];
  const m = masksOf(env, seat);
  const bad = [[99, 0, 0, 0], [Act.Attack, 0, 99, 0], [Act.Attack, 0, 0, 0], [Act.Attack, 0, 0, 101], [Act.Attack, 0, 0, 2.5], [Act.Boat, 0, 0, 0], [Act.Spawn, 0, 0, 0], [Act.BuildCity, -1, 0, 0], [Act.BuildCity, 1e9, 0, 0], [], [Act.Upgrade, -1, 0, 0], [Act.Retreat, 0, 99, 0], [Act.AtomBomb, 1e9, 0, 0], [Act.BuildPort, 0.5, 0, 0]];
  for (const a of bad) assert.equal(decodeAction(env.game, seat, env.frame, m, a).valid, false, `${a}`);
  for (let c = 0; c < env.frame.n; c += 7) decodeAction(env.game, seat, env.frame, m, [Act.Boat, c, 0, 25]);
});

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

test("nukes need a ready silo and no spawn immunity", async () => {
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
  assert.ok(decode(env, seat, [Act.AtomBomb, env.frame.cellOf(ownTile(seat)), 0, 0]).valid, "self-nukes stay legal");
});

// Its own game: at tick 260 the seat holds its spawn land only, too small for a silo and a port 15 tiles apart.
test("warships need a finished port", async () => {
  const env = await EnvGame.create(ONION, "warships");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(1_000_000_000n);
  assert.equal(masksOf(env, seat).types[Act.BuildWarship], 0, "no port");
  const shore = [...seat.tiles()].find((t) => env.game.isShore(t));
  assert.ok(shore !== undefined, "the scripted seat reaches a coast on this seed");
  run(env, decode(env, seat, [Act.BuildPort, env.frame.cellOf(shore), 0, 0]).intent!);
  assert.equal(masksOf(env, seat).types[Act.BuildWarship], 0, "the port is under construction");
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
