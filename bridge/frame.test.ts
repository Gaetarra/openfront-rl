import assert from "node:assert/strict";
import { test } from "node:test";
import { TileRef } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameMap";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { encodeUnits, UNIT_BYTES, UNIT_TYPES } from "./frame";
import { EnvGame } from "./game";
import { ONION, script } from "./testkit";

/** One player's unit records, as encodeUnits sends them. */
function records(env: EnvGame, owner: number) {
  const b = encodeUnits(env.game, env.frame);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const all = Array.from({ length: b.length / UNIT_BYTES }, (_, i) => {
    const o = i * UNIT_BYTES;
    const zone = [dv.getUint16(o + 14, true), dv.getUint16(o + 16, true), dv.getUint16(o + 18, true)];
    return {
      type: UNIT_TYPES[b[o]], level: b[o + 1], owner: dv.getUint16(o + 2, true), troops: dv.getFloat32(o + 8, true),
      ready: b[o + 12], flags: b[o + 13], zone, bound: dv.getUint16(o + 20, true),
    };
  });
  return all.filter((r) => r.owner === owner);
}

test("unit records carry the missiles loaded and the disk each unit acts on", async () => {
  const env = await EnvGame.create(ONION, "zones");
  script(env, 260);
  const g = env.game;
  const config = g.config();
  const seat = env.seats[0];
  const [tile] = seat.tiles();
  const enemy = g.players().find((p) => p !== seat && p.isAlive() && p.numTilesOwned() > 0)!;
  const [target] = enemy.tiles();
  seat.buildUnit(UnitType.SAMLauncher, tile, {}).increaseLevel(); // level 2: its new slot reloads first
  seat.buildUnit(UnitType.MissileSilo, tile, {}).launch(); // level 1, its one missile reloading
  seat.buildUnit(UnitType.DefensePost, tile, {});
  seat.buildUnit(UnitType.DefensePost, tile, {}).setUnderConstruction(true);
  seat.buildUnit(UnitType.AtomBomb, tile, { targetTile: target, trajectory: [] });
  seat.buildUnit(UnitType.City, tile, {});
  const at = (t: TileRef, radius: number) => [g.x(t), g.y(t), Math.round(radius)];
  const none = [0, 0, 0];
  const of = (type: UnitType) => records(env, seat.smallID()).filter((r) => r.type === type);
  assert.deepEqual(
    of(UnitType.SAMLauncher).map((r) => [r.level, r.ready, r.zone]),
    [[2, 1, at(tile, config.samRange(1))]],
    "the range is still level 1's as the upgrade starts",
  );
  assert.deepEqual(of(UnitType.MissileSilo).map((r) => [r.ready, r.zone]), [[0, none]]);
  assert.deepEqual(
    of(UnitType.DefensePost).map((r) => [r.level, r.zone]).sort(),
    [[0, none], [1, at(tile, config.defensePostRange())]],
  );
  assert.deepEqual(
    of(UnitType.AtomBomb).map((r) => [r.ready, r.zone]),
    [[0, at(target, config.nukeMagnitudes(UnitType.AtomBomb).outer)]],
  );
  assert.deepEqual(of(UnitType.City).map((r) => [r.ready, r.zone]), [[0, none]]);
  for (let i = 0; i <= config.samUpgradeDuration(); i++) env.tickWith([]);
  assert.deepEqual(of(UnitType.SAMLauncher)[0].zone, at(tile, config.samRange(2)), "then it ramps to level 2's");
});

test("warship records carry health, veteran level and patrol tile; trade ships their destination's owner", async () => {
  const env = await EnvGame.create(ONION, "ships");
  script(env, 260);
  const g = env.game;
  const seat = env.seats[0];
  const water: TileRef[] = [];
  g.forEachTile((t) => { if (g.isOcean(t) && water.length < 3) water.push(t); });
  const ship = seat.buildUnit(UnitType.Warship, water[0], { patrolTile: water[1] });
  const full = ship.maxHealth();
  ship.modifyHealth(-full / 4);
  while (ship.veterancy() < 1) ship.recordKill(UnitType.Warship); // the cap rises, the health stays
  assert.ok(ship.maxHealth() > full, "the record must use the current cap");
  const enemy = g.players().find((p) => p !== seat && p.isAlive() && p.numTilesOwned() > 0)!;
  const port = enemy.buildUnit(UnitType.Port, [...enemy.tiles()][0], {});
  seat.buildUnit(UnitType.TradeShip, water[2], { targetUnit: port }).setSafeFromPirates();
  const of = (type: UnitType) => records(env, seat.smallID()).filter((r) => r.type === type);
  assert.deepEqual(of(UnitType.Warship).map((r) => [r.troops, r.ready, r.zone, r.bound]),
    [[Math.fround(Number(ship.health()) / ship.maxHealth()), ship.veterancy(), [g.x(water[1]), g.y(water[1]), 0], 0]]);
  assert.deepEqual(of(UnitType.TradeShip).map((r) => [r.flags, r.bound]), [[1, enemy.smallID()]]);
});
