// ranked1v1.json.gz: ranked 1v1 game ct4rBkTH3R, Normal Australia, v0.35.2; usernames, account IDs and tribe names
// blanked.
import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { StampedIntent, Turn } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/packages/engine-lib/src/PseudoRandom";
import { decompressGameRecord } from "../OpenFront_Source_Code/openfrontio/packages/shared/src/SharedUtil";
import { GameRecord, GameRecordSchema } from "../OpenFront_Source_Code/openfrontio/packages/shared/src/WireSchemas";
import { Act, BUILDS } from "./actions";
import { EnvGame, seatClientID } from "./game";
import { labelOf, OUT_OF_SET, readRecord, RecordGame, shareOf, UNMAPPED } from "./replay";
import { gameConfigOf } from "./setups";
import { landTile, ONION, script } from "./testkit";

const FIXTURE = path.join(path.dirname(fileURLToPath(import.meta.url)), "testdata/ranked1v1.json.gz");

test("a label's troop share is the whole percent the player sent", () => {
  assert.equal(shareOf(20, 100), 20);
  assert.equal(shareOf(19.8, 100), 20); // troops change a little between the click and its execution
  assert.equal(shareOf(1000, 4000), 25);
  assert.equal(shareOf(0.3, 100), 1); // a trickle still sends 1%
  assert.equal(shareOf(130, 100), 100); // troops fell since the click
  assert.equal(shareOf(5, 0), 100);
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
  const silo: StampedIntent = { type: "build_unit", unit: UnitType.MIRV, tile: 0, clientID: b };
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
  env.tickWith([]); // an intent executes on the tick after the one that receives it
  const city = seat.units(UnitType.City)[0];
  const upgrade = (unitId: number): StampedIntent => ({ type: "upgrade_structure", unit: UnitType.City, unitId, clientID });
  assert.deepEqual(labelOf(env, seat, upgrade(city.id())), [Act.Upgrade, env.frame.cellOf(city.tile()), 0, 0]);
  assert.deepEqual(labelOf(env, seat, upgrade(999_999)), [UNMAPPED, 0, 0, 0], "a unit that does not exist");
  env.tickWith([{ type: "attack", targetID: null, troops: seat.troops() / 2, clientID }]);
  env.tickWith([]);
  const attack = seat.outgoingAttacks().find((a) => !a.target().isPlayer())!;
  const cancel = (attackID: string): StampedIntent => ({ type: "cancel_attack", attackID, clientID });
  assert.deepEqual(labelOf(env, seat, cancel(attack.id())), [Act.Retreat, 0, 0, 0]);
  assert.deepEqual(labelOf(env, seat, cancel("gone")), [UNMAPPED, 0, 0, 0], "an attack that already ended");
  const enemy = env.game.players().find((p) => p !== seat)!;
  const ally: StampedIntent = { type: "allianceRequest", recipient: enemy.id(), clientID };
  assert.deepEqual(labelOf(env, seat, ally), [OUT_OF_SET, 0, 0, 0]);
});
