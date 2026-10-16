import assert from "node:assert/strict";
import { test } from "node:test";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { attackMask, BUILDS } from "./actions";
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
  encodeSeat(env.game, seat, env.horizonMinutes, env.frame.cs, attack, players, rel, scalars);
  const me = seat.smallID();
  assert.equal(rel[me], Rel.Self);
  assert.deepEqual([...rel].filter((r) => r === Rel.Enemy).length, 3);
  assert.equal(players[me * PLAYER_FEATURES + 2], 1);
  assert.ok(players[7] > 0 && players[7] <= 1, "unowned land share");
  assert.equal(players[12], attack[0]);
  assert.equal(scalars[3], 1, "spawned");
  for (const v of [...players, ...scalars]) assert.ok(Number.isFinite(v) && v >= 0 && v <= 2, `${v}`);
});

test("scalars hold the gold against each build's current cost and the seat's loaded silo missiles", async () => {
  const env = await EnvGame.create(ONION, "costs");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(2_000_000n);
  const [tile] = seat.tiles();
  seat.buildUnit(UnitType.MissileSilo, tile, {}).increaseLevel(); // level 2, its new slot reloading: 1 loaded
  seat.buildUnit(UnitType.MissileSilo, tile, {}).setUnderConstruction(true); // none loaded
  const P = env.slots;
  const scalars = new Float32Array(SCALARS);
  const attack = attackMask(env.game, seat, P);
  encodeSeat(env.game, seat, env.horizonMinutes, env.frame.cs, attack, new Float32Array(P * PLAYER_FEATURES),
    new Uint8Array(P), scalars);
  const gold = Number(seat.gold());
  [...BUILDS.values()].forEach((unit, i) => {
    const cost = Number(env.game.unitInfo(unit).cost(env.game, seat));
    assert.equal(scalars[16 + i], Math.fround(Math.min(gold / cost, 4) / 4), unit);
  });
  assert.ok(scalars[16] > scalars[24], "a city costs less than a hydrogen bomb");
  assert.equal(scalars[25], 1 / 16);
});
