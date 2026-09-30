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
