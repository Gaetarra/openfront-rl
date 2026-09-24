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
