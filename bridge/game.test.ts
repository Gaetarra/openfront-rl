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
