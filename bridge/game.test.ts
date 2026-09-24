import assert from "node:assert/strict";
import { test } from "node:test";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { Act } from "./actions";
import { CellFrame } from "./frame";
import { EnvGame } from "./game";
import { rewardOf } from "./ranking";
import { ONION, script, spawnCell } from "./testkit";

const blob = (blobs: [string, Uint8Array][], name: string) => blobs.find(([n]) => n === name)![1];

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
