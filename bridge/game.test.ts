import assert from "node:assert/strict";
import { test } from "node:test";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { Act } from "./actions";
import { CellFrame } from "./frame";
import { EnvGame } from "./game";
import { rewardOf } from "./ranking";
import { landTile, ONION, RANKED, script, spawnCell } from "./testkit";

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
    env.step([[type, cell, target, rnd.nextInt(1, 101)]]);
    steps++;
  }
  assert.ok(env.done, "game ended");
  const [rank] = env.ranks;
  assert.ok(rank >= 1 && rank <= 4);
  assert.equal(env.nPlayers, 4);
  assert.equal(env.rewards[0], rewardOf(rank, 4));
});

test("a decision can carry a second action, and both execute (the bandwidth probe, PLAN 3B)", async () => {
  const env = await EnvGame.create(ONION, "two-actions");
  script(env, 300);
  const seat = env.seats[0];
  seat.addGold(10_000_000n);
  const target = blob(env.observe(false), "attack").indexOf(1);
  assert.ok(target >= 0, "something to attack");
  env.step([[Act.Attack, 0, target, 25, Act.BuildCity, spawnCell(env), 0, 0]]);
  assert.equal(seat.units(UnitType.City).length, 1);
  assert.equal(seat.outgoingAttacks().length, 1);
  assert.deepEqual(env.invalid, [false]);
});

test("an empty second slot keeps the decision valid, even where a no-op is illegal", async () => {
  const env = await EnvGame.create(ONION, "second-noop");
  let types = blob(env.observe(true), "types");
  while (types[Act.Noop] === 1) {
    env.step([[Act.Noop, 0, 0, 0]]);
    types = blob(env.observe(false), "types");
  }
  env.step([[Act.Spawn, spawnCell(env), 0, 0, Act.Noop, 0, 0, 0]]);
  assert.deepEqual(env.invalid, [false]);
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
  assert.equal(env.terrain().byteLength, 5 * env.frame.n);
  env.step([[Act.Noop, 0, 0, 0]]);
  assert.ok(env.observe(false)[0][1].byteLength < env.frame.n * 11);
});

test("a frame on a wider grid indexes cells by grid row and leaves the padding empty", async () => {
  const env = await EnvGame.create(ONION, "pad");
  script(env, 600);
  const plain = new CellFrame(env.game.map(), env.frame.cs);
  const padded = new CellFrame(env.game.map(), env.frame.cs, 80);
  plain.encode(true);
  padded.encode(true);
  assert.equal(padded.n, plain.ch * 80);
  for (let y = 0; y < plain.ch; y++) {
    for (let x = 0; x < 80; x++) {
      const c = y * 80 + x;
      if (x < plain.cw) {
        assert.equal(padded.o1[c], plain.o1[y * plain.cw + x]);
      } else {
        assert.equal(padded.o1[c], 0);
        assert.deepEqual(padded.tilesOf(c), []);
      }
    }
  }
  const tile = landTile(env);
  assert.ok(padded.tilesOf(padded.cellOf(tile)).includes(tile));
});

test("ranked games draw from the playlist and share the ranked grid", async () => {
  const env = await EnvGame.create(RANKED, "ranked-a");
  assert.equal(env.seats.length, 2);
  assert.equal(env.frame.stride, 256);
  assert.equal(env.slots, 403);
  assert.equal(env.start.config.rankedType, "1v1");
  assert.match(env.start.gameID, /^[A-Za-z0-9]{8}$/);
});
