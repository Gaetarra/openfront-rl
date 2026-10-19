import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ColoredTeams,
  GameMapSize,
  PlayerType,
  UnitType,
} from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { GameMapType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/Maps.gen";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/packages/engine-lib/src/PseudoRandom";
import { Act, N_ACTS } from "./actions";
import { CellFrame } from "./frame";
import { EnvGame } from "./game";
import { nationCount } from "./maps";
import { rewardOf } from "./ranking";
import { PLAYER_FEATURES } from "./obs";
import { shapeOf } from "./setups";
import { FFA, HVN, landTile, LONE, ONION, publicGame, quickGame, RANKED, script, spawnCell } from "./testkit";

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
  assert.deepEqual(blobs.map(([n]) => n), ["cells", "units", "players", "rel", "scalars", "types", "attack", "retreat", "diplomacy"]);
  assert.equal(blob(blobs, "cells").byteLength, env.frame.n * 11);
  assert.equal(blob(blobs, "players").byteLength, env.slots * PLAYER_FEATURES * 4);
  assert.equal(blob(blobs, "diplomacy").byteLength, env.slots * 4);
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

const nationsOf = (env: EnvGame) => env.game.allPlayers().filter((p) => p.type() === PlayerType.Nation);

test("public games get the game's own nations, and the tribes fit the player table", async () => {
  const ffa = await publicGame(FFA, GameMapType.World, GameMapSize.Normal);
  assert.equal(nationsOf(ffa).length, nationCount(GameMapType.World));
  assert.equal(nationsOf(ffa).length, 72);
  assert.equal(nationsOf(await publicGame(FFA, GameMapType.World, GameMapSize.Compact)).length, 18);
  const hvn = await publicGame(HVN, GameMapType.World, GameMapSize.Normal);
  assert.equal(nationsOf(hvn).length, 4);
  assert.ok(hvn.seats.every((s) => s.team() === ColoredTeams.Humans));
  assert.ok(nationsOf(hvn).every((n) => n.team() === ColoredTeams.Nations));
  for (let t = 0; t < 300; t++) ffa.tickWith([]); // past the spawn phase: the tribes have spawned
  const players = ffa.game.allPlayers();
  assert.equal(players.filter((p) => p.type() === PlayerType.Bot).length, 400);
  assert.equal(ffa.slots, 477);
  assert.equal(Math.max(...players.map((p) => p.smallID())), ffa.slots - 1); // 4 seats, 72 nations, 400 tribes: a full table
  ffa.observe(true);
});

/** Random legal play for every seat until the game ends or `maxSteps` pass. */
function playRandom(env: EnvGame, rnd: PseudoRandom, maxSteps: number): void {
  const cells: number[] = [];
  env.game.forEachTile((t) => {
    if (env.game.isLand(t) && !env.game.isImpassable(t)) cells.push(env.frame.cellOf(t));
  });
  const pick = (mask: Uint8Array) => {
    const legal = [...mask.keys()].filter((i) => mask[i] === 1);
    return legal[rnd.nextInt(0, legal.length)];
  };
  const P = env.slots;
  for (let steps = 0; !env.done && steps < maxSteps; steps++) {
    const blobs = env.observe(steps === 0);
    const types = blob(blobs, "types");
    const attack = blob(blobs, "attack");
    env.step(
      env.seats.map((_, i) => {
        const type = pick(types.subarray(i * N_ACTS, (i + 1) * N_ACTS));
        const target = type === Act.Attack ? pick(attack.subarray(i * P, (i + 1) * P)) : 0;
        return [type, cells[rnd.nextInt(0, cells.length)], target, rnd.nextInt(1, 101)];
      }),
    );
  }
}

/** No-op decisions for every seat until the game ends: the steps it took. */
function playNoops(env: EnvGame, maxSteps: number): number {
  let steps = 0;
  while (!env.done && steps < maxSteps) {
    env.step(env.seats.map(() => [Act.Noop, 0, 0, 0]));
    steps++;
  }
  return steps;
}

test("an FFA game ends with every seat ranked among the seats and nations", async () => {
  const env = await quickGame(FFA);
  playRandom(env, new PseudoRandom(5), 400);
  assert.ok(env.done, "game ended");
  const n = 4 + nationsOf(env).length;
  assert.ok(n > 4, "the map has nations");
  assert.equal(env.nPlayers, n);
  env.ranks.forEach((rank, i) => {
    assert.ok(rank >= 1 && rank <= n, `rank ${rank}`);
    assert.equal(env.rewards[i], rewardOf(rank, n));
  });
});

test("four FFA seats that never spawn finish last together", async () => {
  const env = await quickGame(FFA);
  const steps = playNoops(env, 100);
  const n = 4 + nationsOf(env).length;
  assert.ok(env.done && steps <= 41, `steps ${steps}`);
  assert.deepEqual(env.ranks, [n, n, n, n]);
  assert.deepEqual(env.rewards, [-1, -1, -1, -1]);
});

test("a Humans vs Nations team shares its result", async () => {
  const env = await quickGame(HVN);
  playRandom(env, new PseudoRandom(5), 400);
  assert.ok(env.done, "game ended");
  assert.equal(env.nPlayers, 2);
  assert.equal(new Set(env.rewards).size, 1);
  assert.ok([1, -1].includes(env.rewards[0]), `reward ${env.rewards[0]}`);
});

test("a Humans vs Nations team that never spawns loses together", async () => {
  const env = await quickGame(HVN);
  playNoops(env, 100);
  assert.ok(env.done);
  assert.deepEqual(env.ranks, [2, 2, 2, 2]);
  assert.deepEqual(env.rewards, [-1, -1, -1, -1]);
});

test("a Humans vs Nations win is a win for every seat of the team", async () => {
  const env = await quickGame(HVN);
  env.game.setWinner(ColoredTeams.Humans, env.game.stats().stats());
  env.step(env.seats.map(() => [Act.Noop, 0, 0, 0]));
  assert.ok(env.done);
  assert.deepEqual(env.ranks, [1, 1, 1, 1]);
  assert.deepEqual(env.rewards, [1, 1, 1, 1]);
});

test("a game without a timer is observed with a 30-minute horizon", async () => {
  const env = await quickGame(FFA, 0);
  assert.equal(env.horizonMinutes, 30);
  for (let t = 0; t < 150; t++) env.tickWith([]); // inside the spawn phase: the game is still on
  const scalars = blob(env.observe(true), "scalars");
  assert.ok(Math.abs(new Float32Array(scalars.buffer, scalars.byteOffset, 1)[0] - 150 / (200 + 30 * 600)) < 1e-6);
});

test("the backstop marks a game cut only when nobody won and a seat is still in", async () => {
  // One seat alone on the map: nothing kills it and it never expands, so only the clock can end the game.
  for (const [timerMinutes, cut, minutes] of [[0, true, 61], [1, false, 1]] as const) {
    const env = await quickGame(FFA, timerMinutes, LONE);
    const spawn = spawnCell(env);
    let steps = 0;
    while (!env.done && steps < 8000) {
      env.step([[steps === 2 ? Act.Spawn : Act.Noop, spawn, 0, 0], ...[1, 2, 3].map(() => [Act.Noop, 0, 0, 0])]);
      steps++;
    }
    assert.ok(env.done, "game ended");
    assert.equal(env.cut, cut, `timer ${timerMinutes}`);
    assert.equal(env.game.getWinner() === null, cut);
    assert.ok(env.game.ticks() >= env.game.config().numSpawnPhaseTurns() + minutes * 600);
  }
});

test("a game created with a run's shape uses its grid and player table", async () => {
  const env = await EnvGame.create(ONION, "shared", shapeOf([RANKED, FFA, HVN]));
  assert.equal(env.frame.stride, 256);
  assert.equal(env.slots, 477);
  env.observe(true);
  const ffa = await EnvGame.create(FFA, "ffa-a");
  assert.equal(ffa.seats.length, 4);
  assert.equal(ffa.slots, 477);
  assert.equal(ffa.start.config.maxPlayers, 4);
});
