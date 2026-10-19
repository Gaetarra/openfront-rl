import assert from "node:assert/strict";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Act } from "./actions";
import { EnvGame } from "./game";
import { GAME_ROOT } from "./maps";
import { PLAYER_FEATURES } from "./obs";
import { ONION, RANKED, spawnCell } from "./testkit";

const WORKER = path.join(path.dirname(fileURLToPath(import.meta.url)), "worker.ts");
type Reply = { header: any; body: Buffer };

class Client {
  private buf = Buffer.alloc(0);
  private waiting: ((m: Buffer) => void)[] = [];
  readonly proc: ChildProcessWithoutNullStreams = spawn(process.execPath, ["--import", "tsx", WORKER], {
    cwd: GAME_ROOT,
  });

  constructor() {
    this.proc.stdout.on("data", (chunk: Buffer) => {
      this.buf = Buffer.concat([this.buf, chunk]);
      this.drain();
    });
  }

  private drain(): void {
    while (this.waiting.length > 0 && this.buf.length >= 4 && this.buf.length >= 4 + this.buf.readUInt32LE(0)) {
      const n = this.buf.readUInt32LE(0);
      const msg = this.buf.subarray(4, 4 + n);
      this.buf = this.buf.subarray(4 + n);
      this.waiting.shift()!(msg);
    }
  }

  request(msg: object): Promise<Reply> {
    const data = Buffer.from(JSON.stringify(msg));
    const len = Buffer.alloc(4);
    len.writeUInt32LE(data.length);
    this.proc.stdin.write(Buffer.concat([len, data]));
    return new Promise((resolve) => {
      this.waiting.push((m) => {
        const h = m.readUInt32LE(0);
        resolve({ header: JSON.parse(m.subarray(4, 4 + h).toString("utf8")), body: m.subarray(4 + h) });
      });
      this.drain();
    });
  }

  close(): void {
    const data = Buffer.from(JSON.stringify({ cmd: "close" }));
    const len = Buffer.alloc(4);
    len.writeUInt32LE(data.length);
    this.proc.stdin.end(Buffer.concat([len, data]));
  }
}

test("init returns dims and a full first frame per game", async () => {
  const c = new Client();
  const { header, body } = await c.request({ cmd: "init", config: ONION, games: 2, seed: "t", worker: 0 });
  c.close();
  assert.deepEqual(header.dims.grid, [64, 64]);
  assert.equal(header.dims.slots, 5);
  assert.equal(header.games.length, 2);
  const g = header.games[0];
  assert.equal(g.reset, true);
  assert.deepEqual(g.blobs.map((b: [string, number]) => b[0]), ["terrain", "cells", "units", "players", "rel", "scalars", "types", "attack", "retreat", "diplomacy"]);
  assert.equal(g.blobs[1][1], 64 * 64 * 11);
  const total = header.games.flatMap((x: any) => x.blobs).reduce((s: number, b: [string, number]) => s + b[1], 0);
  assert.equal(body.length, total);
});

test("a worker in a mixed run uses the run's grid and player table", async () => {
  const c = new Client();
  const { header } = await c.request({ cmd: "init", config: ONION, mix: [ONION, RANKED], games: 1, seed: "mix", worker: 0 });
  c.close();
  assert.deepEqual(header.dims.grid, [192, 256]);
  assert.equal(header.dims.slots, 403);
  assert.equal(header.dims.seats, 1);
  const players = header.games[0].blobs.find((b: [string, number]) => b[0] === "players");
  assert.equal(players[1], 403 * PLAYER_FEATURES * 4);
});

test("same seed and actions give the same hash in a fresh worker and after other games", async () => {
  const cell = spawnCell(await EnvGame.create(ONION, "cell"));
  const play = async (c: Client, seed: string): Promise<number> => {
    await c.request({ cmd: "init", config: ONION, games: 1, seed, worker: 0 });
    let reply: Reply | null = null;
    for (let step = 0; step < 200; step++) {
      const action = step === 2 ? [Act.Spawn, cell, 0, 0] : step > 45 && step % 10 === 0 ? [Act.Attack, 0, 0, 50] : [Act.Noop, 0, 0, 0];
      reply = await c.request({ cmd: "step", actions: [[action]] });
    }
    return reply!.header.games[0].hash;
  };
  const fresh = new Client();
  const first = await play(fresh, "det");
  fresh.close();
  const reused = new Client();
  await play(reused, "other");
  const second = await play(reused, "det");
  reused.close();
  assert.notEqual(first, 0);
  assert.equal(second, first);
});

test("a finished game is replaced by the next episode in the same reply", async () => {
  const c = new Client();
  await c.request({ cmd: "init", config: { ...ONION, timerMinutes: 1 }, games: 1, seed: "end", worker: 3 });
  let g: any = null;
  for (let step = 0; step < 60; step++) {
    g = (await c.request({ cmd: "step", actions: [[[Act.Noop, 0, 0, 0]]] })).header.games[0];
    if (g.done) break;
  }
  c.close();
  assert.equal(g.done, true);
  assert.equal(g.reset, true);
  assert.equal(g.episode, 1);
  assert.deepEqual(g.ranks, [4]);
  assert.deepEqual(g.rewards, [-1]);
  assert.equal(g.cut, false);
  assert.equal(g.blobs[0][0], "terrain");
});

test("replay mode streams observations with labels and moves on to the next record", async () => {
  const fixture = path.join(path.dirname(WORKER), "testdata/ranked1v1.json.gz");
  const c = new Client();
  try {
    const init = await c.request({
      cmd: "init",
      config: { ...ONION, seats: 2, playlist: "ranked1v1" },
      replay: { records: [fixture] },
      games: 1,
      seed: "rep",
      worker: 0,
    });
    assert.deepEqual(init.header.dims.grid, [192, 256]);
    const g0 = init.header.games[0];
    assert.equal(g0.reset, true);
    assert.equal(g0.seats.length, 2);
    assert.deepEqual([...g0.outcome].sort(), [-1, 1]);
    assert.deepEqual(g0.blobs.at(-1), ["labels", 2 * 4 * 4]);
    let finished: any = null;
    for (let step = 0; step < 2000 && finished === null; step++) {
      const g = (await c.request({ cmd: "step" })).header.games[0];
      if (g.done) finished = g;
    }
    assert.ok(finished !== null, "the record ended and the next one started");
    assert.equal(finished.reset, true);
    assert.equal(finished.desync, false);
    assert.equal(finished.finished, g0.record);
  } finally {
    c.close(); // a failed assertion must not leave the worker, and the test run, hanging
  }
});
