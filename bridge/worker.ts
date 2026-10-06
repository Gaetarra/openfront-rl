/**
 * Env worker: runs K training games for the Python trainer (trainer/env.py) over stdin/stdout.
 *
 * In:  u32 length + JSON: {cmd:"init", config, games, seed, worker, episodes?} starts K training games;
 *      {cmd:"init", config, replay:{records}, games, seed, worker} replays game records instead (behaviour cloning:
 *      every reply carries the seats' observations and labels, bridge/replay.ts); {cmd:"step", actions?, full?};
 *      {cmd:"close"}. "init" may be sent again to restart.
 * Out: u32 size + u32 header length + header JSON + blobs. The header lists each game's blobs as [name, byteLength]
 *      in the order they follow. A finished game is replaced at once by the next episode: its reply has done and
 *      reset set, the finished game's rewards and ranks, and the new game's terrain and full frame.
 * Game seeds are `${seed}-w${worker}-g${game}-e${episode}`.
 * Replay mode cycles through the worker's records in a seeded order; the first record of each slot starts at a random
 * window, so the slots sit at different game phases.
 */
import fs from "node:fs";
import path from "node:path";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { N_ACTS } from "./actions";
import { UNIT_TYPES } from "./frame";
import { Blob, EnvConfig, EnvGame } from "./game";
import { PLAYER_FEATURES, SCALARS } from "./obs";
import { readRecord, RecordGame } from "./replay";
import { gridFor, slotsFor } from "./setups";

// stdout carries replies only; the game logs a lot through console.log/info/warn.
console.log = console.info = console.warn = console.debug = () => {};

interface Init {
  cmd: "init";
  config: EnvConfig;
  games: number;
  seed: string;
  worker: number;
  episodes?: number[];
  replay?: { records: string[] };
  record?: { dir: string; gitCommit: string; games: number };
}
interface Step {
  cmd: "step";
  actions?: number[][][];
  full?: boolean;
}
interface Part {
  header: Record<string, unknown>;
  blobs: Blob[];
}

let init: Init | null = null;
let games: EnvGame[] = [];
let episodes: number[] = [];

const seedOf = (g: number): string => `${init!.seed}-w${init!.worker}-g${g}-e${episodes[g]}`;

let recordStarted = 0;

/** A new training game; the first `record.games` games of this worker are recorded. */
async function newGame(g: number): Promise<EnvGame> {
  const env = await EnvGame.create(init!.config, seedOf(g));
  if (init!.record !== undefined && recordStarted < init!.record.games) {
    env.record();
    recordStarted++;
  }
  return env;
}

let records: RecordGame[] = [];
let order: number[] = [];
let next = 0;
let pick = new PseudoRandom(0);

/** The next record of this worker's list that loads; `skip`: start it at a random window. */
async function nextRecord(skip: boolean): Promise<RecordGame> {
  const files = init!.replay!.records;
  for (let tries = 0; tries < files.length; tries++) {
    if (next >= order.length) {
      order = pick.shuffleArray(order);
      next = 0;
    }
    const file = files[order[next++]];
    try {
      const rg = await RecordGame.create(readRecord(file), gridFor(init!.config)[1], slotsFor(init!.config));
      if (rg.env.seats.length !== init!.config.seats) throw new Error(`${rg.env.seats.length} players`);
      const skipped = skip ? pick.nextInt(0, Math.max(1, Math.floor(rg.left / 2))) : 0;
      for (let i = 0; i < skipped && !rg.done; i++) rg.advance();
      if (!rg.done) return rg;
    } catch (err) {
      console.error(`skipping ${file}: ${err}`);
    }
  }
  throw new Error("none of this worker's records can be replayed");
}

function replayPart(rg: RecordGame, reset: boolean, extra: Record<string, unknown>): Part {
  const labels = new Int32Array(rg.labels().flat());
  const blobs: Blob[] = [
    ...(reset ? [["terrain", rg.env.terrain()] as Blob] : []),
    ...rg.env.observe(reset),
    ["labels", new Uint8Array(labels.buffer)],
  ];
  const header = {
    tick: rg.env.game.ticks(),
    episode: 0,
    done: false,
    reset,
    hash: rg.env.hash,
    invalid: rg.env.seats.map(() => false),
    record: rg.record.info.gameID,
    seats: rg.env.start.players.map((p) => p.clientID),
    outcome: rg.outcome,
    left: rg.left,
    ...(reset ? { cellSize: rg.env.frame.cs } : {}),
    ...extra,
  };
  return { header, blobs };
}

async function replayStep(): Promise<Part[]> {
  const parts: Part[] = [];
  for (let g = 0; g < records.length; g++) {
    const rg = records[g];
    let error: string | undefined;
    try {
      rg.advance();
    } catch (err) {
      error = String(err);
      console.error(`${rg.record.info.gameID}: ${error}`);
      rg.done = true;
    }
    if (!rg.done) {
      parts.push(replayPart(rg, false, {}));
      continue;
    }
    const end = { done: true, finished: rg.record.info.gameID, desync: rg.desync, ...(error ? { error } : {}) };
    records[g] = await nextRecord(false);
    parts.push(replayPart(records[g], true, end));
  }
  return parts;
}

function part(env: EnvGame, g: number, blobs: Blob[], extra: Record<string, unknown>): Part {
  return {
    header: { tick: env.game.ticks(), episode: episodes[g], done: false, reset: false, hash: env.hash, invalid: env.invalid, ...extra },
    blobs,
  };
}

const fresh = (g: number, extra: Record<string, unknown>): Part =>
  part(games[g], g, [["terrain", games[g].terrain()], ...games[g].observe(true)], {
    reset: true,
    cellSize: games[g].frame.cs,
    ...extra,
  });

function dims(cfg: EnvConfig): Record<string, unknown> {
  return {
    grid: gridFor(cfg),
    slots: slotsFor(cfg),
    seats: cfg.seats,
    playerFeatures: PLAYER_FEATURES,
    scalars: SCALARS,
    acts: N_ACTS,
    unitTypes: UNIT_TYPES,
  };
}

function encode(parts: Part[], dimsInfo: Record<string, unknown> | null): Buffer {
  const header = Buffer.from(
    JSON.stringify({
      ...(dimsInfo === null ? {} : { dims: dimsInfo }),
      games: parts.map((p) => ({ ...p.header, blobs: p.blobs.map(([name, b]) => [name, b.byteLength]) })),
    }),
    "utf8",
  );
  const size = 4 + header.length + parts.reduce((s, p) => s + p.blobs.reduce((t, [, b]) => t + b.byteLength, 0), 0);
  const out = Buffer.allocUnsafe(4 + size);
  out.writeUInt32LE(size, 0);
  out.writeUInt32LE(header.length, 4);
  header.copy(out, 8);
  let off = 8 + header.length;
  for (const p of parts) {
    for (const [, b] of p.blobs) {
      out.set(b, off);
      off += b.byteLength;
    }
  }
  return out;
}

async function handle(msg: Init | Step): Promise<Buffer> {
  if (msg.cmd === "init") {
    init = msg;
    if (msg.replay !== undefined) {
      if (msg.replay.records.length === 0) throw new Error("replay mode needs records");
      pick = new PseudoRandom(simpleHash(`${msg.seed}-w${msg.worker}`));
      order = pick.shuffleArray(msg.replay.records.map((_, i) => i));
      next = 0;
      records = [];
      for (let g = 0; g < msg.games; g++) records.push(await nextRecord(true));
      return encode(records.map((rg) => replayPart(rg, true, {})), dims(msg.config));
    }
    episodes = msg.episodes ?? new Array<number>(msg.games).fill(0);
    games = [];
    recordStarted = 0;
    const parts: Part[] = [];
    for (let g = 0; g < msg.games; g++) {
      games.push(await newGame(g));
      parts.push(fresh(g, {}));
    }
    return encode(parts, dims(msg.config));
  }
  if (init === null) throw new Error("step before init");
  if (init.replay !== undefined) return encode(await replayStep(), null);
  const parts: Part[] = [];
  for (let g = 0; g < games.length; g++) {
    const env = games[g];
    env.step(msg.actions?.[g] ?? []);
    if (!env.done) {
      parts.push(part(env, g, env.observe(msg.full === true), {}));
      continue;
    }
    let recorded: string | undefined;
    if (env.recording) {
      fs.mkdirSync(init.record!.dir, { recursive: true });
      recorded = env.start.gameID;
      fs.writeFileSync(path.join(init.record!.dir, `${recorded}.json`), JSON.stringify(env.toRecord(init.record!.gitCommit)));
    }
    const end = {
      done: true,
      rewards: env.rewards,
      ranks: env.ranks,
      players: env.nPlayers,
      endTick: env.game.ticks(),
      hash: env.hash,
      invalid: env.invalid,
      ...(recorded ? { recorded } : {}),
    };
    episodes[g]++;
    games[g] = await newGame(g);
    parts.push(fresh(g, end));
  }
  return encode(parts, null);
}

async function main(): Promise<void> {
  let buf = Buffer.alloc(0);
  for await (const chunk of process.stdin) {
    buf = Buffer.concat([buf, chunk as Buffer]);
    while (buf.length >= 4 && buf.length >= 4 + buf.readUInt32LE(0)) {
      const n = buf.readUInt32LE(0);
      const msg = JSON.parse(buf.subarray(4, 4 + n).toString("utf8")) as Init | Step | { cmd: "close" };
      buf = buf.subarray(4 + n);
      if (msg.cmd === "close") return;
      process.stdout.write(await handle(msg));
    }
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
