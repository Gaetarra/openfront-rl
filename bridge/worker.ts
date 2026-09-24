/**
 * Env worker: runs K training games for the Python trainer (trainer/env.py) over stdin/stdout.
 *
 * In:  u32 length + JSON: {cmd:"init", config, games, seed, worker, episodes?} | {cmd:"step", actions, full?} |
 *      {cmd:"close"}. "init" may be sent again to restart all games.
 * Out: u32 size + u32 header length + header JSON + blobs. The header lists each game's blobs as [name, byteLength]
 *      in the order they follow. A finished game is replaced at once by the next episode: its reply has done and
 *      reset set, the finished game's rewards and ranks, and the new game's terrain and full frame.
 * Game seeds are `${seed}-w${worker}-g${game}-e${episode}`.
 */
import { N_ACTS, TROOP_BINS } from "./actions";
import { UNIT_TYPES } from "./frame";
import { Blob, EnvConfig, EnvGame } from "./game";
import { PLAYER_FEATURES, SCALARS } from "./obs";

// stdout carries replies only; the game logs a lot through console.log/info/warn.
console.log = console.info = console.warn = console.debug = () => {};

interface Init {
  cmd: "init";
  config: EnvConfig;
  games: number;
  seed: string;
  worker: number;
  episodes?: number[];
}
interface Step {
  cmd: "step";
  actions: number[][][];
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

function part(env: EnvGame, g: number, blobs: Blob[], extra: Record<string, unknown>): Part {
  return {
    header: { tick: env.game.ticks(), episode: episodes[g], done: false, reset: false, hash: env.hash, invalid: env.invalid, ...extra },
    blobs,
  };
}

const fresh = (g: number, extra: Record<string, unknown>): Part =>
  part(games[g], g, [["terrain", games[g].terrain()], ...games[g].observe(true)], { reset: true, ...extra });

function dims(env: EnvGame): Record<string, unknown> {
  return {
    grid: [env.frame.ch, env.frame.cw],
    cellSize: env.frame.cs,
    slots: env.slots,
    seats: env.seats.length,
    playerFeatures: PLAYER_FEATURES,
    scalars: SCALARS,
    acts: N_ACTS,
    troopBins: TROOP_BINS,
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
    episodes = msg.episodes ?? new Array<number>(msg.games).fill(0);
    games = [];
    const parts: Part[] = [];
    for (let g = 0; g < msg.games; g++) {
      games.push(await EnvGame.create(msg.config, seedOf(g)));
      parts.push(fresh(g, {}));
    }
    return encode(parts, dims(games[0]));
  }
  if (init === null) throw new Error("step before init");
  const parts: Part[] = [];
  for (let g = 0; g < games.length; g++) {
    const env = games[g];
    env.step(msg.actions[g] ?? []);
    if (!env.done) {
      parts.push(part(env, g, env.observe(msg.full === true), {}));
      continue;
    }
    const end = {
      done: true,
      rewards: env.rewards,
      ranks: env.ranks,
      players: env.nPlayers,
      endTick: env.game.ticks(),
      hash: env.hash,
      invalid: env.invalid,
    };
    episodes[g]++;
    games[g] = await EnvGame.create(init.config, seedOf(g));
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
