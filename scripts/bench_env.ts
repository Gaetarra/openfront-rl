/**
 * Phase 0 spike S2: env-step cost measured on a real game record.
 *
 * Replays the record and, every 5 ticks, updates the per-cell ownership frame
 * (top-2 owners, their tile counts, fallout count; 7 bytes per cell) only for the
 * cells touched by that step's tile updates. Reports simulation time vs frame time,
 * the cost of a full rebuild, and the delta size per step. With --pipe, deltas are
 * written to stdout (u32 length prefix) and the report goes to stderr.
 *
 * Run from the game dir (its tsconfig paths and node_modules are needed):
 *   npx tsx ../../scripts/bench_env.ts <record.json> [--pipe]
 */
import fs from "fs";
import {
  GameMapSize,
  PlayerInfo,
  PlayerType,
} from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { GameStartInfo } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/packages/engine-lib/src/PseudoRandom";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/packages/engine-lib/src/Util";
import { EngineConfig } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/configuration/EngineConfig";
import { Executor } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/execution/ExecutionManager";
import { createGame } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/GameImpl";
import { createNationsForGame } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/NationCreation";
import { GameRunner } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/GameRunner";
import {
  decompressGameRecord,
  toWireGameStartInfo,
} from "../OpenFront_Source_Code/openfrontio/packages/shared/src/SharedUtil";
import { loadFreshTerrain } from "../bridge/maps";

const STEP_TICKS = 5;
const DELTA_BYTES = 11; // u32 cell index + 7-byte cell record

async function main(): Promise<void> {
  const [file, flag] = process.argv.slice(2);
  const pipe = flag === "--pipe";
  console.debug = () => {};
  // The game logs with console.log; in pipe mode stdout carries only frames.
  if (pipe) console.log = console.info = console.warn = console.error;
  const log = console.log;

  // Game setup mirrors tests/replay/ReplayGame.ts.
  const record = decompressGameRecord(JSON.parse(fs.readFileSync(file, "utf8")));
  const info = record.info;
  const gameStart: GameStartInfo = toWireGameStartInfo({
    gameID: info.gameID,
    lobbyCreatedAt: info.lobbyCreatedAt,
    config: info.config,
    players: info.players,
    tribes: info.tribes,
  });
  const terrain = await loadFreshTerrain(info.config.gameMap, info.config.gameMapSize);
  const random = new PseudoRandom(simpleHash(gameStart.gameID));
  const humans = gameStart.players.map(
    (p) =>
      new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.clientID,
        random.nextID(),
        p.isLobbyCreator ?? false,
        p.clanTag,
        p.friends ?? [],
        p.teamIndex ?? null,
      ),
  );
  const nations = createNationsForGame(
    gameStart,
    terrain.nations,
    terrain.additionalNations,
    humans.length,
    random,
  );
  const game = createGame(
    humans,
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    new EngineConfig(info.config, false, gameStart.listed),
    terrain.teamGameSpawnAreas,
  );

  const map = game.map();
  const W = map.width();
  const H = map.height();
  const cs = info.config.gameMapSize === GameMapSize.Compact ? 4 : 8;
  const cw = Math.ceil(W / cs);
  const nCells = cw * Math.ceil(H / cs);
  const state = map.tileStateBuffer();

  // Current frame, one array per field.
  const o1 = new Uint16Array(nCells);
  const o2 = new Uint16Array(nCells);
  const n1 = new Uint8Array(nCells);
  const n2 = new Uint8Array(nCells);
  const fo = new Uint8Array(nCells);

  const dirty = new Uint8Array(nCells);
  const dirtyList = new Int32Array(nCells);
  let nDirty = 0;
  let markMs = 0;

  const cnt = new Uint16Array(4096);
  const touched: number[] = [];
  const res = { o1: 0, o2: 0, n1: 0, n2: 0, fo: 0 };
  const computeCell = (c: number): void => {
    const x0 = (c % cw) * cs;
    const y0 = Math.floor(c / cw) * cs;
    const x1 = Math.min(x0 + cs, W);
    const y1 = Math.min(y0 + cs, H);
    let fallout = 0;
    for (let y = y0; y < y1; y++) {
      for (let i = y * W + x0, end = y * W + x1; i < end; i++) {
        const v = state[i];
        if (v & 0x2000) fallout++;
        const id = v & 0xfff;
        if (id !== 0 && cnt[id]++ === 0) touched.push(id);
      }
    }
    res.o1 = res.o2 = res.n1 = res.n2 = 0;
    for (const id of touched) {
      const n = cnt[id];
      if (n > res.n1) {
        res.o2 = res.o1;
        res.n2 = res.n1;
        res.o1 = id;
        res.n1 = n;
      } else if (n > res.n2) {
        res.o2 = id;
        res.n2 = n;
      }
      cnt[id] = 0;
    }
    touched.length = 0;
    res.fo = fallout;
  };

  let fatal: string | undefined;
  const runner = new GameRunner(
    game,
    new Executor(game, gameStart.gameID, undefined, gameStart.tribes?.map((t) => t.name)),
    (gu) => {
      if ("errMsg" in gu) {
        fatal = gu.errMsg;
        return;
      }
      const t0 = performance.now();
      const packed = gu.packedTileUpdates;
      for (let i = 0; i < packed.length; i += 2) {
        const ref = packed[i];
        const c = Math.floor(Math.floor(ref / W) / cs) * cw + Math.floor((ref % W) / cs);
        if (dirty[c] === 0) {
          dirty[c] = 1;
          dirtyList[nDirty++] = c;
        }
      }
      markMs += performance.now() - t0;
    },
  );
  runner.init();

  const delta = new Uint8Array(4 + nCells * DELTA_BYTES);
  const dv = new DataView(delta.buffer);
  const deltaSizes: number[] = [];
  let simMs = 0;
  let frameMs = 0;
  let writeMs = 0;
  let fullMs = 0;
  let fullRuns = 0;
  let maxUnits = 0;
  let maxPlayers = 0;

  const turns = record.turns;
  for (let t = 0; t < turns.length; t++) {
    runner.addTurn(turns[t]);
    const s0 = performance.now();
    if (!runner.executeNextTick()) throw new Error(`tick failed at ${t}: ${fatal}`);
    simMs += performance.now() - s0;
    if ((t + 1) % STEP_TICKS !== 0) continue;

    const f0 = performance.now();
    let off = 4;
    for (let k = 0; k < nDirty; k++) {
      const c = dirtyList[k];
      dirty[c] = 0;
      computeCell(c);
      if (res.o1 === o1[c] && res.o2 === o2[c] && res.n1 === n1[c] && res.n2 === n2[c] && res.fo === fo[c]) {
        continue;
      }
      o1[c] = res.o1;
      o2[c] = res.o2;
      n1[c] = res.n1;
      n2[c] = res.n2;
      fo[c] = res.fo;
      dv.setUint32(off, c, true);
      dv.setUint16(off + 4, res.o1, true);
      dv.setUint16(off + 6, res.o2, true);
      delta[off + 8] = res.n1;
      delta[off + 9] = res.n2;
      delta[off + 10] = res.fo;
      off += DELTA_BYTES;
    }
    nDirty = 0;
    frameMs += performance.now() - f0;
    deltaSizes.push(off - 4);

    if (pipe) {
      const w0 = performance.now();
      dv.setUint32(0, off - 4, true);
      fs.writeSync(1, delta, 0, off);
      writeMs += performance.now() - w0;
    }
    if (deltaSizes.length % 50 === 0) {
      const u0 = performance.now();
      for (let c = 0; c < nCells; c++) computeCell(c);
      fullMs += performance.now() - u0;
      fullRuns++;
      maxUnits = Math.max(maxUnits, game.units().length);
      maxPlayers = Math.max(maxPlayers, game.players().length);
    }
  }

  const steps = deltaSizes.length;
  const sorted = [...deltaSizes].sort((a, b) => a - b);
  const pct = (p: number) => sorted[Math.min(steps - 1, Math.floor(p * steps))];
  const mean = deltaSizes.reduce((a, b) => a + b, 0) / steps;
  log(
    `${info.gameID} ${info.config.gameMap} (${info.config.gameMapSize}) ${W}x${H} tiles, ` +
      `${cw}x${nCells / cw} cells, ${turns.length} ticks, ${steps} steps`,
  );
  log(
    `  per step: sim ${(simMs / steps).toFixed(2)} ms, frame ${(frameMs / steps).toFixed(3)} ms ` +
      `+ mark ${(markMs / steps).toFixed(3)} ms (${((100 * (frameMs + markMs)) / simMs).toFixed(1)}% of sim)` +
      (pipe ? `, pipe write ${(writeMs / steps).toFixed(3)} ms` : ""),
  );
  log(
    `  full rebuild ${(fullMs / fullRuns).toFixed(2)} ms (${((100 * fullMs) / fullRuns / (simMs / steps)).toFixed(0)}% of a step); ` +
      `full frame ${((nCells * 7) / 1024).toFixed(0)} KB`,
  );
  log(
    `  delta KB/step: mean ${(mean / 1024).toFixed(2)}, p50 ${(pct(0.5) / 1024).toFixed(2)}, ` +
      `p95 ${(pct(0.95) / 1024).toFixed(2)}, p99 ${(pct(0.99) / 1024).toFixed(2)}, max ${(sorted[steps - 1] / 1024).toFixed(1)}`,
  );
  log(`  max alive players ${maxPlayers}, max units ${maxUnits}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
