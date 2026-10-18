/**
 * Traces of replayed records for the mode analyses (PLAN "After 4N" item 2, scripts/mode_analysis.py): every 10 s of
 * game time after the spawn phase, each human's and nation's tiles, troops, gold, structure levels, alliances, rivals
 * attacking it and troops sent away, and every conquest. The records come from a list of "<mode> <record path>"
 * lines. Each trace is appended to the output as one JSON line, and records already there are skipped, so a killed run
 * resumes. A record that desyncs keeps its trace up to the desync. Runs at below-normal priority.
 *
 * Run from the game dir (`mode_analysis.py traces` starts one per worker):
 *   node --import tsx ../../scripts/mode_traces.ts <list.txt> <out.jsonl>
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { PlayerType, UnitType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { Player } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/Game";
import { readRecord, RecordGame } from "../bridge/replay";

const EVERY = 100; // ticks between samples: 10 s
/** Structures and warships, as summed levels (Player.unitCount). */
const UNITS = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
  UnitType.DefensePost,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
  UnitType.Warship,
];
const STRIDE = 2048; // grid width in cells: the frame needs one as wide as the map, and the trace reads no cells

async function trace(mode: string, file: string) {
  const { env, record } = await RecordGame.create(readRecord(file), STRIDE, 0);
  const game = env.game;
  let sides: Player[] | null = null; // the humans and nations, fixed when the spawn phase ends
  const t: number[] = [];
  const tiles: number[][] = [];
  const troops: number[][] = [];
  const gold: number[][] = [];
  const earned: (number | null)[][] = []; // humans' gold earned so far (the game keeps no stats for nations)
  const units: number[][][] = [];
  const allies: number[][] = [];
  const attackers: number[][] = []; // humans and nations with an attack on the side
  const away: number[][] = []; // troops in the side's attacks and boats
  const kills: [tick: number, conqueror: number, conquered: number, type: string][] = [];
  let desync: number | null = null;
  const sample = () => {
    sides ??= game.allPlayers().filter((p) => p.type() !== PlayerType.Bot);
    t.push(game.ticks());
    tiles.push(sides.map((p) => p.numTilesOwned()));
    troops.push(sides.map((p) => Math.round(p.troops())));
    gold.push(sides.map((p) => Number(p.gold())));
    earned.push(sides.map((p) => {
      const g = game.stats().getPlayerStats(p)?.gold;
      return p.type() === PlayerType.Human ? (g ?? []).reduce((a, b) => a + Number(b), 0) : null;
    }));
    units.push(sides.map((p) => UNITS.map((u) => p.unitCount(u))));
    allies.push(sides.map((p) => p.allies().length));
    attackers.push(sides.map((p) => {
      const rivals = p.incomingAttacks().filter((a) => a.attacker().type() !== PlayerType.Bot);
      return new Set(rivals.map((a) => a.attacker())).size;
    }));
    away.push(sides.map((p) => Math.round(
      p.outgoingAttacks().reduce((n, a) => n + a.troops(), 0) +
        p.units(UnitType.TransportShip).reduce((n, u) => n + u.troops(), 0))));
  };
  for (const turn of record.turns) {
    env.tickWith(turn.intents);
    if (turn.hash != null && env.hashTick === turn.turnNumber && env.hash !== turn.hash) {
      desync = turn.turnNumber;
      break;
    }
    for (let k = kills.length; k < env.conquests.length; k++) {
      const c = env.conquests[k];
      const loser = game.player(c.conqueredId);
      kills.push([game.ticks(), game.player(c.conquerorId).smallID(), loser.smallID(), loser.type()]);
    }
    if (game.ticks() % EVERY === 0 && !game.inSpawnPhase()) sample();
  }
  if (desync === null && t[t.length - 1] !== game.ticks()) sample(); // the final state
  const w = game.getWinner();
  return {
    id: record.info.gameID,
    mode,
    map: record.info.config.gameMap,
    size: record.info.config.gameMapSize,
    spawn: game.config().numSpawnPhaseTurns(),
    turns: record.turns.length,
    desync,
    winner: record.info.winner,
    simWinner: w === null ? null : typeof w === "string" ? w : w.smallID(),
    sides: (sides ?? ([] as Player[])).map((p) => [p.smallID(), p.type(), p.clientID() ?? p.name(), p.team()]),
    t,
    tiles,
    troops,
    gold,
    earned,
    units,
    allies,
    attackers,
    away,
    kills,
  };
}

async function main() {
  os.setPriority(0, os.constants.priority.PRIORITY_BELOW_NORMAL);
  console.log = console.warn = console.info = () => {}; // the game's chatter
  const [list, out] = process.argv.slice(2);
  const done = new Set<string>();
  if (fs.existsSync(out)) {
    for (const line of fs.readFileSync(out, "utf8").split("\n")) {
      const id = /^\{"id":"([^"]+)"/.exec(line)?.[1];
      if (id !== undefined) done.add(id);
    }
  }
  for (const line of fs.readFileSync(list, "utf8").split("\n")) {
    const [mode, file] = line.trim().split(/ (.+)/);
    const id = path.basename(file ?? "").replace(".json.gz", "");
    if (!file || done.has(id)) continue;
    const t0 = Date.now();
    let row: object;
    try {
      row = await trace(mode, file);
    } catch (e) {
      row = { id, mode, error: String(e) };
    }
    fs.appendFileSync(out, JSON.stringify(row) + "\n");
    process.stderr.write(`${id} ${mode} ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
