/**
 * The share of its troops a human sends with each attack and boat, from replayed ranked 1v1 records (PLAN 3B, troop
 * shares). A share is taken against the player's troops just before the command executes. Commands within a second of
 * the player's previous one are reported apart: their shares drift, since the client computed them on troops the
 * earlier command had not removed yet. Runs at below-normal priority, next to a training run.
 *
 * Run from the game dir (its tsconfig paths and node_modules are needed), ~10 s per game:
 *   npx tsx ../../scripts/troop_shares.ts [games=60]
 */
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { readRecord, RecordGame } from "../bridge/replay";
import { gridFor, slotsFor } from "../bridge/setups";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RANKED = { map: "Australia", compact: false, nations: 0, difficulty: "Medium", bots: 400, timerMinutes: 15,
  seats: 2, playlist: "ranked1v1" as const };
const GROUPS = [[1, 9], [10, 10], [11, 19], [20, 20], [21, 49], [50, 100], [75, 100]];
/** The five troop sizes of the Phase 2 and 3A networks, and the nearest one in log scale (how their labels were made). */
const SIZES = [0.1, 0.25, 0.5, 0.75, 1.0];
function sizeOf(share: number): number {
  const s = Math.min(Math.max(share, SIZES[0]), 1);
  let best = 0;
  for (let i = 1; i < SIZES.length; i++) {
    if (Math.abs(Math.log(s / SIZES[i])) < Math.abs(Math.log(s / SIZES[best]))) best = i;
  }
  return best;
}

async function main() {
  os.setPriority(0, os.constants.priority.PRIORITY_BELOW_NORMAL);
  const print = console.log;
  console.log = console.warn = () => {}; // the game's chatter
  const n = Number(process.argv[2] ?? 60);
  const dir = path.join(ROOT, "data", "records");
  const files = fs.readdirSync(dir).sort();
  const shares: Record<string, number[]> = { attack: [], boat: [], "follow-up": [] };
  let desync = 0;
  for (let i = 0; i < n; i++) {
    const rg = await RecordGame.create(readRecord(path.join(dir, files[Math.floor((i * files.length) / n)])),
      gridFor(RANKED)[1], slotsFor(RANKED));
    const env = rg.env;
    const seatOf = new Map(env.start.players.map((p, s) => [p.clientID, s]));
    const last = env.seats.map(() => -Infinity);
    for (const turn of rg.record.turns) {
      for (const it of turn.intents) {
        if ((it.type !== "attack" && it.type !== "boat") || it.troops === null) continue;
        const s = seatOf.get(it.clientID);
        if (s === undefined) continue;
        const share = (100 * it.troops) / Math.max(env.seats[s].troops(), 1);
        shares[turn.turnNumber - last[s] <= 10 ? "follow-up" : it.type].push(share);
        last[s] = turn.turnNumber;
      }
      env.tickWith(turn.intents);
      if (turn.hash != null && env.hashTick === turn.turnNumber && env.hash !== turn.hash) {
        desync++;
        break;
      }
    }
  }
  print(`${n} games, ${desync} desynced`);
  for (const [kind, all] of Object.entries(shares)) {
    const pct = (k: number) => `${((100 * k) / all.length).toFixed(1)}%`;
    const whole = all.map((x) => Math.round(x));
    const counts = new Map<number, number>();
    for (const k of whole) counts.set(k, (counts.get(k) ?? 0) + 1);
    const top = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 8);
    print(`\n${kind}: ${all.length}`);
    print("  " + GROUPS.map(([a, b]) => `${a}-${b}%: ${pct(whole.filter((k) => k >= a && k <= b).length)}`)
      .join(", ") + `, over 100%: ${pct(whole.filter((k) => k > 100).length)}`);
    print("  most common: " + top.map(([k, c]) => `${k}% ${pct(c)}`).join(", "));
    if (kind === "follow-up") continue;
    const moved = all.map((x) => Math.abs(x - 100 * SIZES[sizeOf(x / 100)]));
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    print(`  the ${SIZES.length} sizes move a command by ${mean(moved).toFixed(1)} points on average` +
      ` (${(100 * mean(moved.map((m, j) => m / Math.max(all[j], 1)))).toFixed(0)}% of it); within 0.25 points of a whole` +
      ` percent: ${pct(all.filter((x) => Math.abs(x - Math.round(x)) <= 0.25).length)}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
