import fs from "node:fs";
import zlib from "node:zlib";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { StampedIntent } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { Game, Player } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/Game";
import {
  decompressGameRecord,
  toWireGameStartInfo,
} from "../OpenFront_Source_Code/openfrontio/packages/shared/src/SharedUtil";
import { GameRecord } from "../OpenFront_Source_Code/openfrontio/packages/shared/src/WireSchemas";
import { Act, BUILDS } from "./actions";
import { EnvGame, STEP_TICKS } from "./game";

/** Label types of windows that stay out of the loss (PLAN Phase 2, b). */
export const OUT_OF_SET = -1; // a command the actions cannot express: gold donations, rejects, embargoes, chat, MIRVs, ...
export const UNMAPPED = -2; // a command of an expressible type whose label could not be built
/** Commands a seat may have queued; more are dropped and counted (humans rarely issue 2 in one window). */
export const MAX_PENDING = 3;

/** A record from a .json or .json.gz file, with every turn present (records store only non-empty turns). */
export function readRecord(file: string): GameRecord {
  const raw = fs.readFileSync(file);
  const text = (file.endsWith(".gz") ? zlib.gunzipSync(raw) : raw).toString("utf8");
  return decompressGameRecord(JSON.parse(text) as GameRecord);
}

/**
 * The share of its troops a player sent, as a whole percent (1-100). Commands execute a few ticks after the click and
 * troops change meanwhile: 84% of human attacks land within 0.25 points of a whole percent (PLAN 3B).
 */
export function shareOf(troops: number, have: number): number {
  return Math.min(100, Math.max(1, Math.round((100 * troops) / Math.max(have, 1))));
}

/** The action of each unit a build_unit command can place (BUILDS reversed). */
const BUILD_ACT = new Map([...BUILDS].map(([act, unit]) => [unit, act]));

/** A human command as the [type, cell, target, share] action the agent would take for it, from the current state. */
export function labelOf(env: EnvGame, seat: Player, intent: StampedIntent): number[] {
  const { game, frame } = env;
  switch (intent.type) {
    case "spawn":
      return [Act.Spawn, frame.cellOf(intent.tile), 0, 0];
    case "attack": {
      if (intent.troops === null) return [UNMAPPED, 0, 0, 0];
      if (intent.targetID !== null && !game.hasPlayer(intent.targetID)) return [UNMAPPED, 0, 0, 0];
      const target = intent.targetID === null ? 0 : game.player(intent.targetID).smallID();
      return [Act.Attack, 0, target, shareOf(intent.troops, seat.troops())];
    }
    case "boat":
      return [Act.Boat, frame.cellOf(intent.dst), 0, shareOf(intent.troops, seat.troops())];
    case "build_unit": {
      const act = BUILD_ACT.get(intent.unit);
      return act === undefined ? [OUT_OF_SET, 0, 0, 0] : [act, frame.cellOf(intent.tile), 0, 0];
    }
    case "upgrade_structure": {
      const unit = game.unit(intent.unitId);
      return unit === undefined ? [UNMAPPED, 0, 0, 0] : [Act.Upgrade, frame.cellOf(unit.tile()), 0, 0];
    }
    case "cancel_attack": {
      const attack = seat.outgoingAttacks().find((a) => a.id() === intent.attackID);
      if (attack === undefined) return [UNMAPPED, 0, 0, 0];
      const target = attack.target();
      return [Act.Retreat, 0, target.isPlayer() ? target.smallID() : 0, 0];
    }
    case "allianceRequest":
    case "allianceExtension":
    case "breakAlliance":
    case "donate_troops": {
      if (!game.hasPlayer(intent.recipient)) return [UNMAPPED, 0, 0, 0];
      const to = game.player(intent.recipient).smallID();
      if (intent.type === "donate_troops") {
        const sent = intent.troops ?? game.config().defaultDonationAmount(seat);
        return [Act.DonateTroops, 0, to, shareOf(sent, seat.troops())];
      }
      const act = { allianceRequest: Act.AllianceRequest, allianceExtension: Act.RenewAlliance, breakAlliance: Act.BreakAlliance };
      return [act[intent.type], 0, to, 0];
    }
    case "move_warship": {
      const first = game.unit(intent.unitIds[0]);
      if (first === undefined || first.type() !== UnitType.Warship || first.owner() !== seat) return [UNMAPPED, 0, 0, 0];
      return [Act.MoveWarships, frame.cellOf(first.tile()), frame.cellOf(intent.tile), intent.unitIds.length > 1 ? 1 : 0];
    }
    default:
      return [OUT_OF_SET, 0, 0, 0];
  }
}

/** The side of a player: its team, or itself without teams. */
type Sides = { playerByClientID(id: string): { team(): string | null } | null };
const sideOf = (game: Sides, id: string): string => game.playerByClientID(id)?.team() ?? id;

/** The winning side of a record: the winner's team or itself, the winning team, or null for a nations' win. */
function winningSide(info: GameRecord["info"], game: Sides): string | null {
  const w = info.winner;
  return w?.[0] === "player" ? sideOf(game, w[1]) : w?.[0] === "team" ? w[1] : null;
}

/**
 * The clientIDs of the `count` players a replay imitates: the best-placed human of each of the best-placed sides. The
 * winning side comes first, then players alive at the end before the dead (the later death first), then more peak
 * tiles, then record order. Players missing from the game or without land are not eligible. With `count` equal to the
 * record's players, all of them in record order.
 */
export function pickSeats(info: GameRecord["info"], game: Sides, count: number): string[] {
  if (count === info.players.length) return info.players.map((p) => p.clientID);
  const winning = winningSide(info, game);
  const stat = (p: GameRecord["info"]["players"][number]) => ({
    peak: Number(p.stats?.tiles?.[0] ?? 0),
    killed: p.stats?.killedAt === undefined ? Infinity : Number(p.stats.killedAt),
    won: sideOf(game, p.clientID) === winning,
  });
  const order = info.players
    .map((p, i) => ({ id: p.clientID, i, ...stat(p), here: game.playerByClientID(p.clientID) !== null }))
    .filter((p) => p.here && p.peak > 0)
    .sort((a, b) => +b.won - +a.won || b.killed - a.killed || b.peak - a.peak || a.i - b.i);
  if (order.length < count) throw new Error(`only ${order.length} eligible players`);
  const picked: string[] = [];
  const sides = new Set<string>();
  for (const p of order) {
    if (picked.length < count && !sides.has(sideOf(game, p.id))) {
      picked.push(p.id);
      sides.add(sideOf(game, p.id));
    }
  }
  for (const p of order) if (picked.length < count && !picked.includes(p.id)) picked.push(p.id);
  return picked;
}

/**
 * Replays a game record window by window (STEP_TICKS turns) for behaviour cloning. The seats are the record's players
 * (or the best-placed few, pickSeats) and the observations come from the same EnvGame code as training games. A seat's
 * label for a window is the oldest of its queued commands; the commands of the turns about to execute are queued
 * first, so a command labels the observation its player saw just before it executed.
 */
export class RecordGame {
  window = 0;
  done = false;
  /** A recorded hash differed from the replay's: the rest of the record is unusable. */
  desync = false;
  /** Commands dropped because the seat already had MAX_PENDING queued. */
  dropped = 0;
  /** The command behind each seat's label of the last labels() call (undefined for a no-op). */
  chosen: (StampedIntent | undefined)[] = [];
  private readonly pending: StampedIntent[][];
  private readonly seatOf: Map<string, number>;

  private constructor(
    readonly env: EnvGame,
    readonly record: GameRecord,
    /** Per seat: +1 on the winning side (the winner, or its team), -1 otherwise. */
    readonly outcome: number[],
  ) {
    this.pending = env.seats.map(() => []);
    this.seatOf = new Map(env.seats.map((p, i) => [p.clientID()!, i]));
  }

  /** `seats`: how many players to imitate (pickSeats; throws when the record has too few eligible), 0 for all. */
  static async create(record: GameRecord, stride: number, slots: number, seats = 0): Promise<RecordGame> {
    const info = record.info;
    const start = toWireGameStartInfo({
      gameID: info.gameID,
      lobbyCreatedAt: info.lobbyCreatedAt,
      config: info.config,
      players: info.players,
      tribes: info.tribes,
    });
    const choose = seats === 0 ? undefined : (game: Game) => pickSeats(info, game, seats);
    const env = await EnvGame.fromStart(start, stride, slots, info.gameID, choose);
    const winning = winningSide(info, env.game);
    return new RecordGame(env, record, env.seats.map((p) => (sideOf(env.game, p.clientID()!) === winning ? 1 : -1)));
  }

  /** Windows until the record ends. */
  get left(): number {
    return Math.max(0, Math.ceil(this.record.turns.length / STEP_TICKS) - this.window);
  }

  /** One label row per seat for the current observation. Call it once per window, before advance(). */
  labels(): number[][] {
    const t0 = this.window * STEP_TICKS;
    for (const turn of this.record.turns.slice(t0, t0 + STEP_TICKS)) {
      for (const intent of turn.intents) {
        const s = this.seatOf.get(intent.clientID);
        if (s === undefined || intent.type === "mark_disconnected") continue;
        if (this.pending[s].length >= MAX_PENDING) this.dropped++;
        else this.pending[s].push(intent);
      }
    }
    this.chosen = this.env.seats.map((_, s) => this.pending[s].shift());
    return this.env.seats.map((seat, s) => {
      const intent = this.chosen[s];
      if (seat.isDisconnected()) return [UNMAPPED, 0, 0, 0]; // a gone player's window stays out of the loss
      return intent === undefined ? [Act.Noop, 0, 0, 0] : labelOf(this.env, seat, intent);
    });
  }

  /** Executes the window's turns and checks every recorded hash. */
  advance(): void {
    const t0 = this.window * STEP_TICKS;
    for (const turn of this.record.turns.slice(t0, t0 + STEP_TICKS)) {
      this.env.tickWith(turn.intents);
      if (turn.hash != null && this.env.hashTick === turn.turnNumber && this.env.hash !== turn.hash) {
        this.desync = true;
        break;
      }
    }
    this.window++;
    this.done = this.desync || this.window * STEP_TICKS >= this.record.turns.length;
  }
}
