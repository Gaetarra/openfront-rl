import {
  Difficulty,
  GameMode,
  HumansVsNations,
  PlayerType,
  RankedType,
  UnitType,
} from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { GameConfig } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { Game, Player } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/Game";
import { BUILDS } from "./actions";
import { loaded } from "./frame";

export const PLAYER_FEATURES = 23;
/** The lobby kinds of the mode input (scalars MODE..MODE + 6, one-hot); trainer/env.py KINDS lists the same ones. */
export const KINDS = ["ranked1v1", "ranked2v2", "rankedffa", "clanwars", "ffa", "team", "hvn"] as const;
export const MODE = 26; // the first mode scalar: the lobby kind, then the non-bot players, the teams, the difficulty
export const SCALARS = MODE + KINDS.length + 3;

const RANKED_KINDS: Record<RankedType, (typeof KINDS)[number]> = {
  [RankedType.OneVOne]: "ranked1v1",
  [RankedType.TwoVTwo]: "ranked2v2",
  [RankedType.FreeForAll]: "rankedffa",
  [RankedType.ClanWars]: "clanwars",
};
const DIFFICULTY: Record<Difficulty, number> = {
  [Difficulty.Easy]: 0.25,
  [Difficulty.Medium]: 0.5,
  [Difficulty.Hard]: 0.75,
  [Difficulty.Impossible]: 1,
};

/** The index into KINDS of a game's lobby: its ranked type, else its mode (private and singleplayer FFA are "ffa"), else its teams. */
export function lobbyKind(gc: GameConfig): number {
  const kind = gc.rankedType
    ? RANKED_KINDS[gc.rankedType]
    : gc.gameMode === GameMode.FFA
      ? "ffa"
      : gc.playerTeams === HumansVsNations
        ? "hvn"
        : "team";
  return KINDS.indexOf(kind);
}

/** How a player-table row relates to the seat; the trainer maps cells and units through it. */
export enum Rel {
  None = 0,
  Self = 1,
  Friendly = 2,
  Enemy = 3,
  Bot = 4,
}

const logScale = (v: number, div: number) => Math.log1p(Math.max(0, v)) / div;

/**
 * Fills the seat's view of the player table (row = smallID, row 0 = unowned land) and its scalars. The arrays must be
 * zeroed: players [slots * PLAYER_FEATURES], rel [slots], scalars [SCALARS].
 *
 * Player features: 0 exists, 1 alive, 2 self, 3 friendly, 4 enemy, 5 bot, 6 nation, 7 tiles / land, 8 log troops,
 * 9 troops / max troops, 10 log gold, 11 traitor, 12 attackable by the seat, 13 log troops attacking the seat,
 * 14 log troops the seat sends at it, 15 spawned, 16 log troops per tile, 17 allied with the seat (a teammate is not an
 * ally), 18 its alliance request to the seat is pending, 19 the seat's request to it is pending, 20 the alliance's time
 * left / its duration, 21 it agreed to renew the alliance, 22 the seat agreed to renew it.
 * Scalars: 0 game progress, 1 spawn phase, 2 spawn-phase progress, 3 spawned, 4 alive, 5 log troops, 6 troops / max,
 * 7 log gold, 8 gold / city cost, 9 tiles / land, 10 boats / max boats, 11 alive players share, 12 time left,
 * 13 spawn immunity, 14 tiles per cell / 64, 15 share of players ahead in tiles, 16-24 gold / the current cost of
 * each build in BUILDS order (city, defense post, silo, factory, port, SAM, warship, atom bomb, hydrogen bomb; at
 * most 4, / 4), 25 missiles loaded in the seat's silos / 16 (at most 1).
 * The mode input, constant over a game and the same for a replayed record and a training game of one lobby: 26-32 the
 * lobby kind, one-hot in KINDS order (lobbyKind), 33 log2(non-bot players) / 8 (humans and nations, alive or not; at
 * most 1), 34 log2(distinct teams of those players) / 6 (at most 1; 0 without teams), 35 the nations' difficulty
 * (Easy .25, Medium .5, Hard .75, Impossible 1; 0 without a nation).
 */
export function encodeSeat(
  game: Game,
  seat: Player,
  horizonMinutes: number,
  cellSize: number,
  attack: Uint8Array,
  players: Float32Array,
  rel: Uint8Array,
  scalars: Float32Array,
): void {
  const F = PLAYER_FEATURES;
  const slots = rel.length;
  const land = game.numLandTiles();
  const config = game.config();
  const incoming = new Float64Array(slots);
  const outgoing = new Float64Array(slots);
  for (const a of seat.incomingAttacks()) incoming[a.attacker().smallID()] += a.troops();
  for (const a of seat.outgoingAttacks()) outgoing[a.target().smallID()] += a.troops();
  const asked = new Set(seat.incomingAllianceRequests().map((r) => r.requestor().smallID()));
  const askedBy = new Set(seat.outgoingAllianceRequests().map((r) => r.recipient().smallID()));

  let owned = 0;
  for (const p of game.allPlayers()) {
    const s = p.smallID();
    if (s >= slots) throw new Error(`player smallID ${s} does not fit ${slots} table rows`);
    const tiles = p.numTilesOwned();
    const troops = p.troops();
    owned += tiles;
    const self = p === seat;
    const bot = p.type() === PlayerType.Bot;
    const friendly = !self && seat.isFriendly(p);
    rel[s] = self ? Rel.Self : bot ? Rel.Bot : friendly ? Rel.Friendly : Rel.Enemy;
    const o = s * F;
    players[o] = 1;
    players[o + 1] = p.isAlive() ? 1 : 0;
    players[o + 2] = self ? 1 : 0;
    players[o + 3] = friendly ? 1 : 0;
    players[o + 4] = !self && !friendly && !bot ? 1 : 0;
    players[o + 5] = bot ? 1 : 0;
    players[o + 6] = p.type() === PlayerType.Nation ? 1 : 0;
    players[o + 7] = tiles / land;
    players[o + 8] = logScale(troops, 15);
    players[o + 9] = Math.min(troops / Math.max(config.maxTroops(p), 1), 2) / 2;
    players[o + 10] = logScale(Number(p.gold()), 20);
    players[o + 11] = p.isTraitor() ? 1 : 0;
    players[o + 12] = attack[s];
    players[o + 13] = logScale(incoming[s], 15);
    players[o + 14] = logScale(outgoing[s], 15);
    players[o + 15] = p.hasSpawned() ? 1 : 0;
    players[o + 16] = logScale(troops / Math.max(tiles, 1), 5);
    const alliance = seat.allianceWith(p);
    players[o + 17] = alliance ? 1 : 0;
    players[o + 18] = asked.has(s) ? 1 : 0;
    players[o + 19] = askedBy.has(s) ? 1 : 0;
    if (alliance) {
      players[o + 20] = Math.min(Math.max((alliance.expiresAt() - game.ticks()) / config.allianceDuration(), 0), 1);
      players[o + 21] = alliance.agreedToExtend(p) ? 1 : 0;
      players[o + 22] = alliance.agreedToExtend(seat) ? 1 : 0;
    }
  }
  rel[0] = Rel.None;
  players[0] = 1;
  players[1] = 1;
  players[7] = Math.max(0, land - owned) / land;
  players[12] = attack[0];
  players[14] = logScale(outgoing[0], 15);

  const spawnTurns = config.numSpawnPhaseTurns();
  const nonBot = game.allPlayers().filter((p) => p.type() !== PlayerType.Bot);
  const alive = nonBot.filter((p) => p.isAlive());
  const myTiles = seat.numTilesOwned();
  const troops = seat.troops();
  const gold = Number(seat.gold());
  const cityCost = Number(game.unitInfo(UnitType.City).cost(game, seat));
  const ahead = alive.filter((p) => p !== seat && p.numTilesOwned() > myTiles).length;
  scalars[0] = Math.min(game.ticks() / (spawnTurns + horizonMinutes * 600), 1);
  scalars[1] = game.inSpawnPhase() ? 1 : 0;
  scalars[2] = Math.min(game.ticks() / spawnTurns, 1);
  scalars[3] = seat.hasSpawned() ? 1 : 0;
  scalars[4] = seat.isAlive() ? 1 : 0;
  scalars[5] = logScale(troops, 15);
  scalars[6] = Math.min(troops / Math.max(config.maxTroops(seat), 1), 2) / 2;
  scalars[7] = logScale(gold, 20);
  scalars[8] = Math.min(gold / Math.max(cityCost, 1), 4) / 4;
  scalars[9] = myTiles / land;
  scalars[10] = seat.unitCount(UnitType.TransportShip) / Math.max(config.boatMaxNumber(), 1);
  scalars[11] = alive.length / Math.max(nonBot.length, 1);
  scalars[12] = Math.min(1, Math.max(0, 1 - game.elapsedGameSeconds() / (horizonMinutes * 60)));
  scalars[13] = game.isSpawnImmunityActive() ? 1 : 0;
  scalars[14] = (cellSize * cellSize) / 64;
  scalars[15] = seat.isAlive() ? ahead / Math.max(nonBot.length - 1, 1) : 1;
  let k = 16;
  for (const unit of BUILDS.values()) {
    scalars[k++] = Math.min(gold / Math.max(Number(game.unitInfo(unit).cost(game, seat)), 1), 4) / 4;
  }
  const silos = seat.units(UnitType.MissileSilo).reduce((n, u) => n + loaded(u), 0);
  scalars[25] = Math.min(silos, 16) / 16;
  const gc = config.gameConfig();
  scalars[MODE + lobbyKind(gc)] = 1;
  scalars[MODE + KINDS.length] = Math.min(Math.log2(Math.max(nonBot.length, 1)) / 8, 1);
  scalars[MODE + KINDS.length + 1] = Math.min(Math.log2(Math.max(new Set(nonBot.map((p) => p.team())).size, 1)) / 6, 1);
  scalars[MODE + KINDS.length + 2] = nonBot.some((p) => p.type() === PlayerType.Nation) ? DIFFICULTY[gc.difficulty] : 0;
}
