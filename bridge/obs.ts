import { Game, Player, PlayerType, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";

export const PLAYER_FEATURES = 17;
export const SCALARS = 16;

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
 * 14 log troops the seat sends at it, 15 spawned, 16 log troops per tile.
 * Scalars: 0 game progress, 1 spawn phase, 2 spawn-phase progress, 3 spawned, 4 alive, 5 log troops, 6 troops / max,
 * 7 log gold, 8 gold / city cost, 9 tiles / land, 10 boats / max boats, 11 alive players share, 12 time left,
 * 13 spawn immunity, 14 tiles per cell / 64, 15 share of players ahead in tiles.
 */
export function encodeSeat(
  game: Game,
  seat: Player,
  timerMinutes: number,
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
  scalars[0] = Math.min(game.ticks() / (spawnTurns + timerMinutes * 600), 1);
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
  scalars[12] = Math.min(1, Math.max(0, 1 - game.elapsedGameSeconds() / (timerMinutes * 60)));
  scalars[13] = game.isSpawnImmunityActive() ? 1 : 0;
  scalars[14] = (cellSize * cellSize) / 64;
  scalars[15] = seat.isAlive() ? ahead / Math.max(nonBot.length - 1, 1) : 1;
}
