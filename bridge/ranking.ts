import { Game, GameMode, Player, PlayerType, Team } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";

export interface Standing {
  /** The game's winner is this player or its team. */
  won: boolean;
  spawned: boolean;
  /** Placement the game recorded when the player died (non-bot players still standing + 1). */
  deathPosition?: number;
  tiles: number;
}

/**
 * 1 = best. The winner is first; dead players keep the placement recorded at their death; players alive at the end
 * are ordered by tiles; a player who never spawned is last. `others`: tiles of the other alive non-bot players.
 */
export function rankOf(me: Standing, others: number[], n: number): number {
  if (me.won) return 1;
  if (!me.spawned) return n;
  if (me.deathPosition !== undefined) return me.deathPosition;
  return 1 + others.filter((t) => t > me.tiles).length;
}

/** Terminal reward (PLAN decision 9): +1 for a win, otherwise -1 + 0.5 (n - rank) / (n - 1). */
export function rewardOf(rank: number, n: number): number {
  if (rank === 1) return 1;
  return -1 + (0.5 * (n - rank)) / (n - 1);
}

/**
 * The seat's rank among the game's non-bot players (humans and nations). In a Team game the seats of a team share
 * their team's rank among the teams of those players: the winner's team first, the others by their members' summed
 * tiles, a team that never spawned among them (it just holds no tiles).
 */
export function seatStanding(game: Game, seat: Player): { rank: number; n: number } {
  const nonBot = game.allPlayers().filter((p) => p.type() !== PlayerType.Bot);
  const winner = game.getWinner();
  const won = winner !== null && (winner === seat || (typeof winner === "string" && seat.team() === winner));
  if (game.config().gameConfig().gameMode === GameMode.Team) {
    const tiles = new Map<Team | null, number>();
    for (const p of nonBot) tiles.set(p.team(), (tiles.get(p.team()) ?? 0) + p.numTilesOwned());
    const others = [...tiles].filter(([t]) => t !== seat.team()).map(([t, n]) => (t === winner ? Infinity : n));
    const team: Standing = { won, spawned: true, tiles: tiles.get(seat.team()) ?? 0 };
    return { rank: rankOf(team, others, tiles.size), n: tiles.size };
  }
  const others = nonBot.filter((p) => p !== seat && p.isAlive()).map((p) => p.numTilesOwned());
  const standing: Standing = {
    won,
    spawned: seat.hasSpawned(),
    deathPosition: seat.isAlive() ? undefined : (game.stats().getPlayerStats(seat)?.deathPosition ?? undefined),
    tiles: seat.numTilesOwned(),
  };
  return { rank: rankOf(standing, others, nonBot.length), n: nonBot.length };
}
