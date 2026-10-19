import { TileRef } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameMap";
import { GameMapSize } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { GameMapType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/Maps.gen";
import { GameConfig, StampedIntent } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/packages/engine-lib/src/PseudoRandom";
import { EnvConfig, EnvGame, seatClientID, seatPlayers } from "./game";
import { gameIdOf, publicConfig, shapeOf, slotsFor } from "./setups";

/** The Phase 1 gate setup: one agent against 3 Easy nations on Compact Onion, 5-minute timer. */
export const ONION: EnvConfig = {
  map: "Onion",
  compact: true,
  nations: 3,
  difficulty: "Easy",
  bots: 0,
  timerMinutes: 5,
  seats: 1,
};

/** Ranked 1v1 training games: map, size, bots and timer come from the playlist (bridge/setups.ts). */
export const RANKED: EnvConfig = { ...ONION, seats: 2, playlist: "ranked1v1" };

/** Public FFA training games: 4 learner seats against the map's nations, Medium or Hard, with a 30-minute timer. */
export const FFA: EnvConfig = {
  ...ONION,
  seats: 4,
  playlist: "ffa",
  maps: "training",
  difficulty: "Medium,Hard",
  timerMinutes: 30,
};

/** Public Humans vs Nations training games: 4 learner seats against 4 Hard nations. */
export const HVN: EnvConfig = { ...FFA, playlist: "hvn", difficulty: "Hard" };

/** A fixed land tile in the Onion map's lower right, for scripted spawns. */
export function landTile(env: EnvGame): TileRef {
  const g = env.game;
  let found = -1;
  g.forEachTile((t) => {
    if (found < 0 && g.isLand(t) && !g.isImpassable(t) && g.x(t) >= 100 && g.y(t) >= 100) found = t;
  });
  if (found < 0) throw new Error("no land tile found");
  return found;
}

export const spawnCell = (env: EnvGame): number => env.frame.cellOf(landTile(env));

/** Seat 0 spawns at tick 20 and attacks unowned land every 50 ticks after the spawn phase. */
export function script(env: EnvGame, ticks: number, onStep?: () => void): void {
  const tile = landTile(env);
  const clientID = seatClientID(0);
  for (let t = 0; t < ticks; t++) {
    const intents: StampedIntent[] =
      t === 20
        ? [{ type: "spawn", tile, clientID }]
        : t > 250 && t % 50 === 0
          ? [{ type: "attack", targetID: null, troops: 1000, clientID }]
          : [];
    env.tickWith(intents);
    if (t % 5 === 4) onStep?.();
  }
}

/** A public game on a fixed map and size, started as EnvGame.create starts it; `patch` overrides config fields. */
export function publicGame(
  cfg: EnvConfig,
  gameMap: GameMapType,
  gameMapSize: GameMapSize,
  patch: Partial<GameConfig> = {},
): Promise<EnvGame> {
  const bots = gameMapSize === GameMapSize.Compact ? 100 : 400;
  const config = { ...publicConfig(cfg, new PseudoRandom(1)), gameMap, gameMapSize, bots, ...patch };
  const start = { gameID: gameIdOf("nations"), lobbyCreatedAt: 0, config, players: seatPlayers(cfg.seats) };
  return EnvGame.fromStart(start, shapeOf([cfg]).grid[1], slotsFor(cfg), "nations:latency");
}

/** A quick public game: Onion, Compact, with a 1-minute timer unless `timerMinutes` says otherwise. */
export const quickGame = (cfg: EnvConfig, timerMinutes = 1, patch: Partial<GameConfig> = {}) =>
  publicGame({ ...cfg, timerMinutes }, GameMapType.Onion, GameMapSize.Compact, patch);

/** No nations and no tribes: only the seats are in the game. */
export const LONE: Partial<GameConfig> = { nations: "disabled", bots: 0 };

/**
 * Spawns the seats in `which` at tick 20, seat i on the i-th of the game's seats' land tiles spread evenly by x (in a
 * team game whose map has spawn areas, on its team's area, spread among the team's seats), then ticks past the spawn
 * phase. Throws if one of them has not spawned.
 */
export function spawnSeats(env: EnvGame, which: number[] = env.seats.map((_, i) => i)): void {
  const g = env.game;
  const spread = (area: { x: number; y: number; width: number; height: number } | undefined, k: number, n: number) => {
    const land: TileRef[] = [];
    g.forEachTile((t) => {
      const inside = !area || (g.x(t) >= area.x && g.x(t) < area.x + area.width && g.y(t) >= area.y && g.y(t) < area.y + area.height);
      if (inside && g.isLand(t) && !g.isImpassable(t)) land.push(t);
    });
    land.sort((a, b) => g.x(a) - g.x(b)); // stable: ties stay in row order
    return land[Math.floor(((k + 0.5) / n) * land.length)];
  };
  while (g.ticks() < 20) env.tickWith([]);
  env.tickWith(
    which.map((i) => {
      const team = env.seats[i].team();
      const area = team === null ? undefined : g.teamSpawnArea(team);
      const mates = area ? env.seats.map((s, j) => (s.team() === team ? j : -1)).filter((j) => j >= 0) : [];
      const tile = area ? spread(area, mates.indexOf(i), mates.length) : spread(undefined, i, env.seats.length);
      return { type: "spawn", tile, clientID: seatClientID(i) } as StampedIntent;
    }),
  );
  while (g.ticks() < g.config().numSpawnPhaseTurns() + 2) env.tickWith([]);
  for (const i of which) if (!env.seats[i].hasSpawned()) throw new Error(`seat ${i} did not spawn`);
}
