import { TileRef } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameMap";
import { StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { EnvConfig, EnvGame, seatClientID } from "./game";

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
