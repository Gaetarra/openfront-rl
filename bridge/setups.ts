import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  RankedType,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GameConfig } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import type { EnvConfig } from "./game";
import { mapGrid } from "./maps";

const ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/**
 * The game ID of a seed: 8 alphanumerics, as GAME_ID_REGEX requires. A game seeds its RNG from its ID (as the game's
 * client does when it replays a record), so equal seeds give equal games.
 */
export function gameIdOf(seed: string): string {
  const random = new PseudoRandom(simpleHash(seed));
  let id = "";
  for (let i = 0; i < 8; i++) id += ALNUM[random.nextInt(0, ALNUM.length)];
  return id;
}

/** Ranked 1v1 maps with the official weights: Australia 40%, the others 20% each. */
export const RANKED_1V1_MAPS: GameMapType[] = [
  GameMapType.Australia,
  GameMapType.Australia,
  GameMapType.Iceland,
  GameMapType.Asia,
  GameMapType.EuropeClassic,
];

/**
 * A ranked 1v1 game as the official servers create it: a copy of get1v1Config in src/server/MapPlaylist.ts (v0.34.21),
 * which draws with Math.random. Compare it with the game's on every upgrade.
 */
export function ranked1v1Config(random: PseudoRandom): GameConfig {
  const compact = random.next() < 0.2;
  return {
    donateGold: false,
    donateTroops: false,
    gameMap: RANKED_1V1_MAPS[random.nextInt(0, RANKED_1V1_MAPS.length)],
    maxPlayers: 2,
    gameType: GameType.Public,
    gameMapSize: compact ? GameMapSize.Compact : GameMapSize.Normal,
    difficulty: Difficulty.Medium,
    rankedType: RankedType.OneVOne,
    infiniteGold: false,
    infiniteTroops: false,
    maxTimerValue: compact ? 10 : 15,
    instantBuild: false,
    randomSpawn: false,
    nations: "disabled",
    gameMode: GameMode.FFA,
    bots: compact ? 100 : 400,
    spawnImmunityDuration: 30 * 10,
    disabledUnits: [],
  };
}

/** The game config of one training game. `random` is only used for playlist draws. */
export function gameConfigOf(cfg: EnvConfig, random: PseudoRandom): GameConfig {
  if (cfg.playlist === "ranked1v1") {
    if (cfg.seats !== 2) throw new Error("the ranked1v1 playlist needs 2 seats");
    return ranked1v1Config(random);
  }
  return {
    gameMap: cfg.map as GameMapType,
    gameMapSize: cfg.compact ? GameMapSize.Compact : GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Private,
    difficulty: cfg.difficulty as Difficulty,
    nations: cfg.nations > 0 ? cfg.nations : "disabled",
    bots: cfg.bots,
    donateGold: false,
    donateTroops: false,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
    maxTimerValue: cfg.timerMinutes,
  };
}

/** Every map and size a config can produce. */
function mapsOf(cfg: EnvConfig): [GameMapType, GameMapSize][] {
  if (cfg.playlist === "ranked1v1") {
    return [...new Set(RANKED_1V1_MAPS)].flatMap((m): [GameMapType, GameMapSize][] => [
      [m, GameMapSize.Normal],
      [m, GameMapSize.Compact],
    ]);
  }
  return [[cfg.map as GameMapType, cfg.compact ? GameMapSize.Compact : GameMapSize.Normal]];
}

/** The trainer's cell grid [rows, columns]: fits every map the config can produce, padded to multiples of 16 (the CNN
 *  downsamples by 16). */
export function gridFor(cfg: EnvConfig): [number, number] {
  let h = 0;
  let w = 0;
  for (const [map, size] of mapsOf(cfg)) {
    const g = mapGrid(map, size);
    h = Math.max(h, g.h);
    w = Math.max(w, g.w);
  }
  return [Math.ceil(h / 16) * 16, Math.ceil(w / 16) * 16];
}

/** Player-table rows: row 0 (unowned land), the seats, the nations and the most bots any game of the config has. */
export function slotsFor(cfg: EnvConfig): number {
  if (cfg.playlist === "ranked1v1") return 1 + 2 + 400;
  return 1 + cfg.seats + cfg.nations + cfg.bots;
}
