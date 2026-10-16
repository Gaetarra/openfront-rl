import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  HumansVsNations,
  maps as allMaps,
  RankedType,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GameConfig } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import type { EnvConfig } from "./game";
import { mapGrid, nationCount } from "./maps";

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
 * A ranked 1v1 game as the official servers create it: a copy of get1v1Config in src/server/MapPlaylist.ts (v0.34.22),
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

/**
 * The maps held out of FFA and Humans vs Nations training, for Gate 4N's games on unseen maps (4N plan, Decision 1).
 * The draw: the 30 maps that fit the ranked grid, have a rotation weight and are not World, Australia, Iceland or Asia,
 * sorted by name, then `new PseudoRandom(simpleHash("4N held out")).shuffleArray(candidates).slice(0, 4)`. Fixed here,
 * so a game upgrade that adds a map cannot change them.
 */
export const HELD_OUT_MAPS: GameMapType[] = [
  GameMapType.Didier,
  GameMapType.BeringStrait,
  GameMapType.YellowSea,
  GameMapType.Labyrinth,
];

/**
 * The gate's seen maps, trained on and played to measure the gap to the held-out ones (4N plan, Decision 2): drawn the
 * same way from the 26 candidates left, with the seed "4N seen".
 */
export const SEEN_MAPS: GameMapType[] = [
  GameMapType.BlackSea,
  GameMapType.Venice,
  GameMapType.Mars,
  GameMapType.MoreThanLuck,
];

const trainingMaps = new Map<boolean, GameMapType[]>();

/**
 * The maps a public game draws from, uniformly. "training": each map that fits the ranked grid, has a weight above 0
 * and is not held out, repeated by its weight in the FFA rotation (`team`: the Team rotation) as
 * MapPlaylist.buildMapsList does. "heldout" and "seen": their 4 maps once each.
 */
export function publicMaps(team: boolean, which: "training" | "heldout" | "seen"): GameMapType[] {
  if (which === "heldout") return HELD_OUT_MAPS;
  if (which === "seen") return SEEN_MAPS;
  let list = trainingMaps.get(team);
  if (list === undefined) {
    const [rows, cols] = gridOf(bothSizes(RANKED_1V1_MAPS));
    list = allMaps.flatMap(({ type, ffaFrequency, teamFrequency, multiplayerFrequency }) => {
      const own = team ? teamFrequency : ffaFrequency;
      const weight = own >= 0 ? own : multiplayerFrequency;
      if (weight <= 0 || HELD_OUT_MAPS.includes(type)) return [];
      const { h, w } = mapGrid(type, GameMapSize.Normal);
      return h <= rows && w <= cols ? Array<GameMapType>(weight).fill(type) : [];
    });
    trainingMaps.set(team, list);
  }
  return list;
}

/**
 * A public FFA or Humans vs Nations game as the official servers create it: a copy of the FFA and Team branch of
 * MapPlaylist.rollConfig in src/server/MapPlaylist.ts (v0.34.24), which draws with Math.random. Compare it with the
 * game's on every upgrade. Ours: the lobby size is the seats, the map comes from `publicMaps`, the difficulty is one
 * of the config's list and the timer is the config's (the public games have none).
 */
export function publicConfig(cfg: EnvConfig, random: PseudoRandom): GameConfig {
  const team = cfg.playlist === "hvn";
  const pool = publicMaps(team, cfg.maps ?? "training");
  const gameMap = pool[random.nextInt(0, pool.length)];
  const compact = random.next() < 1 / 3; // the rotation makes every third game Compact
  const difficulties = cfg.difficulty.split(",");
  return {
    donateGold: team,
    donateTroops: team,
    gameMap,
    maxPlayers: cfg.seats,
    gameType: GameType.Public,
    gameMapSize: compact ? GameMapSize.Compact : GameMapSize.Normal,
    difficulty: difficulties[random.nextInt(0, difficulties.length)] as Difficulty,
    infiniteGold: false,
    infiniteTroops: false,
    maxTimerValue: cfg.timerMinutes > 0 ? cfg.timerMinutes : undefined,
    instantBuild: false,
    randomSpawn: false,
    nations: "default",
    gameMode: team ? GameMode.Team : GameMode.FFA,
    playerTeams: team ? HumansVsNations : undefined,
    bots: compact ? 100 : 400,
    spawnImmunityDuration: 5 * 10,
    disabledUnits: [],
    disableClanTags: team ? undefined : true,
    overtime: team ? undefined : { enabled: true },
  };
}

/** The game config of one training game. `random` is only used for playlist draws. */
export function gameConfigOf(cfg: EnvConfig, random: PseudoRandom): GameConfig {
  if (cfg.playlist === "ranked1v1") {
    if (cfg.seats !== 2) throw new Error("the ranked1v1 playlist needs 2 seats");
    return ranked1v1Config(random);
  }
  if (cfg.playlist === "ffa" || cfg.playlist === "hvn") return publicConfig(cfg, random);
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

/** The distinct maps of a list, each at both sizes. */
function bothSizes(maps: GameMapType[]): [GameMapType, GameMapSize][] {
  return [...new Set(maps)].flatMap((m): [GameMapType, GameMapSize][] => [
    [m, GameMapSize.Normal],
    [m, GameMapSize.Compact],
  ]);
}

/**
 * Every map and size a config can produce. FFA and Humans vs Nations take every public map that fits the ranked grid
 * (training's and the held-out ones) whatever `maps` they draw from: the network pools its last CNN stage over the
 * whole grid, padding included, so the gate's games on 4 maps (176 x 192 alone) must play on the grid training used.
 */
function mapsOf(cfg: EnvConfig): [GameMapType, GameMapSize][] {
  if (cfg.playlist === "ranked1v1") return bothSizes(RANKED_1V1_MAPS);
  if (cfg.playlist === "ffa" || cfg.playlist === "hvn") {
    return bothSizes([...publicMaps(cfg.playlist === "hvn", "training"), ...HELD_OUT_MAPS]);
  }
  return [[cfg.map as GameMapType, cfg.compact ? GameMapSize.Compact : GameMapSize.Normal]];
}

/** The cell grid [rows, columns] that fits every (map, size) pair, padded to multiples of 16 (the CNN downsamples by
 *  16). */
function gridOf(pairs: [GameMapType, GameMapSize][]): [number, number] {
  let h = 0;
  let w = 0;
  for (const [map, size] of pairs) {
    const g = mapGrid(map, size);
    h = Math.max(h, g.h);
    w = Math.max(w, g.w);
  }
  return [Math.ceil(h / 16) * 16, Math.ceil(w / 16) * 16];
}

/** The trainer's cell grid [rows, columns]: fits every map the config can produce. */
export function gridFor(cfg: EnvConfig): [number, number] {
  return gridOf(mapsOf(cfg));
}

/**
 * Player-table rows: row 0 (unowned land), the seats, the nations and the most bots any game of the config has. Unlike
 * the grid, it follows the config's own `maps`: absent players are masked, so a gate's table may be smaller than
 * training's.
 */
export function slotsFor(cfg: EnvConfig): number {
  switch (cfg.playlist) {
    case "ranked1v1":
      return 1 + 2 + 400;
    case "ffa": {
      // a Normal game has all of its map's nations, a Compact one fewer
      const maps = [...new Set(publicMaps(false, cfg.maps ?? "training"))];
      return 1 + cfg.seats + Math.max(...maps.map(nationCount)) + 400;
    }
    case "hvn": // as many nations as seats
      return 1 + 2 * cfg.seats + 400;
    default:
      return 1 + cfg.seats + cfg.nations + cfg.bots;
  }
}

/** The grid and player table that fit every config of a run, so that its games can share them. */
export function shapeOf(cfgs: EnvConfig[]): { grid: [number, number]; slots: number } {
  const grids = cfgs.map(gridFor);
  return {
    grid: [Math.max(...grids.map(([rows]) => rows)), Math.max(...grids.map(([, cols]) => cols))],
    slots: Math.max(...cfgs.map(slotsFor)),
  };
}
