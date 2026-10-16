import assert from "node:assert/strict";
import { test } from "node:test";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  HumansVsNations,
  maps as allMaps,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GAME_ID_REGEX, GameConfig, GameConfigSchema } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { mapGrid } from "./maps";
import {
  gameIdOf,
  gridFor,
  HELD_OUT_MAPS,
  publicConfig,
  publicMaps,
  ranked1v1Config,
  SEEN_MAPS,
  shapeOf,
  slotsFor,
} from "./setups";
import { FFA, HVN, ONION, RANKED } from "./testkit";

test("game IDs are valid, stable and differ between seeds", () => {
  const id = gameIdOf("run-w0-g0-e0");
  assert.match(id, GAME_ID_REGEX);
  assert.equal(gameIdOf("run-w0-g0-e0"), id);
  assert.notEqual(gameIdOf("run-w0-g0-e1"), id);
});

test("the ranked 1v1 playlist has the official weights and valid configs", () => {
  const random = new PseudoRandom(7);
  const n = 2000;
  const maps = new Map<string, number>();
  let compact = 0;
  for (let i = 0; i < n; i++) {
    const c = ranked1v1Config(random);
    GameConfigSchema.parse(c);
    maps.set(c.gameMap, (maps.get(c.gameMap) ?? 0) + 1);
    const small = c.gameMapSize === GameMapSize.Compact;
    if (small) compact++;
    assert.equal(c.bots, small ? 100 : 400);
    assert.equal(c.maxTimerValue, small ? 10 : 15);
  }
  assert.ok(Math.abs(maps.get(GameMapType.Australia)! / n - 0.4) < 0.04);
  for (const m of [GameMapType.Iceland, GameMapType.Asia, GameMapType.EuropeClassic]) {
    assert.ok(Math.abs(maps.get(m)! / n - 0.2) < 0.03, m);
  }
  assert.ok(Math.abs(compact / n - 0.2) < 0.03);
});

test("the grid fits every map a config can produce, padded to multiples of 16", () => {
  assert.deepEqual(gridFor(ONION), [64, 64]);
  assert.deepEqual(gridFor(RANKED), [192, 256]); // Australia/Iceland 188 rows, all four 250 columns
  assert.equal(slotsFor(ONION), 5);
  assert.equal(slotsFor(RANKED), 403);
});

test("public FFA and Humans vs Nations configs copy the playlist's rules", () => {
  const random = new PseudoRandom(7);
  for (let i = 0; i < 300; i++) {
    const f = publicConfig(FFA, random);
    GameConfigSchema.parse(f);
    assert.equal(f.gameType, GameType.Public);
    assert.equal(f.gameMode, GameMode.FFA);
    assert.equal(f.nations, "default");
    assert.equal(f.maxTimerValue, 30);
    assert.deepEqual(f.overtime, { enabled: true });
    assert.equal(f.bots, f.gameMapSize === GameMapSize.Compact ? 100 : 400);
    assert.equal(f.spawnImmunityDuration, 50);
    assert.equal(f.donateGold || f.donateTroops, false);
    assert.equal(f.disableClanTags, true);
    const h = publicConfig(HVN, random);
    GameConfigSchema.parse(h);
    assert.equal(h.gameMode, GameMode.Team);
    assert.equal(h.playerTeams, HumansVsNations);
    assert.equal(h.difficulty, Difficulty.Hard);
    assert.equal(h.donateGold && h.donateTroops, true);
    assert.equal(h.overtime, undefined);
  }
  assert.equal(publicConfig({ ...FFA, timerMinutes: 0 }, random).maxTimerValue, undefined);
});

test("training maps follow the rotation's weights without the held-out maps, Compact in a third", () => {
  const random = new PseudoRandom(7);
  const n = 6000;
  const draws = Array.from({ length: n }, () => publicConfig(FFA, random));
  const share = (hit: (c: GameConfig) => boolean) => draws.filter(hit).length / n;
  assert.ok(Math.abs(share((c) => c.gameMapSize === GameMapSize.Compact) - 1 / 3) < 0.03);
  assert.ok(Math.abs(share((c) => c.difficulty === Difficulty.Hard) - 0.5) < 0.03);
  const weight = (map: GameMapType) => {
    const info = allMaps.find((m) => m.type === map)!;
    return info.ffaFrequency >= 0 ? info.ffaFrequency : info.multiplayerFrequency;
  };
  const total = [...new Set(publicMaps(false, "training"))].reduce((sum, m) => sum + weight(m), 0);
  assert.ok(Math.abs(share((c) => c.gameMap === GameMapType.World) - weight(GameMapType.World) / total) < 0.02);
  assert.ok(draws.every((c) => !HELD_OUT_MAPS.includes(c.gameMap)));
  const held = Array.from({ length: 2000 }, () => publicConfig({ ...FFA, maps: "heldout" }, random));
  for (const m of HELD_OUT_MAPS) {
    assert.ok(Math.abs(held.filter((c) => c.gameMap === m).length / 2000 - 0.25) < 0.03, m);
  }
});

test("the held-out and seen maps are fixed, fit the ranked grid and stay out of training", () => {
  assert.deepEqual(HELD_OUT_MAPS, [GameMapType.Didier, GameMapType.BeringStrait, GameMapType.YellowSea, GameMapType.Labyrinth]);
  assert.deepEqual(SEEN_MAPS, [GameMapType.BlackSea, GameMapType.Venice, GameMapType.Mars, GameMapType.MoreThanLuck]);
  const ranked = [GameMapType.World, GameMapType.Australia, GameMapType.Iceland, GameMapType.Asia];
  for (const m of [...HELD_OUT_MAPS, ...SEEN_MAPS]) assert.ok(!ranked.includes(m), m);
  assert.ok(HELD_OUT_MAPS.every((m) => !SEEN_MAPS.includes(m)));
  for (const team of [false, true]) {
    const training = new Set(publicMaps(team, "training"));
    assert.equal(training.size, 30);
    assert.ok(SEEN_MAPS.every((m) => training.has(m)));
    assert.ok(HELD_OUT_MAPS.every((m) => !training.has(m)));
    for (const which of ["training", "heldout", "seen"] as const) {
      for (const m of publicMaps(team, which)) {
        const g = mapGrid(m, GameMapSize.Normal);
        assert.ok(g.h <= 192 && g.w <= 256, m);
      }
    }
  }
});

test("a mixed run shares the ranked grid and the largest player table", () => {
  assert.deepEqual(gridFor(FFA), [192, 256]);
  assert.equal(slotsFor(FFA), 1 + 4 + 72 + 400); // World's 72 nations
  assert.equal(slotsFor(HVN), 1 + 4 + 4 + 400);
  assert.equal(slotsFor({ ...FFA, seats: 1, maps: "heldout" }), 1 + 1 + 26 + 400); // Labyrinth
  assert.deepEqual(shapeOf([RANKED, FFA, HVN]), { grid: [192, 256], slots: 477 });
  assert.deepEqual(shapeOf([ONION]), { grid: [64, 64], slots: 5 });
});
