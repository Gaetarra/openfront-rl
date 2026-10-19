/**
 * The map table behaviour cloning filters records with (trainer/env.py map_table): the specialists' cell grid and, per
 * map, its cell grid at Normal size and its nation count. Prints one JSON line on stdout; the game's logs go to stderr.
 * Run from the game dir (its tsconfig paths are needed): node --import tsx ../../bridge/maptable.ts
 */
import { GameMapSize } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { maps } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/Maps.gen";
import { mapGrid, nationCount } from "./maps";
import { shapeOf } from "./setups";
import { FFA, HVN, RANKED } from "./testkit";

console.log = console.info = console.warn = console.debug = console.error;

const table: Record<string, { h: number; w: number; nations: number }> = {};
for (const { type } of maps) {
  const { h, w } = mapGrid(type, GameMapSize.Normal);
  table[type] = { h, w, nations: nationCount(type) };
}
process.stdout.write(JSON.stringify({ grid: shapeOf([RANKED, FFA, HVN]).grid, maps: table }) + "\n");
