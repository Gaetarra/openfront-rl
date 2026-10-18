import assert from "node:assert/strict";
import { test } from "node:test";
import { GameMapSize } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { GameMapType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/Maps.gen";
import { loadFreshTerrain } from "./maps";

test("each call returns new map objects with compact coordinates halved once", async () => {
  const a = await loadFreshTerrain(GameMapType.Onion, GameMapSize.Compact);
  const b = await loadFreshTerrain(GameMapType.Onion, GameMapSize.Compact);
  assert.notEqual(a.gameMap, b.gameMap);
  assert.notEqual(a.gameMap.tileStateBuffer(), b.gameMap.tileStateBuffer());
  assert.equal(a.gameMap.width(), 256);
  assert.equal(a.miniGameMap.width(), 128);
  assert.deepEqual(a.nations.map((n) => n.coordinates), b.nations.map((n) => n.coordinates));
  const normal = await loadFreshTerrain(GameMapType.Onion, GameMapSize.Normal);
  const [x, y] = normal.nations[0].coordinates!;
  assert.deepEqual(a.nations[0].coordinates, [Math.floor(x / 2), Math.floor(y / 2)]);
});
