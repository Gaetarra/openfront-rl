import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { NationMIRVBehavior } from "../OpenFront_Source_Code/openfrontio/src/core/execution/nation/NationMIRVBehavior";
import { TradeShipExecution } from "../OpenFront_Source_Code/openfrontio/src/core/execution/TradeShipExecution";
import { TransportShipExecution } from "../OpenFront_Source_Code/openfrontio/src/core/execution/TransportShipExecution";
import {
  GameMapSize,
  GameMapType,
  TeamGameSpawnAreas,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import {
  genTerrainFromBin,
  MapManifest,
  TerrainMapData,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/TerrainMapLoader";

export const GAME_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../OpenFront_Source_Code/openfrontio",
);

type MapFiles = { manifest: string; map: Uint8Array; map4x: Uint8Array; map16x: Uint8Array };
const files = new Map<GameMapType, MapFiles>();

function mapFiles(map: GameMapType): MapFiles {
  let f = files.get(map);
  if (f === undefined) {
    const key = Object.keys(GameMapType).find((k) => GameMapType[k as keyof typeof GameMapType] === map);
    if (key === undefined) throw new Error(`unknown map: ${map}`);
    const dir = path.join(GAME_ROOT, "resources/maps", key.toLowerCase());
    const read = (name: string) => new Uint8Array(fs.readFileSync(path.join(dir, name)));
    f = {
      manifest: fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
      map: read("map.bin"),
      map4x: read("map4x.bin"),
      map16x: read("map16x.bin"),
    };
    files.set(map, f);
  }
  return f;
}

const half = (c: [number, number] | undefined): [number, number] | undefined =>
  c === undefined ? undefined : [Math.floor(c[0] / 2), Math.floor(c[1] / 2)];

/**
 * Same result as the game's loadTerrainMap, but every call returns new map objects: loadTerrainMap caches one mutable
 * map per map and size, so a second game in the same process would start on the first game's territory.
 */
export async function loadFreshTerrain(map: GameMapType, size: GameMapSize): Promise<TerrainMapData> {
  const f = mapFiles(map);
  const manifest = JSON.parse(f.manifest) as MapManifest;
  const normal = size === GameMapSize.Normal;
  // slice(): a map writes to its terrain bytes (water nukes), so every game gets its own copy.
  const gameMap = await genTerrainFromBin(normal ? manifest.map : manifest.map4x, (normal ? f.map : f.map4x).slice());
  const miniGameMap = await genTerrainFromBin(
    normal ? manifest.map4x : manifest.map16x,
    (normal ? f.map4x : f.map16x).slice(),
  );
  let teamGameSpawnAreas = manifest.teamGameSpawnAreas;
  if (!normal) {
    for (const n of manifest.nations) n.coordinates = half(n.coordinates);
    for (const n of manifest.additionalNations ?? []) n.coordinates = half(n.coordinates);
    if (teamGameSpawnAreas !== undefined) {
      const scaled: TeamGameSpawnAreas = {};
      for (const [key, areas] of Object.entries(teamGameSpawnAreas)) {
        scaled[key] = areas.map((a) => ({
          x: Math.floor(a.x / 2),
          y: Math.floor(a.y / 2),
          width: Math.max(1, Math.floor(a.width / 2)),
          height: Math.max(1, Math.floor(a.height / 2)),
        }));
      }
      teamGameSpawnAreas = scaled;
    }
  }
  return {
    nations: manifest.nations,
    additionalNations: manifest.additionalNations ?? [],
    gameMap,
    miniGameMap,
    teamGameSpawnAreas,
  };
}

/**
 * Static fields in the game code that outlive a game. The real client runs one game per worker, so they start fresh;
 * here several games share a process, so they are reset before each game or a game's outcome would depend on the
 * games that ran before it. On every game upgrade, grep src/core for `private static` fields that are not readonly.
 */
export function resetGameStatics(): void {
  (TransportShipExecution as unknown as { _staggerCounter: number })._staggerCounter = 0;
  (TradeShipExecution as unknown as { _staggerCounter: number })._staggerCounter = 0;
  (NationMIRVBehavior as unknown as { recentMirvTargets: Map<string, number> }).recentMirvTargets.clear();
}
