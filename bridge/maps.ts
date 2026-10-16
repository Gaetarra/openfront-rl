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

type MapFiles = { map: Uint8Array; map4x: Uint8Array; map16x: Uint8Array };
const files = new Map<GameMapType, MapFiles>();
const manifests = new Map<GameMapType, string>();

function mapDir(map: GameMapType): string {
  const key = Object.keys(GameMapType).find((k) => GameMapType[k as keyof typeof GameMapType] === map);
  if (key === undefined) throw new Error(`unknown map: ${map}`);
  return path.join(GAME_ROOT, "resources/maps", key.toLowerCase());
}

/**
 * A map's manifest, parsed anew on every call (loadFreshTerrain edits its copy). It reads no map data: the data of all
 * maps together are over 600 MB, and public games look at every manifest to find the maps that fit the grid.
 */
function manifestOf(map: GameMapType): MapManifest {
  let text = manifests.get(map);
  if (text === undefined) {
    text = fs.readFileSync(path.join(mapDir(map), "manifest.json"), "utf8");
    manifests.set(map, text);
  }
  return JSON.parse(text) as MapManifest;
}

function mapFiles(map: GameMapType): MapFiles {
  let f = files.get(map);
  if (f === undefined) {
    const dir = mapDir(map);
    const read = (name: string) => new Uint8Array(fs.readFileSync(path.join(dir, name)));
    f = { map: read("map.bin"), map4x: read("map4x.bin"), map16x: read("map16x.bin") };
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
  const manifest = manifestOf(map);
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
 * A map's cell grid without building the map: Normal maps use 8x8-tile cells on the full-size map, Compact maps 4x4 on
 * the half-size map, so both sizes of a map share one grid.
 */
export function mapGrid(map: GameMapType, size: GameMapSize): { h: number; w: number; cs: number } {
  const manifest = manifestOf(map);
  const normal = size === GameMapSize.Normal;
  const m = normal ? manifest.map : manifest.map4x;
  const cs = normal ? 8 : 4;
  return { h: Math.ceil(m.height / cs), w: Math.ceil(m.width / cs), cs };
}

/** How many nations a map's manifest lists: a Normal public FFA game has them all, a Compact one a quarter. */
export function nationCount(map: GameMapType): number {
  return manifestOf(map).nations.length;
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
