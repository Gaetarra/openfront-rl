import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { GameMapSize } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { MapManifest } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/MapFiles";
import { GameMapType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/Maps.gen";
import {
  loadTerrainMap,
  TerrainMapData,
} from "../OpenFront_Source_Code/openfrontio/packages/engine-lib/src/game/TerrainMapLoader";

export const GAME_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../OpenFront_Source_Code/openfrontio",
);

type MapBins = { mapBin: Uint8Array; map4xBin: Uint8Array; map16xBin: Uint8Array };
const bins = new Map<GameMapType, MapBins>();
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

function mapBins(map: GameMapType): MapBins {
  let b = bins.get(map);
  if (b === undefined) {
    const dir = mapDir(map);
    const read = (name: string) => new Uint8Array(fs.readFileSync(path.join(dir, name)));
    b = { mapBin: read("map.bin"), map4xBin: read("map4x.bin"), map16xBin: read("map16x.bin") };
    bins.set(map, b);
  }
  return b;
}

/**
 * The game's loadTerrainMap on fresh copies of the map's files, read from disk once per process. It takes the files
 * over: a map writes to its terrain bytes (water nukes), and compact maps scale the manifest's coordinates in place.
 */
export async function loadFreshTerrain(map: GameMapType, mapSize: GameMapSize): Promise<TerrainMapData> {
  const b = mapBins(map);
  return loadTerrainMap({
    map,
    mapSize,
    manifest: manifestOf(map),
    mapBin: b.mapBin.slice(),
    map4xBin: b.map4xBin.slice(),
    map16xBin: b.map16xBin.slice(),
  });
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
