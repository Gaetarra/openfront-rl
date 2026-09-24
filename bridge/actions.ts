import { Game, Player, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { TileRef } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameMap";
import { canBuildTransportShip } from "../OpenFront_Source_Code/openfrontio/src/core/game/TransportShipUtils";
import { Intent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { CellFrame } from "./frame";

/** Phase 1 action types (PLAN "Phase 1": no-op, spawn, attack, boat, build city). */
export enum Act {
  Noop = 0,
  Spawn = 1,
  Attack = 2,
  Boat = 3,
  BuildCity = 4,
}
export const N_ACTS = 5;
/** Troops sent by attack and boat actions, as fractions of the seat's current troops. */
export const TROOP_BINS = [0.1, 0.25, 0.5, 0.75, 1.0];
/** A spawn intent must still execute inside the spawn phase after up to 3 turns of latency. */
const SPAWN_MARGIN = 4;
/** Decisions in the last ticks of the spawn phase must spawn if the seat has not spawned yet. */
const FORCE_SPAWN = 15;

/** Which table rows the seat can attack over land now: row = smallID, row 0 = unowned land. */
export function attackMask(game: Game, seat: Player, slots: number): Uint8Array {
  const mask = new Uint8Array(slots);
  if (game.inSpawnPhase() || !seat.isAlive()) return mask;
  const map = game.map();
  const me = seat.smallID();
  const nbuf = [0, 0, 0, 0];
  for (const border of seat.borderTiles()) {
    const k = map.neighbors4(border, nbuf);
    for (let i = 0; i < k; i++) {
      const t = nbuf[i];
      if (!map.isLand(t) || map.isImpassable(t)) continue;
      const owner = map.ownerID(t);
      if (owner === me || owner >= slots || mask[owner] === 1) continue;
      if (owner === 0) {
        if (!map.hasFallout(t)) mask[0] = 1;
        continue;
      }
      const other = game.playerBySmallID(owner) as Player;
      if (!seat.isFriendly(other) && seat.canAttackPlayer(other)) mask[owner] = 1;
    }
  }
  return mask;
}

/** Which action types are legal now (1 = allowed). */
export function typeMask(game: Game, seat: Player, attack: Uint8Array): Uint8Array {
  const m = new Uint8Array(N_ACTS);
  if (game.inSpawnPhase()) {
    const end = game.config().numSpawnPhaseTurns();
    const open = game.ticks() <= end - SPAWN_MARGIN;
    const forced = open && !seat.hasSpawned() && game.ticks() >= end - FORCE_SPAWN;
    m[Act.Noop] = forced ? 0 : 1;
    m[Act.Spawn] = open ? 1 : 0;
    return m;
  }
  m[Act.Noop] = 1;
  if (!seat.isAlive()) return m;
  m[Act.Attack] = attack.includes(1) ? 1 : 0;
  m[Act.Boat] = seat.unitCount(UnitType.TransportShip) < game.config().boatMaxNumber() ? 1 : 0;
  m[Act.BuildCity] = seat.gold() >= game.unitInfo(UnitType.City).cost(game, seat) ? 1 : 0;
  return m;
}

export interface Decoded {
  intent: Intent | null;
  /** False when the action could not be applied and was dropped (the trainer's cell masks are approximate). */
  valid: boolean;
}
const DROP: Decoded = { intent: null, valid: false };

/** Turns a sampled [type, cell, target, bin] into a game intent (null intent = no-op). */
export function decodeAction(
  game: Game,
  seat: Player,
  frame: CellFrame,
  types: Uint8Array,
  attack: Uint8Array,
  action: number[],
): Decoded {
  const [type, cell, target, bin] = action;
  if (!Number.isInteger(type) || types[type] !== 1) return DROP;
  if (type === Act.Noop) return { intent: null, valid: true };
  if (type === Act.Attack) {
    if (!Number.isInteger(target) || attack[target] !== 1 || !(bin in TROOP_BINS)) return DROP;
    const targetID = target === 0 ? null : (game.playerBySmallID(target) as Player).id();
    return { intent: { type: "attack", targetID, troops: TROOP_BINS[bin] * seat.troops() }, valid: true };
  }
  if (!Number.isInteger(cell) || cell < 0 || cell >= frame.n) return DROP;
  if (type === Act.Spawn) {
    const tile = spawnTile(game, frame, cell);
    return tile === null ? DROP : { intent: { type: "spawn", tile }, valid: true };
  }
  if (type === Act.Boat) {
    if (!(bin in TROOP_BINS)) return DROP;
    for (const tile of frame.tilesOf(cell)) {
      if (!game.isLand(tile) || game.isImpassable(tile) || !game.isShore(tile)) continue;
      if (game.ownerID(tile) === seat.smallID()) continue;
      if (canBuildTransportShip(game, seat, tile) !== false) {
        return { intent: { type: "boat", dst: tile, troops: TROOP_BINS[bin] * seat.troops() }, valid: true };
      }
    }
    return DROP;
  }
  if (type === Act.BuildCity) {
    for (const tile of frame.tilesOf(cell)) {
      if (game.ownerID(tile) === seat.smallID() && seat.canBuild(UnitType.City, tile) !== false) {
        return { intent: { type: "build_unit", unit: UnitType.City, tile }, valid: true };
      }
    }
  }
  return DROP;
}

/** The unowned land tile of the cell closest to its centre. */
function spawnTile(game: Game, frame: CellFrame, cell: number): TileRef | null {
  const [cx, cy] = frame.center(cell);
  let best: TileRef | null = null;
  let bestDist = Infinity;
  for (const t of frame.tilesOf(cell)) {
    if (!game.isLand(t) || game.isImpassable(t) || game.hasOwner(t)) continue;
    const dx = game.x(t) - cx;
    const dy = game.y(t) - cy;
    if (dx * dx + dy * dy < bestDist) {
      bestDist = dx * dx + dy * dy;
      best = t;
    }
  }
  return best;
}
