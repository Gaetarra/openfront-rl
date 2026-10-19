import { TileRef } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameMap";
import { TerraNullius, UnitType } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { Intent } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { Attack, Game, Player, Unit } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/Game";
import { canBuildTransportShip } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/TransportShipUtils";
import { CellFrame } from "./frame";

/** Action types (PLAN 3B "Actions"): Phase 1's five, then every other command ranked 1v1 players give, then step 6's
 * diplomacy and warship moves. Outside the set: gold donations, rejects, embargoes, target marks, chat and emoji, MIRVs,
 * cancelling boats and deleting units. */
export enum Act {
  Noop = 0,
  Spawn = 1,
  Attack = 2,
  Boat = 3,
  BuildCity = 4,
  BuildDefensePost = 5,
  BuildSilo = 6,
  BuildFactory = 7,
  BuildPort = 8,
  BuildSAM = 9,
  BuildWarship = 10,
  AtomBomb = 11,
  HydrogenBomb = 12,
  Upgrade = 13,
  Retreat = 14,
  AllianceRequest = 15, // also accepts a request the target sent
  RenewAlliance = 16,
  BreakAlliance = 17,
  DonateTroops = 18,
  MoveWarships = 19,
}
export const N_ACTS = 20;
/** The types aimed at a player (a row of the player table), in the order of Masks.diplomacy's rows. */
export const DIPLOMACY: readonly Act[] = [Act.AllianceRequest, Act.RenewAlliance, Act.BreakAlliance, Act.DonateTroops];
/** The unit each build action places: structures on own land, a port on own shore, a warship on water, a nuke on any
 * tile the game lets it hit. */
export const BUILDS: ReadonlyMap<Act, UnitType> = new Map([
  [Act.BuildCity, UnitType.City],
  [Act.BuildDefensePost, UnitType.DefensePost],
  [Act.BuildSilo, UnitType.MissileSilo],
  [Act.BuildFactory, UnitType.Factory],
  [Act.BuildPort, UnitType.Port],
  [Act.BuildSAM, UnitType.SAMLauncher],
  [Act.BuildWarship, UnitType.Warship],
  [Act.AtomBomb, UnitType.AtomBomb],
  [Act.HydrogenBomb, UnitType.HydrogenBomb],
]);
/** Attacks and boats send a whole percent of the seat's current troops, 1-100 (PLAN 3B: troop shares). */
const isShare = (s: number) => Number.isInteger(s) && s >= 1 && s <= 100;
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

/** Which table rows the seat is attacking over land now, for a retreat: row = smallID, row 0 = unowned land. */
export function retreatMask(seat: Player, slots: number): Uint8Array {
  const mask = new Uint8Array(slots);
  for (const a of seat.outgoingAttacks()) {
    const row = rowOf(a.target());
    if (a.isActive() && !a.retreating() && row < slots) mask[row] = 1;
  }
  return mask;
}

/** Which table rows each diplomatic type can target now: DIPLOMACY.length rows of `slots` bytes, by the game's own checks. */
export function diplomacyMask(game: Game, seat: Player, slots: number): Uint8Array {
  const mask = new Uint8Array(DIPLOMACY.length * slots);
  if (game.inSpawnPhase() || !seat.isAlive()) return mask;
  for (const p of game.allPlayers()) {
    const s = p.smallID();
    if (p === seat || s >= slots) continue;
    mask[s] = p.isAlive() && seat.canSendAllianceRequest(p) ? 1 : 0;
    mask[slots + s] = seat.allianceInfo(p)?.canExtend === true ? 1 : 0;
    mask[2 * slots + s] = seat.isAlliedWith(p) ? 1 : 0;
    mask[3 * slots + s] =
      seat.canDonateTroops(p) && game.config().maxTroops(p) - p.troops() >= 1 && seat.troops() >= 1 ? 1 : 0;
  }
  return mask;
}

/** What a seat's decision is masked by: the legal types, and the targets of attacks, retreats and diplomacy. */
export interface Masks {
  types: Uint8Array;
  attack: Uint8Array;
  retreat: Uint8Array;
  diplomacy: Uint8Array;
}

export function seatMasks(game: Game, seat: Player, slots: number): Masks {
  const attack = attackMask(game, seat, slots);
  const retreat = retreatMask(seat, slots);
  const diplomacy = diplomacyMask(game, seat, slots);
  return { types: typeMask(game, seat, attack, retreat, diplomacy), attack, retreat, diplomacy };
}

/** Which action types are legal now (1 = allowed). `retreat`: retreatMask, or nothing to cancel. `diplomacy`: diplomacyMask. */
export function typeMask(
  game: Game,
  seat: Player,
  attack: Uint8Array,
  retreat: Uint8Array = new Uint8Array(0),
  diplomacy: Uint8Array = new Uint8Array(0),
): Uint8Array {
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
  for (const [act, unit] of BUILDS) {
    m[act] = !game.config().isUnitDisabled(unit) && seat.gold() >= game.unitInfo(unit).cost(game, seat) ? 1 : 0;
  }
  if (!seat.units(UnitType.Port).some(finished)) m[Act.BuildWarship] = 0;
  if (game.isSpawnImmunityActive() || !seat.units(UnitType.MissileSilo).some((s) => finished(s) && !s.isInCooldown())) {
    m[Act.AtomBomb] = m[Act.HydrogenBomb] = 0;
  }
  m[Act.Upgrade] = seat.units().some((u) => u.isActive() && seat.canUpgradeUnit(u)) ? 1 : 0;
  m[Act.Retreat] = retreat.includes(1) ? 1 : 0;
  const slots = diplomacy.length / DIPLOMACY.length;
  DIPLOMACY.forEach((act, k) => (m[act] = diplomacy.subarray(k * slots, (k + 1) * slots).includes(1) ? 1 : 0));
  m[Act.MoveWarships] = seat.units(UnitType.Warship).some((u) => u.isActive()) ? 1 : 0;
  return m;
}

const finished = (u: Unit) => u.isActive() && !u.isUnderConstruction();
const rowOf = (target: Player | TerraNullius) => (target.isPlayer() ? target.smallID() : 0);

export interface Decoded {
  intent: Intent | null;
  /** False when the action could not be applied and was dropped (the trainer's cell masks are approximate). */
  valid: boolean;
}
const DROP: Decoded = { intent: null, valid: false };

/** Turns a sampled [type, cell, target, share] into a game intent (null intent = no-op). share: the percent of its
 * current troops an attack, boat or donation sends. `masks`: the seat's masks when it chose. */
export function decodeAction(game: Game, seat: Player, frame: CellFrame, masks: Masks, action: number[]): Decoded {
  const { types, attack, retreat, diplomacy } = masks;
  const [type, cell, target, share] = action;
  if (!Number.isInteger(type) || types[type] !== 1) return DROP;
  if (type === Act.Noop) return { intent: null, valid: true };
  if (type === Act.Attack) {
    if (!Number.isInteger(target) || attack[target] !== 1 || !isShare(share)) return DROP;
    const targetID = target === 0 ? null : (game.playerBySmallID(target) as Player).id();
    return { intent: { type: "attack", targetID, troops: (share / 100) * seat.troops() }, valid: true };
  }
  if (type === Act.Retreat) {
    if (!Number.isInteger(target) || retreat[target] !== 1) return DROP;
    const largest = seat
      .outgoingAttacks()
      .filter((a) => a.isActive() && !a.retreating() && rowOf(a.target()) === target)
      .reduce<Attack | null>((best, a) => (best === null || a.troops() > best.troops() ? a : best), null);
    return largest === null ? DROP : { intent: { type: "cancel_attack", attackID: largest.id() }, valid: true };
  }
  const k = DIPLOMACY.indexOf(type);
  if (k >= 0) {
    const slots = diplomacy.length / DIPLOMACY.length;
    if (!Number.isInteger(target) || target < 0 || target >= slots || diplomacy[k * slots + target] !== 1) return DROP;
    const recipient = (game.playerBySmallID(target) as Player).id();
    if (type === Act.AllianceRequest) return { intent: { type: "allianceRequest", recipient }, valid: true };
    if (type === Act.RenewAlliance) return { intent: { type: "allianceExtension", recipient }, valid: true };
    if (type === Act.BreakAlliance) return { intent: { type: "breakAlliance", recipient }, valid: true };
    if (!isShare(share)) return DROP;
    return { intent: { type: "donate_troops", recipient, troops: (share / 100) * seat.troops() }, valid: true };
  }
  if (!Number.isInteger(cell) || cell < 0 || cell >= frame.n) return DROP;
  if (type === Act.MoveWarships) {
    // [type, source cell, destination cell, fleet]: the anchor is the seat's warship in the source cell nearest its centre
    if (!Number.isInteger(target) || target < 0 || target >= frame.n || (share !== 0 && share !== 1)) return DROP;
    const [cx, cy] = frame.center(cell);
    const d = (u: Unit) => (game.x(u.tile()) - cx) ** 2 + (game.y(u.tile()) - cy) ** 2;
    const ships = seat.units(UnitType.Warship).filter((u) => u.isActive());
    const anchor = ships
      .filter((u) => frame.cellOf(u.tile()) === cell)
      .reduce<Unit | null>((best, u) => (best === null || d(u) < d(best) ? u : best), null);
    const sea = anchor === null ? null : game.getWaterComponent(anchor.tile());
    if (anchor === null || sea === null) return DROP;
    const tile = byCentre(game, frame, target).find((t) => game.isWater(t) && game.hasWaterComponent(t, sea));
    if (tile === undefined) return DROP;
    const patrol = anchor.warshipState().patrolTile;
    const unitIds = (share === 0 ? [anchor] : ships.filter((u) => u.warshipState().patrolTile === patrol && game.hasWaterComponent(u.tile(), sea)))
      .map((u) => u.id());
    return { intent: { type: "move_warship", unitIds, tile }, valid: true };
  }
  if (type === Act.Spawn) {
    const tile = byCentre(game, frame, cell).find((t) => game.isLand(t) && !game.isImpassable(t) && !game.hasOwner(t));
    return tile === undefined ? DROP : { intent: { type: "spawn", tile }, valid: true };
  }
  if (type === Act.Boat) {
    if (!isShare(share)) return DROP;
    for (const tile of frame.tilesOf(cell)) {
      if (!game.isLand(tile) || game.isImpassable(tile) || !game.isShore(tile)) continue;
      if (game.ownerID(tile) === seat.smallID()) continue;
      if (canBuildTransportShip(game, seat, tile) !== false) {
        return { intent: { type: "boat", dst: tile, troops: (share / 100) * seat.troops() }, valid: true };
      }
    }
    return DROP;
  }
  if (type === Act.Upgrade) {
    // A cell holds at most one structure (the 15-tile spacing), so the seat's upgradable one in it is the choice.
    const unit = seat.units().find((u) => frame.cellOf(u.tile()) === cell && seat.canUpgradeUnit(u));
    if (unit === undefined) return DROP;
    return { intent: { type: "upgrade_structure", unit: unit.type(), unitId: unit.id() }, valid: true };
  }
  const unit = BUILDS.get(type);
  if (unit === undefined) return DROP;
  const tile = byCentre(game, frame, cell).find((t) => seat.canBuild(unit, t) !== false);
  return tile === undefined ? DROP : { intent: { type: "build_unit", unit, tile }, valid: true };
}

/** The cell's tiles, nearest its centre first (a stable sort: ties keep the frame's order). */
function byCentre(game: Game, frame: CellFrame, cell: number): TileRef[] {
  const [cx, cy] = frame.center(cell);
  const d = (t: TileRef) => (game.x(t) - cx) ** 2 + (game.y(t) - cy) ** 2;
  return [...frame.tilesOf(cell)].sort((a, b) => d(a) - d(b));
}
