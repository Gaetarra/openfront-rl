import { Game, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { GameMap, TileRef } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameMap";

/** Cell record: u32 cell, u16 owner 1, u16 owner 2, u8 tiles of owner 1, u8 tiles of owner 2, u8 fallout tiles. */
export const CELL_BYTES = 11;
/** Unit record: u8 type index (UNIT_TYPES), u8 level (0 = under construction), u16 owner, u32 cell, f32 troops. */
export const UNIT_BYTES = 12;
/** Unit types sent to the trainer, by index. Shells, SAM missiles and trains are not observed. */
export const UNIT_TYPES: UnitType[] = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
  UnitType.MissileSilo,
  UnitType.SAMLauncher,
  UnitType.DefensePost,
  UnitType.TransportShip,
  UnitType.Warship,
  UnitType.TradeShip,
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
  UnitType.MIRVWarhead,
];
const UNIT_INDEX = new Map(UNIT_TYPES.map((t, i) => [t, i]));

/**
 * The map on a grid of cs x cs tiles: per cell its two largest owners with their tile counts, and the fallout count.
 * Only cells touched by the game's tile updates are recomputed (a full rebuild costs over half of the simulation
 * time, see PLAN.md Phase 0 results).
 */
export class CellFrame {
  readonly cw: number;
  readonly ch: number;
  readonly n: number;
  readonly o1: Uint16Array;
  readonly o2: Uint16Array;
  readonly n1: Uint8Array;
  readonly n2: Uint8Array;
  readonly fo: Uint8Array;
  private readonly w: number;
  private readonly h: number;
  private readonly state: Uint16Array;
  private readonly dirty: Uint8Array;
  private readonly dirtyList: Int32Array;
  private nDirty = 0;
  private readonly cnt = new Uint16Array(4096);
  private readonly touched: number[] = [];
  private readonly out: Uint8Array;
  private readonly dv: DataView;

  constructor(
    map: GameMap,
    readonly cs: number,
  ) {
    this.w = map.width();
    this.h = map.height();
    this.cw = Math.ceil(this.w / cs);
    this.ch = Math.ceil(this.h / cs);
    this.n = this.cw * this.ch;
    this.o1 = new Uint16Array(this.n);
    this.o2 = new Uint16Array(this.n);
    this.n1 = new Uint8Array(this.n);
    this.n2 = new Uint8Array(this.n);
    this.fo = new Uint8Array(this.n);
    this.state = map.tileStateBuffer();
    this.dirty = new Uint8Array(this.n);
    this.dirtyList = new Int32Array(this.n);
    this.out = new Uint8Array(this.n * CELL_BYTES);
    this.dv = new DataView(this.out.buffer);
  }

  cellOf(tile: TileRef): number {
    const x = tile % this.w;
    const y = (tile - x) / this.w;
    return Math.floor(y / this.cs) * this.cw + Math.floor(x / this.cs);
  }

  /** The cell's tiles, row by row. */
  tilesOf(cell: number): TileRef[] {
    const x0 = (cell % this.cw) * this.cs;
    const y0 = Math.floor(cell / this.cw) * this.cs;
    const tiles: TileRef[] = [];
    for (let y = y0; y < Math.min(y0 + this.cs, this.h); y++) {
      for (let x = x0; x < Math.min(x0 + this.cs, this.w); x++) tiles.push(y * this.w + x);
    }
    return tiles;
  }

  /** Tile coordinates of the cell's centre. */
  center(cell: number): [number, number] {
    return [(cell % this.cw) * this.cs + this.cs / 2, Math.floor(cell / this.cw) * this.cs + this.cs / 2];
  }

  /** Marks the cells touched by one tick's packedTileUpdates ([tileRef, state] pairs). */
  mark(packed: Uint32Array): void {
    for (let i = 0; i < packed.length; i += 2) {
      const c = this.cellOf(packed[i]);
      if (this.dirty[c] === 0) {
        this.dirty[c] = 1;
        this.dirtyList[this.nDirty++] = c;
      }
    }
  }

  /**
   * Recomputes the cells marked since the last call (or all cells) and returns CELL_BYTES records: every cell when
   * `all`, otherwise only the cells that changed.
   */
  encode(all: boolean): Uint8Array {
    let off = 0;
    const emit = (c: number) => {
      this.dv.setUint32(off, c, true);
      this.dv.setUint16(off + 4, this.o1[c], true);
      this.dv.setUint16(off + 6, this.o2[c], true);
      this.out[off + 8] = this.n1[c];
      this.out[off + 9] = this.n2[c];
      this.out[off + 10] = this.fo[c];
      off += CELL_BYTES;
    };
    if (all) {
      for (let c = 0; c < this.n; c++) {
        this.compute(c);
        emit(c);
      }
    } else {
      for (let k = 0; k < this.nDirty; k++) {
        const c = this.dirtyList[k];
        if (this.compute(c)) emit(c);
      }
    }
    for (let k = 0; k < this.nDirty; k++) this.dirty[this.dirtyList[k]] = 0;
    this.nDirty = 0;
    return this.out.slice(0, off);
  }

  /** Recomputes one cell from the tile states; returns whether it changed. Ties keep the first owner seen. */
  private compute(c: number): boolean {
    const x0 = (c % this.cw) * this.cs;
    const y0 = Math.floor(c / this.cw) * this.cs;
    const x1 = Math.min(x0 + this.cs, this.w);
    const y1 = Math.min(y0 + this.cs, this.h);
    let fallout = 0;
    for (let y = y0; y < y1; y++) {
      for (let i = y * this.w + x0, end = y * this.w + x1; i < end; i++) {
        const v = this.state[i];
        if (v & 0x2000) fallout++;
        const id = v & 0xfff;
        if (id !== 0 && this.cnt[id]++ === 0) this.touched.push(id);
      }
    }
    let o1 = 0;
    let o2 = 0;
    let n1 = 0;
    let n2 = 0;
    for (const id of this.touched) {
      const k = this.cnt[id];
      if (k > n1) {
        o2 = o1;
        n2 = n1;
        o1 = id;
        n1 = k;
      } else if (k > n2) {
        o2 = id;
        n2 = k;
      }
      this.cnt[id] = 0;
    }
    this.touched.length = 0;
    const changed =
      o1 !== this.o1[c] || o2 !== this.o2[c] || n1 !== this.n1[c] || n2 !== this.n2[c] || fallout !== this.fo[c];
    this.o1[c] = o1;
    this.o2[c] = o2;
    this.n1[c] = n1;
    this.n2[c] = n2;
    this.fo[c] = fallout;
    return changed;
  }
}

/** Static terrain per cell, 4 planes of frame.n bytes: land, shore (land next to water) and ocean tiles, and the mean
 *  height (magnitude 0-31) of the land. */
export function terrainCells(map: GameMap, frame: CellFrame): Uint8Array {
  const out = new Uint8Array(4 * frame.n);
  for (let c = 0; c < frame.n; c++) {
    let land = 0;
    let shore = 0;
    let ocean = 0;
    let height = 0;
    for (const t of frame.tilesOf(c)) {
      if (map.isLand(t) && !map.isImpassable(t)) {
        land++;
        height += map.magnitude(t);
        if (map.isShore(t)) shore++;
      } else if (map.isOcean(t)) {
        ocean++;
      }
    }
    out[c] = land;
    out[frame.n + c] = shore;
    out[2 * frame.n + c] = ocean;
    out[3 * frame.n + c] = land > 0 ? Math.round(height / land) : 0;
  }
  return out;
}

/** All observed units as UNIT_BYTES records. */
export function encodeUnits(game: Game, frame: CellFrame): Uint8Array {
  const units = game.units();
  const out = new Uint8Array(units.length * UNIT_BYTES);
  const dv = new DataView(out.buffer);
  let off = 0;
  for (const u of units) {
    const type = UNIT_INDEX.get(u.type());
    if (type === undefined) continue;
    out[off] = type;
    out[off + 1] = u.isUnderConstruction() ? 0 : Math.min(u.level(), 255);
    dv.setUint16(off + 2, u.owner().smallID(), true);
    dv.setUint32(off + 4, frame.cellOf(u.tile()), true);
    dv.setFloat32(off + 8, u.troops(), true);
    off += UNIT_BYTES;
  }
  return out.subarray(0, off);
}
