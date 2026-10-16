import { Config } from "../OpenFront_Source_Code/openfrontio/src/core/configuration/Config";
import { Executor } from "../OpenFront_Source_Code/openfrontio/src/core/execution/ExecutionManager";
import {
  Game,
  GameMapSize,
  Player,
  PlayerInfo,
  PlayerType,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { createGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameImpl";
import { GameUpdateType, HashUpdate } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameUpdates";
import { createNationsForGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/NationCreation";
import { GameRunner } from "../OpenFront_Source_Code/openfrontio/src/core/GameRunner";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import {
  GameRecord,
  GameStartInfo,
  PlayerRecord,
  StampedIntent,
  Turn,
  Winner,
} from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { createPartialGameRecord, simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { Act, attackMask, decodeAction, N_ACTS, retreatMask, typeMask } from "./actions";
import { CellFrame, encodeUnits, terrainCells } from "./frame";
import { LatencyQueue } from "./latency";
import { loadFreshTerrain, resetGameStatics } from "./maps";
import { encodeSeat, PLAYER_FEATURES, SCALARS } from "./obs";
import { rewardOf, seatStanding } from "./ranking";
import { gameConfigOf, gameIdOf, shapeOf } from "./setups";

/** A training game setup, sent by the trainer as JSON. FFA only in Phase 1. */
export interface EnvConfig {
  map: string; // a GameMapType value, e.g. "Onion"
  compact: boolean;
  nations: number; // 0 = no nations
  /** A Difficulty value, e.g. "Easy". "ffa" and "hvn" take a comma-separated list and draw one value per game. */
  difficulty: string;
  bots: number;
  /** In minutes. "ffa" and "hvn": 0 means no timer, as in the public games. */
  timerMinutes: number;
  seats: number; // agent-controlled players
  /**
   * Draw each game's map, size and bots from this playlist (bridge/setups.ts). "ranked1v1" also sets the timer and
   * reads only `seats`; "ffa" and "hvn" (public FFA and Humans vs Nations, against the game's own nations) read `seats`,
   * `maps`, `difficulty` and `timerMinutes`.
   */
  playlist?: "ranked1v1" | "ffa" | "hvn";
  /** "ffa" and "hvn": which maps to draw from (bridge/setups.ts publicMaps). Default "training". */
  maps?: "training" | "heldout" | "seen";
}

/** Ticks per agent decision (PLAN decision 11). */
export const STEP_TICKS = 5;

export const seatClientID = (seat: number): string => `AGENT${seat.toString().padStart(3, "0")}`;

/** The players of a training game, one per seat, named as EnvGame.create names them. */
export const seatPlayers = (seats: number) =>
  Array.from({ length: seats }, (_, i) => ({ clientID: seatClientID(i), username: `agent${i}`, clanTag: null }));

export type Blob = [name: string, bytes: Uint8Array];
const bytes = (a: Float32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

export class EnvGame {
  done = false;
  rewards: number[] = [];
  ranks: number[] = [];
  /** Non-bot players in the game: the n of the reward formula. */
  nPlayers = 0;
  /** Per seat: its last action could not be applied and was dropped. */
  invalid: boolean[];
  private turn = 0;
  private masks: { types: Uint8Array; attack: Uint8Array; retreat: Uint8Array }[] | null = null;
  /** Executed turns while the game is recorded (record()). */
  private turns: Turn[] | null = null;

  private constructor(
    /** How the game was created: a record of it replays from this. */
    readonly start: GameStartInfo,
    readonly game: Game,
    readonly seats: Player[],
    readonly frame: CellFrame,
    /** Rows of the player table: smallID 0 (unowned land) up to the config's largest player count. */
    readonly slots: number,
    private readonly runner: GameRunner,
    private readonly sim: { hash: number; hashTick: number; fatal: string | null },
    private readonly latency: LatencyQueue,
  ) {
    this.invalid = seats.map(() => false);
  }

  /** Latest state hash reported by the game (every 10 ticks). */
  get hash(): number {
    return this.sim.hash;
  }

  /** Tick of the latest hash. */
  get hashTick(): number {
    return this.sim.hashTick;
  }

  get timerMinutes(): number {
    return this.start.config.maxTimerValue ?? 60;
  }

  /**
   * A training game: its game ID and every random draw derive from the seed. `shape`: the grid and player table the
   * game is built on, which the games of one mixed run share (default: the config's own).
   */
  static async create(cfg: EnvConfig, seed: string, shape = shapeOf([cfg])): Promise<EnvGame> {
    const config = gameConfigOf(cfg, new PseudoRandom(simpleHash(`${seed}:setup`)));
    const start: GameStartInfo = { gameID: gameIdOf(seed), lobbyCreatedAt: 0, config, players: seatPlayers(cfg.seats) };
    return EnvGame.fromStart(start, shape.grid[1], shape.slots, `${seed}:latency`);
  }

  /**
   * Builds a game from its start info the way the game's client does (tests/replay/ReplayGame.ts): the RNG is seeded
   * from the game ID and the seats are the start info's players, in order. `stride`: the trainer's grid width in cells.
   */
  static async fromStart(start: GameStartInfo, stride: number, slots: number, latencySeed: string): Promise<EnvGame> {
    resetGameStatics();
    const config = start.config;
    const terrain = await loadFreshTerrain(config.gameMap, config.gameMapSize);
    const random = new PseudoRandom(simpleHash(start.gameID));
    const humans = start.players.map(
      (p) =>
        new PlayerInfo(
          p.username,
          PlayerType.Human,
          p.clientID,
          random.nextID(),
          p.isLobbyCreator ?? false,
          p.clanTag,
          p.friends ?? [],
          p.teamIndex ?? null,
        ),
    );
    const nations = createNationsForGame(start, terrain.nations, terrain.additionalNations, humans.length, random);
    const game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      new Config(config, null, false),
      terrain.teamGameSpawnAreas,
    );
    const frame = new CellFrame(game.map(), config.gameMapSize === GameMapSize.Compact ? 4 : 8, stride);
    const sim = { hash: 0, hashTick: -1, fatal: null as string | null };
    const executor = new Executor(game, start.gameID, undefined, start.tribes?.map((t) => t.name));
    const runner = new GameRunner(game, executor, (gu) => {
      if ("errMsg" in gu) {
        sim.fatal = gu.errMsg;
        return;
      }
      frame.mark(gu.packedTileUpdates);
      for (const h of gu.updates[GameUpdateType.Hash] as HashUpdate[]) {
        sim.hash = h.hash;
        sim.hashTick = h.tick;
      }
    });
    runner.init();
    const seats = humans.map((h) => game.player(h.id));
    const latency = new LatencyQueue(new PseudoRandom(simpleHash(latencySeed)));
    return new EnvGame(start, game, seats, frame, slots, runner, sim, latency);
  }

  /** Records every executed turn from now on (call it before the first step), for toRecord(). */
  record(): void {
    this.turns = [];
  }

  get recording(): boolean {
    return this.turns !== null;
  }

  /** The recorded game as a game record (GameRecordSchema), which the game's own client can replay. */
  toRecord(gitCommit: string): GameRecord {
    if (this.turns === null) throw new Error("this game was not recorded");
    const w = this.game.getWinner();
    const winner: Winner =
      w === null
        ? undefined
        : typeof w === "string"
          ? ["team", w]
          : w.clientID() !== null
            ? ["player", w.clientID()!]
            : ["nation", w.name()];
    const players: PlayerRecord[] = this.start.players.map((p) => ({ ...p, persistentID: null, stats: {} }));
    const end = this.game.ticks() * 100; // ms of game time: 10 ticks a second
    const partial = createPartialGameRecord(this.start.gameID, this.start.config, players, this.turns, 0, end, winner);
    return { ...partial, gitCommit } as GameRecord;
  }

  /** Executes one turn (one tick) with the given intents. */
  tickWith(intents: StampedIntent[]): void {
    const turnNumber = this.turn++;
    this.runner.addTurn({ turnNumber, intents });
    if (!this.runner.executeNextTick()) throw new Error(`tick ${turnNumber} failed: ${this.sim.fatal}`);
    if (this.turns !== null) {
      this.turns.push(this.sim.hashTick === turnNumber ? { turnNumber, intents, hash: this.sim.hash } : { turnNumber, intents });
    }
  }

  /**
   * Applies one decision per seat through the latency queue and advances STEP_TICKS. A decision is one action
   * [type, cell, target, share], or several back to back (the bandwidth probe, PLAN 3B: a troop and a gold action),
   * where a later no-op means no further action; a seat's decision is invalid when any of its actions is.
   */
  step(actions: number[][]): void {
    if (this.done) throw new Error("step() after the game ended");
    const masks = this.masks ?? this.currentMasks();
    this.masks = null;
    this.invalid = this.seats.map((seat, i) => {
      const flat = actions[i] ?? [];
      let invalid = false;
      for (let k = 0; k === 0 || k < flat.length; k += 4) {
        if (k > 0 && flat[k] === Act.Noop) continue;
        const { intent, valid } = decodeAction(
          this.game,
          seat,
          this.frame,
          masks[i].types,
          masks[i].attack,
          flat.slice(k, k + 4),
          masks[i].retreat,
        );
        if (intent !== null) this.latency.push({ ...intent, clientID: seatClientID(i) } as StampedIntent, this.turn);
        invalid ||= !valid;
      }
      return invalid;
    });
    for (let k = 0; k < STEP_TICKS; k++) this.tickWith(this.latency.take(this.turn));
    this.checkDone();
  }

  /** The observation, as blobs in protocol order. `all`: every cell instead of only the changed ones. */
  observe(all: boolean): Blob[] {
    const S = this.seats.length;
    const P = this.slots;
    const players = new Float32Array(S * P * PLAYER_FEATURES);
    const rel = new Uint8Array(S * P);
    const scalars = new Float32Array(S * SCALARS);
    const types = new Uint8Array(S * N_ACTS);
    const attack = new Uint8Array(S * P);
    const retreat = new Uint8Array(S * P);
    this.masks = this.currentMasks();
    this.seats.forEach((seat, i) => {
      const m = this.masks![i];
      types.set(m.types, i * N_ACTS);
      attack.set(m.attack, i * P);
      retreat.set(m.retreat, i * P);
      encodeSeat(
        this.game,
        seat,
        this.timerMinutes,
        this.frame.cs,
        m.attack,
        players.subarray(i * P * PLAYER_FEATURES, (i + 1) * P * PLAYER_FEATURES),
        rel.subarray(i * P, (i + 1) * P),
        scalars.subarray(i * SCALARS, (i + 1) * SCALARS),
      );
    });
    return [
      ["cells", this.frame.encode(all)],
      ["units", encodeUnits(this.game, this.frame)],
      ["players", bytes(players)],
      ["rel", rel],
      ["scalars", bytes(scalars)],
      ["types", types],
      ["attack", attack],
      ["retreat", retreat],
    ];
  }

  terrain(): Uint8Array {
    return terrainCells(this.game.map(), this.frame);
  }

  private currentMasks(): { types: Uint8Array; attack: Uint8Array; retreat: Uint8Array }[] {
    return this.seats.map((seat) => {
      const attack = attackMask(this.game, seat, this.slots);
      const retreat = retreatMask(seat, this.slots);
      return { types: typeMask(this.game, seat, attack, retreat), attack, retreat };
    });
  }

  /** The game ends at a winner, when every seat is dead or never spawned, or 1 min after the timer as a backstop. */
  private checkDone(): void {
    const g = this.game;
    const out = (s: Player) => !g.inSpawnPhase() && (!s.hasSpawned() || !s.isAlive());
    const backstop = g.ticks() >= g.config().numSpawnPhaseTurns() + (this.timerMinutes + 1) * 600;
    if (g.getWinner() === null && !this.seats.every(out) && !backstop) return;
    this.done = true;
    const standings = this.seats.map((s) => seatStanding(g, s));
    this.nPlayers = standings[0].n;
    this.ranks = standings.map((s) => s.rank);
    this.rewards = standings.map((s) => rewardOf(s.rank, s.n));
  }
}
