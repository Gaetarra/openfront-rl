import { Config } from "../OpenFront_Source_Code/openfrontio/src/core/configuration/Config";
import { Executor } from "../OpenFront_Source_Code/openfrontio/src/core/execution/ExecutionManager";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerType,
} from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { createGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameImpl";
import { GameUpdateType, HashUpdate } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameUpdates";
import { createNationsForGame } from "../OpenFront_Source_Code/openfrontio/src/core/game/NationCreation";
import { GameRunner } from "../OpenFront_Source_Code/openfrontio/src/core/GameRunner";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { GameConfig, GameStartInfo, StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";
import { simpleHash } from "../OpenFront_Source_Code/openfrontio/src/core/Util";
import { CellFrame } from "./frame";
import { loadFreshTerrain, resetGameStatics } from "./maps";

/** A training game setup, sent by the trainer as JSON. FFA only in Phase 1. */
export interface EnvConfig {
  map: string; // a GameMapType value, e.g. "Onion"
  compact: boolean;
  nations: number; // 0 = no nations
  difficulty: string; // a Difficulty value, e.g. "Easy"
  bots: number;
  timerMinutes: number;
  seats: number; // agent-controlled players
}

/** Ticks per agent decision (PLAN decision 11). */
export const STEP_TICKS = 5;

export const seatClientID = (seat: number): string => `AGENT${seat.toString().padStart(3, "0")}`;

export class EnvGame {
  private turn = 0;

  private constructor(
    readonly cfg: EnvConfig,
    readonly game: Game,
    readonly seats: Player[],
    readonly frame: CellFrame,
    /** Rows of the player table: smallID 0 (unowned land) up to seats + nations + bots. */
    readonly slots: number,
    private readonly runner: GameRunner,
    private readonly sim: { hash: number; fatal: string | null },
  ) {}

  /** Latest state hash reported by the game (every 10 ticks). */
  get hash(): number {
    return this.sim.hash;
  }

  static async create(cfg: EnvConfig, seed: string): Promise<EnvGame> {
    resetGameStatics();
    const config: GameConfig = {
      gameMap: cfg.map as GameMapType,
      gameMapSize: cfg.compact ? GameMapSize.Compact : GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Private,
      difficulty: cfg.difficulty as Difficulty,
      nations: cfg.nations > 0 ? cfg.nations : "disabled",
      bots: cfg.bots,
      donateGold: false,
      donateTroops: false,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
      maxTimerValue: cfg.timerMinutes,
    };
    const terrain = await loadFreshTerrain(config.gameMap, config.gameMapSize);
    const gameStart: GameStartInfo = { gameID: seed, lobbyCreatedAt: 0, config, players: [] };
    const random = new PseudoRandom(simpleHash(seed));
    const humans = Array.from(
      { length: cfg.seats },
      (_, i) => new PlayerInfo(`agent${i}`, PlayerType.Human, seatClientID(i), random.nextID()),
    );
    const nations = createNationsForGame(gameStart, terrain.nations, terrain.additionalNations, humans.length, random);
    const game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      new Config(config, null, false),
      terrain.teamGameSpawnAreas,
    );
    const frame = new CellFrame(game.map(), cfg.compact ? 4 : 8);
    const sim = { hash: 0, fatal: null as string | null };
    const runner = new GameRunner(game, new Executor(game, seed, undefined), (gu) => {
      if ("errMsg" in gu) {
        sim.fatal = gu.errMsg;
        return;
      }
      frame.mark(gu.packedTileUpdates);
      for (const h of gu.updates[GameUpdateType.Hash] as HashUpdate[]) sim.hash = h.hash;
    });
    runner.init();
    const seats = humans.map((h) => game.player(h.id));
    return new EnvGame(cfg, game, seats, frame, 1 + cfg.seats + nations.length + cfg.bots, runner, sim);
  }

  /** Executes one turn (one tick) with the given intents. */
  tickWith(intents: StampedIntent[]): void {
    this.runner.addTurn({ turnNumber: this.turn++, intents });
    if (!this.runner.executeNextTick()) throw new Error(`tick ${this.turn - 1} failed: ${this.sim.fatal}`);
  }
}
