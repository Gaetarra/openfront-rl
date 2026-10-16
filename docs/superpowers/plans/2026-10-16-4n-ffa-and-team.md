# 4N: FFA and Team Against the Nations (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One PPO run trains StockShip on ranked 1v1, public FFA against the nations and Humans vs Nations at once,
and `trainer.eval` plays Gate 4N's games under the public rules.

**Architecture:**
- **Bridge.**
  - Public FFA and Humans vs Nations configs are copied from the playlist (`MapPlaylist.rollConfig`), as
    `ranked1v1Config` copies `get1v1Config`.
  - They use the 34 public maps that fit the ranked grid, minus 4 held out. The nations come from the game's own rule
    (`nations: "default"` in a public game).
  - A Team game's seats share their team's result.
  - A worker can join a mixed run: it builds its games on the run's shared grid and player table.
- **Trainer.**
  - `VecEnv` takes one config per worker. A game's rows are its seats, so games may have different seat counts.
  - `trainer.ppo --mix ranked1v1=8 ffa=4 hvn=2` learns from one seat of each ranked game (the league's pick plays the
    other) and from every seat of the FFA and Humans vs Nations games.
  - Each setup's curves are logged, and the frozen evaluations play each setup.
  - `trainer.eval --setup ffa|hvn --maps heldout|seen|training` plays Gate 4N's games.

**Tech Stack:** TypeScript bridge (node:test via tsx), Python 3 + torch (unittest), as before.

**Spec:** `docs/PLAN.md`, Phase 4, "4N — FFA and Team against the nations" (approved by the user on 2026-10-15).

## Global Constraints

- Never edit `OpenFront_Source_Code/openfrontio/`. Bridge code lives in `bridge/`, trainer code in `trainer/`.
- Rule values come from the game at runtime. Playlist copies say which game function they copy, so they can be
  compared on every upgrade.
- No `Co-Authored-By` line or any other Claude attribution in commits.
- Learner seats stay at 32: 8 ranked games (8 seats), 4 FFA games (16), 2 Humans vs Nations games (8).
- Training games have a 30-minute timer (the tile leader wins). The gate plays the public rules: no timer; FFA's bar
  drops 2 points a minute after 30 minutes, Team's stays at 80%.
- Rewards follow decision 9, among the seats and the nations; tribes do not count. A team's result is shared.
- Maps: the public maps that fit 192 × 256 cells, weighted like the rotation, minus the 4 held out. Compact in a third
  of the games, with the game's own nation counts.
- FFA: all of the map's nations, Medium or Hard, half each (the gates: Hard), and the tribes (400; 100 on Compact).
  Humans vs Nations: 4 learner seats against 4 Hard nations.
- Unchanged: the network, and the KL reference (the 3B clone).
- Frozen evaluations every 500 updates: ranked 1v1 against the panel, 64 games; FFA, one seat against Hard nations,
  32; Humans vs Nations, 32. All on training maps with the 30-minute timer.
- Bridge tests run on the CPU. Trainer tests start Node workers and need the GPU, so they wait while a GPU run goes
  (`trainer.ctl pause` for a training run).
- The speed check, the baselines and the run are GPU runs: each one needs the user's go.
- Bridge commands run from `OpenFront_Source_Code/openfrontio/`, trainer commands from the repo root.

## Game facts this plan relies on (checked in v0.34.24)

- **`MapPlaylist.rollConfig`**, the FFA and Team branch:
  - FFA:
    - Public, Medium nations, `nations: "default"`;
    - no donations, `disableClanTags: true`;
    - `overtime: { enabled: true }`, no timer;
    - bots 400 (Compact 100), spawn immunity 5 s.
  - Humans vs Nations:
    - Team mode, `playerTeams: HumansVsNations`, Hard nations;
    - donations on, no overtime;
    - the same bots and spawn immunity.
  - Compact in a third of the games (`playlist.length % 3 === 0`).
  - Map weights: `ffaFrequency` / `teamFrequency`, else `multiplayerFrequency` (`buildMapsList`).
- **`createNationsForGame`**, a public game with `nations: "default"`:
  - Normal FFA: every manifest nation;
  - Compact FFA: `max(1, floor(n / 4))` of them, shuffled;
  - Humans vs Nations: as many as the humans.
- **Teams.**
  - `GameImpl.addPlayers` puts the humans of Humans vs Nations in `ColoredTeams.Humans` and the nations in
    `ColoredTeams.Nations`.
  - A team wins as its name: `getWinner()` returns the team string.
- **`WinCheckExecution`:**
  - FFA: the player with the most tiles among `players()` (tribes included) wins at the timer or above the bar;
  - Team: the top team wins, unless it is the tribes' team, in which case nobody does;
  - the bar is 80% of the land without fallout; overtime lowers it after `startMinutes` (30);
  - a hard limit of 170 minutes.
- **`SpawnExecution`:** a chosen spawn tile works anywhere. Team spawn areas apply only to random spawns, so Humans vs
  Nations needs no new spawn mask.
- **Spawn phase:** 200 ticks (40 decisions) in non-singleplayer games without random spawn.
- **The maps that fit 192 × 256 cells** (`mapGrid` on the Normal size; Compact shares it), with FFA / Team weights and
  manifest nations:
  - 34 maps, 31% of the FFA rotation and 32% of Team. World is 30 / 30 with 72 nations.
  - EuropeClassic, a ranked map, is not in the public rotation.

## Decisions (for the user's review)

1. **The held-out maps:** Didier, Bering Strait, Yellow Sea and Labyrinth.
   - How they were drawn: the 30 candidates are the 34 fitting maps without World, Australia, Iceland and Asia,
     sorted by name. The draw is `new PseudoRandom(simpleHash("4N held out")).shuffleArray(candidates).slice(0, 4)`.
   - They are hard-coded, so a game upgrade that adds a map cannot change them.
   - Their FFA nations: 12, 2, 8 and 26 (Compact: 3, 1, 2 and 6).
   - Bering Strait has only 2 nations, which makes it an easy FFA gate map. The draw stands: drawing again for a better
     set would be choosing the test.
2. **The gate's "4 training maps"** (seen, to measure the gap): Black Sea, Venice, Mars and More Than Luck.
   - They are drawn the same way from the 26 candidates left, with the seed "4N seen".
   - Their nations: 9, 15, 6 and 35.
3. **Gate map sets are played equally often**, a quarter of the games on each map. Training maps are drawn by their
   weight.
4. **A game without a timer is observed with a 30-minute horizon**: time scalars 0 and 12 reach their end at 30
   minutes, as in training, which is when FFA's bar starts to drop.
   - Today a game without a timer reads as 60 minutes, and no current setup has one.
   - Under 60, the first 30 minutes of every public game would look like the first half of a game twice as long as
     any the bot trained on.
5. **The backstop still ends a game 1 minute after its timer, or at 61 minutes without one.**
   - A game the backstop ends with no winner is *cut*: it is decided by tiles, and the eval reports the share of cut
     games.
   - The case that can bite is Humans vs Nations under the public rules, where a team must reach 80%.
6. **Dead seats keep their rows.** A dead FFA or Humans vs Nations seat keeps deciding (no-ops) until its game ends.
   `setups/dead_share/<setup>` logs how many learner decisions that costs, before anything is changed.
7. **The run's layout:** `--mix ranked1v1=8 ffa=4 hvn=2 --games-per-worker 1`, so 14 workers with one game each.
   - The frozen evaluations use the same 14 workers.
   - My estimate is ~45 minutes per evaluation round (the spec said ~27), to be measured in Task 6.
8. **The run's pool and panel** are proposed in Task 8, for the user to settle before it starts.

## Review Focus

1. **A mixed run resumed after a crash.** The mix, the rows and the league's slots must come back as they were.
   Task 4's smoke test resumes once.
2. **All 4 FFA seats out before the timer** (they never spawn, or die). The env ends the game at once, and each seat is
   ranked by the game's rules. Task 2: four FFA seats that never spawn finish last, -1 each, right after the spawn
   phase.
3. **A Humans vs Nations team with no winner.** The game names no winner when the tribes lead or the team never
   spawns. The team is then ranked by tiles, and all 4 seats share the result. Task 2: a team that never spawns loses
   together (-1 each, 2 sides).
4. **World, Normal, with 4 seats: 76 real players and a full player table** (smallIDs up to 476 of 477 rows). Task 1
   builds that game, observes it after the tribes spawn, and checks the nation count. Task 4's `cap_binds` test pins
   the counting rule.
5. **Games of one mixed run restarting at different times.** Rows must stay with their game across resets. Task 3
   compares a mixed run's FFA game with the same game played alone, through several restarts.

---

### Task 1: Public FFA and Humans vs Nations configs (bridge)

**Files:**
- Modify: `bridge/setups.ts`, `bridge/maps.ts`, `bridge/game.ts` (`EnvConfig`, `EnvGame.create`), `bridge/testkit.ts`
- Test: `bridge/setups.test.ts`, `bridge/game.test.ts`

**Interfaces:**
- Produces:
  - **`EnvConfig`:**
    - `playlist?: "ranked1v1" | "ffa" | "hvn"`;
    - `maps?: "training" | "heldout" | "seen"` (ffa and hvn; default `"training"`);
    - for ffa and hvn, only `seats`, `maps`, `difficulty` and `timerMinutes` are read;
    - `difficulty` is a comma-separated list, one value drawn per game (e.g. `"Medium,Hard"`);
    - `timerMinutes: 0` means no timer (the public rules).
  - **`bridge/setups.ts`:**
    - `HELD_OUT_MAPS: GameMapType[]` = [Didier, BeringStrait, YellowSea, Labyrinth];
    - `SEEN_MAPS: GameMapType[]` = [BlackSea, Venice, Mars, MoreThanLuck];
    - `publicMaps(team: boolean, which: "training" | "heldout" | "seen"): GameMapType[]`;
    - `publicConfig(cfg: EnvConfig, random: PseudoRandom): GameConfig`;
    - `shapeOf(cfgs: EnvConfig[]): { grid: [number, number]; slots: number }`.
  - **`bridge/maps.ts`:** `nationCount(map: GameMapType): number`, the manifest's nations.
  - **`EnvGame.create(cfg: EnvConfig, seed: string, shape = shapeOf([cfg]))`.**
  - **`bridge/testkit.ts`:**
    - `FFA: EnvConfig = { ...ONION, seats: 4, playlist: "ffa", maps: "training", difficulty: "Medium,Hard", timerMinutes: 30 }`;
    - `HVN = { ...FFA, playlist: "hvn", difficulty: "Hard" }`.

- [ ] **Step 1: Write the failing tests** in `bridge/setups.test.ts`.

```ts
test("public FFA and Humans vs Nations configs copy the playlist's rules", () => {
  const random = new PseudoRandom(7);
  for (let i = 0; i < 300; i++) {
    const f = publicConfig(FFA, random);
    GameConfigSchema.parse(f);
    assert.equal(f.gameType, GameType.Public);
    assert.equal(f.gameMode, GameMode.FFA);
    assert.equal(f.nations, "default");
    assert.equal(f.maxTimerValue, 30);
    assert.deepEqual(f.overtime, { enabled: true });
    assert.equal(f.bots, f.gameMapSize === GameMapSize.Compact ? 100 : 400);
    assert.equal(f.spawnImmunityDuration, 50);
    assert.equal(f.donateGold || f.donateTroops, false);
    assert.equal(f.disableClanTags, true);
    const h = publicConfig(HVN, random);
    GameConfigSchema.parse(h);
    assert.equal(h.gameMode, GameMode.Team);
    assert.equal(h.playerTeams, HumansVsNations);
    assert.equal(h.difficulty, Difficulty.Hard);
    assert.equal(h.donateGold && h.donateTroops, true);
    assert.equal(h.overtime, undefined);
  }
  assert.equal(publicConfig({ ...FFA, timerMinutes: 0 }, random).maxTimerValue, undefined);
});

test("training maps follow the rotation's weights without the held-out maps, Compact in a third", () => {
  // 6000 FFA draws: Compact share within 0.03 of 1/3; Hard share within 0.03 of 0.5;
  // World's share within 0.02 of 30 / (the sum of the 30 training maps' FFA weights);
  // no draw is a HELD_OUT_MAPS map. 2000 draws with maps "heldout": each of the 4 maps within 0.03 of 1/4.
});

test("the held-out and seen maps are fixed, fit the ranked grid and stay out of training", () => {
  // HELD_OUT_MAPS and SEEN_MAPS deepEqual the lists above; neither holds World, Australia, Iceland or Asia;
  // they are disjoint; SEEN_MAPS ⊂ new Set(publicMaps(false, "training"));
  // new Set(publicMaps(false, "training")).size === 30 and the same for team === true;
  // every map of every set: mapGrid(m, GameMapSize.Normal).h <= 192 && .w <= 256.
});

test("a mixed run shares the ranked grid and the largest player table", () => {
  assert.deepEqual(gridFor(FFA), [192, 256]);
  assert.equal(slotsFor(FFA), 1 + 4 + 72 + 400); // World's 72 nations
  assert.equal(slotsFor(HVN), 1 + 4 + 4 + 400);
  assert.equal(slotsFor({ ...FFA, seats: 1, maps: "heldout" }), 1 + 1 + 26 + 400); // Labyrinth
  assert.deepEqual(shapeOf([RANKED, FFA, HVN]), { grid: [192, 256], slots: 477 });
  assert.deepEqual(shapeOf([ONION]), { grid: [64, 64], slots: 5 });
});
```

- [ ] **Step 2: Write the failing nation-count test** in `bridge/game.test.ts`.
  - It builds games with `EnvGame.fromStart` on a fixed map and size: `{ ...publicConfig(cfg, new PseudoRandom(1)),
    gameMap, gameMapSize }`, with players named as `EnvGame.create` names them.
  - FFA, World, Normal: 72 nations (`nationCount(World)`).
  - FFA, World, Compact: 18.
  - Humans vs Nations, World, Normal:
    - 4 nations;
    - every seat's `team()` is `ColoredTeams.Humans`, every nation's `ColoredTeams.Nations`.
  - The Normal FFA game ticks 300 times (past the spawn phase, tribes spawned), and `observe(true)` returns without
    throwing, with `env.slots === 477`.

- [ ] **Step 3: Run them to see them fail.**
  - Run (game dir): `node --import tsx --test ../../bridge/setups.test.ts ../../bridge/game.test.ts`
  - Expected: FAIL. `publicConfig`, `HELD_OUT_MAPS`, `shapeOf` and `nationCount` are not exported.

- [ ] **Step 4: Implement it.**
  - **`publicConfig`** returns the `rollConfig` fields listed in the game facts, plus:
    - `maxPlayers: cfg.seats`;
    - `maxTimerValue`: `cfg.timerMinutes` if it is > 0, else `undefined`;
    - the map, drawn uniformly from `publicMaps(cfg.playlist === "hvn", cfg.maps ?? "training")`;
    - Compact with `random.next() < 1 / 3`;
    - the difficulty, one of `cfg.difficulty.split(",")`.
  - **`publicMaps("training")`:** each map that fits the ranked grid, has a weight above 0 and is not held out,
    repeated by its weight (as `buildMapsList` does). The heldout and seen sets list their 4 maps once each.
  - **The ranked grid:** computed from `RANKED_1V1_MAPS` with the same function as `gridFor`.
  - **`gameConfigOf`** sends ffa and hvn to `publicConfig`.
  - **`mapsOf`** gives ffa and hvn their set's distinct maps, at both sizes.
  - **`slotsFor`:**
    - ffa: `1 + seats + max nationCount over the set + 400`;
    - hvn: `1 + 2 · seats + 400`.
  - **`shapeOf`:** the elementwise maximum of `gridFor`, and the maximum of `slotsFor`.
  - **`EnvGame.create`** passes `shape.grid[1]` and `shape.slots` to `fromStart`.
  - **Comments:** `publicConfig` names `MapPlaylist.rollConfig` (v0.34.24). `HELD_OUT_MAPS` states its draw (Decision
    1) and `SEEN_MAPS` its own (Decision 2).

- [ ] **Step 5: Run the bridge tests and the typecheck.**
  - Run (game dir): `node --import tsx --test "../../bridge/*.test.ts"`, then
    `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`.
  - Expected: all pass, and the typecheck is clean.

- [ ] **Step 6: Commit.**
  - `git add bridge/setups.ts bridge/maps.ts bridge/game.ts bridge/testkit.ts bridge/setups.test.ts bridge/game.test.ts`
  - `git commit -m "4N task 1: public FFA and Humans vs Nations configs copied from the playlist, on the 34 maps that fit the ranked grid minus 4 held out (Didier, Bering Strait, Yellow Sea, Labyrinth), with the game's own nations; a run's games can share one grid and player table"`

### Task 2: Team results, the no-timer horizon and cut games (bridge)

**Files:**
- Modify: `bridge/ranking.ts`, `bridge/game.ts`, `bridge/worker.ts` (the end header)
- Test: `bridge/game.test.ts`

**Interfaces:**
- Consumes: Task 1's `FFA`, `HVN` and `publicConfig`.
- Produces:
  - **`seatStanding(game, seat)`:**
    - in Team mode, the rank of the seat's team among the teams of non-bot players, and `n` = the number of those
      teams;
    - the winner's team is first; the others are ranked by their members' summed tiles (`rankOf` on team tiles);
    - FFA is unchanged.
  - **`EnvGame`:**
    - `NO_TIMER_HORIZON = 30` (exported);
    - `get horizonMinutes()` = `maxTimerValue ?? NO_TIMER_HORIZON`, which replaces `timerMinutes` in `observe`;
    - the backstop is `(maxTimerValue ?? 60) + 1` minutes after the spawn phase;
    - `cut: boolean`: the backstop ended the game and nobody won.
  - **The worker's end header** carries `cut`.

- [ ] **Step 1: Write the failing tests** in `bridge/game.test.ts`.
  - **"an FFA game ends with every seat ranked among the seats and nations":**
    - FFA on Onion, Compact, with a 1-minute timer (via `fromStart`, as in Task 1);
    - random legal play for 4 seats until `done` (at most 400 steps);
    - `env.nPlayers === 4 + nations`;
    - each rank is in 1..n, and `rewards[i] === rewardOf(ranks[i], n)`.
  - **"four FFA seats that never spawn finish last together":** no-ops until done.
    - It ends within 41 steps;
    - every rank is `n`, and every reward is -1.
  - **"a Humans vs Nations team shares its result":** random legal play, 1-minute timer, until done.
    - `nPlayers === 2`;
    - all 4 rewards are equal, and in {1, -1}.
  - **"a Humans vs Nations team that never spawns loses together":** no-ops until done.
    - The ranks are `[2, 2, 2, 2]` and the rewards `[-1, -1, -1, -1]`.
  - **"a game without a timer is observed with a 30-minute horizon":** FFA with `timerMinutes: 0`.
    - `env.horizonMinutes === 30`;
    - after 150 ticks with `tickWith([])` (inside the spawn phase: seats that never spawn end the game at tick 200),
      seat 0's scalar 0 equals `150 / (200 + 30 * 600)` within 1e-6 (read from the `scalars` blob).

- [ ] **Step 2: Run them to see them fail.**
  - Run (game dir): `node --import tsx --test ../../bridge/game.test.ts`
  - Expected: FAIL. The team's ranks differ per seat, and `horizonMinutes` is undefined.

- [ ] **Step 3: Implement it** as in the interfaces.
  - Find the team with `game.config().gameConfig().gameMode === GameMode.Team`.
  - `grep -n timerMinutes bridge/*.ts`, and move every observation use to `horizonMinutes`.

- [ ] **Step 4: Run all bridge tests and the typecheck.**
  - Run (game dir): the commands of Task 1, Step 5.
  - Expected: all pass, and the typecheck is clean.

- [ ] **Step 5: Commit.**
  - `git commit -m "4N task 2: a Team game's seats share their team's result (decision 9 among teams); games without a timer are observed with a 30-minute horizon, ended at 61 minutes, and marked cut when the backstop decides"`

### Task 3: A VecEnv with a setup per worker and games of different sizes

**Files:**
- Modify: `bridge/worker.ts`, `trainer/env.py`
- Test: `bridge/worker.test.ts`, `trainer/test_env.py`

**Interfaces:**
- Consumes: Task 1's `shapeOf`, and Task 2's `cut` in the end header.
- Produces:
  - **The worker's `init`** takes `mix?: EnvConfig[]`, every config of the run. Its `dims` (grid and slots) and its
    games use `shapeOf(mix ?? [config])`. Its `seats` stay its own config's.
  - **`trainer/env.py`:**
    - `FFA = {**RANKED_1V1, "playlist": "ffa", "seats": 4, "maps": "training", "difficulty": "Medium,Hard", "timerMinutes": 30}`;
    - `HVN = {**FFA, "playlist": "hvn", "difficulty": "Hard"}`;
    - `PLAYLISTS = {"ranked1v1": RANKED_1V1, "ffa": FFA, "hvn": HVN}`;
    - `frozen_eval_config(setup)`: for ffa `{**PLAYLISTS["ffa"], "seats": 1, "difficulty": "Hard"}`, else a copy of
      `PLAYLISTS[setup]`. It reads `PLAYLISTS` at call time, so tests can patch it;
    - `gate_config(setup, maps)`: `{**frozen_eval_config(setup), "timerMinutes": 0, "maps": maps}`.
  - **`VecEnv(config, ...)`:**
    - `config` may be a list with one dict per worker;
    - every worker then gets `mix` = the distinct configs;
    - all workers must report the same grid and slots;
    - replay mode keeps a single config.
  - **New `VecEnv` attributes:**
    - `seats` (np int [E]);
    - `first` (np int [E], each game's first row);
    - `row_env` (LongTensor [N] on the device);
    - `S`, the common seat count, or `None` when games differ. Code that needs one fails loudly.
  - **Rows stay game-major:** game e owns rows `first[e] .. first[e] + seats[e] - 1`.
    - `send(actions)` takes [N, k];
    - `_absorb` returns rewards, dones and invalid as [N];
    - infos also carry `"cut"`.

- [ ] **Step 1: Write the failing worker test** in `bridge/worker.test.ts`.
  - "a worker in a mixed run uses the run's grid and player table": init with `{config: ONION, mix: [ONION, RANKED],
    games: 1}`.
  - Expected: `dims.grid` is `[192, 256]`, `dims.slots` 403 and `dims.seats` 1. The `players` blob holds
    `403 * 17 * 4` bytes.

- [ ] **Step 2: Write the failing env tests** in `trainer/test_env.py`.
  - **`test_a_mixed_runs_ffa_game_matches_the_same_game_played_alone`:**
    - `mixed = VecEnv([RANKED_1V1, FFA1, HVN1], 3, 1, seed="mix")` and `alone = VecEnv(FFA1, 2, 1, seed="mix")`, where
      `FFA1 = {**FFA, "timerMinutes": 1}` and `HVN1` likewise. Worker 1 plays FFA in both, with the same game seeds.
    - Assert:
      - `mixed.seats` is `[2, 4, 4]`, `mixed.first` is `[0, 2, 6]`, `mixed.N == 10`, `mixed.S is None` and
        `alone.S == 4`;
      - `mixed.obs["spatial"].shape == (10, N_CHANNELS, 192, 256)` and `mixed.obs["players"].shape[1] == 477`.
    - For 400 steps:
      - the FFA game's rows (mixed 2:6, alone 4:8) get the same actions from their own `Scripted(seed=0)`;
      - every other row gets a no-op;
      - every observation key of those rows is `torch.equal` between the two envs, checked every 50 steps and at the
        end.
    - The 1-minute FFA and Humans vs Nations games restart ~2 times while the ranked game goes on.
  - **`test_units_paint_every_seat_of_their_own_game`:**
    - a `VecEnv.__new__` object on the CPU, with `seats = [2, 4]`, `first = [0, 2]`, a 4 × 4 grid and `P = 8`, plus the
      attributes `_add_units` reads;
    - one City record in game 1 owned by player 5, which is Self for row 4 and Enemy for rows 2, 3 and 5;
    - the city's cell holds `BUILDINGS + 4 · k + r - 1` = 1 on rows 2–5 with each row's own relation;
    - rows 0–1 stay zero.

- [ ] **Step 3: Run them to see them fail.**
  - Run (game dir): `node --import tsx --test ../../bridge/worker.test.ts`
  - Run (repo root, GPU free): `.venv/Scripts/python -m unittest trainer.test_env -k mixed -k paint`
  - Expected: FAIL. `VecEnv` takes one config, and `FFA` does not exist yet.

- [ ] **Step 4: Implement it.**
  - **`_absorb`** fills [N, ...] arrays per game at `first[e]:first[e] + seats[e]`.
  - **`_observe`** uses `self.row_env`.
  - **`_add_units`** loops `s` over `range(seats.max())`, with `row = first[env] + s`. Where `seats[env] <= s`, it
    sets `r = 0`, which drops the record from every channel. The row index is clamped to `N - 1` before indexing.

- [ ] **Step 5: Run the bridge and trainer test suites.**
  - Run (game dir): `node --import tsx --test "../../bridge/*.test.ts"`
  - Run (repo root): `.venv/Scripts/python -m unittest discover -s trainer -t .`
  - Expected: all pass. The BC replay tests and the ranked grid test are unchanged.

- [ ] **Step 6: Commit.**
  - `git commit -m "4N task 3: VecEnv takes a config per worker and games of different seat counts (rows game-major); workers of a mixed run share its grid and player table; the FFA and Humans vs Nations setups"`

### Task 4: PPO on a mix of setups

**Files:**
- Modify: `trainer/ppo.py`, `trainer/control.py` (one chart), `CLAUDE.md` (the training commands)
- Test: `trainer/test_ppo.py`

**Interfaces:**
- Consumes: Task 3's `PLAYLISTS`, `VecEnv(list)`, `seats` and `first`, and `cap_binds`' input
  `obs["players"]`.
- Produces:
  - **`--mix SETUP=WORKERS ...`** (nargs `+`; the setups are `PLAYLISTS`' keys):
    - workers play the setups in the order given;
    - `--workers` becomes their sum, and `--setup` is not used;
    - steps and minibatches default as in ranked1v1 (the same grid);
    - `mix_of(a) -> dict[str, int]` parses it and refuses unknown setups or counts below 1.
  - **`seat_rows(seats, whole, device="cpu") -> (learner, others, slots, owner)`:**
    - `learner`: LongTensor [L] of the learner's rows;
    - `others`: a list (one per other seat) of LongTensor [K], the frozen opponents' rows;
    - `slots`: list[int], the game of each of the K league slots;
    - `owner`: list of `(game, seat)` per learner row;
    - a game with `whole[e]` (ffa, hvn) gives every row. The k-th other game gives seat `k % S` to the learner and the
      rest to `others`.
  - **`cap_binds(players: Tensor [n, P, F]) -> BoolTensor [n]`:**
    - true where the real players (exists, and self, friendly or enemy) plus row 0 outnumber `ATTENDED` (64);
    - tribes do not count.
  - **`Outcomes`:**
    - infos carry `"row"` (the learner row) and `"setup"`;
    - one window of 200 per setup, kept in `state()["by_setup"]`;
    - `summary()` adds `setups/win_rate/<s>`, `setups/mean_reward/<s>` and `setups/ticks/<s>` once two setups have
      finished games;
    - per-seat rates appear only with one setup.
  - **Logged per update:**
    - `actions/<setup>/<act>`;
    - `setups/cap_binds/<setup>` and `setups/dead_share/<setup>` (decisions after the spawn phase with scalar 4, alive,
      at 0);
    - all three are accumulated on the GPU and read once per update.
  - **The league's slots** are the K ranked games. An info goes to `league.finished` with `"env"` replaced by its slot.
  - **`run_info` for a mix** lists each setup's games and learner seats.
  - **A control-page chart** "Win rate by setup", with prefix `setups/win_rate/`.

- [ ] **Step 1: Write the failing unit tests** in `trainer/test_ppo.py`.

```python
def test_the_learner_plays_one_seat_of_ranked_games_and_every_seat_of_ffa_and_hvn(self):
    learner, others, slots, owner = seat_rows([2, 2, 4, 4], [False, False, True, True])
    self.assertEqual(learner.tolist(), [0, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    self.assertEqual([o.tolist() for o in others], [[1, 2]])
    self.assertEqual(slots, [0, 1])
    self.assertEqual(owner[:3], [(0, 0), (1, 1), (2, 0)])
    uniform = seat_rows([2] * 4, [False] * 4)  # today's league: seat e % 2
    self.assertEqual(uniform[0].tolist(), [0, 3, 4, 7])

def test_the_attention_cap_binds_above_63_real_players(self):
    # players [3, 80, 17] zeros; row 0 exists; row r has f0 = 1 and f2 (self) or f4 (enemy) for 63, 64 real players;
    # a third row with 63 real players and 10 bots (f5): cap_binds == [False, True, False]

def test_each_setup_keeps_its_own_window_and_it_survives_a_restart(self):
    # Outcomes(2): finish rows with setups "ranked1v1" and "ffa"; summary has setups/win_rate/ranked1v1 and /ffa;
    # Outcomes(2).load(o.state()) gives the same summary
```

- [ ] **Step 2: Write the failing smoke test** `test_a_mixed_run_learns_from_every_ffa_and_hvn_seat`.
  - Patches: `mock.patch.dict("trainer.env.PLAYLISTS", {"ranked1v1": {**ONION, "seats": 2, "nations": 0, "timerMinutes":
    1}, "ffa": {**FFA, "timerMinutes": 1}, "hvn": {**HVN, "timerMinutes": 1}})`.
  - Run: `main(["--run", RUN, "--mix", "ranked1v1=1", "ffa=1", "hvn=1", "--games-per-worker", "1", "--steps", "48",
    "--minibatches", "2", "--epochs", "1", "--port", "0", "--init", init, "--opponent", init, "--updates", "4"])`,
    then `--resume --updates 5`.
  - Assert:
    - `global_step == 5 * 48 * 9`;
    - `args["mix"] == ["ranked1v1=1", "ffa=1", "hvn=1"]` and `args["workers"] == 3`;
    - TensorBoard holds `setups/win_rate/ffa`, `/hvn` and `/ranked1v1`, `setups/cap_binds/ffa` and
      `actions/hvn/attack`;
    - in `outcomes["by_setup"]["hvn"]`, the episodes of each game (same env and ticks) share one reward.

- [ ] **Step 3: Run them to see them fail.**
  - Run (GPU free): `.venv/Scripts/python -m unittest trainer.test_ppo`
  - Expected: FAIL. `seat_rows` takes (E, S), and `--mix` is unknown.

- [ ] **Step 4: Implement it.**
  - **The rollout:**
    - rewards and dones per learner row are `rew[learner_np]` and `done[learner_np]`;
    - ended infos are the infos whose `(env, seat)` is in `owner`, with `"row"` and `"setup"` added.
  - **Checks:** `--opponent` or `--league` is needed only when `others` is not empty.
  - **`CLAUDE.md`:** add the `--mix` command under Training commands, with one line on the row layout.

- [ ] **Step 5: Run the trainer suite.**
  - Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
  - Expected: all pass. The league and two-seat tests are unchanged.

- [ ] **Step 6: Commit.**
  - `git commit -m "4N task 4: trainer.ppo --mix trains on several setups at once: one learner seat per ranked game against the league, every seat of FFA and Humans vs Nations games; win rate, game length, action mix, attention-cap and dead-seat shares per setup"`

### Task 5: Frozen evaluations per setup, and Gate 4N's games in `trainer.eval`

**Files:**
- Modify: `trainer/eval.py`, `trainer/ppo.py` (`evaluate`, flags), `trainer/control.py` (one chart), `CLAUDE.md`
- Test: `trainer/test_eval.py`, `trainer/test_ppo.py`

**Interfaces:**
- Consumes: Task 3's `frozen_eval_config` and `gate_config`, and Task 4's mix.
- Produces:
  - **`trainer.eval`:**
    - `--setup ffa|hvn` plays `gate_config(setup, a.maps)` with `[a.a] * seats` (ffa one seat, hvn four);
    - `--maps heldout|seen|training` (default heldout);
    - the result JSON adds `"maps"`;
    - `summarize` adds `cut_share`.
  - **`trainer.ppo`:**
    - `--eval-every N` (a multiple of `SNAPSHOT`; default `SNAPSHOT`) and `--eval-games-public` (default 32). Both may
      change on a resume, as `--eval-games` does;
    - in a mixed run, `evaluate()` also plays `frozen_eval_config(s)` for ffa and hvn in the mix, with
      `[spec] * seats`;
    - `panel.jsonl` rows gain `"nations": {setup: summary}`, and their `minutes` cover the whole round. TensorBoard
      gains `eval/nations_<setup>`;
    - the ranked panel plays `PLAYLISTS["ranked1v1"]` in a mixed run.
  - **A control-page chart** "Frozen evaluations against the nations": prefix `eval/nations_`, with a dashed line at
    0.5 for Gate 4N's FFA bar (the note gives HvN's 0.70).

- [ ] **Step 1: Write the failing tests.**
  - **`test_eval`, "the gate's setups play the public rules with every seat policy A":**
    - patch `trainer.eval.play` with a mock returning `{"recorded": [], "win_rate": 0.5}`, and `trainer.eval.RESULTS`
      with a temporary folder;
    - `main(["--setup", "hvn", "--a", "random"])` calls it with specs `["random"] * 4` and config
      `gate_config("hvn", "heldout")` (`timerMinutes` 0). A checkpoint path would be loaded by `replay_dir`;
    - `--setup ffa --maps seen` calls it with `["random"]` and maps `"seen"`.
  - **`test_eval`, "Humans vs Nations seats share one result per game":**
    - `play(["random"] * 4, {**HVN, "timerMinutes": 1}, 2, 1, 2)`;
    - `episodes == 2` and `0 <= cut_share <= 1`.
  - **`test_ppo`:** the smoke test of Task 4 gains `--panel init --eval-every 4 --eval-games 2 --eval-games-public 1
    --save-every 1` under `mock.patch("trainer.ppo.SNAPSHOT", 2)`. `--save-every` must divide `SNAPSHOT`.
    - `panel.jsonl` holds updates `[0, 4]`, and the update 4 row has `nations` with `ffa` and `hvn`;
    - TensorBoard holds `eval/nations_ffa`.

- [ ] **Step 2: Run them to see them fail.**
  - Run (GPU free): `.venv/Scripts/python -m unittest trainer.test_eval trainer.test_ppo`
  - Expected: FAIL. `--setup hvn` is not a choice, and `--eval-every` is unknown.

- [ ] **Step 3: Implement it.**
  - **`CLAUDE.md`** gets the Gate 4N eval commands:
    - `trainer.eval --setup ffa --a <ckpt> --maps heldout --episodes 128 --workers 16 --games-per-worker 2`, then
      `--maps seen`;
    - `trainer.eval --setup hvn --a <ckpt> --maps heldout --episodes 128 --workers 16 --games-per-worker 2`.

- [ ] **Step 4: Run the trainer suite.**
  - Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
  - Expected: all pass.

- [ ] **Step 5: Commit.**
  - `git commit -m "4N task 5: frozen evaluations per setup (FFA one seat against Hard nations, Humans vs Nations) every --eval-every updates; trainer.eval --setup ffa|hvn --maps heldout|seen|training plays Gate 4N's games under the public rules"`

### Task 6: Speed and memory of the mixed run (GPU, ~15 min, the user's go)

**Files:**
- Modify: `docs/PLAN.md` (4N, the results)

**Interfaces:**
- Consumes: Tasks 1–5. The start checkpoint (`checkpoints/league4a2/avg004000.pt` if Gate 4A passed, else
  `checkpoints/swa-lr4a/update1000-2000.pt`).

- [ ] **Step 1: Run 30 updates with the PC otherwise idle.**
  - Run: `.venv/Scripts/python -m trainer.ppo --run speed4n --mix ranked1v1=8 ffa=4 hvn=2 --games-per-worker 1 --init <start> --opponent <start> --kl-ref checkpoints/bc3b-v4/latest.pt --kl-coef 0.2 --lr 2.5e-5 --ent-coef 0 --minibatches 12 --updates 30`
  - Read from TensorBoard (`runs/speed4n`), over updates 10–30:
    - `perf/decisions_per_s`, `perf/env_share`, `perf/private_gb` and `perf/gpu_reserved_gb`;
    - `setups/cap_binds/ffa`.
  - Compare with league v2: 115–137 decisions/s, 19.4–19.9 GiB private, ~16.5 s per update.
- [ ] **Step 2: Time one evaluation round.** Use Step 1's command with `--run speed4n-eval --updates 1 --panel <start>
  --eval-games 64 --eval-games-public 32`. The round plays before the first update. Record its minutes from
  `runs/speed4n-eval/panel.jsonl`.
- [ ] **Step 3: Record the results.** Add them, with the ETA of 4,000 updates and the evaluations' share of the time,
  to PLAN 4N.
- [ ] **Step 4: Commit.**
  - `git commit -m "PLAN 4N: the mixed run's speed and memory"`

### Task 7: Baselines (GPU, ~3–4 h, the user's go)

**Files:**
- Modify: `docs/PLAN.md` (4N, the baselines)

- [ ] **Step 1: Write a launcher.** Write `runs/baseline4n.cmd` (CRLF, started with `explorer.exe`, as
  `runs/gate4a.cmd`). It runs three matches with the start, each saving its JSON to `runs/eval/`:
  - `--setup ffa --maps heldout --episodes 128 --workers 16 --games-per-worker 2`;
  - `--setup ffa --maps seen --episodes 128 --workers 16 --games-per-worker 2`;
  - `--setup hvn --maps heldout --episodes 128 --workers 16 --games-per-worker 2`.
- [ ] **Step 2: Apply the rules fixed in the spec.**
  - FFA held-out ≥ 0.50, or HvN ≥ 0.70: that bar is met without training.
  - Record the win rates, `cut_share` and mean ranks in PLAN 4N.
- [ ] **Step 3: Commit.**
  - `git commit -m "PLAN 4N: the start's baselines on Gate 4N's games"`

### Task 8: The 4N run (the user starts it)

**Files:**
- Create: `runs/ffa4n.cmd` (git-ignored; CRLF; the 6-attempt restart loop of `runs/league4a2.cmd`)
- Modify: `CLAUDE.md` (the run's command), `docs/PLAN.md`

- [ ] **Step 1: Write the launcher** with the command below.
  - The pool and panel are proposed. The user settles them before the run starts.
  - Command: `.venv\Scripts\python.exe -m trainer.ppo --run ffa4n --mix ranked1v1=8 ffa=4 hvn=2 --games-per-worker 1 --init <start> --kl-ref checkpoints/bc3b-v4/latest.pt --kl-coef 0.2 --lr 2.5e-5 --ent-coef 0 --minibatches 12 --updates 4000 --league "checkpoints/league4a2/avg*.pt" checkpoints/swa-lr4a/update1000-2000.pt checkpoints/bc3b-v4/latest.pt checkpoints/ppo-bc2/update005000.pt --panel <start> checkpoints/swa-lr4a/update1000-2000.pt checkpoints/bc3b-v4/latest.pt --eval-every 500 --eval-games 64 --eval-games-public 32`
- [ ] **Step 2: Hand over.** The user starts it with `explorer.exe runs\ffa4n.cmd`. Record the start in PLAN and
  memory.
- [ ] **Step 3: After 4,000 updates, play Gate 4N.**
  - Task 7's three matches on the final average;
  - plus the 1v1 bar: `trainer.eval --a <final avg> --b <start> --episodes 512 --workers 16 --games-per-worker 2`
    ≥ 0.45.
