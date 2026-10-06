# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Deep-RL bot for OpenFront.io. The design, phases, gates and every decision taken so far are in `docs/PLAN.md`:
read it before starting new work, and update it when a decision changes.

## Conventions

- Commits and PRs are authored by the repo owner. **Never add `Co-Authored-By: Claude` or any other Claude
  attribution line to commit messages or PR descriptions.**
- Never edit anything inside the game submodule `OpenFront_Source_Code/openfrontio/`. It must stay a clean checkout of
  an official release tag. Our code lives in `bridge/` (TypeScript), `trainer/` (Python) and `scripts/`.
- Game rule values (costs, ranges, radii, speeds) are read from the game's `Config` at runtime, never hard-coded,
  so balance patches flow through automatically.

## Game version

The submodule is pinned to the production release the official servers run (currently `v0.34.23`, commit `71154c52`).
Check what production runs by reading `gitCommit` from a recent record:
`https://api.openfront.io/public/game/<id>?turns=false` (list games via `/public/games`, see the game's `docs/API.md`).
To upgrade, follow the playbook in `docs/PLAN.md` (decision 14): diff `src/core` between the tags, move the submodule,
run the bridge tests and determinism check, then fine-tune from the latest checkpoint.

## Game commands (run inside `OpenFront_Source_Code/openfrontio/`)

```bash
npx -y npm@12.1.0 ci --ignore-scripts         # install deps: v0.34.21+ needs npm >= 12.1 (engine-strict), the PC has 11;
                                               # never use npm install
npm run perf:game -- --map world --ticks 1800 --bots 400    # headless full-game benchmark
npm run replay:game -- <gameID|record.json> --api-base https://api.openfront.io/public   # re-simulate a record, verify hashes
npx vitest tests/SomeTest.test.ts --run        # one game test file
```

The game has its own `CLAUDE.md` with details on its architecture.

## Bridge commands (run inside `OpenFront_Source_Code/openfrontio/`, which provides `tsx` and the tsconfig paths)

```bash
node --import tsx --test "../../bridge/*.test.ts"              # all bridge tests (node:test)
node --import tsx --test ../../bridge/game.test.ts             # one test file
node node_modules/typescript/bin/tsc -p ../../tsconfig.json    # typecheck bridge/ (tsx does not typecheck)
```

## Benchmarks (Phase 0 results are in `docs/PLAN.md`)

```bash
npx tsx ../../scripts/bench_env.ts <record.json> [--pipe]   # env-step cost on a real record (run from the game dir)
npx tsx ../../scripts/troop_shares.ts 60                    # share of their troops humans send per attack/boat, 60 replayed records (~10 min, game dir)
.venv/Scripts/python scripts/bench_gpu.py                   # GPU encoder throughput (run from the repo root)
.venv/Scripts/python scripts/speed_check.py                 # PPO decisions/s, BC rows/s and peak VRAM, Phase 2 vs Phase 3 networks (~15 min, PC idle)
```

Python: `.venv` at the repo root, `pip install -r requirements.txt` (PyTorch CUDA 13.0 wheels).

## Training commands (repo root)

```bash
.venv/Scripts/python -m unittest discover -s trainer -t .      # trainer tests (start real Node workers, need the GPU)
.venv/Scripts/python -m trainer.ppo --run onion                 # train; control page with the key curves: http://127.0.0.1:8787
.venv/Scripts/python -m trainer.ppo --run onion --resume        # continue from checkpoints/onion/latest.pt
.venv/Scripts/python -m trainer.ctl pause|resume|save|stop|throttle half|status
.venv/Scripts/python -m trainer.crawl                          # download ranked 1v1 records into data/ (hours: detach)
.venv/Scripts/python -m trainer.crawl --purge                  # remove deleted players' records (daily while data is kept)
.venv/Scripts/python -m trainer.bc --run bc3-moe --core moe --workers 14 --games-per-worker 6   # behaviour cloning: one pass over data/ (~14 h); same control page
.venv/Scripts/python -m trainer.bc --run probe --init checkpoints/bc/update0060000.pt --updates 2000 --warmup 100   # fine-tune a checkpoint (its network, a fresh optimizer)
.venv/Scripts/python -m trainer.bc --score checkpoints/bc/update0060000.pt checkpoints/bc3-moe/latest.pt   # NLL per head and bucket on the fixed held-out set (~10 min); table + runs/score/*.json
.venv/Scripts/python -m trainer.eval --a checkpoints/bc/update0060000.pt --b scripted   # ranked 1v1 match, 256 games
.venv/Scripts/python -m trainer.eval --setup onion --a random   # Phase 1 setup: one seat vs 3 Easy nations
.venv/Scripts/python -m trainer.ppo --run ppo-bc2 --setup ranked1v1 --init checkpoints/bc/update0060000.pt --opponent checkpoints/bc/update0060000.pt --kl-coef 0.2 --lr 1e-4 --ent-coef 0 --workers 16 --games-per-worker 2 --updates 5000   # PPO from the Gate A clone (settings: docs/PLAN.md, Phase 2 results)
.venv/Scripts/python -m trainer.eval --a checkpoints/ppo-bc2/update005000.pt --b checkpoints/bc/update0060000.pt --episodes 512 --workers 16 --games-per-worker 2   # Gate B: PPO vs the frozen clone (bar ~0.545), also per seat
.venv/Scripts/python -m trainer.eval --a checkpoints/ppo-bc2/update005000.pt --a-cooldown 1 --b checkpoints/bc/update0060000.pt --episodes 512 --workers 16 --games-per-worker 2   # bandwidth test (PLAN 3B): A's decision after each action is a no-op; every match prints A's meters
.venv/Scripts/python -m trainer.eval --a checkpoints/ppo-bc2/update005000.pt --a-two-bins --b checkpoints/bc/update0060000.pt --episodes 512 --workers 16 --games-per-worker 2   # its upward probe: A takes a troop and a gold action per decision
.venv/Scripts/python -m trainer.eval --a checkpoints/bc/update0060000.pt --b random --episodes 8 --record 1   # save games to replays/<A> vs <B>/, players named after their policies
.venv/Scripts/python -m trainer.watch --dir "bc 60000 vs random"   # watch one matchup's saved games: starts the game's dev client and serves them; open http://localhost:9000/game/<id>, Ctrl+C stops both
.venv/Scripts/python scripts/diag_baselines.py                  # random vs scripted expansion (Phase 1 diagnosis)
.venv/Scripts/tensorboard --logdir runs                         # charts, incl. actions/* mix and episodes/* ranks
```

- Claude Code kills its background tasks when the session ends, and a training run started as one dies with it. Start
  runs longer than a few minutes detached: a short Python launcher calling `subprocess.Popen([...], stdout=log,
  stderr=log, creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB)`, or the user's own
  terminal. Detached runs are still controlled via the page / `trainer.ctl`. They do not survive a session restart: a
  chain of eval matches started this way died when the app restarted the session (2026-10-06 02:28). Runs that must
  outlive the session go in the user's own terminal, or in a `.cmd` file started with `explorer.exe <file>.cmd`. That
  gives a normal console window whose parent is explorer.exe, outside the app (the bandwidth test ran this way,
  `runs/bandwidth.cmd`). `trainer.eval` saves every result to `runs/eval/<date>-<time>.json`.
- The PC's NVIDIA driver occasionally fails (System event nvlddmkm 153) and kills every CUDA program. Checkpoints are
  atomic, and the trainer exits 0 on stop, Save & quit or Ctrl+C and 1 on a crash, so long runs go in a PowerShell
  restart loop: `for ($i = 1; $i -le 5; $i++) { .venv/Scripts/python -m trainer.ppo --run <name> --resume --updates <N>;
  if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 60 }` (a new run starts once without `--resume`; it checkpoints
  every 25 updates).
- `--resume` keeps the run's settings and refuses flags it would ignore. It takes `--updates` (and, for PPO,
  `--kl-coef`) and this sitting's `--port`, `--duty` and `--gpu-temp-limit`. To change anything else, start a new
  run from the checkpoint with `--init`.
- BC: an update averages `--accum` replay windows (default 8), and the learning rate warms up, then decays (cosine)
  over `--updates` (default: one pass over the training records). BC checkpoints from before Phase 3 cannot resume;
  start from them with `--init`. Runs with the same `--seed` replay the same records in the same order.
- Networks: new runs record theirs in the checkpoint (`args["arch"]`, `"core"`); `--init` keeps the checkpoint's. A
  checkpoint without `"arch"` (Phase 2) loads as `PolicyV1`, which rebuilds its 22 input channels from today's 49
  (`v1_spatial`), so the Phase 2 bots still play.
- GPU memory: ranked PPO with a Phase 3 network needs `--minibatches 12` on the 12 GB card (peak ~10.9 GB). At the
  default 8 it overflows by 1.2–1.9 GB, Windows pages the rest into system RAM without an error, and the run gets 2–4×
  slower (watch "Shared GPU memory" in Task Manager).
- `trainer.bc --score` always scores the same rows: the held-out records of games started before 2026-10-01, 12 workers
  x 6 games, seed "score", 2,000 steps. Compare checkpoints within one scoring run (a purge can shift the rows).
- The training win-rate curve is a trend line, not a strength measure: it lags ~170 updates and includes the learner
  adapting to the games in progress (it read 0.84 where Gate B gave 0.63). Compare snapshots with `trainer.eval` on the
  same fixed seeds.
- Training runs and `trainer.watch` all serve on port 8787: stop `trainer.watch` before starting a run. A running
  trainer serves only the games saved directly in `replays/`, so `trainer.watch --dir` needs the port free. With the
  client running but nothing on 8787, the client cannot fetch the game and silently falls back to its lobby page.

## Simulation facts the bridge relies on

- 10 ticks = 1 s of game time. One turn is executed per tick.
- Stepping: `createGameRunner(gameStart, clientID, mapLoader, callback)` (`src/core/GameRunner.ts`), then per tick
  `runner.addTurn({ turnNumber, intents })` and `runner.executeNextTick()`. Intents are `StampedIntent`s
  (intent + the player's `clientID`), defined in `src/core/Schemas.ts`.
- Use `GameType.Private` or `Public` for training games: Singleplayer has no spawn timer.
- Determinism: the core uses `PseudoRandom` (never `Math.random`) and `src/core/DetMath.ts` instead of
  `Math.exp/log/pow/atan2`. Anything we add that feeds the simulation must follow the same rules.
- Ownership: `game.map().tileStateBuffer()` is a live `Uint16Array` (owner = `v & 0xFFF`, fallout = bit 13).
  Read it only between ticks. Each tick's changed tiles arrive as `[tileRef, state]` pairs in the runner callback's
  `packedTileUpdates`; update only the cells they touch (a full rebuild costs over half of the sim time).
- The game logs with `console.log`: a process whose stdout carries frames must redirect it to stderr.
- A game's RNG is seeded from its game ID (as in the game's client), so training games get a valid 8-character ID from
  their seed (`bridge/setups.ts: gameIdOf`) and `EnvGame.fromStart` builds env games and replayed records alike. A
  recorded game (`EnvGame.record/toRecord`) replays in the game's own client.
- All games of a trainer share one padded cell grid (`gridFor`: 192 x 256 for ranked 1v1). Cell indexes everywhere
  are grid indexes y x W + x; padding cells hold no tiles.
- Records store only non-empty turns: expand them with `decompressGameRecord` (`bridge/replay.ts: readRecord`).
- Replays: archived records contain only human intents; nations and bots re-simulate from the game seed. A record
  replays in sync when no simulation code changed between its `gitCommit` and the pin (v0.34.17-v0.34.23 are
  identical); the hashes it carries every 10 turns decide, not the commit string.
- v0.34.x has no game snapshots. `loadTerrainMap` caches one mutable map per map and size per process, so the bridge
  builds a fresh map for every game (`bridge/maps.ts: loadFreshTerrain`) and resets the game's cross-game static
  fields before each game (`resetGameStatics`); several games can then share a worker process. On every upgrade, grep
  `src/core` for non-readonly `private static` fields and add new ones to `resetGameStatics`.
- Code that only builds observations may use `Math.*`; code whose output reaches the simulation (intents, latency)
  must use seeded `PseudoRandom`.
- Structures (city, port, factory, silo, SAM, defense post) of all players stay at least `structureMinDist()` = 15
  tiles apart (`PlayerImpl.validStructureSpawnTiles`), so a cell (at most 8 × 8 tiles) holds at most one structure.
  The trainer's building channels and `v1_spatial` rely on it: recheck it on upgrades.
