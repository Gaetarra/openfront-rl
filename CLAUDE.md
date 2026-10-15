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

The submodule is pinned to the production release the official servers run (currently `v0.34.24`, commit `3f5633b9`).
Check what production runs by reading `gitCommit` from a recent record:
`https://api.openfront.io/public/game/<id>?turns=false` (list games via `/public/games`, see the game's `docs/API.md`).
To upgrade, follow the playbook in `docs/PLAN.md` (decision 14): diff `src/core` between the tags, move the submodule,
run the bridge tests and determinism check, then fine-tune from the latest checkpoint. A simulation change also needs
a replay fixture recorded at the new pin, and a worktree for the old simulation (below).

Records of an older simulation desync at the pin, so each older simulation keeps a worktree at its last tag in `pins/`
(git-ignored). `pins/v0.34.23` replays the records of v0.34.17-v0.34.23, which is all of `data/` today.
- It holds the same bridge and trainer code on a detached HEAD. After code changes, update it with
  `git -C pins/v0.34.23 checkout --detach <branch>`.
- Its game folder is a worktree of the game repo at the old tag.
- Junctions lead to the main checkout's `data`, `checkpoints`, `runs` and the game's `node_modules` (same
  dependencies; if a later pin changes them, give the old one its own `npm ci`).
- Run trainer commands from it with the main venv: `cd pins/v0.34.23`, then `../../.venv/Scripts/python -m trainer.bc
  ...`. Checkpoints record the game commit they were trained on.
- Made with `git worktree add --detach pins/<tag> <commit>`, then `git -C OpenFront_Source_Code/openfrontio worktree
  add --detach <absolute path>/pins/<tag>/OpenFront_Source_Code/openfrontio <tag>`, then a PowerShell
  `New-Item -ItemType Junction` for each of the four links.

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
.venv/Scripts/python scripts/coverage.py                    # Gate 3B's coverage: commands outside the 15 actions over all held-out records (seconds)
.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12   # PPO decisions/s, time split and peak VRAM of the Phase 2 network and a v4 network, BC rows/s of the v4 network (~15 min, PC idle); --v4 <checkpoint> times a trained one (Gate 3B)
.venv/Scripts/python scripts/profile_ppo.py                     # where PPO's time goes: one update of the real loop with the 3B clone under torch.profiler, per part: CPU, idle GPU, syncs, GPU time (~2 min, PC idle; runs/profile/*.json)
.venv/Scripts/python scripts/ab_speed.py --old pins/ab-base --label <change>   # code A/B of PPO speed: old, new, new, old probes and the rule's verdict (~15 min, PC idle; runs/ab/)
```

The A/B's old code lives in the worktree `pins/ab-base` (git-ignored), a detached HEAD at the commit before the change:
`git worktree add --detach pins/ab-base <commit>` (later `git -C pins/ab-base checkout --detach <commit>`), then
replace its empty `OpenFront_Source_Code/openfrontio` and add `checkpoints` and `runs` as PowerShell junctions
(`New-Item -ItemType Junction`) to this checkout's.

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
.venv/Scripts/python -m trainer.bc --run bc3b --core moe --twin moe-swiglu --workers 14 --games-per-worker 6   # two new networks on the same windows (PLAN 3B): checkpoints/bc3b and checkpoints/bc3b-moe-swiglu
.venv/Scripts/python -m trainer.bc --run probe --init checkpoints/bc/update0060000.pt --updates 2000 --warmup 100   # fine-tune a checkpoint (its network, a fresh optimizer)
.venv/Scripts/python -m trainer.bc --score checkpoints/bc/update0060000.pt checkpoints/bc3-moe/latest.pt   # NLL per head and bucket on the fixed held-out set (~10 min); table + runs/score/*.json
.venv/Scripts/python -m trainer.eval --a checkpoints/bc/update0060000.pt --b scripted   # ranked 1v1 match, 256 games
.venv/Scripts/python -m trainer.eval --setup onion --a random   # Phase 1 setup: one seat vs 3 Easy nations
.venv/Scripts/python -m trainer.ppo --run ppo-bc2 --setup ranked1v1 --init checkpoints/bc/update0060000.pt --opponent checkpoints/bc/update0060000.pt --kl-coef 0.2 --lr 1e-4 --ent-coef 0 --workers 16 --games-per-worker 2 --updates 5000   # PPO from the Gate A clone (settings: docs/PLAN.md, Phase 2 results)
.venv/Scripts/python -m trainer.ppo --run ppo4a-s0 --setup ranked1v1 --init checkpoints/bc3b-v4/latest.pt --opponent checkpoints/bc3b-v4/latest.pt --kl-coef 0.2 --lr 1e-4 --ent-coef 0 --workers 16 --games-per-worker 2 --minibatches 12 --updates 3000   # 4A Stage 0 (PLAN "League design"): PPO from the 3B clone against its frozen self
.venv/Scripts/python -m trainer.ppo --run league4a --setup ranked1v1 --init checkpoints/ppo4a-s0/update001000.pt --kl-ref checkpoints/bc3b-v4/latest.pt --kl-coef 0.2 --lr 1e-4 --ent-coef 0 --workers 16 --games-per-worker 2 --minibatches 12 --updates 10000 --league checkpoints/bc3b-v4/latest.pt checkpoints/bc3-moe/latest.pt checkpoints/ppo-bc2/update005000.pt checkpoints/bc/update0060000.pt "checkpoints/ppo4a-s0/update*.pt" --panel checkpoints/bc3b-v4/latest.pt checkpoints/bc/update0060000.pt checkpoints/ppo-bc2/update005000.pt --eval-games 128   # the 4A league (PLAN "League design"): PFSP over a pool, frozen evaluations against the panel every 250 updates
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
- The PC's NVIDIA driver occasionally fails (System event nvlddmkm 153) and kills every CUDA program (bc3b died this
  way at update ~5,225, 2026-10-06 23:40). Checkpoints are atomic, and the trainer exits 0 on stop, Save & quit or
  Ctrl+C and 1 on a crash, so long runs go in one PowerShell loop that starts the run or resumes it, safe to rerun:
  `for ($i = 1; $i -le 6; $i++) { if (Test-Path checkpoints/<name>/latest.pt) { <trainer> --run <name> --resume }
  else { <trainer> --run <name> <flags> }; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 60 }`.
  - A new run refuses a folder that already holds a run.
  - A resume hides from the curves what the crashed sitting logged after its last checkpoint.
  - BC saves every 250 updates and keeps every 2,500th; PPO saves every 25.
- `--resume` keeps the run's settings and refuses flags it would ignore. It takes `--updates` (and, for PPO,
  `--kl-coef`) and this sitting's `--port`, `--duty` and `--gpu-temp-limit`. To change anything else, start a new
  run from the checkpoint with `--init`.
- BC: an update averages `--accum` replay windows (default 8), and the learning rate warms up, then decays (cosine)
  over `--updates` (default: one pass over the training records). BC checkpoints from before Phase 3 cannot resume;
  start from them with `--init`. Runs with the same `--seed` replay the same records in the same order.
- Networks: new runs record theirs in the checkpoint (`args["arch"]`, `"core"`): v1 is Phase 2, v2 is Phase 3A (5 troop
  sizes), v3 is Phase 3B (100 troop shares), v4 adds every action (15 types, 12 heat maps, a retreat pointer) and the
  3B inputs (57 channels, 26 scalars); `--init` keeps the checkpoint's. A checkpoint without `"arch"` (Phase 2) loads
  as `PolicyV1`, which rebuilds its 22 input channels from the first 49 of today's 57 (`v1_spatial`), so the Phase 2
  bots still play. v1–v3 see only their 5 types and 3 heat maps of the 15 (`ActionDist`), and only the 49 channels and
  16 scalars they were trained on; BC scores each network on the rows its types can express.
- Troops are a whole percent of current troops (1–100) in actions and labels. v1 and v2 networks send
  10/25/50/75/100%, exactly what their five bins sent. v3 and v4 networks read the share with a player's token: the attack's
  target, or the main owner of the boat's landing cell.
  - BC labels the human's share, rounded and clamped to 1–100. The loss puts 0.05 on each neighbouring percent
    (`SHARE_SMOOTH`).
  - In `--score`, `bc/nll` is like for like (a share network's shares summed into the five sizes' ranges, 1–15, 16–35,
    36–61, 62–86 and 87–100%), and `bc/nll_shares` is a share network's own NLL.
  - The cell heads are also scored by closeness (the user's idea, 2026-10-07), since a near miss is almost as good as
    a hit. `bc/*_cell_near` is the chance the bot's cell lands within 16 tiles of the human's, and `bc/*_cell_dist`
    is its expected distance in tiles. `*_uniform` gives each one's baseline, a uniform pick over the legal cells.
  - `--twin CORE` trains a second network on the same windows, from the same weights outside the core. It saves to
    `checkpoints/<run>-<core>`, and the run resumes both together.
- Actions (bridge/actions.ts `Act`): no-op, spawn, attack, boat, build city / defense post / silo / factory / port /
  SAM / warship, atom bomb, hydrogen bomb, upgrade (the seat's structure in the chosen cell) and retreat (cancels
  the largest attack on the chosen target).
  - The bridge masks types by gold, a ready silo outside spawn immunity, a finished port, an upgradable structure and
    an attack to cancel; the GPU masks the cells, the upgrade map exactly (own, finished, gold for its type).
  - The game's `canBuild` / `canUpgradeUnit` decide in the worker, and a failed action is a counted no-op.
  - Outside the set: diplomacy, chat, warship moves and MIRVs.
- Observations (3B step 3, part 2):
  - Unit records carry the missiles loaded in silos and SAMs, and each unit's zone in tiles from the game's Config:
    a finished SAM's current range, a finished defense post's range, a nuke's outer blast around its target.
  - The GPU paints 8 channels after the 49 (`disk_cells`): each structure's level and loaded missiles (/ 16; level 0
    while under construction), and the SAM, defense-post and nuke zones of each side. Ours is self and friends;
    theirs is enemies and bots.
  - Scalars 16–25: the gold against each build's current cost, and the seat's loaded silo missiles.
- GPU memory (12 GB card):
  - PPO keeps its rollout rows in pinned system RAM (`HostRows`), with the cell masks as bits (`pack_bits`). A worker
    thread copies each minibatch to the GPU on its own stream while the previous one trains.
    - On the ranked grid with the KL reference's outputs, it locks 7.6 GiB of RAM (pinned pages cannot be paged out).
      PyTorch's caching host allocator rounds pinned buffers up to a power of two (12.9 GiB here), so the trainers'
      allocator config skips the rounding above 1 MB (`trainer/__init__.py`).
    - Its worker thread lives as long as the run. A thread's first parallel CPU op starts an OpenMP team that outlives
      the thread, and a new thread per update leaked ~21 GiB over the first league run. Run CPU torch work only on
      threads that live as long as the process.
    - On Windows, a process's private memory also counts its GPU memory, so the trainer's figure in Task Manager is not
      all RAM. PPO logs it every update as `perf/private_gb` (PLAN 4A, "First league run").
  - The trainers run with expandable allocator segments (`trainer/__init__.py`), and PPO empties the cache before
    learning.
  - Ranked PPO with a Phase 3 network runs at `--minibatches 12`. 96- and 128-row minibatches cost the same per row,
    and 192 rows (the default 8) cost 49% more. At 12, v4 with every 3B input holds ~8.7 GB of the card, 6.0 GB of it
    in use.
  - When the card overflows, Windows pages the rest into system RAM without an error, and the run gets 2–4× slower
    (watch "Shared GPU memory" in Task Manager).
  - PPO logs `perf/gpu_reserved_gb` (the allocator's peak, which reads past the card when it spills),
    `perf/gpu_allocated_gb` (in use), and the time split `perf/rollout_share` and `perf/env_share` (the workers'
    simulation and observations).
- PPO speed (PLAN 4A, "Speed work"; `scripts/profile_ppo.py` shows where the time goes):
  - Learning is GPU-bound. `.item()` and a copy from pageable memory (`torch.from_numpy(...).to(device)`, a constant
    CPU tensor, a Python list as an index of a GPU tensor) wait for everything queued on the GPU, so keep them out of
    a minibatch's work (`_on` in `trainer/model.py` keeps constants on the GPU).
  - Keep CPU torch work out of the rollouts: torch's OpenMP pool (14 threads) spins for up to ~200 ms after each
    parallel op and takes the cores from the workers' simulation (lever 2 lost 14% that way).
  - Absolute speeds swing ±15% with the PC's state (a PC cold after hours idle runs fast at first): credit a speed
    change only with `scripts/ab_speed.py`.
- `trainer.bc --score` always scores the same rows: the held-out records of games started before 2026-10-01, 12 workers
  x 6 games, seed "score", 2,000 steps. Compare checkpoints within one scoring run (a purge can shift the rows).
- `data/` holds only records of v0.34.17-v0.34.22, which replay in full only in `pins/v0.34.23` ("Game version").
  Step 4's BC night and its scoring ran from there (user decision 2026-10-10, PLAN 3B). `trainer.bc` takes every
  indexed record, so after a crawl `data/` mixes in v0.34.24 records: a BC run on both needs them split by version
  (PLAN 3B, "Step 4's records", option 3). `--score` stays valid from the pin (its records predate 2026-10-01).
- The league (PLAN 4A, "League design"):
  - `--league` takes checkpoints and globs. Quote the globs: Python expands them.
  - Every 250 updates PPO also saves `avg<update>.pt`, the average of the weights of the last 5 kept checkpoints
    (1,000 updates; `--init` counts as update 0). The averages, not the raw snapshots, join the pool and play the
    frozen evaluations: PPO's snapshots scatter around a stronger mean policy (PLAN 4A, "Checks before the next
    league").
  - The first `--panel` checkpoint anchors the Bradley-Terry rating at 0 Elo.
  - Each frozen evaluation is appended to `runs/<run>/panel.jsonl`, and a new run refuses a `runs/<run>` that already
    holds one.
  - Training waits during an evaluation: ~18 min at the default 128 games, ~9 min at 64. `--eval-games` can change on a
    resume.
- The training win-rate curve is a trend line, not a strength measure: it lags ~170 updates and can be off either way.
  It read 0.84 where Gate B gave 0.63 (Phase 2: the learner adapting to the games in progress). In 4A Stage 0 it read
  0.57 where the frozen check gave 0.74, but that snapshot was a lucky one: at lr 1e-4 Stage 0's snapshots swung from
  0.20 to 0.77 against the clone, most near 0.5 (PLAN 4A, "Learning test"). Compare snapshots with `trainer.eval` on
  the same fixed seeds, and never judge a run by a single snapshot.
- PPO with a v4 network runs at lr 2.5e-5: at 1e-4 its updates were mostly noise, so snapshot strength swung far
  beyond the evaluations' noise (PLAN 4A, "Learning test").
- Training runs and `trainer.watch` all serve on port 8787: stop `trainer.watch` before starting a run. A running
  trainer serves only the games saved directly in `replays/`, so `trainer.watch --dir` needs the port free. With the
  client running but nothing on 8787, the client cannot fetch the game and silently falls back to its lobby page.

## Simulation facts the bridge relies on

- 10 ticks = 1 s of game time. One turn is executed per tick.
- Stepping: `createGameRunner(gameStart, clientID, mapLoader, callback)` (`src/core/GameRunner.ts`), then per tick
  `runner.addTurn({ turnNumber, intents })` and `runner.executeNextTick()`. Intents are `StampedIntent`s
  (intent + the player's `clientID`), defined in `src/core/Schemas.ts`.
- An intent's execution starts on the tick after the one that receives it: a built unit or a launched attack exists
  one tick later, so tests tick once more before reading it.
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
  identical); the hashes it carries every 10 turns decide, not the commit string. v0.34.24 changed attacks, and
  older records desync there within ~500 turns (median): replay them in `pins/v0.34.23`.
- v0.34.x has no game snapshots. `loadTerrainMap` caches one mutable map per map and size per process, so the bridge
  builds a fresh map for every game (`bridge/maps.ts: loadFreshTerrain`) and resets the game's cross-game static
  fields before each game (`resetGameStatics`); several games can then share a worker process. On every upgrade, grep
  `src/core` for non-readonly `private static` fields and add new ones to `resetGameStatics`.
- Code that only builds observations may use `Math.*`; code whose output reaches the simulation (intents, latency)
  must use seeded `PseudoRandom`.
- Structures (city, port, factory, silo, SAM, defense post) of all players stay at least `structureMinDist()` = 15
  tiles apart (`PlayerImpl.validStructureSpawnTiles`), so a cell (at most 8 × 8 tiles) holds at most one structure.
  The trainer's building channels and `v1_spatial` rely on it: recheck it on upgrades.
