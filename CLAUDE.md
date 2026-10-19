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

The submodule is pinned to the production release the official servers run (currently `v0.35.2`, commit `08b63c05`).
Check what production runs by reading `gitCommit` from a recent record:
`https://api.openfront.io/public/game/<id>?turns=false` (list games via `/public/games`, see the game's `docs/API.md`).
To upgrade, follow the playbook in `docs/PLAN.md` (decision 14): diff the engine between the tags (`packages/` since
v0.35, `src/core` before), move the submodule,
run the bridge tests and determinism check, then fine-tune from the latest checkpoint. A simulation change also needs
a replay fixture recorded at the new pin, and a worktree for the old simulation (below).

Records of an older simulation desync at the pin, so each older simulation keeps a worktree at its last tag in `pins/`
(git-ignored). `pins/v0.34.23` replays the records of v0.34.17-v0.34.23, which is all of `data/` today.
- It holds our bridge and trainer code on a detached HEAD, frozen before the v0.35 upgrade: later commits import the
  engine from `packages/`, which v0.34 lacks. It sits at bac9a8b with the user's uncommitted `trainer/eval.py` edits;
  0c0cb02 is the last commit it can run.
- Its game folder is a worktree of the game repo at the old tag.
- Junctions lead to the main checkout's `data`, `checkpoints`, `runs` and the game's `node_modules` (same runtime
  dependencies through v0.35.2; if a later pin changes them, give the old one its own `npm ci`).
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
.venv/Scripts/python scripts/coverage.py                    # Gate 3B's coverage: commands outside the 20 actions over all held-out records (seconds)
.venv/Scripts/python scripts/mode_analysis.py play          # how humans play each mode, per player-minute, over every record of data/v0.35.0/ (~2 min; PLAN "After 4N" item 2)
.venv/Scripts/python scripts/mode_analysis.py traces        # replays a stable sample of 600 records per mode (scripts/mode_traces.ts; ~2 h on 12 workers, resumable: detach, runs\modes035.cmd)
.venv/Scripts/python scripts/mode_analysis.py win           # what predicts winning, per mode and minute, from the traces (seconds; runs/modes/win.json); `selftest` checks its fit
.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12   # PPO decisions/s, time split and peak VRAM of the Phase 2 network and a v4 network, BC rows/s of a v5 network (~15 min, PC idle); --v4 <checkpoint> times a trained one (Gate 3B); --v5 also times a fresh v5 network in the same PPO setup, as a share of v4's decisions/s (step 6's budget: at least 95%, no spill)
.venv/Scripts/python scripts/profile_ppo.py                     # where PPO's time goes: one update of the real loop with the 3B clone under torch.profiler, per part: CPU, idle GPU, syncs, GPU time (~2 min, PC idle; runs/profile/*.json)
.venv/Scripts/python scripts/ab_speed.py --old pins/ab-base --label <change>   # code A/B of PPO speed: old, new, new, old probes and the rule's verdict (~15 min, PC idle; runs/ab/)
```

The A/B's old code lives in the worktree `pins/ab-base` (git-ignored), a detached HEAD at the commit before the change:
`git worktree add --detach pins/ab-base <commit>` (later `git -C pins/ab-base checkout --detach <commit>`), then
replace its empty `OpenFront_Source_Code/openfrontio` and add `checkpoints` and `runs` as PowerShell junctions
(`New-Item -ItemType Junction`) to this checkout's. Its game is this checkout's, so both commits of an A/B must come
after the v0.35 upgrade.

Python: `.venv` at the repo root, `pip install -r requirements.txt` (PyTorch CUDA 13.0 wheels).

## Training commands (repo root)

```bash
.venv/Scripts/python -m unittest discover -s trainer -t .      # trainer tests (start real Node workers, need the GPU)
.venv/Scripts/python -m trainer.ppo --run onion                 # train; control page with the key curves: http://127.0.0.1:8787
.venv/Scripts/python -m trainer.ppo --run onion --resume        # continue from checkpoints/onion/latest.pt
.venv/Scripts/python -m trainer.ctl pause|resume|save|stop|throttle half|status
.venv/Scripts/python -m trainer.crawl --modes ranked1v1 rankedffa ffa=0.25   # records of the pin's simulation into data/v0.35.0/; MODE=SHARE keeps that share of a mode's games, the default is ranked1v1 alone; --limit N per mode (hours: detach)
.venv/Scripts/python -m trainer.crawl --purge                  # remove deleted players' records from every store (daily while data is kept)
.venv/Scripts/python -m trainer.crawl --reindex                # refill the index fields step 7 added (playerTeams, modifiers, disabled units) for the entries already in data/v0.35.0; run it with no crawl writing the index, after the crawl ends
.venv/Scripts/python -m trainer.bc --run bc3-moe --core moe --workers 14 --games-per-worker 6   # behaviour cloning: one pass over data/ (~14 h); same control page. The next three lines ran before step 7, when trainer.bc read the old pin's ranked 1v1 store data/: they are the record of what ran, and today's trainer.bc does not read data/, so don't rerun them expecting it
.venv/Scripts/python -m trainer.bc --run bc3b --core moe --twin moe-swiglu --workers 14 --games-per-worker 6   # (before step 7) two new networks on the same windows (PLAN 3B): checkpoints/bc3b and checkpoints/bc3b-moe-swiglu
.venv/Scripts/python -m trainer.bc --run probe --init checkpoints/bc/update0060000.pt --updates 2000 --warmup 100   # (before step 7) fine-tune a checkpoint (its network, a fresh optimizer)
.venv/Scripts/python -m trainer.bc --run bc7 --init checkpoints/bc3b-v4/latest.pt --grow --lr 1e-4 --mix ranked1v1=3 ranked2v2=2 rankedffa=1 ffa=4 team=3 hvn=1 --games-per-worker 6 --updates 24000   # step 7's BC night (PLAN "After 4N" item 4; ~15 h): the 3B clone grown into v5, on the records of every mode in data/v0.35.0, one family per replay worker (after the crawl and --reindex); the user starts it with explorer.exe runs\bc7.cmd, which resumes the run created with --updates 0
.venv/Scripts/python -m trainer.bc --score checkpoints/bc3b-v4/latest.pt checkpoints/bc7/latest.pt   # NLL per head, bucket and lobby kind on the fixed v0.35 held-out set (~10 min); table + runs/score/*.json
.venv/Scripts/python -m trainer.eval --a checkpoints/bc/update0060000.pt --b scripted   # ranked 1v1 match, 256 games
.venv/Scripts/python -m trainer.eval --setup onion --a random   # Phase 1 setup: one seat vs 3 Easy nations
.venv/Scripts/python -m trainer.ppo --run ppo-bc2 --setup ranked1v1 --init checkpoints/bc/update0060000.pt --opponent checkpoints/bc/update0060000.pt --kl-coef 0.2 --lr 1e-4 --ent-coef 0 --workers 16 --games-per-worker 2 --updates 5000   # PPO from the Gate A clone (settings: docs/PLAN.md, Phase 2 results)
.venv/Scripts/python -m trainer.ppo --run ppo4a-s0 --setup ranked1v1 --init checkpoints/bc3b-v4/latest.pt --opponent checkpoints/bc3b-v4/latest.pt --kl-coef 0.2 --lr 1e-4 --ent-coef 0 --workers 16 --games-per-worker 2 --minibatches 12 --updates 3000   # 4A Stage 0 (PLAN "League design"): PPO from the 3B clone against its frozen self
.venv/Scripts/python -m trainer.ppo --run league4a2 --setup ranked1v1 --init checkpoints/swa-lr4a/update1000-2000.pt --kl-ref checkpoints/bc3b-v4/latest.pt --kl-coef 0.2 --lr 2.5e-5 --ent-coef 0 --workers 16 --games-per-worker 2 --minibatches 12 --updates 4000 --league checkpoints/swa-lr4a/update1000-2000.pt checkpoints/bc3b-v4/latest.pt checkpoints/bc3-moe/latest.pt checkpoints/ppo-bc2/update005000.pt checkpoints/bc/update0060000.pt --panel checkpoints/swa-lr4a/update1000-2000.pt checkpoints/bc3b-v4/latest.pt checkpoints/ppo-bc2/update005000.pt --eval-games 128   # league v2 (PLAN 4A, "League v2"): PFSP over a pool, frozen evaluations of the weight averages every 250 updates; launcher with a restart loop: explorer.exe runs\league4a2.cmd
.venv/Scripts/python -m trainer.ppo --run ffa4n --mix ranked1v1=8 ffa=4 hvn=2 --games-per-worker 1 --init checkpoints/league4a2/avg004000.pt --kl-ref checkpoints/bc3b-v4/latest.pt --kl-coef 0.2 --lr 2.5e-5 --ent-coef 0 --minibatches 12 --updates 4000 --league "checkpoints/league4a2/avg*.pt" checkpoints/swa-lr4a/update1000-2000.pt checkpoints/bc3b-v4/latest.pt checkpoints/ppo-bc2/update005000.pt --panel checkpoints/league4a2/avg004000.pt checkpoints/swa-lr4a/update1000-2000.pt checkpoints/bc3b-v4/latest.pt --eval-every 500 --eval-games 56 --eval-games-public 28   # the 4N run (PLAN 4N): 14 workers on three setups at once; its pool and panel are the plan's proposal (Task 8); the eval sizes are multiples of the 14 game slots (play() rounds up to a multiple of the slots). It failed (PLAN 4N "Results": mixing the setups cost 1v1 ~300 Elo), so don't rerun it as is
.venv/Scripts/python -m trainer.eval --a checkpoints/ppo-bc2/update005000.pt --b checkpoints/bc/update0060000.pt --episodes 512 --workers 16 --games-per-worker 2   # Gate B: PPO vs the frozen clone (bar ~0.545), also per seat
.venv/Scripts/python -m trainer.eval --a checkpoints/ppo-bc2/update005000.pt --a-cooldown 1 --b checkpoints/bc/update0060000.pt --episodes 512 --workers 16 --games-per-worker 2   # bandwidth test (PLAN 3B): A's decision after each action is a no-op; every match prints A's meters
.venv/Scripts/python -m trainer.eval --a checkpoints/ppo-bc2/update005000.pt --a-two-bins --b checkpoints/bc/update0060000.pt --episodes 512 --workers 16 --games-per-worker 2   # its upward probe: A takes a troop and a gold action per decision
.venv/Scripts/python -m trainer.eval --setup ffa --a <ckpt> --maps heldout --episodes 128 --workers 16 --games-per-worker 2   # Gate 4N (PLAN 4N): policy A in one FFA seat against the map's Hard nations, public rules (no timer), 128 games on the 4 held-out maps; then --maps seen (4 training maps: the gap between seen and unseen). Bar 0.50 on decided_win_rate (games won and not cut: a lead at the env's backstop is no win); the result JSON adds maps and cut_share (games the backstop ended)
.venv/Scripts/python -m trainer.eval --setup hvn --a <ckpt> --maps heldout --episodes 128 --workers 16 --games-per-worker 2   # Gate 4N: policy A in all 4 seats of the team against 4 Hard nations, held-out maps, no timer; one result per game (bar 0.70, on decided_win_rate). --b is not used. --difficulty and --timer change the nations and the timer (training's games: --maps training --timer 30; it trains against Medium or Hard nations)
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
  `--kl-coef`, `--eval-games`, `--eval-every` and `--eval-games-public`) and this sitting's `--port`, `--duty` and
  `--gpu-temp-limit`. To change anything else, start a new run from the checkpoint with `--init`.
- BC: an update averages `--accum` replay windows (default 8), and the learning rate warms up, then decays (cosine)
  over `--updates` (default: one pass over the training records). BC checkpoints from before Phase 3 cannot resume;
  start from them with `--init`. Runs with the same `--seed` replay the same records in the same order.
  - `--init <v4 checkpoint> --grow` trains that network grown into v5 (`grow()` in `trainer/model.py`): every trained
    v4 weight is kept, the new inputs get zero weights and the 5 new types a bias of -8, so the grown network starts as
    the v4 one (its logits for v4's 15 types and 12 maps, pointers, shares, value and router gates match). A resume
    loads the v5 checkpoint, without `--grow`.
  - `--mix FAMILY=WORKERS ...` gives each replay worker one family's records: `ranked1v1`, `ranked2v2`, `rankedffa`,
    `ffa`, `team` (with Clan Wars) and `hvn`. A family's share of the rows is its share of the workers. Without it,
    every worker plays ranked 1v1.
  - `--val-workers N` (N > 0) validates with one worker per family that has held-out records, 2 games each. A new run
    validates once at update 0: the baseline of the forgetting check.
- Networks: new runs record theirs in the checkpoint (`args["arch"]`, `"core"`): v1 is Phase 2, v2 is Phase 3A (5 troop
  sizes), v3 is Phase 3B (100 troop shares), v4 adds every action (15 types, 12 heat maps, a retreat pointer) and the
  3B inputs (57 channels, 26 scalars); `--init` keeps the checkpoint's. A checkpoint without `"arch"` (Phase 2) loads
  as `PolicyV1`, which rebuilds its 22 input channels from the first 49 of today's 57 (`v1_spatial`), so the Phase 2
  bots still play. v1–v3 see only their 5 types and 3 heat maps of the 20 (`ActionDist`), and only the 49 channels and
  16 scalars they were trained on; BC scores each network on the rows its types can express.
  - v5 (step 6) adds the diplomacy and warship actions: 20 types, 13 heat maps, four diplomacy pointers, a donate kind
    of the share head, and an autoregressive warship-move head (a source heat map, a destination map conditioned on the
    chosen source, alone or fleet). It also reads their inputs: 64 channels and 31 player features. New PPO and BC
    runs are v5. Step 7 adds the mode input: v5 reads 36 scalars, and v2–v4 keep reading their 26.
  - v2–v4 read only their own inputs and heads (`V4_CHANNELS`, `V4_FEATURES`, `V4_SCALARS`, `V4_ACTS`, `V4_MAPS` in
    `trainer/model.py`), so their checkpoints play unchanged. Later steps append inputs after v5's in the same way.
  - A v5 learner needs a v5 `--kl-ref`: there is no KL across architectures.
  - `RandomPolicy` plays v5's heads since step 6 (20 types, 1–100% shares), so results against `random` from before
    step 6 don't compare.
- Troops are a whole percent of current troops (1–100) in actions and labels. v1 and v2 networks send
  10/25/50/75/100%, exactly what their five bins sent. v3 and v4 networks read the share with a player's token: the attack's
  target, or the main owner of the boat's landing cell. v5 adds a donate kind, read with the recipient's token.
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
  the largest attack on the chosen target). Step 6 adds types 15–19:
  - an alliance request (which also accepts a pending one), renew and break, encoded as `[type, 0, smallID, 0]`;
  - donate troops, `[18, 0, smallID, share]`;
  - move warships, `[19, source cell, destination cell, fleet]`. The anchor is the seat's warship nearest the
    source cell's centre; fleet 1 adds every warship sharing its patrol point in its water body. The destination is
    the water tile of that body nearest the destination cell's centre.
  - The bridge masks types by gold, a ready silo outside spawn immunity, a finished port, an upgradable structure and
    an attack to cancel; the GPU masks the cells, the upgrade map exactly (own, finished, gold for its type).
  - Diplomacy has exact per-player masks from the game's own checks (the worker's "diplomacy" blob, all zero in the
    spawn phase and for a dead seat):
    - `canSendAllianceRequest`, `allianceInfo().canExtend` and `isAlliedWith`;
    - `canDonateTroops`, plus room under the recipient's troop cap.
    - A donation that would truncate to 0 troops drops.
  - The game's `canBuild` / `canUpgradeUnit` decide in the worker, and a failed action is a counted no-op. PPO logs
    each type's drop share as `actions/invalid_rate/<type>`.
  - Outside the set: gold donations, rejects, embargoes, target marks, chat and emoji, MIRVs, cancelling boats and
    deleting units.
- Observations (3B step 3, part 2):
  - Unit records carry the missiles loaded in silos and SAMs, and each unit's zone in tiles from the game's Config:
    a finished SAM's current range, a finished defense post's range, a nuke's outer blast around its target.
  - The GPU paints 8 channels after the 49 (`disk_cells`): each structure's level and loaded missiles (/ 16; level 0
    while under construction), and the SAM, defense-post and nuke zones of each side. Ours is self and friends;
    theirs is enemies and bots.
  - Scalars 16–25: the gold against each build's current cost, and the seat's loaded silo missiles.
- Observations (step 6, v5):
  - Unit records are 22 bytes.
    - A warship's troops field holds its health ÷ max health, `ready` its veteran level, and its zone its patrol tile.
    - A trade ship carries its safe-from-pirates flag and the smallID of its destination port's owner.
  - Channels 57–63:
    - warship health and veteran levels, ours and theirs;
    - our patrol points;
    - trade ships that pay us or our friends;
    - trade ships we could capture.
  - Player features 25–30, after the 8 building features: ally, the two pending requests, the alliance's time left,
    and who asked to renew.
  - `cell_mask` is [N, 14, H·W]: the 13 heat maps, then the warship destination (`DST`, ocean cells).
    `diplomacy_mask` is [N, 4, P].
- Observations (step 7, the mode input; scalars 26–35):
  - The lobby kind, one-hot (26–32: ranked 1v1, ranked 2v2, Ranked FFA, Clan Wars, FFA, Team, Humans vs Nations), from
    the config's ranked type, mode and teams; then log2(non-bot players) / 8, log2(teams) / 6 (0 in the FFA modes) and
    the nations' difficulty (Easy 0.25 to Impossible 1, 0 without nations).
  - A replayed record and a training game of the same mode give the same values; the training setups are `ranked1v1`,
    `ffa` and `hvn`. Item 5 has to know one mismatch: the FFA training setup has 4 seats, where public lobbies have
    ~25 humans (the players input tells them apart).
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
- `trainer.bc --score` always scores the same rows: the usable held-out (5% by game ID) records of `data/v0.35.0`
  started before 2026-10-19 12:00 UTC, each family on min(2, its records) workers x 6 games, seed "score", 1,500 steps
  (`SCORE_CUTOFF`, `SCORE_LAYOUT` in `trainer/bc.py`). Every statistic is also logged per lobby kind. The result JSON
  adds the records per family. Compare checkpoints within one scoring run (a purge can shift the rows).
  - `bc/nll_common*` (and `bc/nll_common_<kind>`) is the NLL over the legal rows labelled with one of v4's 15 types: the
    like-for-like key between a v4 and a v5 network.
  - `labels/illegal/<type>` is the share of the rows labelled with that type that fall outside the masks.
  - The cell, target and destination heads come with uniform baselines (`bc/nll_<map>_cell_uniform`,
    `bc/nll_<type>_target_uniform`, `bc/nll_warship_dst_uniform`). The donation share's baseline is log 100 = 4.605.
  - The old set (games before 2026-10-01 in `data/`, 12 workers x 6 games, 2,000 steps) can be scored only from
    `pins/v0.34.23`.
- Records live in one store per simulation (`trainer/crawl.py`), since they desync at a pin with another one. Each
  store holds `records/`, `index.jsonl` (one entry per record, with its mode) and `skipped.txt`.
  - `data/` holds the old pin's records, v0.34.17-v0.34.22. They replay in full only in `pins/v0.34.23` ("Game
    version"), where Step 4's BC night and its scoring ran (user decision 2026-10-10, PLAN 3B).
  - Until step 7, `trainer.bc` read only `data/` (ranked 1v1); the old score set is valid only from the pin.
  - `data/v0.35.0/` holds v0.35.0-v0.35.2 records of every mode, which replay in this checkout. The crawler fills it
    and skips records of other game commits (`SIM_COMMITS`). `trainer.bc` reads it by family (step 7, PLAN "After 4N"
    item 4) and keeps the records whose map fits the specialists' grid (`bridge/maptable.ts`,
    `trainer.env.map_table()`, [192, 256]) and that set no modifier but Compact and disable no unit.
    - Its `index.jsonl` needs `python -m trainer.crawl --reindex` once, for the fields the filters read; `trainer.bc`
      and `--score` stop with that message until then.
    - `scripts/coverage.py` keeps the old store's held-out set (games before 2026-10-01): it is Gate 3B's coverage
      measure. The v0.35 coverage per mode is `mode_analysis`'s out-of-set row.
- Replays in BC (step 7): a record imitates two players a game, not every human.
  - The best-placed human of each of the two best-placed sides (players in the FFA modes, teams in the team modes, the
    humans as one side in Humans vs Nations), the winning side first. Placement: alive at the end, then a later
    `killedAt`, then more peak tiles (`stats.tiles[0]`); a human is eligible with peak tiles above 0. A record with
    fewer than two eligible humans is skipped by its worker.
  - Team wins count: a seat on the winning side (`["team", name, ...]`, with `name` as `team()` returns it) is +1, every
    other seat -1; a nations' win (`["nation", ...]` or `["team", "Nations"]`) is -1 for both players.
  - A window where the seat's player is disconnected (`isDisconnected()`) labels UNMAPPED, so the policy loss skips it
    (it would teach "do nothing"); the value loss keeps the row.
  - Elo (`--min-elo`, `crawl.elo`) is computed over the ranked 1v1 entries only.
  - The player table fits the run's largest game (`slots_of`, over every usable record of the store); `bridge/obs.ts`
    throws when a smallID does not fit.
  - On an upgrade that changes the simulation, give the crawler the new `SIM`, `SIM_COMMITS` and `SINCE`; one that
    does not only adds its commit to `SIM_COMMITS`.
- Mixed runs (`--mix SETUP=WORKERS ...`, PLAN 4N): workers play the setups in the order given; rows are game-major
  (game e owns `first[e]` .. `first[e] + seats[e] - 1`). The learner has one seat of each ranked game (the league's pick
  the other) and every seat of an FFA or Humans vs Nations game (`WHOLE` in ppo.py); `global_step` counts learner rows.
  - A mixed run couples the setups' training signals in three ways: one value head, advantages normalised over mixed
    minibatches, and one summed gradient. The 4N run lost ~300 Elo in 1v1 that way (PLAN 4N, "Results").
  - Modes now train as separate specialists (PLAN decision 5).
  - Before setups share a PPO run again, add per-setup advantage normalisation and value targets, and play the control
    run that PLAN 4N describes.
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
    resume. It counts the whole panel's games: 128 over three members gives each ~43, so a single point swings ±0.07;
    pool several evaluations before reading a trend.
- The training win-rate curve is a trend line, not a strength measure: it lags ~170 updates and can be off either way.
  It read 0.84 where Gate B gave 0.63 (Phase 2: the learner adapting to the games in progress). In 4A Stage 0 it read
  0.57 where the frozen check gave 0.74, but that snapshot was a lucky one: at lr 1e-4 Stage 0's snapshots swung from
  0.20 to 0.77 against the clone, most near 0.5 (PLAN 4A, "Learning test"). Compare snapshots with `trainer.eval` on
  the same fixed seeds, and never judge a run by a single snapshot. In a league it mostly tracks PFSP's current
  opponent: league v2's read 0.17–0.59 while its averages beat the start 0.66 (PLAN 4A, "League v2").
- PPO with a v4 network runs at lr 2.5e-5: at 1e-4 its updates were mostly noise, so snapshot strength swung far
  beyond the evaluations' noise (PLAN 4A, "Learning test").
- Training runs and `trainer.watch` all serve on port 8787: stop `trainer.watch` before starting a run. A running
  trainer serves only the games saved directly in `replays/`, so `trainer.watch --dir` needs the port free. With the
  client running but nothing on 8787, the client cannot fetch the game and silently falls back to its lobby page.

## Simulation facts the bridge relies on

- 10 ticks = 1 s of game time. One turn is executed per tick.
- Stepping: `createGameRunner(gameStart, clientID, mapFiles, callback)` (`packages/engine/src/GameRunner.ts`), then per
  tick
  `runner.addTurn({ turnNumber, intents })` and `runner.executeNextTick()`. Intents are `StampedIntent`s
  (intent + the player's `clientID`), defined in `packages/engine-api/src/Schemas.ts`.
- An intent's execution starts on the tick after the one that receives it: a built unit or a launched attack exists
  one tick later, so tests tick once more before reading it.
- Use `GameType.Private` or `Public` for training games: Singleplayer has no spawn timer.
- Determinism: the core uses `PseudoRandom` (never `Math.random`) and `packages/engine-lib/src/DetMath.ts` instead of
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
  older records desync there within ~500 turns (median): replay them in `pins/v0.34.23`. v0.35 changed it again
  (boat paths, trade ships and trains, the nations' AI), so only v0.35.2 records replay at the pin; no v0.34.24 records
  were crawled.
- `loadTerrainMap` takes over the map files it is given: a map writes to its terrain bytes, and compact maps scale the
  manifest in place. So the bridge passes fresh copies for every game (`bridge/maps.ts: loadFreshTerrain`).
- v0.35 left no mutable static fields in the engine (v0.34's needed a reset before each game), so several games share
  a worker process. On every upgrade, grep `packages/` for non-readonly `static` fields and module-level state; a new
  one needs a reset before each game.
- v0.35 has game snapshots (`GameRunner.snapshot`, `createGameRunnerFromSnapshot`).
- A conquest (a player's last land taken) arrives as a `ConquestEvent` update, which `EnvGame.conquests` keeps. The
  game's stats skip a conquered human who never attacked anyone (no gold moves), so they count one kill fewer there.
- Code that only builds observations may use `Math.*`; code whose output reaches the simulation (intents, latency)
  must use seeded `PseudoRandom`.
- Structures (city, port, factory, silo, SAM, defense post) of all players stay at least `structureMinDist()` = 15
  tiles apart (`PlayerImpl.validStructureSpawnTiles`), so a cell (at most 8 × 8 tiles) holds at most one structure.
  The trainer's building channels and `v1_spatial` rely on it: recheck it on upgrades.
- Every map has one ocean water body: the map generator marks only the largest as ocean, and the others are lakes. The
  warship destination mask is ocean-only, so a lake warship's moves drop. The bot builds warships on ocean cells only.
- Tribes (bots) accept every alliance request, renewals too. When a public lobby switches donations off, only
  donations to humans are refused, so an allied nation still receives them.
- A replay window's masks are read at its first tick, and the game checks a command when it executes, a few ticks
  later. So a human command sent right as a cooldown ends can label illegal: one donation of 21 in the Team fixture
  did.
