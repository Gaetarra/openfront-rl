# OpenFront DRL Bot — Project Plan

## Context
Goal: train the strongest possible deep-RL bot for OpenFront — Ranked 1v1/2v2, public FFA and Team, with modifiers and
unseen maps — on a student budget: one PC (i5-13600KF 14C/20T, 32 GB RAM, RTX 4070 Ti 12 GB, Windows 11 without WSL,
Node 24, Python 3.11 with PyTorch 2.14 cu130). User decisions: bootstrap from human replays; reward = terminal
rank-based only (shaping only if learning stalls); final use = private lobbies with consenting humans and tests in
official casual lobbies. Upstream moves fast (~9 commits/day), so the game is pinned to the current production
release and the base model follows each new release.
**Maintainers (OpenFront owner on Discord, 2026-09-30):** using the public API for training is fine (no data dump:
we crawl it ourselves); the bot may be tested in casual multiplayer games (not ranked, no abuse); the user's account
has an AI-DEV tag.

## Ambition ladder (honest expectations)
| Level | Target | Outlook |
|---|---|---|
| L1 | Beats random/scripted bots and Easy nations | very likely (Phase 3–4) |
| L2 | Consistently beats Impossible nations (FFA) | likely: scripted AIs are formulaic and RL exploits them |
| L3 | Beats average ranked humans in 1v1/2v2 | plausible after a few months of BC + self-play |
| L4 | Beats top human players consistently | stretch goal: where AlphaStar-scale league breadth mattered |
OpenFront is far simpler than StarCraft II / Dota 2 (no fog of war, ~20 mostly strategic action types, units act on their
own, cheap deterministic sim we control, 10–15 min ranked games), so the plan uses scaled-down versions of the same
ingredients (human BC, self-play league, PPO, careful obs/action engineering). Rough experience budget: ~1k decisions/s ×
0.5 s ≈ 1.4 years of game time per day ≈ 40 years/month, vs ~200 years per agent for AlphaStar 2019.

## Findings that shape the design
- Headless sim runs in Node today: `createGameRunner` (src/core/GameRunner.ts:40), `tests/replay/ReplayGame.ts`,
  `tests/perf/fullgame/FullGamePerf.ts`. ≈1.07 ms/tick late-game World Normal with 261 AI players. 10 ticks = 1 s.
- **Production runs release tags `v0.34.x` (branch `origin/v34`), not `main`** (split 2026-09-18; 16 core commits on
  main, 6 on v34). Snapshots (`src/core/snapshot`) and `fresh` per-game maps exist only on main.
- No fog of war → state is essentially Markov → no LSTM.
- Ownership: zero-copy `Uint16Array` `game.map().tileStateBuffer()` (owner = v & 0xFFF, fallout bit 13, defense bit 14),
  live during a tick. Moving units keep `unit.tile()` current every tick.
- `attack` = {targetID | null, absolute troops}: player-level. Boats/builds/nukes/warships take a tile.
- SAM interception is deterministic. `build_unit.amount` (1–50) makes salvos native; `rocketDirectionUp` flips the arc.
- **Ranked** (src/server/MapPlaylist.ts:490-551): 1v1 = FFA with 2 players, 2v2 = 2 teams with donations; maps
  Australia 40%, Iceland/Asia/EuropeClassic 20% each; Compact 20% (1v1) / 50% (2v2); nations disabled; 100/400 bots;
  timer 10 min Compact / 15 Normal; spawn immunity 30/60 s. Win = last human/team standing, else 80% land or tile leader
  at the timer (WinCheckExecution.ts:96-105, 147+). A 2v2 is void unless all 4 spawn.
- **Public games**: FFA/Team 50/50; teams 2–7, Duos/Trios/Quads, Humans-vs-Nations (MapPlaylist.ts:44-54); up to 3
  modifiers from a weighted pool: random spawn, compact, crowded, hard nations, starting gold 1M/5M/25M, gold multiplier,
  alliances/nukes/SAMs disabled, peace time, water nukes, doomsday clock; plus per-map forced/disabled modifiers
  (MapPlaylist.ts:68-118, 270-330).
- Maps enter the sim only through `GameMapLoader.getMapData` → {mapBin, map4xBin, map16xBin, manifest}
  (src/core/game/GameMapLoader.ts:4-16), so a custom loader can serve augmented/procedural maps from memory; `gameMap`
  otherwise only picks tribe-name themes. Official maps are built from PNGs by the Go tool `map-generator/` (Go not installed).
- Replays: `GET api.openfront.io/public/games` (≤2-day windows, ≤1000/req, filters type/mode/rankedType) +
  `/public/game/:id` (`?turns=false` for metadata). Only human intents are recorded; multiplayer records carry hashes
  every 10 turns (sync check). Deleted-players endpoint must be honoured.
- Official servers require a Turnstile anti-bot token on first join (src/server/JoinVerify.ts) → self-hosted server.
- Stats give each clientID player a `deathPosition` (StatsImpl.ts:463-467) → agent rank is built in.

## Key decisions
1. **No C++ port.** Measured in Phase 0: PPO learner ≈0.7–1k decisions/s (~6M-param net), while ~10 cores of TS sim
   give ≈2k (Normal) to ≈7k (Compact) ranked 1v1 decisions/s, so the GPU is the bottleneck. A port would break BC
   (replays need the exact TS sim) and can't track upstream.
2. **CNN on the whole map**, fixed cell grid: 1 cell = 8×8 tiles Normal = 4×4 Compact (same geography → same grid;
   World = 250×125). Plus a player-entity transformer and a scalar MLP. Fully convolutional with padding masks,
   batches grouped by map, tiles-per-cell input, range channels in tile units. Start with maps ≤ ~100k cells.
3. **Coordinates = spatial categorical heat map, not regression.** One logit per cell, illegal cells masked to −∞,
   softmax, sample (DeepMind SC2-style). Multimodal-safe, rule-enforcing (build: own land/coast; warship: connected
   water; boat: target's shore; spawn: unowned land; nuke: not teammates), explores naturally. Conditioned (FiLM) on the
   action/unit type. Inside the cell a rule picks the tile (engine snaps structures ≤15 tiles, boats ≤50).
   Autoregressive: type → unit type → player pointer | cell → modifiers; log-prob = sum; entropy per head.
4. **Missiles: the policy never guesses.** SAM-coverage channels first; then an exact interception predictor (copy of
   `SAMTargetingSystem` driving a mini-sim of SAMs + nukes) → per cell and arc: interceptor slots on the path.
5. **Many different situations** (spawn, early/mid/late, many vs few players, modifiers, ranked vs public) at 3 levels:
   a. everything is an input: phase, alive players, mode, ranked type, every modifier as scalars;
   b. rule-routed specialisation where decisions differ: a dedicated spawn head; optional per-mode specialist checkpoints
      (e.g. a ranked-1v1 fine-tune) picked at deployment by lobby type;
   c. learned MoE: placed after the encoders (the router needs the CNN/player/scalar features), over the fused vector or
      the 1/16-res spatial + player tokens; swappable dense ↔ Soft-MoE (4–8 experts, the variant that helped deep-RL
      scaling in 2024 work). It adds capacity at ~zero extra compute (the CNN, which always runs, is ~95% of the
      FLOPs); it does not reduce compute. A/B during BC on held-out accuracy per situation bucket, confirmed in RL.
6. **Model size: spend parameters where they are cheap.** Compute ≈ CNN width² × map cells, so the CNN ("eyes") stays
   modest and is sized by the GPU benchmark (doubling its width ≈ 4× compute ≈ 4× less experience per day). The
   "brain" (core, player transformer, heads, MoE experts) costs ~1 MAC per parameter per decision, so it can grow from
   ~1.5M to ~10–30M params almost free. Start ~6M total (M width, Phase 0); pick the final size from BC scaling curves
   (S/M/L on the same human data), since more experience usually beats a bigger net on one GPU.
7. **Maps & generalisation:** official maps weighted like the real playlist + dihedral augmentation (mirror/rotate the
   bins and nation coordinates; the sim stays self-consistent) + procedural maps (TS port of the map-generator pipeline,
   validated byte-for-byte on official maps; noise continents/islands/rivers/mountains, random sizes/aspects, random
   nation spots), served by an in-memory `GameMapLoader`. Held-out official maps measure the generalisation gap.
8. **Spawn phase = a ~40-decision mini-game** (200 ticks / Δ5): pick, wait, relocate (a new spawn intent moves you) while
   watching others. Spawn heat map over unowned land; forced spawn near the end; masked under random spawn.
   Learned from top human spawns (BC) + terminal reward via the value function.
9. **Reward terminal only.** Win = +1, else r = −1 + 0.5·(N−rank)/(N−1) (1v1/2v2 → ±1). Agent rank: `deathPosition` if
   dead; alive at end: 1 + alive non-bot players with more tiles; never spawned = last. Teams: team placement shared.
   If learning stalls: reverse curriculum (start from mid/late-game states), only then annealed potential-based shaping.
10. **PPO** (CleanRL-style), GAE, masked logits, KL-to-BC (annealed), territory-delta channel; PyTorch eager + bf16.
11. **Δ = 5 ticks (0.5 s), one action per decision**: ~2,400 decisions per 20-min game instead of 12,000. Extra human
    intents inside a BC window queue into following windows. Measured on 61 replays (2026-09-30): median 21 commands/min
    per player in ranked 1v1 (p90 32), 16 in 2v2; only 10–15% of active 0.5 s windows hold 2+ commands, so Δ = 5 holds.
12. **Deployment requirements baked in**: same pinned commit on the server; random intent latency L ∈ {1,2,3} turns
    (p = .6/.3/.1), seat order shuffled per turn; ≤150 intents/min/seat; real-time CPU inference.
13. **Curriculum: Ranked 1v1 → Ranked 2v2 → public FFA/Team with nations + modifiers → augmented/procedural maps.**
    Ranked first: 4 maps, no nations, 2–4 players (natural self-play), 10–15 min games, best human data, clean win/loss.
14. **Game updates: the base model tracks the official servers.** The game submodule is pinned to the production release
    tag (currently v0.34.x) and moved to each new gameplay-relevant release via the playbook below. Upstream ships a minor
    version every ~5–7 weeks (v0.26 Oct 2025 → v0.34 Aug 2026), patch tags often, and gameplay changes land roughly weekly
    (last 6 weeks: economy saturation ×3, attack speed/loss ×5, spawn phase 30→20 s, SAM range ramp, overtime, doomsday,
    MIRV rules, nation AI). A small `scripts/release_watch` lists new release tags and whether they touch `src/core`.
    Built in from day one:
    - bridge reads every rule constant from `Config` at runtime (costs, ranges, radii, speeds) — nothing hard-coded;
    - current unit prices and key rule values (SAM range per level, nuke radii, spawn length, win threshold) are scalar
      inputs, so a balance patch shifts inputs instead of silently invalidating the policy;
    - channel/action registry + zero-init "network surgery" tooling (OpenAI Five trained ~10 months across Dota patches
      this way instead of restarting);
    - optional later: rule randomisation (±10–20% on balance constants via a `Config` subclass) for patch robustness.
    Upgrade playbook: `git diff <old>..<new> -- src/core` → classify (none / balance / rules / new mechanic / nation AI) →
    move the submodule to the new tag → bridge tests + determinism + predictor Gate A (flags copied SAM logic drift) →
    fine-tune from the latest checkpoint (league keeps old checkpoints as opponents) → eval vs the previous bot on the new
    rules → optional BC refresh from new-version replays. Balance/rule patches: fine-tune only (hours–days of GPU);
    new mechanic: bridge code + surgery + longer fine-tune; never retrain from scratch. Keep one worktree per past pin so
    old replays stay re-extractable.
    Log: v0.34.17 → **v0.34.21** (2026-10-01): 32 commits of lobby, telemetry, auth, record metadata and npm changes, no
    simulation change (nothing in `src/core/game`, `execution`, `configuration`); ranked 1v1 records from v0.34.17,
    v0.34.20 and v0.34.21 replay in sync at v0.34.21 (5/5, every hash). v0.34.21 requires npm ≥ 12.1 (`engine-strict`).

## Architecture
```
Python trainer (GPU)                                    Node env workers (CPU, ~12 processes)
 PPO/BC learner ◄─ buffer ◄─ batched inference ◄─ frame deltas per game  ◄─ K games each, stepped round-robin
                                    │  ─► actions per seat ─► decoder → latency queue → GameRunner.addTurn
                         GPU ego-obs builder (per seat)
```
- **Shared frame per game**: static terrain once (terrain deltas when water nukes are on); per cell top-2 owners +
  land fractions, fallout; unit list; players table from `allPlayers()`; alliance/embargo/request pairs. After reset only
  changed cells are sent; target ≤ ~20 KB per step.
- **Per-seat ego tensors on the GPU** via lookups (relation[seat, owner] → self/ally/teammate/enemy/bot/unowned; owner
  features per cell; units by relation). Extra self-play seats cost almost nothing.
- Transport: Node child processes over stdin/stdout, length-prefixed binary.

| Model part | Design | Params |
|---|---|---|
| CNN encoder | ~48 ch × cell grid; 4×4 stride-4 stem; 3 IMPALA-style ResNet stages (2 blocks of 3×3 convs, no norm, zero-init residual) 96/192/256 ch to 1/16 res; avg+max pool | ~4.3M |
| Heat-map decoder | U-Net-lite back to 1/4 res, pixel shuffle to cell res + 1×1 skip from the input, FiLM on action/unit type; separate spawn head | ~0.3M |
| Player encoder | 2-layer transformer d=128 over ≤64 players (features + spatial features pooled over their cells) | ~0.4M |
| Scalars + core | MLPs → 512-d (dense or Soft-MoE) | ~0.8M |
| Heads | action type, player pointer, bins, arc, value | ~0.3M |
| Total (v1) | ≈2 GMAC per decision on a 32k-cell grid, ~95% of it in the CNN; measured ~6.5k decisions/s inference, ~2.3k/s training (4070 Ti, bf16) | ~6M |
| Scale-up path | wider core / transformer / Soft-MoE experts, chosen from BC scaling curves; CNN widened only if it underfits | ~10–30M |

**Parallelism (this PC):** ~12 Node workers × 2–4 games (round-robin); one Python process batches GPU inference for all
waiting seats (64–256) and runs PPO; two worker groups alternate so CPU and GPU overlap; spare cores run eval and replay
extraction. BC: CPU workers replay human games into a shuffle buffer while the GPU trains. ~40 games ≈ 1–2 GB RAM.

**Training control (the PC is also used daily):** the trainer exposes a local-only control API (127.0.0.1) with two
thin front ends: a local web page and a CLI (`python -m trainer.ctl pause|resume|save|stop|throttle|status`).
- Pause (workers idle, CPU/GPU freed), resume, save, save & quit, throttle profile (e.g. 4 cores + small GPU batch);
  Ctrl+C saves before exit.
- Checkpoints hold everything to resume exactly: model, optimizer, league pool, counters, RNG states, config, game
  version. In-progress games are dropped on quit (fine for on-policy PPO).
- Live status: training metrics (+ link to TensorBoard), decisions/s, CPU/RAM (psutil), GPU load/VRAM/power/temperature
  (`nvidia-smi` query), CPU temperature only if LibreHardwareMonitor is running (Windows doesn't expose it reliably);
  optional auto-pause above a configurable GPU temperature.

**Observations:** terrain; ownership ego one-hots + owner troop density + traitor; fallout; defense-post coverage;
territory delta; units by relation (cities, ports, factories, silos + ready slots, SAMs + level/slots, defense posts,
warships, transports + troops, trade ships, nukes in flight + targets); enemy SAM coverage. Scalars: tick, phase,
immunity, own troops/max/gold/tiles/rates, ready silo slots, boats in flight, alive players, mode, teams, ranked type,
all modifiers, win-threshold progress, time left, tiles-per-cell.

## Action space (v1)
| Type | Args | Intent / decoding rule |
|---|---|---|
| no-op | — | — (in spawn phase = wait / keep spot) |
| spawn | cell (spawn head) | `spawn` on unowned land; re-pick allowed; forced spawn at ~tick 195; masked if random spawn |
| attack | player ptr or unowned land, troop bin | `attack`; mask = shares land border (land-only check) + `canAttackPlayer` |
| retreat | player ptr or unowned land | `cancel_attack` on the largest attack vs that target |
| boat | target (player or unowned), cell, troop bin | `boat` to a tile of that target in/near the cell; ≤3 boats |
| build | unit type, cell | `build_unit` on an own tile; warship → water in a port's water body |
| upgrade | unit type, cell, amount bin | `upgrade_structure` on the own unit of that type nearest the cell |
| nuke | cell, arc bit, amount bin | `build_unit` Atom/Hydrogen; amount ≤ ready silo slots; blocked in immunity / near teammates |
| MIRV | cell | `build_unit` MIRV on an owned target tile |
| move warships | source cell, target cell | `move_warship` for own warships near source; target in the same water body |
| diplomacy | player ptr × {request/accept, reject, break, extend, donate gold, donate troops, embargo on/off, target} | alliance* / donate* / embargo / targetPlayer |
Dropped in v1: emoji, quick_chat, delete_unit, cancel_boat, embargo_all. Masks: cheap approximate grid masks (mostly
derived on the GPU from observation channels); exact check only on the sampled action; failures = logged no-ops.
Coverage of human commands (61 replays, 2026-09-30): this list 99.9% ranked 1v1, 99.8% 2v2, 99.5% FFA and Team; the
Phase 1 subset (no-op, spawn, attack, boat, build city) 94.7%, 92.1%, 76.0%, 81.3%. The Phase 1 subset misses what
decides the mid and late game: per ranked 1v1 game ~5 nukes, 3–4 silos, 3–4 defense posts, 2 factories, 1 port, 1 SAM,
upgrades; FFA/Team add alliances (~6%), troop donations (~3% in Team) and warships.

## Repo layout
New standalone GitHub repo **`Gaetarra/openfront-rl` (private; rename or make public any time)**, rooted at
`C:\applicazioni\Visual studio test folder\openFront RL\`. Only our work is in its history; the game is a submodule
(a pointer to one official commit, nothing redistributed), so the contributors list stays yours. Commits are authored by
you with **no Claude co-author trailer** (recorded in the root CLAUDE.md so every session follows it).
```
openFront RL/                          (repo root)
  OpenFront_Source_Code/openfrontio/   git submodule → openfrontio/OpenFrontIO @ production release tag (untouched)
  bridge/     TS + vitest: EnvWorker, Configs (ranked/public samplers), Maps (loader, augmentation, procedural),
              Frame, Actions, Ranking, NukePredictor, ReplayExtract; imports the game via tsconfig paths
              (`resources/*`, `src/*` → submodule) and relative paths
  trainer/    Python: env, obs, model, ppo, bc, crawl, league, eval
  scripts/    release_watch, benchmarks
  docs/PLAN.md   this plan
  CLAUDE.md   project guidance for future sessions (commands, architecture, determinism rules, commit convention)
  README.md, .gitignore, package.json (tsx/vitest for bridge), requirements.txt
```
The game's own deps are installed inside the submodule (`npm run inst`); bridge deps at the root.

## First session after approval (setup)
1. In the game clone: restore the upstream `CLAUDE.md` (my earlier edits move into the root CLAUDE.md, keeping the
   submodule clean) and check out the latest production release tag (v0.34.17 today; confirmed by spike S1).
2. `git init` at `openFront RL/`, register the existing clone as the submodule, add `docs/PLAN.md`, `CLAUDE.md`,
   `README.md`, `.gitignore`; first commit authored by you.
3. `gh repo create openfront-rl --private --source . --push`.
4. Start Phase 0. It installs the game's npm deps (`npm run inst`, lockfile only) and a Python venv with PyTorch CUDA
   (~2–3 GB download), and makes ~100 polite metadata requests to the public replay API.

## Phases (each ends with a gate)
**Phase 0 — Spikes (≈1 week)**
- S1 version/replay: fetch ~100 recent public records (`?turns=false`, incl. ranked) → gitCommit histogram (confirm the
  production tag) and ranked games/day; replay ~20 ranked + FFA/Team records at the pinned tag → in-sync rate.
  v0.34.x lacks main's snapshots and per-game fresh maps: until a release ships them, run one game per Node process and
  re-simulate seed + turns instead of snapshots.
- S2 env step: Compact Australia/World, 5 ticks + frame (full vs changed cells) + pipe transfer; target env overhead ≤30%
  of sim time, ≤ ~20 KB/step; `npm run perf:game` table (Compact/Normal, 0–400 bots).
- S3 GPU: PyTorch CUDA venv; eager bf16 fwd+bwd samples/s of the encoder at 250×125 → model width.
- Message the maintainers about bulk replay downloads (courtesy, avoids throttling, maybe a dump). Done: the owner
  answered on 2026-09-30 (see Context).
- Gate: pin chosen; throughput numbers recorded.

**Phase 0 results (2026-09-24, gate passed; scripts: `scripts/bench_env.ts`, `scripts/bench_gpu.py`)**
- S1: 100/100 sampled public records ran v0.34.17 (`59da4bc1`), so the pin is right. Last 24 h: 10,460 unranked,
  2,439 ranked 1v1 and 667 ranked 2v2 public games. 20/20 replays (ranked 1v1/2v2, FFA, Team, up to 85 players) stayed
  in sync at every hash checkpoint. Some ranked records are void (1 player, ~210 ticks): the crawler must drop records
  with fewer than 2 (1v1) or 4 (2v2) spawned players.
- S2, one core, real records, 5-tick steps: sim 2.7 ms (Compact Asia 2v2), 9 ms (Australia 2v2, 400 bots), 18 ms
  (Giant World FFA, 85 players). Per-cell frame updated only for cells in the tick's `packedTileUpdates`: 0.7–2.2% of
  sim time (a full rebuild would cost 53–73%). Delta per step (11 B per changed cell): mean 3–6 KB, p95 11–17 KB,
  max 60 KB; full frame 256–857 KB. Pipe write 0.03 ms/step. The game logs with `console.log`, so the bridge must route
  it to stderr when stdout carries frames. `perf:game` Australia, first 3,000 ticks: 100 bots 1,063 ticks/s, 400 bots
  452 ticks/s: bots are most of the CPU cost. Up to 404 (ranked) / 584 (Giant FFA) players are alive at once, so the
  player transformer's ≤64 slots need a selection rule (Phase 2).
- S3 (RTX 4070 Ti, torch 2.14 cu130, 32k-cell grid, batch 64): the ConvNeXt encoder was memory-bound (depthwise 7×7,
  LayerNorm, layout conversions; no Triton on Windows, so no `torch.compile` fusion): 3.8k inference / 1.5k training
  samples/s even with channels-last. IMPALA-style ResNet, channels-last bf16:
  S 64/128/192 (2.3M) 11.4k / 3.8k, **M 96/192/256 (4.4M) 7.1k / 2.4k (chosen)**, L 128/256/384 (9.0M) 4.9k / 1.7k.
  The card measures 68 TFLOP/s (bf16 matmul) and 348 GB/s (copy); forward cost S/M/L = 1.9/3.9/7.0 GFLOP per
  decision, so the ResNet runs at 31/41/51% of peak: mostly compute-bound, the rest is memory-bound small ops at full
  resolution (a Linux/WSL2 `torch.compile` could fuse some of them; untested).
  VRAM ≤ 2 GB at batch 128. PPO with 2–3 epochs → ~0.7–1k decisions/s, GPU-bound; CPU has 2–3× headroom, so fewer
  workers (6–8) can leave cores free for daily use.

**Phase 1 — Vertical slice (≈2–3 weeks)**
- Minimal bridge + Python vec env, GPU ego builder, small model, PPO; reduced actions (no-op, spawn, attack, boat, build
  city); `Ranking.ts`; latency queue; determinism test (same seed + actions → same hash, fresh process and after other games).
- Training control: checkpoint save/resume, control API + local web page + CLI (pause/resume/save/quit/throttle, metrics,
  system status and temperatures).
- Gate: tiny Compact map (Onion/Pangaea), 1 agent vs 2–3 Easy nations, 5-min timer, terminal reward → PPO clearly beats
  the random-legal agent within hours; value head predicts outcomes.

**Phase 1 results (2026-09-25): pipeline done, learning gate NOT met.** Closed on 2026-10-01 as "pipeline done"
(`phase1` merged into `main`); the learning test moves to Phase 2, after behaviour cloning. Implementation plan:
`docs/superpowers/plans/2026-09-24-phase1-vertical-slice.md`.
- Built as planned. Bridge: fresh per-game maps + `resetGameStatics` (K games per worker process), incremental cell frame,
  seeded latency queue, rank reward, masks + decoding (exact checks in the worker, failures = counted no-ops), per-seat
  observation, stdin/stdout worker protocol. Trainer: `VecEnv` with the GPU observation builder, ResNet 64/128/192
  policy with a masked joint action distribution (entropy per head divided by log of its size), control API + page +
  CLI (`X-Control` header), PPO with checkpoints/resume, fixed-seed eval. Tests: 22 bridge (node:test instead of
  vitest), 22 trainer (unittest); determinism holds in a fresh worker and after other games. Pause, resume, throttle,
  stop and resume-from-checkpoint verified live.
- Speed: Onion Compact sim ≈0.06 ms/tick. The trainer's single Python thread is the bottleneck (82% of a core; workers
  11–17%; GPU mostly idle, 40–55 °C). 8 workers × 4 games: 1.1–1.9k decisions/s; × 8 games: 1.8–2.4k (27M decisions in
  4 h). Later fixes: overlap two worker groups (planned above), bigger batches, profile and trim per-step Python work.
- Gate: random-legal baseline 0/256 wins, mean rank 3.98/4, mean reward −0.997. PPO run `onion-gate` (27M decisions,
  3,300 updates): mean rank flat at 3.93–3.97 throughout, no wins after the first minutes, value/outcome correlation
  0.1–0.2. Eval on the same 256 seeds: rank 3.99, reward −0.998, i.e. no better than random.
- Diagnosis (64 games each): a scripted "attack unowned land with 25% of troops every 2 s, no boats" ranks 2.77 at the
  5-min timer (8% wins, 44% top-2) and 2.41 at 2 min (38% wins); random ranks 3.97 / 3.78 (`scripts/diag_baselines.py`).
  The task is learnable, but uniform-random exploration (≈1/3 boats, random attacks on nations, random amounts)
  finishes last in ~97% of games, so a terminal rank reward carries almost no signal. This is the cold start behaviour
  cloning exists for; the gate should not have asked from-scratch PPO to pass it.
- Value gate: sign accuracy is trivially high while the agent rarely wins, so it was replaced by the per-phase
  correlation between value and final reward (`value/outcome_corr_*`; gate: late ≥ 0.5, mid ≥ early).
- Exploration-bonus defect (found 2026-09-25, fixed 2026-10-01): `ActionDist.entropy` weighted each type's argument
  entropy by the type's probability, so the bonus pushed the type head toward the types with the most arguments
  (attack, boat) and away from no-op: no-op fell from 44% to 1–18% of actions during the run. It did not cause the
  stall (the missing reward signal did). The weights are now detached; `test_entropy_bonus_does_not_favour_types_with_more_arguments`.
- Open minors from the final review: the status shows "running" for a moment when stop arrives while paused; a hung
  worker is killed but not reaped; `--resume` silently ignores flags outside `KEEP_ON_RESUME` (fix: explicit flags
  override, but refuse worker/game-count changes, since episode counters are per slot).
- Checkpoints: `checkpoints/onion-gate/` (every 250 updates + `latest.pt`), `checkpoints/onion-gate-g4-stopped/` (first
  attempt, killed with its Claude Code session at update 375). Long runs must be started detached (see CLAUDE.md).

**Phase 2 — Behaviour-cloning slice on ranked 1v1 (adopted 2026-10-01)**
From-scratch PPO with a terminal reward gets no signal (Phase 1). Behaviour cloning (BC) fixes the cold start, and the
Phase 1 actions already cover 94.7% of ranked-1v1 human commands. The slice also tests the whole BC pipeline (crawl →
replay → observations and labels → training → evaluation) before the full environment is built. User decisions
(2026-10-01): the 5 Phase 1 actions now, the full action set right after; the learning test runs in ranked 1v1 (Onion
becomes optional); games are watched in the game's own client; BC imitates all players.
- a. `trainer/crawl.py`: ranked 1v1 records since v0.34.17 went live (≈2.4k games/day), ≈1 req/s, gzip cache in the
  git-ignored `data/records/`; usernames blanked (clan tags and team indexes kept: sim inputs), deleted players
  honoured, void games (fewer than 2 spawned players) dropped.
- b. Replay extractor (`bridge/replay.ts` + a worker mode): the game is built from the record as in
  `tests/replay/ReplayGame.ts` (`toWireGameStartInfo`); every 5 ticks it streams the same blobs as `EnvGame.observe`
  for each human seat, through the same protocol and GPU builder, so BC and RL see identical inputs. Labels: no-op /
  spawn (cell) / attack (target, bin) / boat (cell, bin) / build city (cell). Extra commands queue into later windows
  (decision 11); commands outside the 5 actions and labels illegal under our approximate masks are counted and left
  out of the loss, never relabelled as no-op. Recorded hashes are checked every 10 turns and a record is cut at its
  first mismatch: the hashes, not an exact `gitCommit` match, decide which records are usable.
- c. `trainer/bc.py`: loss = −log π(label) through `ActionDist.log_prob` with the masks; held-out games; accuracy and
  NLL per head and per game phase; the 4 ranked maps padded to one grid. Every sample keeps the game outcome and the
  player's Elo (computed from `publicID` over the crawled games), so the data can later be filtered or weighted
  without re-extracting.
- d. Ranked 1v1 environment: `EnvConfig` gains the ranked 1v1 setup (2 seats, 100/400 bots, no nations, spawn
  immunity, 10/15-min timer, map weights), reusing `MapPlaylist.get1v1Config` if it imports cleanly.
- e. Watching games: eval games are saved as game records (`createPartialGameRecord`, `src/core/Util.ts`) and replayed
  in the game's own client in dev mode. On localhost the client loads archived games from
  `http://localhost:8787/game/<id>` (`src/client/ApiBase.ts`, our control port) and skips the version check when its
  `gitCommit` is "DEV" (`src/client/JoinLobbyModal.ts`), so the control server can serve them. Starts with a short spike.
- Gate (ranked 1v1, fixed seeds, win rates with standard errors): the BC agent clearly beats random and the scripted
  expander (moved from `scripts/diag_baselines.py` into `trainer/eval.py`); then PPO from the BC checkpoint with
  KL-to-BC beats a frozen BC copy. The trainer speed fix (overlapping worker groups) only if profiling on ranked maps
  (8–11× Onion's cells) still shows the Python thread as the bottleneck.

**Phase 2 results (2026-10-02): Gate A and Gate B passed.** Plan:
`docs/superpowers/plans/2026-10-01-phase2-bc-slice.md`; branch `phase2` (local, not merged).
- Data: 16,107 usable ranked 1v1 records started 2026-09-23 18:00 to 2026-09-30 23:10 UTC; 2,355 void ones dropped
  (13%); 765 held out (5%, by a hash of the game ID). BC replayed 7,242 records with 0 errors and 0 desyncs. Labels:
  no-op in 83.4% of windows, a command in 16.6%; commands outside the 5 actions 1.0%, labels our masks forbid 0.6%
  (held-out 0.3–1.5%); the latency queue dropped 3.1% of commands (20-record sample). The profile showed the simulation
  workers, not the trainer, as the bottleneck: no trainer speed fix needed.
- Watching works: public records and our recorded games play in the game's dev client (`trainer.watch`, or a running
  trainer, serves them on 8787). A download manager that intercepts the map `.bin` fetches (Internet Download Manager)
  hangs the replay at "Game is starting...".
- BC run `bc` (14 workers × 6 games): ~745 rows/s (validation every 500 updates costs ~15%). It learnt in the first
  ~10k updates (~40 min) and then stayed flat; stopped by choice at update 72,729 of 200,000 (4 h 52 min, 12.2M rows,
  7,242 games). On the same 224 held-out games every 10k-update snapshot scores within noise (NLL 1.065–1.10); 60k is
  best: NLL 1.065 (spawn 0.73, early 1.08, mid 1.15, late 0.60), value/outcome correlation 0.557, attack target 36%,
  attack size 52%, spawn cell 1.4%, boat cell 3.5%, action type 84% (the no-op share: the likeliest choice is to wait,
  sampling acts at the human rate). The in-run held-out checks (12 games each) are too noisy to pick a checkpoint.
  Likely cause of the plateau: a constant learning rate with small, correlated batches (one 5-turn window of the same
  84 games per update, ~28 command rows); the next BC run gets a decaying learning rate and several windows per update.
- Gate A PASSED with `checkpoints/bc/update0060000.pt` (256 games each, seats swapped): 0.984 (SE 0.008) against
  random, 0.949 (SE 0.014) against the scripted expander, value/outcome correlation 0.557. In 8 recorded games against
  the expander the clone won all 8: it builds 5–17 cities (the expander none), launches boats, attacks less often, and
  re-clicks its spawn during the spawn phase as humans do.
- PPO run `ppo-bc` (16 workers × 2 games, 48 steps, lr 2.5e-4, entropy bonus 0.01, KL to the clone 0.05 from a
  one-sample estimate, learner always in seat 0): ~190 decisions/s (8 × 4: ~166). The training win rate (last 200 games)
  rose to ~0.6 at updates 480–600, then collapsed to ~0.2 over the last ~60 games by update 718, while the KL estimate
  spiked (31, 82, 300+). Gate B on update 500 (512 games, seats swapped): 0.471 (SE 0.022), FAILED: the training curve
  overstates Gate B (run 2 shows it again) and update 500 was picked at its top.
- Fixes (2026-10-01): the KL penalty is the exact joint KL (`ActionDist.kl`; the one-sample estimate explodes when the
  learner samples a map cell it gives almost no probability that the clone favours); the learner plays both seats, half
  the games each, with the win rate logged per seat; checkpoints keep the win-rate window across resumes; the control
  page charts the key curves and says what is training; eval reports the win rate per seat and prints its progress.
- PPO run `ppo-bc2` (from the clone: exact KL 0.2, lr 1e-4, no entropy bonus, 16 × 2, ~190 decisions/s; 5,000 updates
  from 2026-10-01 21:26 to 2026-10-02 10:42): the training win rate rose from ~0.6 to ~0.8 in the first ~500 updates
  (~1 h) and stayed there (200-game windows 0.74–0.885), with the KL to the clone at 0.05–0.06 and entropy at the
  clone's ~0.43 throughout. Gate B PASSED on the same 512 games, seats swapped: update 1000 0.730 (SE 0.020), update
  5000 0.773 (SE 0.019). Seat does not matter, as expected in a real-time symmetric game: 0.727 / 0.734 and
  0.770 / 0.777 per seat. The training curve reads ~0.1 above Gate B (0.885 at update 1000) and its swings do not rank
  snapshots (update 5000 sat lower on it and scored higher): snapshots are picked by Gate B only. A GPU driver error
  (System event nvlddmkm 153, also logged on this PC before the project) killed the run at update 1157; it resumed from
  the update-1150 checkpoint, and long runs now go in a restart loop (CLAUDE.md).
- Plateau and the KL-0.05 test: against a frozen opponent the KL penalty looked like the ceiling, so updates 5000–6000
  ran with KL 0.05 (same anchor). The KL levelled at ~0.15, attacks rose from 17% to 19% of decisions, games got ~7%
  shorter, and the training win rate stayed at ~0.81; but Gate B on update 6000 fell to 0.576 (SE 0.022; seats
  0.566 / 0.586) on the same games where update 5000 scored 0.773. The looser leash made the bot worse against the
  clone, and the training curve did not show it. Updates 5500 and 5750 scored 0.578 and 0.633 (128 games each) where
  the curve read 0.83–0.84: the whole stretch was worse. The Phase 2 bot is `checkpoints/ppo-bc2/update005000.pt`.
- Open: the training curve reads 0.05–0.25 above frozen snapshots (5 of 5 tested) and missed the weaker-KL decline.
  Likely cause: a game lasts ~20 updates and the learner trains on the 32 games it is playing, so it adapts to those
  games while it plays them; a frozen copy facing new games cannot, and a looser leash allows more of it. Test: the
  training loop with lr 0 from update 5750 (if its curve drops to Gate B's level, the cause is confirmed). For RL:
  measure strength only with frozen copies on fixed seeds, during training too; more games per update and smaller
  steps. Also open: the BC plateau fix; evaluations replay fixed seeds, so their recordings reuse game IDs and
  overwrite each other in `replays/` (now update 6000's Gate B games).

**Phase 3 — Full environment + BC refresh**
- Full v1 actions + masks, spawn head, observation channels for every unit type, nukes in flight and SAM coverage,
  added by network surgery (zero-init heads, decision 14); the extractor maps every intent type. `Configs.ts`: ranked
  2v2 (`get2v2Config`), public sampler (playlist map weights, team configs, modifier pool and exclusions); `Maps.ts`:
  in-memory loader + dihedral augmentation; eval suite (fixed seeds, held-out maps) + TrueSkill table.
- Crawl ranked 2v2, then top public FFA/Team games; BC refresh with every action.
- Player encoder as designed above, never built so far: Phase 1 averaged all player rows into one vector, harmless with
  4 players on Onion, but in ranked 1v1 the opponent is one row among ~400 bots, so its troops, gold and troops ÷ max
  barely reach the decision (user decision 2026-10-02: fix it here). Each player token = its features, its relation and
  the CNN features pooled over its cells (where it is, what it borders); a 2-layer transformer over the tokens;
  attention pooling from the core with several heads (so it can focus on the biggest threats in FFA, where one mean over
  all enemies would bury a strong neighbour among weak distant ones), plus the self token and one mean per relation
  (self, ally, enemy, bot); the attack pointer reads the transformer outputs. All players first (masked: attention is
  small next to the CNN); the ≤64-slot selection rule only if decisions/s drop too far. Compared with the current mean
  in the BC model experiments below: held-out NLL and attack-target accuracy (36% in Phase 2).
- Map channels per owner: the CNN sees land only as mine / ally / enemy / bot plus each owner's troops per tile, so it
  cannot tell a 500k-troop neighbour's border from a 10k one's when their densities match, and buildings are only mine
  or not mine. Paint each owner's total troops (log), troops ÷ max, gold (log) and troops attacking me onto its cells
  (the same gather as the density channel), and split building channels by relation (mine / ally / enemy / bot).
- Model experiments on the same data (cheap, supervised): dense vs Soft-MoE and S/M/L sizes; held-out accuracy per
  bucket (spawn, early, mid, late-many, late-few, ranked, modifiers) plus decisions/s → pick the RL architecture.
- Gate: 1,000 games across modes without errors; invalid-action rate < 1%; throughput within the Phase 0 budget; the
  refreshed BC agent beats the Phase 2 clone (`checkpoints/bc/update0060000.pt`) in ranked 1v1 and Easy/Medium nations
  in Compact FFA. Beating the Phase 2 PPO agent is left to Phase 4's RL: a clone is not expected to.

**Phase 4 — RL on Ranked (ongoing)**
- 4A ranked 1v1: self-play league from BC init — current policy vs pool (past checkpoints + BC), prioritized fictitious
  self-play, KL-to-BC annealed. 4B ranked 2v2: teammates = copies of the current policy.
- Stall protocol: reverse curriculum → shaping. Gate: ≥70% vs BC policy and a rising TrueSkill vs past checkpoints.

**Phase 5 — Exact nuke-interception predictor (≈1–2 weeks)**: silo choice as `PlayerImpl.nukeSpawn`,
`UniversalPathFinding.Parabola`, targetable window, copied SAM targeting (slots, cooldowns, upgrade ramp, stacked-launch
stagger). Gate A: frozen-world harness matches 100% per nuke. Gate B (metric): in-game agreement, unexplained mismatches = 0.

**Phase 6 — Broaden & scale**: public FFA/Team with nations (Easy→Impossible) + all modifiers (water nukes via terrain
deltas); procedural maps (TS generator port, byte-for-byte test on official maps); Normal-size maps; league adds
nations as opponents; per-mode specialist fine-tunes if they win; bigger model; optional rented GPU run.

**Phase 7 — Deployment**: ONNX export; headless Node bot client (core sim + onnxruntime) joining private lobbies on a
self-hosted server at the pinned release; checkpoint chosen by lobby mode (AGPL: publish source if hosting a modified
server for others). Official servers: the owner allows testing in casual multiplayer games from the AI-DEV account (not
ranked, no abuse). The connection must keep Turnstile (e.g. the official web client in the user's own browser session,
or a path the maintainers provide); never bypass it. To decide when the bot is strong enough to be worth testing.

## Reused code
| Need | Existing code |
|---|---|
| Build/step games | `createGameRunner`, `GameRunner.addTurn/executeNextTick`, `tests/util/ScriptedGame.ts` (main only) |
| Maps | `GameMapLoader`/`MapData` interface, `loadTerrainMap`, `tests/perf/fullgame/NodeGameMapLoader.ts`, `map-generator/map_generator.go` (logic to port) |
| Game configs | `MapPlaylist.get1v1Config/get2v2Config`, modifier pool + exclusions (src/server/MapPlaylist.ts) |
| Replay records | `tests/replay/ReplayGame.ts` (keep its `toWireGameStartInfo`) |
| Legality | `Player.canAttackPlayer/canBuild/buildableUnits/canSendAllianceRequest/canDonate*/canTarget`, land-border pattern `AiAttackBehavior.ts:842-860` |
| Nukes/SAMs | `PathFinder.Parabola.ts`, `Line.ts`, `PlayerImpl.nukeSpawn`, `Config.dynamicSamRange`, `SAMLauncherExecution.ts` |
| Ranking | `stats().getPlayerStats(p).deathPosition`, `game.getWinner()` |
| Reset/fork | `GameRunner.snapshot`, `createGameRunnerFromSnapshot` (main only; usable once a release ships them) |

## Compute & money
- Local: electricity only; ≈0.35–0.45 kW at full load ≈ 9 kWh/day (≈ €2–3.5/day at €0.25–0.35/kWh).
- Rented RTX 4090 (vast.ai, RunPod): ~$0.3–0.7/h ≈ $50–120/week with ≥16 CPU cores on the same machine; checkpoint often.
  AWS/GCP/Azure are poor value here (~$0.7–1.2/h for weaker GPUs, $3+/h A100, GPU quota requests) unless free student
  credits cover it. Kaggle's free weekly GPU quota suits BC experiments; Colab is too disconnect-prone for RL.
- Everything runs locally; one optional 1–2 week rented run (~$50–150) only after the pipeline is proven.
- Replay data is public (docs/API.md) and the owner cleared its use for training; crawl politely (≈1 req/s, cached).

## Verification
- `node --import tsx --test "../../bridge/*.test.ts"` (node:test, run from the game dir): decoder rules, masks, ranking, latency queue, frame deltas == full recompute,
  determinism, map augmentation (flip twice = identity; sim runs), procedural port byte-for-byte vs official map.bin,
  predictor Gate A.
- Smoke: random-agent games to completion in every mode; replay extraction hashes match recorded hashes.
- Fixed eval suite (ranked 1v1/2v2 maps, public modes × modifiers, held-out and procedural maps, fixed seeds) per checkpoint.
- Metrics (TensorBoard):
  - Strength: win rate + mean placement vs BC policy, past checkpoints, nations (Easy→Impossible); TrueSkill across the
    league; generalisation gap (training vs held-out maps); later results vs humans.
  - Learning health: value explained variance and outcome accuracy by game time (incl. at spawn end), entropy per head,
    approx KL, clip fraction, KL to BC, grad norm.
  - Behaviour: invalid-action rate, intercepted-nuke rate, boats landed/sent, APM, territory curves, betrayals.
  - Systems: decisions/s, GPU/CPU utilisation, time split (sim / frame / IPC / inference / training).
  - BC: per-head and per-bucket accuracy and log-likelihood on held-out human games.
