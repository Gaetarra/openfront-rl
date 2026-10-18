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
**The bot: StockShip** (named by the user on 2026-10-05, after the chess engine Stockfish). The goal is the strongest
bot possible, not a copy of human play:
- human habits (a few commands per minute, single-missile nukes) bind only behaviour cloning;
- RL may go past them as fast as the frozen evaluations allow. The KL leash keeps early RL near the clone, and
  loosening it too early made the Phase 2 bot worse (0.773 → 0.576);
- the game's rules always bind, and the server's command limits count as game rules.

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
- `attack` = {targetID | null, absolute troops}: player-level. Boats/builds/nukes/warships take a tile. An attack runs
  on after its command until one of these happens (`AttackExecution.ts`):
  - its troops are spent;
  - it is cancelled;
  - nothing is left to conquer;
  - an alliance forms.
  Since v0.34.24, an attack that has nothing to conquer when it starts (a target the attacker does not border, or no
  wilderness next to it) is refunded at once.
- **Nukes and SAMs** (checked in v0.34.22 on 2026-10-05):
  - A SAM holds as many missiles as its level and reloads each one in 9 s (`SAMCooldown`).
  - Interception is deterministic, with no hit chance.
  - Each nuke is claimed by one SAM, so overlapping SAMs add their slots.
  - Range: 150 − 480 / (level + 5) tiles, 70 at level 1.
  - Among the nukes due this tick, hydrogen bombs get a bonus worth 70 tiles of distance, then nearer targets come
    first, then the sooner explosion (`SAMLauncherExecution.ts`).
  - An upgrade adds its slot empty: `increaseLevel` puts it in the reload queue, so it is ready 9 s later. The range
    ramps to the new level's over 4.5 s. Silos work the same way.
  - Atom and hydrogen bombs fly 10 tiles per tick, so a strike lands within seconds. An upgrade during a strike only
    widens the range of the slots already loaded.
  - `build_unit.amount` (1–50) repeats only atom and hydrogen bombs, one tick apart per silo. Each missile needs its own
    loaded silo slot and its gold; the extras fail silently.
  - `upgrade_structure.amount` (1–50) buys levels until the gold runs out.
  - Each new structure takes its own intent.
  - `rocketDirectionUp` flips the arc.
  - The client offers atom bombs ×1, ×2, ×5 or ×max (capped by loaded slots and gold), hydrogen bombs ×1 only, and
    upgrades ×1, ×5, ×10 or ×max.
  - So 2 atom bombs arriving less than 9 s apart beat a level-1 SAM; the game's own comment calls ×2 the standard answer
    to a single SAM. Two single nukes 0.5 s apart do the same.
- **Structures:**
  - Building and upgrading cost the same.
  - The price counts min(levels owned, levels ever built), and ports and factories share one count. City, port and
    factory cost min(1M, 2^n × 125k).
  - Construction times: city and factory 2 s, port and defense post 5 s, silo 10 s, SAM 30 s. Upgrades are instant.
  - Structures stay ≥ 15 tiles apart.
  - Spreading versus stacking:
    - a nuke deletes every building within its outer radius, with all its levels (atom 30 tiles, hydrogen 100);
    - a captured tile hands its building to the captor with every level. Defense posts are destroyed instead;
    - so a nuke on spread level-1 buildings costs one level per building hit, while a stack loses all its levels, and
      a captured stack goes to the enemy whole;
    - spreading protects against one bomb only when the buildings are farther apart than its blast's diameter: 60
      tiles for an atom bomb, 200 for a hydrogen bomb, against a minimum spacing of 15;
    - each new building takes one decision, and so does each upgrade level while amounts are deferred.
- **The server's command limits** (`src/server/ClientMsgRateLimiter.ts`; the simulation does not apply them):
  - per client, 10 intents per second and 150 per minute (wall-clock token buckets that start full, plus a count per
    fixed window);
  - intents over the limit are dropped silently, for humans too;
  - an intent over 2,000 bytes, or 5 MB sent in total, gets the client kicked.
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
- Troop growth (`Config.troopIncreaseRate`, applied every tick by `PlayerExecution`, v0.34.21): (10 + troops^0.73 / 4) ×
  (1 − troops / cap), so it peaks at ~42% of the cap (0.73 / 1.73) and is about a third of the peak at 5% and at 90%;
  cap = 2 × (tiles^0.6 × 1000 + 50,000) + 250,000 per city level (bots: half the growth, a third of the cap; nations
  0.9–1.05× by difficulty). The model sees troops ÷ cap for itself (scalar 6) and for every player (player feature 9);
  the formulas stay in the game's Config, never in our code.
- **Economy, kills and diplomacy** (checked in v0.34.24 on 2026-10-18; v0.35.2 changes only the trade-ship and train
  spawn rates):
  - **Gold.** Income is flat, 100 a tick for humans (1,000/s), and land makes none.
    - Trade pays the full route value to both port owners: 75,000 / (1 + e^(−0.03(d − 300))) + 50d for d tiles
      travelled. That is 14k at 200 tiles, 52k at 300 and 125k at 1,000.
    - Trains pay both owners at each stop.
  - **Warships.**
    - They capture trade ships near their patrol tile. The captor alone is paid the route's value when the ship
      reaches its own port.
    - They sink a transport boat with one shell, from up to 130 tiles away.
  - **Attacks.** One troop pool defends all of a player's land, and an attack removes its troops at once
    (`AttackExecution.ts:117-123`). Per tile taken (`Config.attackLogic`):
    - the attacker's time grows with the defender's home troops ÷ the attack's troops, clamped to 0.82–7.5;
    - the attacker's loss grows with the same ratio, clamped to 0.6–2, times a base plus 0.0039 × the defender's troops
      per tile;
    - the defender loses its troops per tile.
    - Large territories are up to 30% cheaper and faster to attack into.
    - Attacking a human or a nation gives the victim a 5-minute embargo on the attacker.
  - **Kills.** The attack that takes a player under 100 tiles conquers it: the rest of its land, all of a nation's gold,
    half of a human's (`AttackExecution.handleDeadDefender`, `GameImpl.conquerPlayer`). A player killed by a nuke
    credits nobody, and its gold is destroyed.
  - **Nukes.** Each tile hit costs the target 5 × troops ÷ tiles left. Losing 10% of the land therefore costs ~41% of
    the troops, attacks and boats included.
  - **Team.** Teammates cannot attack each other or land boats on each other. Donations are on in Team and off in FFA:
    up to a third of the stock, every 10 s per recipient.
  - **Alliances.** They last 5 minutes, and both sides can extend them. A betrayer is a traitor for 30 s: attacks on it
    lose half the troops and take tiles ~25% faster.
  - **Win.** Over 80% of the land without fallout, in every mode. Public FFA lowers the bar 2 points a minute after 30
    minutes.

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
   Autoregressive: type → unit type → player pointer | cell → modifiers; log-prob = sum; entropy per head. (Phases 1–3A
   draw the troop share independently of the target: the Phase 1 task list dropped the order without saying so, found
   2026-10-04. 3B restores it.)
4. **Missiles: the policy never guesses.** SAM-coverage channels first; then an exact interception predictor (copy of
   `SAMTargetingSystem` driving a mini-sim of SAMs + nukes) → per cell and arc: interceptor slots on the path.
   - The coverage channels count the ready interceptors on each cell: over the covering SAMs, level minus missiles
     reloading (3B). So a level-1 and a level-14 SAM look different.
   - That count is a lower bound for a strike on the cell. The predictor adds the SAMs along the path.
5. **Many different situations** (spawn, early/mid/late, many vs few players, modifiers, ranked vs public) at 3 levels:
   a. everything is an input: phase, alive players, mode, ranked type, every modifier as scalars;
   b. rule-routed specialisation where decisions differ: a dedicated spawn head; optional per-mode specialist checkpoints
      (e.g. a ranked-1v1 fine-tune) picked at deployment by lobby type;
   c. learned MoE: placed after the encoders (the router needs the CNN/player/scalar features), over the fused vector or
      the 1/16-res spatial + player tokens; swappable dense ↔ Soft-MoE (4–8 experts, the variant that helped deep-RL
      scaling in 2024 work). It adds capacity at ~zero extra compute (the CNN, which always runs, is ~95% of the
      FLOPs); it does not reduce compute. A/B during BC on held-out accuracy per situation bucket, confirmed in RL.
   **Specialists now, distillation later** (the user's call, 2026-10-18, after 4N):
   - Behaviour cloning trains one model on all modes (5a). RL then trains one copy of it per mode family.
   - Mixed RL coupled the modes' training signals and cost 1v1 ~300 Elo (Phase 4, 4N "Results"). Mixed BC has no such
     coupling: its labels are fixed human actions, with no value baseline and no advantage normalisation.
   - The MoE (5c) did not separate the modes under PPO: its router used the same 3 of 8 experts in every mode.
   - **The merge** is policy distillation (Rusu et al. 2016). The student matches each specialist's action
     probabilities on that mode's positions, a supervised loss against fixed targets.
     - A weight average of the specialists is a cheap first try, since they start from one clone ("model soups",
       Wortsman et al. 2022). It is unproven for RL fine-tunes.
     - Until then, deployment picks the specialist by lobby type (5b).
6. **Model size: spend parameters where they are cheap.** Compute ≈ CNN width² × map cells, so the CNN ("eyes") stays
   modest and is sized by the GPU benchmark (doubling its width ≈ 4× compute ≈ 4× less experience per day). The
   "brain" (core, player transformer, heads, MoE experts) costs ~1 MAC per parameter per decision, so it can grow from
   ~1.5M to ~10–30M params almost free. Start ~6M total (M width, Phase 0); pick the final size from BC scaling curves
   (S/M/L on the same human data), since more experience usually beats a bigger net on one GPU. Phases 1–3 use S
   (64/128/192): PPO is GPU-bound at ~190 decisions/s and M measured ~1.6× slower, so the CNN size experiments wait for
   Phase 6, and the brain grows first, through the MoE core (Phase 3A).
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
   **Reviewed 2026-10-18, after 4N** (the user's questions on rank, kills and land):
   - FFA's rank-graded reward carried no usable signal in 4N: the training games' mean reward sat at −0.86 for 4,000
     updates.
   - **No kill bonus.**
     - A kill goes to the attack that takes the victim under 100 tiles, so kills can be stolen. A nuke kill credits
       nobody.
     - Conquest already pays: gold, land and buildings.
     - Tribes and weak nations would be farmed.
   - **No raw land reward.** Land makes no gold, and a big empire with few troops per tile is cheaper and faster to take
     (Findings, "Economy, kills and diplomacy").
   - **Potential-based shaping is the safe form** (Ng, Harada & Russell 1999).
     - The reward is the change of a potential, so land lost later takes its reward back, and the best policy does not
       change.
     - The potential comes from the analysis of what predicts winning in each mode (Phase 4, "After 4N" item 2).
   - **Order when a mode's results stay flat:** late-game starts first, now practical since v0.35's snapshots save and
     restore any tick; then shaping.
10. **PPO** (CleanRL-style), GAE, masked logits, KL-to-BC (annealed), territory-delta channel; PyTorch eager + bf16.
11. **Δ = 5 ticks (0.5 s), one action per decision**: ~2,400 decisions per 20-min game instead of 12,000. Extra human
    intents inside a BC window queue into following windows. Measured on 61 replays (2026-09-30): median 21 commands/min
    per player in ranked 1v1 (p90 32), 16 in 2v2; only 10–15% of active 0.5 s windows hold 2+ commands, so Δ = 5 holds.
    **Under review (2026-10-06):** the user proposed three action bins (troops, gold, diplomacy), each taking up to one
    action per decision.
    - The reasons:
      - one softmax answers two questions: a build in a window delays a new attack, and the reverse. The bot re-reads
        the state every decision, so an emergency comes first, but it interrupts the plan: a counter-attack on a
        landing 200k transport pauses a run of SAM and city builds;
      - bursts: the server allows 10 intents/s, one action per decision only 2/s.
    - Against:
      - human play shows almost no parallel use. In 2,101 ranked 1v1 records, only 0.02% of 570k command windows mix a
        troop command with a gold command (10.3% hold 2+ commands, nearly all of one kind);
      - the server holds every client to 150/min, so bins add at most 25% sustained;
      - bins would make the server's limits reachable, so the environment would need the limiter and a budget input.
    - Kept for 3B by the bandwidth test (2026-10-06, results in 3B): the throttle lost nothing and the probe gained
      nothing. The probe reruns on the 3B clone before 4A.
12. **Deployment requirements baked in**: same pinned commit on the server; random intent latency L ∈ {1,2,3} turns
    (p = .6/.3/.1), seat order shuffled per turn; real-time CPU inference.
    - The server's command limits (10 intents/s, 150/min; "Findings") hold by construction. Decision 11 sends at most
      one intent per 0.5 s: 120/min, and at most 3 in any second with the latency jitter.
    - So the environment needs no copy of the limiter while decision 11 holds.
13. **Curriculum: Ranked 1v1 → Ranked 2v2 → public FFA/Team with nations + modifiers → augmented/procedural maps.**
    **Changed 2026-10-15 (the user's call):** Ranked 1v1 → FFA and Humans vs Nations against the nations (4N) →
    alliance and donation actions → human data of every mode (public FFA/Team, ranked 2v2) with the MoE routing test
    → modifiers → bigger and augmented/procedural maps. Public lobbies are where the bot will play (Phase 4, "4N").
    **Changed 2026-10-18 (after 4N; the user's calls):** v0.35.2 → human games of every mode and their analyses →
    alliance and donation actions → BC on all modes → RL specialists per mode family (ranked 1v1, Ranked FFA, public FFA
    against the nations, Team) → distillation → modifiers → bigger maps (Phase 4, "After 4N").
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
    - One worktree per simulation is enough: `pins/<its last tag>` (CLAUDE.md).
    - A simulation change can desync every older record, not just games that use the changed rule: the replay
      re-simulates the tribes and nations (v0.34.24).
    - Then the replay fixture (`bridge/testdata/ranked1v1.json.gz`) must be replaced by a record of the new pin. The
      v0.34.24 entry below lists what the tests need from it.
    Log: v0.34.17 → **v0.34.21** (2026-10-01): 32 commits of lobby, telemetry, auth, record metadata and npm changes, no
    simulation change (nothing in `src/core/game`, `execution`, `configuration`); ranked 1v1 records from v0.34.17,
    v0.34.20 and v0.34.21 replay in sync at v0.34.21 (5/5, every hash). v0.34.21 requires npm ≥ 12.1 (`engine-strict`).
    v0.34.21 → **v0.34.22** (2026-10-03; live since 2026-10-01): 6 commits of client, server, telemetry and payments
    changes; the only core file is `src/core/ApiSchemas.ts` (an account API field), no simulation change and no
    dependency change; 5/5 recent v0.34.22 ranked 1v1 records replay in sync at v0.34.22 (2,519 hash checkpoints).
    v0.34.22 → **v0.34.23** (2026-10-06; live by 2026-10-06): 1 commit, client only (the WebGL map recovers after the GPU
    process crashes, `src/client/render/gl`), no `src/core` or dependency change; 5/5 recent v0.34.23 ranked 1v1
    records replay in sync at v0.34.23 (1,819 hash checkpoints).
    v0.34.23 → **v0.34.24** (2026-10-10; live since 2026-10-06 between 18:30 and 19:00 UTC): 1 commit and a
    **simulation change** (class: rules). `AttackExecution.init` now refunds at once an attack with nothing to conquer
    (a target the attacker does not border, or no wilderness next to it). Before, its troops sat out one regeneration
    step and came back a tick later, which let players grow past the troop cap. No dependency or static-field change.
    - The bot never sends such an attack: our attack mask only offers targets with a conquerable land tile next to the
      seat's border, a subset of the engine's test.
    - Older records desync. 64 records spread over `data/` (v0.34.17–v0.34.22) replay with 0 desyncs at v0.34.23
      (~27,400 hash checkpoints), but 61 of them desync at v0.34.24, at turns 250–3,120 (median 510): only 13.7% of
      their turns come before the desync. 60 of the 61 hold no human attack with nothing to conquer, so re-simulated
      attacks take the new branch. Most likely the tribes': they check the border when they decide, but their attack
      starts at the end of the tick, after that tick's other attacks have moved the borders (`GameImpl.executeNextTick`).
      Boat landings start attacks too. The game's own `NationGoldPerMinute` snapshot changed as well.
    - So the last pin of the old simulation is kept as a worktree, as the playbook says: `pins/v0.34.23` (git-ignored).
      It has the same bridge and trainer code, its game folder is a worktree of the game repo at v0.34.23, and junctions
      lead to the main checkout's `node_modules` (same dependencies), `data`, `checkpoints` and `runs`. Records of
      v0.34.17–v0.34.23 replay there in sync (commands: CLAUDE.md).
    - The replay fixture (a v0.34.21 record) desynced. It is now ranked 1v1 game `c5QTvwaU5x` (Compact Europe
      Classic, 272 s, v0.34.24), picked for what the trainer tests need: a nuke in flight, a finished defense post,
      at least 5 structures (it has 7), at most 560 windows (543), and an out-of-set command (an emoji and a quick
      chat; the coverage test's count now says 2). Short games with a nuke are rare: the first candidate had no nuke,
      and the second's only defense post was never built. Typecheck, 43 bridge tests (determinism included) and 127
      trainer tests pass.
    - Not applicable this time: predictor Gate A (Phase 5), and the fine-tune and the match against the previous bot
      (step 4 trains the v4 network, and Gate 3B measures it on v0.34.24).
    - v0.34.24 records: 7,973 ranked 1v1 games from go-live to 2026-10-10 03:00 UTC (2,392 a day), 6,899 of them
      with two players; none crawled yet. Which records step 4 uses is decided in 3B ("Step 4's records").
    v0.34.24 → **v0.35.2** (scanned 2026-10-18; the upgrade is next, Phase 4 "After 4N" item 1). Production runs it
    since 2026-10-16: commit 08b63c05, read from 5 public games of 2026-10-17. It is 227 commits away.
    - **The engine moved.** `src/core` is now `packages/{engine,engine-lib,engine-api,shared,zbin}`, and 20 bridge and
      script files import it. The map:
      - **`engine-lib`:** Config, PseudoRandom, Util, DetMath;
      - **`engine`:** GameRunner, the executions, `*Impl`;
      - **`engine-api`:** Schemas, the game types, GameMap, GameUpdates;
      - **`shared`:** records and the wire schemas.
      - `Config` becomes `EngineConfig(config, false)`, and `createGameRunner` takes map files.
    - **Simulation changes** (old records desync):
      - trade-ship and train spawn rates, with a boost for small economies (balance);
      - the water graph's edge order, which changes boat paths (rules);
      - a train fix;
      - nukes annex cut-off land again (rules);
      - a MIRV fix;
      - a large nation AI rewrite for Medium and harder, whose boat routes the tribes share (nation AI). Hard and
        harder add 1% beachhead boats, warship squads, nuking back, and a gang-up on a runaway FFA leader.
    - **Unchanged:** `attackLogic`, troop growth, gold, unit costs, warships, SAMs and the intents.
    - **New:**
      - **Ranked FFA** (`getFfaConfig`): 4–10 players on six continent maps at Normal size, nations off, 400 tribes,
        20 minutes; the last human wins.
      - **Ranked Clan Wars:** 2–5 a side, up to 4 clans, donations on.
      - **Game snapshots in the engine:** `GameRunner.snapshot`, `createGameRunnerFromSnapshot`.
      - Lobby sizes now come from the map's land, up to ~210.
      - No mutable statics are left, so `resetGameStatics` goes.
    - **The steps** (`docs/superpowers/plans/2026-10-18-after-4n-v035.md`):
      - move the submodule and `npm ci`;
      - rewrite the imports;
      - re-copy the playlist configs and add Ranked FFA's;
      - fix the tsconfig;
      - record a new replay fixture;
      - typecheck, run the tests and the determinism check, and replay 5 production records in sync.
      - `pins/v0.34.23` stays at the last pre-upgrade commit, since its code cannot import `packages/`.
    - Node 24.18.1 meets the new engines (Node ≥ 24.15, npm ≥ 12.1 via npx).

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
| Player encoder | a token for every player (features + building counts + spatial features pooled over their cells); 2-layer transformer d=128 attending to the 64 most relevant | ~0.4M |
| Scalars + core | MLPs → 512-d (dense or Soft-MoE) | ~0.8M |
| Heads | action type, player pointer, troop share (1–100%, read with the chosen target), arc, value | ~0.3M |
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
| attack | player ptr or unowned land, then troop share 1–100% read with that target | `attack`; mask = shares land border (land-only check) + `canAttackPlayer` |
| retreat | player ptr or unowned land | `cancel_attack` on the largest attack vs that target |
| boat | target (player or unowned), cell, then troop share 1–100% read with that target | `boat` to a tile of that target in/near the cell; ≤3 boats |
| build | unit type, cell | `build_unit` on an own tile; warship → water in a port's water body |
| upgrade | unit type, cell, amount bin | `upgrade_structure` on the own unit of that type nearest the cell |
| nuke | cell, arc bit, amount bin | `build_unit` Atom/Hydrogen; amount ≤ ready silo slots; blocked in immunity / near teammates |
| MIRV | cell | `build_unit` MIRV on an owned target tile |
| move warships | source cell, target cell | `move_warship` for own warships near source; target in the same water body |
| diplomacy | player ptr × {request/accept, reject, break, extend, donate gold, donate troops, embargo on/off, target} | alliance* / donate* / embargo / targetPlayer |
Dropped in v1: emoji, quick_chat, delete_unit, cancel_boat, embargo_all. Masks: cheap approximate grid masks (mostly
derived on the GPU from observation channels); exact check only on the sampled action; failures = logged no-ops.
**Amounts and the arc are deferred (2026-10-06).** 3B sends one missile on the default arc and upgrades by one level.
- Why:
  - two single nukes 0.5 s apart beat a level-1 SAM just like a ×2 salvo;
  - bulk upgrades cost the same per level;
  - humans flip the arc on 0.4% of nukes.
- They return if 4A's intercepted-nuke rate stays high, or with the Phase 5 predictor (amount = interceptor slots on
  the path + 1). One salvo decision is easier for RL to find than a barrage over several decisions.
- Rule set when they return (user decision 2026-10-05):
  - training, self-play and private lobbies follow the server's rules (1–50);
  - in official casual lobbies, a deployment switch holds the bot to what the client offers ("Findings").
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
`docs/superpowers/plans/2026-10-01-phase2-bc-slice.md`; branch `phase2`, merged into `main` on 2026-10-03 after a review
(32 bridge and 50 trainer tests green).
- Data: 16,107 usable ranked 1v1 records started 2026-09-23 18:00 to 2026-09-30 23:10 UTC; 2,355 void ones dropped
  (13%); 765 held out (5%, by a hash of the game ID). BC replayed 7,242 records with 0 errors and 0 desyncs. Labels:
  no-op in 83.4% of windows, a command in 16.6%; commands outside the 5 actions 1.0%, labels our masks forbid 0.6%
  (held-out 0.3–1.5%); the latency queue dropped 3.1% of commands (20-record sample). The profile showed the simulation
  workers, not the trainer, as the bottleneck: no trainer speed fix needed.
- Watching works: public records and our recorded games play in the game's dev client (`trainer.watch`, or a running
  trainer, serves them on 8787). A download manager that intercepts the map `.bin` fetches (Internet Download Manager)
  hangs the replay at "Game is starting...".
  - 2026-10-03: with the client running but nothing on 8787, the client cannot fetch the game and silently falls back
    to its lobby page, where joining by game ID fails.
  - Since then `trainer.watch` also starts the game's dev client, and Ctrl+C stops both, the client's child processes
    included. Before, stopping `npm run dev` left vite and the game server running.
  - All 17 saved games replay in sync at v0.34.22, and play in the client with every hash verified.
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
- PPO run `ppo-bc2` (from the clone: exact KL 0.2, lr 1e-4, no entropy bonus, 16 × 2, ~190 decisions/s with the GPU at
  ~97% and 10.7 of 12 GB, the user's apps holding ~2 GB, CPU ~22%; 5,000 updates from 2026-10-01 21:26 to 2026-10-02
  10:42): the training win rate rose from ~0.6 to ~0.8 in the first ~500 updates
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
- Every evaluation against the frozen clone (`checkpoints/bc/update0060000.pt`; fixed seed `eval`, seats swapped; the
  128-game runs are the first 4 games of each slot of the 512):

  | Snapshot | KL coef | Training curve there | Gate B (games) |
  |---|---|---|---|
  | `ppo-bc` update 500 | 0.05, one-sample | ~0.6 | 0.471 ± 0.022 (512) |
  | `ppo-bc2` update 1000 | 0.2 | 0.885 | 0.730 ± 0.020 (512) |
  | `ppo-bc2` update 5000 (the Phase 2 bot) | 0.2 | 0.825 | 0.773 ± 0.019 (512) |
  | `ppo-bc2` update 5500 | 0.05 | 0.830 | 0.578 ± 0.044 (128) |
  | `ppo-bc2` update 5750 | 0.05 | 0.840 | 0.633 ± 0.043 (128) |
  | `ppo-bc2` update 6000 | 0.05 | 0.770 | 0.576 ± 0.022 (512) |

  Frozen probe (run `probe-frozen`): update 5750 in the same training loop with lr 0 for 250 updates read 0.705 ± 0.032.
- Reading the training curve: each point is the last 200 finished games, ~170 updates at ~1.2 finished games per
  update, so it lags the policy by that much and swings ±0.055 (95%) from luck alone; after a start or resume all 32
  games restart together and short games finish first. It also includes the learner adapting to the games in progress
  (below), so it is a trend line, not a strength measure: snapshots are ranked with `trainer.eval` on the same fixed
  games (paired comparisons).
- Open, for the next planning session:
  - The training curve reads 0.05–0.25 above frozen snapshots (all 5 tested) and missed the weaker-KL decline. Cause,
    confirmed by the frozen probe: a game lasts ~20 updates and the learner trains on the 32 games it is playing, so it
    adapts to them while it plays them; a frozen copy facing new games cannot, and a looser leash allows more of it.
    Ruled out: the eval seeds' map mix (Australia Normal 27% vs 32% expected, Compact 24.5% vs 20%), repeated games
    (512 of 512 and 8,000 of 8,000 distinct) and code differences between eval and training (same seat rows, bf16,
    sampling). For RL: frozen copies on fixed seeds measure strength, during training too; more games per update and
    smaller steps.
  - Why PPO stalled at ~0.77 against the clone: not the KL leash alone (loosening it made Gate B worse). Candidates: the
    within-game adaptation above (the gradient rewards fitting the current games over general play), a single frozen
    opponent (exploits rather than skill), a terminal reward with ~1 finished game per update (at 80% wins, one loss
    every ~4 updates), the 5 actions, and the player-encoder and map weaknesses that Phase 3 fixes.
  - The BC plateau fix (a decaying learning rate, several windows per update), for Phase 3's BC refresh.
  - Phase 4's gate (≥70% against the BC policy) is already met by Phase 2's PPO against its own clone (0.773): set a
    new bar (e.g. against the Phase 2 PPO agent and the Phase 3 clone, or a league rating).
  - Evaluations replay fixed seeds, so their recordings reuse game IDs and overwrite each other in `replays/` (now
    update 6000's Gate B games).
  - The detailed run log, with every ruling and probe number, is the local, git-ignored
    `.superpowers/sdd/2026-10-01-phase2-bc-slice/progress.md`.
  - Taken up by the 2026-10-03 planning session:
    - the scoring command, the plateau fix and the recordings, in Phase 3A;
    - the new gate, the frozen evaluations during training and the league, in Phase 4.

**Phase 3 — Ranked 1v1 in full, then 2v2 (re-planned 2026-10-03)**
Three stages. Each changes one thing and is measured against the previous best on the same 224 held-out games and
the same evaluation seeds. 3A needs no bridge change and tests the encoder decisions on their own; ranked 1v1 has the
data, the environment and the Phase 2 bots; Phase 4A needs only 3A and 3B.
- What holds the Phase 2 bot back:
  - the 5 actions cover 93.7% of human commands (1,500 crawled ranked 1v1 games) but miss what decides the mid and late
    game; per game, both players together: 4.0 defense posts, 3.9 atom bombs, 3.1 silos, 2.9 factories, 1.0 port,
    0.9 SAMs, 0.8 hydrogen bombs, 1.1 upgrades, 0.5 warships. With those and retreat, the coverage is 99.6%;
  - the player summary is one mean over ~400 rows;
  - BC stopped improving after ~10k updates;
  - PPO against one frozen opponent levelled at 0.77.
- **3A — See better, learn better** (ranked 1v1, the same 5 actions; trainer code only):
  - **Scoring**: `trainer.bc --score <checkpoints>` scores the fixed 224 held-out games. It is committed, replacing the
    throw-away script behind the Phase 2 snapshot table. It reports NLL overall, per head (type, attack target, attack
    size, spawn / boat / city cell) and per bucket (spawn, early, mid, late, Compact, Normal). The Phase 2 clone is the
    baseline.
  - **BC plateau fix**:
    - gradient accumulation over k consecutive windows (default 8: ~220 command rows per step instead of ~28). The
      workers are the bottleneck, so bigger batches come from accumulation, not more games;
    - warmup, then a cosine decay of the learning rate;
    - `--init` (a checkpoint's weights with a fresh optimizer);
    - `--resume` refuses the flags it would ignore.
    - Probe: the Phase 2 clone fine-tuned with the fix (~1 h), then scored.
  - **Map channels 22 → 48**:
    - each owner's numbers painted onto its cells (total troops, troops ÷ max, gold, troops attacking me, troops I send
      at it), with the same gather as the density channel;
    - buildings by type and relation: city, port, factory, silo, SAM, defense post, transport and warship × self / ally
      / enemy / bot. A structure's value grows with its level.
  - **Player encoder** (the design above, user decision 2026-10-02):
    - one token per player: its 17 features, its building counts by type, and the CNN features (1/8 resolution)
      averaged over its land;
    - a 2-layer transformer (d 128, 4 heads, absent players masked) over the 64 most relevant players: the seat and
      unowned land, the other non-bot players, the players it can attack or that fight it, then the biggest;
    - 4-head attention pooling queried by the map and scalar summary, plus the self token and one mean each over
      allies, enemies and bots, into the core;
    - the attack pointer reads the tokens.
    - In 1v1 the enemy mean is the opponent itself; in FFA, attention can single out the strong neighbour.
    - Every player keeps its token. With all 403 tokens in the transformer, the speed check failed (120–128
      decisions/s), so it attends to the 64 most relevant (user decision 2026-10-03). The other players skip only the
      attention step: the pointer, the pooling and the means still read them.
  - **MoE core** (decision 5c; user decision 2026-10-03):
    - a switch between a dense core and a Soft-MoE core with 8 experts. With one fused vector per decision it works as
      a router that blends the experts by situation. It costs well under 1% more compute, since the core is tiny next
      to the CNN;
    - A/B: two BC runs that differ only in the core (same data, record order and schedule), compared on held-out NLL
      overall and per bucket, router use per bucket, and decisions/s;
    - keep the MoE unless it is clearly worse or >5% slower. The RL phase should run on the architecture we keep, and
      architecture choices are cheapest in BC. Experts can be added later by copying trained ones (upcycling), when 2v2
      and FFA arrive.
  - **Inputs rounded to 1/64** (exact in bf16 and 8 bits; ≤1/128 change), so PPO rollouts can be stored as 8 bits. At 48
    channels, 1,536 rows then take 3.6 GB instead of 7.2 GB in bf16 (3.3 GB today).
  - **Old checkpoints keep working**: the Phase 2 network stays as `PolicyV1`. `load_net` recognises it and feeds it the
    old 22 channels, rebuilt by summing the new ones (tested against the Phase 2 code). Needed for every match against
    the Phase 2 clone and PPO bot.
  - **Recordings**: evaluations keep their games in `replays/<A> vs <B>/`, and `trainer.watch --dir` serves one folder
    (fixed seeds: the same games played by different bots).
  - **BC refresh**, from scratch (BC learns in ~1 h; network surgery matters for RL-trained weights): ~21,000 records
    after the 2026-10-03 crawls; one pass ≈ 14 h per run (the MoE night took 14.0 h), one run dense and one MoE.
  - **Gate 3A** (on the arm the MoE rule keeps):
    - lower held-out NLL and higher attack-target accuracy than the Phase 2 clone on the fixed scoring set
      (`trainer.bc --score`, which replaced the 224 games; the clone scores NLL 1.039, attack target 37.5%);
    - beats the Phase 2 clone ≥ 0.545 over 512 fixed games;
    - a 10-update PPO probe keeps ≥ 75% of Phase 2's decisions/s.
  - **3A done (2026-10-05): Gate 3A passed on the MoE clone, and the MoE rule keeps it (dense tied on NLL); 3B is
    next.** Task lists:
    - `docs/superpowers/plans/2026-10-03-phase3a-encoder.md`;
    - `docs/superpowers/plans/2026-10-03-phase3a-attention64.md`.
    Branch `phase3`; 79 trainer and 32 bridge tests green. Ledgers with every ruling:
    `.superpowers/sdd/2026-10-03-phase3a-*/progress.md`.
    - Built:
      - 49 map channels as uint8 64ths, with the cell owners and the players' buildings in each row;
      - `PolicyV1`, which keeps the Phase 2 bots playing (inputs rebuilt within 1/128 of the Phase 2 code);
      - the Phase 3 network: the player transformer and a dense or Soft-MoE core;
      - `--init`, `--core`, and `--resume` that refuses ignored flags;
      - BC: 8 windows per step and warmup + cosine over one pass;
      - `trainer.bc --score`, recordings per matchup, and `scripts/speed_check.py`.
    - Sizes: the Phase 2 network 3.0M parameters, Phase 3 dense 3.8M, Phase 3 MoE 5.7M.
    - Rulings:
      - 49 channels, not 48: PLAN's own lists add up to 49.
      - The scoring set is the held-out records of games started before 2026-10-01 (766), 12 workers × 6 games,
        seed "score", 2,000 steps (~288k rows). The Phase 2 script's 224 games depended on the old record list and
        cannot be rebuilt.
      - A structure counts its level, while the Phase 2 network still counts it as 1. Structures stay ≥ 15 tiles
        apart, so a cell holds at most one.
      - One BC pass over today's 19,140 training records is ~32M rows, ~12 h.
    - **BC speed:** 758 rows/s (Phase 2 network) and 751 (MoE network), peak 7.9 GB. The refresh stays
      worker-bound at ~12 h per pass.
    - **PPO speed and memory** (16 × 2 games, 48 steps, `scripts/speed_check.py`):
      - The Phase 2 network: 174 decisions/s at 8 minibatches (11.5 GB peak with 2.5 GB of other programs) and 161
        at 12.
      - The Phase 3 networks overflow the 12 GB card at 8 minibatches by ~1.7 GB. Windows pages that into system
        RAM, and speed falls to 36–44 decisions/s.
      - At 12 minibatches they fit (11.6 GB): dense 120, MoE 128 decisions/s (the two cores cost the same). That is
        74–79% of the Phase 2 network at the same settings, but 63–67% of Phase 2's 190: short of the gate.
      - One 128-row training minibatch: the Phase 2 network 62–64 ms, the Phase 3 network 93 ms. The player
        transformer over all 403 players is the largest extra cost (~16 ms); with 1 layer: 85 ms; with 64 tokens:
        72 ms.
    - **User decision (2026-10-03): attention over the 64 most relevant players, every player keeping its token.**
      Tribes are harmless fillers, so they never need a slot.
    - **Speed with it** (one run, 12 minibatches, 0.8 GB of other programs):
      - the Phase 2 network 172 decisions/s; Phase 3 dense 142 and MoE 138: 80–82% of the Phase 2 network, and
        73–75% of Phase 2's recorded 190;
      - peak 10.9 GB, no spill. At 8 minibatches the Phase 3 networks still overflow (1.2–1.9 GB);
      - MoE is 2.7% slower than dense, within the MoE rule's 5%.
    - **Ruling:** Gate 3A's speed criterion counts as passed. The same-run comparison (80–82%) is the controlled
      one; the old 190 was measured with the old observation code, and the Phase 2 network itself ranged 134–174 on
      this PC today.
    - **Crawl** (after the purge at 15:52 UTC): 927 new games (122 void), 21,011 records, 20,018 of them for
      training. One pass is now 24,981 updates, ~12.5–13 h.
    - **BC plateau probe** (`bc-probe`: the Phase 2 clone plus 2,000 updates with the fix, ~1 h), on the fixed scoring
      set (228 games, 288,000 rows; `runs/score/20261003-200446.json`), Phase 2 clone → probe:
      - NLL 1.039 → 0.989, lower in every head and every bucket (spawn 0.73 → 0.72, early 1.09 → 1.03, mid 1.09 →
        1.04, late 0.62 → 0.61);
      - attack size 1.28 → 1.01 (accuracy 52% → 59%), boat size 1.25 → 1.03 (accuracy 56% → 52%), spawn / boat /
        city cell 9.56 / 5.84 / 7.02 → 9.41 / 5.70 / 7.00;
      - value/outcome correlation 0.546 → 0.607;
      - barely moved: attack target 2.09 → 2.07 (accuracy 37.5% → 38.0%) and action type 0.459 → 0.454.
      - Reading: the plateau was the optimizer. One hour with the fix gained more than the whole spread of the Phase 2
        snapshots (1.065–1.10 from update 10k to 72.7k), and those kept seeing new games, so new data does not
        explain it. The target and type heads look limited by the inputs, which the new observations and the player
        transformer address.
      - The probe is the reference for the nights: a new network that beats the clone but not the probe gained from
        the optimizer, not from its new inputs.
    - **MoE night** (`bc3-moe`: 2026-10-03 20:54 to 2026-10-04 10:57, all 24,981 updates, even pace throughout), scored
      in one run with the clone and the probe (`runs/score/20261004-113904.json`; their numbers repeat 2026-10-03's
      exactly). Phase 2 clone → probe → MoE:
      - NLL 1.039 → 0.989 → 0.843, lower in every head and bucket (spawn 0.73 → 0.72 → 0.67, early 1.09 → 1.03 → 0.90,
        mid 1.09 → 1.04 → 0.84, late 0.62 → 0.61 → 0.53, Compact 0.87 → 0.82 → 0.74, Normal 1.08 → 1.03 → 0.87);
      - attack target 2.09 → 2.07 → 1.61 (accuracy 37.5% → 38.0% → 49.2%);
      - attack size 1.28 → 1.01 → 0.67 (accuracy 52% → 59% → 73%), boat size 1.25 → 1.03 → 0.85 (56% → 52% → 62%);
      - spawn / boat / city cell 9.56 / 5.84 / 7.02 → 9.41 / 5.70 / 7.00 → 8.61 / 4.98 / 6.79 (accuracy 0.7 / 4.1 /
        0.4% → 0.8 / 4.6 / 0.5% → 2.2 / 9.7 / 1.0%);
      - value/outcome correlation 0.546 → 0.607 → 0.696;
      - action type 0.459 → 0.454 → 0.431, accuracy 84.3% → 84.3% → 84.4%, barely above the 84.2% of rows that are
        no-ops: when a human clicks stays mostly unpredictable at 0.5 s.
      - Router (mean expert weights): experts 2 and 6 take 36–45% and 23–38%, experts 5 and 4 most of the rest, and
        experts 0, 1, 3 and 7 get 2–3% each. Late game leans more on expert 6 (38%, 23% at spawn), spawn more on
        expert 5 (18%).
      - Reading: Gate 3A's scoring criterion holds for the MoE arm by a wide margin (NLL −0.196, attack target +11.7
        points). Against the probe, the gain is −0.146 and +11.2 points. The attack target, which the optimizer fix
        did not move, is where the new inputs show most. The probe fine-tuned for 2,000 updates rather than a full pass,
        so part of the −0.146 may come from the longer training.
    - **Gate 3A match, MoE clone vs the Phase 2 clone** (512 fixed games, seats swapped, `--record 1`; run detached
      2026-10-04 12:08–13:07, 59 min, where Phase 2 networks took 35): **0.631 (SE 0.021), PASSED** (bar 0.545). Seat 0
      0.609, seat 1 0.652 (within noise), mean rank 1.37. For scale: on the same games, the Phase 2 PPO bot scored 0.773
      against the same clone. Log `runs/gate3a-bc3-moe.log`; 16 games in `replays/bc3-moe 24981 vs bc 60000/`.
    - **Dense night and the A/B** (`bc3-dense`: 2026-10-04 13:40 to 2026-10-05 03:24, 13 h 44 min, all 24,981 updates,
      no restart), scored with the other three in one run (`runs/score/20261005-034226.json`; the clone, probe and MoE
      repeat 2026-10-04's numbers exactly). Dense vs MoE:
      - NLL 0.8424 vs 0.8430 (dense 0.0006 lower), every bucket within 0.002 except spawn (0.677 vs 0.671);
      - attack target 1.602 vs 1.607 (accuracy 49.4% vs 49.2%); action type, both sizes and the value within 0.001;
      - the cell heads lean MoE: spawn cell 8.72 vs 8.61 (accuracy 2.0% vs 2.2%), city cell 6.84 vs 6.79 (0.4% vs
        1.0%), boat cell 4.99 vs 4.98 (7.3% vs 9.7%). These heads score few rows (spawns, cities and boats are rare
        commands), so this is suggestive only. It fits the router giving spawns their own expert mix;
      - the training curves on the same windows: dense lower by up to 0.016 NLL around updates 2,000–4,000, the MoE
        lower by at most 0.0025 from update 9,000 on.
      - **MoE rule: keep the MoE.** Its NLL is within 0.02 of dense's, and it is 2.7% slower in PPO, within 5% (6.7%
        faster in the speed run before, so the gap is noise). Gate 3A passed on the MoE clone on all three criteria, so
        3A is done and the dense clone needs no match.
      - Reading: in ranked 1v1 BC, the experts add no measurable accuracy overall and cost no measurable speed. The
        published deep-RL gains of Soft-MoE came from feeding it the encoder's output as many tokens (Obando-Ceron et
        al. 2024; Sokar et al., ICLR 2025, found the tokens, not the number of experts, drove them), while ours gets
        one fused vector per decision (decision 5c allowed both). The case for the MoE rests on the mixed modes ahead
        (3C, Phase 6), where the router can blend the experts by mode.
    - **Next:** 3B (below): the troop shares (the bandwidth test kept decision 11). Until their BC night is measured,
      run only the purge, not the crawl, so that night trains on the same records as the 3A clone it is compared with.
      The purge ran
      2026-10-05 02:09 UTC: 3 records removed (21,008 left), none from the scoring set (still 766); the next is due by
      ~2026-10-12.
- **3B — Every human action** (ranked 1v1; bridge and trainer):
  - **3B done (2026-10-11): Gate 3B passed**, with the speed bar accepted at ~72% by the user (Gate 3B, below). The 3B
    clone is `checkpoints/bc3b-v4/latest.pt` (arch v4, SwiGLU experts): 0.700 against the 3A clone. Next, the user's
    choice: 4A (needs only 3A and 3B; first the bandwidth probe on the 3B clone) or 3C (ranked 2v2, needed before 4B).
  - **Reviewed 2026-10-06** against the StockShip brief (`docs/superpowers/specs/2026-10-05-stockship-3b-brief.md`).
    The brief's game facts were checked in the code: corrections are in "Findings", and the ideas are decided below
    and in decisions 4, 11 and 12. Order:
    1. the bandwidth test;
    2. the troop shares with a second network, in one night;
    3. the actions, observations and extractor;
    4. one BC refresh;
    5. Gate 3B.
  - **Bandwidth test** (first; it decides the action bins of decision 11). Three matches of the Phase 2 PPO bot
    against the Phase 2 clone, on the 512 fixed games where it scored 0.773 (~35 min each):
    1. baseline, unchanged: it should repeat 0.773, and gives the meters at today's bandwidth;
    2. throttle (`trainer.eval --a-cooldown 1`): after any action, the bot's next decision is a no-op (at most 1
       action/s, no back-to-back actions, spawn phase exempt). It measures what today's bot loses with less bandwidth;
    3. upward probe (`--a-two-bins`, the user's design): each decision samples one troop action (attack, boat or
       no-op) and one gold action (city or no-op). Each kind keeps the policy's own probability, as bins trained by
       imitation would, so the bot acts as often as before and only acting in parallel is new.
    - Meters, from every match:
      - the share of decisions that act, and that act twice (the probe);
      - the share of actions that follow an action;
      - the share of decisions where the policy puts ≥ 25% on both a troop and a gold action.
    - Rule, set in advance:
      - bins join 3B's actions if the throttle loses more than 0.05 (~2.5 SE) or the probe gains more than 0.05. They
        bring the server's limiter and a command budget into the environment;
      - otherwise decision 11 stays for 3B. The probe then runs again on the 3B clone before 4A, with its ten gold
        actions, and the meters run in 4A (Phase 4).
    - Limits (raised by the user, 2026-10-06):
      - today's bot learned from humans who almost never act in parallel (0.02% of windows), so it never learned to.
        Its only gold action is a city, while 3B has ten;
      - the throttle shows what today's bot loses with less bandwidth, not what a bot would gain with bins;
      - a gain in the probe is strong evidence for bins; a loss is weak, since the bot was never trained for it. Its
        city builds are few per game, so the two kinds of action rarely coincide, and a null result says little.
    - **Results (2026-10-06)**, on the same 512 games (`runs/bandwidth-{baseline,throttle,probe}.log`):

      | Match | Win rate (SE) | Acts in | Notes |
      |---|---|---|---|
      | baseline | 0.793 (0.018) | 17.6% of decisions | 19.1% of its actions follow an action (17.6% if independent) |
      | throttle | 0.820 (0.017) | 15.5% | the cooldown dropped 16.6% of its sampled actions |
      | probe | 0.773 (0.019) | 17.6% | two actions in 0.19% of decisions |

      - The baseline is 0.793, not Gate B's 0.773: since 3A the Phase 2 networks read inputs rebuilt from the new
        channels (within 1/128), so the same seeds play out differently, within noise.
      - The bot put ≥ 25% on both a troop and a gold action in 1 of 648k decisions.
      - **Decision 11 stays for 3B.** The bar was 0.05, and neither match came close:
        - the throttle lost nothing: +0.027, ~1.1 SE of the difference;
        - the probe gained nothing: −0.020, ~0.8 SE.
      - Reading: today's bot is not limited by bandwidth. It acts in one decision in six, rarely twice in a row, and
        almost never wants a troop and a gold action at once.
      - As agreed, this says little about a bot with ten gold actions. So the probe reruns on the 3B clone before 4A,
        and the meters run in 4A (Phase 4).
    - **Rerun on the 3B clone** (2026-10-11, `runs/bandwidth3b.cmd`, `runs/eval/20261011-170544.json`): the clone
      with a troop and a gold action per decision against the same clone, 512 games. Its baseline is 0.5 by symmetry,
      so the bar was 0.55.
      - **0.482** (SE 0.022; seats 0.480 / 0.484): −0.018, ~0.8 SE. **Decision 11 stays for 4A.**
      - It acts in 14.4% of decisions, twice in 0.19%; 21.4% of its actions follow an action.
      - It put ≥ 25% on both a troop and a gold action in **none** of 815,407 decisions, so even with ten gold
        actions it never wants both at once. The meters keep running in 4A, since PPO may change that.
  - **Troop shares read with the target** (first, measured alone). Found 2026-10-04: the attack and boat sizes are
    drawn independently of the target and the landing cell (against decision 3), so the bot cannot size an attack to
    the player it attacks; and its five sizes (10/25/50/75/100% of current troops) fit human play badly.
    - Measured on 60 replayed ranked 1v1 games (`scripts/troop_shares.ts`): 9,405 attacks, 605 boats, and 6,852
      follow-up clicks within a second of the player's previous command (their shares drift, so they are kept apart):
      - attacks: 1–9% of troops 5.3%, 10% 7.4%, 11–19% 12.0%, 20% (the client's default) 41.1%, 21–49% 29.4%, 50% or
        more 4.8% (75% or more: 1.9%);
      - boats: 1–9% 11.4%, 10% 6.9%, 11–19% 8.8%, 20% 30.4%, 21–49% 36.0%, 50% or more 6.4%;
      - snapping to the five sizes moves an attack by 4.7 points on average (28% of it), a boat by 5.1: the default 20%
        is learned as 25%;
      - follow-up clicks: 12.0% send 1–9%, 5.2% exactly 1%: humans trickle troops into a front;
      - 84% of attacks fall within 0.25 points of a whole percent (troops change between the click and its execution).
    - The game: the server takes any troop count; the client sends whole percents of current troops (slider 1–100,
      keys ±10 by default, default 20%); a land attack on a target already under attack merges into it. Troop growth,
      (10 + troops^0.73 / 4) × (1 − troops / max), peaks at 42% of max, and 38–46% keeps ≥ 99% of the peak.
    - Design:
      - 100 choices, 1–100% of current troops. The label is the human's share rounded to a whole percent, with soft
        labels (a little weight on the neighbouring percents);
      - attack: the share reads the chosen target's token (in BC, the human's target). The exact entropy and KL compute
        the shares for every target, which is cheap next to the CNN;
      - boat: the share reads the token of the landing cell's main owner, so the exact entropy and KL sum the boat
        map's probabilities per owner;
      - the bridge takes a percent; 5-size checkpoints map theirs to 10/25/50/75/100%.
    - Measured alone: one BC night with the core 3A keeps, scored against the 3A clone (the 100 shares summed back into
      the five sizes' ranges, for a like-for-like NLL), then 512 games against it. Then the rest of 3B.
    - **A second network in the same night: real MoE experts** (user decision 2026-10-05; the brief's idea F).
      - Each of the 8 experts becomes a 2-layer SwiGLU network, 512 → 1024 → 512, instead of one 512 × 512 layer.
        That adds 12.6M parameters and ~12.6M multiply-adds per decision, ~1% of the CNN's ~1.4 GMAC on the 192 × 256
        grid: decision 6's brain growth. It is a new core value in the checkpoint, so old checkpoints keep working.
      - One BC run trains both networks on the same replayed windows, paired row by row. BC is limited by the replay
        workers, not the GPU: the Phase 2 and Phase 3 networks ran at 758 and 751 rows/s although their training costs
        differ.
      - A speed check comes first: two networks must keep ≥ 90% of one network's rows/s. Otherwise the second network
        takes its own night.
      - Two comparisons, each differing in one thing:
        - shares alone against the 3A clone gives the shares' effect;
        - shares with the new experts against shares alone gives the experts' effect.
      - Keep rule, like the MoE rule: keep the new experts unless the old core's NLL is more than 0.02 lower, or they
        make PPO more than 5% slower.
      - Expectation: the deep-RL Soft-MoE paper behind decision 5c (Obando-Ceron et al. 2024) also used single-layer
        experts, and its one-vector-per-sample input did worst. So the 3A tie more likely comes from the single fused
        vector than from the experts' depth (3A reading). The routing test belongs in 3C.
    - The 3A nights keep the five sizes: both arms share them, so the dense / MoE comparison stays fair, and changing
      the heads mid-A/B would cost the running night and two more.
    - **Code (2026-10-06)**, `docs/superpowers/plans/2026-10-06-phase3b-troop-shares.md` (102 trainer + 35 bridge
      tests):
      - actions and labels carry a whole percent (1–100); the 5-size networks send 10/25/50/75/100%, the same troops
        as before;
      - the share head (arch v3): per kind, a 64-wide hidden layer reads the core state and one player's token, then
        100 logits. It runs on every token, so the entropy and the KL to the clone stay exact;
      - soft labels: 0.9 on the human's percent, 0.05 on each neighbour;
      - scores: `bc/nll` like for like (shares summed into the five sizes' ranges: 1–15, 16–35, 36–61, 62–86 and
        87–100%), and `bc/nll_shares` at full resolution;
      - `trainer.bc --twin moe-swiglu` trains the experts' network on the same windows, from the same weights outside
        the core (`checkpoints/<run>-moe-swiglu`). Parameters: 5.8M with the MoE core, 16.3M with SwiGLU experts.
    - Rules fixed before the night (2026-10-06):
      - speed check: the pair keeps ≥ 90% of one network's rows/s, or the twin gets its own night;
      - shares: they stay unless they lose the 512 games to the 3A clone by more than 2 SE (below ~0.456). A loss that
        large points to a bug, which is found before the rest of 3B;
      - experts: the keep rule above, on `bc/nll_shares` (both networks have 100 shares) and the speed check's PPO
        runs.
    - Checks (2026-10-06, `runs/3b-score.log`, `runs/3b-speed.log`):
      - score: the 3A clone scores `bc/nll` 0.8425 under the percent labels (0.8430 before, same 288,000 rows). Only
        its attack size moved (0.6739 → 0.6705): a few rounded percents cross a size boundary;
      - speed (`--ppo-minibatches 12`): BC 779 rows/s with one network, 549 with the pair (71%, below the 90% bar), so
        **the twin gets its own night**. One network's GPU work takes ~70% of the time the workers need to replay a
        window, so two networks make the GPU the bottleneck;
      - PPO: Phase 2 network 178 decisions/s, v3 MoE 150 (79% of Phase 2's 190; Gate 3B's bar is 75%), v3 SwiGLU
        experts 147 (98% of the MoE core, within the keep rule's 5%). Peak VRAM 11.4–11.5 GB of 12.3 (2.1 GB held by
        other programs). Both margins are thin before step 3 adds 9 heat maps and the coverage channels.
    - Night 1 (`bc3b`, started 2026-10-06 20:32): the GPU driver failed at 23:40 (nvlddmkm 153), at update ~5,225.
      The run was then restarted from update 0 under the same name by mistake (01:03), which made the curves step
      back. It was stopped at update 81; `latest.pt` was restored from `update0005000.pt`, and the run resumed from
      update 5000.
      - Fixes: a new run now refuses a folder that holds a run, and a resume hides the curves' points logged after
        the last checkpoint (TensorBoard's purge_step).
      - The launch command is one loop that starts or resumes, safe to rerun (CLAUDE.md).
    - **Results (2026-10-07)**: night 1 (`bc3b`, shares with the 3A core) finished at 12:50.
      - Score (`runs/score/20261007-130049.json`, the same 288,000 rows), 3A clone → shares:
        - `bc/nll` 0.8425 → 0.8364;
        - attack size 0.6705 → 0.6324 (accuracy 73.6% → 74.5%); boat size 0.8458 → 0.8466;
        - attack target 1.6065 → 1.6020; type 0.4310 → 0.4304;
        - the rare cell heads slightly worse: spawn +0.10, city +0.08, boat +0.03;
        - its own 100-share NLL: 1.126 overall, attack 2.55, boat 2.92.
      - Match (`runs/eval/20261007-133639.json`, 512 games): **shares 0.660 (SE 0.021)** against the 3A clone (seat
        0 0.637, seat 1 0.684).
        - **The shares stay**: the bar was 0.456.
        - 0.660 already clears Gate 3B's 0.545 against the 3A clone, a bar the full 3B clone must pass.
      - Reading: a small gain in NLL (−0.006 overall, −0.038 on attack sizes) gave a large gain in play. Sizing an
        attack to its target decides fights, yet it is a small part of the loss, so NLL understates such decisions.
        The user expected a big leap. My caution (a modest gain under imitation) was wrong.
      - The twin's own night (`bc3b-moe-swiglu`) ran 2026-10-07 14:09 → 2026-10-08 04:17: all 24,981 updates in one
        sitting.
        - It survived one driver event (nvlddmkm id 13, not the fatal 153).
        - Its checkpoint holds 16.3M parameters (195 MB).
      - **Rescore** of the three networks, 2026-10-08 (`runs/score/20261008-151318.json`), bc3-moe / bc3b /
        bc3b-moe-swiglu:
        - `bc/nll` 0.8425 / 0.8364 / 0.8386; `bc/nll_shares` 1.1255 (bc3b) against 1.1296 (SwiGLU).
        - **The SwiGLU experts stay**: 0.004 behind, inside the 0.02 fixed in advance, and their PPO speed passed (98%
          of the MoE core). They lead on nothing, though: spawn cell NLL 8.89 against bc3b's 8.72.
        - Closeness, the first numbers (close for all three): city 3.4–3.7% within 16 tiles, 177–181 tiles expected
          distance; boat 9.4–10.6%, 322–349 tiles; spawn 1.3–2.0%, 396–417 tiles. The heat maps are spread wide.
          Reading them needs a baseline, still missing: what a uniform pick over the legal cells gets.
      - The heat maps are also scored by closeness (the user's suggestion, 2026-10-07). Exact-cell accuracy and NLL
        treat a near miss like a far one, so two closeness numbers are added:
        - `bc/*_cell_near`: the chance the bot's cell lands within 16 tiles of the human's;
        - `bc/*_cell_dist`: its expected distance in tiles.
        All three were rescored after the twin's night (above). A spatial soft label (the human's cell spread over
        its neighbours, like the shares' 0.05) is a candidate for step 4's BC refresh, decided on these numbers.
        - **Deferred to the next BC refresh (user decision 2026-10-10)**: the closeness numbers cannot decide it.
          - A cell is 8 × 8 tiles, so 0.05 on each neighbour moves at most 10% of the mass by one cell, inside the
            16-tile circle: near rate and expected distance barely move by construction (≤ ~1 tile).
          - What it might change is how fast the rare maps learn, which only two nights and a match would show.
          - Adding it would put a second untested change into the night Gate 3B judges, and would raise the exact-cell
            NLL by up to 0.1 nats, so step 4's cell scores would lose their like-for-like comparison.
          - Step 4's validation logs every map's near rate next to its uniform baseline (`*_cell_near_uniform`), and
            its score gives the final numbers: a map that stays near uniform is the evidence for a spatial label (or
            another fix) in the next refresh.
  - **Actions**, as flat types:
    - today's 5;
    - build defense post / silo / factory / port / SAM / warship;
    - atom bomb and hydrogen bomb;
    - upgrade: the own structure in the chosen cell, on its own heat map (see the decisions below);
    - retreat: a pointer; it cancels the largest attack on that target.
    - One heat map per cell action (12 instead of 3).
    - Nukes use the default arc and one missile, and upgrades are by one level ("Action space": amounts deferred). In
      2,101 ranked 1v1 records:
      - 98.7% of 9,255 human nukes were single missiles, and 0.4% flipped the arc;
      - 10% came within 2 s of the same player's previous nuke. Humans build barrages from single clicks, and so can
        the bot at one action per 0.5 s;
      - 92% of upgrades were by one level, and upgrades are 0.3% of commands.
    - Left out of the loss: alliances, embargo, emoji, chat, donations, cancel boat, warship moves and MIRV (0.4% of 1v1
      commands).
    - **Task list, part 1**, `docs/superpowers/plans/2026-10-07-phase3b-actions.md`: written 2026-10-07, reviewed by
      the user 2026-10-08, coded 2026-10-08 (3dbc878..; 41 bridge and 114 trainer tests pass).
      - Code: 15 types, 12 heat maps (spawn, boat, nine builds, upgrade) and a retreat pointer (arch v4); masks in
        the bridge (types) and on the GPU (cells, the upgrade map exact); a cell type needs a legal cell; the extractor
        labels every command except diplomacy, chat, warship moves and MIRVs; the v1–v3 networks keep their 5 types
        and 3 maps, and BC scores each network only on the rows its types express (`labels/beyond_net`).
      - Coverage check (2026-10-09, `runs/score/20261009-030123.json`, bc3b on the fixed rows): Gate 3B counts
        commands, not rows. `labels/out_of_set_commands` is 0.60%, against the bar of 0.5% and ~7% with the five
        actions. Other labels:
        - `labels/out_of_set` 0.10% of rows (1.2% before);
        - `labels/unmapped` 0;
        - `labels/illegal` 0.59% of rows;
        - `labels/beyond_net` 1.04%: the new actions' rows, which a v3 network cannot express.

        The share is heavy-tailed.
        - The extractor's label stream, emulated on all 766 held-out records (its queue and windows, no simulation),
          gives 0.40%: 927 of 231,453 commands.
        - One game alone holds 153 of the 927 (123 troop donations and 21 alliance requests, of its 286 commands).
        - Random 228-game subsets (the score replays 228) read 0.37% at the median, 0.58% at the 90th percentile and
          0.64% at the 95th.

        Outside the set, as a share of all commands:

        | Commands | Share |
        |---|---|
        | alliance requests | 0.16% |
        | troop donations | 0.06% |
        | embargoes | 0.04% |
        | emoji | 0.04% |
        | breaking, rejecting or extending alliances | 0.04% |
        | cancelled boats | 0.03% |
        | warship moves | 0.02% |
        | quick chat | 0.01% |
        | MIRVs | 0.006% |
        | deleted units | 0.004% |

        **Decided by the user (2026-10-09): Gate 3B's coverage is read on all held-out records: 0.40%, passed.** Which
        games the 228-game subset holds moves it by ±0.15 points. `scripts/coverage.py` gives the number in seconds,
        and `trainer/test_env.py` checks it against the extractor's labels on the fixture record.
      - Decisions:
        - retreat gets its own pointer;
        - **self-nukes stay legal**: top players nuke their own buildings before a capture, so the enemy gets nothing,
          and an over-extended front leaves their own land open. The game refuses only a teammate's land;
        - a cell action is legal only while its heat map has a legal cell;
        - new networks are arch v4;
        - upgrades point at a cell of their own heat map, not at a list of buildings (the user asked, 2026-10-08).
          Masked to the buildings the seat can upgrade, the map is that list's pointer, and each score reads the map
          around its building. The mask is exact: own, finished, and the gold for its type;
        - Gate 3B is judged on match strength, and a network is scored only on the rows its actions can express.
      - **Task list, part 2** (the observations), `docs/superpowers/plans/2026-10-09-phase3b-observations.md`: written
        2026-10-09 and approved by the user the same day, with the coverage counts by side (ours, theirs) rather than
        by relation. It measures v4's PPO speed and memory first (Part 1 took the heat maps from 3 to 12), against a
        rule fixed in advance.
        - Coded 2026-10-09..10 (6df6dc1..83a4db9; 43 bridge and 126 trainer tests pass). Its decisions:
          1. zones by side, ours (self and friends) and theirs (enemies and bots), not by relation:
             - the two are the same in 1v1;
             - bots never build SAMs or posts, and never launch nukes;
             - a friend's SAM also covers our land;
             - it saves 3 channels;
          2. scales: level and loaded missiles / 16 (exact up to 63). A SAM's zone counts its loaded interceptors
             / 16, a post's or a nuke's 1;
          3. under construction: the level channel reads 0 while the type channel reads 1;
          4. unit costs are scalars, gold over each build's current cost, capped at 4 and divided by 4. Rule values
             reach the network through the zones' radii;
          5. v4 is redefined to read the new inputs (no v4 checkpoint existed);
          6. a cell is in a zone when its centre lies within the radius:
             - the arithmetic stays in integers (positions doubled);
             - zones also cover padding cells, which the network marks with VALID.
        - **Task 1's speed check** (2026-10-10, `--ppo-minibatches 12`; a video playing in Chrome kept the GPU ~20%
          busy, the same for every arm):
          - PPO: Phase 2 network 164.5 decisions/s, v4 11.2: **7%** of it (bar: 75%).
          - Memory: 1,447 MiB in use before the runs; v4's peak 11,965 MiB, so **10,518 MiB its own** (rule: ≤ 9,400).
            The peak comes from nvidia-smi, which cannot read past the card's 12,282 MiB: v4 filled the card, Windows
            paged the rest into system RAM, and the run got 13× slower (v3's spills cost 2–4×).
          - BC: v4 611 rows/s, peak 9,880 MiB (v3: 779 rows/s with one network).
          - **Verdict: Task 2 runs.** The masks are probably not the whole excess. Each learning minibatch builds its
            12 heat maps' distributions in float32 (128 rows × 12 × 49,152 cells, 0.3 GB a tensor), keeps the exact
            KL's and the entropy's for the backward pass, and v3 had 3 maps.
        - **Task 2** (masks as bits, 16825a1; PPO now logs its share of time in rollouts and the allocator's peak,
          which reads past the card, f6d8150), re-measured 2026-10-10 with `--skip-bc`:
          - 12 minibatches: v1 152.7, v4 40.6 decisions/s (27%). v4's allocator peaked at 11,992 MiB, so it still
            spilled, and learning took 99% of the time.
          - 16 minibatches (the plan's fallback): v1 170.6, v4 99.3 decisions/s (**58%**), with no spill (allocator
            10,460 MiB; own 11,172 MiB, against the rule's 9,400).
          - **Neither holds the rule, so the plan stops here for the user.**
          - Profile of one learning minibatch (96 rows, the ranked grid, every type legal; the log-probability, the
            entropy and the exact KL as PPO computes them):
            - v3: 104 ms, 86 of them the network;
            - v4: 155 ms, 96 of them the network.
            - v4's 12 heat maps cost 59 ms of distribution math, v3's 3 cost 18.
          - Rollouts also take 7.3 s per update, against v1's 5.2. Not yet split between GPU work and the random
            network's play: it acts on most decisions and builds every unit type.
          - Triton is not installed, so `torch.compile` cannot fuse that math.
        - **Optimization** (the user's call, 2026-10-10: optimize, then re-measure; all runs at 16 minibatches):
          - The heads take their distributions from one log_softmax (f7e81f5): v4's heat-map math went from 59 to 36
            ms a minibatch, with the same values and gradients. PPO's entropy enters the graph only with a weight
            (c294f13). Result: 52%, because v4 still spilled (other programs held 1.2 GB).
          - Expandable allocator segments, now the trainers' default (988c6d8): v4's reserved peak fell from 10.5 to
            9.9 GB, the spill stopped, and v4 reached **71%**.
          - Emptying the cache before learning: 9,552 MiB reserved, 8,155 in use, own 9,673 MiB; 69%.
          - Per update, v4 against v1:
            - learning 6.6 s against 3.9;
            - the rollouts' GPU work 1.9 s against 1.2;
            - simulation and observations 5.0 s against 4.5 (a random network plays more than the clone).
          - Still open:
            - the memory rule: own 9,673 MiB against 9,400, before Part 2's ~0.8 GB;
            - the 75% bar.
        - **The user's calls (2026-10-10):**
          - The rollout buffer moves to system RAM (0b8c7ff, `HostRows`): ~4.3 GB off the card.
            - Learning is ~4% slower in a synthetic epoch: 147 ms a minibatch against 138.
            - 96- and 128-row minibatches cost the same per row and 192 rows 49% more, so PPO stays at 12 minibatches.
          - The 75% bar is judged at Gate 3B, with the BC-trained v4.
        - Then Task 6 (the 8 channels, 83a4db9) and Task 8's docs.
        - **Final speed check** (2026-10-10, `--ppo-minibatches 12`, every input; other programs held 1,147 MiB):
          - PPO: v1 153.0 decisions/s, v4 112.4 (**73%**; the bar is judged at Gate 3B).
          - Memory: v4 had 5,978 MiB in use, down from 8,155 before the RAM buffer, and that includes Part 2's
            channels. It still reserved 9,698 MiB (own 9,861) because each epoch took a new copy stream, and PyTorch
            caches freed memory per stream, so every epoch's blocks stayed reserved. Fixed in b86fb53.
          - BC: v4 642 rows/s (611 before the optimizations), peak 9,871 MiB.
        - Re-measured after the fix (`--skip-bc`, other programs held 1,104 MiB):
          - v1 148.9 decisions/s, v4 108.1 (73%);
          - v4: 8,474 MiB reserved, 5,978 in use, **own 8,692 MiB, under the 9,400 rule** with every input.
        - **Part 2 is done.** Next is 3B step 4, the BC refresh: a v4 network with SwiGLU experts, one night at ~640
          rows/s.
  - **Masks**: action types in the bridge (gold against each unit's current cost, a ready silo, a port); cells on the
    GPU from the channels, and the upgrade map from the unit records (exact). The exact check stays in the worker,
    and failures are counted no-ops.
  - **Observations** (coded 2026-10-10, task list part 2): nukes in flight and their targets, SAM and defense-post
    coverage, silo readiness, and unit costs as scalars. The bridge sends nuke targets and unit levels.
    - Buildings need their state too (found 2026-10-08): one under construction reads as a finished level 1 today,
      and levels from 4 up saturate at 3.98, the top of the inputs' 64ths byte.
    - SAM coverage counts the ready interceptors on each cell (the brief's idea D): over the covering SAMs, level minus
      missiles reloading.
      - It counts by side: ours (self and friends) and theirs (enemies and bots).
      - Decision 4 first said "by relation"; part 2's decision 1 changed it, with its reasons.
    - SAMs under construction are left out, and each SAM's current range is used (it ramps after an upgrade).
    - Coded as 8 channels and 10 scalars:
      - channels 49–56: level, loaded missiles, and the SAM, post and nuke zones of each side;
      - scalars 16–25: the gold against each build's cost, and the loaded silo missiles.
  - **Extractor**: every intent type is mapped. A wrapper masks the new types for 5-action checkpoints. Then a BC
    refresh.
  - **Step 4's records** (after the v0.34.24 upgrade, 2026-10-10; **user decision 2026-10-10: option 1**). `data/`
    holds 21,008 records of v0.34.17–v0.34.22, and at the new pin they desync within ~500 turns (decision 14's log).
    Options:
    1. **Chosen:** the BC night and its scoring run from `pins/v0.34.23` on these records.
       - Every game replays in full on its own simulation.
       - They are the 3A night's records (less 3 purged), so the scores stay paired with the 3A clone and probe.
       - Gate 3B's matches, PPO and evals run in the main checkout on v0.34.24. No code change.
       - The new rule only touches attacks with nothing to conquer, and labels our masks forbid already stay out of
         the loss (Phase 2, b), so the old records teach the same play.
    2. v0.34.24 records only: crawl from 2026-10-06 19:00 UTC (6,899 two-player games so far, ~2,100 more a day), BC
       at the new pin.
       - A third of the data today.
       - A new scoring set (the fixed one holds old records), so no paired comparison with the 3A networks.
       - `trainer.bc` needs a version filter: it takes every indexed record.
    3. Both, each at its own pin: the most data, but `trainer.bc` would need two worker groups split by record
       version. A candidate for the next refresh.
    - No crawl until the night is done: a crawl would add v0.34.24 records to `data/`, and `trainer.bc` would mix them
      into the run at the old pin.
  - **Step 4, the BC night** (`bc3b-v4`: arch v4, SwiGLU experts, from scratch, seed 1, 14 workers × 6 games, one pass;
    started by the user from `pins/v0.34.23` with the start-or-resume loop, 2026-10-10):
    - the purge runs first, so the night trains on purged records; while its last sync is under 7 days old (before
      2026-10-12 01:09 UTC) it is one API call;
    - ~17 h at the speed check's 642 rows/s (the v3 nights: ~14 h at ~750);
    - first hours: every map's `val/bc/*_cell_near` should rise above its `*_cell_near_uniform`, and `val/bc/nll` fall;
    - then the score from the pin (the 3A clone, bc3b, bc3b-moe-swiglu, bc3b-v4), and Gate 3B in the main checkout.
    - **Done 2026-10-10 06:10 → 21:44** (15.6 h, ~600 rows/s): all 24,976 updates in one sitting, no driver event.
      Its last validations (held-out games, 5 averaged; v4's `val/bc/nll` 0.977 covers rows older networks cannot
      express, so it is not comparable with theirs): every map's near rate beats its uniform baseline, by 1.3× (port)
      to 21× (boat, warship); the maps of whole-map targets far more (atom 0.29 against ~0). Weakest against uniform:
      port 1.3×, upgrade 1.5×, city, silo and factory ~2×. `val/labels/illegal` 0.47%.
    - **Score** (2026-10-10, from the pin, `runs/score/20261010-232613.json`, the same 288,000 rows), bc3b-v4 against
      bc3-moe / bc3b / bc3b-moe-swiglu:
      - `bc/nll` 0.950 against 0.8386 is not like for like: v4 is scored on the rows of all 15 actions
        (`labels/beyond_net` 0 against 1.04%), and the new maps' cells are rare and hard (NLL 4.6–7.5).
      - On the heads all four share, v4 is level or best: attack target 1.5998 (bc3b 1.6020), attack share 2.5518
        (2.5532), attack size 0.6288 (0.6324), city cell 6.7884 (6.8667), outcome correlation 0.700 (0.697). It is
        slightly behind bc3b on boats (cell 5.014 against 5.004, share 2.933 against 2.919) and on the spawn cell
        (8.896, like the SwiGLU twin, against 8.718).
      - Near rate against a uniform pick over the legal cells: atom 26% (~650×), hydrogen 1.8% (58×), warship 2.3%
        (29×), boat 10% (16×), spawn 1.4% (13×), SAM 3.0% (6.5×), defense post 5.2% (4.5×), silo 2.6×, factory
        2.5×, city 2.4×, port 8.5% (1.2×), upgrade 34% (1.15×; the exact building 40%). Retreat target 91%.
        - No map stays at uniform. The two closest, port and upgrade, have few legal cells (coast; the seat's own
          upgradable buildings), so a uniform pick is already near, and a spatial label could not spread onto
          illegal neighbours: the deferred spatial label still has no case.
      - It acts in 17.2% of rows, the humans in 16.7%.
      - Router: one expert takes 54–77% of the weight and a second 17–33%; the other six get ≤ 8% (bc3b: two at
        27% and 38–45%). For 3C's routing test (upcycling, a load-balancing term or router noise).
    - **Gate 3B's run** (`runs/gate3b.cmd`, started 2026-10-10 23:26 via `explorer.exe`, main checkout, v0.34.24):
      1,024 games against the 3A clone (the 0.545 bar, invalid actions from the eval's new `invalid_share`, and the
      1,000 games without errors), then the PPO speed probe with this clone (`speed_check.py --v4`).
  - **Gate 3B**:
    - commands outside the action set < 0.5%: **passed**, 0.40% on all held-out records (`scripts/coverage.py`; the
      user's reading, 2026-10-09; 0.60% on the score's 228 games);
    - invalid actions < 1%;
    - 1,000 ranked games without errors;
    - beats the 3A clone ≥ 0.545;
    - a 10-update PPO probe keeps ≥ 75% of Phase 2's decisions/s, compared within one run as in Gate 3A. 3B adds 9 heat
      maps, coverage channels and maybe 12.6M parameters.
      - The user decided on 2026-10-10 to measure it with the BC-trained v4.
      - A network with random weights reached 73%, but it acts on most decisions and costs more simulation than a
        trained one.
      - If the bar is missed, one exact optimization is left: compute only the heat maps of each row's legal types.
    - **Results (2026-10-11, `runs/gate3b.cmd`, main checkout at v0.34.24; the 3B clone bc3b-v4 against the 3A clone
      bc3-moe):**

      | Criterion | Bar | Result | |
      |---|---|---|---|
      | commands outside the set | < 0.5% | 0.40% (2026-10-09) | passed |
      | invalid actions | < 1% | 0.11%: 263 of its 249,702 actions | passed |
      | games without errors | 1,000 | 1,024, in one attempt (exit 0) | passed |
      | against the 3A clone | ≥ 0.545 | **0.700** (SE 0.014; seats 0.705 / 0.695) | passed |
      | PPO speed (10 updates, 12 minibatches) | ≥ 75% | **71%**: Phase 2 network 145.2 decisions/s, v4 103.2 | missed; **accepted** (below) |

      - Match (`runs/eval/20261011-011546.json`, 1,024 games, 16 recorded): the clone acts in 17.2% of decisions, and
        22.9% of its actions follow an action. bc3b (shares alone) scored 0.660 on 512 games at v0.34.23: +0.04, ~1.6
        SE apart, with the game version changed in between.
      - Speed (`runs/gate3b-speed.log`): v4's own memory 8,826 MiB, under the 9,400 rule; other programs held 1,892
        MiB. The trained network is no faster than random weights (73% twice): the simulation now takes the same time
        per update as the Phase 2 network's. In units of the Phase 2 network's update time, v4 spends 0.73 learning
        (Phase 2: 0.40) and 0.18 on the rollouts' GPU work (0.12). 75% needs ~5% off v4's total time, ~10% of its
        learning.
      - **The fallback** (the user's call, 2026-10-11: measure, then optimize):
        - Measured first (1,600 steps of clone games, 102,400 rows): a row can take **4.65 of the 12 cell types** on
          average. Boat 95%, defense post 83%, factory 60%, city 48%, port 46%, upgrade 42%, silo 31%, SAM 20%, atom
          19%, warship 12%, hydrogen 6%, spawn 4%.
        - Coded (234ad4c): `ActionDist` normalizes only those maps (`_Maps`). It is exact: an untakeable type has
          probability 0, so its map adds nothing to the log-probability, the entropy or the KL. A test compares
          values and gradients with torch's Categorical over every map.
        - A/B against the old code (one minibatch of 96 rows, those masks): v4's learning −9% (125–132 ms against
          136–144). Rollout steps are unchanged for v4 and 6% slower for the Phase 2 network (the packing waits for
          the GPU once per distribution). This predicts ~75%.
        - The probe after it (02:23, `runs/gate3b-speed2.log`) read **79%** (v1 121.6, v4 96.5), but both arms ran
          slower than at 01:15, the Phase 2 network by 16%: far more than the code explains, so probably another load
          on the GPU during its run. A clean re-measure with the PC idle decides.
        - **Re-measured with the PC idle** (rule fixed before: the mean of two probes; `runs/gate3b-speed3{a,b}.log`):
          81% (v1 118.8, v4 96.1) and 80% (123.0, 98.9). **The pass is an artifact**: with the PC idle, both networks
          stay slower than with the old code (v1 145–153, v4 103–112). The ratio rose because the Phase 2 network
          slowed more (−17%) than v4 (−5%). v4 itself got slower.
          - Cause unknown. First suspect: picking the used pairs with boolean masks makes the CPU wait for the GPU,
            ~12 times a learning minibatch. But a loop shaped like PPO's (the RAM buffer and its thread, `.item()`
            after each step, rollout steps; no simulation) shows no difference between the old code, the packing and
            a one-wait version: Phase 2 network 87–92 ms a minibatch, v4 145–156, rollout steps equal. Nothing else
            used the CPU or the GPU afterwards either (the replay viewer idle, 19.6 GB of RAM free).
          - Not counted as a pass. The one-wait version stays (fewer waits, same values). Next: probes of the old and
            the new code back to back, in the same conditions.
        - **Old against new code, back to back** (PC idle; `runs/gate3b-speed4-{new,old}.log`):
          - 03:01, the new code (one wait): Phase 2 network 122.7, v4 96.2: 78%;
          - 03:06, the code before the packing (2b83c02, swapped in, then restored from git): 118.9 and 95.3: **80%**.
          - **The code is not the cause.** Since ~02:20 the PC runs the Phase 2 network ~18% slower and v4 ~7%
            slower than at 01:15, with either code, for no visible reason.
          - The packing changes nothing measurable in PPO (v4 96.2 against 95.3). Its 9% showed only in isolated
            benchmarks.
          - The ratio depends on those conditions. The Phase 2 network spends 60% of its time in rollouts, which
            are mostly CPU, so a slowdown there hits it harder. When it runs at its usual 145–153 (01:15 and every
            check on 2026-10-10), the ratio is 71–73%.
          - **Reading: the bar is missed by ~3 points in normal conditions.** The 78–81% of tonight's probes come
            from the slow conditions. The user decides how the gate reads this (asked 2026-10-11 ~03:10).
      - **User decision (2026-10-11): accept ~72% and close Gate 3B.** 3B's 15 actions and inputs cost ~28% of PPO's
        speed against the small Phase 2 network; the one exact lever left gained nothing; the clone beat the 3A clone
        0.700 and passed every other bar. **The packing is reverted** (no measurable gain, less code; in git as
        234ad4c and a0d5821). Lesson: credit a speed change only from old and new code probed back to back.
      - **PPO speed is 4A's first work item** (reviewed with the user, 2026-10-11). The packing cut v4's GPU work by 9%
        in isolation and gained nothing in PPO. The CPU's launch time per minibatch roughly equals the GPU's (3B step
        3, part 2), so the real loop is probably bound by kernel launches: cutting GPU work alone cannot help. (The
        4A profile refuted this for learning: Phase 4, 4A, "Speed work".)
        Profile the real loop first (torch.profiler), then try, each judged by old and new code probed back to back:
        1. fewer launches: a CUDA graph of the fixed-shape learning step, or `torch.compile` through the community
           `triton-windows` build (a new dependency). Possibly 20–40% of learning, with Windows risk;
        2. the 12-map entropy recomputed after every minibatch only for the log: log it once per update;
        3. the frozen clone's outputs for the KL computed once per update instead of once per epoch (a third of
           those passes at 3 epochs, ~2 GB more pinned RAM).
- **3C — Ranked 2v2 and a nations check** (before Phase 4B; after 4N since 2026-10-15, Phase 4 "4N"):
  - a 2v2 setup (a copy of `get2v2Config`), the team relation, and troop and gold donations;
  - a crawl of ranked 2v2 (~670 games/day), then BC on 1v1 + 2v2 with the mode as an input;
  - Compact FFA against Easy and Medium nations, as an evaluation only.
  - **The routing test** (the brief's idea G):
    - the MoE core against a dense core with the same parameters and compute, e.g. 8 SwiGLU experts of 512 → 1024 →
      512 against one dense SwiGLU layer of 512 → 8192 → 512, so that only the router's gates differ;
    - 1v1 + 2v2 is the first place where the router can blend by mode. In 1v1 alone, a tie keeps the MoE by rule;
    - if new modes crowd into the experts the 3A router favours (it left 4 of 8 at 2–3% weight), try upcycling (copying
      a trained expert), a load-balancing term or router noise.
  - **Gate 3C**:
    - 1,000 2v2 games without errors;
    - the clone team beats scripted-expander teams ≥ 0.75;
    - against 3 Easy nations in Compact FFA it wins more often than the scripted expander (8% in Phase 1's diagnosis).
  - **Nations probe** (the user's go, 2026-10-15; `runs/eval/20261015-155801.json`): the weight average of lr4a, one
    seat against 3 Easy nations on Onion (Phase 1's setup: Compact, 5 minutes, no tribes), never trained on FFA, won
    **0.395 ± 0.031** of 256 games (mean rank 1.89 of 4). It acted in 6.3% of its decisions, half its rate in ranked
    1v1 (13%).
- **Moved to Phase 6**: the public FFA/Team sampler with modifiers, dihedral augmentation, procedural maps, and the S/M/L
  CNN sizes. The CNN does most of the compute, and the next size up (M) measured ~1.6× slower in Phase 0; PPO is already
  limited by the GPU. The TrueSkill table moves to Phase 4, for the league.

**Phase 4 — RL on Ranked (ongoing)**
- **4A, ranked 1v1** (after 3B): first PPO's speed ("Speed work", below). A self-play league from the 3B clone. The current policy plays a pool of past snapshots
  and the clones, with prioritized fictitious self-play: opponents that beat the learner more often are picked more
  often. The KL to the clone starts at 0.2 and is lowered only while the frozen panel keeps improving (0.05 made the
  Phase 2 bot worse).
  - **Speed work** (2026-10-11):
    - **Profile** (`scripts/profile_ppo.py`, `runs/profile/20261011-053112.json`, PC idle): the real loop with the 3B
      clone as Gate 3B ran it (16 workers × 2 games, 12 minibatches), with update 4 recorded. It ran at 101.5
      decisions/s, against 101.0 and 104.1 for the unprofiled updates around it, so the profiler barely slowed it. An
      update takes 15.1 s:
      - **Learning, 7.6 s: not launch-bound.** The GPU is busy 85% of it, and the main thread waits for the GPU 49% of
        it. GPU work 6.4 s: the backward pass 2.33, the forward pass 1.53, the KL reference's forward pass 1.50, the
        distributions 0.87 (0.20 of it the entropy computed only for the log), Adam and clipping 0.13.
      - Its 1.2 s of idle GPU sit behind ~11 syncs per minibatch, each of which empties the GPU's queue:
        - ActionDist copies constant index tensors (`MAP_TYPES_T`, `MAP_OF`) from pageable memory, and such a copy
          waits for the GPU;
        - the troop `Categorical` checks its arguments;
        - `kl.item()` runs before the backward pass, then the stats' `.item()`;
        - the first minibatch of each epoch waits for its copy (0.2 s).
      - **Rollouts, 7.55 s: the GPU is idle 69% of it, and the workers simulate only 2.9 s of it** (the trainer waits).
        The rest is the trainer's own work, and the workers wait meanwhile:
        - building the observations, 2.45 s: GPU work 1.1 s, and ~70 syncs a step, since every
          `torch.from_numpy(...).to(device)` copies from pageable memory;
        - inference and sampling for both seats, ~2.0 s: at 32 rows a network call costs 12.8 ms of CPU time and 9.5
          ms of GPU time.
      - So Gate 3B's launch-bound hypothesis was wrong for learning. Graphs or compilation could recover at most its
        16% of idle GPU, and syncs cause most of that idle time.
    - **Levers, with the gain the profile predicts**, in the order the user chose (2026-10-11), each with its own A/B:
      1. no sync within a learning minibatch: the constants kept on the GPU, no argument check, and `kl.item()` with
         the other stats. Also the next epoch's first minibatch fetched ahead, and the entropy for the log computed
         once per update: −1.0 to −1.4 s an update, **+7–10%**. Exact, ~20 lines.
         - The logged entropy then comes from one minibatch, a noisier curve.
         - Alone the entropy change is ~1%, under the A/B bar, so it goes with this lever.
         - Coded (`_on` in `trainer/model.py`, `HostRows.batches` over every epoch): one wait per minibatch is left,
           the stats' `.item()` after `opt.step()`, so the run-ahead (and the GPU memory) stays as before. The same
           ActionDist fix also removes the rollout's distribution syncs, first listed under lever 2.
         - **A/B** (2026-10-11 12:10, `runs/ab/20261011-121018-lever1.json`; old deae627, new 6b71cac): old 119.2,
           new 122.4, new 118.6, old 108.1 decisions/s. Mean gain +6.0%, but the second new probe is under the first
           old one: **no measurable gain by the rule.** Own GPU peaks 8,466–8,749 MiB.
           - The old code drifted 9% in ten minutes, more than the effect. The drift sat in the rollouts (45% → 53% of
             an update), and the first probe started from a PC idle and cool for hours (i5-13600KF, hybrid cores,
             Balanced power plan).
           - A straight drift fits all four probes: d = 3.7 decisions/s per probe from the old pair, 3.8 from the new
             pair, and an effect of ~+5.7%. The ABBA mean cancels such a drift exactly; "both new above both old"
             fails whenever a probe's drift is about half the effect.
         - **Re-run with a warm-up probe** (the user's call; 12:23, `runs/ab/20261011-122339-lever1b.json`): warm-up
           99.6 (not counted), then old 102.3, new 107.0, new 107.9, old 101.0. **Credited: +5.7%** (the prediction
           was +7–10%). The warm PC drifted 1%. Warm, the old code runs ~101 decisions/s, as at Gate 3B and in the
           profile: the cold PC's 119 was a start-up artifact.
      2. observations without syncs (copies from pinned memory that do not wait, the zone count on the CPU), and the
         same constants in the rollout's distributions: −1.0 to −1.2 s, **+7–8%**. Exact. It changes
         `trainer/env.py`, which BC and the eval share, so they gain too.
         - Coded (the constants came with lever 1):
           - every numpy → GPU copy goes through pinned memory without waiting (`pinned`);
           - `_add_units` picks its records on the CPU and copies only what it paints, and the zones' cells are
             computed on the CPU;
           - the mask update indexes with a GPU tensor, not a Python list;
           - the rollout counts actions from its CPU copy.
           - The observation build went from ~70 waits a step to none.
         - Checked bit for bit: 600 steps of 16 clone games, replayed through both versions with the same actions,
           give identical observations (every tensor's hash at every step).
         - **A/B** (12:56, `runs/ab/20261011-125640-lever2.json`; old 04156c1 = lever 1, new ce5eff7): warm-up 125.8,
           old 125.5, new 102.8, new 107.2, old 118.3. **−13.9%: slower.**
         - **Why** (profiles of the same games as the morning's, `runs/profile-v4-lever2*.log`):
           - The observation build did shrink (2.45 → 1.08 s an update), but the wait for the workers' simulation
             grew (2.91 → 4.37 s), and the rollout's GPU kernels ran 1.4–5× longer (the same kernels, as many).
           - Lever 2 brought CPU work into torch: the `pin_memory()` copies and the zones' cells. torch runs it on
             its OpenMP pool (14 threads), whose threads then spin for up to ~200 ms. The actions reach the workers
             ~70 ms later, so the spinning threads take the cores from the workers (below-normal priority) during
             their simulation.
           - The same profile with `OMP_NUM_THREADS=1`: **124–131 decisions/s** (morning, old code: 101–104). The
             workers' wait went to 2.48 s, the observations to 0.65 s, the rollout's forward passes back to 0.83 s
             of GPU time. Learning went to 6.44 s: its forward pass's CPU time halved (1.0 → 0.42 s), since the
             RAM buffer's CPU gather spins the pool during learning too, in the old code as well.
           - So one torch CPU thread in the trainer looked like a lever of its own, maybe for the old code too.
         - **One CPU thread, alone on lever 1's code** (the user's call: two A/Bs; lever 2 set aside in d192cbd,
           `torch.set_num_threads(1)` in 880ebcc; `runs/ab/20261011-133232-threads.json`): warm-up 105.8, old 108.6,
           new 108.0, new 107.9, old 109.9. **No measurable gain (−1.2%).** Lever 1's code does no CPU torch work in
           the rollouts, so nothing spins there, and learning is GPU-bound, so its faster CPU side does not show. The
           line stays as lever 2's enabler, and lever 2 is A/B'd on top of it.
         - **Lever 2 on one thread** (2164cb0 against 880ebcc, `runs/ab/20261011-134823-lever2b.json`): warm-up
           114.7, old 113.8, new 115.7, new 118.2, old 114.6. Both new above both old, but +2.4%: **no measurable gain
           by the rule.**
           - The workers' simulation plus the observations fell from ~4.45 to ~3.25 s an update, but the rollout
             only ~0.65 s: with no wait inside the observation build, the CPU waits once, before sending the actions,
             for the same GPU work. The rollout's trainer side is now limited by its GPU time per step, so lever 4
             (a graph only cuts the CPU's launches) would gain less than estimated.
           - If it stays, BC and the eval, which share `trainer/env.py`, need the one-thread line too, or its CPU
             torch work spins over their workers as it did in PPO (−14%).
         - **The user's call (2026-10-11): both reverted**, the observation rewrite and the one-thread line, as the
           heat-map packing was (under the bar, and a risk for BC and the eval). The code is lever 1's again; lever 2
           stays in git (ce5eff7, 2164cb0) with its bit-for-bit check.
      3. the KL reference's outputs computed once per update (one forward pass per row instead of three), kept in
         the RAM buffer: −1.0 s, **+7%**. Exact, ~2 GB more pinned RAM (32 GB, ~20 GB free).
         - Coded on lever 1's code: the rollout runs the reference on the learner's rows as it stores them, the RAM
           buffer keeps its outputs with the observations, and learning reads them back.
         - The check: PPO at learning rate 0 (the learner stays the reference, with random heads so each row's policy
           is its own) must log a KL of ~0. It reads 2e-7; with the stored rows shifted by one it read 4e-3 and
           1.3e-2, and the test failed.
         - **A/B** (79f6779 against 5fcb152 = lever 1's code, `runs/ab/20261011-144809-lever3.json`): warm-up 113.7,
           old 115.1, new 119.4, new 120.4, old 113.8. **Credited: +4.8%** (predicted +7%). Own GPU peak 8,115–8,285
           MiB against 8,465: the reference's activations left learning.
           - From the time split: learning ~1.5 s an update faster, the rollouts ~0.8 s slower. The reference's
             forward pass at 32 rows is bound by its CPU launches (12.8 ms against 9.5 of GPU), so it costs the
             rollout more than its GPU time.
      4. a CUDA graph of the rollout's inference (a fixed 32 rows): −0.8 s, **+5%**. Moderate risk: graphs allow no
         syncs, need static inputs, and the sampling's random draws move into the graph.
         - Re-estimated after lever 3: a graph would cut the CPU launches of the rollout's three 32-row forward
           passes (learner, reference, opponent; 12.8 ms of CPU against 9.5 of GPU each), ~0.5 s an update, ~3–4%.
      - **Speed work closed (the user's call, 2026-10-11):** levers 1 and 3 credited, +5.7% and +4.8% (~+10.8%
        together); lever 2 and the one-thread line reverted; lever 4 parked (near the bar, moderate risk).
        - Also for the league's design: when the opponent is the KL reference itself, one forward pass over both
          seats' rows would give the opponent's actions and the reference's outputs, saving a 32-row pass a step.
      - Together: ~15.1 → ~11 s an update, ~140 decisions/s (+35–40%).
      - Parked:
        - `torch.compile` through `triton-windows`: learning is GPU-bound, so only its fusion could help, and it is an
          unofficial dependency;
        - overlapping the workers' simulation with the trainer's work (two worker groups): up to −2.9 s, but a larger
          change, worth it only after levers 2 and 4;
        - 2 epochs instead of 3 ("Speed", below): −2.5 s, but it changes what PPO learns per sample, so it is judged
          on learning curves, not by a speed probe.
    - **A/B rule** (the user's call, 2026-10-11; fixed before any run):
      - the probe: `scripts/speed_check.py --ppo-minibatches 12 --skip-bc --v4 checkpoints/bc3b-v4/latest.pt` without
        its Phase 2 run (a new `--skip-v1`): 10 updates with the 3B clone, the mean decisions/s of updates 3–10, with
        the PC idle;
      - old, new, new, old, back to back in one `.cmd` started via `explorer.exe`. The old code runs from a worktree
        of the commit before the change (`pins/ab-base`);
      - from lever 1's re-run on (the user's call, 2026-10-11): a warm-up probe of the old code first, not counted,
        so no counted probe starts from a cold PC;
      - a lever is credited if both new probes beat both old probes and the mean gain is ≥ 3%; its credit is the mean
        gain. Otherwise it reads "no measurable gain";
      - each probe's own GPU peak stays ≤ 9,400 MiB (the 3B rule).
  - **League design** (the user's calls, 2026-10-11, after the speed work and the bandwidth probe):
    - **Stage 0 first:** plain PPO from the 3B clone against its frozen self, with Phase 2's settings (exact KL 0.2 to
      the clone, lr 1e-4, no entropy bonus, 16 × 2 games, 12 minibatches), 3,000 updates (~10–11 h at ~120
      decisions/s).
      - Why: no v4 network (15 actions, 12 heat maps, 100 shares) has been trained by PPO yet, so this checks the
        signal before the league machinery is built.
      - Checked like Gate B: its snapshots against the frozen clone on the 512 fixed games, bar 0.545. Its snapshots
        seed the league's pool.
      - **Results (2026-10-13): passed.** `ppo4a-s0` ran 3,000 updates in one sitting (2026-10-11 23:32 to 2026-10-12
        10:49, 114 decisions/s).
        - Training was stable: the KL to the clone held at 0.06–0.07 (Phase 2: 0.05–0.06), the KL per update fell from
          0.02 to 0.01 with ~6% of samples clipped, and the value head predicted the winner (correlation ~0.35 early
          in a game, ~0.9 late). The bot acted in 13–14% of decisions and attacked in 11–12%.
        - **The check** (rule fixed before: update 3,000 ≥ 0.545): update 1,000 0.736 ± 0.019 (seats 0.773 / 0.699),
          update 3,000 0.695 ± 0.020 (0.699 / 0.691). Phase 2's PPO scored 0.730 at update 1,000 and 0.773 at 5,000.
          Nothing was gained after update 1,000 (−0.041, 1.5 SE unpaired).
        - **The training curve read below the frozen checks**, the reverse of Phase 2 (0.05–0.25 above): 0.57 at
          update 1,000 and 0.58 at 3,000. It wandered from 0.35 (near update 600) to 0.71 (near 2,400), ~0.56 on
          average; luck alone moves it ±0.07.
          - Ruled out by reading the code: layers that act differently in training (there are none), forward passes
            that change the observations in place (lever 3's reference pass included), and differences in setup,
            seat rotation or sampling between the loop and `trainer.eval`.
          - The 12 snapshots' windows (disjoint, 2,400 games) by game length: the learner won 0.74 of the games that
            ended before 3,000 ticks (377 games), 0.61 at 3,000–6,000 (535), **0.36 at 6,000–9,000 (647)** and 0.66
            after (841).
          - Two explanations remained: the loop makes the learner play worse than its snapshots (a bug the league
            would inherit), or learning during a game hurts it (a game spans ~30 updates).
          - **Frozen probe** (the user's call; run `probe-s0-1000`, 2026-10-13 00:51–01:38): the loop with lr 0 from
            update 1,000 for 250 updates. Rule fixed before, on its last 200 games: ≥ 0.66 means the loop is sound,
            ≤ 0.62 a handicap to find before the league. It read **0.735 ± 0.031** (the eval: 0.736): **the loop is
            sound**, and learning during the games cost them 0.12–0.17.
            - By game length: 0.88, 0.85, **0.52** and 0.79 (26, 39, 58 and 77 games). The 6,000–9,000-tick games are
              this policy's weak spot even frozen; learning lowered every length.
          - In Phase 2 the training games read 0.05–0.16 above the frozen checks instead, at the same step sizes (KL per
            update ~0.010, ~7% of samples clipped, KL to its clone 0.05–0.06). Why the sign flipped is open. It matters
            only if it slows the league, whose frozen evaluations measure strength anyway.
    - **Opponents one at a time, rotated:** every 25 updates PFSP picks the opponent for new games; games in progress
      finish with theirs, so at most 2 are active. Each active opponent costs one more forward pass per rollout step
      (~0.6 s an update), and nobody takes over a game halfway.
      - The pool: the 3B clone, the 3A clone, the Phase 2 PPO bot, the Phase 2 clone, and the learner's snapshots
        every 250 updates.
      - PFSP: an opponent's weight grows with how often it beats the learner in the training games against it.
      - The KL reference stays the 3B clone.
    - **Frozen evaluations inside the trainer:** every 250 updates training pauses, a second set of workers plays ~128
      fixed games against the panel (the 3B clone, the Phase 2 clone, the Phase 2 PPO bot), then training resumes:
      ~15% of the time, no contention, results on the control page.
    - **Ratings: a Bradley-Terry fit** over all frozen-evaluation results, anchored on the panel, with bootstrap
      uncertainty. It replaces the planned TrueSkill table: a few lines of numpy, no new dependency, and every game
      is a snapshot against a fixed panel member, which one fit handles directly. Gate 4A reads it.
    - **Coded (2026-10-13)**: `trainer/league.py`, and `trainer.ppo` takes `--league`, `--panel`, `--eval-games` and
      `--kl-ref` (the command is in CLAUDE.md).
      - Opponents: each game slot keeps its opponent until its game ends, and every 25 updates PFSP picks the opponent
        of the games that start next.
        - The weight is (1 − x)² (AlphaStar's f_hard), where x is the learner's win rate in its last 50 training games
          against that member, with one win and one loss added, so a new member starts at 1/2.
        - The pool is the checkpoints given (globs expand: Stage 0's 12 snapshots and the 4 fixed ones), plus the
          learner's snapshots every 250 updates.
        - A network that no game uses leaves the GPU.
        - Correction: games last up to ~38 updates (the timer ends them near 9,200 ticks, at 48 decisions an update).
          So after a rotation, a long game can keep a third opponent active for a few updates.
      - `--kl-ref` sets the KL reference (the 3B clone) apart from `--init` (a Stage 0 snapshot).
      - Frozen evaluations:
        - Which: `--init`'s at update 0, then every snapshot's. A resume redoes one a stop or crash cut short.
        - The games: `--eval-games` (default 128) on Gate B's seeds and layout ("eval", 16 × 2). Game slot e plays
          panel member e % 3 (44, 44 and 40 games), its first 4 episodes.
        - Training waits meanwhile, and pause and stop still work.
        - Results are appended to `runs/<run>/panel.jsonl`. The page shows the pool, the evaluations and two charts.
      - Ratings: Newton's method with a Gaussian prior (sd 10 logits), so a perfect score stays finite. 200 parametric
        bootstrap refits give the uncertainty. The trend is the slope of the last 5 snapshots' ratings, with the share
        of refits in which it rises.
      - **The time, re-estimated:** the 512-game check of update 3,000 took 60.4 min, 0.149 s a step with 32 games at
        once.
        - Each slot plays its games back to back, and the evaluation waits for the slowest. So 128 games take ~18 min:
          ~24% of the time, against Stage 0's ~56 min per 250 updates, not ~15%.
        - 64 games take ~9 min, ~13%.
        - `--eval-games` can change on a resume.
      - **The trend's resolution** (simulated with the fit itself, 60 draws for each case):
        - At 128 games, the slope over 5 snapshots has an sd of ~12 Elo per 250 updates. At 64 games it is ~18.
        - A true rise of 25 per 250 updates reads as rising in ≥ 90% of the refits in 72% of draws (47% at 64
          games). A rise of 2.5 per 250 updates, Phase 2's pace after its update 1,000 (0.730 → 0.773 against its
          clone, 40 Elo over 4,000 updates), reads that way in 10% of draws, about the false-alarm rate.
        - So Gate 4A's "still rises over the last 5 snapshots" detects only a fast league. Its rule must be fixed
          before it is read, e.g. over a longer window.
      - Tests:
        - `trainer/test_league.py` covers PFSP and the ratings;
        - a league run in `test_ppo.py`: a snapshot every 2 updates, a pick every update, and a resume;
        - `test_eval.py` covers `play()`'s per-slot opponents and its early stop.
    - **First league run** (`league4a`, 2026-10-13 03:44–21:57, 4,028 updates, stopped by the user's call):
      - The setup: from Stage 0's update 1,000, KL 0.2 to the 3B clone, 64 games per frozen evaluation. The pool had 16
        members at the start and 32 at the end, and 4,312 training games were played.
      - **No progress.** The snapshots of updates 250–4,000, pooled, beat the 3B clone 0.599 ± 0.026 (352 games) and
        the Phase 2 PPO bot 0.588 ± 0.028. The start scored 0.736 ± 0.019 against the 3B clone (512 games).
      - **The snapshots' strength swings.**
        - Their win rates vary more than the evaluation's noise allows (χ² p = 0.009, < 0.001 and 0.04 against the three
          panel members).
        - They rise and fall together across the members (r ≈ 0.7), whose games differ.
        - Against the 3B clone they ran from 0.32 (update 2,000) to 0.82 (update 3,250).
        - The 5-snapshot trend at the end: −36 Elo per 250 updates.
      - In training games, the learner beat its own start 0.34 of the time, flat over the run (0.26–0.46 for each ~50
        games). That is about Stage 0's cost of learning during a game, against an equal opponent.
        - PFSP spent 514 games on Stage 0's update 250, which still won 79% of them.
      - **The swings start in Stage 0** (read from the league's PFSP results after the stop).
        - The learner's win rate in training games against Stage 0's snapshots jumps between neighbours 250 updates
          apart. By update: 250: 0.21 (514 games), 500: 0.88 (66), 750: 0.44 (364), 1,000: 0.34 (300), 1,250: 0.78
          (74), 1,500: 0.64, 1,750: 0.67, 2,000: 0.75, 2,250: 0.70, 2,500: 0.34 (235), 2,750: 0.73, 3,000: 0.40 (317).
        - The league's own snapshots do the same: update 750 0.97 (34 games), 1,000 0.37 (220).
        - The learner itself is about even with the 3B clone (0.53 over 368 games). So some snapshots are far weaker
          than the clone and others far stronger.
        - Meanwhile Stage 0's KL to the clone sat at 0.061–0.070 from update 250 on, and its KL per update fell from
          0.017 to 0.010.
        - So PPO at this step size never settles: the problem is instability from the start, not a plateau after
          update 1,000. Stage 0's two frozen checks (updates 1,000 and 3,000) landed on strong snapshots.
        - Caveat: each member's games are spread over the run, against a learner that kept changing, and PFSP gives
          the strong members more games.
      - **Where strong and weak snapshots differ.** In a diagnostic, the clone played both seats of 32 games for 2,400
        steps, and 10 snapshots, strong and weak by the PFSP results, were scored on the same observations.
        - Not the spawn. In the spawn phase every snapshot's KL to the clone is 0.008–0.021, strong and weak alike.
        - The KL to the clone does not separate them in any phase (0.07–0.18 a decision). For all of them it sits
          mostly in the action type and the attack's target and troop share.
        - Their action mixes differ 2–5x on the rare actions (cities, boats, retreats). No single action sorts strong
          from weak once ~40 comparisons are counted.
        - The best lead is early-game cities: 0.27–0.49% of the weak snapshots' decisions, against 0.52–1.12% of the
          strong ones' (the clone: 0.50%).
        - Reading: PPO's noise moves the probabilities of rare, decisive actions by large factors, and a KL averaged
          over all decisions barely sees it.
      - Training statistics stayed level: the KL to the clone 0.061–0.068, the KL per update ~0.01, ~6% of samples
        clipped, the entropy 0.27 → 0.25.
      - **Speed:** ~13 s an update up to update 3,500 (~19:15), Stage 0's pace. Simulation was slower only as games grew
        longer. Then every part slowed at once, to 16.2 s at the end (learning 5.1 → 5.85 s): RAM had run out.
      - **Memory:** at the end the trainer held 46 GB of private memory, and the PC's RAM was full (0.4 GB free; the
        stop freed 64 GB of commit). Measured afterwards:
        - Windows counts a process's GPU memory in its private memory: 2 GiB on the card added 1.98 GB. That is ~8 GB
          of the trainer's.
        - PyTorch's caching host allocator rounds pinned buffers up to a power of two, and pinned pages cannot be paged
          out. The rollout buffer holds 7.6 GiB of data in 12.9 GiB of pinned RAM: the observations' 4.0 GiB take
          8 GiB.
        - The other ~21 GiB was a leak, found with probes of one operation each. Every learning pass started a new
          thread to gather the minibatches, and that thread's first parallel CPU op started an OpenMP team of 13
          threads that outlived it. That is 13 threads and ~5.4 MiB an update: ~21 GiB and ~52,000 threads over 4,028
          updates. Forward passes at new batch sizes (cuDNN benchmark), loading 28 different checkpoints, and starting
          and closing 16 workers were flat.
        - Each frozen evaluation adds 16 game workers, ~0.6 GB each.
      - **Fixed (2026-10-13):**
        - The gathering thread lives as long as the run. A test checks that the thread count stays flat over learning
          passes. The probe's passes 50–200 at the real row size: +0.00 GiB, against +0.80 before.
        - Pinned buffers above 1 MB keep their exact size (`pinned_max_round_threshold_mb:1` in the allocator config,
          PyTorch 2.14), so the rollout buffer locks 7.6 GiB instead of 12.9.
        - PPO logs its private memory every update (`perf/private_gb`, the page's "Trainer memory" chart).
    - **Learning test** (the user's go, 2026-10-14; `runs/learn4a.cmd`, ~7 h). The rules were fixed before any result:
      1. **Weight average.** The uniform average of Stage 0's 9 snapshots of updates 1,000–3,000
         (`checkpoints/swa-s0/update1000-3000.pt`) plays 256 games against the 3B clone. Its KL to the clone is 0.029,
         against 0.037 and 0.059 for updates 1,000 and 3,000. Compared with update 1,000's 0.736:
         - ≥ 0.78: the league plays and evaluates averaged weights, and the average is the strongest bot so far;
         - ≤ 0.70: dropped;
         - in between: decided with test 3.
      2. **Are the swings real?** Stage 0's updates 250, 500, 750, 1,250, 1,500, 1,750 and 2,000 play 128 games each
         against the 3B clone. The swings are confirmed if the snapshots differ beyond noise (χ² p < 0.01; noise alone
         moves a 128-game result by ±0.04).
      3. **A smaller step.** Run `lr4a` uses Stage 0's setup from its update 1,000 (KL 0.2 to the 3B clone, which is
         also the opponent) at lr 2.5e-5 instead of 1e-4. It runs 1,000 updates, with a frozen evaluation against the
         clone every 250 (128 games). Its 4 snapshots are compared with Stage 0's at the same distance (updates
         1,250–2,000):
         - adopted for the next league if their mean is ≥ 0.04 higher, or if test 2 confirmed the swings and its
           snapshots are steady (χ² p ≥ 0.05) with a mean at most 0.02 lower;
         - otherwise the step size is not the cause. Next come more games per update (RAM permitting) or a KL that
           weighs rare actions more.
      - **Results (2026-10-14).** Windows Update restarted the PC at 07:20 (a scheduled restart by MoUsoCoreWorker), so
        run `lr4a` stopped at update 800, after its evaluation of update 750.
        1. The weight average scored **0.742 ± 0.027** (256 games), between the bars, so it was decided with test 3
           (below).
        2. **The swings are real** (χ² 84.9 on 6 df, p < 0.001). Stage 0's snapshots against the 3B clone, 128 games
           each, by update: 250: 0.766, 500: **0.195**, 750: 0.531, 1,250: 0.500, 1,500: 0.469, 1,750: 0.516, 2,000:
           0.477.
           - So a typical Stage 0 snapshot was about even with the clone. The checks of updates 1,000 (0.736) and
             3,000 (0.695) landed on lucky ones, and Stage 0 passed its check on them.
           - Correction: the 0.12–0.17 "cost of learning during the games" was measured against the lucky update
             1,000. Stage 0's training curve (0.54–0.59) sat about where its typical snapshot did.
        3. **A smaller step works.** Run `lr4a` started from Stage 0's update 1,000, which read 0.680 on these 128
           games. Its snapshots scored **0.773** (update 250), **0.750** (500) and **0.883** (750).
           - Their mean, 0.802, is 0.31 above Stage 0's 0.490 at the same distance (updates 1,250–2,000).
           - Update 1,000 cannot change the verdict: the rule needs a mean ≥ 0.530, and even a 0 there leaves 0.60.
           - **Adopted: lr 2.5e-5 for the next league.**
           - The KL per update fell from ~0.010 to ~0.002, the clipped share from ~6% to ~2.5%, and the KL to the clone
             from ~0.065 to ~0.05.
           - The training curve rose with the frozen evaluations: 0.60 (update 50), 0.80 (250), 0.85 (500), 0.89
             (700–750).
        - Test 1 against test 3: the weight average beat Stage 0's typical snapshot by ~0.25. That is the same reading
          as test 3: at lr 1e-4 the snapshots were noise around a decent policy. The smaller step alone does better
          (0.80 against 0.74), so averaging is not adopted now.
        - The memory fix held: the trainer's private memory stayed at 19.2–19.5 GiB over 808 updates and three
          evaluations (the first league run reached 46 GB by update 4,028). Speed: 113–137 decisions/s.
        - **Extended to update 2,000** (resumed at the user's request at 10:51, finished 15:40, one attempt):
          - Against the 3B clone (128 games each), by update: 1,000: **0.672**, 1,250: **0.812**, 1,500: **0.812**,
            1,750: **0.750**, 2,000: **0.844**.
          - Updates 250–1,000 averaged 0.770 (the rule's four snapshots), and updates 1,250–2,000 averaged 0.805.
          - All eight snapshots pooled: **0.787 ± 0.013** (1,024 games), against Stage 0's ~0.49 at the same distance.
          - The swings shrank but did not vanish: 0.67–0.88, still beyond noise (χ² 22.8 on 7 df, p = 0.002). Stage 0
            ranged 0.20–0.77.
          - The statistics held level: KL per update ~0.002, ~2.5% of samples clipped, KL to the clone ~0.052, entropy
            ~0.27. The training curve read 0.81–0.89.
          - Memory stayed flat over all 2,000 updates and nine evaluations (19.0–20.3 GiB). Speed ~137 decisions/s.
      - **Checks before the next league** (the user's go, 2026-10-14; `runs/check4a.cmd`, ~1 h). The rule was fixed
        before any result. The weight average of lr4a's updates 1,000–2,000 (5 snapshots,
        `checkpoints/swa-lr4a/update1000-2000.pt`) plays 256 games against the 3B clone and 256 against the Phase 2 PPO
        bot.
        - ≥ 0.83 against the clone: the next league starts from the average, and the trainer gets weight averaging.
          Otherwise the league starts from lr4a's update 2,000.
        - The game against the Phase 2 PPO bot reads Gate 4A's second bar (0.70).
        - **Results (2026-10-14, 20:14–21:03):** the average beat the 3B clone **0.945 ± 0.014** (seats 0.961 / 0.930)
          and the Phase 2 PPO bot **0.969 ± 0.011** (0.969 / 0.969), 256 games each.
          - Its five snapshots scored 0.672–0.844 against the clone (mean 0.778, 128 games each), so averaging added
            ~0.17, and 0.10 over the best of them. Stage 0's average added ~0.25 over its typical snapshot. At both step
            sizes the snapshots scatter around a stronger mean policy.
          - By the rule: the next league starts from the average, and the trainer gets weight averaging.
          - Gate 4A: the average meets both match bars (0.80 against the 3B clone, 0.70 against the Phase 2 PPO bot).
            The third, a league rating that still rises, waits for the league; its rule is fixed before the league
            runs.
        - **Weight averaging in the trainer** (the user's go, 2026-10-15): every 250 updates PPO also saves
          `avg<update>.pt`, the uniform average of the last 5 kept checkpoints (1,000 updates, `--init` counting as
          update 0), the recipe that scored 0.945. The averages join the league's pool and play the frozen evaluations;
          the learner keeps training its own weights.
      - **League v2** (`league4a2`, the user's go, 2026-10-15; `runs/league4a2.cmd`, ~16 h). Fixed before it runs:
        - From the lr4a average at lr 2.5e-5, KL 0.2 to the 3B clone, 16 × 2 games, 12 minibatches, 4,000 updates,
          weight averaging on.
        - Pool: the lr4a average, the 3B clone, the 3A clone, the Phase 2 PPO bot and the Phase 2 clone; the run's
          averages join every 250 updates. Stage 0's and the first league's snapshots are left out: their strength
          swung.
        - Panel: the lr4a average (the rating's anchor), the 3B clone and the Phase 2 PPO bot, 128 games every 250
          updates. The old panel cannot measure progress any more: the average beats each of its members ~95% of the
          time.
        - **Gate 4A's third bar:** the final average (updates 3,000–4,000) plays 512 fixed games against the lr4a
          average, and ≥ 0.55 passes (2.3 SE above even). It replaces "a league rating that still rises over the last
          5 snapshots": that trend is too noisy to read (above), and consecutive averages share 4 of their 5
          snapshots.
        - **Results (2026-10-15 16:13 to 2026-10-16 11:29, one sitting, 4,697 training games):**
          - The averages beat the lr4a average **0.656 ± 0.018** pooled over the 16 frozen evaluations (704 games,
            ~+110 Elo), the 3B clone 0.962 ± 0.007 and the Phase 2 PPO bot 0.994 ± 0.003. `--eval-games` counts the
            whole panel: each member got ~44 of the 128 games, so single points swing ±0.07 (0.57–0.75 after update
            1,000).
          - Against the lr4a average by quarter: 0.625 (updates 250–1,000), 0.682 (1,250–2,000), 0.665 (2,250–3,000),
            0.653 (3,250–4,000), each ± 0.036. The gain came by update ~1,250; the last 2,750 updates (~13 h) added
            nothing measurable against the start.
          - The raw learner never beat the start in its training games: 0.478 over 728 games (0.47, 0.45, 0.57, 0.43 in
            chronological quarters), while its averages beat the start 0.66. It won 0.391 against the run's averages
            (79% of its games) and 0.948 against the four fixed bots (5%). The league's gain shows only through the
            averaging, as in lr4a (snapshots 0.778 against the clone, their average 0.945).
          - The training curve read 0.17–0.59 after update 500 (median 0.31 over the last 1,000 updates). PFSP gave the
            start and the averages ~99% of the weight, and the curve (the last 200 games, ~170 updates) follows the
            opponent picked every 25 updates.
          - The lr4a average was in the pool (728 training games against it), so 0.66 is against a known opponent. The
            held-out tests are 4N's nations and, later, humans.
          - Gate 4A: the final average meets both match bars (0.977 against the 3B clone, 0.975 against the Phase 2 PPO
            bot; 44 and 40 games).
          - **Third bar (2026-10-16, 15:30–16:31, `runs/gate4a.cmd`):** the final average beat the lr4a average **0.592 ±
            0.022** over 512 fixed games (seats 0.586 / 0.598), 1.9 SE above the 0.55 bar: **Gate 4A passes**, and 4N
            starts from `checkpoints/league4a2/avg004000.pt`.
          - The edge is ~+65 Elo, below the panel's pooled 0.656 (2.3 SE apart). The panel played 44 games per point on
            another fixed set of games, so the 512 games are the number to trust.
    - Later, each measured on its own: more games per update with fewer steps each (Phase 2's open question), a lower
      KL (only while the frozen panel improves), and lever 4 if more opponents at once are wanted.
- **4B, ranked 2v2** (after 3C; both come after 4N since 2026-10-15, below): teammates are copies of the current
  policy.
- **Gate** (re-set on 2026-10-03: Phase 2's PPO already beat its own clone 0.773, above the old ≥70% bar). A fixed panel
  of frozen opponents on fixed seeds: the Phase 3 clone, the Phase 2 clone, and the Phase 2 PPO bot
  (`checkpoints/ppo-bc2/update005000.pt`). Gate 4A:
  - ≥ 0.80 against the Phase 3 clone;
  - ≥ 0.70 against the Phase 2 PPO bot;
  - the league's final weight average beats its start ≥ 0.55 over 512 fixed games (since 2026-10-15, "League v2"
    above; it was a league rating that still rises over the last 5 snapshots).
- **Frozen evaluations during training**: every 250 updates, 64–128 frozen games against the panel (~13% of the time at
  64 and ~24% at 128, "League design" above).
  The training curve is off either way: in Phase 2 it included the learner adapting to the games in progress and
  missed the KL-0.05 decline; in 4A Stage 0 it read 0.12–0.17 below the frozen checks.
- **Speed**: Phase 2's PPO ran at ~190 decisions/s with the GPU at ~97%. Most of it was learning (3 epochs of forward
  and backward passes) and the KL reference's forward pass in every epoch. Levers to measure: 2 epochs, and the
  reference computed once per rollout. Rollouts are stored as 8 bits (3A).
- **Stall protocol**: reverse curriculum, then shaping.
- **Bandwidth meters** (from 3B's bandwidth test):
  - the share of decisions that act, per game phase;
  - the share of actions that follow an action;
  - the share of decisions with ≥ 25% on both a troop and a gold action.
  - If 3B keeps decision 11, bins come in if any of these holds:
    - the upward probe on the 3B clone gains more than 0.05 (before 4A; it did not: −0.018, 2026-10-11);
    - the bot acts in more than half of a phase's decisions;
    - the bot wants both kinds of action in more than 10% of its decisions.
- **Intercepted-nuke rate**: if it stays high, salvo amounts come back ("Action space").
- **Dormant units** (Sokar et al., ICML 2023):
  - the share of each layer's units whose mean |activation| is at most a small fraction of the layer's mean, logged in
    PPO, where the league shifts the data;
  - the 3B clone's checkpoint gives the baseline with one forward pass;
  - if the share climbs: leaky ReLU in the CNN, or resetting the dormant units. Abbas et al. 2023 and Dohare et al.
    (Nature, 2024) report the same loss of plasticity in long training on shifting data.
- **Also from Phase 2**:
  - the exact KL, never a one-sample estimate;
  - more games per update and smaller steps;
  - a restart loop for long runs (the PC's GPU driver occasionally fails).
- **4N — FFA and Team against the nations** (after 4A; the user's call, 2026-10-15, designed with the user section by
  section):
  - **Why now:** public lobbies (FFA and Team, 50/50) are where the bot will play, the only place the owner allows.
    Ranked 1v1 stays the training ground and the benchmark. The 1v1 bot already plays FFA: the lr4a average won 0.395
    against 3 Easy nations on Onion without FFA training (3C's nations probe).
  - **What moves:** ranked 2v2 (3C's setup, crawl and BC, then 4B) and the MoE routing test come after 4N, with the
    human data (below). Their reasons stand: team mechanics, and the router's first mixed modes. 4N reaches the
    deployment modes first and needs no new data or actions.
  - **Facts** (`MapPlaylist.ts`, v0.34.24):
    - Lobbies rotate FFA, Team and special (FFA or Team with 1–3 modifiers): 50/50 FFA and Team, a third with
      modifiers.
    - 111 FFA maps and 112 Team maps. All games of a run share one padded grid, and the CNN's cost grows with its
      cells. 192 × 256 (today's) fits 34 maps, 32% of the rotation, with World (the most played map) and the ranked
      maps; 256 × 256 fits 55 (48%), 256 × 320 74 (65%), 256 × 384 84 (75%).
    - Lobbies hold up to 125 players (~70 on average at the largest size tier; every 7th lobby at most 25).
    - FFA: all of a map's nations (~30 on average, a quarter on Compact maps), Medium; 400 tribes (100 on Compact).
      No timer: 80% of the land wins, and after 30 minutes the bar drops 2 points a minute.
    - Team: donations on. No nations except in Humans vs Nations (20% of Team games: as many Hard nations as humans).
      2–7 teams in 60% of Team games, Duos, Trios or Quads in 20%.
  - **The games** (one run, each worker playing one kind):
    1. ranked 1v1 as in the league (the learner against the pool), so 1v1 is kept;
    2. FFA against nations: 4 learner seats (shared weights, all learning), all of the map's nations (Medium or Hard,
       half each) and the tribes. The 34 maps that fit today's grid, weighted like the public rotation, minus 4
       held-out maps, drawn once with a fixed seed among the 34 without World and the 4 ranked maps; Compact in a
       third of the games, with a quarter of the nations, as the game does;
    3. Humans vs Nations: 4 learner seats as one team against 4 Hard nations.
    - Training games have a 30-minute timer (the tile leader wins), about half the longest public game. The gate
      plays the public rules.
    - Rewards as decision 9, among the seats and the nations (tribes do not count). A team's result is shared.
    - Learner seats stay at today's 32, since each learner decision keeps a ~2.8 MB observation in pinned RAM:
      8 ranked games (8 seats), 4 FFA games (16), 2 Humans vs Nations games (8).
    - A 30-minute game is ~2.4× a typical ranked game, so results, the only reward, come ~2.4× less often per
      decision.
  - **The code:**
    - `bridge/setups.ts`: public FFA and Humans vs Nations configs copied from `MapPlaylist` (as `ranked1v1Config`
      is), over the 34 maps with the public weights minus the 4 held out, with the 30-minute timer and Medium or
      Hard nations. A test checks the drawn configs against the game's rules (nation counts, Compact every third).
    - Mixed runs: each worker gets its own setup, and games may have different seat counts (`trainer/env.py`,
      `bridge/worker.ts`).
    - PPO learns from every seat of FFA and Humans vs Nations games; ranked games keep one learner seat and the
      league's pick in the other.
    - Frozen evaluations per setup, with each setup's win rate on the control page.
    - `trainer.eval --setup ffa | hvn`, on the training or the held-out maps, under the public rules.
    - Unchanged: the network (it already sees which players are nations and which are teammates) and the KL
      reference (the 3B clone). The start is league v2's final average if it passes Gate 4A, otherwise the lr4a
      average.
    - A known limit: World has 72 nations, so with 4 seats a game holds 76 real players. The transformer attends to
      the 64 most relevant (its relevance rule), and every player keeps a token.
    - Tests: a mixed run with one worker per setup in the smoke test, and a game of each new setup played to its end.
  - **Gate 4N** (the final weight average, under the public rules: no timer, the bar drops after 30 minutes):
    - FFA: one bot seat against all of the map's Hard nations and the tribes, 128 games on the 4 held-out maps: it
      wins ≥ 0.50. The same 128 games on 4 training maps show the gap between seen and unseen maps;
    - Humans vs Nations: 4 bot seats against 4 Hard nations, 128 games on the held-out maps: the team wins ≥ 0.70;
    - 1v1: against its start in ranked 1v1, 512 fixed games: ≥ 0.45.
    - One bot seat in the FFA gate, not training's four, so a win is the bot's own and not shared among copies.
    - Baselines first: the start plays the FFA and Humans vs Nations games before training (~3–4 h of GPU). A bar it
      already meets is done without training.
    - Gate games are Normal size, Compact in a third, as in training.
  - **During training:** frozen evaluations every 500 updates (FFA games are long): ranked 1v1 against the panel, 64
    games; FFA against Hard nations, 32; Humans vs Nations, 32; on training maps with the 30-minute timer, ~27 minutes
    per ~2 hours of training. 4,000 updates (~16–20 h), then the gate; if it fails, the per-setup curves show where
    it is stuck before anything is decided. Also logged: each setup's training win rate, action mix and game length,
    how often the 64-player cap binds, and decisions/s.
  - **The code** (2026-10-16, `docs/superpowers/plans/2026-10-16-4n-ffa-and-team.md`, commits 029a955..761fd5e).
    - **How it was written** (the user's call): Sonnet subagents implemented each task, a Sonnet reviewer checked each
      one, and an Opus review covered the whole branch. Its three fixes are in 761fd5e. Bridge 58 tests, trainer 155.
    - **Decisions taken in the plan:**
      - **Held-out maps:** Didier, Bering Strait, Yellow Sea and Labyrinth. Seed "4N held out", over the 30 candidates
        (the 34 fitting maps without World, Australia, Iceland and Asia; EuropeClassic is not in the rotation).
        - Bering Strait has only 2 nations, which makes it an easy FFA gate map. The draw stands.
      - **Seen maps (the gap):** Black Sea, Venice, Mars and More Than Luck, drawn the same way with the seed "4N
        seen".
      - **Gate map sets** are played equally often.
      - **No-timer horizon:** a game without a timer is observed with a 30-minute horizon, as in training.
      - **The backstop** ends a game 1 minute after its timer, or at 61 minutes without one. A game it ends with no
        winner is *cut* and ranked by tiles.
      - **Dead seats** keep their rows. `setups/dead_share/<setup>` logs the cost.
      - **Humans vs Nations** is Compact in a third of games on every map. The rotation drops Compact on 3 small maps,
        ~2.6% of its games.
      - **The training grid everywhere:** every 4N game (training, evaluations, gates) plays on the ranked grid,
        192 × 256. The network averages its features over the whole grid, padding included, so a held-out game on a
        smaller grid would read differently (the final review measured it).
    - **Rulings during execution:**
      - **Gate 4N's bars read `decided_win_rate`:** games won and not cut, over all games. A cut game would play on
        under the public rules, so a lead at minute 61 is no win.
        - `cut_share` is reported beside it.
        - If the baselines show more than 10% of games cut, the user hears it before any bar is read.
      - **Frozen evaluations: 56 ranked games and 28 each for FFA and Humans vs Nations**, instead of 64 and 32.
        - Both are multiples of the run's 14 game slots: `play()` rounds up to a multiple of the slots.
        - Layout: `--mix ranked1v1=8 ffa=4 hvn=2 --games-per-worker 1`.
  - **The speed check** (2026-10-17, 02:36–03:05, the user's go; `runs/speed4n`): the run's full command for 30
    updates, exit 0.
    - **Speed:** 132 decisions/s over updates 10–30 (96–142), 11.6 s per update. League v2: 115–137, ~16.5 s per update
      with its evaluations.
    - **Memory:** the trainer 20.2 GiB private (league v2: 19.4–19.9), the GPU 8.5 GB reserved and 6.6 GB in use. The
      Node workers hold at most 8.2 GB in the evaluation round (28 alive) and 4.3 GB in training (14).
    - **The evaluation round:** 23 minutes. 4,000 updates: ~13 h of training plus 9 rounds (~3.5 h), ~16.5 h in all, the
      evaluations ~19% of it.
      - Both will grow as the bot survives longer: the env ends a game when all its seats are dead, and the FFA games
        ended early.
    - **Dead seats:** 39% of the FFA rows and 22% of the Humans vs Nations rows over updates 10–30, ~25% of all learner
      rows.
      - The 64-player cap never bound (only 4 FFA games), and no game was cut.
    - **The start against the nations** (the round at update 0: training maps, the 30-minute timer):
      - FFA, one seat against Hard nations: 1 win in 28, mean rank 19.2.
      - Humans vs Nations: 0 wins in 28.
      - In training games, counting only seats still alive, the bot attacked in ~7% of its decisions in FFA and Humans
        vs Nations, against 16% in ranked 1v1, and did nothing in ~91% (81% in 1v1). The action shares count dead seats'
        rows as no-ops.
        - The user's reading: FFA and Team games sit closer to an equilibrium, where timing (and the ocean, with
          warships) matters more than speed, so humans may attack less there too. Public FFA records can test it.
      - Humans vs Nations has two teams, so its reward is ±1 (decision 9). Without wins its only signal is the
        discount's push to lose later (γ 0.999 per decision: a loss 10 minutes away weighs −0.30). FFA's reward is
        graded by rank, so its games carry signal without wins.
  - **The checks before the run** (2026-10-17, 03:52–04:04, the user's choice; `runs/checks4n.cmd`):
    - **Humans vs Nations against Medium nations** (training maps, the 30-minute timer, 28 games): the start won **24**
      (0.857 ± 0.067), against 0 of 28 against Hard. Medium and Hard nations are far apart.
    - **The rule, fixed before the check:** 3 or more wins, and Humans vs Nations trains against Medium or Hard nations,
      half each, as FFA does. Its frozen evaluations and the gate stay against Hard (`trainer/env.py`). Fewer would have
      taken it out of the mix.
    - **Games to watch** (`trainer.watch --dir`): FFA and Humans vs Nations against Hard nations on Venice
      ("league4a2 4000": 3Ky1Z3nU and atVyzIap), and ranked 1v1, league v2's final average against the lr4a average
      ("league4a2 4000 vs lr4a swa 1000-2000", 4 games).
      - The Humans vs Nations game overwrote the second FFA game: the gates' games of one policy and seed share their
        IDs. Recorded games now go to a folder per setup, maps, difficulty and timer.
    - **The baselines (Task 7) wait.** The round at update 0 already shows no bar met. The start can play Gate 4N's
      games after the run, on the same fixed games, as the before-and-after reference.
    - `trainer.eval --difficulty` and `--timer` change a match's nations and timer.
    - **Open (the user's idea, 2026-10-17):** team games of StockShip copies only, as in public Team games (no
      nations), beside or instead of Humans vs Nations in training.
      - A game between copies always has a winning team, so even two teams give a signal, and more teams rank them.
      - Public Team games are ~60% "2–7 teams" and ~20% Duos, Trios or Quads; Humans vs Nations is the other ~20%.
      - Whether the 4N run waits for them is the user's call.
  - **The run** started 2026-10-17 at 04:38 (the user's go; `explorer.exe runs\ffa4n.cmd`, the 6-attempt restart loop;
    attempts in `runs/ffa4n-runs.log`, evaluations in `runs/ffa4n/panel.jsonl`).
    - The pool and panel are as proposed: league v2's 16 averages, the lr4a average, the 3B clone and the Phase 2 PPO
      bot; the panel is league v2's final average (0 Elo), the lr4a average and the 3B clone.
    - Team games are not in it: the recommendation was to add them after 4N.
    - ETA ~16–20 h. Then Gate 4N, with the start's baselines on the same games.
    - **Results (04:38–22:36, one sitting, exit 0): the run failed.**
      - **Ranked 1v1 collapsed after update ~2,000.**
        - The averages against the start (league v2's final average), 20 games each: 0.35, 0.55 and 0.65 at updates
          0–1,000, then 0.35, 0.25, 0.15, 0.15, 0.10 and 0.15.
        - Against the lr4a average: 0.70 at update 0, 0.05–0.35 from update 2,500. Against the 3B clone: 0.94, then
          0.75–0.81.
        - The training win rate against the league, by quarter: 0.27, 0.19, 0.06, 0.08 (league v2: median 0.31).
      - **Against Hard nations nothing moved.**
        - FFA: 1 win in 28 up to update 1,000, then 0; the mean rank went from 19.2 to 20.75.
        - Humans vs Nations: 0 of 28 at every evaluation.
      - **Training games.**
        - FFA: win rate 0.01–0.02, mean reward flat at −0.86.
        - Humans vs Nations (Medium and Hard): 0.14–0.19, mean reward −0.73 to −0.62. The check gave 0.86 against
          Medium alone, so the Medium games won far less in training.
      - **PPO's statistics looked normal.**
        - KL to the clone ~0.037, approx_kl 0.002, 2.3% clipped, entropy ~0.19, explained variance ~0.97.
        - 134–139 decisions/s, memory flat at ~19.9 GiB, evaluation rounds of 20–30 minutes, no worker errors.
      - In ranked 1v1 the bot attacked less as it trained: 12% of decisions in the first quarter, 10% in the last (the
        start: 16%). In FFA, among live seats, it attacked slightly more (~10% to ~11%).
      - **Why 1v1 collapsed (2026-10-18): the setups' training signals were coupled.** This is the user's reading, and
        the evidence supports it. One network learned from all three setups at once, and their rewards reached it
        together in three ways.
        - **The value baseline.** One value head fit every setup's returns, so each setup's GAE advantages rest on a
          baseline shaped by the others. Over all setups' games, its mid-game outcome correlation fell from 0.72 in
          the first quarter to 0.49–0.55. League v2, on 1v1 only, held 0.76–0.82.
        - **Advantage normalisation.** Each minibatch normalises the advantages over all its rows. About 75% of them
          were FFA rows (mean reward flat at −0.86) or Humans vs Nations rows, so their returns set the mean and the
          spread that every 1v1 advantage is shifted and scaled by.
        - **One summed gradient.** Every row's gradient adds into one update, and the network did not keep the modes
          apart:
          - A gate probe found the MoE router using the same 3 of its 8 experts (0, 1, 5) in every setup. The other 5
            got 1–6% each.
          - From the gates alone, a nearest-centroid guess finds the setup in 51% of positions for the start and 59%
            for 4N's final average, against 33% for chance.
          - 4N moved those experts the same way in every setup: expert 0 down, expert 5 up.
          - The probe: 6 games (2 per setup, seed "gates-diag") for 500 decisions, with the start policy acting; it
            counts live seats after decision 60.
          - The 1v1 attack share drifted toward FFA's: 12.1% → 10.0% by quarter. League v2 held 12–14%.
        - **The cost.** 25% of the learner's rows came from 1v1, on their own ±1 rewards, and the averages still ended
          ~300 Elo below the start. The KL to the 3B clone stayed ~0.037, so the leash did not protect 1v1.
        - **No indexing bug.** The code was reread: `seat_rows` and the rollout in `trainer/ppo.py`, `trainer/env.py`,
          `bridge/worker.ts`, `bridge/game.ts` and `bridge/ranking.ts`. Every 1v1 learner row carries its own seat's
          observation, action and reward.
          - A control run checks the mixed code path before the next mixed PPO run: 4N's flags with only
            `ranked1v1=8` and `--minibatches 3`, ~4 h.
          - The user's call (2026-10-18): deferred until that run.
        - **If setups share a PPO run again,** the fixes follow from the three couplings:
          - per-setup advantage normalisation;
          - per-setup value targets: PopArt (Hessel et al. 2019), or a value head per setup;
          - a check of each setup's share of the gradient.
        - **Still open:** Humans vs Nations won ~0.15 in training. Its half-Medium, half-Hard games predict ~0.43 from
          the checks (0.857 against Medium, 0 against Hard). It is measured again at v0.35.2.
      - **Gate 4N is not played:** the evaluations already show the final average below its start in 1v1 and no gain
        against Hard nations. The 1v1 bot stays league v2's final average, `checkpoints/league4a2/avg004000.pt`.
  - **After 4N** (re-planned 2026-10-18 with the user, `docs/superpowers/plans/2026-10-18-after-4n-v035.md`). The
    user's calls: upgrade first, then specialists; the control run is deferred.
    1. **v0.35.2** (production since 2026-10-16). Decision 14 holds its scan.
       - The upgrade.
       - The 1v1 bot against the 3B clone at the new pin, 256 games (0.977 at v0.34.24). A short fine-tune if it lost
         strength.
       - The nations probes again: Easy, Medium and Hard nations, FFA with one seat and Humans vs Nations, 28 games
         each. The nations' AI changed, so 4N's numbers against them are history.
    2. **Human games of every mode at v0.35.2.** The crawler takes Ranked FFA, public FFA, public Team, Clan Wars,
       ranked 2v2 and ranked 1v1, kept apart from the old pin's records. Two analyses, whose tables go into this plan:
       - **What predicts winning, per mode** (the user's reward theories):
         - replayed records log tiles, troops, gold, kills and rivals alive per player every 10 s;
         - the win probability is fit per mode at minutes 5, 10, 15 and 20, with each measure's AUC and calibration;
         - it tests whether kills add anything once land is known, and whether a big empire with few troops per tile
           loses land later;
         - the best predictor is the potential if a mode needs shaping (decision 9).
       - **How humans play each mode**, per player-minute:
         - attacks, boats and builds by type;
         - nukes, alliances and donations;
         - the share of commands outside the 15 actions.
         - It tests the user's picture of FFA and Team, and it sizes item 3.
    3. **Alliance and donation actions**, perhaps warship moves, designed with the user while the crawl runs.
       - The crawl moves ahead of this item (it was second). Its analysis sizes the actions.
       - BC still waits for both, which was the reason the actions came first.
    4. **BC on all modes** with the mode as an input (decision 5a), with 3C's MoE routing test.
    5. **RL specialists, one per mode family, all from that clone** (decision 5, 2026-10-18). Each has its own gate.
       - The families:
         - ranked 1v1 (the league);
         - Ranked FFA (copies, on Asia and Australia first: Phase 6);
         - public FFA against the nations;
         - Team (copies as in Clan Wars, and Humans vs Nations).
       - Rewards stay terminal. Where a mode's results stay flat, late-game starts from v0.35's snapshots come first,
         then potential-based shaping (decision 9).
    6. **Distillation** of the specialists into one network, or one specialist per lobby type (decision 5b).
    7. **Then:** modifiers; bigger maps and the 64-player cap (Phase 6).

**Phase 5 — Exact nuke-interception predictor (≈1–2 weeks)**: silo choice as `PlayerImpl.nukeSpawn`,
`UniversalPathFinding.Parabola`, targetable window, copied SAM targeting (slots, cooldowns, upgrade ramp, stacked-launch
stagger). Gate A: frozen-world harness matches 100% per nuke. Gate B (metric): in-game agreement, unexplained mismatches = 0.
- Why it stays here (reviewed 2026-10-06):
  - SAMs are rare in ranked 1v1: 0.9 per game for both players, against 4.4–4.7 nukes;
  - 3B's coverage counts already show the interceptors at each target (decision 4);
  - humans already barrage with single clicks;
  - the predictor must match the game's targeting exactly, on every upgrade.
- It is an input, never a mask: the bot may strike a defended cell with enough nukes.

**Phase 6 — Broaden & scale**:
- public FFA/Team with nations (Easy→Impossible) and all modifiers (water nukes via terrain deltas), through the public
  sampler (playlist map weights, team configs, modifier pool and exclusions);
- an in-memory map loader with dihedral augmentation;
- procedural maps (a TS port of the generator, tested byte-for-byte on official maps);
- Normal-size maps;
- nations join the league as opponents;
- per-mode specialist fine-tunes if they win;
- a bigger model: the S/M/L CNN picked from BC scaling curves, and more MoE experts by upcycling;
- the player transformer's attention cap (64 players, Phase 3A) revisited for FFA and Team:
  - Casual lobbies often hold more than 64 real players (the user, 2026-10-03).
  - Tribes are fillers: harmless, and usually extinct before the end, so they never need a slot. The relevance rule
    already ranks every real player above every tribe.
  - Options: a larger cap for those modes, or ranking real players by threat and distance. The transformer's weights
    do not depend on the cap, so raising it keeps the trained network (a short fine-tune at most).
- Ranked FFA's maps (v0.35) at Normal size, in cells (W × H):

  | Map | Cells |
  |---|---|
  | Europe | 363 × 209 |
  | North America | 350 × 181 |
  | South America | 218 × 297 |
  | Africa | 244 × 254 |
  | Australia | 250 × 188 |
  | Asia | 250 × 150 |

  - Only Asia and Australia fit the ranked grid (256 × 192).
  - All six need ~368 × 304 cells, 2.3× today's grid, or 368 × 256 (1.9×) with the two tall maps transposed.
- an optional rented GPU run.

The sampler, augmentation and CNN sizes were moved here from Phase 3 on 2026-10-03.

**Phase 7 — Deployment**: ONNX export; headless Node bot client (core sim + onnxruntime) joining private lobbies on a
self-hosted server at the pinned release; checkpoint chosen by lobby mode (AGPL: publish source if hosting a modified
server for others). Official servers: the owner allows testing in casual multiplayer games from the AI-DEV account (not
ranked, no abuse). The connection must keep Turnstile (e.g. the official web client in the user's own browser session,
or a path the maintainers provide); never bypass it. To decide when the bot is strong enough to be worth testing.
- **The user's client** (the user's requirement, 2026-10-15): the user plays multiplayer games themselves, with a button
  that hands their seat to StockShip and takes it back, and a view of the moves it makes. The user expects the owner
  may also allow ranked tests; until the owner confirms it, only casual lobbies from the AI-DEV account.
  - Proposed: a browser extension (or userscript) on the official client, so the user's own session keeps Turnstile.
    It forwards the game's start and every turn to a local StockShip server, which replays the same game with the
    bridge (as it replays records: the hashes keep it in sync), runs the network and returns the seat's next intent;
    the extension sends it as the user's. Decision 12's limits hold (one intent per 0.5 s). An overlay lists the moves
    it makes, and could show what it would do while switched off.
  - When: after 4N's code, while 4N trains (it needs no training time on the GPU). To be designed with the user first.

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
