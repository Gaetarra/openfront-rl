# Step 7: behaviour cloning on all modes (design)

Designed with the user on 2026-10-19 and approved by them the same day. This is step 7 of the after-4N plan
(`docs/superpowers/plans/2026-10-18-after-4n-v035.md`) and PLAN "After 4N" item 4. The numbers below were measured on
2026-10-19 in `data/v0.35.0/` and in the code of the pinned game (v0.35.2).

## Why

- Item 5's RL specialists all start from one behaviour-cloned network (decision 5, 2026-10-18):
  - ranked 1v1 (the league);
  - Ranked FFA;
  - public FFA against the nations;
  - Team, with Clan Wars and Humans vs Nations.
- Today's clones learned ranked 1v1 only, and none of them has step 6's diplomacy and warship actions.
- Decision 5a makes the mode an input, so one network can play each mode the way its humans do.
- Mixed BC has none of the couplings that cost 4N ~300 Elo in 1v1. Its labels are fixed human actions, with no value
  baseline and no advantage normalisation (decision 5).

## Goals

- One arch v5 network that imitates the humans of every mode: ranked 1v1, ranked 2v2, Ranked FFA, Clan Wars, public
  FFA, public Team and Humans vs Nations.
- In ranked 1v1 it is at least as good as the 3B clone (`bc3b-v4`), the KL reference of today's league.
- In the other modes it uses the new actions the way humans do: alliances in FFA, donations in Team, and warship moves.
- It becomes every specialist's start and KL reference.

## Non-goals

- Maps that don't fit the 192 × 256 cell grid (Phase 6).
- Records with modifiers (item 7).
- The MoE-vs-dense routing test, which moves to Phase 6 (below).
- RL (item 5), and any new action or input beyond the mode input.

## Anti-goals (failures even if everything else works)

- A clone weaker than the 3B clone in 1v1. The league's next specialist would start lower, and what ~21,000 older
  records taught would be lost.
- A value head that learns the outcome from how its players were picked rather than from the position. If both of a
  game's players came from the winning team, both would always be +1.
- Rows that teach "do nothing" because the human was away.
- A mode input that differs between a replayed record and a training game of the same mode.

## What the data allows (2026-10-19)

- **Only v0.35 records replay at the pin.**
  - `data/` holds 20,900 ranked 1v1 records of v0.34.17–v0.34.22. They replay only in `pins/v0.34.23`, whose code
    predates arch v5.
  - The 3B clone learned from them: 33.6M rows in one pass. Its weights are the only way that data can reach v5.
- **The grid.** A record can train the network only when its map fits the specialists' cell grid (192 × 256). Both
  sizes of a map share one grid.

  | Mode | Records | Fit the grid | And no modifiers |
  |---|---|---|---|
  | Ranked 1v1 | 2,852 | 2,852 | 2,852 |
  | Ranked 2v2 | 682 | 682 | 682 |
  | Ranked FFA | 664 | 230 | 230 |
  | Clan Wars | 9 | 9 | 9 |
  | Public FFA | 2,269 | 628 | ~580 |
  | Public Team | 1,610 | 486 | ~410 |
  | Humans vs Nations | 300 | 95 | ~75 |

  - The misses are the big maps: Brain Rot, Europe, Africa, the Americas, Giant World Map and others.
  - Every map the specialists will train on fits.
- **The top-up crawl** (the user, 2026-10-19; `runs/crawl035b.cmd`, ~7 h of API):
  - It takes every mode in full since v0.35 went live, so public FFA and Team also get the games the first crawl's
    shares left out.
  - Expected afterwards, fitting the grid and without modifiers: ~5,800 ranked 1v1, ~1,470 ranked 2v2, ~420 Ranked FFA,
    ~4,400 public FFA, ~1,560 Team and ~280 Humans vs Nations records.
- **Players per record**, among those that fit:
  - 2–10 in the ranked modes;
  - a median of 25 in public FFA (22 sent a command), 28 in Team (24) and 15 in Humans vs Nations (14), up to 123.
- **Replay cost per 5-tick window**, on one worker:
  - the simulation takes 8–11 ms on Normal maps in every mode, and 2 ms on a Compact 1v1 map;
  - each observed player adds ~0.25 ms.
  - Public games cost no more simulation than ranked ones: the map and the 400 tribes dominate.
- **BC speed is set by the GPU.**
  - The 3B night ran at ~600 rows/s, with 168 rows a window (14 workers × 6 games × 2 players). So a 15-hour night
    is ~32M rows.
  - Rows per window can't grow much: at 168 rows, BC peaked at 9.9 GB of the 12-GB card.

## The design (decided)

### Starting weights: the 3B clone, grown into v5 (the user, 2026-10-19)

- `checkpoints/bc3b-v4/latest.pt` (arch v4, 8 SwiGLU experts) becomes an arch v5 network. Every trained weight is
  kept.
- **Layers v5 keeps unchanged** are copied as they are.
- **New inputs get zero weights**, so the grown network computes exactly what v4 computed:
  - the 7 new channels in the stem and in the maps' skip convolution;
  - the 6 new player features in the player embedding. They sit between v4's 25 features and the 128 land features,
    so this layer is not a prefix copy;
  - the mode scalars in the scalar layer.
- **New outputs keep v5's own initialisation**:
  - the warship-source map;
  - the donation's share kind;
  - the 4 diplomacy pointers;
  - the warship move's destination and fleet heads.
- **The 5 new types' rows of the type head start with zero weights and a bias of −8** (Claude's call).
  - At logit 0, the new types that are legal in 1v1 would take a few percent of the probability: alliance requests to
    tribes, and warship moves once there is a warship. That raises the start's 1v1 NLL above the 3B clone's.
  - At −8, the grown start matches the 3B clone's action distribution to ~1e-3 NLL. So the pre-night check, Gate 7's 1v1
    bar and the forgetting trigger below all share one baseline.
  - The new types still learn fast. Adam moves each of the row's 512 weights by about the learning rate per update,
    so a type that labels rows gains logit within a few hundred updates.
- **What equal means at the start.** On the same observation, the grown network's logits for v4's 15 types and 12
  maps equal the 3B clone's, and so do its pointers, its attack and boat shares, its value and its router's gates
  (a test). On the 1v1 scoring rows its NLL is the 3B clone's to ~1e-3.
- **Fine-tuning:** a fresh optimizer with a peak learning rate of 1e-4, a third of the 3B night's 3e-4, then the usual
  warmup and cosine.
- **The new clone becomes every specialist's start and KL reference.** A v5 learner needs a v5 reference.

### Which players are imitated

The user chose the best-placed players (2026-10-19). The count and the per-side rule are Claude's calls.

- **Two players per game, as in ranked 1v1 today:** the best-placed human of each of the two best-placed sides.
- **Sides:**
  - in the FFA modes (ranked 1v1, Ranked FFA, public FFA), each human is a side;
  - in the team modes (ranked 2v2, Clan Wars, Team), each team is a side;
  - in Humans vs Nations the humans form a single side, so both players come from it.
- **Order:** the winning side first, then the other sides by their best-placed human.
  - The winner is the record's `winner`: `["player", id]`, or `["team", name, ...the winning humans' clientIDs]`.
  - When a nation or the nations' team won, no side of humans won, and the order is by placement alone.
  - That happens in 169 of the 300 Humans vs Nations games, recorded as `["team", "Nations"]`, and in 1 of 2,269
    public FFA games, as `["nation", name]`.
  - Such records are kept, and both their players are −1.
- **Each human's team** comes from the replayed game: the game assigns the teams when it is created, before the first
  window. So the players are picked once the game exists.
- **A human's placement**, from the record's player stats:
  - alive at the end ranks above dead;
  - among the dead, a later `killedAt` ranks higher;
  - ties go to the higher peak tiles (`stats.tiles[0]`).
- **Who can be picked:** a human whose stats show peak tiles above 0.
  - The players without stats are almost all lobby joiners who never sent a command: 1,043 of the 1,074 in 400 public
    FFA records. Every winner has stats.
  - Ranked 1v1 keeps both players, as today. The crawler already drops records where a player never spawned.
  - If only one side has two eligible humans, both come from it. A record with fewer than two eligible humans is
    skipped.
- **Why two players per game:**
  - Whatever the count, the GPU caps a night's rows, and the workers have CPU to spare. Two players per game spread
    those rows over 2–4× as many games.
  - One player per side keeps the value targets balanced.
- **Why the best-placed:** public lobbies are mostly casual players, and the best-placed are the better ones. RL then
  improves on them (item 5).

### Value targets

- A player on the winning side gets +1, every other player −1, discounted as today (γ to the windows left).
- **A fix:** replays give every player of a team game −1 today. A team win is recorded as
  `["team", name, ...clientIDs]`, and `RecordGame` reads only `["player", id]`.
- **Not decision 9's rank-graded FFA reward.** A record holds no nation's death order, and each specialist fits its
  own value head anyway.

### Rows out of the loss

- **Disconnected players:** while a player is disconnected (the game's `isDisconnected()`, set from the server's
  `mark_disconnected` intents), its rows label UNMAPPED, which the loss skips. They would teach "do nothing".
- **Dead players:** their rows stay. Only the no-op is legal for them, so they add nothing to the policy loss.

### The mode input (Claude's calls, decision 5a)

- **Ten scalars after v4's 26, so 36 in all.** `bridge/obs.ts` computes them from the game's config and players, so a
  replayed record and a training game of the same mode give the same values:
  - **the lobby kind**, one-hot: ranked 1v1, ranked 2v2, Ranked FFA, Clan Wars, FFA (unranked), Team (unranked, not
    Humans vs Nations) and Humans vs Nations.
    - It comes from the config's `rankedType`, `gameMode` and `playerTeams`.
    - So the training setups match: `ranked1v1` is ranked 1v1, `ffa` is FFA, and `hvn` is Humans vs Nations.
  - **the players:** log2(humans + nations) / 8;
  - **the teams:** log2(the number of teams) / 6, and 0 in the FFA modes;
  - **the nations' difficulty:** Easy 0.25, Medium 0.5, Hard 0.75, Impossible 1, and 0 without nations.
- **Older networks are unaffected.** v2–v4 networks read only their 26 scalars (`V4_SCALARS`), so they play unchanged.
  v5 reads all 36, and the only v5 networks so far, the speed check's fresh ones, need no conversion.
- **No modifier inputs**, since modifier records are left out. Item 7 adds both.

### Records (Claude's calls)

- **Source:** `data/v0.35.0/` only.
- **Kept:** records whose map fits the grid, with no `publicGameModifiers` set except `isCompact` (a size, which the
  network already reads) and no `disabledUnits`.
- **The index** gets the fields these filters need (`playerTeams`, the modifiers and the disabled units), so the
  trainer can filter without opening records. A one-off pass fills them in for the records already there.
- **Held out:** 5% by game ID (`is_val`), as today.
- **Elo** (`--min-elo`, `crawl.elo`) stays a ranked 1v1 filter, computed over the ranked 1v1 entries only. Today it
  assumes two players in every entry.
- **Game version:** production ran v0.35.2 at 11:50 UTC on 2026-10-19, and the crawler skips records of any other
  game commit. If a later release changes the simulation, its records wait for a pin upgrade (decision 14), which is a
  step of its own.

### Mode shares: replay workers per family (Claude's calls)

- **How shares work:** as with PPO's `--mix`, each worker replays one family's records. Every game slot gives 2 rows a
  window, so a family's share of the rows is its share of the workers.
- **The families:** ranked 1v1, ranked 2v2, Ranked FFA, public FFA, Team (with Clan Wars) and Humans vs Nations.
- **The proposed night** (14 workers × 6 games, ~32M rows; the passes assume the crawl's expected counts):

  | Family | Workers | Rows | Passes over its records |
  |---|---|---|---|
  | Ranked 1v1 | 3 | ~6.9M | ~0.7 |
  | Ranked 2v2 | 2 | ~4.6M | ~1.9 |
  | Ranked FFA | 1 | ~2.3M | ~2 |
  | Public FFA | 4 | ~9.2M | ~0.5 |
  | Team and Clan Wars | 3 | ~6.9M | ~1.2 |
  | Humans vs Nations | 1 | ~2.3M | ~3 |

- **Why this split:**
  - Public lobbies, where the bot will play (4N), get ~57% of the rows.
  - Ranked 1v1 gets a fifth, since the grown network already holds what the old records taught.
- The worker counts are knobs. The plan recomputes the passes after the crawl.

### The routing test moves to Phase 6

This move follows from the user's choice of the grown start (2026-10-19).

- **What 3C's test does:** it pits the MoE core against a dense core of equal parameters and compute, to pick the core
  once the router can blend by mode.
- **Why it can't run here:** the grown clone inherits the 3B clone's core (8 SwiGLU experts). Only a run from scratch
  could pick another core, and it would lose the old 1v1 data.
- **Where it goes:** the next time the core is chosen, which is Phase 6's model growth (more experts by upcycling, a
  bigger model).
- **What step 7 still answers:** 3C's other question, whether the modes crowd into the same experts.
  - It logs the router's mean weight per expert in each mode, in training, validation and the score.
  - If the modes crowd into the experts the 3B clone already favours (one took 54–77% of the weight), Phase 6's growth
    starts with upcycling, a load-balancing term or router noise.

### Scoring and validation per mode

- **A new fixed scoring set:** the held-out records of every family among games started before 2026-10-19 12:00 UTC,
  with a fixed layout per family.
  - It is fixed once the top-up crawl has run, since later crawls add only later games.
  - `--score` scores it. The old set's records desync at v0.35, so they can be scored only from the old pin.
- **Every statistic of `measure` is also logged per lobby kind** (`bc/nll_<kind>`, `router/<kind>_<i>`), read from the
  mode input, in training, validation and the score.
- **Step 6's leftover check:** labels outside the masks are counted per action type (`labels/illegal/<type>`). This
  checks the window-start timing and lake warships, and runs on the scoring set before the night.
- **Validation:** one worker per family.

### The run

- `trainer.bc --init checkpoints/bc3b-v4/latest.pt` with the growth and the mix (the flag names are left to the plan).
  The user starts it in the start-or-resume loop.
- About 24,000 updates of 8 windows, ~15 h at the 3B night's speed. The smoke run measures v5's real speed.

### Risks, and what to do if they show

- **1v1 forgets:** in the first 2,500 updates the ranked 1v1 validation NLL rises more than 0.02 above the grown
  start's. Then stop, and lower the learning rate (5e-5) or give ranked 1v1 more workers.
- **A small family overfits** (Humans vs Nations, ~3 passes): its validation NLL turns up while training's falls. Then
  move its worker to public FFA.
- **Memory:** public records have a larger player table (up to ~590 rows against 403), and that grows the land-feature
  tensors. The smoke run checks the peak against the card.
- **Item 5 has to know about one mismatch:** the FFA training setup has 4 seats, where public lobbies have ~25 humans.
  The players input tells the two apart.

## Gate 7 (bars fixed before the night)

- **Score**, on the fixed set, against the 3B clone and on the rows both can express:
  - ranked 1v1's `bc/nll` within +0.01 of the 3B clone's;
  - every other lobby kind's `bc/nll` below the 3B clone's;
  - the new types' heads beat a uniform pick over their legal choices (target, cell, share).
- **Play:**
  - ranked 1v1 against the 3B clone, 256 games: at least 0.45, which is 2 SE under even;
  - the re-baseline's nations probes for both clones: training maps, 30-minute timer, 56 games each, in FFA with one
    seat and in Humans vs Nations, against Medium and Hard nations. The new clone may lose none of them by more than
    2 SE (win rate, plus mean rank in FFA);
  - no errors, and the bridge drops under 1% of the clone's actions (`invalid_share`), with the share per type reported.
- **Reported, not gated:** the clone's action mix per mode against the humans' (alliances in FFA, donations in Team).

## Verification

- **Tests:**
  - the grown network equals the 3B clone on v4's outputs (logits, pointers, shares, value, gates) on real
    observations;
  - the mode input for each lobby kind, with a record and a training game of the same kind giving the same values;
  - the player choice on fixtures:
    - in the Team fixture, the winning team's best human and the best of the runner-up team;
    - in a 1v1 fixture, both players;
    - a nations' win: both players by placement, both −1;
  - team wins give +1 to the winning team's players;
  - a disconnected player's rows label UNMAPPED;
  - the replay mix: each worker's rows come from its family's records;
  - the index's new fields and the one-off fill.
- **Before the night:**
  - on ranked 1v1, the grown checkpoint scores the same as the 3B clone, to about 1e-3;
  - the per-type illegal labels;
  - a short smoke run: finite losses, rows/s and peak memory.
- **After the night:** the score, then Gate 7's matches.

## Left to the builder

- Flag and config names, and how a replay worker is told its family and its player count.
- How the trainer learns which maps fit the grid. The bridge's `mapGrid` and `shapeOf` are the source of truth.
- How the index fields are filled in, and where the per-kind buckets live in `measure`.
- The validation and scoring layouts per family.
