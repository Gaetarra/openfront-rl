# Step 6: diplomacy and warship actions (design)

Approved by the user on 2026-10-19; the implementation plan is
`docs/superpowers/plans/2026-10-19-step6-diplomacy-and-warships.md`. Built on 2026-10-19 in commits d741a5a..c68fd8f
on phase3; the results are in PLAN "After 4N" item 3. This is step 6 of the after-4N plan
(`docs/superpowers/plans/2026-10-18-after-4n-v035.md`) and PLAN "After 4N" item 3. It was designed with the user on
2026-10-18/19, and the game facts below were checked in the code of the pinned game (v0.35.2) and in the human
records of `data/v0.35.0/`.

## Why

- None of the bot's 15 actions (`bridge/actions.ts`, `Act`) is diplomacy or a warship move.
  - It cannot request, accept, renew or break an alliance, donate troops or move a warship.
  - Alliance requests that nations send it lapse unanswered after 20 seconds.
- In public FFA and Team, these commands are most of what humans do beyond the 15 actions
  (`scripts/mode_analysis.py play`, per player-minute):

  | | Public FFA | Public Team |
  |---|---|---|
  | Alliance requests (and accepts) | 0.69 (87% of players) | 0.19 (45%) |
  | Alliance renewals | 0.18 (37%) | 0.04 (11%) |
  | Alliances broken | 0.05 (25%) | 0.01 (9%) |
  | Troop donations | 0.002 (1%) | 0.52 (53%) |
  | Warship moves | 0.06 (10%) | 0.11 (15%) |
  | Commands outside the actions: today | 8.7% | 8.2% |
  | Commands outside the actions: after this step | 1.0% | 1.2% |

  Ranked FFA falls from 2.9% to 0.6%, Humans vs Nations from 6.1% to 0.9%, ranked 2v2 from 2.0% to 0.8%.
- Step 7 (behaviour cloning on all modes) can only learn what the bot can express, and step 8's RL specialists can
  only use actions that exist. So the actions come first.
- The user's picture of the modes, checked in the code and the records:
  - **FFA:**
    - Alliances are almost essential.
    - Breaking one sometimes wins despite the traitor debuff. Otherwise the alliance runs out after 5 minutes,
      with no penalty.
    - Donations are useless: public FFA lobbies switch them off. An allied nation can still receive them, but
      humans almost never donate (0.002 a minute).
  - **Team:**
    - Donations are almost essential. Players behind the front can only use their troops by donating them to the
      teammate who is fighting (the "donation race").
    - Alliances matter more as teams get smaller. 17% of players request one in two-team games, 52% with 3–7 teams,
      70–82% in Duos to Quads, and 87% in public FFA.
  - **Warships:**
    - Piracy on busy routes earns gold and cuts the enemy's trade.
    - In Team games, control of the sea decides the invasions.
    - Fleets snowball through veteran levels.
    - Humans move warships rarely because their attention is limited, not because moves are weak. A bot has no
      such limit, so this is where it can beat humans.

## Goals

1. Five new action types, aimed the way a human aims them:
   - request an alliance (which also accepts one);
   - renew an alliance;
   - break an alliance;
   - donate troops;
   - move warships.
2. The inputs the bot needs to use them well.
3. Behaviour-cloning labels for these commands from the human records.
4. Legality masks that follow the game's own rules exactly.
5. A new network version (v5) with the heads for the new actions. Older checkpoints keep playing unchanged.
6. Tests, and updated docs.

## Non-goals

- **Training.** BC on all modes is step 7, and the RL specialists are step 8.
- **Commands left out** (the user, 2026-10-19). Each is rare in public FFA and Team, per player-minute:
  - gold donations: 0 and 0.016;
  - embargoes: 0.008 and 0.019;
  - target marks: 0.010 and 0.023;
  - rejecting a request: 0.024 and 0.012. A request the bot ignores lapses after 20 seconds anyway;
  - chat and emoji: 0.064 and 0.063;
  - MIRVs, cancelling a boat, deleting a unit: at most 0.013 each.
- **Nations' opinion of the bot** (`relation`). The game client doesn't show it, so the bot doesn't see it either.
- **Alliances between other players** (who is allied with whom).
- **A switch to turn diplomacy off in a setup.** See "Risks".

## Anti-goals (failures even if everything else works)

- **A v4 checkpoint that plays differently than today.** The 1v1 bot (`checkpoints/league4a2/avg004000.pt`) and
  every league member must play exactly as before.
- **A rule value written into our code.** Cooldowns, durations and ranges come from the game's `Config` at
  runtime (CLAUDE.md).
- **An edit inside the game submodule.**
- **Masks looser than the game's rules.**
  - A diplomatic action the mask allows must execute.
  - A move whose destination is in another sea is the only expected drop.
  - The drop rate is measured (Verification).

## The game's rules this design relies on (v0.35.2)

**Alliances** (`packages/engine/src/execution/alliance/`, `PlayerImpl`):
- **Requesting:**
  - A request (`allianceRequest`) to a player who has one pending to you accepts it.
  - A request lasts 20 s (`allianceRequestDuration`), then counts as rejected.
  - You can request the same player again only after 30 s (`allianceRequestCooldown`), unless they have a request
    pending to you.
  - Requests are refused to a friend or a disconnected player, and when the lobby disables alliances
    (`canSendAllianceRequest`).
- **Duration:** 5 minutes by default (`allianceDuration`; a lobby can set 1–15).
- **Renewing:**
  - The game offers renewal in the last 30 s (`allianceExtensionPromptOffset`, `allianceInfo().canExtend`).
  - The alliance renews only if both sides send `allianceExtension`.
  - An alliance that runs out ends with no penalty.
- **Breaking** (`breakAlliance`):
  - The breaker becomes a traitor for 30 s, unless the other side already is one or has disconnected.
  - While the flag lasts, attackers lose half as many troops per tile against the traitor and take its tiles 25%
    faster.
  - Nations refuse a traitor's requests 9 times in 10.
- **While allied:**
  - Neither side can attack the other, and attacks already under way retreat.
  - Trains pay 35k instead of 25k at each other's stations.
  - Ports send trade ships to an ally's ports more often.
  - The alliance lifts the 5-minute trade embargo that an attack sets.
  - Warships and SAMs ignore the ally's ships and nukes. In Team games this holds even when they are aimed at
    teammates.
  - Nukes in flight between the two are destroyed when the alliance forms.
  - A nuke breaks the alliance if its blast would hit one of the ally's structures, or more than 100 of the ally's
    tiles (tiles in the outer ring count half).
- **Who answers requests:**
  - Nations decide by rules: threat, relation, strength and team game. In team games, Hard nations refuse three
    requests in four, unless the requester threatens them.
  - Tribes accept every request and every renewal.
  - Humans answer by clicking.

**Donations** (`DonateTroopExecution`, `PlayerImpl.canDonateTroops`):
- **Recipients:** only friends, meaning teammates or allies.
- **Public lobbies:**
  - Donations are switched off outside Team mode (`donateTroops()`).
  - The switch only refuses donations to humans, so a nation can still receive them.
- **Cooldown:** one donation per recipient every 10 s (`donateCooldown`), counting troops and gold together.
- **Cap:** a donation is cut to the recipient's room under its troop cap.
- **Amount:** the client sends a share of the sender's troops, set by the troop slider. With no amount given, the
  game sends a third (`defaultDonationAmount`).

**Warships** (`WarshipExecution`, `MoveWarshipExecution`, `TradeShipExecution`):
- **Moving:**
  - `move_warship {unitIds, tile}` sets the patrol point of each listed warship. Ships in another water body are
    skipped.
  - A box selection gives every selected ship the same patrol point.
- **Patrolling:**
  - A warship roams within 50 tiles of its patrol point (`warshipPatrolRange` / 2) and picks its own targets.
  - Order of targets: transport ships first (one shot each), then enemy warships, then trade ships.
  - Trade ships count only within 100 tiles of the patrol point; any target must be within 130 tiles of the
    ship (`warshipTargettingRange`).
- **Piracy:**
  - Chasing a trade ship, a warship moves 2 tiles a tick instead of 1, and captures it within 5 tiles.
  - The captured ship sails to the captor's nearest port in the same water body and pays on arrival. The payout
    is up to 75k, plus 50 gold per tile sailed, times the gold multiplier.
  - A trade ship is safe for 2 s after each coastal tile (`safeFromPiratesCooldownMax`).
- **Veterancy:**
  - Levels go from 0 to 3. Each adds 20% max health and 20% shell damage, so 1,000 health becomes 1,600 at
    level 3.
  - Sinking an enemy warship gives a level at once. Sinking 10 transport ships or capturing 25 trade ships also
    gives one.
- **Healing:**
  - Within 150 tiles of one of its owner's ports, a warship heals 1 health a tick.
  - A port docks as many ships as its level and heals them 5 health a tick per level, shared among them.
  - A ship below 75% health retreats to a port, except in the 5 s after a manual move.
- **How humans move them** (records): one warship in 87–94% of move commands, four or more in 2–4%.

## The design (decided)

### The actions

The five types follow the 15 (`Act` 15–19). This is the user's scope, 2026-10-19:

| Type | Arguments | Intent | Legal when (exact, from the game's own checks) |
|---|---|---|---|
| Alliance request | a player | `allianceRequest` (also accepts a pending request from that player) | `seat.canSendAllianceRequest(p)` |
| Renew alliance | a player | `allianceExtension` | `seat.allianceInfo(p)?.canExtend` |
| Break alliance | a player | `breakAlliance` | `seat.isAlliedWith(p)` |
| Donate troops | a player; a share of 1–100% of the bot's troops | `donate_troops` | `seat.canDonateTroops(p)` and `p` below its troop cap |
| Move warships | a source cell; alone or with its fleet; a destination cell | `move_warship` | a cell holding one of the bot's warships; a water cell |

- **Players:**
  - Players are rows of the player table, as for attacks and retreats.
  - Tribes can be chosen wherever the game allows it. They accept every request (see "Risks").
- **Donation amount:** an attack's or a boat's share sends `share`% of the bot's current troops. A donation sends
  the same.
- **Type mask:** a type is legal when at least one of its targets is.

### Move warships (Claude's calls, agreed with the user on 2026-10-19)

- **Source:** a cell holding one of the bot's warships. The anchor is the bot's warship in that cell nearest the
  cell's centre.
- **Alone or fleet:**
  - "Alone" moves the anchor only, which takes it out of its fleet.
  - "Fleet" moves every warship of the bot that shares the anchor's patrol point.
- **Destination:**
  - A water cell. The patrol point is the water tile nearest the cell's centre that lies in the anchor's water
    body.
  - Ships sent to the same cell therefore share a patrol point and form a fleet.
  - If the anchor's water body has no tile in that cell, the action is dropped as a counted no-op.
- **Choice order:** the destination is chosen knowing the source (autoregressive), so the bot can send two fleets
  to different places.
  - When acting, the destination head reads the sampled source.
  - When learning, it reads the source of the stored action.
  - Its entropy and KL terms use that same source, which makes them a one-sample estimate over sources.
- **Docked ships can be moved,** as humans move them. A manual move cancels a repair retreat.

### The inputs (Claude's calls, agreed with the user on 2026-10-19)

Each player row (`bridge/obs.ts`, 17 features today) gains:
- **ally:** an alliance, not a teammate. Today's "friendly" flag covers both.
- **pending requests:** one flag for a request from them to the bot, one for a request from the bot to them.
- **alliance time left,** as a share of the alliance duration (0 without an alliance).
- **renewal:** one flag for "they asked to renew", one for "the bot asked to renew".

The map channels gain:
- **warship health and veteran level,** for the bot's warships and the others'.
- **patrol points:** how many of the bot's warships have their patrol point in each cell.
- **trade ships,** which the bot does not see at all today:
  - the ones it could capture: bound for a port that is neither its own nor a friend's, and not safe near a coast;
  - the ones that pay it or its friends: their own, and those bound for their ports.

The game client shows a human player the alliance timers, the renewal prompts, and each warship's health and veteran
level.

### Labels (`bridge/replay.ts`, `labelOf`)

| Human command | Label |
|---|---|
| `allianceRequest` | alliance request, the recipient's row |
| `allianceExtension` | renew alliance, the recipient's row |
| `breakAlliance` | break alliance, the recipient's row |
| `donate_troops` | donate troops, the recipient's row; share = round(100 × troops ÷ the sender's troops), clamped to 1–100; with no amount, 33 |
| `move_warship` | move warships; source = the cell of the command's first warship; fleet when the command lists two or more; destination = the cell of the clicked tile |

- A recipient or warship that no longer exists makes the label `UNMAPPED`.
- Every other command stays `OUT_OF_SET`.
- **Legality:** as today (`trainer/bc.py`, `label_legal`), a label that the masks forbid stays out of the loss and
  is counted.
- **Fleet labels are approximate.** A human box selection may span several fleets, while the bot's fleet means
  ships that share a patrol point.

### The network (arch v5)

- **Types:** 20 action types.
- **Pointers:** four new player pointers, one per diplomatic type. Each has its own query over the shared player
  keys, like the retreat pointer.
- **Donation amount:** a third kind of the share head, read from the recipient's token. The other two kinds are
  attacks and boats.
- **Warship moves:**
  - two new heat maps, one for the source and one for the destination, which reads the chosen source;
  - an alone-or-fleet choice, which also reads the chosen source.
- **Older checkpoints:** v4 and earlier keep playing exactly as today.
  - They read only the types, masks and inputs they have, as `ActionDist` already does for v1–v3.
  - Under the new masks, they never take the new types.
- **Starting from v4 weights:** a v5 network can start from v4 weights, with every new input weight and head at
  zero. Whether step 7 does this is step 7's call.
- **The mode as an input** (decision 5a) stays in step 7.
  - Step 6 trains nothing, so no v5 checkpoint exists before step 7.
  - Step 7 can therefore add the mode inputs to v5 itself before training it, with no v6 and no second
    compatibility pass.

### How far into the trainer v5 goes (Claude's call)

- **In step 6:** v5 works in every code path that exists today:
  - the worker protocol and the environment (the observations and masks);
  - `ActionDist` (sampling, log-probabilities, entropy and KL);
  - PPO's rollout storage and learning (`HostRows`, the stored action and masks, the minibatch path that feeds the
    stored source to the destination head);
  - behaviour cloning's labels, loss and scoring;
  - `trainer.eval` and game recording.
- **Not in step 6** (step 7's work): behaviour cloning reading `data/v0.35.0/` with all its modes, and the mode
  inputs.
- **Why:** after step 6, any command in CLAUDE.md runs with a v5 network. So step 7 and step 8 start from training
  and don't have to finish this step first.

### Speed and memory (Claude's call, from the project's own rule)

- **What v5 adds:**
  - more input channels;
  - two heat maps, each with a per-row cell mask in PPO's pinned memory;
  - the destination head.
- **Current headroom:** ranked PPO at `--minibatches 12` already holds about 8.7 GB of the 12 GB card and 7.6 GiB
  of pinned RAM. When the card overflows, Windows pages silently and a run gets 2–4× slower.
- **Budget:** v5 keeps at least 95% of v4's PPO decisions/s and doesn't spill the card at `--minibatches 12`.
  - This is the 5% rule the project applies to architecture changes (PLAN 3A's MoE rule, 3B's experts).
  - It is measured with `scripts/speed_check.py`, which gets a v5 option, on an idle PC.
- **If v5 misses the budget:**
  - first the cheaper choices: the destination reading the source through an embedding added to the shared decoder
    features rather than its own pass, and fewer or packed channels;
  - then, if it still misses, the user decides.

### Risks, and what to do if they show

- **Collusion in self-play.**
  - Two copies of the bot could ally and stop fighting. The game allows alliances in ranked 1v1, and 15% of ranked
    1v1 players request one.
  - Today's league opponents are v4 networks and cannot accept, so the risk starts only when v5 snapshots play each
    other.
  - Step 8 decides whether the 1v1 specialist masks diplomacy, which is one line in the type mask.
- **Allying tribes.**
  - Tribes accept everything, and an allied tribe can't be attacked for 5 minutes. The bot could block its own
    expansion that way.
  - BC from humans and RL should learn to avoid it. If they don't, mask tribes out of requests.
- **Request spam toward humans in casual lobbies.** The game's 30-s cooldown per recipient bounds it.

## Verification

- **Bridge tests (node:test), for each new type, in small games of the bridge's setups:**
  - Masks: the mask equals the game's own check on every player row.
  - Decoding: decoding gives the expected intent.
  - Execution, one tick later:
    - an alliance forms when both sides request;
    - a renewal registers;
    - a break marks the traitor;
    - a donation moves troops and is capped at the recipient's room;
    - a warship move sets the patrol point of the anchor alone, or of its whole fleet.
- **Labels:**
  - A new replay fixture: a v0.35.2 public Team or FFA record from `data/v0.35.0/` that holds every new command,
    and as short as possible.
  - The new commands' labels are legal under the masks, and decode back to the human's intent wherever the mapping
    is exact.
  - The counts of `UNMAPPED` and illegal labels are reported.
- **Trainer tests:**
  - Sampled v5 actions are legal.
  - Labels' log-probabilities, entropy and KL are finite.
  - The destination's distribution changes with the source.
  - A v4 checkpoint's outputs and sampled actions are unchanged on the same observation and seed.
- **Coverage:** `scripts/coverage.py` (its `IN_SET`) and `mode_analysis.py play` report the out-of-set shares in
  the table above.
- **The training paths, as smoke tests that train nothing:**
  - a few PPO updates with a fresh v5 network on each setup;
  - a few behaviour-cloning updates on the new fixture.
  - Both take minutes of GPU time.
- **Speed and memory:** `scripts/speed_check.py` with v5 against v4, judged by the budget above.
- **Smoke games:** a few games per setup (ranked 1v1, FFA, Humans vs Nations) with a random v5 policy:
  - no crash;
  - the valid and dropped counts per new type;
  - a drop rate near 0, except warship destinations in another sea.
- **Docs:**
  - PLAN: item 3's result, and the action space.
  - CLAUDE.md: the actions, the inputs, arch v5, and the out-of-set list.
  - The header of `bridge/actions.ts`.

Done means: the tests and the typecheck pass, the docs are updated, and the work is committed on `phase3`
without a push.

## Left to the builder

- How an action is encoded: today it is four integers `[type, cell, target, troops]`. A warship move needs a
  source, a destination and the fleet choice, so either a fifth field, or reusing `target` and `troops` by type.
- Channel and feature layouts and scales, and the head shapes, within the decisions above.
- The fixture record, the file organisation, the test names and the order of the work.
