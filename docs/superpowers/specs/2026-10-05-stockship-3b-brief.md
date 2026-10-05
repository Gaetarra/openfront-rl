# StockShip: brief for the 3B planning session (2026-10-05)

Input for a max-effort plan-mode session before any 3B code. It collects a teaching session's findings and the
user's decisions. **Nothing here is in `docs/PLAN.md` yet**: the planning session decides what goes in, and every
change to a PLAN decision states its reason against PLAN's own rationale.
The findings are facts checked in the code. Everything after them is **ideas to weigh, not instructions**: only the
items marked "user decision" were decided, and even those leave the details open.

## The goal: StockShip
The user named the bot **StockShip**, after Stockfish. The goal is the strongest possible bot, not a copy of human
play. Human habits (few commands per minute, single-missile nukes) only bind the BC step: RL may go past them. The game's
rules bind always, and the server's command limits (below) are game rules.
Caveat from Phase 2: the KL leash to the clone keeps early RL close to human style, and loosening it too early made the
bot worse (0.773 → 0.576). Human style fades as frozen evaluations allow, not at once.

## Findings (checked in the v0.34.22 code)
**SAMs**
- A SAM holds as many missiles as its level: it is in cooldown when `missileTimerQueue.length === level`
  (`UnitImpl.isInCooldown`). Each fired missile reloads in `SAMCooldown()` = 90 ticks (9 s).
- Interception is deterministic: no hit chance (`SAMLauncherExecution`, `SAMMissileExecution`).
- Each nuke is claimed by one SAM (`targetedBySAM`), so overlapping SAMs add their slots.
- Target priority: hydrogen bombs first, then nukes aimed closer to the SAM, then the ones exploding sooner
  (`computeTargetScore`).
- Range: `samRange(level) = 150 − 480 / (level + 5)`: 70 tiles at level 1, approaching 150.
- An upgrade raises the level, and with it the slots, **at once** (`increaseLevel`). The range ramps to the new level's
  over `samUpgradeDuration()` = 45 ticks (4.5 s). A defender can upgrade while nukes are in flight.
- So 2 atom bombs beat a level-1 SAM: one is intercepted, one lands.

**Salvos and bulk upgrades**
- `build_unit.amount` is 1–50 (`MAX_UPGRADE_AMOUNT`, `Schemas.ts:744`). `ConstructionExecution.completeConstruction`
  multiplies only atom and hydrogen bombs. `NukeExecution` staggers same-tick launches per silo.
- `upgrade_structure.amount` is 1–50 too (`UpgradeStructureExecution`).
- **New structures cannot be bulk-built**: one intent per structure, each on its own tile.
- The client offers (`RadialMenuElements.ts:470`, `InputHandler.ts:1121`, `Game.ts:993`):
  - atom bombs: radial menu ×1 / ×2 / ×5 / ×max, where max = min(affordable, loaded silo slots); pressing the hotkey
    twice toggles ×5;
  - hydrogen bombs: ×1 only;
  - upgrades: ×1 / ×5 / ×10 / ×max (max = what gold buys, ≤ 50).
- Humans still mostly click one at a time: 98% of ranked 1v1 nukes are single missiles (PLAN 3B).

**Building versus upgrading**
- Same price. A structure's cost counts the levels already owned: `unitsOwned` adds `unit.level()` (a structure under
  construction counts 1). City, port and factory: min(1M, 2^n × 125k). Each city level adds the same troop cap.
- Differences:
  - construction time: city and factory 2 s, port 5 s; upgrades are instant;
  - clicks: one per new building; upgrades bulk;
  - space: a new building needs a free tile ≥ 15 tiles from other structures;
  - risk: spread buildings lose one level to a nuke or a lost border, a stacked one loses all.
- No rule says one is better: the bot learns the trade-off.

**The server's command limits** (`src/server/ClientMsgRateLimiter.ts`, `SocketIngress.ts:159`)
- Per client: **10 intents per second and 150 per minute** (token buckets from the `limiter` package). Over the limit,
  the intent is **dropped** (logged "limit"), for humans too. An intent over 2,000 bytes gets the client kicked.
- The limiter is in `src/server`, not `src/core`: the simulation does not apply it. PLAN decision 12 lists "≤ 150
  intents/min/seat" as a deployment requirement. If the training environment does not copy the limiter, the bot learns
  plans it cannot carry out online.
- Humans are far below it (median 21 commands/min in ranked 1v1, p90 32), so BC teaches nothing about the budget.

**One action per decision (PLAN decision 11)**
- 0.5 s steps: at most 120 actions per minute, below the server's 150.
- An attack keeps running after its command until its troops are spent or it is cancelled (game knowledge, not checked
  in the code). A build decision therefore delays the next *new* attack by 0.5 s; it does not stop running ones.
- Evidence behind decision 11: only 10–15% of active 0.5 s windows hold 2+ human commands (61 replays, whole games).

## Ideas for the planning session (possible directions, not decisions)
**A. Action bins** (user decision 2026-10-05: three bins; the details are open). Each decision takes up to one action
from each bin, and every bin has its own no-op:
| Bin | Spends | Actions |
|---|---|---|
| Troops | troops | attack (target, share), boat (target, cell, share), retreat |
| Gold | gold | build (city, port, factory, silo, SAM, defense post), warship, upgrade (+ amount), atom / hydrogen bomb (+ amount, arc), MIRV |
| Diplomacy | nothing (except donations) | alliance request / accept / reject / break / extend, embargo, target player, donations? |
- Why: the bins split along the two resources, so actions within a bin compete and actions across bins barely do. They
  give more actions per decision at almost no GPU cost (one network pass, three small heads); a shorter step (Δ < 5)
  would multiply the decisions per game, and PPO is GPU-bound.
- It would replace decision 11 (one action per decision).
- The likelihood of a decision would be the product over bins, so log-probabilities, entropies and KLs would add per
  bin, as they do across today's heads.
- Ranked 1v1 has no diplomacy: that bin would stay masked to no-op until 2v2 (donations) and FFA.

**B. Command budget**
- The environment could copy the server limiter (10/s, 150/min), so training and online play obey the same budget.
- The bot could see its remaining budget as an input, with bins masked when it runs out, so it never sends a command
  the server would drop.
- With three bins the bot could ask for up to 6 commands per second: the budget binds in busy moments, and commands
  become a resource to spend well. Salvos and bulk upgrades save commands (15 nukes = 1 command).

**C. Salvos and bulk upgrades**
- 3B, as planned today: one missile per nuke, default arc, upgrades by one level. Nothing in PLAN brings the amount and
  arc back for FFA (Phase 6).
- Idea: restore the amount (nukes, upgrades) and the arc in the action set.
- Rule set, user decision: training, self-play and private lobbies follow the server's rules (1–50). In official casual
  lobbies the owner allowed tests with "no abuse": a deployment switch can hold the bot to what the client offers.
  The client already allows ×max atom bombs and ×max upgrades, so only hydrogen salvos and amounts between the client's
  steps differ.

**D. SAM coverage counts interceptors** (3B observations)
- The coverage channel could hold the ready interceptor slots covering each cell (sum of the covering SAMs' levels, minus
  missiles reloading), not a yes/no. A level-1 and a level-14 SAM must look different to learn "send one more than
  there are slots".

**E. Why the exact nuke predictor could stay in Phase 5**
- It is an input (interceptor slots on the path), never a mask: the bot must be free to attack defended cells with
  enough nukes, and the defender can react (an instant upgrade adds a slot).
- Reasons to keep it in Phase 5: SAMs are rare in ranked 1v1 (~0.9 per game for both players, against ~4.7 nukes);
  it costs 1–2 weeks with a 100%-match gate; BC on human games already teaches salvos at defended spots; D covers the
  main gap meanwhile.

**F. Activations and real MoE experts** (user decision 2026-10-05: add it)
- Today (`trainer/model.py`): ReLU everywhere. That covers the CNN (`Res`, line 112; between stages; the heat-map
  decoder), the scalar layer, the trunk, after the core (line 185), and the player transformer's feed-forward part
  (PyTorch's `TransformerEncoderLayer` defaults to ReLU, line 137).
- **The MoE experts are single linear layers** (`SoftMoE`, line 92: `nn.Linear(d, d)` per expert). The core blends 8
  linear maps by the router's weights, and one ReLU follows the blend. Standard MoE experts are 2-layer networks with
  an activation between the layers (today usually SwiGLU). The design was deliberate (the docstring: "each expert is one
  d x d layer"), and it matched the dense arm (one 512 x 512 layer), so the A/B was fair. But it gives the experts
  almost no capacity of their own: a second likely reason the MoE tied dense, next to the single fused vector (PLAN 3A
  reading). The router also left 4 of 8 experts nearly idle (2–3% weight each).
  - Idea: each expert could become a 2-layer SwiGLU network, 512 → 1024 → 512 (~1.6M multiply-adds and ~1.6M parameters
    per expert). For 8 experts that is ~13M multiply-adds, ~0.6% of the CNN's ~2 billion per decision, and ~12M more
    parameters: the cheap brain growth decision 6 asks for ("~1.5M to ~10–30M params almost free").
- **Dead neurons.** ReLU's gradient is zero for negative inputs. A unit whose input is negative for every observation
  stops learning and never recovers. Deep-RL studies report such units piling up in long training on shifting data:
  "dormant neurons" (Sokar et al., ICML 2023), "loss of plasticity" (Abbas et al. 2023, Dohare et al., Nature 2024);
  check them before relying on them. A self-play league (Phase 4) is shifting data by design. Leaky ReLU, GELU and SiLU
  never have a zero gradient.
- **Cost on this PC.**
  - ReLU, leaky ReLU, GELU and SiLU cost about the same here: the work is one read and one write of the whole tensor,
    and without `torch.compile` on Windows (no Triton) it cannot be fused into the convolutions (Phase 0).
  - SwiGLU is a gated layer with an extra projection: nearly free in the brain, costly in the CNN.
- **One possible split:**
  | Part | Now | Idea |
  |---|---|---|
  | CNN (~95% of compute) | ReLU | Keep (the IMPALA-style standard in RL); revisit if dead units show up |
  | Transformer feed-forward | ReLU | GELU (`activation="gelu"`) or SwiGLU |
  | MoE experts | one linear layer | 2-layer SwiGLU networks |
  | PPO and BC monitoring | none | Fraction of dormant units per layer (a unit's mean \|activation\| ≤ τ × its layer's mean, as in Sokar et al.); if it climbs, leaky ReLU in the CNN or resetting dormant units |
- One possible timing: a new network needs a new BC night anyway, so it could join 3B's BC refresh, as an A/B like dense vs MoE, so its
  effect is not mixed up with the new actions. A cheaper first look: two short from-scratch runs, scored on the fixed
  set.
- Old checkpoints keep working: the network kind is stored in the checkpoint (`args["arch"]`, `"core"`), so the new
  experts are a new value there.

**G. The MoE in mixed modes** (ideas from the 2026-10-05 discussion)
- The user's view: OpenFront has many situations (early, mid and late game, rich starts, 1v1, 2v2, FFA, Team, custom
  maps), and one dense network may suffer "internal conflict" between them. The textbook name is gradient interference
  (negative transfer): situations that pull a shared weight in opposite directions. Reducing it is the benefit an MoE
  could bring here.
- What our MoE does not bring: compute savings. Sparse MoEs (language models) run only the top 1–2 experts. Our
  Soft-MoE gets one vector per decision and runs all 8. It is cheap only because the core is tiny next to the CNN.
- What a dense network already does: each ReLU unit is on or off depending on the input, so a ReLU network is itself a
  large implicit mixture of linear pieces, and the situation is an input (decision 5a). Large dense networks handle wide
  task mixes (OpenAI Five: one network for every Dota 2 hero). Interference bites mostly when capacity is small, and
  our brain is small (3.8M dense, 5.7M MoE), which leans the bet towards the MoE. Neither side is proven for OpenFront.
- The idle experts (4 of 8 at 2–3% weight in 3A) read two ways:
  - spare capacity: 1v1 needed about 4, so 4 are free for new modes;
  - a risk of expert collapse: an expert with 2% of the weight gets about 2% of the learning signal, stays untrained,
    and the router keeps avoiding it. New modes may then crowd into the trained experts.
  - Possible remedies: upcycling (copy a trained expert into an idle slot; PLAN mentions it), a load-balancing term in
    the loss, router noise. A hint of specialisation already exists: in 3A the late game leaned on expert 6, the spawn on
    expert 5.
- A possible fair comparison (open question 11): 8 SwiGLU experts of 512 → 1024 → 512 against one dense SwiGLU network
  of 512 → 8192 → 512. Both have ~12.6M parameters and the same compute, so the only difference is the routing. If the
  MoE wins, routing and specialisation helped.
- Where it could be tested: ranked 1v1 alone has one mode, the hardest place for an MoE to show a gain. Mixed modes
  are the natural test: 3C (1v1 + 2v2 with the mode as an input) and Phase 6 (FFA, Team).
- The dense night of 3A was the control group: it showed that, in 1v1 with single-layer experts, the routing added
  nothing measurable. That limits the 3A conclusion; it does not rule out an MoE with real experts in mixed modes.

## Considered and dropped
- Measuring late-game FFA command bursts to decide whether one action per decision is enough. The user dropped it: the
  bins are decided for strength, and human co-occurrence cannot show what a bot with more bandwidth could do.

## Open questions for the planning session
Points the ideas above leave undecided. None is answered here; they are listed so the planning session does not miss
them.
1. Donations: troop donations in the troops bin and gold donations in the gold bin, or both in diplomacy?
2. Spawn phase: which bin holds spawn, and are the others masked until the spawn ends?
3. Warship moves (spend nothing, Phase 6): gold bin or their own?
4. Do the bins choose independently, or does the gold bin see the troops bin's choice (autoregressive)? The cost of
   conditioning, and the cases where it matters (e.g. a nuke and an attack on the same player).
5. Budget: what the bot sees (tokens left in each bucket), what happens when two bins want the last token, and whether
   the 150-per-minute bucket starts full.
6. BC labels: a window with a human attack and a build labels both bins; extra commands of the same bin still queue
   into later windows (decision 11's rule, per bin).
7. Order inside 3B: the troop shares first, measured alone (one BC night against the 3A clone), as PLAN says; then
   the bins with the new actions, C and D, and one BC refresh.
8. Gate 3B: add a check that the bins are used and the budget holds (no dropped commands), plus the existing criteria.
9. Old checkpoints: a wrapper so 5-action and 3A bots keep playing (one bin, no amounts).
10. Speed: three heads plus the bigger action set against the 75%-of-Phase-2 decisions/s rule.
11. The fair baseline for F: against what would the new MoE core compete? A dense SwiGLU network with one expert's
    size (same parameters per expert), or with the MoE's total compute (all 8 experts run on every decision)? The
    answer decides what "the MoE wins" means. G suggests one answer.
12. The night budget: the troop-share night, then the new actions' BC refresh, now possibly ×2 for F's A/B, at ~14 h
    each. Which comparisons can share a night, and which must stay alone ("change one thing at a time")?
13. Idle experts: with richer experts, does the router still leave half of them unused, and does it matter before the
    mixed modes (3C, Phase 6)? G lists possible remedies.
