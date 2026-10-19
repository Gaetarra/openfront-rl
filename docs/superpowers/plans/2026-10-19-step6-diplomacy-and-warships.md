# Step 6: Diplomacy and Warship Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the bot five new actions: request an alliance (which also accepts one), renew one, break one, donate
troops, and move warships. Add the inputs it needs to use them, the behaviour-cloning labels, exact masks, and an arch
v5 network that works in every code path that exists today.

**Architecture:**
- **Bridge:** it computes exact per-player masks from the game's own checks and decodes the new types into intents. The
  worker sends one more mask blob, six more player features and richer unit records.
- **Trainer:** it appends the new channels and features after v4's, so older networks keep reading only their own
  inputs.
- **Arch v5:** it adds four player pointers, a third kind of the share head (donations), and an autoregressive
  warship-move head. That head is a source heat map; a destination map conditioned (FiLM) on the chosen source cell;
  and an alone-or-fleet choice.
- **Action rows** stay four integers: a warship move is `[19, source cell, destination cell, fleet]`.

**Tech Stack:**
- TypeScript in `bridge/`: node:test, run with tsx.
- Python and PyTorch in `trainer/`: unittest. The trainer tests start real Node workers and need the GPU.
- The pinned game v0.35.2, a read-only submodule.

**Spec:** `docs/superpowers/specs/2026-10-19-diplomacy-and-warship-actions-design.md` (approved 2026-10-19). Read it
first: it holds the game rules every task relies on.

## Global Constraints

- Never edit anything inside `OpenFront_Source_Code/openfrontio/`.
- Rule values are read from the game at runtime (its `Config` or its own checks): cooldowns, durations, ranges, troop
  caps and the default donation. Never write them into our code or tests.
- v4 and older checkpoints play exactly as before: they read only the types, masks and inputs they have.
- New spatial channels go after v4's 57, and new player features after v4's 25.
- Masks never allow what the game's own checks refuse.
- Speed budget: v5 keeps at least 95% of v4's PPO decisions/s, without spilling the card at `--minibatches 12`.
- Work on branch `phase3`, commit after each task, and never push. Commit messages carry no `Co-Authored-By` or other
  Claude attribution line.
- Leave the user's untracked files alone: `0.34.24 to 0.35.3 changelog.txt` and `scripts/*dashboard*/`.
- Commands:
  - Bridge tests, from `OpenFront_Source_Code/openfrontio/`: `node --import tsx --test "../../bridge/*.test.ts"` (one
    file: `node --import tsx --test ../../bridge/actions.test.ts`).
  - Typecheck, from the same folder: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`.
  - Trainer tests, from the repo root: `.venv/Scripts/python -m unittest discover -s trainer -t .` (one file:
    `.venv/Scripts/python -m unittest trainer.test_model`).

## Review Focus

1. **A seat outside the game:** a dead or unspawned seat, or any seat during the spawn phase, gets all-zero new masks,
   and every new action from it drops (Task 4).
2. **Players beyond the attention cap:** a donation target outside the 64 attended players can still be chosen,
   because the pointers read every token (Task 8).
3. **Warship destinations:** a move whose destination cell holds no water of the anchor's water body drops. It never
   sends a land tile, or a tile in another sea (Task 5).
4. **Vanished targets in records:** a human command whose recipient or warship no longer exists labels `UNMAPPED`,
   never a wrong player (Task 6).
5. **The move head stays finite:** its logits are finite with and without autocast, and on rows where the seat has no
   warship (Task 8).

---

### Task 1: Older networks read only their own inputs

**Files:**
- Modify: `trainer/model.py` (`Policy.__init__`, `Policy.forward`, constants)
- Modify: `trainer/env.py` (`VecEnv._observe`: the order of the player features)
- Test: `trainer/test_model.py`

**Interfaces:**
- **Produces, in `trainer/model.py`:**
  - `V4_CHANNELS = 57` and `V4_FEATURES = 25`.
  - v2 and v3 build on `OLD_CHANNELS` channels; v4 builds on `V4_CHANNELS`. Both use `V4_FEATURES` player features,
    whatever `dims` says.
  - `forward` reads `obs["spatial"][:, :c_in]` (it already does) and `obs["players"][..., :self.n_feat]`.
- **Produces, in `trainer/env.py`:**
  - `WORKER_V4_FEATURES = 17`.
  - `obs["players"]` is [the worker's features 0–16 | the buildings by type (8) | the worker's features from 17 on].

- [ ] **Step 1: Write the failing test** in `trainer/test_model.py`. Let `nets()` take `dims=DIMS`, then add:

```python
def test_older_networks_read_only_their_own_inputs(self):
    """v5 appends channels and player features: every older network loads into a model built for the wider inputs
    and gives the same outputs on them."""
    wide = {**DIMS, "channels": DIMS["channels"] + 7, "playerFeatures": DIMS["playerFeatures"] + 6}
    obs = fake_obs()
    wider = dict(obs, spatial=torch.cat([obs["spatial"], torch.full_like(obs["spatial"][:, :7], 40)], 1),
                 players=torch.cat([obs["players"], torch.rand(N, P, 6, device="cuda")], 2))
    for (name, net), twin in zip(nets().items(), nets(wide).values()):
        net = randomized(net).cuda().eval()
        twin.load_state_dict(net.state_dict())
        twin = twin.cuda().eval()
        with torch.no_grad():
            a, b = net(obs), twin(wider)
        for k, v in a.items():
            self.assertTrue(torch.allclose(v, b[k], atol=1e-5), (name, k))
```

- [ ] **Step 2: Run it:** `.venv/Scripts/python -m unittest trainer.test_model -k older_networks`. Expected: FAIL,
  because a state_dict size mismatch shows in the embed or stem layer.
- [ ] **Step 3: Implement** the input sizes in `Policy.__init__` and slice the players in `forward`. In `_observe`,
  build `players` as `torch.cat([pl[..., :WORKER_V4_FEATURES], torch.log1p(buildings[row_env]) / 3,
  pl[..., WORKER_V4_FEATURES:]], 2)`. That is a no-op while the worker sends 17.
- [ ] **Step 4: Run** `trainer.test_model`, then the whole trainer suite. Expected: PASS.
- [ ] **Step 5: Commit** `trainer/model.py trainer/env.py trainer/test_model.py`.

### Task 2: Warship and trade-ship unit records

**Files:**
- Modify: `bridge/frame.ts` (`UNIT_BYTES`, the record doc comment, `encodeUnits`)
- Modify: `trainer/env.py` (the `UNIT` dtype)
- Test: `bridge/frame.test.ts`

**Interfaces:**
- **Produces:** `UNIT_BYTES = 22`, laid out as follows:
  - 0: u8 type;
  - 1: u8 level;
  - 2: u16 owner;
  - 4: u32 cell;
  - 8: f32 troops (warships: health ÷ max health);
  - 12: u8 ready (silos and SAMs: missiles loaded; warships: veteran level);
  - 13: u8 flags (bit 0: a trade ship safe from pirates);
  - 14–19: u16 zx, zy, zr (the zone; warships: zx and zy hold the patrol tile, and zr is 0);
  - 20: u16 bound (trade ships: the smallID of their destination port's owner; else 0).
- **Produces:** `trainer/env.py` `UNIT` gets `("flags", "u1")` in place of `("pad", "u1")` and a trailing
  `("bound", "<u2")`, and its itemsize check becomes `assert UNIT.itemsize == 22`.

- [ ] **Step 1: Write the failing test.**
  - First extend `records()` in `bridge/frame.test.ts` to also return `troops` (f32 at 8), `flags` (byte 13) and
    `bound` (u16 at 20).
  - Then add:

```ts
test("warship records carry health, veteran level and patrol tile; trade ships their destination's owner", async () => {
  const env = await EnvGame.create(ONION, "ships");
  script(env, 260);
  const g = env.game;
  const seat = env.seats[0];
  const water: TileRef[] = [];
  g.forEachTile((t) => { if (g.isOcean(t) && water.length < 3) water.push(t); });
  const ship = seat.buildUnit(UnitType.Warship, water[0], { patrolTile: water[1] });
  ship.modifyHealth(-250); // 750 of 1,000
  ship.recordKill(UnitType.Warship); // veteran level 1: the cap rises to 1,200, the health stays 750
  const enemy = g.players().find((p) => p !== seat && p.isAlive() && p.numTilesOwned() > 0)!;
  const port = enemy.buildUnit(UnitType.Port, [...enemy.tiles()][0], {});
  seat.buildUnit(UnitType.TradeShip, water[2], { targetUnit: port }).setSafeFromPirates();
  const of = (type: UnitType) => records(env, seat.smallID()).filter((r) => r.type === type);
  assert.deepEqual(of(UnitType.Warship).map((r) => [r.troops, r.ready, r.zone, r.bound]),
    [[750 / 1200, 1, [g.x(water[1]), g.y(water[1]), 0], 0]]);
  assert.deepEqual(of(UnitType.TradeShip).map((r) => [r.flags, r.bound]), [[1, enemy.smallID()]]);
});
```

- [ ] **Step 2: Run** `node --import tsx --test ../../bridge/frame.test.ts`. Expected: FAIL (troops 0, no bound).
- [ ] **Step 3: Implement** in `encodeUnits`, after the zone:
  - warships: `Number(u.health()) / u.maxHealth()` at 8, `u.veterancy()` at 12, and the patrol tile's x and y at 14
    and 16;
  - trade ships: `isSafeFromPirates()` as bit 0 of byte 13, and `targetUnit()?.owner().smallID()` at 20.
  - Update the doc comment, then the `UNIT` dtype.
- [ ] **Step 4: Run** the bridge tests, the typecheck and `.venv/Scripts/python -m unittest trainer.test_env`.
  Expected: PASS.
- [ ] **Step 5: Commit** `bridge/frame.ts bridge/frame.test.ts trainer/env.py`.

### Task 3: Diplomacy player features

**Files:**
- Modify: `bridge/obs.ts` (`PLAYER_FEATURES`, the doc comment, `encodeSeat`)
- Modify: `bridge/testkit.ts`
- Modify: `bridge/game.test.ts` (it imports the moved helpers; the players blob size uses `PLAYER_FEATURES`)
- Test: `bridge/obs.test.ts`

**Interfaces:**
- **Produces:** `PLAYER_FEATURES = 23`. Features 17–22 of row p, from the seat's view:
  - 17 ally (`seat.isAlliedWith(p)`, never a teammate);
  - 18 p's alliance request to the seat is pending;
  - 19 the seat's request to p is pending;
  - 20 the alliance's time left, `(expiresAt − ticks) / config.allianceDuration()`, clamped to 0–1, and 0 without an
    alliance;
  - 21 p agreed to renew (`alliance.agreedToExtend(p)`);
  - 22 the seat agreed to renew.
- **Produces, in `bridge/testkit.ts`** (moved here from `game.test.ts`):
  - `publicGame(cfg, gameMap, gameMapSize, patch?)` and `quickGame(cfg, timerMinutes = 1, patch = {})`;
  - `LONE: Partial<GameConfig> = { nations: "disabled", bots: 0 }`;
  - `spawnSeats(env: EnvGame, which: number[] = all seats): void`.
    - It spawns seat i (for i in `which`) on the i-th of `env.seats.length` land tiles spread evenly by x, at tick 20
      with `tickWith`, then ticks past the spawn phase (`numSpawnPhaseTurns() + 2`).
    - In team games whose map has spawn areas, the tiles come from the seat's team area.
    - It throws if a seat in `which` has not spawned.

- [ ] **Step 1: Write the failing test** in `bridge/obs.test.ts`:

```ts
test("diplomacy features: ally, pending requests, the alliance's time left and who asked to renew", async () => {
  const env = await quickGame(FFA, 5, { ...LONE, customAllianceDuration: 1 });
  spawnSeats(env);
  const g = env.game;
  const [a, b] = env.seats;
  const send = (i: number, intent: Intent) => { env.tickWith([{ ...intent, clientID: seatClientID(i) } as StampedIntent]); env.tickWith([]); };
  const view = () => {
    const players = new Float32Array(env.slots * PLAYER_FEATURES);
    encodeSeat(g, a, env.horizonMinutes, env.frame.cs, attackMask(g, a, env.slots), players, new Uint8Array(env.slots),
      new Float32Array(SCALARS));
    return [17, 18, 19, 20, 21, 22].map((f) => players[b.smallID() * PLAYER_FEATURES + f]);
  };
  send(1, { type: "allianceRequest", recipient: a.id() });
  assert.deepEqual(view(), [0, 1, 0, 0, 0, 0], "b asked a");
  send(0, { type: "allianceRequest", recipient: b.id() });
  const fresh = view();
  assert.deepEqual([fresh[0], fresh[1], fresh[2]], [1, 0, 0], "allied, nothing pending");
  assert.ok(fresh[3] > 0.99, "a new alliance");
  const alliance = a.allianceWith(b)!;
  while (g.ticks() < alliance.expiresAt() - g.config().allianceExtensionPromptOffset()) env.tickWith([]);
  send(1, { type: "allianceExtension", recipient: a.id() });
  const late = view();
  assert.deepEqual([late[4], late[5]], [1, 0], "b asked to renew, a not yet");
  assert.ok(late[3] < 0.5);
});
```

- [ ] **Step 2: Run** `node --import tsx --test ../../bridge/obs.test.ts`. Expected: FAIL, because the helpers
  don't exist yet and `PLAYER_FEATURES` is 17.
- [ ] **Step 3: Implement the helpers.** Move `publicGame` and `quickGame` into `testkit.ts` and add `LONE` and
  `spawnSeats`. In `game.test.ts`, import them and change `env.slots * 17 * 4` to `env.slots * PLAYER_FEATURES * 4`.
- [ ] **Step 4: Implement the features** in `encodeSeat`. Collect the seat's incoming requestors and outgoing
  recipients once, as sets, before the player loop.
- [ ] **Step 5: Run** the bridge tests, the typecheck and the trainer suite. Expected: PASS. Task 1 puts the 6 new
  features after the buildings, so `dims["playerFeatures"]` becomes 31.
- [ ] **Step 6: Commit** `bridge/obs.ts bridge/obs.test.ts bridge/testkit.ts bridge/game.test.ts`.

### Task 4: Diplomacy masks and decoding

**Files:**
- Modify: `bridge/actions.ts` (the header comment, `Act`, `N_ACTS`, the new mask functions, `typeMask`,
  `decodeAction`)
- Modify: `bridge/game.ts` (`currentMasks` → `seatMasks`, the `step` decode call, the "diplomacy" blob)
- Create: `bridge/diplomacy.test.ts`
- Modify: `bridge/actions.test.ts` (its helpers and direct calls use `seatMasks` and the new `decodeAction`)
- Modify: `bridge/game.test.ts` and `bridge/worker.test.ts` (the blob lists)

**Interfaces:**
- **Produces, in `bridge/actions.ts`:**
  - The new types: `Act.AllianceRequest = 15`, `Act.RenewAlliance = 16`, `Act.BreakAlliance = 17`,
    `Act.DonateTroops = 18`, `Act.MoveWarships = 19`, and `N_ACTS = 20`.
  - `DIPLOMACY: readonly Act[] = [Act.AllianceRequest, Act.RenewAlliance, Act.BreakAlliance, Act.DonateTroops]`.
  - `interface Masks { types: Uint8Array; attack: Uint8Array; retreat: Uint8Array; diplomacy: Uint8Array }`.
    `diplomacy` holds `DIPLOMACY.length × slots` bytes; row k is the targets of `DIPLOMACY[k]`.
  - `diplomacyMask(game: Game, seat: Player, slots: number): Uint8Array`. It is all zero during the spawn phase or when
    the seat is dead. Otherwise, for each player p ≠ seat with smallID s < slots:
    - request: `p.isAlive() && seat.canSendAllianceRequest(p)`;
    - renew: `seat.allianceInfo(p)?.canExtend === true`;
    - break: `seat.isAlliedWith(p)`;
    - donate: `seat.canDonateTroops(p) && game.config().maxTroops(p) - p.troops() >= 1 && seat.troops() >= 1`.
  - `seatMasks(game: Game, seat: Player, slots: number): Masks`.
  - `typeMask(game, seat, attack, retreat = new Uint8Array(0), diplomacy = new Uint8Array(0)): Uint8Array`.
    - `m[DIPLOMACY[k]]` is set when any target of row k is.
    - `m[Act.MoveWarships]` is set when the seat has an active warship.
    - Both stay 0 in the spawn phase and for a dead seat (the function already returns early there).
  - `decodeAction(game: Game, seat: Player, frame: CellFrame, masks: Masks, action: number[]): Decoded`.
    - Diplomacy actions are `[type, 0, row, share]`, and the target's bit in its row must be set.
    - The intents: `{ type: "allianceRequest", recipient }`, `{ type: "allianceExtension", recipient }`,
      `{ type: "breakAlliance", recipient }`, and `{ type: "donate_troops", recipient, troops: share / 100 ×
      seat.troops() }`. The share must be a whole 1–100.
    - `MoveWarships` drops until Task 5.
- **Produces, in `bridge/game.ts`:** `observe()` returns `[..., "retreat", "diplomacy"]`. The "diplomacy" blob is
  `S × DIPLOMACY.length × slots` bytes.

- [ ] **Step 1: Write the failing tests** in `bridge/diplomacy.test.ts`.
  - Helpers:
    - `row(m, act, p)` reads `m.diplomacy[DIPLOMACY.indexOf(act) * slots + p.smallID()]`.
    - `act(env, picks)` observes, then calls `env.step` with `picks[i]` for seat i and no-ops elsewhere.
  - The tests:

```ts
test("a request is legal toward live non-friends, and a request back forms the alliance", async () => {
  const env = await quickGame(FFA, 5, LONE);
  spawnSeats(env);
  const [a, b] = env.seats;
  const m = seatMasks(env.game, a, env.slots);
  assert.deepEqual([row(m, Act.AllianceRequest, b), row(m, Act.AllianceRequest, a)], [1, 0]);
  assert.deepEqual([m.types[Act.AllianceRequest], m.types[Act.BreakAlliance]], [1, 0]);
  assert.deepEqual(decodeAction(env.game, a, env.frame, m, [Act.AllianceRequest, 0, b.smallID(), 0]).intent,
    { type: "allianceRequest", recipient: b.id() });
  act(env, { 0: [Act.AllianceRequest, 0, b.smallID(), 0] });
  assert.equal(row(seatMasks(env.game, a, env.slots), Act.AllianceRequest, b), 0, "a request is pending");
  act(env, { 1: [Act.AllianceRequest, 0, a.smallID(), 0] });
  assert.ok(a.isAlliedWith(b));
  assert.deepEqual(env.invalid, [false, false, false, false]);
});

test("renewal opens in the alliance's last offer window and renews when both ask", async () => {
  const env = await quickGame(FFA, 5, { ...LONE, customAllianceDuration: 1 });
  spawnSeats(env);
  const [a, b] = env.seats;
  act(env, { 0: [Act.AllianceRequest, 0, b.smallID(), 0], 1: [Act.AllianceRequest, 0, a.smallID(), 0] });
  const alliance = a.allianceWith(b)!;
  assert.equal(row(seatMasks(env.game, a, env.slots), Act.RenewAlliance, b), 0, "too early");
  const expires = alliance.expiresAt();
  while (env.game.ticks() < expires - env.game.config().allianceExtensionPromptOffset()) env.tickWith([]);
  assert.equal(row(seatMasks(env.game, a, env.slots), Act.RenewAlliance, b), 1);
  act(env, { 0: [Act.RenewAlliance, 0, b.smallID(), 0], 1: [Act.RenewAlliance, 0, a.smallID(), 0] });
  assert.ok(a.allianceWith(b)!.expiresAt() > expires);
});

test("breaking marks the seat a traitor; an alliance that runs out marks nobody", async () => {
  // Two alliances in one 1-minute game: a breaks the one with b and lets the one with c expire.
  // Assertions: break legal toward b only; after it a.isTraitor() and !a.isAlliedWith(b);
  // past c's expiry: c is no longer allied, and c.isTraitor() is false.
});

test("donations go to teammates below their cap as a share of the seat's troops; public FFA refuses humans", async () => {
  const hvn = await quickGame(HVN, 5);
  spawnSeats(hvn);
  const [a, b] = hvn.seats;
  const m = seatMasks(hvn.game, a, hvn.slots);
  assert.equal(row(m, Act.DonateTroops, b), 1, "a teammate");
  assert.deepEqual(decodeAction(hvn.game, a, hvn.frame, m, [Act.DonateTroops, 0, b.smallID(), 30]).intent,
    { type: "donate_troops", recipient: b.id(), troops: 0.3 * a.troops() });
  assert.equal(decodeAction(hvn.game, a, hvn.frame, m, [Act.DonateTroops, 0, b.smallID(), 0]).valid, false);
  const [giverBefore, takerBefore] = [a.troops(), b.troops()];
  act(hvn, { 0: [Act.DonateTroops, 0, b.smallID(), 30] });
  assert.ok(a.troops() < 0.8 * giverBefore, "the giver lost about 30%"); // growth over 5 ticks is far smaller
  assert.ok(b.troops() > takerBefore + 0.2 * giverBefore, "the teammate received it");
  const ffa = await quickGame(FFA, 5, LONE);
  spawnSeats(ffa);
  const [c, d] = ffa.seats;
  act(ffa, { 0: [Act.AllianceRequest, 0, d.smallID(), 0], 1: [Act.AllianceRequest, 0, c.smallID(), 0] });
  assert.equal(row(seatMasks(ffa.game, c, ffa.slots), Act.DonateTroops, d), 0, "an allied human, donations off");
});

test("a seat outside the game has no diplomacy and no warship moves", async () => {
  const env = await quickGame(FFA, 5, LONE);
  const zero = (m: Masks) => m.diplomacy.every((v) => v === 0) && DIPLOMACY.every((t) => m.types[t] === 0)
    && m.types[Act.MoveWarships] === 0;
  assert.ok(zero(seatMasks(env.game, env.seats[0], env.slots)), "spawn phase");
  spawnSeats(env, [1, 2, 3]); // seat 0 never spawns
  const m = seatMasks(env.game, env.seats[0], env.slots);
  assert.ok(zero(m), "an unspawned seat after the spawn phase");
  assert.equal(decodeAction(env.game, env.seats[0], env.frame, m,
    [Act.AllianceRequest, 0, env.seats[1].smallID(), 0]).valid, false);
});
```

  - In `bridge/game.test.ts`, add `"diplomacy"` to the expected blob names, and assert that its size is
    `env.slots * 4`.
  - In `bridge/worker.test.ts`, add it to line 68's list.
- [ ] **Step 2: Run** the bridge tests. Expected: FAIL (`seatMasks` is not exported).
- [ ] **Step 3: Implement** the new `Act` members, `DIPLOMACY`, `Masks`, `diplomacyMask`, `seatMasks`, the
  `typeMask` additions and the new `decodeAction` signature with the diplomacy branch.
  - `EnvGame` stores `Masks` per seat and passes them to `decodeAction`. `observe()` writes the diplomacy blob after
    "retreat".
  - Update every call site in `bridge/actions.test.ts` to pass a `Masks`, building it as
    `{ ...seatMasks(env.game, seat, env.slots), types }` where a test needs other types.
  - Rewrite the header comment of `actions.ts`: what is outside the set is now gold donations, rejects, embargoes,
    target marks, chat and emoji, MIRVs, cancelling boats and deleting units.
- [ ] **Step 4: Run** the bridge tests, the typecheck and the trainer suite. Expected: PASS. The trainer ignores the
  new blob until Task 7, and v4 networks read `type_mask[:, :15]`.
- [ ] **Step 5: Commit** `bridge/actions.ts bridge/game.ts bridge/diplomacy.test.ts bridge/actions.test.ts
  bridge/game.test.ts bridge/worker.test.ts`.

### Task 5: Warship moves

**Files:**
- Modify: `bridge/actions.ts` (the `MoveWarships` branch of `decodeAction`)
- Test: `bridge/actions.test.ts`

**Interfaces:**
- **Consumes:** `Masks` and `decodeAction` (Task 4).
- **Produces:** `[Act.MoveWarships, sourceCell, destinationCell, fleet]` decodes to
  `{ type: "move_warship", unitIds, tile }`. The rules:
  - **Anchor:** the seat's active warship whose tile lies in `sourceCell`, nearest the cell's centre (`frame.center`).
  - **unitIds:** `fleet === 1` takes every active warship of the seat whose `warshipState().patrolTile` equals the
    anchor's and that lies in the anchor's water body (`game.hasWaterComponent(u.tile(),
    game.getWaterComponent(anchor.tile())!)`). `fleet === 0` takes `[anchor]`.
  - **tile:** the tile of `destinationCell` nearest its centre (the existing `byCentre`) that is water in the
    anchor's water body.
  - **Drops:** a cell outside 0..frame.n−1, a fleet other than 0 or 1, no anchor, or no such tile.

- [ ] **Step 1: Write the failing tests.**
  - Add the helper `oceanCells(env)`: the cells holding ocean tiles of the water body of the first ocean tile found,
    each with one tile of it.
  - Then add:

```ts
test("a warship move sends the anchor alone or with its fleet to the water nearest the cell's centre", async () => {
  const env = await EnvGame.create(ONION, "moves");
  script(env, 260);
  const seat = env.seats[0];
  const [home, near, far] = oceanCells(env); // [cell, a water tile in it]
  const s1 = seat.buildUnit(UnitType.Warship, home[1], { patrolTile: home[1] });
  const s2 = seat.buildUnit(UnitType.Warship, home[1], { patrolTile: home[1] });
  const lone = seat.buildUnit(UnitType.Warship, near[1], { patrolTile: near[1] });
  assert.equal(seatMasks(env.game, seat, env.slots).types[Act.MoveWarships], 1);
  const fleet = decode(env, seat, [Act.MoveWarships, home[0], far[0], 1]).intent;
  assert.ok(fleet?.type === "move_warship");
  assert.deepEqual(new Set(fleet.unitIds), new Set([s1.id(), s2.id()]));
  assert.equal(env.frame.cellOf(fleet.tile), far[0]);
  assert.ok(env.game.isWater(fleet.tile));
  const alone = decode(env, seat, [Act.MoveWarships, home[0], far[0], 0]).intent;
  assert.ok(alone?.type === "move_warship" && alone.unitIds.length === 1);
  run(env, fleet);
  env.tickWith([]);
  assert.deepEqual([s1, s2, lone].map((s) => s.warshipState().patrolTile), [fleet.tile, fleet.tile, near[1]]);
});

test("a move drops without a warship in the source cell or water of its sea in the destination cell", async () => {
  // Same setup with one warship at home.
  // Each of these is invalid:
  //   [MoveWarships, <a land-only cell of the seat>, far, 0]
  //   [MoveWarships, home, <a land-only cell>, 0]
  //   [MoveWarships, home, far, 2]
  //   [MoveWarships, home, 1e9, 0]
});
```

- [ ] **Step 2: Run** `node --import tsx --test ../../bridge/actions.test.ts`. Expected: FAIL (the move drops).
- [ ] **Step 3: Implement** the branch as specified.
- [ ] **Step 4: Run** the bridge tests and the typecheck. Expected: PASS.
- [ ] **Step 5: Commit** `bridge/actions.ts bridge/actions.test.ts`.

### Task 6: Labels and the Team fixture

**Files:**
- Modify: `bridge/replay.ts` (`labelOf`, the `OUT_OF_SET` comment)
- Create: `bridge/testdata/team.json.gz`
- Test: `bridge/replay.test.ts`

**Interfaces:**
- **Consumes:** `Act`, `DIPLOMACY` and `seatMasks` (Task 4).
- **Produces:** `labelOf` maps the new commands:

  | Command | Label |
  |---|---|
  | `allianceRequest` | `[Act.AllianceRequest, 0, smallID, 0]` |
  | `allianceExtension` | `[Act.RenewAlliance, 0, smallID, 0]` |
  | `breakAlliance` | `[Act.BreakAlliance, 0, smallID, 0]` |
  | `donate_troops` | `[Act.DonateTroops, 0, smallID, shareOf(troops ?? game.config().defaultDonationAmount(seat), seat.troops())]` |
  | `move_warship` | `[Act.MoveWarships, cellOf(first.tile()), cellOf(tile), unitIds.length > 1 ? 1 : 0]`, where `first = game.unit(unitIds[0])` |

  - A recipient the game doesn't have, or a `first` that is missing or is not the seat's warship, gives
    `[UNMAPPED, 0, 0, 0]`.
  - Every other command stays `OUT_OF_SET`.
- **Produces, the fixture:** `bridge/testdata/team.json.gz` is record `cy6fkbw55W` from `data/v0.35.0/records/`.
  - It is public Team, 6 teams, Between Two Seas, Compact, 18 players, 3,921 turns, v0.35.2, with donations on.
  - It holds 20 alliance requests, 1 renewal, 4 breaks, 21 troop donations and 1 warship move.
  - Usernames, `publicID` and `persistentID` are blanked.

- [ ] **Step 1: Make the fixture:**

```bash
.venv/Scripts/python -c "import gzip, json; from trainer import crawl; r = json.loads(gzip.decompress(open('data/v0.35.0/records/cy6fkbw55W.json.gz', 'rb').read())); crawl.blank(r, public_ids=True); [p.update(persistentID=None) for p in r['info']['players']]; open('bridge/testdata/team.json.gz', 'wb').write(gzip.compress(json.dumps(r).encode()))"
```

- [ ] **Step 2: Write the failing tests.**
  - Rename the last test of `bridge/replay.test.ts` to "every command maps to an action, except gold donations,
    rejects, embargoes, chat and MIRVs".
  - Replace its alliance assertion. With `other = env.game.players().find((p) => p !== seat)!` and two warships
    `w1, w2` built on ocean tiles, the expected labels are:
    - `allianceRequest` to other: `[Act.AllianceRequest, 0, other.smallID(), 0]`;
    - the same to `"nobody"`: `[UNMAPPED, 0, 0, 0]`;
    - `allianceExtension` and `breakAlliance`: their acts with `other.smallID()`;
    - `donate_troops` with `troops: seat.troops() / 4`: share 25;
    - `donate_troops` with `troops: null`: `shareOf(env.game.config().defaultDonationAmount(seat), seat.troops())`;
    - `move_warship` with `[w1.id(), w2.id()]`: `[Act.MoveWarships, cellOf(w1.tile()), cellOf(tile), 1]`;
    - `move_warship` with `[w1.id()]`: fleet 0;
    - `move_warship` with `[999_999]`: `UNMAPPED`;
    - `donate_gold`, `allianceReject` and `embargo`: `OUT_OF_SET`.
  - Then add:

```ts
const TEAM = path.join(path.dirname(fileURLToPath(import.meta.url)), "testdata/team.json.gz");

test("a blanked Team record replays in sync, and its diplomacy and warship labels are legal", async () => {
  const rg = await RecordGame.create(readRecord(TEAM), 256, 160);
  const seen = new Map<number, number>();
  const legal = new Map<number, number>();
  while (!rg.done) {
    rg.env.observe(rg.window === 0);
    const masks = rg.env.seats.map((s) => seatMasks(rg.env.game, s, rg.env.slots));
    rg.labels().forEach(([t, cell, target, troops], i) => {
      if (t < Act.AllianceRequest) return;
      seen.set(t, (seen.get(t) ?? 0) + 1);
      const k = DIPLOMACY.indexOf(t);
      const ok = k >= 0
        ? masks[i].diplomacy[k * rg.env.slots + target] === 1
        : rg.env.seats[i].units(UnitType.Warship).some((u) => rg.env.frame.cellOf(u.tile()) === cell)
          && rg.env.frame.tilesOf(target).some((tile) => rg.env.game.isWater(tile)) && troops <= 1;
      if (ok) legal.set(t, (legal.get(t) ?? 0) + 1);
    });
    rg.advance();
  }
  assert.equal(rg.desync, false);
  // Pin the exact counts the first passing run prints (the fixture is fixed): every new type appears, and each
  // type's legal labels are at least 3/4 of its labels.
});
```

- [ ] **Step 3: Run** `node --import tsx --test ../../bridge/replay.test.ts`. Expected: FAIL (the labels are
  `OUT_OF_SET`).
- [ ] **Step 4: Implement** the `labelOf` cases. Then print `seen` and `legal` once, and replace the comment above
  with exact `assert.deepEqual` checks on both maps.
- [ ] **Step 5: Run** the bridge tests and the typecheck. Expected: PASS.
- [ ] **Step 6: Commit** `bridge/replay.ts bridge/replay.test.ts bridge/testdata/team.json.gz`.

### Task 7: Trainer observation

**Files:**
- Modify: `trainer/env.py` (the channel constants, `MAP_TYPES`, `DST`, `_absorb`, `_observe`, `_add_units`, and
  `__init__` reading the trade-ship type index)
- Modify: `trainer/model.py` (constants only)
- Test: `trainer/test_env.py`

**Interfaces:**
- **Consumes:** the "diplomacy" blob (Task 4), the 23 worker features (Task 3), and the records' `flags` and `bound`
  and the warships' fields (Task 2).
- **Produces, in `trainer/env.py`:**
  - **Channels:**
    - `HEALTH = 57`: 57 ours, 58 theirs. The summed warship health ÷ max health, ÷ 4.
    - `VETERAN = 59`: 59 ours, 60 theirs. The summed veteran levels, ÷ 12.
    - `PATROL = 61`: the seat's warships whose patrol tile lies in the cell, ÷ 16. The cell is
      `(zy // side) * W + zx // side`, with `side = sqrt(area)`.
    - `TRADE = 62`. Channel 62 holds the trade ships that pay the seat or its friends (the owner's or the bound
      port's relation is Self or Friendly), ÷ 16. Channel 63 holds the ones the seat could capture: an Enemy owner, a
      bound port that is neither the seat's nor a friend's, and `flags & 1` clear, ÷ 16.
    - `N_CHANNELS = 64`.
    - "Ours" means Self or Friendly, "theirs" Enemy or Bot, as for `ZONES`.
  - **Heat maps:** `MAP_TYPES = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19)`. The 13th heat map is the warship
    source.
  - **Cell masks:** `DST = len(MAP_TYPES)`. `obs["cell_mask"]` is [N, 14, H·W]: the 13 maps' masks, then `DST`, the
    destination.
    - The source mask is `x[:, BUILDINGS + 4 * BUILDING_TYPES.index("Warship")] > 0` (the seat's own warships).
    - The destination mask is `x[:, OCEAN] > 0`.
    - Both are `& valid`.
  - **Type mask:** `type_mask[:, list(MAP_TYPES)] &= cell_mask[:, :len(MAP_TYPES)].any(-1)`, then
    `type_mask[:, 19] &= cell_mask[:, DST].any(-1)`.
  - **Diplomacy masks:** `obs["diplomacy_mask"]` is a bool [N, 4, P].
- **Produces, in `trainer/model.py`:**
  - `ACT_NAMES` gains `"alliance_request", "renew_alliance", "break_alliance", "donate_troops", "move_warships"`.
  - `REQUEST, RENEW, BREAK, DONATE, MOVE = 15, 16, 17, 18, 19` and `DIPLOMACY = (REQUEST, RENEW, BREAK, DONATE)`.
  - `MAP_NAMES` gains `"warship_src"`.

- [ ] **Step 1: Write the failing tests** in `trainer/test_env.py`:

```python
def test_warship_and_trade_channels_follow_their_definitions(self):
    env = VecEnv.__new__(VecEnv)  # one game of 2 seats on a 4 x 4 grid, as in test_units_paint_every_seat...
    env.device = torch.device("cpu")
    env.E, env.N, env.P, env.H, env.W = 1, 2, 4, 4, 4
    env.seats, env.first = np.array([2]), np.array([0])
    env.building = torch.tensor([BUILDING_TYPES.index("Warship"), -1])
    env.zone_kind, env.transport, env.trade = torch.tensor([-1, -1]), -1, 1
    env.area = torch.full((1,), 64.0)
    rel = torch.tensor([[0, 1, 3, 3], [0, 3, 1, 3]])  # row 0 is player 1, row 1 is player 2; player 3 an enemy of both
    units = np.zeros(3, UNIT)
    units["type"], units["owner"], units["cell"] = (0, 1, 1), (1, 2, 2), (5, 6, 7)
    units["troops"][0], units["ready"][0], units["zx"][0], units["zy"][0] = 0.5, 2, 11, 19  # patrol tile in cell 9
    units["bound"][1:] = 3, 1  # one ship bound for player 3's port, one for player 1's
    x = torch.zeros(2, N_CHANNELS, 16)
    env._add_units(x, [(0, units)], rel, torch.ones(2, UPGRADABLE, dtype=torch.bool))
    near = lambda got, want: self.assertAlmostEqual(got.item(), want, places=6)
    near(x[0, HEALTH, 5], 0.5 / 4)
    near(x[0, VETERAN, 5], 2 / 12)
    near(x[0, PATROL, 9], 1 / 16)
    near(x[0, TRADE + 1, 6], 1 / 16)  # an enemy's ship bound for an enemy: capturable
    near(x[0, TRADE, 7], 1 / 16)  # an enemy's ship bound for row 0's port: it pays row 0
    near(x[1, HEALTH + 1, 5], 0.5 / 4)  # the same warship is the enemy's for row 1
    near(x[1, PATROL].sum(), 0)  # and not its patrol
    near(x[1, TRADE, 6] + x[1, TRADE, 7], 2 / 16)  # row 1's own ships
```

- [ ] **Step 2: Add a live-shape test.** `test_the_observation_carries_v5_masks_and_inputs` runs ONION with 1 worker
  and 1 game, and takes 3 random legal steps. It asserts:
  - `spatial` is [1, 64, H, W];
  - `players` is [1, P, 31];
  - `cell_mask` is [1, 14, H·W];
  - `diplomacy_mask` is [1, 4, P] of dtype bool;
  - `type_mask` is [1, 20].
- [ ] **Step 3: Update the channel definitions.** Extend `new_channels` with the definitions of channels 57–63, so
  `test_level_missile_and_zone_channels_follow_their_definitions` checks them on replayed games.
- [ ] **Step 4: Run** `.venv/Scripts/python -m unittest trainer.test_env`. Expected: FAIL.
- [ ] **Step 5: Implement** the env changes and the model constants.
- [ ] **Step 6: Run** the trainer suite. Expected: PASS. v4 networks read `cell_mask[:, :12]`, `type_mask[:, :15]`
  and their own inputs.
- [ ] **Step 7: Commit** `trainer/env.py trainer/model.py trainer/test_env.py`.

### Task 8: The v5 network and its action distribution

**Files:**
- Modify: `trainer/model.py` (`Policy`, `RandomPolicy`, `ActionDist`, `arch_of`, `new_net`, `describe`)
- Modify: `trainer/ppo.py` (`network_args`: new runs are `"v5"`)
- Test: `trainer/test_model.py`, `trainer/test_ppo.py`

**Interfaces:**
- **Consumes:** the Task 7 constants and `obs["diplomacy_mask"]`, `obs["cell_mask"][:, DST]`.
- **Produces, the network:**
  - `Policy(dims, widths, core, shares=False, full=False, v5=False)`. `v5=True` implies shares and full, and reads
    `dims` for its inputs.
  - It has 20 types, 13 heat maps and 3 share kinds: attack, boat and donate. The donate kind is read with the
    recipient's token.
  - The new layers:
    - `diplomacy_query = Linear(512, 4 * D)`;
    - `dst_film = Linear(512 + c1, 2 * c1)`;
    - `dst_shuffle = Conv2d(c1, 16, 1)`;
    - `dst_skip = Conv2d(c_in, 1, 1)`;
    - `fleet_head = Linear(512 + c1, 2)`.
    - All start at zero.
  - v5's outputs add:
    - `"diplomacy"`: [N, 4, P], the pointer logits in `DIPLOMACY` order;
    - `"move"`: a callable `move(src: LongTensor[N]) -> (dst_logits [N, H·W], fleet_logits [N, 2])`. It reads the
      decoder features `g` at the 1/4-resolution block of `src` together with the core `h`, FiLMs `g` with them, and
      decodes one map like `maps`. Because it runs outside the caller's autocast, it re-enters bf16 autocast when `g`
      is bf16.
- **Produces, the action distribution:**
  - `ActionDist(out, obs)` also accepts precomputed `"dst_logits"` and `"fleet_logits"` tensors in place of `"move"`
    (the KL reference, computed at a known source).
  - `ActionDist.src` is a LongTensor [N]: every row's source cell from the last `complete`/`sample`.
  - `ActionDist.entropy(src=None)` and `ActionDist.kl(ref, src=None)` evaluate the destination and fleet terms at
    `src`. By default `src` is the source map's most likely cell.
  - The encodings: `[t, 0, row, 0]` for request, renew and break; `[18, 0, row, share]` for donations;
    `[19, source cell, destination cell, fleet]` for moves.
- **Produces, around the network:**
  - `arch "v5"` gives `Policy(..., shares=True, full=True, v5=True)`, and `describe` says "20 actions".
  - `RandomPolicy` returns zeros for every head the observation has masks for. For v5 observations that means
    `"shares"` with 3 kinds and a zero `"move"`.

- [ ] **Step 1: Write the failing tests.**
  - Extend the helpers in `trainer/test_model.py`:
    - `DIMS_V5 = {"channels": N_CHANNELS, "playerFeatures": 31, "scalars": 26}`;
    - `fake_obs` gains 6 player features, the v5 masks (`type_mask` [N, 20], `cell_mask` [N, 14, H·W],
      `diplomacy_mask`) and `N_CHANNELS` channels;
    - `nets()` gains `"v5": Policy(DIMS_V5, v5=True)`;
    - `v5_obs(seed)`, `v5_out(seed)`: `small_obs` and `share_out` extended to 20 types, 14 masks, diplomacy masks
      [3, 4, 4], `"shares"` [3, 3, 4, SHARES], and a `"move"` whose logits depend on `src` (for example, the source
      cell's index added to one destination's logit).
  - `brute_kl` enumerates the new types: `[t, 0, p, 0]`, `[DONATE, 0, p, s]` and `[MOVE, c, d, f]`.
  - The tests:

```python
def test_v5_joint_probabilities_sum_to_one(self):
    """Over every joint action of v5_obs, including each source x destination x fleet move."""

def test_v5_samples_are_legal_and_the_destination_reads_the_source(self):
    """Every sampled action is legal under its masks. A move's log-probability changes when only its source does."""

def test_v5_kl_is_exact_when_the_source_is_certain(self):
    """With one legal source cell, kl(ref) equals brute_kl over every joint action."""

def test_a_pointer_reaches_players_outside_the_attended_64(self):
    """80 player rows, only row 79 a legal donation and the least relevant: every sampled donation targets it."""

def test_the_move_head_is_finite_with_and_without_autocast_and_without_warships(self):
    """A randomized v5 network: move(src) is finite under bf16 autocast and in fp32, also on rows whose source mask is
    empty."""

def test_v5_starts_uniform_over_legal_actions(self):
    """A fresh v5 network: zero logits on every new head, so its first policy is uniform, as for v4."""

def test_a_v4_network_acts_the_same_on_a_v5_observation(self):
    """A randomized v4 network on fake_obs: its outputs, and the actions ActionDist samples from them under one torch
    seed, are identical whether the observation carries v5's extra masks, channels and features or only v4's."""
```

  - Also, in `trainer/test_ppo.py`, `test_new_runs_are_v5`: `network_args` without `--init` sets `a.arch == "v5"`.
- [ ] **Step 2: Run** `trainer.test_model` and `trainer.test_ppo -k new_runs`. Expected: FAIL.
- [ ] **Step 3: Implement** `Policy` for v5, the `"move"` callable, and `ActionDist`.
  - `ActionDist` gets the diplomacy pointers, the third share kind (`_at` and `_weights`), and the move term in
    `complete`, `log_prob`, `entropy` and `kl`.
  - Then `RandomPolicy`, the arch helpers and the `network_args` default.
- [ ] **Step 4: Run** the trainer suite. Expected: PASS, including Task 1's test with `nets()` now listing v5.
- [ ] **Step 5: Commit** `trainer/model.py trainer/ppo.py trainer/test_model.py trainer/test_ppo.py`.

### Task 9: Random v5 play on every setup, and an evaluation match

**Files:**
- Test: `trainer/test_env.py`, `trainer/test_eval.py`

**Interfaces:**
- **Consumes:** everything above. The test also proves the anti-goal that a diplomatic action the mask allows is never
  dropped.

- [ ] **Step 1: Write the test:**

```python
def test_random_v5_play_drops_no_diplomatic_action(self):
    """The spec's smoke games: a random v5 policy plays 300 decisions on each setup through real workers."""
    for config in (RANKED_1V1, FFA, HVN):
        env = VecEnv(config, workers=1, games_per_worker=1, seed="v5-random")
        try:
            policy, sent, dropped = RandomPolicy(env.dims).cuda(), collections.Counter(), collections.Counter()
            for _ in range(300):
                obs = env.obs
                act = ActionDist(policy(obs), obs).sample()
                invalid = env.step(act.cpu().numpy())[4]
                for t, bad in zip(act[:, 0].tolist(), invalid):
                    sent[t] += 1
                    dropped[t] += int(bad)
            for t in DIPLOMACY:
                self.assertEqual(dropped[t], 0, (config["playlist"], t))
            self.assertGreater(sent[REQUEST], 0, config["playlist"])
            if config is HVN:
                self.assertGreater(sent[DONATE], 0)
            self.assertLessEqual(dropped[MOVE], 0.1 * max(sent[MOVE], 1))
        finally:
            env.close()
```

- [ ] **Step 2: Write `test_a_v5_checkpoint_plays_a_match`** in `trainer/test_eval.py`. It is the existing match
  test with a fresh v5 checkpoint as A against `random`, on ranked 1v1 for 2 episodes. It asserts both episodes finish
  (`trainer.eval` and game recording carry the new actions).
- [ ] **Step 3: Run** `.venv/Scripts/python -m unittest trainer.test_env -k random_v5` and
  `.venv/Scripts/python -m unittest trainer.test_eval -k v5`. Expected: PASS. If a diplomatic type drops, the mask is
  looser than the game's check: fix the mask in `bridge/actions.ts`, never the test.
- [ ] **Step 4: Commit** `trainer/test_env.py trainer/test_eval.py`.

### Task 10: PPO with v5

**Files:**
- Modify: `trainer/ppo.py` (`REF_OUTPUTS`, `stored`, `rollout`, `learn`)
- Test: `trainer/test_ppo.py`

**Interfaces:**
- **Consumes:** `ActionDist.src`, `entropy(src)`, `kl(ref, src)`, and the `"move"` callable (Task 8).
- **Produces:**
  - `REF_OUTPUTS` gains `"diplomacy"`.
  - `stored(o, src)` keeps:
    - `"dst_src"` = src, the rollout's sampled source for every row;
    - with a v5 KL reference: `"ref_dst_logits"` and `"ref_fleet_logits"`, the reference's `move(src)`.
  - In `rollout`, `host.put(step, stored(mine, dist.src))` moves after the sample.
  - In `learn`, `out_r` maps those keys to `"dst_logits"` and `"fleet_logits"`, the entropy is
    `dist.entropy(o["dst_src"])`, and the KL is `dist.kl(ActionDist(out_r, o), o["dst_src"])`.

- [ ] **Step 1: Write the failing test.** `test_a_v5_network_trains_against_itself_with_kl` is
  `test_two_seats_against_a_frozen_opponent_with_kl` with a fresh v5 checkpoint:
  `torch.save({"model": new_net(dims, args).state_dict(), "args": args, "update": 0}, init)`, with
  `args = {"arch": "v5", "core": "moe", "widths": "64,128,192"}` and dims `{"channels": N_CHANNELS,
  "playerFeatures": 31, "scalars": 26}`. It asserts the run finishes 2 updates and logs a finite `loss/kl_ref`.
- [ ] **Step 2: Run** `.venv/Scripts/python -m unittest trainer.test_ppo -k v5`. Expected: FAIL. KL raises on a
  missing move head, or `dst_src` is missing.
- [ ] **Step 3: Implement** as specified.
- [ ] **Step 4: Run** the trainer suite. Expected: PASS (the v4 runs are unchanged).
- [ ] **Step 5: Commit** `trainer/ppo.py trainer/test_ppo.py`.

### Task 11: Behaviour cloning with v5

**Files:**
- Modify: `trainer/bc.py` (`label_legal`, `measure`)
- Test: `trainer/test_bc.py`

**Interfaces:**
- **Consumes:** `DIPLOMACY`, `DONATE`, `MOVE` and `DST`, and the Team fixture (Task 6).
- **Produces:**
  - **`label_legal`:**
    - A diplomacy label needs `obs["diplomacy_mask"][row, k, target]`, where k is its index in `DIPLOMACY`.
    - A donation also needs a share of 1–100.
    - A move needs its source cell via `MAP_OF`, as every map type does, plus `obs["cell_mask"][row, DST, target]` and
      a fleet of 0 or 1.
  - **`measure` adds:**
    - `bc/{name}_target_acc` and `bc/nll_{name}_target` per diplomacy type;
    - `bc/nll_donate_share`;
    - `bc/nll_warship_dst` and `bc/warship_fleet_acc`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_new_labels_need_their_target_or_destination(self):
    """Like test_new_labels_need_their_heat_map_cell_or_an_attack_to_cancel. Synthetic obs with type_mask [n, 20],
    cell_mask [n, 14, 16] and diplomacy_mask [n, 4, 5]. Legal: a request to a masked-in row, a donation with share 40,
    a move from a legal source to a legal destination with fleet 1. Illegal: the same to a masked-out row, a donation
    with share 0, a move to an illegal destination, a move with fleet 2."""

def test_a_v5_network_learns_from_the_team_record(self):
    """Replay bridge/testdata/team.json.gz. config = {"map": "Between Two Seas", "compact": True, "nations": 0,
    "bots": 100, "difficulty": "Medium", "timerMinutes": 0, "seats": 18}. Keep at most 8 windows holding a new-type
    label, stopping at 400 windows. Asserts:
    - a new-type label was seen and is legal;
    - bc_loss with a fresh v5 network is finite;
    - 20 Adam steps on the kept windows lower their loss."""
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m unittest trainer.test_bc -k "new_labels_need_their_target or
  team_record"`. Expected: FAIL.
- [ ] **Step 3: Implement** `label_legal` and the `measure` metrics.
- [ ] **Step 4: Run** the trainer suite. Expected: PASS. The v4 scoring tests are unchanged, and a v4 network counts
  the new labels in `labels/beyond_net`.
- [ ] **Step 5: Commit** `trainer/bc.py trainer/test_bc.py`.

### Task 12: Coverage and the speed check

**Files:**
- Modify: `scripts/coverage.py` (`IN_SET`)
- Modify: `scripts/speed_check.py` (the `--v5` flag)

**Interfaces:**
- **Produces:**
  - `IN_SET` gains `"allianceRequest", "allianceExtension", "breakAlliance", "donate_troops", "move_warship"`.
    `scripts/mode_analysis.py` imports it.
  - `speed_check.py --v5` also times a fresh v5 network (moe-swiglu, as the v4 run) in the same PPO setup. It prints
    v5's decisions/s as a share of v4's and its own peak memory.

- [ ] **Step 1: Implement** both. Then run `.venv/Scripts/python -m unittest trainer.test_env -k coverage`.
  Expected: PASS (the coverage script and the extractor agree).
- [ ] **Step 2: Run** `.venv/Scripts/python scripts/mode_analysis.py play`. Expected: the out-of-set row reads ranked
  2v2 0.82%, Ranked FFA 0.62%, public FFA 1.00%, public Team 1.22% and Humans vs Nations 0.88%.
- [ ] **Step 3: The speed check needs an idle PC and about 15 minutes of GPU,** so ask the user to run it in their
  own terminal:
  `.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12 --skip-bc --skip-v1 --v5`.
  - The budget: v5 at least 95% of v4's decisions/s, and no spill.
  - If v5 misses it, first switch the destination head to the cheaper form: the source's embedding added to the
    shared decoder features, without its own pass. Then fewer or packed channels. Measure again.
  - If it still misses, the user decides.
- [ ] **Step 4: Commit** `scripts/coverage.py scripts/speed_check.py`.

### Task 13: Docs and final verification

**Files:**
- Modify: `docs/PLAN.md`:
  - "After 4N" item 3: what was built, the coverage numbers, the speed check's result;
  - "Action space (v1)": the diplomacy row now names the five actions and what stays out.
- Modify: `CLAUDE.md`:
  - "Actions": 20 types, the diplomacy masks, the move encoding, and what stays outside;
  - "Observations": 64 channels, 31 player features, the warship and trade-ship records;
  - "Networks": v5, and new runs are v5;
  - "Simulation facts": tribes accept every alliance request; donations to nations stay open in public FFA.
- Modify: `docs/superpowers/specs/2026-10-19-diplomacy-and-warship-actions-design.md`: the status line names the
  implementing commits.

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Run** the typecheck, all bridge tests and the whole trainer suite. Expected: PASS everywhere.
- [ ] **Step 3: Commit** `docs/PLAN.md CLAUDE.md docs/superpowers/specs/2026-10-19-diplomacy-and-warship-actions-design.md`.
