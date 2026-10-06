# Phase 3B, Step 2: Troop Shares and SwiGLU Experts in One Night (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attacks and boats send a troop share of 1–100%, read with the target's token (the Phase 3B network, arch
"v3"). BC trains it with soft labels and scores it like for like against the 3A clone. In the same run, a second
network with SwiGLU experts trains on the same windows, ready for one night.

**Architecture:**
- **The troop column becomes a percent (1–100) end to end.**
  - The bridge decodes percents.
  - The extractor labels the human's share as a whole percent.
  - The Phase 2 and 3A networks map their five sizes to 10/25/50/75/100%, which send exactly the troops they sent
    before.
- **`Policy(shares=True)`** (arch "v3") gets a troop-share head.
  - Per kind (attack, boat), a 64-wide hidden layer reads the core state and one player's token, then gives 100
    logits.
  - It runs on every token, so `ActionDist` keeps entropies and KLs exact. Attack shares are weighted by the
    pointer, boat shares by the boat map summed per main owner of each cell.
- **BC.**
  - Soft labels: 0.9 on the human's percent, 0.05 on each neighbour.
  - Scores report `bc/nll` like for like (a 100-share network's shares summed into the five sizes' ranges) and the
    100-way `bc/nll_shares`.
- **The `moe-swiglu` core:** 8 Soft-MoE experts, each a 2-layer SwiGLU network 512 → 1024 → 512 (12.6M parameters).
- **`trainer.bc --twin CORE`** trains a second network on the same windows, row by row, from the same weights outside
  the core. It saves to `checkpoints/<run>-<core>`.

**Tech Stack:** TypeScript bridge (node:test via tsx), Python 3 + torch (unittest), as in Phase 3A.

**Spec:** `docs/PLAN.md`, Phases, "3B — Every human action": "Troop shares read with the target", including "A
second network in the same night". The 3B order puts this second: "the troop shares with a second network, in one
night".

## Global Constraints

- No edits inside `OpenFront_Source_Code/openfrontio/`, and no new dependencies.
- Commits are the repo owner's: never a `Co-Authored-By` or any other Claude line.
- Nothing is pushed: `phase3` stays local until the user says so.
- Ask the user before any run longer than ~10 minutes (the score check, the speed check, the night, the match). The
  PC is shared.
- Files stay LF. Edit with the Edit tool; Python's `Path.write_text` writes CRLF on Windows (fix with
  `sed -i 's/\r$//' <file>`).
- Code whose output reaches the simulation stays deterministic. Percent to troops is plain arithmetic.
- Test commands:
  - trainer: `.venv/Scripts/python -m unittest discover -s trainer -t .` from the repo root (needs the GPU and Node);
  - bridge: `node --import tsx --test "../../bridge/*.test.ts"`, from `OpenFront_Source_Code/openfrontio/`;
  - typecheck: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`, from the same folder.
- Baseline: 89 trainer + 34 bridge tests green at e7e51f8.

## Review Focus

1. **Old bots through the new protocol.** The Phase 2 and 3A checkpoints play every comparison of 3B.
   - A 5-size network must send exactly 10/25/50/75/100% of its troops, the same floats as before.
   - Tests: bridge "the five sizes of earlier networks send what they sent before"; trainer
     `test_five_size_networks_send_their_sizes_as_percents`.
2. **Labels outside 1–100%.**
   - Troops change between a click and its execution, so shares above 100% exist.
   - Trickles round to 0%, and a seat can have 0 troops.
   - The label clamps to 1–100. Test: bridge "a label's troop share is the whole percent the player sent".
3. **Boats to unowned land or to the seat's own cells.**
   - A boat's share reads the main owner of the landing cell. That can be row 0 (unowned land) or the seat itself.
   - Tests: `test_an_attack_reads_its_targets_share_and_a_boat_its_landing_cells_main_owner`, and the brute-force KL
     over every boat cell (owners 0–3 on a small map).
4. **Mixing troop heads.**
   - A KL between a 100-share and a 5-size network has no meaning: `test_the_kl_needs_the_same_troop_head`.
   - A v3 network plays a v1 network through the bridge: `test_old_and_new_networks_play_each_other`.
5. **A twin run that crashes.**
   - The night runs in a restart loop. A resume must restore both networks at the same update, or refuse clearly.
   - Test: `test_a_twin_trains_on_the_same_windows_and_resumes_with_the_run`.

The share tables' GPU memory ([rows, 2, 403, 100] floats) is not a unit test. The speed check measures it as peak
VRAM (Task 7).

## Files

| File | Change |
|---|---|
| `bridge/actions.ts` | `decodeAction` reads a percent; `TROOP_BINS` removed |
| `bridge/replay.ts` | `shareOf` replaces `binOf`; labels carry percents |
| `bridge/worker.ts`, `bridge/game.ts` | dims lose `troopBins`; comments |
| `bridge/{actions,replay,game,worker}.test.ts` | percents |
| `scripts/troop_shares.ts` | keeps its 5-size statistic with local helpers |
| `scripts/diag_baselines.py` | percents |
| `trainer/model.py` | `SIZES`, `SHARES`, `SIZE_OF`; `ActionDist` with percents and shares; the share head; arch v3; SwiGLU experts; `CORES`; `describe` |
| `trainer/env.py` | docstrings |
| `trainer/eval.py` | `Scripted(share=25)`; `two_bin_sample` |
| `trainer/bc.py` | soft labels; like-for-like `measure`; `--twin` |
| `trainer/ppo.py` | new networks are v3; `CORES` |
| `trainer/control.py` | the loss chart shows the twin |
| `scripts/speed_check.py` | the 3B run set |
| `trainer/test_{model,bc,eval,ppo}.py` | as each task says |
| `CLAUDE.md`, `docs/PLAN.md` | Task 8 |

---

### Task 1: The troop column becomes a percent, end to end

**Files:**
- Modify:
  - bridge: `bridge/actions.ts`, `bridge/replay.ts`, `bridge/worker.ts`, `bridge/game.ts`;
  - scripts: `scripts/troop_shares.ts`, `scripts/diag_baselines.py`;
  - trainer: `trainer/model.py`, `trainer/env.py`, `trainer/eval.py`, `trainer/bc.py`.
- Test: `bridge/actions.test.ts`, `bridge/replay.test.ts`, `bridge/game.test.ts`, `bridge/worker.test.ts`,
  `trainer/test_model.py`, `trainer/test_eval.py`, `trainer/test_ppo.py`.

**Interfaces:**
- Produces:
  - action and label column 3 = troops as a whole percent of current troops (1–100);
  - bridge: `shareOf(troops: number, have: number): number`;
  - trainer: `SIZES = (10, 25, 50, 75, 100)`, `SHARES = 100`, `SIZE_OF` (long tensor [101]: percent → nearest size
    index in log scale), `SIZE_PERCENT` (tensor of `SIZES`);
  - `ActionDist.troop_sample(target, cell) -> [N, 2]` percents (attack, boat);
  - `ActionDist.size_log_probs(target, cell) -> [N, 2, 5]`.

- [ ] **Step 1: Write the failing bridge tests**

In `bridge/actions.test.ts`, replace the test "after spawning, unowned land is attackable and the troop bin sets the
troops" with these two:

```ts
test("after spawning, unowned land is attackable and the troop share sets the troops", async () => {
  const env = await EnvGame.create(ONION, "attack");
  script(env, 260);
  const seat = env.seats[0];
  const attack = attackMask(env.game, seat, env.slots);
  assert.equal(attack[0], 1);
  const types = typeMask(env.game, seat, attack);
  assert.equal(types[Act.Attack], 1);
  const { intent, valid } = decodeAction(env.game, seat, env.frame, types, attack, [Act.Attack, 0, 0, 37]);
  assert.ok(valid);
  assert.deepEqual(intent, { type: "attack", targetID: null, troops: 0.37 * seat.troops() });
  const far = [...attack.keys()].find((s) => s > 1 && attack[s] === 0);
  if (far !== undefined) {
    assert.equal(decodeAction(env.game, seat, env.frame, types, attack, [Act.Attack, 0, far, 10]).valid, false);
  }
});

test("the five sizes of earlier networks send what they sent before", async () => {
  const env = await EnvGame.create(ONION, "sizes");
  script(env, 260);
  const seat = env.seats[0];
  const attack = attackMask(env.game, seat, env.slots);
  const types = typeMask(env.game, seat, attack);
  for (const [share, fraction] of [[10, 0.1], [25, 0.25], [50, 0.5], [75, 0.75], [100, 1.0]] as const) {
    const { intent } = decodeAction(env.game, seat, env.frame, types, attack, [Act.Attack, 0, 0, share]);
    assert.deepEqual(intent, { type: "attack", targetID: null, troops: fraction * seat.troops() }, `${share}%`);
  }
});
```

Make two changes in "malformed or unusable actions are dropped, never thrown":
- In the `bad` list, replace `[Act.Attack, 0, 0, 7]` with
  `[Act.Attack, 0, 0, 0], [Act.Attack, 0, 0, 101], [Act.Attack, 0, 0, 2.5], [Act.Boat, 0, 0, 0]`.
- In the boat loop, replace `[Act.Boat, c, 0, 1]` with `[Act.Boat, c, 0, 25]`.

In `bridge/replay.test.ts`, change the import to `import { OUT_OF_SET, readRecord, RecordGame, shareOf } from
"./replay";`. Then replace the test "troop shares map to the nearest bin in log scale" with:

```ts
test("a label's troop share is the whole percent the player sent", () => {
  assert.equal(shareOf(20, 100), 20);
  assert.equal(shareOf(19.8, 100), 20); // troops change a little between the click and its execution
  assert.equal(shareOf(1000, 4000), 25);
  assert.equal(shareOf(0.3, 100), 1); // a trickle still sends 1%
  assert.equal(shareOf(130, 100), 100); // troops fell since the click
  assert.equal(shareOf(5, 0), 100);
});
```

Change three literal troop columns:
- `bridge/game.test.ts`, the random-play test: `rnd.nextInt(0, 5)` becomes `rnd.nextInt(1, 101)` (`nextInt` is
  `[min, max)`).
- `bridge/game.test.ts`, the second-action test: `[Act.Attack, 0, target, 1, Act.BuildCity, ...` becomes
  `[Act.Attack, 0, target, 25, Act.BuildCity, ...`.
- `bridge/worker.test.ts` line 80: `[Act.Attack, 0, 0, 2]` becomes `[Act.Attack, 0, 0, 50]`.

- [ ] **Step 2: Run the bridge tests to see them fail**

Run (from `OpenFront_Source_Code/openfrontio/`): `node --import tsx --test "../../bridge/*.test.ts"`
Expected: FAIL.
- `replay.test.ts` fails to load: `shareOf` is not exported.
- The 37% attack is dropped: `valid` is false, since `37 in TROOP_BINS` is false.
- `[Act.Attack, 0, 0, 0]` is valid today (bin 0).

- [ ] **Step 3: Implement the bridge side**

`bridge/actions.ts`: delete the `TROOP_BINS` export and its comment, and add below `N_ACTS`:

```ts
/** Attacks and boats send a whole percent of the seat's current troops, 1-100 (PLAN 3B: troop shares). */
const isShare = (s: number) => Number.isInteger(s) && s >= 1 && s <= 100;
```

In `decodeAction`:
- Update the doc comment: `/** Turns a sampled [type, cell, target, share] into a game intent (null intent = no-op).
  share: the percent of its current troops an attack or boat sends. */`
- Destructure `const [type, cell, target, share] = action;`.
- The attack branch:

```ts
  if (type === Act.Attack) {
    if (!Number.isInteger(target) || attack[target] !== 1 || !isShare(share)) return DROP;
    const targetID = target === 0 ? null : (game.playerBySmallID(target) as Player).id();
    return { intent: { type: "attack", targetID, troops: (share / 100) * seat.troops() }, valid: true };
  }
```

- The boat branch: `if (!isShare(share)) return DROP;` and `troops: (share / 100) * seat.troops()`.
- `share / 100` is computed first because 10/100, 25/100, 50/100, 75/100 and 100/100 are the same doubles as the
  old bins. So the old networks send exactly what they sent before.

`bridge/replay.ts`:
- The import becomes `import { Act } from "./actions";`.
- Replace `binOf` with:

```ts
/** The share of its troops a player sent, as a whole percent (1-100). Commands execute a few ticks after the click and
 * troops change meanwhile: 84% of human attacks land within 0.25 points of a whole percent (PLAN 3B). */
export function shareOf(troops: number, have: number): number {
  return Math.min(100, Math.max(1, Math.round((100 * troops) / Math.max(have, 1))));
}
```

- In `labelOf`, the attack returns `[Act.Attack, 0, target, shareOf(intent.troops, seat.troops())]` and the boat
  returns `[Act.Boat, frame.cellOf(intent.dst), 0, shareOf(intent.troops, seat.troops())]`.
- Its doc comment says `[type, cell, target, share]`.

`bridge/worker.ts`:
- The import becomes `import { N_ACTS } from "./actions";`.
- Delete the `troopBins: TROOP_BINS,` line from `dims()`.
- Run `grep -n "bin" bridge/worker.ts`. Expected: no line mentions troop bins. Reword any that does to "share".

`bridge/game.ts`: in the `step` doc comment, `[type, cell, target, bin]` becomes `[type, cell, target, share]`.

`scripts/troop_shares.ts`: its comparison with the five sizes stays reproducible with local helpers.
- Delete the imports of `TROOP_BINS` and `binOf`, keeping `readRecord, RecordGame`.
- Add below `GROUPS`:

```ts
/** The five troop sizes of the Phase 2 and 3A networks, and the nearest one in log scale (how their labels were made). */
const SIZES = [0.1, 0.25, 0.5, 0.75, 1.0];
function sizeOf(share: number): number {
  const s = Math.min(Math.max(share, SIZES[0]), 1);
  let best = 0;
  for (let i = 1; i < SIZES.length; i++) {
    if (Math.abs(Math.log(s / SIZES[i])) < Math.abs(Math.log(s / SIZES[best]))) best = i;
  }
  return best;
}
```

- In `main`, `TROOP_BINS[binOf(x, 100)]` becomes `SIZES[sizeOf(x / 100)]` and `TROOP_BINS.length` becomes
  `SIZES.length`.

- [ ] **Step 4: Run the bridge tests and the typecheck**

Run (game dir): `node --import tsx --test "../../bridge/*.test.ts"`, then
`node node_modules/typescript/bin/tsc -p ../../tsconfig.json`.
Expected: 35 tests pass (34 + the sizes test), and tsc prints nothing.

- [ ] **Step 5: Write the failing trainer tests**

`trainer/test_model.py`:
- Import `SIZES` and `SIZE_OF` from `trainer.model`.
- Drop `"troopBins"` from `DIMS`: `DIMS = {"channels": N_CHANNELS, "playerFeatures": 25, "scalars": 16, "acts": 5}`.
- In `test_kl_is_the_exact_kl_of_the_joint_action`, both `for b in range(5)` become `for b in SIZES`.
- Add to `ModelTest`:

```python
    def test_five_size_networks_send_their_sizes_as_percents(self):
        self.assertEqual(SIZE_OF[list(SIZES)].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(SIZE_OF[[1, 15, 16, 35, 36, 61, 62, 86, 87, 100]].tolist(), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        out = RandomPolicy(DIMS)(self.obs)
        out["type"][:, ATTACK] = 30.0  # rows 3-5 attack (their only target is player 2); rows 0-2 cannot
        out["bins"][:, 0, 1] = 30.0  # and send 25%
        dist = ActionDist(out, self.obs)
        self.assertEqual(dist.sample()[3:, [0, 2, 3]].tolist(), [[ATTACK, 2, 25]] * 3)
        lp = dist.log_prob(torch.tensor([[ATTACK, 0, 2, 20]] * N, device="cuda"))  # 20% lies in 25%'s range
        self.assertTrue(torch.allclose(lp[3:], torch.zeros(3, device="cuda"), atol=1e-4))
```

`trainer/test_eval.py`:
- In the `dist()` helper's docstring, "the attack troop bin is almost surely bin 1" becomes "attacks almost surely send
  25%".
- In `test_each_kind_of_action_keeps_its_own_probability_and_both_can_happen_together`, the attack assertion becomes
  `self.assertEqual(a[attack][:, 1:4].unique(dim=0).tolist(), [[0, 2, 25]])  # target 2, 25%`.

`trainer/test_ppo.py` line 144: drop `"troopBins": [...]` from the dims dict.

- [ ] **Step 6: Run the trainer tests to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model trainer.test_eval -v 2>&1 | tail -5`
Expected: FAIL with `ImportError: cannot import name 'SIZES' from 'trainer.model'`.

- [ ] **Step 7: Implement the trainer side**

`trainer/model.py`: add below `ACT_NAMES`/`SPAWN...`:

```python
SIZES = (10, 25, 50, 75, 100)  # the troop sizes of the Phase 2 and 3A networks, in percent of current troops
SHARES = 100  # the Phase 3B network's troop shares: 1-100% (PLAN 3B)
# Each percent's nearest size in log scale, as the Phase 2-3A labels were made: 1-15% -> 10, 16-35 -> 25, 36-61 -> 50,
# 62-86 -> 75, 87-100 -> 100.
SIZE_OF = torch.tensor([min(range(len(SIZES)), key=lambda i: abs(math.log(max(p, SIZES[0]) / SIZES[i])))
                        for p in range(SHARES + 1)])
SIZE_PERCENT = torch.tensor(SIZES)
```

Replace every `len(dims["troopBins"])`. In `Policy`, `PolicyV1` and `RandomPolicy`, `self.n_bins =
len(dims["troopBins"])` becomes `self.n_bins = len(SIZES)`. The networks no longer read the troop sizes from the
worker.

In `ActionDist`:

```python
class ActionDist:
    """The joint action [type, cell, target, troops]: the type picks which argument heads are used.

    spawn: spawn map cell; attack: player pointer + troops; boat: boat map cell + troops; build city: city map cell.
    troops: the percent of its current troops an attack or boat sends (1-100). The Phase 2 and 3A networks pick one of
    the five SIZES per kind, whatever the target."""

    def __init__(self, out, obs):
        self.type = _masked(out["type"].float(), obs["type_mask"])
        self.cells = _masked(out["maps"].float(), obs["cell_mask"])  # batch [N, 3]
        self.ptr = _masked(out["ptr"].float(), obs["attack_mask"])
        self.bins = Categorical(logits=out["bins"].float())  # batch [N, 2]: attack, boat

    def troop_sample(self, target, cell):
        """Percents [N, 2] sampled for an attack on `target` and a boat to `cell`."""
        return SIZE_PERCENT.to(target.device)[self.bins.sample()]

    def size_log_probs(self, target, cell):
        """Log-probabilities [N, 2, 5] of the five SIZES for an attack on `target` and a boat to `cell`."""
        return self.bins.logits

    def sample(self):
        t, c, p = self.type.sample(), self.cells.sample(), self.ptr.sample()
        zero = torch.zeros_like(t)
        cell = torch.where(t == SPAWN, c[:, 0], torch.where(t == BOAT, c[:, 1], torch.where(t == BUILD, c[:, 2], zero)))
        target = torch.where(t == ATTACK, p, zero)
        pct = self.troop_sample(p, c[:, 1])
        troops = torch.where(t == ATTACK, pct[:, 0], torch.where(t == BOAT, pct[:, 1], zero))
        return torch.stack([t, cell, target, troops], 1)

    def log_prob(self, actions):
        t, cell, target, troops = actions.unbind(1)
        lc = self.cells.log_prob(cell[:, None].expand(-1, N_MAPS))
        lb = self.bins.log_prob(SIZE_OF.to(troops.device)[troops.clamp(0, SHARES)][:, None].expand(-1, 2))
        zero = torch.zeros_like(lc[:, 0])
        return (self.type.log_prob(t)
                + torch.where(t == SPAWN, lc[:, 0], zero)
                + torch.where(t == ATTACK, self.ptr.log_prob(target) + lb[:, 0], zero)
                + torch.where(t == BOAT, lc[:, 1] + lb[:, 1], zero)
                + torch.where(t == BUILD, lc[:, 2], zero))
```

`entropy`, `kl` and `_kl` stay as they are.

`trainer/eval.py`:
- `Scripted`:

```python
    def __init__(self, share=25, every=4, seed=0):
        self.share, self.every, self.rng, self.step = share, every, np.random.default_rng(seed), 0
```

  - In its docstring, "with troop bin `bin`" becomes "sending `share` percent of its troops".
  - The attack row becomes `act = [2, 0, int(self.rng.choice(targets)), self.share]`.
- In `two_bin_sample`, the samples and the first action become:

```python
    cells, target, zero = dist.cells.sample(), dist.ptr.sample(), torch.zeros_like(troop)
    pct = dist.troop_sample(target, cells[:, 1])
    first = torch.stack([troop, torch.where(troop == BOAT, cells[:, 1], zero),
                         torch.where(troop == ATTACK, target, zero),
                         torch.where(troop == ATTACK, pct[:, 0], torch.where(troop == BOAT, pct[:, 1], zero))], 1)
```

`trainer/bc.py`:
- The model import gains `SHARES, SIZE_OF`.
- In `measure`, replace the `lb = dist.bins.log_prob(...)` line and the two bin accuracies:

```python
    size = SIZE_OF.to(b.device)[b.clamp(0, SHARES)]  # the label's percent as one of the five sizes
    ls = dist.size_log_probs(tgt, cell)  # [N, attack/boat, 5]
    lb = ls.gather(2, size[:, None, None].expand(-1, 2, 1))[:, :, 0]  # [N, attack/boat]
```

- `bc/attack_bin_acc` compares `ls[:, 0].argmax(1) == size`, and `bc/boat_bin_acc` compares
  `ls[:, 1].argmax(1) == size`.

`trainer/env.py`:
- In the `send()` docstring, `[type, cell, target, bin]` becomes `[type, cell, target, troops %]`.
- The labels comment becomes `the label [type, cell, target, troops %]`.

`scripts/diag_baselines.py`:
- `BINS = [...]` becomes `SIZES = [10, 25, 50, 75, 100]`.
- The docstring says `(troop share in percent, attack every n decisions)`.
- The random branch's `rng.integers(5)` becomes `int(rng.choice(SIZES))`.
- `troop_bin, every = policy` becomes `share, every = policy`, and the attack row becomes `[2, 0, 0, share]`.
- The policies become `("random", (10, 1), (25, 4))`, and the name becomes `f"expand {policy[0]}% every
  {policy[1]}"`.

- [ ] **Step 8: Run all the tests**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . 2>&1 | tail -4` and the bridge suite again.
Expected: 90 trainer tests OK, and 35 bridge tests pass.

- [ ] **Step 9: Commit**

```bash
git add bridge scripts/troop_shares.ts scripts/diag_baselines.py trainer
git commit -m "Troop column is a percent end to end (PLAN 3B): the bridge decodes 1-100% of current troops, labels carry the human's share rounded to a whole percent (clamped to 1-100), and the Phase 2/3A networks map their five sizes to 10/25/50/75/100%, which send exactly the troops their bins sent"
```

---

### Task 2: ActionDist reads troop shares with a player's token

**Files:**
- Modify: `trainer/model.py` (`ActionDist`, constants)
- Test: `trainer/test_model.py`

**Interfaces:**
- Consumes (Task 1): `SIZES`, `SHARES`, `SIZE_OF`, `SIZE_PERCENT`, `troop_sample`, `size_log_probs`.
- Produces:
  - `ActionDist(out, obs)` accepts `out["shares"]` [N, 2, P, SHARES] (attack and boat × player token; needs
    `obs["owners"]`) or `out["bins"]` [N, 2, 5];
  - `dist.shares: bool`;
  - `troop_log_prob(target, cell, percent, smooth=0.0, sizes=False) -> [N, 2]`;
  - `size_log_probs(target, cell) -> [N, 2, 5]` for both kinds of network;
  - `log_prob(actions, smooth=0.0, sizes=False)`;
  - `kl(ref)` raises `ValueError` when the troop heads differ;
  - `SIZE_RANGES` [5, SHARES].

- [ ] **Step 1: Write the failing tests**

In `trainer/test_model.py`:
- Add `from torch.distributions import Categorical` and import `SHARES` from `trainer.model`.
- Add these helpers below `exact_fp32`:

```python
def brute_kl(p_out, q_out, obs, troops):
    """Per row, the sum over every joint action of p log(p / q); `troops` are the percents a troop head can send."""
    hw, players = obs["cell_mask"].shape[-1], obs["attack_mask"].shape[-1]
    acts = torch.tensor([[0, 0, 0, 0]] + [[SPAWN, c, 0, 0] for c in range(hw)]
                        + [[ATTACK, 0, t, s] for t in range(players) for s in troops]
                        + [[BOAT, c, 0, s] for c in range(hw) for s in troops] + [[BUILD, c, 0, 0] for c in range(hw)])
    rows = []
    for i in range(len(obs["type_mask"])):
        one = lambda d: {k: v[i:i + 1].expand(len(acts), *v.shape[1:]) for k, v in d.items()}
        lp = ActionDist(one(p_out), one(obs)).log_prob(acts)
        lq = ActionDist(one(q_out), one(obs)).log_prob(acts)
        rows.append(float((lp.exp() * (lp - lq)).sum()))
    return rows


def small_obs(seed=0):
    """3 rows on 16 cells with 4 player rows (0: unowned land, 1: the seat, 2-3: others), every type legal and the
    cells' main owners drawn among all four: small enough to sum over every joint action."""
    g = torch.Generator().manual_seed(seed)
    return {"type_mask": torch.ones(3, 5, dtype=torch.bool), "cell_mask": torch.rand(3, 3, 16, generator=g) < 0.6,
            "attack_mask": torch.tensor([[True, False, True, True]] * 3),
            "owners": torch.randint(0, 4, (3, 2, 16), generator=g).to(torch.int16)}


def share_out(seed=0):
    """Random outputs of a 100-share network for small_obs."""
    g = torch.Generator().manual_seed(seed)
    return {"type": torch.randn(3, 5, generator=g), "maps": torch.randn(3, 3, 16, generator=g) * 3,
            "ptr": torch.randn(3, 4, generator=g), "shares": torch.randn(3, 2, 4, SHARES, generator=g) * 2}
```

The existing `test_kl_is_the_exact_kl_of_the_joint_action` now uses the helper. Replace its brute-force loop (from
`hw = range(H * W)` to the end of the `for i in range(N)` loop) with:

```python
        for i, want in enumerate(brute_kl(p_out, q_out, obs, SIZES)):
            self.assertAlmostEqual(float(kl[i]), want, places=4)
```

Add the class:

```python
class ShareDistTest(unittest.TestCase):
    """ActionDist with a 100-share troop head (PLAN 3B), on small_obs."""

    def test_an_attack_reads_its_targets_share_and_a_boat_its_landing_cells_main_owner(self):
        obs, out = small_obs(), share_out()
        out["shares"].zero_()
        for who in range(4):
            out["shares"][:, 0, who, 10 * who + 4] = 50.0  # an attack on player `who` sends 10 * who + 5 %
            out["shares"][:, 1, who, 10 * who + 9] = 50.0  # a boat to a cell `who` mainly owns: 10 * who + 10 %
        dist = ActionDist(out, obs)
        target, cell = torch.tensor([0, 2, 3]), torch.tensor([0, 5, 15])
        owner = obs["owners"][torch.arange(3), 0, cell].long()
        pct = dist.troop_sample(target, cell)
        self.assertEqual(pct[:, 0].tolist(), (10 * target + 5).tolist())
        self.assertEqual(pct[:, 1].tolist(), (10 * owner + 10).tolist())
        lp = dist.troop_log_prob(target, cell, 10 * owner + 10)
        self.assertTrue(torch.allclose(lp[:, 1], torch.zeros(3), atol=1e-4))
        out["type"] = torch.tensor([[-30.0, -30.0, -30.0, 30.0, -30.0]] * 3)  # boats everywhere
        a = ActionDist(out, obs).sample()
        self.assertEqual(a[:, 3].tolist(), (10 * obs["owners"][torch.arange(3), 0, a[:, 1]].long() + 10).tolist())

    def test_kl_is_the_exact_kl_of_the_joint_action(self):
        obs, p_out, q_out = small_obs(), share_out(1), share_out(2)
        kl = ActionDist(p_out, obs).kl(ActionDist(q_out, obs))
        for i, want in enumerate(brute_kl(p_out, q_out, obs, range(1, SHARES + 1))):
            self.assertAlmostEqual(float(kl[i]), want, delta=1e-4 * max(1.0, want))
        self.assertTrue(torch.allclose(ActionDist(p_out, obs).kl(ActionDist(p_out, obs)), torch.zeros(3), atol=1e-6))

    def test_the_share_entropy_weights_each_token_by_its_pointer_or_boat_probability(self):
        obs, out = small_obs(), share_out()
        d, flat = ActionDist(out, obs), ActionDist(dict(out, shares=torch.zeros_like(out["shares"])), obs)
        h = Categorical(logits=out["shares"]).entropy() / math.log(SHARES)  # [3, 2, 4]; uniform shares give 1
        boat = torch.zeros(3, 4)
        for i in range(3):
            for c in range(16):
                boat[i, int(obs["owners"][i, 0, c])] += d.cells.probs[i, 1, c]
        pt = d.type.probs
        want = pt[:, ATTACK] * ((d.ptr.probs * h[:, 0]).sum(1) - 1) + pt[:, BOAT] * ((boat * h[:, 1]).sum(1) - 1)
        self.assertTrue(torch.allclose(d.entropy() - flat.entropy(), want, atol=1e-5))

    def test_a_soft_label_spreads_onto_the_neighbouring_percents(self):
        obs, out = small_obs(), share_out()
        d = ActionDist(out, obs)
        target, cell = torch.full((3,), 2), torch.zeros(3, dtype=torch.long)
        lp = out["shares"][:, 0, 2].log_softmax(-1)  # the attack on player 2; index k is k + 1 percent
        for percent, near in ((20, (18, 19, 20)), (1, (0, 0, 1)), (100, (98, 99, 99))):
            got = d.troop_log_prob(target, cell, torch.full((3,), percent), smooth=0.1)[:, 0]
            want = 0.05 * lp[:, near[0]] + 0.9 * lp[:, near[1]] + 0.05 * lp[:, near[2]]
            self.assertTrue(torch.allclose(got, want, atol=1e-5), percent)
        self.assertTrue(torch.allclose(d.troop_log_prob(target, cell, torch.full((3,), 20))[:, 0], lp[:, 19]))

    def test_uniform_shares_sum_into_the_five_sizes_by_their_ranges(self):
        obs, out = small_obs(), share_out()
        out["shares"].zero_()
        d = ActionDist(out, obs)
        p = d.size_log_probs(torch.full((3,), 2), torch.zeros(3, dtype=torch.long)).exp()
        want = torch.tensor([15, 20, 26, 25, 14]) / 100  # 1-15%, 16-35, 36-61, 62-86, 87-100
        self.assertTrue(torch.allclose(p, want.expand(3, 2, 5), atol=1e-6))
        acts = torch.tensor([[ATTACK, 0, 2, 20]] * 3)
        gap = d.log_prob(acts, sizes=True) - d.log_prob(acts)  # 25%'s 20 percents against one
        self.assertTrue(torch.allclose(gap, torch.full((3,), math.log(20)), atol=1e-5))

    def test_the_kl_needs_the_same_troop_head(self):
        obs, out = small_obs(), share_out()
        sizes = {k: v for k, v in out.items() if k != "shares"} | {"bins": torch.zeros(3, 2, 5)}
        with self.assertRaisesRegex(ValueError, "same troop head"):
            ActionDist(out, obs).kl(ActionDist(sizes, obs))
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v 2>&1 | tail -5`
Expected: the 6 `ShareDistTest` tests fail with `KeyError: 'bins'`. The refactored sizes KL test passes.

- [ ] **Step 3: Implement**

`trainer/model.py`: add below `SIZE_PERCENT`:

```python
SIZE_RANGES = torch.where(SIZE_OF[1:] == torch.arange(len(SIZES))[:, None], 0.0, -math.inf)  # [5, SHARES]: each size's percents
```

Replace the whole `ActionDist` class with:

```python
class ActionDist:
    """The joint action [type, cell, target, troops]: the type picks which argument heads are used.

    spawn: spawn map cell; attack: player pointer + troops; boat: boat map cell + troops; build city: city map cell.
    troops: the percent of its current troops an attack or boat sends (1-100).
    - A Phase 3B network ("shares" in its outputs, [N, 2, P, SHARES]) reads the share from a player's token: the
      attack's target, or the main owner of the boat's landing cell (obs "owners").
    - The Phase 2 and 3A networks ("bins", [N, 2, 5]) pick one of the five SIZES per kind, whatever the target: a
      single token."""

    def __init__(self, out, obs):
        self.type = _masked(out["type"].float(), obs["type_mask"])
        self.cells = _masked(out["maps"].float(), obs["cell_mask"])  # batch [N, 3]
        self.ptr = _masked(out["ptr"].float(), obs["attack_mask"])
        self.shares = "shares" in out
        if self.shares:
            self.troops = Categorical(logits=out["shares"].float())  # batch [N, 2, P]: attack, boat x token
            self.owner = obs["owners"][:, 0].long()  # [N, cells]: each cell's main owner
        else:
            self.troops = Categorical(logits=out["bins"].float()[:, :, None])  # batch [N, 2, 1]

    def _at(self, target, cell):
        """The troop logits [N, 2, choices] of an attack on `target` and a boat to `cell`."""
        logits = self.troops.logits
        if not self.shares:
            return logits[:, :, 0]
        who = torch.stack([target, self.owner.gather(1, cell[:, None])[:, 0]], 1)  # [N, 2]
        return logits.gather(2, who[:, :, None, None].expand(-1, -1, 1, SHARES))[:, :, 0]

    def _weights(self):
        """[N, 2, tokens]: how likely an attack or a boat reads each token's troop head, given its type."""
        if not self.shares:
            return torch.ones_like(self.troops.logits[..., 0])
        boat = torch.zeros_like(self.ptr.probs).scatter_add_(1, self.owner, self.cells.probs[:, 1])
        return torch.stack([self.ptr.probs, boat], 1)

    def troop_sample(self, target, cell):
        """Percents [N, 2] sampled for an attack on `target` and a boat to `cell`."""
        k = Categorical(logits=self._at(target, cell)).sample()
        return k + 1 if self.shares else SIZE_PERCENT.to(k.device)[k]

    def size_log_probs(self, target, cell):
        """Log-probabilities [N, 2, 5] of the five SIZES for an attack on `target` and a boat to `cell`. A 100-share
        network's shares are summed over the percents each size stands for (SIZE_OF), for like-for-like scores."""
        lp = self._at(target, cell).log_softmax(-1)
        return (lp[..., None, :] + SIZE_RANGES.to(lp.device)).logsumexp(-1) if self.shares else lp

    def troop_log_prob(self, target, cell, percent, smooth=0.0, sizes=False):
        """[N, 2]: the log-probability that an attack on `target` or a boat to `cell` sends `percent`.
        - sizes: of the size `percent` falls in (SIZE_OF), for every network;
        - smooth (100-share networks): the label keeps 1 - smooth of its weight and puts smooth / 2 on each neighbouring
          percent (back on itself at 1% and 100%); the result is the expected log-probability."""
        if sizes or not self.shares:
            size = SIZE_OF.to(percent.device)[percent.clamp(0, SHARES)]
            return self.size_log_probs(target, cell).gather(2, size[:, None, None].expand(-1, 2, 1))[:, :, 0]
        lp = self._at(target, cell).log_softmax(-1)
        k = (percent - 1).clamp(0, SHARES - 1)[:, None, None].expand(-1, 2, 1)
        if not smooth:
            return lp.gather(2, k)[:, :, 0]
        near = (k + torch.tensor([-1, 0, 1], device=k.device)).clamp(0, SHARES - 1)  # [N, 2, 3]
        weight = torch.tensor([smooth / 2, 1 - smooth, smooth / 2], device=lp.device).expand(near.shape)
        return (torch.zeros_like(lp).scatter_add_(2, near, weight) * lp).sum(2)

    def sample(self):
        t, c, p = self.type.sample(), self.cells.sample(), self.ptr.sample()
        zero = torch.zeros_like(t)
        cell = torch.where(t == SPAWN, c[:, 0], torch.where(t == BOAT, c[:, 1], torch.where(t == BUILD, c[:, 2], zero)))
        target = torch.where(t == ATTACK, p, zero)
        pct = self.troop_sample(p, c[:, 1])
        troops = torch.where(t == ATTACK, pct[:, 0], torch.where(t == BOAT, pct[:, 1], zero))
        return torch.stack([t, cell, target, troops], 1)

    def log_prob(self, actions, smooth=0.0, sizes=False):
        """log pi(actions) [N]; `smooth` and `sizes` as in troop_log_prob."""
        t, cell, target, troops = actions.unbind(1)
        lc = self.cells.log_prob(cell[:, None].expand(-1, N_MAPS))
        lt = self.troop_log_prob(target, cell, troops, smooth, sizes)
        zero = torch.zeros_like(lc[:, 0])
        return (self.type.log_prob(t)
                + torch.where(t == SPAWN, lc[:, 0], zero)
                + torch.where(t == ATTACK, self.ptr.log_prob(target) + lt[:, 0], zero)
                + torch.where(t == BOAT, lc[:, 1] + lt[:, 1], zero)
                + torch.where(t == BUILD, lc[:, 2], zero))

    def entropy(self):
        """Exact joint entropy, H(type) + sum_k p(k) H(args | k), with every head divided by the log of its size so
        the 4k-cell maps do not drown out the type head. A troop head's entropy is averaged over the tokens it reads,
        weighted by the pointer or the boat map. p(k) and those weights are detached: otherwise the bonus pushes the
        type head toward the types with the most arguments (attack, boat) and away from no-op, as it did in Phase 1,
        and the pointer toward the targets with the least certain shares."""
        hc = self.cells.entropy() / math.log(self.cells.logits.shape[-1])
        hp = self.ptr.entropy() / math.log(max(self.ptr.logits.shape[-1], 2))
        ht = (self._weights().detach() * self.troops.entropy()).sum(2) / math.log(self.troops.logits.shape[-1])
        pt = self.type.probs.detach()
        return (self.type.entropy() / math.log(pt.shape[-1]) + pt[:, SPAWN] * hc[:, 0]
                + pt[:, ATTACK] * (hp + ht[:, 0]) + pt[:, BOAT] * (hc[:, 1] + ht[:, 1]) + pt[:, BUILD] * hc[:, 2])

    def kl(self, ref):
        """Exact KL(self || ref) of the joint action, KL(type) + sum_k p(k) KL(args | k); a troop head's KL is averaged
        over the tokens it reads, weighted by this policy's pointer or boat map. The one-sample estimate from the taken
        action is not usable: it explodes when that action is one this policy all but rules out and ref does not,
        which a spread-out 49k-cell map makes common."""
        if self.shares != ref.shares:
            raise ValueError("the KL needs two networks with the same troop head (100 shares or 5 sizes)")
        kc = _kl(self.cells, ref.cells)  # [N, 3]
        kt = (self._weights() * _kl(self.troops, ref.troops)).sum(2)  # [N, 2]
        pt = self.type.probs
        return (_kl(self.type, ref.type) + pt[:, SPAWN] * kc[:, 0] + pt[:, ATTACK] * (_kl(self.ptr, ref.ptr) + kt[:, 0])
                + pt[:, BOAT] * (kc[:, 1] + kt[:, 1]) + pt[:, BUILD] * kc[:, 2])
```

- [ ] **Step 4: Run the model tests, then everything**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v 2>&1 | tail -5`, then the full trainer suite.
Expected: all pass, 96 trainer tests. The full suite matters because `bc.measure` and `eval.two_bin_sample` call the
methods that changed.

- [ ] **Step 5: Commit**

```bash
git add trainer/model.py trainer/test_model.py
git commit -m "ActionDist reads troop shares with a player's token (PLAN 3B): an attack's share at its target, a boat's at its landing cell's main owner; exact entropy and KL weight each token by the pointer or the boat map; soft labels and the five sizes' ranges for like-for-like scores; a KL between a 100-share and a 5-size network is refused"
```

---

### Task 3: Policy v3, the troop-share head

**Files:**
- Modify: `trainer/model.py` (`Policy`, `DS`, `arch_of`, `new_net`, `describe`), `trainer/ppo.py` (`network_args`),
  `trainer/bc.py` (`run_info`)
- Test: `trainer/test_model.py`, `trainer/test_bc.py`, `trainer/test_eval.py`

**Interfaces:**
- Consumes (Task 2): `ActionDist` with `out["shares"]`.
- Produces:
  - `Policy(dims, widths=(64, 128, 192), core="moe", shares=False)`; with `shares=True`, `out["shares"]` is
    [N, 2, P, SHARES] and there is no `out["bins"]`;
  - arch "v3" (new runs);
  - `describe(args) -> str`.

- [ ] **Step 1: Write the failing tests**

`trainer/test_model.py`:
- `nets()` returns `{"v1": PolicyV1(DIMS), "dense": Policy(DIMS, core="dense"), "moe": Policy(DIMS, core="moe"),
  "v3": Policy(DIMS, shares=True)}`.
- In `test_output_shapes`, replace the `bins` assertion with:

```python
                if name == "v3":
                    self.assertEqual(out["shares"].shape, (N, 2, P, SHARES))
                else:
                    self.assertEqual(out["bins"].shape, (N, 2, 5))
```

Add to `ModelTest`:

```python
    def test_a_new_share_head_starts_uniform(self):
        with torch.no_grad():
            out = Policy(DIMS, shares=True).cuda()(self.obs)
        self.assertEqual(int(torch.count_nonzero(out["shares"])), 0)
```

In `PolicyTest`, each player test covers both troop heads:

```python
    def test_absent_players_change_nothing(self):
        exact_fp32(self)
        for shares in (False, True):
            with self.subTest(shares=shares):
                net = randomized(Policy(DIMS, shares=shares)).cuda().eval()
                noisy = dict(self.obs, players=self.obs["players"].clone())
                noisy["players"][:, 4, 1:] = 3.0  # the absent row (feature 0 = 0) holds garbage, relation flags included
                with torch.no_grad():
                    a, b = net(self.obs), net(noisy)
                self.assertGreater(float(a["ptr"][:, :4].std()), 1e-2, "the pointer depends on the players")
                pairs = [(a[k], b[k]) for k in ("type", "maps", "value", "gates")] + [(a["ptr"][:, :4], b["ptr"][:, :4])]
                pairs.append((a["shares"][:, :, :4], b["shares"][:, :, :4]) if shares else (a["bins"], b["bins"]))
                for x, y in pairs:
                    self.assertTrue(torch.allclose(x, y, rtol=1e-4, atol=1e-5))

    def test_players_can_come_in_any_order(self):
        exact_fp32(self)
        perm = torch.tensor([0, 3, 1, 4, 2], device="cuda")  # new row i holds old row perm[i]
        moved = dict(self.obs, players=self.obs["players"][:, perm], attack_mask=self.obs["attack_mask"][:, perm],
                     owners=torch.argsort(perm)[self.obs["owners"].long()].to(torch.int16))
        for shares in (False, True):
            with self.subTest(shares=shares):
                net = randomized(Policy(DIMS, shares=shares)).cuda().eval()
                with torch.no_grad():
                    a, b = net(self.obs), net(moved)
                pairs = [(a[k], b[k]) for k in ("type", "maps", "value", "gates")] + [(a["ptr"][:, perm], b["ptr"])]
                pairs.append((a["shares"][:, :, perm], b["shares"]) if shares else (a["bins"], b["bins"]))
                for x, y in pairs:
                    self.assertTrue(torch.allclose(x, y, rtol=1e-4, atol=1e-4))
```

In `LoadTest`:
- Add `({"arch": "v3", "core": "moe", "widths": "64,128,192"}, Policy)` to the cases.
- After the `moe` assertion, add `self.assertEqual(getattr(net, "shares", False), args.get("arch") == "v3")`.

`trainer/test_bc.py`:
- `CloneTest` builds `Policy(dims, core="moe", shares=True)`. This is the network the night trains, on real replayed
  observations with 403 player rows.
- In `test_the_control_page_says_what_is_training`, set `a.arch = "v3"` and expect
  `"player transformer, Soft-MoE core (8 experts), 100 troop shares"`.
- In `MainTest.test_train_save_and_resume`, expect `("v3", "moe")`.

`trainer/test_eval.py` `test_old_and_new_networks_play_each_other`: the "new" network becomes
`{"arch": "v3", "core": "moe", "widths": "64,128,192"}`. That is 100 shares against the Phase 2 network's 5 sizes,
through the bridge.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model trainer.test_bc trainer.test_eval 2>&1 | tail -5`
Expected: errors, `TypeError: Policy.__init__() got an unexpected keyword argument 'shares'`.

- [ ] **Step 3: Implement**

`trainer/model.py`: add below `ATTENDED`:

```python
DS = 64  # hidden width of the troop-share head, per kind (attack, boat)
```

Add to the `Policy` docstring's list:
`- troops: with shares=True (arch v3, PLAN 3B), per kind a hidden layer reads the core and one player's token, then
  100 shares; otherwise the 3A network's 5 sizes from the core alone.`

The signature becomes `def __init__(self, dims, widths=(64, 128, 192), core="moe", shares=False):`. Replace the
lines from `self.type_head = ...` through the zero-init loop with:

```python
        self.type_head = nn.Linear(512, dims["acts"])
        self.shares = shares
        if shares:  # per kind (attack, boat): a hidden layer reading the core and one player's token, then the shares
            self.share_h, self.share_t = nn.Linear(512, 2 * DS), nn.Linear(D, 2 * DS)
            self.share_out = nn.ModuleList(nn.Linear(DS, SHARES) for _ in range(2))
            troop_heads = tuple(self.share_out)
        else:
            self.bin_head = nn.Linear(512, 2 * len(SIZES))
            troop_heads = (self.bin_head,)
        self.query, self.key = nn.Linear(512, D), nn.Linear(D, D)
        self.value_head = nn.Linear(512, 1)
        # Policy outputs start at zero: a uniform policy over legal actions.
        for layer in (self.type_head, *troop_heads, self.shuffle, self.skip, self.query):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
```

- Delete `self.n_bins = len(SIZES)` from `Policy`. The v2 branch above builds the layers in the old order, so a v2
  network still starts from the same weights for a given seed.
- The core stays last.
- At the end of `forward`, the output becomes:

```python
        ptr = torch.einsum("nd,npd->np", self.query(h), self.key(tokens)) / math.sqrt(D)
        out.update(type=self.type_head(h), maps=maps.flatten(2), ptr=ptr, value=self.value_head(h).squeeze(1))
        if self.shares:
            n, p = tokens.shape[:2]
            z = F.relu(self.share_h(h).view(n, 1, 2, DS) + self.share_t(tokens).view(n, p, 2, DS))  # [N, P, kind, DS]
            out["shares"] = torch.stack([head(z[:, :, k]) for k, head in enumerate(self.share_out)], 1)
        else:
            out["bins"] = self.bin_head(h).view(-1, 2, len(SIZES))
        return out
```

`arch_of` gets the docstring: `"""The network a run's settings describe: arch v1 (Phase 2), v2 (Phase 3A, 5 troop
sizes) or v3 (Phase 3B, 100 troop shares). Phase 2 checkpoints predate "arch": they are PolicyV1."""`

`new_net` builds:

```python
    if spec["arch"] == "v1":
        return PolicyV1(dims, widths)
    return Policy(dims, widths, spec["core"], shares=spec["arch"] == "v3")
```

Add below `new_net`:

```python
CORE_NAMES = {"dense": "dense core", "moe": "Soft-MoE core (8 experts)"}


def describe(args):
    """A network in words, from a run's settings (the control page)."""
    spec = arch_of(args)
    if spec["arch"] == "v1":
        return "the Phase 2 network"
    troops = "100 troop shares" if spec["arch"] == "v3" else "5 troop sizes"
    return f"player transformer, {CORE_NAMES[spec['core']]}, {troops}"
```

`trainer/ppo.py` `network_args`:
- `a.arch = "v2"` becomes `a.arch = "v3"`.
- The docstring's last sentence becomes "Without it, a Phase 3B network (100 troop shares) from --core and
  --widths."

`trainer/bc.py`:
- Import `describe`.
- `run_info` becomes:

```python
def run_info(a, train, val):
    """What the control page says about the run."""
    return {"Name": a.run, "Method": "behaviour cloning (imitating humans)", "Network": describe(vars(a)),
            "Starts from": a.init or "random weights",
            "Data": f"{len(train):,} ranked 1v1 replays, {len(val):,} held out",
            "Games at once": f"{a.workers * a.games_per_worker} ({a.workers} workers x {a.games_per_worker})",
            "Updates": f"{a.updates:,} of {a.accum} windows each"}
```

- [ ] **Step 4: Run all the trainer tests**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . 2>&1 | tail -4`
Expected: OK, 97 tests.

- [ ] **Step 5: Commit**

```bash
git add trainer
git commit -m "Policy v3 (PLAN 3B): a troop-share head per kind reads the core and one player's token and gives 100 shares, zero-initialised (uniform); new runs are v3, v2 checkpoints keep their five sizes; the control page names the troop head"
```

---

### Task 4: BC on shares: soft labels and like-for-like scores

**Files:**
- Modify: `trainer/bc.py` (`SHARE_SMOOTH`, `bc_loss`, `backward_windows`, `measure`, `score`, `validate`)
- Test: `trainer/test_bc.py`

**Interfaces:**
- Consumes (Task 2): `log_prob(actions, smooth, sizes)`, `troop_log_prob`, `size_log_probs`, `dist.shares`;
  (Task 1) `SIZE_OF`, `SHARES`.
- Produces:
  - `SHARE_SMOOTH = 0.1`;
  - `bc_loss(model, obs, labels, legal, target, vf_coef, smooth=0.0) -> (loss, dist, value, gates)`;
  - `measure(m, obs, labels, legal, dist, value, target, gates=None)`;
  - metric keys: `bc/nll` (like for like), `bc/nll_shares`, `bc/nll_attack_share`, `bc/nll_boat_share`.

- [ ] **Step 1: Write the failing tests**

`trainer/test_bc.py`:
- Imports: add `measure` from `trainer.bc`, and `ATTACK, SHARES, ActionDist` from `trainer.model`.
- Add:

```python
class MeasureTest(unittest.TestCase):
    def test_a_share_networks_troops_score_like_for_like_and_on_its_own_shares(self):
        obs = fake_obs()  # rows 3-5: no-op, attack (only on player 2) and city are legal
        hw, players = obs["cell_mask"].shape[-1], obs["attack_mask"].shape[-1]
        out = {"type": torch.zeros(ROWS, 5, device="cuda"), "maps": torch.zeros(ROWS, 3, hw, device="cuda"),
               "ptr": torch.zeros(ROWS, players, device="cuda"),
               "shares": torch.zeros(ROWS, 2, players, SHARES, device="cuda")}
        labels = torch.tensor([[ATTACK, 0, 2, 20]] * ROWS, device="cuda")
        legal = label_legal(obs, labels)
        self.assertEqual(legal.tolist(), [False] * 3 + [True] * 3)
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, legal, ActionDist(out, obs), zero, zero)
        s = m.means()
        self.assertAlmostEqual(s["bc/nll_attack_bin"], -math.log(0.20), places=4)  # 20% lies in 25%'s range, 16-35%
        self.assertAlmostEqual(s["bc/nll_attack_share"], math.log(SHARES), places=4)
        self.assertAlmostEqual(s["bc/nll"] - s["bc/nll_type"], -math.log(0.20), places=4)  # one legal target adds 0
        self.assertAlmostEqual(s["bc/nll_shares"] - s["bc/nll_type"], math.log(SHARES), places=4)
```

- `ScoreTest`: add a third checkpoint, `("shares", {"arch": "v3", "core": "moe", "widths": "64,128,192"})`.
- `old, moe = first["metrics"]` becomes `old, moe, shares = first["metrics"]`, and add:

```python
        self.assertIn("bc/nll_shares", shares)
        self.assertNotIn("bc/nll_shares", moe)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v 2>&1 | tail -6`
Expected: FAIL.
- `MeasureTest` fails with `TypeError` (`measure()` still takes `logp`).
- `ScoreTest` fails on `bc/nll_shares`.

- [ ] **Step 3: Implement**

`trainer/bc.py`: add below `SCORE_LAYOUT`:

```python
SHARE_SMOOTH = 0.1  # a troop-share label's weight on its two neighbouring percents (PLAN 3B: soft labels)
```

`bc_loss`:

```python
def bc_loss(model, obs, labels, legal, target, vf_coef, smooth=0.0):
    """-log pi(label) summed over the rows that can enter the loss and divided by all rows, plus the value loss. With
    `smooth`, a troop-share label keeps 1 - smooth of its weight and spreads the rest onto the neighbouring percents.
    Every window has the same rows, so k accumulated windows give the gradient of one batch of all of them."""
    with torch.autocast("cuda", torch.bfloat16):
        out = model(obs)
    dist = ActionDist(out, obs)
    safe = torch.where(legal[:, None], labels, torch.zeros_like(labels))  # other rows: a no-op
    nll = -(dist.log_prob(safe, smooth=smooth) * legal).sum() / len(legal)
    value = out["value"].float()
    return nll + vf_coef * 0.5 * (value - target).pow(2).mean(), dist, value, out.get("gates")
```

`backward_windows` (still one model):

```python
    for obs, labels, legal, target in windows:
        loss, dist, value, gates = bc_loss(model, obs, labels, legal, target, a.vf_coef, SHARE_SMOOTH)
        (loss / a.accum).backward()
        measure(meter, obs, labels, legal, dist, value, target, gates)
```

`measure`:

```python
@torch.no_grad()
def measure(m, obs, labels, legal, dist, value, target, gates=None):
    """For one batch:
    - label statistics;
    - accuracy and NLL per head. bc/nll and the troop heads' bc/*_bin are like for like for every network: a
      100-share network's shares are summed into the five sizes' ranges (SIZE_OF). Its own 100-share NLL is
      bc/nll_shares, with bc/nll_*_share per head;
    - NLL per bucket (game phase and map size);
    - the router's mean weight per expert and bucket, for MoE networks."""
    t = labels[:, 0]
    safe = torch.where(legal[:, None], labels, torch.zeros_like(labels))  # rows outside the loss: a no-op
    _, cell, tgt, b = safe.unbind(1)
    n, nl = len(t), legal.sum()
    logp = dist.log_prob(safe, sizes=True)
    m.add("labels/out_of_set", (t == -1).sum(), n)
    m.add("labels/unmapped", (t == -2).sum(), n)
    m.add("labels/illegal", ((t >= 0) & ~legal).sum(), n)
    m.add("labels/noop", ((t == 0) & legal).sum(), nl)
    m.add("bc/nll", -(logp * legal).sum(), nl)
    if dist.shares:
        m.add("bc/nll_shares", -(dist.log_prob(safe) * legal).sum(), nl)
    m.add("bc/nll_type", -(dist.type.log_prob(torch.where(legal, t, 0)) * legal).sum(), nl)
    m.add("bc/type_acc", ((dist.type.logits.argmax(1) == t) & legal).sum(), nl)
    m.add("bc/act_rate_human", ((t > 0) & legal).sum(), nl)
    m.add("bc/act_rate_policy", ((1 - dist.type.probs[:, 0]) * legal).sum(), nl)
    lc = dist.cells.log_prob(cell[:, None].expand(-1, 3))  # [N, maps]
    rows = legal & (t == ATTACK)
    m.add("bc/attack_target_acc", ((dist.ptr.logits.argmax(1) == tgt) & rows).sum(), rows.sum())
    m.add("bc/nll_attack_target", -(dist.ptr.log_prob(tgt) * rows).sum(), rows.sum())
    size = SIZE_OF.to(b.device)[b.clamp(0, SHARES)]  # the label's percent as one of the five sizes
    ls = dist.size_log_probs(tgt, cell)  # [N, attack/boat, 5]
    lt = dist.troop_log_prob(tgt, cell, b) if dist.shares else None  # [N, attack/boat]
    for k, (typ, name) in enumerate(((ATTACK, "attack"), (BOAT, "boat"))):
        rows = legal & (t == typ)
        m.add(f"bc/{name}_bin_acc", ((ls[:, k].argmax(1) == size) & rows).sum(), rows.sum())
        m.add(f"bc/nll_{name}_bin", -(ls[:, k].gather(1, size[:, None])[:, 0] * rows).sum(), rows.sum())
        if lt is not None:
            m.add(f"bc/nll_{name}_share", -(lt[:, k] * rows).sum(), rows.sum())
    for typ, name in ((SPAWN, "spawn"), (BOAT, "boat"), (BUILD, "city")):
        rows = legal & (t == typ)
        hit = dist.cells.logits[:, CELL_HEAD[typ]].argmax(1) == cell
        m.add(f"bc/{name}_cell_acc", (hit & rows).sum(), rows.sum())
        m.add(f"bc/nll_{name}_cell", -(lc[:, CELL_HEAD[typ]] * rows).sum(), rows.sum())
```

The buckets and `bc/value_mse` that follow stay unchanged, and they use `logp`.

In `score()` and `validate()`, `_, dist, logp, value, gates = bc_loss(...)` becomes `_, dist, value, gates =
bc_loss(...)`, and `measure(m, obs, labels, legal, dist, logp, value, target, gates)` becomes `measure(m, obs,
labels, legal, dist, value, target, gates)`.

- [ ] **Step 4: Run all the trainer tests**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . 2>&1 | tail -4`
Expected: OK, 98 tests. `AccumulationTest` runs a v2 network: smoothing does not touch the five sizes, so both sides
agree.

- [ ] **Step 5: Commit**

```bash
git add trainer/bc.py trainer/test_bc.py
git commit -m "BC on troop shares (PLAN 3B): soft labels (0.9 on the human's percent, 0.05 on each neighbour); bc/nll and the troop heads' NLL are like for like for every network (a share network's shares summed into the five sizes' ranges), and a share network also reports its own 100-share NLL"
```

- [ ] **Step 6: Run the like-for-like regression check (R1, ~10 min GPU; ask the user first)**

Run: `.venv/Scripts/python -m trainer.bc --score checkpoints/bc3-moe/latest.pt`
Expected: `bc/nll` within 0.003 of 0.8430, the 3A clone's score on the same rows (`runs/score/20261005-034226.json`;
the purge left the 766 scoring records alone). Only labels whose exact share and rounded percent fall on different
sides of a size boundary changed. A larger gap means the label change broke something: stop and debug before the
night.

---

### Task 5: SwiGLU experts, the `moe-swiglu` core

**Files:**
- Modify: `trainer/model.py` (`SoftMoE`, `Policy` core, `CORES`, `CORE_NAMES`), `trainer/bc.py` and `trainer/ppo.py`
  (`--core` choices)
- Test: `trainer/test_model.py`

**Interfaces:**
- Consumes (Task 3): `Policy(..., shares=...)`, `CORE_NAMES`.
- Produces:
  - `SoftMoE(d, experts=8, hidden=None)`; with `hidden`, the parameters `gate`, `up` ([experts, hidden, d]) and
    `down` ([experts, d, hidden]);
  - core `"moe-swiglu"`;
  - `CORES = ("dense", "moe", "moe-swiglu")`.

- [ ] **Step 1: Write the failing tests**

`trainer/test_model.py`:
- Import `torch.nn.functional as F`, and import `CORES` from `trainer.model`.
- `nets()` gains `"swiglu": Policy(DIMS, core="moe-swiglu", shares=True)`, and `test_output_shapes` checks `shares`
  for it: `if name in ("v3", "swiglu")`.
- Replace `test_the_two_cores_start_alike_everywhere_else` with:

```python
    def test_the_cores_start_alike_everywhere_else(self):
        states = {}
        for core in CORES:
            torch.manual_seed(3)
            states[core] = Policy(DIMS, core=core, shares=True).state_dict()
        for core in CORES[1:]:
            for k, v in states["dense"].items():
                if not k.startswith("core."):
                    self.assertTrue(torch.equal(v, states[core][k]), (core, k))
```

Add to `SoftMoETest`:

```python
    def test_swiglu_experts_are_two_layer_networks_blended_by_the_router(self):
        torch.manual_seed(0)
        moe = SoftMoE(16, experts=4, hidden=32)
        torch.nn.init.normal_(moe.router.weight)
        x = torch.randn(5, 16)
        y, gates = moe(x)
        experts = torch.stack([(F.silu(x @ moe.gate[e].T) * (x @ moe.up[e].T)) @ moe.down[e].T for e in range(4)], 1)
        self.assertTrue(torch.allclose(y, (gates[..., None] * experts).sum(1), atol=1e-5))

    def test_the_swiglu_experts_hold_plans_12_6m_parameters(self):
        moe = SoftMoE(512, hidden=1024)
        self.assertEqual(sum(p.numel() for n, p in moe.named_parameters() if not n.startswith("router")),
                         8 * 3 * 512 * 1024)  # 12,582,912
```

In `LoadTest`:
- Add `({"arch": "v3", "core": "moe-swiglu", "widths": "64,128,192"}, Policy)`.
- The `moe` assertion becomes `self.assertEqual(getattr(net, "moe", False), args.get("core", "dense") != "dense")`.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model 2>&1 | tail -5`
Expected: `ImportError: cannot import name 'CORES'`.

- [ ] **Step 3: Implement**

`trainer/model.py`: `SoftMoE` becomes:

```python
class SoftMoE(nn.Module):
    """Soft-MoE (PLAN decision 5c) for one vector per row. With a single token, every expert's slot receives the row's
    vector, and the combine weights are a softmax over the experts: the layer blends the experts' outputs by
    situation. Each expert is one d x d layer, or with `hidden` a 2-layer SwiGLU network d -> hidden -> d without
    biases (PLAN 3B, the brief's idea F). Running them all costs little next to the CNN. More experts can be added
    later by copying trained ones (upcycling)."""

    def __init__(self, d, experts=8, hidden=None):
        super().__init__()
        self.router = nn.Linear(d, experts)
        nn.init.zeros_(self.router.weight)  # every expert weighs the same at the start
        nn.init.zeros_(self.router.bias)
        self.hidden = hidden
        if hidden is None:
            layers = [nn.Linear(d, d) for _ in range(experts)]
            self.w = nn.Parameter(torch.stack([layer.weight.detach() for layer in layers]))  # [experts, out, in]
            self.b = nn.Parameter(torch.stack([layer.bias.detach() for layer in layers]))  # [experts, out]
        else:
            stack = lambda i, o: nn.Parameter(torch.stack([nn.Linear(i, o, bias=False).weight.detach()
                                                           for _ in range(experts)]))  # [experts, out, in]
            self.gate, self.up, self.down = stack(d, hidden), stack(d, hidden), stack(hidden, d)

    def forward(self, x):
        gates = self.router(x).softmax(-1)  # [N, experts]
        if self.hidden is None:
            y = torch.einsum("ni,eoi->neo", x, self.w) + self.b  # every expert's output [N, experts, out]
        else:
            z = F.silu(torch.einsum("ni,eoi->neo", x, self.gate)) * torch.einsum("ni,eoi->neo", x, self.up)
            y = torch.einsum("nei,eoi->neo", z, self.down)
        return torch.einsum("ne,neo->no", gates, y), gates
```

Add `CORES = ("dense", "moe", "moe-swiglu")` below `ATTENDED`. In `Policy.__init__`, the last two lines become:

```python
        self.moe = core != "dense"
        self.core = SoftMoE(512, hidden=1024 if core == "moe-swiglu" else None) if self.moe else nn.Linear(512, 512)
```

The `Policy` docstring's core line becomes: `- core: "dense" (one 512 x 512 layer), "moe" (a Soft-MoE layer of 8
experts in its place) or "moe-swiglu" (8 SwiGLU experts, 512 -> 1024 -> 512).`

`CORE_NAMES` gains `"moe-swiglu": "Soft-MoE core (8 SwiGLU experts)"`.

`trainer/bc.py` and `trainer/ppo.py`:
- Import `CORES`.
- `p.add_argument("--core", choices=("dense", "moe"), ...)` becomes `choices=CORES`.

- [ ] **Step 4: Run all the trainer tests**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . 2>&1 | tail -4`
Expected: OK, 100 tests.

- [ ] **Step 5: Commit**

```bash
git add trainer
git commit -m "The moe-swiglu core (PLAN 3B, the brief's idea F): each of the 8 Soft-MoE experts is a 2-layer SwiGLU network 512 -> 1024 -> 512 without biases (12.6M parameters); every core starts from the same weights outside the core for a given seed"
```

---

### Task 6: Two networks in one BC run (`--twin`)

**Files:**
- Modify: `trainer/bc.py` (parser, `run_info`, `backward_windows`, `main`), `trainer/control.py` (loss chart)
- Test: `trainer/test_bc.py`

**Interfaces:**
- Consumes (Task 4): `bc_loss(..., smooth)`, `measure(...)`; (Task 5) `CORES`; (Task 3) `describe`.
- Produces:
  - `trainer.bc --twin CORE`;
  - `backward_windows(models, windows, a, meters)`;
  - the twin's checkpoints in `checkpoints/<run>-<twin>/` (args `run=<run>-<twin>`, `core=<twin>`, `twin=None`) and
    its curves in `runs/<run>-<twin>/`;
  - the run's own TensorBoard gets `twin/bc/nll` and `twin/val/bc/nll`.

- [ ] **Step 1: Write the failing tests**

`trainer/test_bc.py`:
- `AccumulationTest` calls `backward_windows([model], windows, argparse.Namespace(accum=3, vf_coef=0.5),
  [Meter()])`.
- In `MainTest`:
  - `tearDown` also removes the twin's folders: `for run in (self.RUN, f"{self.RUN}-moe-swiglu")`, for both
    `checkpoints` and `runs`.
  - `latest` becomes:

```python
    def latest(self, run=None):
        return torch.load(ROOT / "checkpoints" / (run or self.RUN) / "latest.pt", weights_only=False)
```

Add to `MainTest`:

```python
    def test_a_twin_starts_from_the_same_weights_outside_its_core(self):
        self.run_main(self.common + ["--updates", "0", "--twin", "moe-swiglu"])
        main, twin = self.latest(), self.latest(f"{self.RUN}-moe-swiglu")
        self.assertEqual((main["args"]["core"], twin["args"]["core"]), ("moe", "moe-swiglu"))
        self.assertEqual((twin["args"]["run"], twin["args"]["twin"]), (f"{self.RUN}-moe-swiglu", None))
        for k, v in main["model"].items():
            if not k.startswith("core."):
                self.assertTrue(torch.equal(v, twin["model"][k]), k)

    def test_a_twin_trains_on_the_same_windows_and_resumes_with_the_run(self):
        self.run_main(self.common + ["--updates", "2", "--twin", "moe-swiglu"])
        main, twin = self.latest(), self.latest(f"{self.RUN}-moe-swiglu")
        self.assertEqual((twin["update"], twin["global_step"]), (2, main["global_step"]))
        self.assertTrue((ROOT / "runs" / f"{self.RUN}-moe-swiglu").is_dir())
        self.run_main(self.common + ["--updates", "3", "--resume"])
        self.assertEqual((self.latest()["update"], self.latest(f"{self.RUN}-moe-swiglu")["update"]), (3, 3))
        (ROOT / "checkpoints" / f"{self.RUN}-moe-swiglu" / "latest.pt").unlink()
        with self.assertRaisesRegex(SystemExit, "is missing"):
            self.run_main(self.common + ["--updates", "4", "--resume"])
        with self.assertRaisesRegex(SystemExit, "drop --init"):
            self.run_main(self.common + ["--init", str(ROOT / "checkpoints" / self.RUN / "latest.pt"), "--twin", "moe",
                                         "--updates", "1"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v 2>&1 | tail -6`
Expected: FAIL.
- The new tests stop with `SystemExit: 2` (argparse: unrecognized `--twin`).
- `AccumulationTest` fails with `TypeError` (a list where a model is expected).

- [ ] **Step 3: Implement**

`trainer/bc.py`: the parser gains, after `--core`:

```python
    p.add_argument("--twin", choices=CORES,
                   help="also train a second new network with this core on the same windows, row by row, from the same "
                        "weights outside the core (PLAN 3B); it saves to checkpoints/<run>-<twin>")
```

In `run_info`, the "Network" value becomes
`describe(vars(a)) + (f"; twin: {describe({**vars(a), 'core': a.twin})}" if a.twin else "")`.

`backward_windows`:

```python
def backward_windows(models, windows, a, meters):
    """Accumulates each network's gradient over a.accum windows, each (obs, labels, legal rows, value targets): their
    mean loss gives the gradient of one batch of all their rows. Every network trains on every window (--twin)."""
    for obs, labels, legal, target in windows:
        for model, meter in zip(models, meters):
            loss, dist, value, gates = bc_loss(model, obs, labels, legal, target, a.vf_coef, SHARE_SMOOTH)
            (loss / a.accum).backward()
            measure(meter, obs, labels, legal, dist, value, target, gates)
```

`main`:
- Right after `if a.score: ...`, add:

```python
    if a.twin and a.init:
        raise SystemExit("--twin trains two new networks from the same random weights: drop --init")
```

- In the `if a.resume:` branch, after `a = resume_args(...)`, add:

```python
        if a.twin:
            path = ROOT / "checkpoints" / f"{a.run}-{a.twin}" / "latest.pt"
            if not path.exists():
                raise SystemExit(f"{path} is missing: this run also trains a {a.twin} twin, and both resume together")
            twin_state = torch.load(path, map_location="cpu", weights_only=False)
            if twin_state["update"] != state["update"]:
                raise SystemExit(f"the twin's checkpoint is at update {twin_state['update']} and the run's at "
                                 f"{state['update']}: both must resume from the same update")
```

  Initialise `state = twin_state = None` where `state = None` is today.
- Replace the block from `model = new_net(envs.dims, vars(a)).to(device)` through `opt = torch.optim.Adam(...)` with:

```python
    torch.manual_seed(a.seed)  # again before each network: a twin starts from the same weights outside its core
    nets = [new_net(envs.dims, vars(a)).to(device).to(memory_format=torch.channels_last)]
    net_args, dirs, writers = [vars(a)], [ckpt_dir], [SummaryWriter(run_dir)]
    if a.twin:
        torch.manual_seed(a.seed)
        twin_run = f"{a.run}-{a.twin}"
        net_args.append({**vars(a), "run": twin_run, "core": a.twin, "twin": None})
        nets.append(new_net(envs.dims, net_args[1]).to(device).to(memory_format=torch.channels_last))
        dirs.append(ROOT / "checkpoints" / twin_run)
        writers.append(SummaryWriter(ROOT / "runs" / twin_run))
    model, writer = nets[0], writers[0]
    opts = [torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5) for net in nets]
    meters = [Meter() for _ in nets]
```

- Delete the later `writer = SummaryWriter(run_dir)` line. In `meter, done = Meter(), {...}`, keep only `done = {...}`.
- When resuming, load every network:

```python
    if state:
        for net, opt, s in zip(nets, opts, (state, twin_state)):
            net.load_state_dict(s["model"])
            opt.load_state_dict(s["optimizer"])
        update, global_step = state["update"], state["global_step"]
```

  The RNG restore lines follow unchanged.
- `checkpoint()`:

```python
    def checkpoint():
        """Every network's checkpoint, at the same update."""
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all()}
        for net, opt, args, d in zip(nets, opts, net_args, dirs):
            s = dict(kind="bc", model=net.state_dict(), optimizer=opt.state_dict(), args=args, update=update,
                     global_step=global_step, game_commit=commit, rng=rng)
            save_checkpoint(d / "latest.pt", **s)
            if update % (10 * a.save_every) == 0 and update > 0:
                save_checkpoint(d / f"update{update:07d}.pt", **s)
        control.note(f"saved checkpoint at update {update}")
```

- `validate()` scores every network on the same windows and returns one dict per network:

```python
    def validate():
        """Every network's held-out statistics, on the same windows."""
        ms, values, outcomes = [Meter() for _ in nets], [[] for _ in nets], []
        with torch.no_grad():
            for _ in range(a.val_steps):
                obs, labels = val_envs.obs, val_envs.labels
                target = value_targets(val_envs, a.gamma, device)
                legal = label_legal(obs, labels)
                for net, m, v in zip(nets, ms, values):
                    _, dist, value, gates = bc_loss(net, obs, labels, legal, target, a.vf_coef)
                    measure(m, obs, labels, legal, dist, value, target, gates)
                    v.append(value.cpu().numpy())
                outcomes.append(val_envs.outcome.reshape(-1).copy())
                val_envs.step()
        o, results = np.concatenate(outcomes), []
        for m, v in zip(ms, values):
            out, v = m.means("val/"), np.concatenate(v)
            if v.std() > 0 and o.std() > 0:
                out["val/value_outcome_corr"] = float(np.corrcoef(v, o)[0, 1])
            results.append(out)
        return results
```

- The training step:

```python
            t0 = time.perf_counter()
            for opt in opts:
                for group in opt.param_groups:
                    group["lr"] = lr_at(update, a)
                opt.zero_grad(set_to_none=True)
            backward_windows(nets, windows(), a, meters)
            for net, opt in zip(nets, opts):
                nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
                opt.step()
            update += 1
```

- Logging:
  - `metrics = meter.means()` becomes `metrics = meters[0].means()`;
  - `"train/lr": opt.param_groups[0]["lr"]` becomes `opts[0].param_groups[0]["lr"]`;
  - after `for k, v in metrics.items(): writer.add_scalar(k, v, global_step)`, add:

```python
                if a.twin:
                    twin = meters[1].means()
                    for k, v in twin.items():
                        writers[1].add_scalar(k, v, global_step)
                    if "bc/nll" in twin:
                        writer.add_scalar("twin/bc/nll", twin["bc/nll"], global_step)  # the control page's loss chart
```

- Validation:

```python
            if val_envs is not None and update % a.val_every == 0:
                results = validate()
                for w, metrics in zip(writers, results):
                    for k, v in metrics.items():
                        w.add_scalar(k, v, global_step)
                if a.twin and "val/bc/nll" in results[1]:
                    writer.add_scalar("twin/val/bc/nll", results[1]["val/bc/nll"], global_step)
```

- In `finally`, `writer.close()` becomes `for w in writers: w.close()`.

`trainer/control.py`, the BC loss chart: `lines: [["bc/nll", "training"], ["val/bc/nll", "held-out"]]` becomes
`lines: [["bc/nll", "training"], ["val/bc/nll", "held-out"], ["twin/bc/nll", "twin, training"],
["twin/val/bc/nll", "twin, held-out"]]`. Lines without data are already hidden.

- [ ] **Step 4: Run all the trainer tests**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . 2>&1 | tail -4`
Expected: OK, 102 tests.

- [ ] **Step 5: Commit**

```bash
git add trainer/bc.py trainer/control.py trainer/test_bc.py
git commit -m "trainer.bc --twin CORE (PLAN 3B): a second new network trains on the same windows, row by row, from the same weights outside its core, with its own optimizer, checkpoints (checkpoints/<run>-<core>) and curves; both resume together or not at all, and the control page's loss chart shows the twin"
```

---

### Task 7: The speed check for the pair

**Files:**
- Modify: `scripts/speed_check.py`

**Interfaces:**
- Consumes: `--twin` (Task 6) and the v3 networks (Tasks 3 and 5).

This script orchestrates other commands and has no unit test. Its check is running it (R2).

- [ ] **Step 1: Rewrite the run set and the report**

The docstring:

```python
"""Speed and memory check (docs/PLAN.md, 3B): PPO decisions/s and peak GPU memory for the Phase 2 network and the
Phase 3B networks (100 troop shares; MoE and SwiGLU-expert cores, random weights: speed does not depend on them), then
BC rows/s for one network and for two trained on the same windows (--twin). Run it with the PC otherwise idle (~20 min,
full GPU):

  .venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12
"""
```

In `main`, replace the checkpoint loop and `runs`:

```python
    for core in ("moe", "moe-swiglu"):
        args = {"arch": "v3", "core": core, "widths": "64,128,192", "run": f"speed-{core}"}
        torch.manual_seed(0)
        torch.save({"model": new_net(DIMS, args).state_dict(), "args": args, "update": 0}, speed / f"{core}.pt")
```

```python
    moe, swiglu = str(speed / "moe.pt"), str(speed / "moe-swiglu.pt")
    runs = {"speed-ppo-v1": ("trainer.ppo", ppo + ["--init", str(PHASE2), "--opponent", str(PHASE2)]),
            "speed-ppo-moe": ("trainer.ppo", ppo + ["--init", moe, "--opponent", moe]),
            "speed-ppo-moe-swiglu": ("trainer.ppo", ppo + ["--init", swiglu, "--opponent", swiglu]),
            "speed-bc-one": ("trainer.bc", bc + ["--core", "moe"]),
            "speed-bc-pair": ("trainer.bc", bc + ["--core", "moe", "--twin", "moe-swiglu"])}
```

The cleanup loop also removes the pair's twin:

```python
        for name in runs:
            for d in (name, f"{name}-moe-swiglu"):
                shutil.rmtree(ROOT / "checkpoints" / d, ignore_errors=True)
                shutil.rmtree(ROOT / "runs" / d, ignore_errors=True)
```

After the per-run lines, print the verdicts:

```python
    ppo_moe, ppo_swiglu = results["speed-ppo-moe"][0], results["speed-ppo-moe-swiglu"][0]
    print(f"\nPPO, SwiGLU experts: {ppo_swiglu / ppo_moe:.0%} of the MoE core's decisions/s (keep rule: at most 5% slower)")
    if "speed-bc-pair" in results:
        pair = results["speed-bc-pair"][0] / results["speed-bc-one"][0]
        print(f"BC, two networks: {pair:.0%} of one network's rows/s (PLAN 3B: >= 90% to train both in one night)")
```

- [ ] **Step 2: Check that it imports and parses**

Run: `.venv/Scripts/python scripts/speed_check.py --help`
Expected: the usage text, with no error.

- [ ] **Step 3: Commit**

```bash
git add scripts/speed_check.py
git commit -m "Speed check for 3B: PPO decisions/s and peak VRAM of the Phase 2 network and the 3B networks (MoE and SwiGLU experts), BC rows/s of one network and of a twin pair, with the PLAN's two bars (pair >= 90% of one, SwiGLU at most 5% slower in PPO)"
```

- [ ] **Step 4: Run it (R2, ~20 min with the PC idle; ask the user first)**

Run: `.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12`
Expected:
- the pair keeps ≥ 90% of one network's rows/s. If not, the twin gets its own night (R3, second form);
- no PPO run's peak exceeds the card's 12,282 MiB (watch "Shared GPU memory" too). The share tables add
  [rows, 2, 403, 100] floats per minibatch;
- note the PPO speeds for the experts' keep rule and for Gate 3B (≥ 75% of Phase 2's 190 decisions/s).

---

### Task 8: Docs

**Files:**
- Modify: `CLAUDE.md`, `docs/PLAN.md`; memory `openfront-rl-project.md`

- [ ] **Step 1: CLAUDE.md**

Benchmarks: the `speed_check.py` line becomes:

```
.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12   # PPO decisions/s and peak VRAM (Phase 2, 3B MoE, 3B SwiGLU experts), BC rows/s of one network and of a twin pair (~20 min, PC idle)
```

Training commands: below the `bc3-moe` line, add:

```
.venv/Scripts/python -m trainer.bc --run bc3b --core moe --twin moe-swiglu --workers 14 --games-per-worker 6   # two new networks on the same windows (PLAN 3B): checkpoints/bc3b and checkpoints/bc3b-moe-swiglu
```

The "Networks:" bullet's first sentence becomes "Networks: new runs record theirs in the checkpoint (`args["arch"]`,
`"core"`): v1 is Phase 2, v2 is Phase 3A (5 troop sizes), v3 is Phase 3B (100 troop shares); `--init` keeps the
checkpoint's." Add a bullet after it:

```
- Troops are a whole percent of current troops (1–100) in actions and labels. v1 and v2 networks send
  10/25/50/75/100%, exactly what their five bins sent. v3 networks read the share with a player's token: the attack's
  target, or the main owner of the boat's landing cell.
  - BC labels the human's share, rounded and clamped to 1–100. The loss puts 0.05 on each neighbouring percent
    (`SHARE_SMOOTH`).
  - In `--score`, `bc/nll` is like for like (a share network's shares summed into the five sizes' ranges, 1–15, 16–35,
    36–61, 62–86 and 87–100%), and `bc/nll_shares` is a share network's own NLL.
  - `--twin CORE` trains a second network on the same windows, from the same weights outside the core. It saves to
    `checkpoints/<run>-<core>`, and the run resumes both together.
```

- [ ] **Step 2: docs/PLAN.md**

In 3B, "Troop shares read with the target", add after "The 3A nights keep the five sizes: ...":

```
    - **Code (2026-10-06)**, `docs/superpowers/plans/2026-10-06-phase3b-troop-shares.md`:
      - actions and labels carry a whole percent (1–100); the 5-size networks send 10/25/50/75/100%, the same troops
        as before;
      - the share head (arch v3): per kind, a 64-wide hidden layer reads the core state and one player's token, then
        100 logits. It runs on every token, so the entropy and the KL to the clone stay exact;
      - soft labels: 0.9 on the human's percent, 0.05 on each neighbour;
      - scores: `bc/nll` like for like (shares summed into the five sizes' ranges: 1–15, 16–35, 36–61, 62–86 and
        87–100%), and `bc/nll_shares` at full resolution;
      - `trainer.bc --twin moe-swiglu` trains the experts' network on the same windows, from the same weights outside
        the core (`checkpoints/<run>-moe-swiglu`).
    - Rules fixed before the night (2026-10-06):
      - speed check: the pair keeps ≥ 90% of one network's rows/s, or the twin gets its own night;
      - shares: they stay unless they lose the 512 games to the 3A clone by more than 2 SE (below ~0.456). A loss that
        large points to a bug, which is found before the rest of 3B;
      - experts: the keep rule above, on `bc/nll_shares` (both networks have 100 shares) and the speed check's PPO
        runs.
```

The results go in after R4 and R5 (below).

- [ ] **Step 3: Memory**

In `memory/openfront-rl-project.md`, replace the "**Next:** 3B step 2 ..." lines with the state at that point:
- the code is done (commit range);
- the R1 and R2 results;
- the night's command and its run name `bc3b`;
- next: R3 to R5.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/PLAN.md
git commit -m "Docs for 3B step 2: troops as percents, the v3 share head, like-for-like scores, --twin, the speed check, and the rules fixed before the night"
```

---

## Runs (each needs the user's go)

| # | When | Command | Time | Read / rule |
|---|---|---|---|---|
| R1 | after Task 4 | `.venv/Scripts/python -m trainer.bc --score checkpoints/bc3-moe/latest.pt` | ~10 min | `bc/nll` within 0.003 of 0.8430 |
| R2 | after Task 7 | `.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12` | ~20 min, PC idle | pair ≥ 90% of one; peak VRAM; SwiGLU PPO speed |
| R3 | after Task 8 | the night, below | ~14 h | both checkpoints reach the last update |
| R4 | after R3 | `.venv/Scripts/python -m trainer.bc --score checkpoints/bc3-moe/latest.pt checkpoints/bc3b/latest.pt checkpoints/bc3b-moe-swiglu/latest.pt` | ~10 min | shares vs the 3A clone on `bc/nll`, `bc/nll_attack_bin`, `bc/nll_boat_bin`; experts vs shares alone on `bc/nll_shares` (keep unless shares alone is lower by > 0.02, or SwiGLU PPO > 5% slower) |
| R5 | after R4 | `.venv/Scripts/python -m trainer.eval --a checkpoints/bc3b/latest.pt --b checkpoints/bc3-moe/latest.pt --episodes 512 --workers 16 --games-per-worker 2` | ~1 h | the shares stay unless below ~0.456 (2 SE under 0.5) |

R3 runs in the user's own PowerShell (repo root). It uses a restart loop for the GPU driver's occasional failure:

```powershell
.venv/Scripts/python -m trainer.bc --run bc3b --core moe --twin moe-swiglu --workers 14 --games-per-worker 6
for ($i = 1; $i -le 5 -and $LASTEXITCODE -ne 0; $i++) { Start-Sleep 60; .venv/Scripts/python -m trainer.bc --run bc3b --resume }
```

If R2's pair falls below 90%, the night runs without `--twin`. A second night then trains
`--run bc3b-moe-swiglu --core moe-swiglu` with the same seed: each run seeds before building its network, so the two
still start alike outside the core and replay the same records in the same order.

After R4 and R5, write the results in PLAN 3B ("Results"), apply the rules, and update the memory. Then the rest of
3B (step 3: the actions, observations and extractor).
