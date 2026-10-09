# Phase 3B, Step 3 (Part 2): What StockShip Sees (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** StockShip sees what its new actions act on:
- each structure's level, and the missiles loaded in each silo and SAM;
- where SAMs, defense posts and nukes in flight reach, on its own side and on the enemy's;
- the gold it holds against each build's cost, and its loaded silo missiles.
The networks before keep their inputs. Part 1 took the heat maps from 3 to 12, so v4's speed and memory are measured
first.

**Architecture:**
- **Measure first** (Tasks 1–2). The v4 network's PPO speed and peak memory are checked against a rule fixed below. If
  v4 leaves no room for Part 2's inputs, the PPO rollouts keep their cell masks as bits, in an eighth of the memory.
- **Bridge** (Tasks 4–5).
  - Each unit record adds the missiles loaded (silos and SAMs) and a zone: a centre and a radius in tiles.
    - A finished SAM: its current range around it.
    - A finished defense post: its range around it.
    - A nuke in flight: its outer blast around its target.
  - The scalars add gold against each build's current cost, and the seat's loaded silo missiles.
- **Trainer** (Tasks 3, 6).
  - 8 channels after the 49: level, loaded missiles, and three zones for each side. The zones are SAM interceptors,
    defense posts and nukes in flight; the sides are ours and theirs. The GPU paints the channels from the records.
  - v4 networks read the 57 channels and 26 scalars. v1–v3 networks read their 49 and 16.
- **Scores** (Task 7). Each heat map's closeness gets a baseline: a uniform pick over the legal cells.

**Tech Stack:** TypeScript bridge (node:test via tsx), Python 3 + torch (unittest).

**Spec:** `docs/PLAN.md`, read with this list:
- 3B "Observations", and decision 4 (the coverage channels);
- the "Nukes and SAMs" and "Structures" findings;
- Gate 3B's speed bar.

The BC refresh (3B step 4) comes after this list.

**Status (2026-10-09):** written, waiting for the user's review.

## Global Constraints

- Never edit `OpenFront_Source_Code/openfrontio/`. Bridge code lives in `bridge/`, trainer code in `trainer/`, scripts
  in `scripts/`.
- Rule values come from the game's `Config` at runtime, never hard-coded:
  - `config.dynamicSamRange(sam, game.ticks())`;
  - `config.defensePostRange()`;
  - `config.nukeMagnitudes(type).outer`;
  - `game.unitInfo(type).cost(game, player)`.
- Spatial inputs reach the model as uint8 in 64ths (`quantize_`), so a channel holds 0 to 255/64.
- v1–v3 networks must play unchanged. They read only channels 0–48 and scalars 0–15.
- No `Co-Authored-By` line or any other Claude attribution in commits.
- Speed checks are heavy runs. Ask the user first, and run them with the PC otherwise idle.
- Run bridge commands from `OpenFront_Source_Code/openfrontio/`, trainer commands from the repo root.

## Game facts this plan relies on (checked in v0.34.23)

- **SAMs and silos:**
  - loaded missiles = `level() - missileTimerQueue().length`;
  - `increaseLevel()` adds the new slot to the reload queue, so the slot starts empty, and `launch()` queues one more;
  - `config.dynamicSamRange(sam, tick)` ramps from the old level's range to the new one's over
    `samUpgradeDuration()` ticks after an upgrade. `samRange(1)` is 70 tiles;
  - a SAM intercepts any nuke whose owner is neither the SAM's owner nor a friend
    (`SAMLauncherExecution.isValidNukeTarget`). So a friend's SAM protects the whole side.
- **Defense posts:** a conquered tile gets the defender's bonus when a finished, active post of the defender lies
  within `defensePostRange()` (`AttackExecution`, `hasUnitNearby`).
- **Nukes:**
  - `targetTile()` is where the nuke lands;
  - `nukeMagnitudes(type)`, inner / outer radius: atom bomb 12 / 30, hydrogen bomb 80 / 100, MIRV warhead 12 / 18.
    The MIRV carrier has none (it throws);
  - a blast deletes the buildings within its outer radius.
- **Construction:** `ConstructionExecution` calls `buildUnit`, then `setUnderConstruction(true)`.
  - A unit made by `player.buildUnit` alone is finished and has no execution, so it stays put and never reloads. The
    tests build their units this way.
  - `buildUnit(UnitType.AtomBomb, tile, { targetTile, trajectory: [] })` makes a nuke with a target.
- **Grid:**
  - `frame.cellOf(tile)` is floor(y / cs) · W + floor(x / cs);
  - the trainer knows each game's tiles per cell as `env.area` = cs² (64 on Normal maps, 16 on Compact);
  - structures stay at least 15 tiles apart and a cell is at most 8 × 8 tiles, so a cell holds one structure at most.
    One level channel and one missiles channel therefore hold every structure.

## Decisions (for the user's review)

1. **Zones by side, not by relation.** PLAN 3B says SAM coverage counts "by relation" (self, friendly, enemy, bot).
   This list uses two sides: ours (self and friends) and theirs (enemies and bots).
   - In 1v1 the two are the same: self, and the enemy.
   - Bots never build SAMs or posts, and never launch nukes.
   - A friend's SAM shoots down enemy nukes over our land too (the game fact above), so it belongs to our side.
   - It saves 3 channels, ~225 MB of PPO rollouts.
2. **Scales.**
   - Level and loaded missiles are / 16, exact up to 63. The old channels' level saturates at 3.98, and they stay as
     they are for v1–v3.
   - SAM zones: the interceptors loaded over the cell, / 16.
   - Defense-post and nuke zones: 1 per post or nuke.
3. **Under construction** is a structure whose level channel reads 0 while its type channel reads 1. It gets no
   channel of its own.
4. **Unit costs** are scalars of gold over each build's current cost, capped at 4 and divided by 4, as scalar 8
   already does for cities. The rule values (SAM ranges, blast radii) reach the network through the zones' radii, so
   they need no scalars.
5. **v4 is redefined** to read the new inputs. No v4 checkpoint exists yet: Part 1 ran no training.
6. **A cell is inside a zone when its centre lies within the radius**, by Euclidean distance in tiles.
   - Positions are doubled so the arithmetic stays in integers, and the trainer and its test agree exactly.
   - Zones also cover padding cells. The network has the VALID channel.

## Review Focus

1. **A Compact map (4 × 4 tiles per cell) in the same batch as a Normal one.** Each zone must use its own game's cell
   side. Tested by Task 6's `DiskTest`.
2. **A zone at the map's corner or edge.** Its disk is clipped to the grid, and no index wraps into the next row.
   Tested by Task 6's `DiskTest`, with a disk at tile 0, 0.
3. **No zone in the batch** (the spawn phase). Painting does nothing and raises nothing. Tested by Task 6's
   `DiskTest`.
4. **A v1–v3 checkpoint saved before this list, loaded with today's wider inputs.** It loads, and its outputs ignore
   the new inputs. Tested by Task 3's tests.
5. **Rollout masks stored as bits.** Every minibatch gets the exact masks back. Tested by Task 2's `PackTest`, and
   end to end by test_ppo's runs.

---

### Task 1: Measure v4's PPO speed and memory

**Files:**
- Modify: `scripts/speed_check.py`
- Modify: `CLAUDE.md` (its speed-check line), `docs/PLAN.md` (3B, the results)

**Interfaces:**
- Consumes: Part 1's v4 network (`new_net(DIMS, {"arch": "v4", "core": "moe-swiglu", ...})`), `trainer.ppo` and
  `trainer.bc`. New runs are v4 (`trainer.ppo.network_args`).
- Produces: the rule's verdict, which decides whether Task 2 runs.

**The rule, fixed before the run.**
- own = the `speed-ppo-v4` peak minus the GPU memory used before the runs (other programs), in MiB.
- Task 2 is skipped only if both of these hold:
  - own ≤ 9,400 MiB:
    - the card has 12,282 MiB;
    - other programs held 2,100 MiB in the 3B check;
    - Part 2's 8 channels need ~800 MiB (1,536 rollout rows × 8 × 49,152 bytes, plus each minibatch's input);
  - v4 makes at least 75% of the v1 arm's decisions/s in the same run (Gate 3B's bar).
- Expected: Task 2 runs.
  - v3 used ~9,350 MiB on its own in the 3B check.
  - v4's rollouts hold 12 cell masks instead of 3 (+680 MiB), and each minibatch adds 12 maps' logits.

- [ ] **Step 1: Point the check at v4**

In `scripts/speed_check.py`, replace the docstring's first paragraph with:

```python
"""Speed and memory check (docs/PLAN.md, 3B). It measures:
- PPO decisions/s and peak GPU memory for the Phase 2 network and a v4 network. The v4 network has every action and
  the SwiGLU experts the BC refresh trains, with random weights: speed does not depend on them;
- BC rows/s for the v4 network.
Gate 3B: v4 keeps at least 75% of the Phase 2 network's decisions/s in the same run. Run it with the PC otherwise idle
(~15 min, full GPU):

  .venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12
"""
```

Replace the two v3 checkpoints:

```python
    for core in ("moe", "moe-swiglu"):
        args = {"arch": "v3", "core": core, "widths": "64,128,192", "run": f"speed-{core}"}
        torch.manual_seed(0)
        torch.save({"model": new_net(DIMS, args).state_dict(), "args": args, "update": 0}, speed / f"{core}.pt")
```

with

```python
    args = {"arch": "v4", "core": "moe-swiglu", "widths": "64,128,192", "run": "speed-v4"}
    torch.manual_seed(0)
    v4 = str(speed / "v4.pt")
    torch.save({"model": new_net(DIMS, args).state_dict(), "args": args, "update": 0}, v4)
```

Replace the runs:

```python
    moe, swiglu = str(speed / "moe.pt"), str(speed / "moe-swiglu.pt")
    runs = {"speed-ppo-v1": ("trainer.ppo", ppo + ["--init", str(PHASE2), "--opponent", str(PHASE2)]),
            "speed-ppo-moe": ("trainer.ppo", ppo + ["--init", moe, "--opponent", moe]),
            "speed-ppo-moe-swiglu": ("trainer.ppo", ppo + ["--init", swiglu, "--opponent", swiglu]),
            "speed-bc-one": ("trainer.bc", bc + ["--core", "moe"]),
            "speed-bc-pair": ("trainer.bc", bc + ["--core", "moe", "--twin", "moe-swiglu"])}
```

with

```python
    runs = {"speed-ppo-v1": ("trainer.ppo", ppo + ["--init", str(PHASE2), "--opponent", str(PHASE2)]),
            "speed-ppo-v4": ("trainer.ppo", ppo + ["--init", v4, "--opponent", v4]),
            "speed-bc-v4": ("trainer.bc", bc + ["--core", "moe-swiglu"])}
```

The cleanup no longer meets twins. Replace

```python
        for name in runs:
            for d in (name, f"{name}-moe-swiglu"):
                shutil.rmtree(ROOT / "checkpoints" / d, ignore_errors=True)
                shutil.rmtree(ROOT / "runs" / d, ignore_errors=True)
```

with

```python
        for name in runs:
            shutil.rmtree(ROOT / "checkpoints" / name, ignore_errors=True)
            shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
```

Replace the closing lines

```python
    ppo_moe, ppo_swiglu = results["speed-ppo-moe"][0], results["speed-ppo-moe-swiglu"][0]
    print(f"\nPPO, SwiGLU experts: {ppo_swiglu / ppo_moe:.0%} of the MoE core's decisions/s "
          "(keep rule: at most 5% slower)")
    if "speed-bc-pair" in results:
        pair = results["speed-bc-pair"][0] / results["speed-bc-one"][0]
        print(f"BC, two networks: {pair:.0%} of one network's rows/s (PLAN 3B: >= 90% to train both in one night)")
```

with

```python
    rate, peak = results["speed-ppo-v4"]
    print(f"\nPPO, v4: {rate / v1:.0%} of the Phase 2 network's decisions/s in this run (Gate 3B: >= 75%); "
          f"its own peak {peak - (idle or 0)} MiB")
```

- [ ] **Step 2: Check that it starts**

Run: `.venv/Scripts/python scripts/speed_check.py --help`
Expected: the usage text with the new docstring. The imports and syntax are fine.

- [ ] **Step 3: Commit**

```bash
git add scripts/speed_check.py
git commit -m "Speed check for 3B step 3: the Phase 2 network against a v4 network with SwiGLU experts (PPO decisions/s and peak memory), and BC rows/s of the v4 network"
```

- [ ] **Step 4: Run it (ask the user first)**

Ask: "The v4 speed and memory check is ready: ~15 min, full GPU, PC otherwise idle. OK to start?" After the go, run:

Run: `.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12 > runs/3b-speed-v4.log 2>&1`
Expected: three rows (`speed-ppo-v1`, `speed-ppo-v4`, `speed-bc-v4`), then the `PPO, v4:` line.

- [ ] **Step 5: Apply the rule and record it**

In `docs/PLAN.md` 3B, under "Task list, part 2", write:
- the numbers:
  - v1 and v4 decisions/s, and v4's share of v1;
  - the MiB used before the runs, v4's peak and its own MiB;
  - BC rows/s and peak;
- the verdict: "Task 2 skipped" or "Task 2 runs".

In `CLAUDE.md`, replace the speed-check line under "Benchmarks" with:

```
.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12   # PPO decisions/s and peak VRAM of the Phase 2 network and a v4 network, BC rows/s of the v4 network (~15 min, PC idle)
```

```bash
git add docs/PLAN.md CLAUDE.md
git commit -m "PLAN 3B: v4's PPO speed and memory before Part 2's inputs, and the rule's verdict"
```

### Task 2: Rollouts keep their cell masks as bits (only if Task 1's rule calls for it)

If Task 1's verdict is "skipped", write `Task 2: skipped by Task 1's rule` in the ledger and go to Task 3.

**Files:**
- Modify: `trainer/ppo.py`
- Test: `trainer/test_ppo.py`

**Interfaces:**
- Consumes: the `rows(obs, learner)` dicts, with `obs["cell_mask"]` bool [N, maps, H·W]. H·W is a multiple of 8,
  since grids are multiples of 16.
- Produces: `pack_bits(m)` → uint8 [..., n / 8], and `unpack_bits(b)` → bool [..., 8 · n] (trainer/ppo.py).

- [ ] **Step 1: Write the failing test**

In `trainer/test_ppo.py`, replace the `trainer.ppo` import with:

```python
from trainer.ppo import (Outcomes, gae, main, pack_bits, parse_args, parser, resume_args, run_info, seat_rows,
                         unpack_bits)
```

and add:

```python
class PackTest(unittest.TestCase):
    def test_cell_masks_come_back_exactly_from_their_bits(self):
        m = torch.rand(3, 12, 64, device="cuda") < 0.3
        b = pack_bits(m)
        self.assertEqual((b.dtype, tuple(b.shape)), (torch.uint8, (3, 12, 8)))
        self.assertTrue(torch.equal(unpack_bits(b), m))
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python -m unittest trainer.test_ppo.PackTest -v`
Expected: ImportError: cannot import name `pack_bits` from `trainer.ppo`.

- [ ] **Step 3: Write the code**

In `trainer/ppo.py`, at module level before `main`:

```python
def pack_bits(m):
    """A bool tensor [..., n] (n a multiple of 8) as uint8 [..., n / 8]. The rollout keeps its cell masks this way: 12
    heat maps of 49,152 cells take 74 KB a row instead of 590 KB."""
    bits = torch.arange(8, device=m.device, dtype=torch.uint8)
    return (m.unflatten(-1, (-1, 8)).to(torch.uint8) << bits).sum(-1, dtype=torch.uint8)


def unpack_bits(b):
    """pack_bits undone."""
    bits = torch.arange(8, device=b.device, dtype=torch.uint8)
    return ((b.unsqueeze(-1) >> bits) & 1).bool().flatten(-2)
```

Make three replacements. The buffer:

```python
    buf = {k: torch.zeros((T,) + tuple(v.shape), dtype=v.dtype, device=device)
           for k, v in rows(obs, learner).items()}
```

becomes

```python
    def stored(o):
        """A seat's rows as the rollout keeps them: the cell masks as bits."""
        return {**o, "cell_mask": pack_bits(o["cell_mask"])}

    buf = {k: torch.zeros((T,) + tuple(v.shape), dtype=v.dtype, device=device)
           for k, v in stored(rows(obs, learner)).items()}
```

The rollout:

```python
            mine = rows(obs, learner)
            for k, v in mine.items():
                buf[k][step] = v
```

becomes

```python
            mine = rows(obs, learner)
            for k, v in stored(mine).items():
                buf[k][step] = v
```

The minibatches:

```python
                o = {k: v[mb] for k, v in flat.items()}
```

becomes

```python
                o = {k: v[mb] for k, v in flat.items()}
                o["cell_mask"] = unpack_bits(o["cell_mask"])
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_ppo -v`
Expected: all pass. The PPO runs in this file store their masks as bits and read them back.

- [ ] **Step 5: Commit**

```bash
git add trainer/ppo.py trainer/test_ppo.py
git commit -m "PPO rollouts keep their cell masks as bits: the 12 heat maps took 0.9 GB of the 12 GB card in 1,536 rows, now 0.11 GB"
```

- [ ] **Step 6: Measure again (ask the user first, ~8 min)**

Run: `.venv/Scripts/python scripts/speed_check.py --ppo-minibatches 12 --skip-bc > runs/3b-speed-v4-bits.log 2>&1`
Expected: v4's own peak is ~800 MiB lower than in Task 1. Then:
- If Task 1's rule holds, record the numbers in PLAN 3B and commit.
- If it does not, run once more with `--ppo-minibatches 16`, since smaller minibatches hold less in memory. If that
  holds, 16 becomes the setting for ranked PPO with a v4 network: record it in PLAN 3B and in CLAUDE.md's "GPU
  memory" bullet, and commit.
- If neither holds, stop and ask the user: the plan is wrong about where the memory goes.

### Task 3: Older networks read only the inputs they were trained on

**Files:**
- Modify: `trainer/model.py`
- Test: `trainer/test_model.py`

**Interfaces:**
- Consumes: `BUILDINGS` and `BUILDING_TYPES` (trainer/env.py).
- Produces: `OLD_CHANNELS = 49` and `OLD_SCALARS = 16` (trainer/model.py).
  - v1–v3 networks are built for exactly these inputs and read only them.
  - v4 networks read `dims["channels"]` and `dims["scalars"]`.

Until Tasks 5 and 6, the observations hold exactly 49 channels and 16 scalars, so nothing that runs changes. The tests
feed wider inputs.

- [ ] **Step 1: Write the failing tests**

In `trainer/test_model.py`, add `OLD_CHANNELS, OLD_SCALARS` to the `trainer.model` import. Add this class:

```python
class InputsTest(unittest.TestCase):
    """Part 2 of 3B step 3 adds channels and scalars after the old ones. The v1-v3 networks keep reading theirs."""

    def setUp(self):
        torch.manual_seed(0)
        self.obs = fake_obs()
        self.wide = {**DIMS, "channels": self.obs["spatial"].shape[1] + 8, "scalars": self.obs["scalars"].shape[1] + 10}
        extra = torch.full((N, 8, H, W), 64, dtype=torch.uint8, device="cuda")
        self.more = dict(self.obs, spatial=torch.cat([self.obs["spatial"], extra], 1),
                         scalars=torch.cat([self.obs["scalars"], torch.ones(N, 10, device="cuda")], 1))

    def test_older_networks_ignore_the_inputs_added_after_theirs(self):
        exact_fp32(self)
        old = {"v1": PolicyV1(self.wide), "v2": Policy(self.wide), "v3": Policy(self.wide, shares=True)}
        for name, net in old.items():
            with self.subTest(name):
                net = randomized(net).cuda().eval()
                with torch.no_grad():
                    a, b = net(self.obs), net(self.more)
                for k in a:
                    self.assertTrue(torch.allclose(a[k], b[k], rtol=1e-5, atol=1e-6), k)

    def test_v4_networks_read_every_input(self):
        net = randomized(Policy(self.wide, shares=True, full=True)).cuda().eval()
        self.assertEqual((net.stem.in_channels, net.scalars[0].in_features),
                         (self.wide["channels"], self.wide["scalars"]))
        with torch.no_grad():
            a, b = net(self.more), net(dict(self.more, scalars=self.more["scalars"] * 0))
        self.assertFalse(torch.allclose(a["value"], b["value"]))
```

In `LoadTest`, add:

```python
    def test_networks_saved_before_part_2_load_with_todays_wider_inputs(self):
        old = {**DIMS, "channels": OLD_CHANNELS, "scalars": OLD_SCALARS}
        wide = {**DIMS, "channels": OLD_CHANNELS + 8, "scalars": OLD_SCALARS + 10}
        with tempfile.TemporaryDirectory() as d:
            for args in ({"widths": "64,128,192"}, {"arch": "v3", "core": "moe-swiglu", "widths": "64,128,192"}):
                with self.subTest(args.get("arch", "v1")):
                    path = Path(d) / "net.pt"
                    torch.save({"model": new_net(old, args).state_dict(), "args": args}, path)
                    load_net(path, wide)  # a shape mismatch raises
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v`
Expected: ImportError: cannot import name `OLD_CHANNELS`. With only the constant added, the three new tests fail on
shape mismatches.

- [ ] **Step 3: Write the code**

In `trainer/model.py`, under `OLD_ACTS, OLD_MAPS = 5, 3 ...`:

```python
OLD_CHANNELS, OLD_SCALARS = BUILDINGS + 4 * len(BUILDING_TYPES), 16  # the inputs before 3B step 3, part 2 (v1-v3)
```

In `v1_spatial`, replace

```python
    b = spatial[:, BUILDINGS:].unflatten(1, (len(BUILDING_TYPES), 4))  # [N, type, relation, H, W]
```

with

```python
    b = spatial[:, BUILDINGS:OLD_CHANNELS].unflatten(1, (len(BUILDING_TYPES), 4))  # [N, type, relation, H, W]
```

In the `Policy` docstring, replace

```
    - full=True (arch v4, 3B step 3): 15 action types, 12 heat maps (MAP_TYPES) and the retreat pointer; otherwise the
      5 types and 3 heat maps of the networks before."""
```

with

```
    - full=True (arch v4, 3B step 3): 15 action types, 12 heat maps (MAP_TYPES), the retreat pointer and every input;
      otherwise the 5 types, 3 heat maps, OLD_CHANNELS channels and OLD_SCALARS scalars of the networks before."""
```

In `Policy.__init__`, after `c_in, n_feat, n_scal = dims["channels"], dims["playerFeatures"], dims["scalars"]`:

```python
        if not full:  # v2 and v3 networks were trained on the inputs before 3B step 3, part 2
            c_in, n_scal = OLD_CHANNELS, OLD_SCALARS
```

In `Policy.forward`, replace `x = _spatial_input(obs["spatial"], self.stem.weight)` with

```python
        x = _spatial_input(obs["spatial"][:, :self.stem.in_channels], self.stem.weight)
```

and `s = self.scalars(obs["scalars"])` with

```python
        s = self.scalars(obs["scalars"][:, :self.scalars[0].in_features])
```

In `PolicyV1`:
- In the docstring, replace "It reads its 22 input channels and 17 player features, rebuilt from today's
  observations." with "It reads its 22 input channels, 17 player features and 16 scalars, rebuilt from today's
  observations.".
- In `__init__`, replace `n_scal = dims["scalars"]` with `n_scal = OLD_SCALARS`.
- In `forward`, replace `self.scalars(obs["scalars"])` with `self.scalars(obs["scalars"][:, :OLD_SCALARS])`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v`
Expected: all pass.

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . > <workspace>/t3.log 2>&1` and read the tail.
Expected: all pass. The observations still hold 49 channels and 16 scalars.

- [ ] **Step 5: Commit**

```bash
git add trainer/model.py trainer/test_model.py
git commit -m "Networks before 3B step 3 part 2 read only the 49 channels and 16 scalars they were trained on, so their checkpoints load and play unchanged when the inputs grow; v4 networks read every input"
```

### Task 4: Unit records carry loaded missiles and zones

**Files:**
- Modify: `bridge/frame.ts`, `trainer/env.py` (the `UNIT` record)
- Create: `bridge/frame.test.ts`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `UNIT_BYTES = 20`. After `troops`, each record holds:
    - `ready`, u8 at 12;
    - a zero byte at 13;
    - `zx` u16 at 14, `zy` u16 at 16, `zr` u16 at 18.
  - `export const loaded = (u: Unit) => number` (bridge/frame.ts). Task 5 uses it.
  - In the trainer, the `UNIT` dtype with the fields `ready, pad, zx, zy, zr`. Task 6 reads them.

- [ ] **Step 1: Write the failing test**

Create `bridge/frame.test.ts`:

```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { TileRef } from "../OpenFront_Source_Code/openfrontio/src/core/game/GameMap";
import { encodeUnits, UNIT_BYTES, UNIT_TYPES } from "./frame";
import { EnvGame } from "./game";
import { ONION, script } from "./testkit";

/** One player's unit records, as encodeUnits sends them. */
function records(env: EnvGame, owner: number) {
  const b = encodeUnits(env.game, env.frame);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const all = Array.from({ length: b.length / UNIT_BYTES }, (_, i) => {
    const o = i * UNIT_BYTES;
    const zone = [dv.getUint16(o + 14, true), dv.getUint16(o + 16, true), dv.getUint16(o + 18, true)];
    return { type: UNIT_TYPES[b[o]], level: b[o + 1], owner: dv.getUint16(o + 2, true), ready: b[o + 12], zone };
  });
  return all.filter((r) => r.owner === owner);
}

test("unit records carry the missiles loaded and the disk each unit acts on", async () => {
  const env = await EnvGame.create(ONION, "zones");
  script(env, 260);
  const g = env.game;
  const config = g.config();
  const seat = env.seats[0];
  const [tile] = seat.tiles();
  const enemy = g.players().find((p) => p !== seat && p.isAlive() && p.numTilesOwned() > 0)!;
  const [target] = enemy.tiles();
  seat.buildUnit(UnitType.SAMLauncher, tile, {}).increaseLevel(); // level 2: its new slot reloads first
  seat.buildUnit(UnitType.MissileSilo, tile, {}).launch(); // level 1, its one missile reloading
  seat.buildUnit(UnitType.DefensePost, tile, {});
  seat.buildUnit(UnitType.DefensePost, tile, {}).setUnderConstruction(true);
  seat.buildUnit(UnitType.AtomBomb, tile, { targetTile: target, trajectory: [] });
  seat.buildUnit(UnitType.City, tile, {});
  const at = (t: TileRef, radius: number) => [g.x(t), g.y(t), Math.round(radius)];
  const none = [0, 0, 0];
  const of = (type: UnitType) => records(env, seat.smallID()).filter((r) => r.type === type);
  assert.deepEqual(
    of(UnitType.SAMLauncher).map((r) => [r.level, r.ready, r.zone]),
    [[2, 1, at(tile, config.samRange(1))]],
    "the range is still level 1's as the upgrade starts",
  );
  assert.deepEqual(of(UnitType.MissileSilo).map((r) => [r.ready, r.zone]), [[0, none]]);
  assert.deepEqual(
    of(UnitType.DefensePost).map((r) => [r.level, r.zone]).sort(),
    [[0, none], [1, at(tile, config.defensePostRange())]],
  );
  assert.deepEqual(
    of(UnitType.AtomBomb).map((r) => [r.ready, r.zone]),
    [[0, at(target, config.nukeMagnitudes(UnitType.AtomBomb).outer)]],
  );
  assert.deepEqual(of(UnitType.City).map((r) => [r.ready, r.zone]), [[0, none]]);
  for (let i = 0; i <= config.samUpgradeDuration(); i++) env.tickWith([]);
  assert.deepEqual(of(UnitType.SAMLauncher)[0].zone, at(tile, config.samRange(2)), "then it ramps to level 2's");
});
```

- [ ] **Step 2: Run it to see it fail**

Run: `node --import tsx --test ../../bridge/frame.test.ts`
Expected: FAIL on the first `deepEqual`. The record is 12 bytes, so the decoded fields hold the next record's bytes,
or the record count is off.

- [ ] **Step 3: Write the code**

In `bridge/frame.ts`, add `Unit` to the `Game` import:

```ts
import { Game, Unit, UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
```

Replace the unit record's comment and size:

```ts
/** Unit record: u8 type index (UNIT_TYPES), u8 level (0 = under construction), u16 owner, u32 cell, f32 troops. */
export const UNIT_BYTES = 12;
```

with

```ts
/**
 * Unit record: u8 type index (UNIT_TYPES), u8 level (0 = under construction), u16 owner, u32 cell, f32 troops,
 * u8 missiles loaded (silos and SAMs), u8 zero, then the unit's zone: u16 x, u16 y and u16 radius in tiles (radius 0:
 * none). A zone is a finished SAM's current range or a finished defense post's range around it, or a nuke's outer
 * blast around its target.
 */
export const UNIT_BYTES = 20;

/** Missiles a finished silo or SAM holds: its level minus the ones reloading. */
export const loaded = (u: Unit): number =>
  u.isUnderConstruction() ? 0 : Math.max(0, u.level() - u.missileTimerQueue().length);

/** The disk a unit acts on, as [centre, radius in tiles], or null. */
function zone(game: Game, u: Unit): [TileRef, number] | null {
  const config = game.config();
  switch (u.type()) {
    case UnitType.SAMLauncher:
      return u.isUnderConstruction() ? null : [u.tile(), config.dynamicSamRange(u, game.ticks())];
    case UnitType.DefensePost:
      return u.isUnderConstruction() ? null : [u.tile(), config.defensePostRange()];
    case UnitType.AtomBomb:
    case UnitType.HydrogenBomb:
    case UnitType.MIRVWarhead: {
      const target = u.targetTile();
      return target === undefined ? null : [target, config.nukeMagnitudes(u.type()).outer];
    }
    default:
      return null;
  }
}
```

In `encodeUnits`, after `dv.setFloat32(off + 8, u.troops(), true);`:

```ts
    if (u.type() === UnitType.MissileSilo || u.type() === UnitType.SAMLauncher) out[off + 12] = Math.min(loaded(u), 255);
    const z = zone(game, u);
    if (z !== null) {
      dv.setUint16(off + 14, game.x(z[0]), true);
      dv.setUint16(off + 16, game.y(z[0]), true);
      dv.setUint16(off + 18, Math.min(Math.round(z[1]), 65535), true);
    }
```

- [ ] **Step 4: Run the bridge tests and the typecheck**

Run: `node --import tsx --test "../../bridge/*.test.ts"`
Expected: all pass (42).

Run: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: no output.

- [ ] **Step 5: See the trainer fail on the longer records**

Run: `.venv/Scripts/python -m unittest trainer.test_env -v` (repo root)
Expected: errors. `np.frombuffer` reads 12-byte records from 20-byte ones: either the buffer size is not a multiple
of the element size, or the channel checks fail.

- [ ] **Step 6: Read the new record in the trainer**

In `trainer/env.py`, replace

```python
UNIT = np.dtype([("type", "u1"), ("level", "u1"), ("owner", "<u2"), ("cell", "<u4"), ("troops", "<f4")])
assert CELL.itemsize == 11 and UNIT.itemsize == 12
```

with

```python
UNIT = np.dtype([("type", "u1"), ("level", "u1"), ("owner", "<u2"), ("cell", "<u4"), ("troops", "<f4"),
                 ("ready", "u1"), ("pad", "u1"), ("zx", "<u2"), ("zy", "<u2"), ("zr", "<u2")])  # bridge/frame.ts
assert CELL.itemsize == 11 and UNIT.itemsize == 20
```

Run: `.venv/Scripts/python -m unittest trainer.test_env -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add bridge/frame.ts bridge/frame.test.ts trainer/env.py
git commit -m "Unit records carry the missiles loaded in silos and SAMs and each unit's zone from the game's Config: a finished SAM's current range, a finished defense post's range, a nuke's outer blast around its target"
```

### Task 5: Scalars: gold against every build's cost, and the loaded silo missiles

**Files:**
- Modify: `bridge/obs.ts`, `trainer/test_model.py` (`DIMS`, `fake_obs`)
- Test: `bridge/obs.test.ts`

**Interfaces:**
- Consumes: `BUILDS` (bridge/actions.ts: the nine builds in action order), `loaded` (Task 4).
- Produces: `SCALARS = 26`.
  - Scalars 16–24: gold / current cost of each build in `BUILDS` order, capped at 4, / 4.
  - Scalar 25: missiles loaded in the seat's silos / 16, capped at 1.

- [ ] **Step 1: Write the failing test**

In `bridge/obs.test.ts`, add `UnitType` and `BUILDS` to the imports:

```ts
import { UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { attackMask, BUILDS } from "./actions";
```

(the second line replaces `import { attackMask } from "./actions";`) and add:

```ts
test("scalars hold the gold against each build's current cost and the seat's loaded silo missiles", async () => {
  const env = await EnvGame.create(ONION, "costs");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(2_000_000n);
  const [tile] = seat.tiles();
  seat.buildUnit(UnitType.MissileSilo, tile, {}).increaseLevel(); // level 2, its new slot reloading: 1 loaded
  seat.buildUnit(UnitType.MissileSilo, tile, {}).setUnderConstruction(true); // none loaded
  const P = env.slots;
  const scalars = new Float32Array(SCALARS);
  const attack = attackMask(env.game, seat, P);
  encodeSeat(env.game, seat, env.timerMinutes, env.frame.cs, attack, new Float32Array(P * PLAYER_FEATURES),
    new Uint8Array(P), scalars);
  const gold = Number(seat.gold());
  [...BUILDS.values()].forEach((unit, i) => {
    const cost = Number(env.game.unitInfo(unit).cost(env.game, seat));
    assert.equal(scalars[16 + i], Math.fround(Math.min(gold / cost, 4) / 4), unit);
  });
  assert.ok(scalars[16] > scalars[24], "a city costs less than a hydrogen bomb");
  assert.equal(scalars[25], 1 / 16);
});
```

- [ ] **Step 2: Run it to see it fail**

Run: `node --import tsx --test ../../bridge/obs.test.ts`
Expected: FAIL. `SCALARS` is 16, so `scalars[16]` is undefined.

- [ ] **Step 3: Write the code**

In `bridge/obs.ts`, add the imports:

```ts
import { BUILDS } from "./actions";
import { loaded } from "./frame";
```

Set `export const SCALARS = 26;`. In the docstring, replace

```
 * 13 spawn immunity, 14 tiles per cell / 64, 15 share of players ahead in tiles.
```

with

```
 * 13 spawn immunity, 14 tiles per cell / 64, 15 share of players ahead in tiles, 16-24 gold / the current cost of
 * each build in BUILDS order (city, defense post, silo, factory, port, SAM, warship, atom bomb, hydrogen bomb; at
 * most 4, / 4), 25 missiles loaded in the seat's silos / 16 (at most 1).
```

At the end of `encodeSeat`, after `scalars[15] = ...`:

```ts
  let k = 16;
  for (const unit of BUILDS.values()) {
    scalars[k++] = Math.min(gold / Math.max(Number(game.unitInfo(unit).cost(game, seat)), 1), 4) / 4;
  }
  const silos = seat.units(UnitType.MissileSilo).reduce((n, u) => n + loaded(u), 0);
  scalars[25] = Math.min(silos, 16) / 16;
```

In `trainer/test_model.py`, set `DIMS = {"channels": N_CHANNELS, "playerFeatures": 25, "scalars": 26, "acts": 5}`.
In `fake_obs`, after `obs["retreat_mask"] = torch.zeros(N, P, dtype=torch.bool)`, add:

```python
    obs["scalars"] = torch.cat([obs["scalars"], torch.rand(N, 10, generator=g)], 1)  # the 10 scalars of 3B part 2
```

- [ ] **Step 4: Run the tests**

Run: `node --import tsx --test "../../bridge/*.test.ts"` (game dir)
Expected: all pass (43).

Run: `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`
Expected: no output.

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . > <workspace>/t5.log 2>&1` and read the tail.
Expected: all pass. The environment now gives 26 scalars: v4 networks read them all, and older ones read their 16
(Task 3).

- [ ] **Step 5: Commit**

```bash
git add bridge/obs.ts bridge/obs.test.ts trainer/test_model.py
git commit -m "Scalars 16-25: the gold against each build's current cost, and the missiles loaded in the seat's silos"
```

### Task 6: The level, missiles and zone channels

**Files:**
- Modify: `trainer/env.py`, `trainer/test_model.py` (`fake_obs`)
- Test: `trainer/test_env.py`

**Interfaces:**
- Consumes: Task 4's record fields (`ready`, `zx`, `zy`, `zr`), and `env.area` (cs² per game).
- Produces (trainer/env.py):
  - `LEVEL = 49` and `READY = 50`;
  - `ZONES = 51`: the channel ZONES + 2 · kind + side, with kind 0 SAM, 1 post, 2 nuke and side 0 ours, 1 theirs;
  - `ZONE_KINDS`, `STRUCTURES = 6`, and `N_CHANNELS = 57`;
  - `disk_cells(zx, zy, zr, side, H, W) -> (disk index, cell index)`.

- [ ] **Step 1: Write the failing tests**

In `trainer/test_env.py`, add `LEVEL, ZONE_KINDS, disk_cells` to the `trainer.env` import. Add, after
`phase2_spatial`:

```python
def new_channels(units, rel, names, side, H, W):
    """One seat's channels from LEVEL on, from the unit records by their definitions in trainer/env.py."""
    want = np.zeros((N_CHANNELS - LEVEL, H * W))
    cy, cx = np.divmod(np.arange(H * W), W)
    for u in units:
        name, r = names[u["type"]], int(rel[u["owner"]])
        if r == 0:
            continue
        if name in BUILDING_TYPES[:6]:  # structures, one per cell at most
            want[0, u["cell"]] += u["level"] / 16
            want[1, u["cell"]] += u["ready"] / 16
        kind = ZONE_KINDS.get(name)
        if kind is not None and u["zr"] > 0:
            x, y, radius = int(u["zx"]), int(u["zy"]), int(u["zr"])
            near = ((2 * cx + 1) * side - 2 * x) ** 2 + ((2 * cy + 1) * side - 2 * y) ** 2 <= (2 * radius) ** 2
            want[2 + 2 * kind + int(r >= 3), near] += u["ready"] / 16 if kind == 0 else 1
    return want
```

Add a class:

```python
class DiskTest(unittest.TestCase):
    def test_a_disk_covers_the_cells_whose_centres_lie_within_its_radius(self):
        H, W = 6, 8
        long = lambda *v: torch.tensor(v, dtype=torch.long, device="cuda")
        disks = [(4, 4, 8, 8), (0, 0, 4, 4)]  # x, y, radius, tiles per cell side: a Normal map's, a Compact corner's
        n, cell = disk_cells(*(long(*v) for v in zip(*disks)), H, W)
        want = {(k, c) for k, (x, y, r, s) in enumerate(disks) for c in range(H * W)
                if ((c % W + 0.5) * s - x) ** 2 + ((c // W + 0.5) * s - y) ** 2 <= r * r}
        self.assertEqual({(int(a), int(c)) for a, c in zip(n, cell)}, want)
        self.assertIn((0, 1), want, "a centre exactly 8 tiles away is inside")
        self.assertEqual(len(disk_cells(*(long() for _ in range(4)), H, W)[0]), 0, "no disks, no cells")
```

In `ChannelTest.setUpClass`, after `cls.upgrade = np.zeros(3, int) ...`:

```python
        cls.new_error, cls.zones = 0, np.zeros(3, int)  # worst error from LEVEL on; observations with each zone kind
```

and at the end of `keep`, after the `if cls.seen is None ...` block:

```python
            side = round(float(env.area[0]) ** 0.5)
            for seat in range(2):
                want = new_channels(rec, rel[0][seat], env.dims["unitTypes"], side, env.H, env.W)
                got = obs["spatial"][seat, LEVEL:].reshape(N_CHANNELS - LEVEL, -1).cpu().numpy().astype(np.int64)
                cls.new_error = max(cls.new_error, int(np.abs(got - np.clip(np.round(want * 64), 0, 255)).max()))
                cls.zones += want[2:].reshape(3, 2, -1).any((1, 2))
```

In `test_owner_and_building_channels_follow_their_definitions`, replace the two comparison lines

```python
            got = s["spatial"][seat].reshape(N_CHANNELS, -1)[OWNER:].astype(np.int64)
            self.assertLessEqual(np.abs(got - np.clip(np.round(want[OWNER:] * 64), 0, 255)).max(), 1)
```

with

```python
            got = s["spatial"][seat].reshape(N_CHANNELS, -1)[OWNER:LEVEL].astype(np.int64)
            self.assertLessEqual(np.abs(got - np.clip(np.round(want[OWNER:LEVEL] * 64), 0, 255)).max(), 1)
```

and add to `ChannelTest`:

```python
    def test_level_missile_and_zone_channels_follow_their_definitions(self):
        self.assertEqual(self.new_error, 0)  # 1/16 steps and whole counts are exact in 64ths
        self.assertGreater(self.zones[1], 0, "a defense post's zone was seen")
        self.assertGreater(self.zones[2], 0, "a nuke in flight was seen")
```

The fixture record has no SAM. SAM zones go through the same code with the loaded interceptors as their value, and
Task 4's bridge test checks their records.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python -m unittest trainer.test_env -v`
Expected: ImportError: cannot import name `LEVEL` from `trainer.env`.

- [ ] **Step 3: Write the code**

In `trainer/env.py`, replace `N_CHANNELS = BUILDINGS + 4 * len(BUILDING_TYPES)` with:

```python
LEVEL = BUILDINGS + 4 * len(BUILDING_TYPES)  # 49: the cell's structure's level / 16; 0 while under construction
READY = LEVEL + 1  # 50: the missiles loaded in the cell's silo or SAM / 16
ZONES = READY + 1  # 51..56: ZONES + 2 * kind + side (0 ours: self and friends; 1 theirs: enemies and bots)
ZONE_KINDS = {"SAM Launcher": 0, "Defense Post": 1, "Atom Bomb": 2, "Hydrogen Bomb": 2, "MIRV Warhead": 2}
N_CHANNELS = ZONES + 2 * 3  # a SAM's zone counts its loaded interceptors / 16, a post's or a nuke's 1
```

Under `UPGRADABLE = 5 ...`:

```python
STRUCTURES = 6  # BUILDING_TYPES[:6] are structures, at most one per cell
```

After `quantize_`:

```python
def disk_cells(zx, zy, zr, side, H, W):
    """The cells of an H x W grid that disks cover, as (disk index, cell index) pairs. A cell is in when its centre lies
    within zr tiles of the tile (zx, zy), by Euclidean distance. All int64 tensors; side is the tiles per cell side of
    each disk's game. The arithmetic stays in integers (positions doubled), so the edge is exact."""
    if len(zr) == 0:
        return zr, zr
    R = int(((zr + side - 1) // side).max()) + 1  # cells from a centre's cell to the farthest one inside
    # ponytail: one window size for every disk (the largest); chunk the disks if their memory ever matters
    d = torch.arange(-R, R + 1, device=zr.device)
    s = side[:, None, None]
    gy = (zy // side)[:, None, None] + d[:, None]  # [n, 2R + 1, 1]
    gx = (zx // side)[:, None, None] + d  # [n, 1, 2R + 1]
    dy, dx = (2 * gy + 1) * s - 2 * zy[:, None, None], (2 * gx + 1) * s - 2 * zx[:, None, None]
    near = (dx * dx + dy * dy <= (2 * zr[:, None, None]) ** 2) & (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    n, i, j = near.nonzero(as_tuple=True)
    return n, gy[n, i, 0] * W + gx[n, 0, j]
```

In `VecEnv.__init__`, after `self.transport = d["unitTypes"].index("Transport")`:

```python
        self.zone_kind = torch.tensor([ZONE_KINDS.get(t, -1) for t in d["unitTypes"]], device=dev)
```

In `_add_units`, replace the docstring's first two sentences

```
        """Paints each seat's buildings into x by type and relation, and returns every player's buildings by type
        [E, P, len(BUILDING_TYPES)]. A structure counts its level (under construction: 1), a transport its log troops
        / 10, a warship 1.
```

with

```
        """Paints each seat's units into x, and returns every player's buildings by type [E, P, len(BUILDING_TYPES)].
        - By type and relation, a structure counts its level (under construction: 1), a transport its log troops / 10,
          a warship 1.
        - LEVEL and READY hold each structure's level (0 under construction) and its loaded missiles, / 16.
        - ZONES hold each zone (bridge/frame.ts) by kind and by its owner's side. A SAM's zone counts its loaded
          interceptors / 16, a post's or a nuke's 1.
```

After `troops = torch.from_numpy(rec["troops"].astype(np.float32)).to(dev)`:

```python
        ready = torch.from_numpy(rec["ready"].astype(np.float32)).to(dev)
        zr = torch.from_numpy(rec["zr"].astype(np.int64)).to(dev)
        zk = self.zone_kind[typ]
        zone = ((zk >= 0) & (zr > 0)).nonzero().squeeze(1)  # the records with a zone
        zx, zy = (torch.from_numpy(rec[f].astype(np.int64)).to(dev)[zone] for f in ("zx", "zy"))
        side = self.area[env[zone]].sqrt().round().long()  # tiles per cell side of each zone's game
        disk, zcell = disk_cells(zx, zy, zr[zone], side, self.H, self.W)
        zi = zone[disk]  # the record behind each covered cell
        zvalue = torch.where(zk[zi] == 0, ready[zi] / 16, 1.0)
```

In the seat loop, after the `x.index_put_((row[k], BUILDINGS + ...), value[k], accumulate=True)` line:

```python
            st = k & (kind < STRUCTURES)
            x[:, LEVEL].index_put_((row[st], cell[st]), level[st] / 16, accumulate=True)
            x[:, READY].index_put_((row[st], cell[st]), ready[st] / 16, accumulate=True)
            rz = r[zi]
            on = rz > 0
            x.index_put_((row[zi][on], ZONES + 2 * zk[zi][on] + (rz[on] >= 3).long(), zcell[on]), zvalue[on],
                         accumulate=True)
```

In `trainer/test_model.py`, keep `fake_obs`'s old draws as they were: draw `"spatial"` with `OLD_CHANNELS` channels
instead of `N_CHANNELS`:

```python
        "spatial": torch.randint(0, 65, (N, OLD_CHANNELS, H, W), generator=g).to(torch.uint8),
```

and, after the scalars line added in Task 5:

```python
    new = torch.randint(0, 65, (N, N_CHANNELS - OLD_CHANNELS, H, W), generator=g).to(torch.uint8)
    obs["spatial"] = torch.cat([obs["spatial"], new], 1)  # the 8 channels of 3B part 2
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_env -v`
Expected: all pass, including `DiskTest` and the new channel test.

Run: `.venv/Scripts/python -m unittest discover -s trainer -t . > <workspace>/t6.log 2>&1` and read the tail.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add trainer/env.py trainer/test_env.py trainer/test_model.py
git commit -m "Channels 49-56: each structure's level and loaded missiles, and the zones of SAMs (loaded interceptors), defense posts and nukes in flight for each side, painted on the GPU from the unit records"
```

### Task 7: A uniform-pick baseline for the closeness scores

**Files:**
- Modify: `trainer/bc.py`
- Test: `trainer/test_bc.py`

**Interfaces:**
- Consumes: `measure`'s `tiles` [N, cells] and `rows`, and `obs["cell_mask"]` [N, 12, cells].
- Produces: `bc/{map}_cell_near_uniform` and `bc/{map}_cell_dist_uniform` for each of the network's heat maps.

- [ ] **Step 1: Write the failing test**

In `trainer/test_bc.py`'s `MeasureTests`, add:

```python
    def test_closeness_has_a_uniform_pick_as_its_baseline(self):
        obs = fake_obs()
        obs["scalars"][:, 14] = 1.0  # a Normal map: cells of 8 x 8 tiles
        obs["type_mask"][:] = True
        w = obs["spatial"].shape[-1]
        human = 10 * w + 10
        obs["cell_mask"][:] = False
        obs["cell_mask"][:, 2, [human, human + 1, human + 3 * w]] = True  # 0, 8 and 24 tiles from the human's city
        labels = torch.tensor([[BUILD, human, 0, 0]] * ROWS, device="cuda")
        m, zero = Meter(), torch.zeros(ROWS, device="cuda")
        measure(m, obs, labels, label_legal(obs, labels), ActionDist(RandomPolicy5()(obs), obs), zero, zero)
        s = m.means()
        self.assertAlmostEqual(s["bc/city_cell_near_uniform"], 2 / 3, places=5)  # 2 of the 3 within 16 tiles
        self.assertAlmostEqual(s["bc/city_cell_dist_uniform"], 32 / 3, places=4)  # (0 + 8 + 24) / 3
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python -m unittest trainer.test_bc.MeasureTests -v`
Expected: KeyError: `bc/city_cell_near_uniform`.

- [ ] **Step 3: Write the code**

In `trainer/bc.py`'s `measure`, after the `bc/{name}_cell_dist` line:

```python
        u = obs["cell_mask"][:, k].float()
        u = u / u.sum(1, keepdim=True).clamp(min=1)  # the baseline: a uniform pick over the legal cells
        m.add(f"bc/{name}_cell_near_uniform", ((u * (tiles <= NEAR_TILES)).sum(1) * rows).sum(), rows.sum())
        m.add(f"bc/{name}_cell_dist_uniform", ((u * tiles).sum(1) * rows).sum(), rows.sum())
```

In its docstring, replace

```
      hit: the chance it falls within NEAR_TILES (bc/*_cell_near) and its expected distance in tiles (bc/*_cell_dist);
```

with

```
      hit: the chance it falls within NEAR_TILES (bc/*_cell_near) and its expected distance in tiles (bc/*_cell_dist),
      and both for a uniform pick over the legal cells, their baseline (bc/*_uniform);
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v > <workspace>/t7.log 2>&1` and read the tail.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add trainer/bc.py trainer/test_bc.py
git commit -m "BC scores: each heat map's closeness gets its baseline, a uniform pick over the legal cells"
```

### Task 8: Docs, and the speed check with every input (Gate 3B's probe)

**Files:**
- Modify: `CLAUDE.md`, `docs/PLAN.md`

- [ ] **Step 1: CLAUDE.md**

In the "Networks" bullet, replace "v4 adds every action (15 types, 12 heat maps, a retreat pointer)" with "v4 adds
every action (15 types, 12 heat maps, a retreat pointer) and the 3B inputs (57 channels, 26 scalars)". Replace "v1–v3
see only their 5 types and 3 heat maps of the 15 (`ActionDist`)" with "v1–v3 see only their 5 types and 3 heat maps of
the 15 (`ActionDist`), and only the 49 channels and 16 scalars they were trained on".

After the "Actions" bullet, add:

```
- Observations (3B step 3, part 2):
  - Unit records carry the missiles loaded in silos and SAMs, and each unit's zone in tiles from the game's Config:
    a finished SAM's current range, a finished defense post's range, a nuke's outer blast around its target.
  - The GPU paints 8 channels after the 49 (`disk_cells`): each structure's level and loaded missiles (/ 16; level 0
    while under construction), and the SAM, defense-post and nuke zones of each side. Ours is self and friends;
    theirs is enemies and bots.
  - Scalars 16–25: the gold against each build's current cost, and the seat's loaded silo missiles.
```

If Task 2 ran, add to the "GPU memory" bullet: "PPO rollouts keep their cell masks as bits (`pack_bits`)", and the
minibatch setting Task 2 settled.

- [ ] **Step 2: PLAN.md**

In 3B:
- In "Observations", write that the coverage counts are by side (ours, theirs), with this list's decision 1 and its
  reasons. Mark the bullet as coded (2026-10-09), with the channels and scalars.
- After "Task list, part 1", add "Task list, part 2": `docs/superpowers/plans/2026-10-09-phase3b-observations.md`,
  written 2026-10-09, reviewed by the user (date), coded (date, first commit..; the bridge and trainer test counts).
  Give one line per decision (1–6 of this list) and Task 1's numbers.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/PLAN.md
git commit -m "Docs for 3B step 3, part 2: the observations (unit zones and loaded missiles, 8 channels, 10 scalars), and how older networks keep their inputs"
```

- [ ] **Step 4: The speed check with every input (ask the user first, ~15 min)**

Run: `.venv/Scripts/python scripts/speed_check.py --ppo-minibatches <12, or the setting of Task 2> > runs/3b-speed-part2.log 2>&1`
Expected: v4 keeps at least 75% of v1's decisions/s in the same run (Gate 3B's speed criterion), with no spill. A
spill shows as decisions/s falling 2–4×, or as "Shared GPU memory" rising in Task Manager. Then:
- Record the numbers in PLAN 3B, and the BC rows/s, which sets the BC refresh's night.
- If the bar is missed, stop and ask the user.

```bash
git add docs/PLAN.md
git commit -m "PLAN 3B: v4's speed and memory with every 3B input"
```
