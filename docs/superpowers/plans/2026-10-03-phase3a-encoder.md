# Phase 3A — See Better, Learn Better: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The ranked 1v1 bot (same 5 actions) sees the map and the other players better, the behaviour-cloning trainer
gets past its plateau, and every comparison runs on a fixed scoring set. The dense-versus-MoE core question is settled
by an A/B test before the expensive RL phase.

**Architecture:**
- **Observations** (`trainer/env.py`): 22 → 49 map channels.
  - Each cell's owners paint their strength onto it: troops, troops ÷ max, gold, troops attacking the seat, troops the
    seat sends at them.
  - Buildings are split by type and relation, and a structure counts its level.
  - The map is sent as uint8 in 64ths of a unit, so PPO rollouts store as 8 bits.
  - Each row also carries the cell owners' IDs and shares, and every player's buildings by type.
- **Networks** (`trainer/model.py`):
  - The Phase 2 network stays as `PolicyV1` and rebuilds its old inputs from the new ones, so its checkpoints keep
    playing.
  - The new `Policy` adds a 2-layer transformer over all players. Each token holds the player's features, its
    buildings, and the CNN features averaged over its land.
  - Its core is dense or Soft-MoE (8 experts), picked by a switch.
- **Behaviour cloning** (`trainer/bc.py`):
  - Each optimizer step averages 8 replay windows.
  - The learning rate warms up, then decays (cosine).
  - `--init` fine-tunes a checkpoint.
  - `--score` scores checkpoints on a fixed held-out set: NLL per head and per bucket, and router use per bucket.
- **Housekeeping:**
  - `--resume` refuses flags it would ignore (BC and PPO).
  - Evaluations keep their recordings in one folder per matchup, served with `trainer.watch --dir`.

**Tech Stack:** as Phase 2: Python 3.11, torch 2.14 cu130 (bf16 autocast), numpy, psutil, tensorboard, `unittest`.
Node workers unchanged.

**Spec:** `docs/PLAN.md`:
- "Phase 3 — Ranked 1v1 in full, then 2v2 (re-planned 2026-10-03)", 3A;
- decisions 5c (MoE) and 6 (model size);
- the Architecture table (player encoder row).

## Global Constraints

- **Code location:** never edit anything inside `OpenFront_Source_Code/openfrontio/`. The bridge (`bridge/`) is
  untouched in 3A: trainer code only.
- **Commits:** plain messages, authored by the repo owner, with no `Co-Authored-By` or any other Claude attribution.
- **Rule values** come from the game's `Config` at runtime. One structural fact is relied on and written down where it
  is used: structures stay `structureMinDist()` = 15 tiles apart (`PlayerImpl.validStructureSpawnTiles`, all
  players), and a cell spans at most 8 × 8 tiles, so a cell holds at most one structure. Recheck it on upgrades.
- **Dependencies:** no new Python or npm packages.
- **GPU runs:**
  - The test suite and the short speed checks of Task 8 (≤ 10 min of GPU) run in this session.
  - The BC probe (~1 h), the BC refresh nights (~12 h each) and the 512-game gate match are the user's. Give the exact
    commands; the user starts them in their own terminal.
- **Branch:** work on `phase3`. Nothing is pushed until the user says so.
- **Agents:** no subagents unless the user asks.
- **Where commands run:** Python from the repo root with `.venv/Scripts/python`. Bridge tests from
  `OpenFront_Source_Code/openfrontio/`.
- **GPU memory:** a ranked row is now ~2.7 MB (49 uint8 channels plus owners); PPO's 1,536-row rollout ≈ 4.4 GB
  (Phase 2: ~3.6 GB). Task 8 measures the peak.

## Review Focus

- **Old and new networks in one match:** a Phase 2 checkpoint (no `"arch"`) and a Phase 3 one play each other from
  the same observation rows. Test: `test_old_and_new_networks_play_each_other` (Task 5).
- **Levels and stacked ships:**
  - A structure above level 1 still counts as one structure for the Phase 2 network, and several warships in a cell
    count as their number.
  - Values above 255/64 saturate instead of wrapping around.
  - Tests: `test_a_structure_counts_once_whatever_its_level`, `test_values_are_rounded_to_64ths_and_saturate`
    (Task 3).
- **Checkpoints from before this plan:** Phase 2 PPO runs resume as `PolicyV1`; Phase 2 BC runs refuse, with an
  `--init` hint. Tests: `test_checkpoints_load_as_the_network_they_hold` (Task 5), `test_old_bc_checkpoints_cannot_resume`
  (Task 6).
- **Absent and reordered players:** players that do not exist (exists = 0) but carry garbage, and players in any
  order, do not change the type, value, maps, gates, or the existing players' pointer logits. Tests:
  `test_absent_players_change_nothing`, `test_players_can_come_in_any_order` (Task 4).
- **A missing watch folder:** `trainer.watch --dir` with a folder that does not exist names the saved matchups and
  stops before starting the client. Test: `test_a_missing_folder_stops_before_anything_starts` (Task 2).

## File structure

| File | Change |
|---|---|
| `trainer/ppo.py` | `parser()`, `given_flags`, `resume_args` (Task 1); `--core`, `network_args`, `new_net` (Task 5) |
| `trainer/bc.py` | resume check (1); `--init`/`--core` (5); accumulation, schedule, one pass (6); `--score` (7) |
| `trainer/eval.py` | `tag`, `replay_dir`, recordings per matchup (Task 2) |
| `trainer/control.py` | `serve(..., replays=)` (Task 2) |
| `trainer/watch.py` | `--dir` (Task 2) |
| `trainer/env.py` | 49 channels, `quantize_`, owners, buildings by type (Task 3) |
| `trainer/model.py` | `PolicyV1` + `v1_spatial` (3); `Policy`, `SoftMoE` (4); `arch_of`, `new_net`, `load_net` (5) |
| `trainer/test_*.py` | the tests of each task |
| `CLAUDE.md`, `docs/PLAN.md` | commands and results (Tasks 1, 2, 8) |

---

### Task 1: `--resume` refuses the flags it would ignore

**Files:**
- Modify: `trainer/ppo.py` (parser, `KEEP_ON_RESUME` → `SESSION`, `main` resume block)
- Modify: `trainer/bc.py` (parser, `KEEP_ON_RESUME`, `main` resume block)
- Modify: `CLAUDE.md` (the `--resume` bullet)
- Test: `trainer/test_ppo.py`

**Interfaces:**
- Produces (`trainer/ppo.py`):
  - `parser() -> argparse.ArgumentParser` and `parse_args(argv=None)`, in both `trainer/ppo.py` and `trainer/bc.py`;
  - `SESSION = ("port", "duty", "gpu_temp_limit")`;
  - `given_flags(p, argv) -> dict`: the explicitly given flags;
  - `resume_args(p, argv, saved: dict, may_change: tuple) -> argparse.Namespace`, or `SystemExit` naming the
    refused flags and `--init`.

- [ ] **Step 1: Write the failing test** (append to `trainer/test_ppo.py`; extend its import line)

```python
from trainer.ppo import Outcomes, gae, main, parse_args, parser, resume_args, run_info, seat_rows


class ResumeArgsTest(unittest.TestCase):
    SAVED = vars(parse_args(["--run", "r", "--setup", "ranked1v1", "--lr", "1e-4", "--kl-coef", "0.2"]))

    def resume(self, *flags):
        return resume_args(parser(), ["--run", "r", "--resume", *flags], self.SAVED, ("updates", "kl_coef"))

    def test_a_resume_keeps_the_runs_settings(self):
        a = self.resume("--port", "0", "--updates", "9")
        self.assertEqual((a.setup, a.lr, a.kl_coef, a.updates, a.port, a.resume), ("ranked1v1", 1e-4, 0.2, 9, 0, True))

    def test_a_given_kl_coef_replaces_the_runs(self):
        self.assertEqual(self.resume("--kl-coef", "0.05").kl_coef, 0.05)

    def test_flags_a_resume_would_ignore_are_refused(self):
        with self.assertRaisesRegex(SystemExit, r"--lr would be ignored.*--init checkpoints/r/latest\.pt"):
            self.resume("--lr", "2.5e-4")  # the default value, but not the run's
        self.resume("--lr", "1e-4", "--setup", "ranked1v1")  # the run's own values pass
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_ppo -v`
Expected: ERROR, `ImportError: cannot import name 'parser' from 'trainer.ppo'`.

- [ ] **Step 3: Implement** in `trainer/ppo.py`

Replace `KEEP_ON_RESUME = (...)` with:

```python
SESSION = ("port", "duty", "gpu_temp_limit")  # settings of this sitting, not of the run: a resume takes them as given
```

Rename `def parse_args(argv=None):` to `def parser():`. Keep every `add_argument` line, and end with `return p`
instead of `return p.parse_args(argv)`. Then add:

```python
def parse_args(argv=None):
    return parser().parse_args(argv)


def given_flags(p, argv):
    """The flags given on the command line, with their parsed values (an explicit value equal to the default counts)."""
    unset = object()
    ns = argparse.Namespace(**{k: unset for k in vars(p.parse_args([]))})
    return {k: v for k, v in vars(p.parse_args(argv, namespace=ns)).items() if v is not unset}


def resume_args(p, argv, saved, may_change):
    """A resumed run's settings: the checkpoint's (`saved`), this sitting's SESSION flags and the given `may_change`
    flags. Any other flag would be ignored, so one given with a value other than the run's is refused."""
    given, a = given_flags(p, argv), p.parse_args(argv)
    refused = [k for k, v in given.items() if k not in (*SESSION, *may_change, "run", "resume") and saved.get(k) != v]
    if refused:
        flags = ", ".join("--" + k.replace("_", "-") for k in refused)
        raise SystemExit(f"--resume keeps the run's settings, so {flags} would be ignored. Drop them, or start a new "
                         f"run from its weights: --run <new name> --init checkpoints/{a.run}/latest.pt")
    return argparse.Namespace(**{**vars(p.parse_args([])), **saved, **{k: getattr(a, k) for k in SESSION},
                                 **{k: given[k] for k in may_change if k in given}, "resume": True})
```

In `main`, replace the resume block (from `if a.resume:` through `a.kl_coef = given.kl_coef`) with:

```python
    if a.resume:
        state = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        a = resume_args(parser(), argv, state["args"], ("updates", "kl_coef"))
```

In `trainer/bc.py`:
- Delete `KEEP_ON_RESUME`.
- Make the same `parser()` / `parse_args()` split.
- Import `resume_args` from `trainer.ppo`, next to `game_commit` and `save_checkpoint`.
- Replace its resume line `a = argparse.Namespace(**{**state["args"], ...})` with
  `a = resume_args(parser(), argv, state["args"], ("updates",))`.

In `CLAUDE.md`, replace the bullet that starts "`--resume` keeps the checkpoint's settings except `KEEP_ON_RESUME`"
with:

```markdown
- `--resume` keeps the run's settings and refuses flags it would ignore. It takes `--updates` (and, for PPO,
  `--kl-coef`) and this sitting's `--port`, `--duty` and `--gpu-temp-limit`. To change anything else, start a new
  run from the checkpoint with `--init`.
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_ppo trainer.test_bc -v`
Expected: all pass. The existing resume smoke tests pass the run's own values, so they are accepted.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (54 tests).

```bash
git add trainer/ppo.py trainer/bc.py trainer/test_ppo.py CLAUDE.md
git commit -m "--resume refuses the flags it would ignore and points to --init (BC and PPO)"
```

---

### Task 2: Recordings in one folder per matchup

**Files:**
- Modify: `trainer/eval.py` (new `tag`, `replay_dir`; `main` and the docstring)
- Modify: `trainer/control.py` (`serve` takes `replays`)
- Modify: `trainer/watch.py` (`--dir`)
- Modify: `CLAUDE.md` (the eval `--record` and watch lines)
- Test: `trainer/test_eval.py`, `trainer/test_control.py`, `trainer/test_watch.py`

**Interfaces:**
- Produces:
  - `trainer.eval.tag(spec) -> str`: "random", "scripted", or `"<run> <update>"` (the run is the checkpoint's
    `args["run"]`, else its folder name);
  - `trainer.eval.replay_dir(specs) -> Path`: `REPLAYS / "<tag A> vs <tag B>"`;
  - `trainer.control.serve(control, port, sampler=True, log_dir=None, replays=REPLAYS)`.

- [ ] **Step 1: Write the failing tests**

`trainer/test_eval.py` (extend the imports to `from trainer.control import REPLAYS` and
`from trainer.eval import label, name_players, play, replay_dir, seats_for, summarize, tag`):

```python
    def test_recordings_go_to_a_folder_per_matchup(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, old = Path(d) / "bc" / "update0060000.pt", Path(d) / "x.pt"
            ckpt.parent.mkdir()
            torch.save({"args": {"run": "bc"}, "update": 60000}, ckpt)
            torch.save({"args": {}, "update": 7}, old)
            self.assertEqual(tag(str(ckpt)), "bc 60000")
            self.assertEqual(tag(str(old)), f"{Path(d).name} 7")  # no run in its settings: its folder
            self.assertEqual(replay_dir([str(ckpt), "random"]), REPLAYS / "bc 60000 vs random")
```

`trainer/test_control.py` (add `import tempfile` and `from pathlib import Path`):

```python
    def test_games_are_served_from_the_given_folder(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "Test0002.json").write_text('{"info": {"x": 1}}')
            server = serve(Control(), 0, sampler=False, replays=Path(d))
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}/game/Test0002"
                with urllib.request.urlopen(url, timeout=5) as r:
                    self.assertEqual(json.load(r), {"info": {"x": 1}})
            finally:
                server.shutdown()
                server.server_close()
```

`trainer/test_watch.py` (add `from unittest import mock`):

```python
class DirTest(unittest.TestCase):
    def test_a_missing_folder_stops_before_anything_starts(self):
        with mock.patch.object(watch, "start_client") as start, mock.patch.object(watch, "serve") as serve:
            with self.assertRaisesRegex(SystemExit, "no folder .*no such matchup"):
                watch.main(["--dir", "no such matchup"])
        start.assert_not_called()
        serve.assert_not_called()
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python -m unittest trainer.test_eval trainer.test_control trainer.test_watch -v`
Expected:
- test_eval: `ImportError: cannot import name 'replay_dir'`;
- test_control: `TypeError: serve() got an unexpected keyword argument 'replays'`;
- test_watch: `error: unrecognized arguments: --dir`, raised as `SystemExit: 2`, which does not match the regex.

- [ ] **Step 3: Implement**

`trainer/eval.py`: add after `label`:

```python
def tag(spec):
    """A policy's short name in folder names: a checkpoint's run and update (runs reuse update numbers)."""
    if spec in ("random", "scripted"):
        return spec
    ckpt = torch.load(spec, map_location="cpu", weights_only=False)
    return f"{ckpt['args'].get('run', Path(spec).parent.name)} {ckpt['update']}"


def replay_dir(specs):
    """Where a match's recorded games go: one folder per matchup, since fixed seeds give every match the same game
    IDs."""
    return REPLAYS / " vs ".join(tag(s) for s in specs)
```

In `main`, replace the lines from `record = {"dir": str(REPLAYS), ...` to the end of the function with:

```python
    specs = [a.a, a.b][:config["seats"]]
    folder = replay_dir(specs)
    record = {"dir": str(folder), "gitCommit": game_commit(), "games": a.record} if a.record > 0 else None
    result = play(specs, config, a.episodes, a.workers, a.games_per_worker, seed=a.seed, record=record)
    print(json.dumps({"a": a.a, "b": a.b, "setup": a.setup, **result}, indent=2))
    if result["recorded"]:
        print(f'watch: python -m trainer.watch --dir "{folder.name}", then open')
        for game_id in result["recorded"]:
            print(f"  http://localhost:9000/game/{game_id}")
```

Change the `--record` help to `"games saved per worker into replays/<A> vs <B>/, to watch them"`.

`trainer/control.py`:
- Change the `REPLAYS` comment to `# saved games: trainer.eval keeps one folder per matchup`.
- Change the signature to `def serve(control, port, sampler=True, log_dir=None, replays=REPLAYS):` and append to its
  docstring: "Saved games are served from `replays`."
- In `do_GET`, change `path = REPLAYS / f"{game[1]}.json"` to `path = replays / f"{game[1]}.json"`.

`trainer/watch.py`:
- Docstring: change the first usage line to
  `.venv/Scripts/python -m trainer.watch --dir "bc 60000 vs random"   then open http://localhost:9000/game/<id>; Ctrl+C stops both`
  and add the line
  `.venv/Scripts/python -m trainer.watch                    (the games saved directly in replays/, e.g. by --fetch)`.
- After the `--port` argument, add:

```python
    p.add_argument("--dir", help='serve this folder of replays/ (one per matchup, e.g. "bc 60000 vs random")')
```

After the `--fetch` block, add:

```python
    folder = REPLAYS / a.dir if a.dir else REPLAYS
    if not folder.is_dir():
        saved = sorted(d.name for d in REPLAYS.iterdir() if d.is_dir())
        raise SystemExit(f"no folder {folder}. Matchups saved so far: {', '.join(saved) or 'none'}")
```

Then, in the rest of `main`:
- Call `serve(Control(), a.port, sampler=False, replays=folder)`.
- Print `f"serving {folder} on http://127.0.0.1:{a.port}"`.
- Make the port-taken message
  `f"port {a.port} is taken: a running trainer's control server serves {REPLAYS} (not --dir); stop it to use --dir"`.
- List the newest links from `folder.glob("*.json")` instead of `REPLAYS.glob(...)`.

`CLAUDE.md`:
- In the eval `--record 1` line, change the comment to `# save games to replays/<A> vs <B>/, players named after their policies`.
- Replace the watch line with:

```bash
.venv/Scripts/python -m trainer.watch --dir "bc 60000 vs random"   # watch one matchup's saved games: starts the game's dev client and serves them; open http://localhost:9000/game/<id>, Ctrl+C stops both
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_eval trainer.test_control trainer.test_watch -v`
Expected: all pass.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (57 tests).

```bash
git add trainer/eval.py trainer/control.py trainer/watch.py trainer/test_eval.py trainer/test_control.py trainer/test_watch.py CLAUDE.md
git commit -m "Evaluations keep their recordings in replays/<A> vs <B>/ and trainer.watch --dir serves one matchup"
```

---

### Task 3: 49 map channels as uint8, and the Phase 2 network keeps its inputs

**Files:**
- Modify: `trainer/env.py` (channel constants, `quantize_`, `VecEnv.__init__` dims and building index, `_observe`,
  `_add_units`)
- Modify: `trainer/model.py` (`Policy` → `PolicyV1` with `v1_spatial` and `_spatial_input`; `load_net`)
- Modify: `trainer/bc.py`, `trainer/ppo.py` (`Policy` → `PolicyV1` until Task 5)
- Test: `trainer/test_env.py`, `trainer/test_model.py`, `trainer/test_bc.py`, `trainer/test_ppo.py`

**Interfaces:**
- Produces (`trainer/env.py`):
  - channel constants `LAND..FALLOUT` (unchanged), `OWNER = 12`, `OWNER_FEATURES = (8, 9, 10, 13, 14)`,
    `BUILDINGS = 17`, `BUILDING_TYPES` (8 unit type names), `N_CHANNELS = 49`;
  - `quantize_(x) -> uint8` (overwrites x);
  - the observation dict:
    - `"spatial"`: uint8 [N, 49, H, W], in 64ths;
    - `"players"`: float32 [N, P, 25]; features 17–24 are log1p(buildings by type) / 3;
    - `"owners"`: int16 [N, 2, H·W];
    - `"owner_land"`: uint8 [N, 2, H·W], in 64ths;
    - `"scalars"`, `"type_mask"`, `"attack_mask"`, `"cell_mask"`, as before;
  - `VecEnv.dims["channels"] == 49`, `dims["playerFeatures"] == 25`.
- Produces (`trainer/model.py`):
  - `V1_CHANNELS = 22`, `V1_FEATURES = 17`;
  - `v1_spatial(spatial uint8) -> int16 [N, 22, H, W]` in 64ths;
  - `_spatial_input(x, weight)`;
  - `PolicyV1(dims, widths)`, with the Phase 2 state-dict keys, reading the new observation dict.

- [ ] **Step 1: Write the failing tests**

`trainer/test_model.py`: replace the imports, `DIMS` and `fake_obs`, and run the three network tests on every
network kind:

```python
import math
import unittest

import torch

from trainer.env import BUILDINGS, N_CHANNELS
from trainer.model import ATTACK, BOAT, BUILD, SPAWN, ActionDist, PolicyV1, RandomPolicy, v1_spatial

DIMS = {"channels": N_CHANNELS, "playerFeatures": 25, "scalars": 16, "acts": 5,
        "troopBins": [0.1, 0.25, 0.5, 0.75, 1.0]}
N, P, H, W = 6, 5, 32, 32


def fake_obs(device="cuda", seed=0):
    """Rows as trainer/env.py builds them. Player 0 is unowned land, 1 the seat, 2 an enemy, 3 a bot, 4 absent."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cell_mask = torch.rand(N, 3, H * W, generator=g) < 0.1
    cell_mask[0] = False  # a row without any legal cell must not produce NaNs
    type_mask = torch.tensor([[1, 1, 0, 0, 0]] * 3 + [[1, 0, 1, 0, 1]] * 3, dtype=torch.bool)
    attack_mask = torch.zeros(N, P, dtype=torch.bool)
    attack_mask[3:, 2] = True
    players = torch.rand(N, P, 25, generator=g)
    players[..., :6] = 0
    players[:, :4, 0] = 1  # exists
    players[:, 1, 2] = players[:, 2, 4] = players[:, 3, 5] = 1  # self, enemy, bot
    players[:, 4] = 0
    owners = torch.randint(0, 4, (N, 2, H * W), generator=g)
    obs = {
        "spatial": torch.randint(0, 65, (N, N_CHANNELS, H, W), generator=g).to(torch.uint8),
        "scalars": torch.rand(N, 16, generator=g),
        "players": players,
        "owners": owners.to(torch.int16),
        "owner_land": (torch.randint(1, 33, (N, 2, H * W), generator=g) * (owners > 0)).to(torch.uint8),
        "type_mask": type_mask, "attack_mask": attack_mask, "cell_mask": cell_mask,
    }
    return {k: v.to(device) for k, v in obs.items()}


def nets():
    """One network of every kind."""
    return {"v1": PolicyV1(DIMS)}
```

In `ModelTest`:
- `setUp` keeps `self.obs = fake_obs()` and drops `self.model`.
- `test_output_shapes`, `test_samples_respect_masks_and_log_probs_are_finite` and `test_gradients_are_finite` loop
  with `for name, net in nets().items():` and `with self.subTest(name):`, use `net.cuda()` where they used
  `self.model`, and keep their bodies. The two random-policy tests and the KL test are unchanged.

Add:

```python
class V1InputsTest(unittest.TestCase):
    def test_a_structure_counts_once_whatever_its_level(self):
        x = torch.zeros(1, N_CHANNELS, 2, 2, dtype=torch.uint8)
        x[0, :12] = 7  # the first 12 channels are the same in both layouts
        x[0, BUILDINGS + 4 * 0 + 0, 0, 0] = 3 * 64  # a level-3 city of the seat
        x[0, BUILDINGS + 4 * 4 + 2, 0, 1] = 2 * 64  # a level-2 SAM of an enemy
        x[0, BUILDINGS + 4 * 7 + 3, 1, 0] = 2 * 64  # two warships of a bot
        x[0, BUILDINGS + 4 * 6 + 0, 1, 1] = 40  # the seat's transport: log troops / 10 = 40/64
        v = v1_spatial(x).float() / 64
        self.assertEqual(tuple(v.shape), (1, 22, 2, 2))
        self.assertTrue(torch.equal(v[0, :12], torch.full((12, 2, 2), 7 / 64)))
        self.assertEqual(float(v[0, 12, 0, 0]), 1.0)  # city, own
        self.assertEqual(float(v[0, 17, 0, 1]), 1.0)  # other structure, other
        self.assertEqual(float(v[0, 21, 1, 0]), 2.0)  # warship, other
        self.assertEqual(float(v[0, 18, 1, 1]), 40 / 64)  # transport, own
        self.assertEqual(float(v[0, 12:].sum()), 1 + 1 + 2 + 40 / 64)
```

`trainer/test_env.py`: new imports, the frozen Phase 2 reference, and the new tests:

```python
from trainer.env import (BUILDING_TYPES, BUILDINGS, LAND, N_CHANNELS, OWN, OWNER, OWNER_FEATURES, RANKED_1V1, ROOT,
                         VALID, VecEnv, quantize_)
from trainer.model import v1_spatial

FIXTURE = ROOT / "bridge" / "testdata" / "ranked1v1.json.gz"
UNIT_GROUP_V1 = {"City": 0, "Port": 1, "Factory": 2, "Missile Silo": 2, "SAM Launcher": 2, "Defense Post": 2,
                 "Transport": 3, "Warship": 4}


def watch_observations(env, keep):
    """Wraps env._observe so keep(args, obs) sees every observation with its inputs, while the env's cell state still
    matches them."""
    observe = env._observe

    def spy(*args):
        obs = observe(*args)
        keep(args, obs)
        return obs

    env._observe = spy


def phase2_spatial(env, players, rel, units):
    """The 22 channels Phase 2's VecEnv._observe built (copied from it at commit b6c05ea), to check v1_spatial."""
    N, S, P, hw, dev = env.N, env.S, env.P, env.H * env.W, env.device
    row_env = torch.arange(N, device=dev) // S
    area = env.area[row_env][:, None]
    rel_t = torch.from_numpy(rel.reshape(N, P).astype(np.int64)).to(dev)
    pl = torch.from_numpy(players.reshape(N, P, env.F)).to(dev)
    x = torch.zeros(N, 22, hw, device=dev)
    terr = env.terrain[row_env]
    x[:, 0:3] = terr[:, 0:3] / area[:, None]
    x[:, 3] = terr[:, 3] / 31
    x[:, 4] = (terr[:, 4] > 0).float()
    flat = row_env[:, None] * hw + torch.arange(hw, device=dev)
    own = torch.zeros(N, 4, hw, device=dev)
    for o, n in ((env.o1, env.n1), (env.o2, env.n2)):
        owner = o[flat]
        frac = n[flat] / area
        r = torch.gather(rel_t, 1, owner)
        own.scatter_add_(1, (r - 1).clamp(min=0).unsqueeze(1), (frac * (r > 0)).unsqueeze(1))
        x[:, 10] += frac * torch.gather(pl[..., 16], 1, owner)
    x[:, 5:9] = own
    x[:, 9] = (x[:, 0] - own.sum(1)).clamp(min=0)
    x[:, 11] = env.fo[flat] / area
    rec = np.concatenate([u for _, u in units])
    if len(rec):
        group_of = torch.tensor([UNIT_GROUP_V1.get(t, -1) for t in env.dims["unitTypes"]], device=dev)
        transport = env.dims["unitTypes"].index("Transport")
        e = torch.from_numpy(np.concatenate([np.full(len(u), i, np.int64) for i, u in units])).to(dev)
        typ = torch.from_numpy(rec["type"].astype(np.int64)).to(dev)
        owner = torch.from_numpy(rec["owner"].astype(np.int64)).to(dev)
        cell = torch.from_numpy(rec["cell"].astype(np.int64)).to(dev)
        troops = torch.from_numpy(rec["troops"].astype(np.float32)).to(dev)
        group = group_of[typ]
        value = torch.where(typ == transport, torch.log1p(troops) / 10, torch.ones_like(troops))
        for s in range(S):
            row = e * S + s
            r = rel_t[row, owner]
            keep = (group >= 0) & (r > 0)
            ch = 12 + 2 * group + (r != 1).long()
            x.index_put_((row[keep], ch[keep], cell[keep]), value[keep], accumulate=True)
    return x.view(N, 22, env.H, env.W)


class QuantizeTest(unittest.TestCase):
    def test_values_are_rounded_to_64ths_and_saturate(self):
        q = quantize_(torch.tensor([0.0, 0.004, 0.0079, 0.5, 1.0, 3.99, 5.0]))
        self.assertEqual(q.dtype, torch.uint8)
        self.assertEqual(q.tolist(), [0, 0, 1, 32, 64, 255, 255])  # 5.0 saturates: it does not wrap to 64


class ChannelTest(unittest.TestCase):
    """One replay of the fixture record, checked against the channel definitions and against the Phase 2 code."""

    @classmethod
    def setUpClass(cls):
        env = VecEnv(RANKED_1V1, workers=1, games_per_worker=1, seed="channels", records=[FIXTURE])
        cls.seen, cls.legacy_error = None, 0.0
        structures = [env.dims["unitTypes"].index(t) for t in BUILDING_TYPES[:6]]

        def keep(args, obs):
            players, rel, _, _, _, units = args
            old = phase2_spatial(env, players, rel, units).to(torch.bfloat16).float()
            new = v1_spatial(obs["spatial"]).to(torch.bfloat16).float() / 64  # what PolicyV1 reads
            cls.legacy_error = max(cls.legacy_error, float((new - old).abs().max()))
            rec = np.concatenate([u for _, u in units])
            n = int(np.isin(rec["type"], structures).sum())
            if cls.seen is None or n >= cls.seen["structures"]:  # keep the observation with the most structures
                cls.seen = {"structures": n, "players": players[0].copy(), "rel": rel[0].copy(), "units": rec.copy(),
                            "spatial": obs["spatial"].cpu().numpy(),
                            "features": obs["players"][:, :, 17:].cpu().numpy(),
                            "cells": [t[:env.H * env.W].cpu().numpy() for t in (env.o1, env.n1, env.o2, env.n2)],
                            "area": float(env.area[0])}

        try:
            watch_observations(env, keep)
            for _ in range(560):  # past the end of the record (552 windows), whatever window it started at
                env.step()
            cls.names, cls.hw, cls.P = env.dims["unitTypes"], env.H * env.W, env.P
        finally:
            env.close()

    def test_old_networks_get_the_phase2_inputs(self):
        self.assertLessEqual(self.legacy_error, 1 / 32)  # 1/64 rounding of up to 3 summed channels, then bf16

    def test_owner_and_building_channels_follow_their_definitions(self):
        s = self.seen
        self.assertGreaterEqual(s["structures"], 5, "the record's structures were seen")
        o1, n1, o2, n2 = s["cells"]
        for seat in range(2):
            players, rel = s["players"][seat], s["rel"][seat]
            want = np.zeros((N_CHANNELS, self.hw))
            for o, n in ((o1, n1), (o2, n2)):
                share = n / s["area"] * (rel[o] > 0)
                for i, f in enumerate(OWNER_FEATURES):
                    want[OWNER + i] += share * players[o, f]
            levels = np.zeros((self.P, len(BUILDING_TYPES)))
            for u in s["units"]:
                name = self.names[u["type"]]
                if name in BUILDING_TYPES:
                    k = BUILDING_TYPES.index(name)
                    value = np.log1p(u["troops"]) / 10 if name == "Transport" else max(int(u["level"]), 1)
                    want[BUILDINGS + 4 * k + int(rel[u["owner"]]) - 1, u["cell"]] += value
                    levels[u["owner"], k] += value
            got = s["spatial"][seat].reshape(N_CHANNELS, -1)[OWNER:].astype(np.int64)
            self.assertLessEqual(np.abs(got - np.clip(np.round(want[OWNER:] * 64), 0, 255)).max(), 1)
            np.testing.assert_allclose(s["features"][seat], np.log1p(levels) / 3, atol=1e-5)
```

In `VecEnvTest.test_observations_during_random_play`, the first shape checks become:

```python
            self.assertEqual(obs["spatial"].shape, (2, N_CHANNELS, 64, 64))
            self.assertEqual(obs["spatial"].dtype, torch.uint8)
            self.assertEqual(obs["players"].shape, (2, 5, 25))
            self.assertEqual((obs["owners"].shape, obs["owners"].dtype), ((2, 2, 64 * 64), torch.int16))
            self.assertEqual(obs["owner_land"].dtype, torch.uint8)
```

and the loop reads `x = obs["spatial"].float() / 64`.

In `trainer/test_bc.py` and `trainer/test_ppo.py`, import `PolicyV1` instead of `Policy` and construct
`PolicyV1(dims)`. That is the CloneTest model and the PPO two-seat init checkpoint.

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python -m unittest trainer.test_env trainer.test_model -v`
Expected: ERROR, `ImportError: cannot import name 'BUILDING_TYPES' from 'trainer.env'` (and `PolicyV1` from
`trainer.model`).

- [ ] **Step 3: Implement `trainer/env.py`**

Replace the "Spatial channels" block (from `# Spatial channels` through `DENSITY_FEATURE = 16`) with:

```python
# Spatial channels. They reach the model as uint8 in 64ths of a unit (quantize_; trainer/model.py divides by 64): tile
# shares stay exact, other values move by at most 1/128, and PPO stores its rollouts as 8 bits. Tile counts are divided
# by the tiles per cell.
LAND, SHORE, OCEAN, HEIGHT, VALID = 0, 1, 2, 3, 4
OWN = 5  # 5..8: land owned by self, friendly, enemy, bot (relation codes 1..4 from the worker)
UNOWNED, DENSITY, FALLOUT = 9, 10, 11
OWNER = 12  # 12..16: the cell's owners' player features OWNER_FEATURES, weighted by their shares of its land
OWNER_FEATURES = (8, 9, 10, 13, 14)  # log troops, troops / max, log gold, troops attacking the seat, sent at it
BUILDINGS = 17  # 17..48: BUILDING_TYPES x relation (self, friendly, enemy, bot), see VecEnv._add_units
BUILDING_TYPES = ("City", "Port", "Factory", "Missile Silo", "SAM Launcher", "Defense Post", "Transport", "Warship")
N_CHANNELS = BUILDINGS + 4 * len(BUILDING_TYPES)
DENSITY_FEATURE = 16  # player feature: log troops per tile


def quantize_(x):
    """x in 64ths as uint8 (x itself is overwritten on the way). Values above 255/64 (a structure above level 3,
    several warships in one cell) saturate."""
    return x.mul_(64).round_().clamp_(0, 255).to(torch.uint8)
```

In `VecEnv.__init__`, replace the three lines `self.dims = ...`, `dev, hw = ...` and `self.unit_group = ...` with:

```python
        self.dims = {**d, "channels": N_CHANNELS, "playerFeatures": self.F + len(BUILDING_TYPES), "H": self.H,
                     "W": self.W}
        dev, hw = self.device, self.H * self.W
        self.building = torch.tensor([BUILDING_TYPES.index(t) if t in BUILDING_TYPES else -1 for t in d["unitTypes"]],
                                     device=dev)
```

Keep the `self.transport = ...` line that follows.

Replace `_observe` and `_add_units` with:

```python
    def _observe(self, players, rel, scalars, types, attack, units):
        """The model's inputs, one row per seat:
        - "spatial": uint8 [N, N_CHANNELS, H, W], in 64ths;
        - "players": [N, P, F + len(BUILDING_TYPES)], the worker's features, then the player's buildings by type
          (log1p of their summed values, / 3);
        - "owners" (int16) and "owner_land" (uint8) [N, 2, H * W]: each cell's two largest owners and their shares of
          it in 64ths, for the model's per-player map features;
        - the scalars and the masks."""
        N, S, P, hw, dev = self.N, self.S, self.P, self.H * self.W, self.device
        row_env = torch.arange(N, device=dev) // S
        area = self.area[row_env][:, None]  # [N, 1]
        rel_t = torch.from_numpy(rel.reshape(N, P).astype(np.int64)).to(dev)
        pl = torch.from_numpy(players.reshape(N, P, self.F)).to(dev)
        x = torch.zeros(N, N_CHANNELS, hw, device=dev)
        terr = self.terrain[row_env]
        valid = terr[:, 4] > 0
        x[:, LAND:OCEAN + 1] = terr[:, 0:3] / area[:, None]
        x[:, HEIGHT] = terr[:, 3] / 31
        x[:, VALID] = valid.float()
        flat = row_env[:, None] * hw + torch.arange(hw, device=dev)
        own = torch.zeros(N, 4, hw, device=dev)
        owners, shares = [], []
        for o, n in ((self.o1, self.n1), (self.o2, self.n2)):
            owner = o[flat]
            frac = n[flat] / area
            r = torch.gather(rel_t, 1, owner)
            held = frac * (r > 0)  # owner 0 (no owner) holds nothing
            own.scatter_add_(1, (r - 1).clamp(min=0).unsqueeze(1), held.unsqueeze(1))
            x[:, DENSITY] += frac * torch.gather(pl[..., DENSITY_FEATURE], 1, owner)
            for i, f in enumerate(OWNER_FEATURES):
                x[:, OWNER + i] += held * torch.gather(pl[..., f], 1, owner)
            owners.append(owner)
            shares.append(frac)
        x[:, OWN:OWN + 4] = own
        x[:, UNOWNED] = (x[:, LAND] - own.sum(1)).clamp(min=0)
        x[:, FALLOUT] = self.fo[flat] / area
        buildings = self._add_units(x, units, rel_t)
        cell_mask = torch.stack([x[:, UNOWNED] > 0,
                                 (x[:, SHORE] > 0) & (x[:, LAND] - x[:, OWN] > 1e-6),
                                 x[:, OWN] > 0], 1) & valid[:, None]
        return {
            "spatial": quantize_(x).view(N, N_CHANNELS, self.H, self.W),
            "scalars": torch.from_numpy(scalars.reshape(N, self.K)).to(dev),
            "players": torch.cat([pl, torch.log1p(buildings[row_env]) / 3], 2),
            "owners": torch.stack(owners, 1).to(torch.int16),
            "owner_land": quantize_(torch.stack(shares, 1)),
            "type_mask": torch.from_numpy(types.reshape(N, self.T)).to(dev),
            "attack_mask": torch.from_numpy(attack.reshape(N, P)).to(dev),
            "cell_mask": cell_mask,
        }

    def _add_units(self, x, units, rel_t):
        """Paints each seat's buildings into x by type and relation, and returns every player's buildings by type
        [E, P, len(BUILDING_TYPES)]. A structure counts its level (under construction: 1), a transport its log troops
        / 10, a warship 1."""
        buildings = torch.zeros(self.E, self.P, len(BUILDING_TYPES), device=self.device)
        rec = np.concatenate([u for _, u in units])
        if len(rec) == 0:
            return buildings
        dev = self.device
        env = torch.from_numpy(np.concatenate([np.full(len(u), e, np.int64) for e, u in units])).to(dev)
        typ = torch.from_numpy(rec["type"].astype(np.int64)).to(dev)
        owner = torch.from_numpy(rec["owner"].astype(np.int64)).to(dev)
        cell = torch.from_numpy(rec["cell"].astype(np.int64)).to(dev)
        level = torch.from_numpy(rec["level"].astype(np.float32)).to(dev)
        troops = torch.from_numpy(rec["troops"].astype(np.float32)).to(dev)
        kind = self.building[typ]
        value = torch.where(typ == self.transport, torch.log1p(troops) / 10, level.clamp(min=1))
        keep = kind >= 0
        buildings.index_put_((env[keep], owner[keep], kind[keep]), value[keep], accumulate=True)
        for s in range(self.S):
            row = env * self.S + s
            r = rel_t[row, owner]
            k = keep & (r > 0)
            x.index_put_((row[k], BUILDINGS + 4 * kind[k] + r[k] - 1, cell[k]), value[k], accumulate=True)
        return buildings
```

- [ ] **Step 4: Implement `trainer/model.py`** (the Phase 2 network becomes `PolicyV1`)

Change the module docstring to: `"""Policy networks (PLAN "Architecture"). PolicyV1 is the Phase 2 network, kept for its
checkpoints."""`. Add `from trainer.env import BUILDING_TYPES, BUILDINGS, OWNER` to the imports and, after `MASK_VALUE`:

```python
V1_CHANNELS, V1_FEATURES = 22, 17  # the Phase 2 network's inputs


def v1_spatial(spatial):
    """The Phase 2 network's 22 input channels, in 64ths (int16), from the N_CHANNELS of trainer/env.py.
    - The first 12 channels are the same.
    - Its unit channels were (city, port, other structure, transport, warship) x (own, other), and a structure counted
      1 whatever its level.
    - The game keeps structures structureMinDist = 15 tiles apart and a cell spans at most 8 x 8 tiles, so a cell holds
      at most one structure: its count is whether its channel is non-zero."""
    total = lambda t: t.sum(1, dtype=torch.int16)
    b = spatial[:, BUILDINGS:].unflatten(1, (len(BUILDING_TYPES), 4))  # [N, type, relation, H, W]
    built = (b[:, :6] > 0).to(torch.int16) * 64
    ships = b[:, 6:].to(torch.int16)
    groups = (built[:, 0], built[:, 1], total(built[:, 2:]), ships[:, 0], ships[:, 1])  # each [N, relation, H, W]
    units = torch.stack([torch.stack([g[:, 0], total(g[:, 1:])], 1) for g in groups], 1).flatten(1, 2)
    return torch.cat([spatial[:, :OWNER].to(torch.int16), units], 1)


def _spatial_input(x, weight):
    """Channels in 64ths (uint8 or int16) as the conv input, channels-last. Under autocast it is bf16: k/64 is exact
    for k < 256, and larger sums round as the bf16 cast of the Phase 2 values did. Otherwise it is the weights' dtype."""
    dtype = torch.bfloat16 if torch.is_autocast_enabled("cuda") else weight.dtype
    return x.contiguous(memory_format=torch.channels_last).to(dtype).mul_(1 / 64)
```

Rename `class Policy` to `class PolicyV1` and give it the docstring `"""The Phase 2 network, kept for its checkpoints
(those without "arch"). It reads its 22 input channels and 17 player features, rebuilt from today's observations."""`.

In `__init__`:
- Replace `c_in, n_feat, n_scal = dims["channels"], dims["playerFeatures"], dims["scalars"]` with
  `n_scal = dims["scalars"]`.
- Use `V1_CHANNELS` in `self.stem` and `self.skip`, and `V1_FEATURES` in the first `nn.Linear` of `self.players`.

In `forward`, replace the first line with:

```python
        x = _spatial_input(v1_spatial(obs["spatial"]), self.stem.weight)
        players = obs["players"][..., :V1_FEATURES]
```

Then use `players` where the body read `obs["players"]`: in `emb = self.players(players)` and
`exists = players[..., :1]`.

In `load_net`, construct `PolicyV1(...)` instead of `Policy(...)`.

In `trainer/bc.py` and `trainer/ppo.py`, import and construct `PolicyV1` instead of `Policy` (Task 5 replaces both
with `new_net`).

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_env trainer.test_model -v`
Expected: all pass. `ChannelTest` replays ~560 windows, which takes about a minute.

- [ ] **Step 6: Full suite and commit**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (61 tests).

```bash
git add trainer/env.py trainer/model.py trainer/bc.py trainer/ppo.py trainer/test_env.py trainer/test_model.py trainer/test_bc.py trainer/test_ppo.py
git commit -m "Observations: 49 map channels (owners' strength; buildings by type, relation and level) as uint8 64ths; the Phase 2 network keeps its inputs"
```

---

### Task 4: The player transformer and the dense / Soft-MoE core

**Files:**
- Modify: `trainer/model.py` (new `D`, `masked_mean`, `land_features`, `SoftMoE`, `Policy`)
- Test: `trainer/test_model.py`

**Interfaces:**
- Consumes: the Task 3 observation dict; `_spatial_input`; `Res`; `N_MAPS`.
- Produces:
  - `Policy(dims, widths=(64, 128, 192), core="moe")` → the outputs `type, maps, bins, ptr, value`, plus `gates`
    [N, 8] when `core == "moe"`;
  - attributes `Policy.moe: bool` and `Policy.core`;
  - `SoftMoE(d, experts=8)` with `.router`, `.w` [E, d, d] and `.b` [E, d]; `forward(x) -> (y, gates)`.

- [ ] **Step 1: Write the failing tests** (`trainer/test_model.py`)

Extend the model import with `Policy, SoftMoE`. Make `nets()` return:

```python
    return {"v1": PolicyV1(DIMS), "dense": Policy(DIMS, core="dense"), "moe": Policy(DIMS, core="moe")}
```

Add:

```python
def randomized(net):
    """net with random weights everywhere, scaled by fan-in: the zero-initialised heads would hide what the inputs
    do."""
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() > 1:
                fan_in = p.shape[-1] if p.dim() == 3 else p[0].numel()  # SoftMoE's experts: [experts, out, in]
                p.normal_(0, fan_in ** -0.5)
    return net


class PolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.obs = fake_obs()

    def test_absent_players_change_nothing(self):
        net = randomized(Policy(DIMS)).cuda().eval()
        noisy = dict(self.obs, players=self.obs["players"].clone())
        noisy["players"][:, 4, 1:] = 3.0  # the absent row (feature 0 = 0) holds garbage, relation flags included
        with torch.no_grad():
            a, b = net(self.obs), net(noisy)
        self.assertGreater(float(a["ptr"][:, :4].std()), 1e-2, "the pointer depends on the players")
        for k in ("type", "maps", "bins", "value", "gates"):
            self.assertTrue(torch.allclose(a[k], b[k], rtol=1e-4, atol=1e-5), k)
        self.assertTrue(torch.allclose(a["ptr"][:, :4], b["ptr"][:, :4], rtol=1e-4, atol=1e-5))

    def test_players_can_come_in_any_order(self):
        net = randomized(Policy(DIMS)).cuda().eval()
        perm = torch.tensor([0, 3, 1, 4, 2], device="cuda")  # new row i holds old row perm[i]
        moved = dict(self.obs, players=self.obs["players"][:, perm], attack_mask=self.obs["attack_mask"][:, perm],
                     owners=torch.argsort(perm)[self.obs["owners"].long()].to(torch.int16))
        with torch.no_grad():
            a, b = net(self.obs), net(moved)
        for k in ("type", "maps", "bins", "value", "gates"):
            self.assertTrue(torch.allclose(a[k], b[k], rtol=1e-4, atol=1e-4), k)
        self.assertTrue(torch.allclose(a["ptr"][:, perm], b["ptr"], rtol=1e-4, atol=1e-4))

    def test_the_two_cores_start_alike_everywhere_else(self):
        torch.manual_seed(3)
        dense = Policy(DIMS, core="dense").state_dict()
        torch.manual_seed(3)
        moe = Policy(DIMS, core="moe").state_dict()
        for k, v in dense.items():
            if not k.startswith("core."):
                self.assertTrue(torch.equal(v, moe[k]), k)

    def test_64ths_are_exact_in_bf16(self):
        k = torch.arange(256)
        self.assertTrue(torch.equal(k.to(torch.uint8).to(torch.bfloat16).mul(1 / 64).float(), k.float() / 64))


class SoftMoETest(unittest.TestCase):
    def test_the_output_blends_the_experts_by_the_routers_weights(self):
        torch.manual_seed(0)
        moe = SoftMoE(16, experts=4)
        torch.nn.init.normal_(moe.router.weight)  # a router that already prefers some experts
        x = torch.randn(5, 16)
        y, gates = moe(x)
        self.assertTrue(torch.allclose(gates.sum(1), torch.ones(5)))
        experts = torch.stack([x @ moe.w[e].T + moe.b[e] for e in range(4)], 1)  # [5, experts, 16]
        self.assertTrue(torch.allclose(y, (gates[..., None] * experts).sum(1), atol=1e-5))

    def test_every_expert_learns(self):
        torch.manual_seed(0)
        moe = SoftMoE(16, experts=4)
        y, _ = moe(torch.randn(5, 16))
        y.pow(2).sum().backward()
        self.assertTrue(all(float(moe.w.grad[e].abs().sum()) > 0 for e in range(4)))
        self.assertGreater(float(moe.router.weight.grad.abs().sum()), 0)
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v`
Expected: ERROR, `ImportError: cannot import name 'Policy' from 'trainer.model'`.

- [ ] **Step 3: Implement** in `trainer/model.py`

Update the module docstring to name `Policy` (the Phase 3 network) and `PolicyV1`. After `V1_CHANNELS, V1_FEATURES`,
add `D = 128  # width of the player tokens`. After `_spatial_input`, add:

```python
def masked_mean(x, mask):
    """Mean of x [N, P, D] over the rows where mask [N, P] holds; zeros where none does."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


def land_features(obs, f):
    """Per player, the mean of the feature map f [N, C, h, w] over its land, at f's resolution (h x w blocks of the
    cell grid). It comes from each cell's two largest owners and their shares of it (obs "owners", "owner_land" in
    64ths). Players without land get zeros."""
    n, _, h, w = f.shape
    H, W = obs["spatial"].shape[-2:]
    r = H // h  # cells per block side
    cell = torch.arange(H * W, device=f.device)
    block = cell // W // r * w + cell % W // r
    index = (obs["owners"].long() * (h * w) + block).flatten(1)
    weight = torch.zeros(n, obs["players"].shape[1] * h * w, device=f.device)
    weight = weight.scatter_add_(1, index, obs["owner_land"].flatten(1).float()).unflatten(1, (-1, h * w))
    weight = weight / weight.sum(2, keepdim=True).clamp(min=1)
    return torch.bmm(weight.to(f.dtype), f.flatten(2).transpose(1, 2))


class SoftMoE(nn.Module):
    """Soft-MoE (PLAN decision 5c) for one vector per row. With a single token, every expert's slot receives the row's
    vector, and the combine weights are a softmax over the experts: the layer blends the experts' outputs by
    situation. Each expert is one d x d layer, so running them all costs little next to the CNN. More experts can be
    added later by copying trained ones (upcycling)."""

    def __init__(self, d, experts=8):
        super().__init__()
        self.router = nn.Linear(d, experts)
        nn.init.zeros_(self.router.weight)  # every expert weighs the same at the start
        nn.init.zeros_(self.router.bias)
        layers = [nn.Linear(d, d) for _ in range(experts)]
        self.w = nn.Parameter(torch.stack([layer.weight.detach() for layer in layers]))  # [experts, out, in]
        self.b = nn.Parameter(torch.stack([layer.bias.detach() for layer in layers]))  # [experts, out]

    def forward(self, x):
        gates = self.router(x).softmax(-1)  # [N, experts]
        y = torch.einsum("ni,eoi->neo", x, self.w) + self.b  # every expert's output [N, experts, out]
        return torch.einsum("ne,neo->no", gates, y), gates


class Policy(nn.Module):
    """The Phase 3 network (PLAN Phase 3A).
    - The map encoder, heat-map decoder and heads are PolicyV1's.
    - Each player becomes a token: its features, its buildings by type, and the map features (1/8 resolution)
      averaged over its land.
    - A 2-layer transformer runs over all the tokens; absent players are masked out.
    - The core reads an attention summary queried by the map and the scalars, the seat's own token, and the means over
      its friends, its enemies and the bots.
    - The attack pointer reads the tokens.
    - core: "dense" (one 512 x 512 layer) or "moe" (a Soft-MoE layer of 8 experts in its place)."""

    def __init__(self, dims, widths=(64, 128, 192), core="moe"):
        super().__init__()
        c_in, n_feat, n_scal = dims["channels"], dims["playerFeatures"], dims["scalars"]
        self.n_bins = len(dims["troopBins"])
        c1, c2, c3 = widths
        self.stem = nn.Conv2d(c_in, c1, 4, stride=4)
        self.s1 = nn.Sequential(Res(c1), Res(c1))
        self.d2, self.s2 = nn.Conv2d(c1, c2, 2, stride=2), nn.Sequential(Res(c2), Res(c2))
        self.d3, self.s3 = nn.Conv2d(c2, c3, 2, stride=2), nn.Sequential(Res(c3), Res(c3))
        self.embed = nn.Linear(n_feat + c2, D)
        self.encoder = nn.ModuleList(nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.0, batch_first=True,
                                                                norm_first=True) for _ in range(2))
        self.encoder_norm = nn.LayerNorm(D)
        self.scalars = nn.Sequential(nn.Linear(n_scal, D), nn.ReLU())
        self.ask = nn.Linear(2 * c3 + D, D)
        self.pool = nn.MultiheadAttention(D, 4, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(2 * c3 + 6 * D, 512), nn.ReLU())
        self.film = nn.Linear(512, 2 * c1)
        self.u3, self.u2 = nn.Conv2d(c3, c2, 1), nn.Conv2d(c2, c1, 1)
        self.shuffle = nn.Conv2d(c1, N_MAPS * 16, 1)
        self.skip = nn.Conv2d(c_in, N_MAPS, 1)
        self.type_head = nn.Linear(512, dims["acts"])
        self.bin_head = nn.Linear(512, 2 * self.n_bins)
        self.query, self.key = nn.Linear(512, D), nn.Linear(D, D)
        self.value_head = nn.Linear(512, 1)
        # Policy outputs start at zero: a uniform policy over legal actions.
        for layer in (self.type_head, self.bin_head, self.shuffle, self.skip, self.query):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Built last, so for a given seed both cores start every other layer from the same weights (the A/B).
        self.moe = core == "moe"
        self.core = SoftMoE(512) if self.moe else nn.Linear(512, 512)

    def forward(self, obs):
        x = _spatial_input(obs["spatial"], self.stem.weight)
        f1 = self.s1(self.stem(x))
        f2 = self.s2(self.d2(F.relu(f1)))
        f3 = self.s3(self.d3(F.relu(f2)))
        pooled = torch.cat([f3.mean((2, 3)), f3.amax((2, 3))], 1)
        players = obs["players"]
        exists = players[..., 0] > 0
        t = self.embed(torch.cat([players, land_features(obs, f2).to(players.dtype)], 2))
        for layer in self.encoder:
            t = layer(t, src_key_padding_mask=~exists)
        tokens = self.encoder_norm(t)
        s = self.scalars(obs["scalars"])
        ask = self.ask(torch.cat([pooled, s], 1)).unsqueeze(1)
        attended = self.pool(ask, tokens, tokens, key_padding_mask=~exists, need_weights=False)[0].squeeze(1)
        groups = [masked_mean(tokens, (players[..., k] > 0) & exists) for k in (2, 3, 4, 5)]  # self, friends, enemies, bots
        h = self.trunk(torch.cat([pooled, attended, *groups, s], 1))
        out = {}
        if self.moe:
            h, out["gates"] = self.core(h)
        else:
            h = self.core(h)
        h = F.relu(h)
        g = f2 + F.interpolate(self.u3(F.relu(f3)), scale_factor=2)
        g = f1 + F.interpolate(self.u2(F.relu(g)), scale_factor=2)
        gamma, beta = self.film(h).chunk(2, 1)
        g = g * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        maps = F.pixel_shuffle(self.shuffle(F.relu(g)), 4) + self.skip(x)
        ptr = torch.einsum("nd,npd->np", self.query(h), self.key(tokens)) / math.sqrt(D)
        return {**out, "type": self.type_head(h), "maps": maps.flatten(2),
                "bins": self.bin_head(h).view(-1, 2, self.n_bins), "ptr": ptr, "value": self.value_head(h).squeeze(1)}
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v`
Expected: all pass. The three network tests now run on v1, dense and moe.

- [ ] **Step 5: Check the sizes**

Run:

```bash
.venv/Scripts/python -c "from trainer.model import Policy, PolicyV1; from trainer.test_model import DIMS; [print(n, sum(p.numel() for p in m.parameters())) for n, m in (('v1', PolicyV1(DIMS)), ('dense', Policy(DIMS, core='dense')), ('moe', Policy(DIMS, core='moe')))]"
```

Expected: dense ≈ 0.8M more parameters than v1 (transformer ~0.39M, wider trunk), and moe ≈ 1.84M more than dense
(7 extra 512 × 512 experts plus the router). Note the numbers in the ledger.

- [ ] **Step 6: Full suite and commit**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (67 tests).

```bash
git add trainer/model.py trainer/test_model.py
git commit -m "Phase 3 network: a transformer over all players (features, buildings, map features over their land) and a dense or Soft-MoE core"
```

---

### Task 5: Networks follow the checkpoint's settings (`--core`, `--init`)

**Files:**
- Modify: `trainer/model.py` (`arch_of`, `new_net`, `load_net`)
- Modify: `trainer/ppo.py` (`--core`, `network_args`, `new_net`)
- Modify: `trainer/bc.py` (`--init`, `--core`, `network_args`, `new_net`, `run_info`)
- Test: `trainer/test_model.py`, `trainer/test_eval.py`, `trainer/test_bc.py`, `trainer/test_ppo.py`

**Interfaces:**
- Consumes: `given_flags`, `resume_args` (Task 1); `Policy`, `PolicyV1` (Tasks 3, 4).
- Produces:
  - `trainer.model.arch_of(args: dict) -> {"arch", "core", "widths"}`; a missing `"arch"` means `"v1"`;
  - `trainer.model.new_net(dims, args: dict) -> nn.Module`;
  - `trainer.model.load_net(path, dims, device)`, which builds the checkpoint's own network;
  - `trainer.ppo.network_args(p, argv, a) -> a`: `--init`'s network, else `arch = "v2"` with `--core`/`--widths`;
  - checkpoints of new runs hold `args["arch"]` and `args["core"]`.

- [ ] **Step 1: Write the failing tests**

`trainer/test_model.py` (add `import tempfile` and `from pathlib import Path`; import `load_net, new_net`):

```python
class LoadTest(unittest.TestCase):
    def test_checkpoints_load_as_the_network_they_hold(self):
        obs = fake_obs()
        with tempfile.TemporaryDirectory() as d:
            for args, kind in (({"widths": "64,128,192"}, PolicyV1),  # Phase 2 checkpoints have no "arch"
                               ({"arch": "v2", "core": "dense", "widths": "64,128,192"}, Policy),
                               ({"arch": "v2", "core": "moe", "widths": "64,128,192"}, Policy)):
                path = Path(d) / "net.pt"
                torch.save({"model": new_net(DIMS, args).state_dict(), "args": args}, path)
                net = load_net(path, DIMS)
                self.assertIsInstance(net, kind)
                self.assertEqual(getattr(net, "moe", False), args.get("core") == "moe")
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    self.assertEqual(tuple(net(obs)["value"].shape), (N,))
```

`trainer/test_eval.py` (import `from trainer.model import new_net` and `from trainer.test_model import DIMS`):

```python
    def test_old_and_new_networks_play_each_other(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, args in (("new", {"arch": "v2", "core": "moe", "widths": "64,128,192"}),
                               ("old", {"widths": "64,128,192"})):
                path = Path(d) / f"{name}.pt"
                torch.save({"model": new_net(DIMS, args).state_dict(), "args": args, "update": 1}, path)
                paths.append(str(path))
            r = play(paths, ONION2, episodes=2, workers=1, games_per_worker=2, seed="mix")
        self.assertEqual(r["episodes"], 2)
```

`trainer/test_bc.py`:
- Import `DIMS` from `trainer.test_model`, and `Policy, PolicyV1` from `trainer.model`.
- Turn `MainTest` into a fixture with helpers, keeping its existing test.
- Make CloneTest construct `Policy(dims, core="moe")` (the network BC now trains).

```python
class MainTest(unittest.TestCase):
    RUN = "unittest-bc"

    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        (self.data / "records").mkdir()
        shutil.copy(FIXTURE, self.data / "records" / "fixture.json.gz")
        entry = {"id": "fixture", "start": 0, "commit": None, "map": "x", "size": "Compact", "turns": 1,
                 "players": [{"clientID": "a", "publicID": None}, {"clientID": "b", "publicID": None}], "winner": "a"}
        (self.data / "index.jsonl").write_text(json.dumps(entry) + "\n")
        self.common = ["--run", self.RUN, "--workers", "1", "--games-per-worker", "1", "--val-workers", "0",
                       "--val-share", "0", "--port", "0", "--log-every", "1", "--save-every", "2"]
        self.seeds = []

    def tearDown(self):
        for d in ("checkpoints", "runs"):
            shutil.rmtree(ROOT / d / self.RUN, ignore_errors=True)
        shutil.rmtree(self.data, ignore_errors=True)

    def run_main(self, argv):
        real = bc.VecEnv

        def spy(*args, **kwargs):
            self.seeds.append(kwargs["seed"])
            return real(*args, **kwargs)

        with mock.patch.multiple(crawl, DATA=self.data, RECORDS=self.data / "records",
                                 INDEX=self.data / "index.jsonl"), mock.patch.object(bc, "VecEnv", spy):
            main(argv)

    def latest(self):
        return torch.load(ROOT / "checkpoints" / self.RUN / "latest.pt", weights_only=False)

    def test_train_save_and_resume(self):
        self.run_main(self.common + ["--updates", "3"])
        first = self.latest()
        self.assertEqual((first["kind"], first["update"]), ("bc", 3))
        self.assertEqual((first["args"]["arch"], first["args"]["core"]), ("v2", "moe"))
        self.run_main(self.common + ["--updates", "4", "--resume"])
        self.assertEqual(self.latest()["update"], 4)
        self.assertTrue((ROOT / "runs" / self.RUN).is_dir())
        self.assertNotEqual(self.seeds[0], self.seeds[1], "a resumed run replays the records in a new order")

    def test_fine_tuning_keeps_the_checkpoints_network(self):
        init = ROOT / "checkpoints" / self.RUN / "init.pt"
        init.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": PolicyV1(DIMS).state_dict(), "args": {"widths": "64,128,192"}}, init)
        self.run_main(self.common + ["--init", str(init), "--updates", "1"])
        self.assertEqual(self.latest()["args"]["arch"], "v1")
        with self.assertRaisesRegex(SystemExit, "drop --core"):
            self.run_main(self.common + ["--init", str(init), "--core", "moe", "--updates", "1"])
```

`trainer/test_bc.py`, the control-page test:

```python
    def test_the_control_page_says_what_is_training(self):
        a = bc.parse_args(["--workers", "14", "--games-per-worker", "6"])
        a.arch = "v2"
        info = bc.run_info(a, ["r"] * 3, ["v"])
        self.assertEqual(info["Method"], "behaviour cloning (imitating humans)")
        self.assertEqual(info["Network"], "player transformer, Soft-MoE core (8 experts)")
        self.assertEqual(info["Starts from"], "random weights")
        self.assertEqual(info["Data"], "3 ranked 1v1 replays, 1 held out")
        self.assertEqual(info["Games at once"], "84 (14 workers x 6)")
```

`trainer/test_ppo.py`: in `SmokeTest.test_train_save_and_resume`, after the `first["update"]` check, add:

```python
        self.assertEqual((first["args"]["arch"], first["args"]["core"]), ("v2", "moe"), "a new run records its network")
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model trainer.test_eval trainer.test_bc trainer.test_ppo -v`
Expected:
- `ImportError: cannot import name 'load_net'`... from `new_net` in test_model and test_eval;
- `KeyError: 'arch'` in the BC and PPO smoke tests;
- `KeyError: 'Network'` in the control-page test.

- [ ] **Step 3: Implement**

`trainer/model.py`: replace `load_net` with:

```python
def arch_of(args):
    """The network a run's settings describe. Phase 2 checkpoints predate "arch": they are PolicyV1."""
    return {"arch": args.get("arch", "v1"), "core": args.get("core", "dense"), "widths": args["widths"]}


def new_net(dims, args):
    """A fresh network of the kind a run's settings (a dict like a checkpoint's "args") describe."""
    spec = arch_of(args)
    widths = tuple(int(w) for w in spec["widths"].split(","))
    return PolicyV1(dims, widths) if spec["arch"] == "v1" else Policy(dims, widths, spec["core"])


def load_net(path, dims, device="cuda"):
    """A frozen policy from a trainer.bc or trainer.ppo checkpoint, in eval mode: the network it holds."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = new_net(dims, ckpt["args"])
    net.load_state_dict(ckpt["model"])
    return net.to(device).to(memory_format=torch.channels_last).eval().requires_grad_(False)
```

`trainer/ppo.py`:
- Import `ActionDist, arch_of, load_net, new_net` from `trainer.model`; `PolicyV1` is no longer imported.
- In `parser()`, after `--widths`, add:

```python
    p.add_argument("--core", choices=("dense", "moe"), default="moe", help="the core of a new network (not with --init)")
```

Add after `resume_args`:

```python
def network_args(p, argv, a):
    """A new run's network settings. With --init they are the checkpoint's: its weights decide the network, and a
    different --core or --widths given alongside is refused. Without it, a Phase 3 network from --core and
    --widths."""
    if not a.init:
        a.arch = "v2"
        return a
    spec = arch_of(torch.load(a.init, map_location="cpu", weights_only=False)["args"])
    given = given_flags(p, argv)
    clash = [k for k in ("core", "widths") if k in given and given[k] != spec[k]]
    if clash:
        raise SystemExit(f"--init {a.init} holds a {spec['arch']} network ({spec['core']} core, widths "
                         f"{spec['widths']}): drop {', '.join('--' + k for k in clash)}")
    vars(a).update(spec)
    return a
```

In `main`:
- After the `if a.resume:` block, add `else:` with `a = network_args(parser(), argv, a)`.
- Construct the model with `model = new_net(envs.dims, vars(a)).to(device)`.

`trainer/bc.py`:
- Import `new_net` instead of `PolicyV1`: `from trainer.model import ATTACK, BOAT, BUILD, SPAWN, ActionDist, new_net`.
- Import `network_args` from `trainer.ppo`.
- In `parser()`, after `--resume`, add:

```python
    p.add_argument("--init", help="start from this checkpoint's weights and network, with a fresh optimizer")
    p.add_argument("--core", choices=("dense", "moe"), default="moe", help="the core of a new network (not with --init)")
```

In `main`:
- After the `if a.resume:` block, add `else:` with `a = network_args(parser(), argv, a)`.
- Construct `model = new_net(envs.dims, vars(a)).to(device)`.
- After the `if state:` block that restores the model, add:

```python
    elif a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
```

Replace `run_info` with:

```python
def run_info(a, train, val):
    """What the control page says about the run."""
    network = "the Phase 2 network" if a.arch == "v1" else (
        "player transformer, " + ("Soft-MoE core (8 experts)" if a.core == "moe" else "dense core"))
    return {"Name": a.run, "Method": "behaviour cloning (imitating humans)", "Network": network,
            "Starts from": a.init or "random weights",
            "Data": f"{len(train):,} ranked 1v1 replays, {len(val):,} held out",
            "Games at once": f"{a.workers * a.games_per_worker} ({a.workers} workers x {a.games_per_worker})",
            "Updates": f"{a.updates:,}"}
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_model trainer.test_eval trainer.test_bc trainer.test_ppo -v`
Expected: all pass.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (70 tests).

```bash
git add trainer/model.py trainer/ppo.py trainer/bc.py trainer/test_model.py trainer/test_eval.py trainer/test_bc.py trainer/test_ppo.py
git commit -m "Checkpoints say which network they hold: new runs pick --core, --init keeps the checkpoint's network, Phase 2 ones load as PolicyV1"
```

---

### Task 6: The BC plateau fix: 8 windows per step, warmup then cosine, one pass

**Files:**
- Modify: `trainer/bc.py`:
  - `bc_loss` normalisation;
  - new `lr_at`, `one_pass` and `backward_windows`;
  - parser defaults and flags;
  - `run_info`'s "Updates";
  - `main`: the old-checkpoint check, the record seed, the training loop and the lr metric.
- Test: `trainer/test_bc.py`

**Interfaces:**
- Consumes: `new_net` (Task 5); `fake_obs`, `DIMS` and `randomized` from `trainer/test_model.py` (tests only).
- Produces:
  - `lr_at(update, a) -> float`, from `a.lr`, `a.warmup` and `a.updates`;
  - `one_pass(entries, a) -> int`, from `a.workers`, `a.games_per_worker` and `a.accum`;
  - `backward_windows(model, windows, a, meter)`: windows are `(obs, labels, legal, target)` and `a` has `accum`
    and `vf_coef`;
  - new flags `--accum` (default 8) and `--warmup` (default 500); `--updates` defaults to one pass;
  - new defaults `--val-every 250`, `--log-every 25`, `--save-every 250`;
  - the record-order seed is `f"bc-{a.seed}-u{start}"`;
  - `bc_loss`'s NLL is divided by the number of rows.

- [ ] **Step 1: Write the failing tests** (`trainer/test_bc.py`)

Add `import argparse` and import `backward_windows, lr_at, one_pass` from `trainer.bc`. Also import
`fake_obs, randomized` from `trainer.test_model`, and `N` as `ROWS`.

```python
class ScheduleTest(unittest.TestCase):
    def test_warmup_then_cosine_to_zero(self):
        a = argparse.Namespace(lr=1e-3, warmup=10, updates=110)
        lrs = [lr_at(u, a) for u in range(111)]
        self.assertAlmostEqual(lrs[0], 1e-4)
        self.assertAlmostEqual(lrs[9], 1e-3)
        self.assertAlmostEqual(lrs[10], 1e-3)
        self.assertAlmostEqual(lrs[60], 5e-4)
        self.assertAlmostEqual(lrs[110], 0.0)
        self.assertTrue(all(x >= y for x, y in zip(lrs[10:], lrs[11:])))

    def test_one_pass_counts_the_windows_of_every_record(self):
        a = argparse.Namespace(workers=2, games_per_worker=3, accum=4)
        self.assertEqual(one_pass([{"turns": 5 * 120}, {"turns": 5 * 360}], a), 20)  # 480 windows / 24 per update


class AccumulationTest(unittest.TestCase):
    def test_accumulated_windows_give_the_gradient_of_one_batch(self):
        torch.manual_seed(0)
        model = randomized(Policy(DIMS, core="moe")).cuda().to(memory_format=torch.channels_last)
        windows = []
        for i, n_legal in enumerate((6, 2, 4)):  # unequal legal rows: a mean of window means would differ
            labels = torch.zeros(ROWS, 4, dtype=torch.long, device="cuda")
            legal = torch.arange(ROWS, device="cuda") < n_legal
            windows.append((fake_obs(seed=i), labels, legal, torch.full((ROWS,), 0.5, device="cuda")))
        model.zero_grad()
        backward_windows(model, windows, argparse.Namespace(accum=3, vf_coef=0.5), Meter())
        accumulated = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        obs = {k: torch.cat([w[0][k] for w in windows]) for k in windows[0][0]}
        bc_loss(model, obs, *(torch.cat([w[i] for w in windows]) for i in (1, 2, 3)), 0.5)[0].backward()
        for p, g in zip(model.parameters(), accumulated):
            self.assertLessEqual(float((p.grad - g).norm()), 0.05 * float(p.grad.norm()) + 1e-6)
```

In `MainTest`:
- Add `"--accum", "2", "--warmup", "1"` to `self.common`.
- In `test_train_save_and_resume`, add `self.assertEqual(first["global_step"], 3 * 2 * 2)` (3 updates of 2 windows of
  2 rows) and `self.assertEqual(self.seeds[0], "bc-1-u0")` (runs with one seed replay one order).

Add two tests:

```python
    def test_a_resume_refuses_flags_it_would_ignore(self):
        self.run_main(self.common + ["--updates", "1"])
        with self.assertRaisesRegex(SystemExit, "--lr would be ignored"):
            self.run_main(self.common + ["--updates", "2", "--lr", "1e-3", "--resume"])

    def test_old_bc_checkpoints_cannot_resume(self):
        path = ROOT / "checkpoints" / self.RUN / "latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"args": {"run": self.RUN, "widths": "64,128,192"}, "update": 5}, path)  # no "accum"
        with self.assertRaisesRegex(SystemExit, "predates.*--init"):
            self.run_main(["--run", self.RUN, "--resume"])
```

In the control-page test, parse `["--workers", "14", "--games-per-worker", "6", "--updates", "10"]` and add
`self.assertEqual(info["Updates"], "10 of 8 windows each")`.

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v`
Expected: ERROR, `ImportError: cannot import name 'backward_windows' from 'trainer.bc'`.

- [ ] **Step 3: Implement** in `trainer/bc.py`

- Add `import math`.
- Update the docstring's usage lines:

```
  Train:   .venv/Scripts/python -m trainer.bc --run bc3-moe --core moe   (one pass over data/ from trainer.crawl)
  Tune:    .venv/Scripts/python -m trainer.bc --run probe --init checkpoints/bc/update0060000.pt --updates 2000
  Resume:  .venv/Scripts/python -m trainer.bc --run bc3-moe --resume
```

Parser changes:
- `--val-every` default 250.
- `--updates`: `type=int, default=None, help="optimizer steps (default: one pass over the training records)"`.
- After `--updates`, add:
  - `p.add_argument("--accum", type=int, default=8, help="replay windows averaged per optimizer step")`;
  - `--lr` help `"peak learning rate"`;
  - `p.add_argument("--warmup", type=int, default=500, help="updates of linear warmup before the cosine decay")`.
- `--seed` help: `"network init and record order: runs with one seed replay the same records in the same order"`.
- `--log-every` default 25.
- `--save-every` default 250, help `"updates between checkpoints (every 10th is kept)"`.

In `bc_loss`, set the docstring to `"""-log pi(label) summed over the rows that can enter the loss and divided by all
rows, plus the value loss. Every window has the same rows, so k accumulated windows give the gradient of one batch of
all of them."""` and change the NLL line to:

```python
    nll = -(logp * legal).sum() / len(legal)
```

Add after `measure`:

```python
def lr_at(update, a):
    """The learning rate of an update: a linear warmup over a.warmup updates, then a cosine decay to 0 at a.updates."""
    if update < a.warmup:
        return a.lr * (update + 1) / a.warmup
    done = min(1.0, (update - a.warmup) / max(1, a.updates - a.warmup))
    return a.lr * 0.5 * (1 + math.cos(math.pi * done))


def one_pass(entries, a):
    """Updates that replay every record of `entries` about once: each update takes a.accum windows of 5 turns from every
    game slot."""
    windows = sum(e["turns"] for e in entries) / 5
    return max(1, round(windows / (a.workers * a.games_per_worker * a.accum)))


def backward_windows(model, windows, a, meter):
    """Accumulates the gradient of a.accum windows, each (obs, labels, legal rows, value targets). Their mean loss gives
    the gradient of one batch of all their rows."""
    for obs, labels, legal, target in windows:
        loss, dist, logp, value = bc_loss(model, obs, labels, legal, target, a.vf_coef)
        (loss / a.accum).backward()
        measure(meter, obs, labels, legal, dist, logp, value, target)
```

Change `run_info`'s last entry to `"Updates": f"{a.updates:,} of {a.accum} windows each"`.

In `main`, right after `state = torch.load(...)` in the resume block, add:

```python
        if "accum" not in state["args"]:
            raise SystemExit(f"checkpoints/{a.run} predates the Phase 3 optimizer (accumulation, learning-rate "
                             f"schedule), so it cannot resume. Start a new run from it: --run <new name> --init "
                             f"checkpoints/{a.run}/latest.pt")
```

Next, in `main`:
- Keep `train_entries = [e for e in entries if not is_val(e["id"], a.val_share)]` and build `train` from it.
- Before the comment about the record order, add `if a.updates is None: a.updates = one_pass(train_entries, a)`.
- Replace that comment and the seed:

```python
    # Runs with one --seed replay the same records in the same order (the dense/MoE A/B), and a resumed run draws a new
    # order: with the old one, every stop and resume would replay the same records.
    start = state["update"] if state else 0
    envs = VecEnv(RANKED_1V1, a.workers, a.games_per_worker, seed=f"bc-{a.seed}-u{start}", device=device,
                  records=train, log_dir=run_dir / "workers")
```

Add, inside `main` after `elo_rows`:

```python
    def windows():
        """The next a.accum windows; the workers replay each next window while the GPU trains on the current one."""
        nonlocal global_step
        for _ in range(a.accum):
            obs, labels = envs.obs, envs.labels
            target = value_targets(envs, a.gamma, device)
            legal = label_legal(obs, labels) & (elo_rows(envs) >= a.min_elo)
            envs.send()
            yield obs, labels, legal, target
            for i in envs.recv()[3]:
                done["records"] += 1
                done["desync"] += int(i["desync"])
                done["errors"] += int(i["error"] is not None)
            global_step += envs.N
```

Replace the body of the training `while` loop (from `t0 = time.perf_counter()` through `control.throttle(...)`) with:

```python
            t0 = time.perf_counter()
            for group in opt.param_groups:
                group["lr"] = lr_at(update, a)
            opt.zero_grad(set_to_none=True)
            backward_windows(model, windows(), a, meter)
            nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
            opt.step()
            update += 1
            control.throttle(time.perf_counter() - t0)
```

In the logging block, add `"train/lr": opt.param_groups[0]["lr"]`, and make the speed
`"perf/samples_per_s": a.log_every * a.accum * envs.N / (time.perf_counter() - t_log)`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v`
Expected: all pass.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (75 tests).

```bash
git add trainer/bc.py trainer/test_bc.py
git commit -m "BC plateau fix: each step averages 8 windows, the learning rate warms up then decays (cosine) over one pass, one seed one record order"
```

---

### Task 7: `trainer.bc --score`, a fixed scoring set with NLL per head and bucket

**Files:**
- Modify: `trainer/bc.py`:
  - `bc_loss` also returns the gates;
  - `measure` adds per-head NLL, the Compact/Normal buckets and router use;
  - `backward_windows` and `validate` pass the gates on;
  - new `SCORE_CUTOFF`, `SCORE_LAYOUT`, `score_records`, `score`, `table` and `score_main`;
  - the `--score` flag and the early return in `main`.
- Test: `trainer/test_bc.py`

**Interfaces:**
- Consumes: `load_net` (Task 5); `trainer.eval.tag` (Task 2).
- Produces:
  - `bc_loss(...) -> (loss, dist, logp, value, gates or None)`;
  - `measure(m, obs, labels, legal, dist, logp, value, target, gates=None)`;
  - `score(paths, records, workers, games_per_worker, steps, device="cuda") -> {"checkpoints", "records", "games",
    "rows", "metrics": [dict per checkpoint]}`;
  - `table(result) -> str`;
  - CLI `trainer.bc --score CKPT [CKPT ...]` prints the table and writes `runs/score/<YYYYmmdd-HHMMSS>.json`.

- [ ] **Step 1: Write the failing test** (`trainer/test_bc.py`; add `import math`, import `score, table` from
`trainer.bc` and `new_net` from `trainer.model`)

```python
class ScoreTest(unittest.TestCase):
    def test_scores_are_repeatable_and_per_head(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, args in (("old", {"widths": "64,128,192"}),
                               ("moe", {"arch": "v2", "core": "moe", "widths": "64,128,192"})):
                torch.manual_seed(0)
                path = Path(d) / name / "update0000001.pt"
                path.parent.mkdir()
                torch.save({"model": new_net(DIMS, args).state_dict(), "args": {**args, "run": name}, "update": 1},
                           path)
                paths.append(path)
            first = score(paths, [FIXTURE], workers=1, games_per_worker=1, steps=30)
            second = score(paths, [FIXTURE], workers=1, games_per_worker=1, steps=30)
            text = table(first)
        self.assertEqual((first["rows"], first["games"], first["records"]), (60, 1, 1))
        for a, b in zip(first["metrics"], second["metrics"]):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assertTrue(math.isclose(a[k], b[k], abs_tol=1e-4) or (math.isnan(a[k]) and math.isnan(b[k])), k)
        old, moe = first["metrics"]
        for k in ("bc/nll", "bc/nll_type", "bc/nll_compact", "bc/type_acc", "bc/value_mse", "value/outcome_corr"):
            self.assertIn(k, old)
        self.assertFalse(any(k.startswith("router/") for k in old), "the Phase 2 network has no router")
        self.assertAlmostEqual(sum(moe[f"router/compact_{i}"] for i in range(8)), 1.0, places=4)
        self.assertIn("old 1", text)
        self.assertIn("bc/nll_type", text)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/Scripts/python -m unittest trainer.test_bc.ScoreTest -v`
Expected: ERROR, `ImportError: cannot import name 'score' from 'trainer.bc'`.

- [ ] **Step 3: Implement** in `trainer/bc.py`

Imports: add `import datetime`, `import json`, `from trainer.eval import tag`, and `load_net` to the
`trainer.model` import. Update the docstring's usage with
`  Score:   .venv/Scripts/python -m trainer.bc --score checkpoints/bc/update0060000.pt checkpoints/bc3-moe/latest.pt`.

After `PHASES`, add:

```python
SCORE_CUTOFF = 1790812800000  # 2026-10-01 00:00 UTC, in ms: the scoring set comes from the Phase 2 crawl's games
SCORE_LAYOUT = {"workers": 12, "games_per_worker": 6, "steps": 2000}  # fixed, so every scoring run sees the same rows
```

In `parser()`, after `--init`, add:

```python
    p.add_argument("--score", nargs="+", metavar="CHECKPOINT",
                   help="score these checkpoints on the fixed held-out set (~10 min) instead of training")
```

`bc_loss`: return `out.get("gates")` as a fifth value:
`return nll + vf_coef * 0.5 * (value - target).pow(2).mean(), dist, logp, value, out.get("gates")`.
Callers unpack five:
- `backward_windows` passes `gates` to `measure`;
- `validate` (`_, dist, logp, value, gates = ...`) passes `gates` to `measure`;
- CloneTest's `loss, *_ =` is unchanged.

Replace `measure` with:

```python
@torch.no_grad()
def measure(m, obs, labels, legal, dist, logp, value, target, gates=None):
    """For one batch:
    - label statistics;
    - accuracy and NLL per head;
    - NLL per bucket (game phase and map size);
    - the router's mean weight per expert and bucket, for MoE networks."""
    t, cell, tgt, b = labels.unbind(1)
    n, nl = len(t), legal.sum()
    m.add("labels/out_of_set", (t == -1).sum(), n)
    m.add("labels/unmapped", (t == -2).sum(), n)
    m.add("labels/illegal", ((t >= 0) & ~legal).sum(), n)
    m.add("labels/noop", ((t == 0) & legal).sum(), nl)
    m.add("bc/nll", -(logp * legal).sum(), nl)
    m.add("bc/nll_type", -(dist.type.log_prob(torch.where(legal, t, 0)) * legal).sum(), nl)
    m.add("bc/type_acc", ((dist.type.logits.argmax(1) == t) & legal).sum(), nl)
    m.add("bc/act_rate_human", ((t > 0) & legal).sum(), nl)
    m.add("bc/act_rate_policy", ((1 - dist.type.probs[:, 0]) * legal).sum(), nl)
    lc = dist.cells.log_prob(cell.clamp(0, dist.cells.logits.shape[-1] - 1)[:, None].expand(-1, 3))  # [N, maps]
    lb = dist.bins.log_prob(b.clamp(0, dist.bins.logits.shape[-1] - 1)[:, None].expand(-1, 2))  # [N, attack/boat]
    rows = legal & (t == ATTACK)
    m.add("bc/attack_target_acc", ((dist.ptr.logits.argmax(1) == tgt) & rows).sum(), rows.sum())
    m.add("bc/nll_attack_target", -(dist.ptr.log_prob(torch.where(rows, tgt, 0)) * rows).sum(), rows.sum())
    m.add("bc/attack_bin_acc", ((dist.bins.logits[:, 0].argmax(1) == b) & rows).sum(), rows.sum())
    m.add("bc/nll_attack_bin", -(lb[:, 0] * rows).sum(), rows.sum())
    rows = legal & (t == BOAT)
    m.add("bc/boat_bin_acc", ((dist.bins.logits[:, 1].argmax(1) == b) & rows).sum(), rows.sum())
    m.add("bc/nll_boat_bin", -(lb[:, 1] * rows).sum(), rows.sum())
    for typ, name in ((SPAWN, "spawn"), (BOAT, "boat"), (BUILD, "city")):
        rows = legal & (t == typ)
        hit = dist.cells.logits[:, CELL_HEAD[typ]].argmax(1) == cell
        m.add(f"bc/{name}_cell_acc", (hit & rows).sum(), rows.sum())
        m.add(f"bc/nll_{name}_cell", -(lc[:, CELL_HEAD[typ]] * rows).sum(), rows.sum())
    phase = torch.where(obs["scalars"][:, 1] > 0, 0, 1 + (obs["scalars"][:, 0] * 3).long().clamp(max=2))
    buckets = {name: phase == i for i, name in enumerate(PHASES)}
    normal = obs["scalars"][:, 14] > 0.5  # tiles per cell / 64: Normal maps 1, Compact 0.25
    buckets.update(compact=~normal, normal=normal)
    for name, inside in buckets.items():
        rows = legal & inside
        m.add(f"bc/nll_{name}", -(logp * rows).sum(), rows.sum())
        if gates is not None:
            for i in range(gates.shape[1]):
                m.add(f"router/{name}_{i}", (gates[:, i].float() * inside).sum(), inside.sum())
    m.add("bc/value_mse", (value - target).pow(2).sum(), n)
```

Add before `main`:

```python
def score_records():
    """The fixed scoring set: the held-out records (5%, by game ID) of games started before SCORE_CUTOFF, by game ID.
    Later crawls do not change it; a purge can drop a record (the result counts them)."""
    entries = [e for e in crawl.load_index() if e["start"] < SCORE_CUTOFF and is_val(e["id"], 0.05)]
    paths = [crawl.record_path(e["id"]) for e in sorted(entries, key=lambda e: e["id"])]
    return [p for p in paths if p.exists()]


def score(paths, records, workers, games_per_worker, steps, device="cuda"):
    """Every checkpoint's statistics (measure, plus the value/outcome correlation) on the same replayed rows. The rows
    depend only on the records, the layout and the seed "score", so every comparison is paired."""
    envs = VecEnv(RANKED_1V1, workers, games_per_worker, seed="score", device=device, records=records)
    try:
        nets = [load_net(p, envs.dims, device) for p in paths]
        meters, values, outcomes, games = [Meter() for _ in nets], [[] for _ in nets], [], set()
        with torch.no_grad():
            for _ in range(steps):
                obs, labels = envs.obs, envs.labels
                target = value_targets(envs, 0.999, device)
                legal = label_legal(obs, labels)
                for net, m, v in zip(nets, meters, values):
                    _, dist, logp, value, gates = bc_loss(net, obs, labels, legal, target, 0.0)
                    measure(m, obs, labels, legal, dist, logp, value, target, gates)
                    v.append(value.cpu().numpy())
                outcomes.append(envs.outcome.reshape(-1).copy())
                games.update(envs.record_of)
                envs.step()
    finally:
        envs.close()
    o = np.concatenate(outcomes)
    metrics = []
    for m, v in zip(meters, values):
        stats, v = m.means(), np.concatenate(v)
        stats["value/outcome_corr"] = float(np.corrcoef(v, o)[0, 1]) if v.std() > 0 and o.std() > 0 else float("nan")
        metrics.append(stats)
    return {"checkpoints": [str(p) for p in paths], "records": len(records), "games": len(games), "rows": len(o),
            "metrics": metrics}


def table(result):
    """The scores as text: one row per statistic and one column per checkpoint (its run and update), then the
    routers' mean expert weights per bucket."""
    names = [tag(p) for p in result["checkpoints"]]
    keys = sorted({k for s in result["metrics"] for k in s if not k.startswith("router/")})
    w = max(len(k) for k in keys)
    lines = [f"{result['games']} games, {result['rows']:,} rows ({result['records']} held-out records before "
             f"2026-10-01)", " " * w + "".join(f"{n:>18}" for n in names)]
    lines += [k.ljust(w) + "".join(f"{s.get(k, float('nan')):18.4f}" for s in result["metrics"]) for k in keys]
    for name, s in zip(names, result["metrics"]):
        for bucket in (*PHASES, "compact", "normal"):
            weights = [s[k] for k in sorted(s) if k.startswith(f"router/{bucket}_")]
            if weights:
                lines.append(f"router {name} {bucket:>7}: " + " ".join(f"{x:.2f}" for x in weights))
    return "\n".join(lines)


def score_main(paths):
    records = score_records()
    if not records:
        raise SystemExit("no held-out records of games before 2026-10-01 in data/: run trainer.crawl first")
    result = score(paths, records, **SCORE_LAYOUT)
    out = ROOT / "runs" / "score" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(table(result))
    print(f"saved {out}")
```

In `main`, right after `a = parse_args(argv)`:

```python
    if a.score:
        return score_main(a.score)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_bc -v`
Expected: all pass.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (76 tests).

```bash
git add trainer/bc.py trainer/test_bc.py
git commit -m "trainer.bc --score: a fixed held-out scoring set, NLL per head and per bucket, and router use per bucket"
```

---

### Task 8: Speed and memory checks, then the docs

**Files:**
- Modify: `CLAUDE.md` (training commands and notes), `docs/PLAN.md` (Phase 3A progress)
- Scratch: `checkpoints/speed/`, `runs/speed-*` (deleted at the end)

Measurements only: no production code changes unless the decision point below says so.

- [ ] **Step 1: Full suites**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (76 tests).

From `OpenFront_Source_Code/openfrontio/`:
- `node --import tsx --test "../../bridge/*.test.ts"`
- `node node_modules/typescript/bin/tsc -p ../../tsconfig.json`

Expected: 32 bridge tests pass and the typecheck is clean (the bridge is unchanged).

- [ ] **Step 2: Random-weight Phase 3 checkpoints for the speed checks** (speed does not depend on the weights)

```bash
.venv/Scripts/python - <<'EOF'
from pathlib import Path
import torch
from trainer.model import new_net
from trainer.test_model import DIMS
for core in ("dense", "moe"):
    args = {"arch": "v2", "core": core, "widths": "64,128,192", "run": f"speed-{core}"}
    torch.manual_seed(0)
    path = Path(f"checkpoints/speed/{core}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": new_net(DIMS, args).state_dict(), "args": args, "update": 0}, path)
EOF
```

- [ ] **Step 3: PPO speed and VRAM, 10 updates each** (~3 min each, Phase 2's `ppo-bc2` settings)

For each pair of `<name>` and `<init>`, run the probe while sampling the GPU memory:
- `speed-v1`: `checkpoints/bc/update0060000.pt`;
- `speed-dense`: `checkpoints/speed/dense.pt`;
- `speed-moe`: `checkpoints/speed/moe.pt`.

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > runs/<name>-vram.log &
.venv/Scripts/python -m trainer.ppo --run <name> --setup ranked1v1 --init <init> --opponent <init> --kl-coef 0.2 --lr 1e-4 --ent-coef 0 --workers 16 --games-per-worker 2 --updates 10 --port 0
kill %1
```

Read the decisions per second, averaged over updates 3–10, and the peak memory:

```bash
.venv/Scripts/python - <<'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for name in ("speed-v1", "speed-dense", "speed-moe"):
    acc = EventAccumulator(f"runs/{name}"); acc.Reload()
    v = [e.value for e in acc.Scalars("perf/decisions_per_s")][2:]
    mem = max(int(l) for l in open(f"runs/{name}-vram.log") if l.strip())
    print(name, round(sum(v) / len(v), 1), "decisions/s,", mem, "MiB peak")
EOF
```

Expected:
- every run completes without running out of memory;
- the Phase 3 networks keep ≥ 75% of the v1 number and of Phase 2's ~190 decisions/s (the Gate 3A bar);
- moe is within 5% of dense (the MoE rule).

- [ ] **Step 4: BC speed, 30 updates each** (~2 min each)

```bash
.venv/Scripts/python -m trainer.bc --run speed-bc-v1 --init checkpoints/bc/update0060000.pt --workers 14 --games-per-worker 6 --val-workers 0 --updates 30 --log-every 5 --port 0
.venv/Scripts/python -m trainer.bc --run speed-bc-moe --core moe --workers 14 --games-per-worker 6 --val-workers 0 --updates 30 --log-every 5 --port 0
```

Read `perf/samples_per_s` from both runs (same script as above, tag changed, skipping the first point).
Expected: both near Phase 2's ~860 rows/s without validation. The CPU workers set the pace, so a one-pass refresh
stays at ~12 h.

- [ ] **Step 5: Decision point** (rule on it in the ledger)

- **The checks pass:** go on.
- **The Phase 3 networks fall below 75% of the v1 run:**
  - Profile one update (`torch.profiler`, 3 updates) to see whether the player transformer is the cost.
  - If it is, the ≤64-token cap of PLAN 3A comes in before any BC refresh, through its own short plan: the
    selection rule decides which players can be attacked, so it needs its own tests.
  - Otherwise, report the profile to the user.
- **PPO runs out of memory:** add `--minibatches 12` to the Phase 3 PPO commands. It cuts activation memory by a
  third without touching the code; record it as the setting.

- [ ] **Step 6: Clean up the speed runs**

```bash
rm -rf checkpoints/speed checkpoints/speed-v1 checkpoints/speed-dense checkpoints/speed-moe checkpoints/speed-bc-v1 checkpoints/speed-bc-moe runs/speed-*
```

- [ ] **Step 7: Docs**

`CLAUDE.md`, training commands. Replace the `trainer.bc --run bc` line with:

```bash
.venv/Scripts/python -m trainer.bc --run bc3-moe --core moe --workers 14 --games-per-worker 6   # behaviour cloning: one pass over data/ (~12 h); same control page
.venv/Scripts/python -m trainer.bc --run probe --init checkpoints/bc/update0060000.pt --updates 2000 --warmup 100   # fine-tune a checkpoint (its network, a fresh optimizer)
.venv/Scripts/python -m trainer.bc --score checkpoints/bc/update0060000.pt checkpoints/bc3-moe/latest.pt   # NLL per head and bucket on the fixed held-out set (~10 min); table + runs/score/*.json
```

`CLAUDE.md` notes:
- **BC:** an update averages `--accum` windows (default 8). The learning rate warms up, then decays over `--updates`
  (default: one pass). BC checkpoints from before Phase 3 cannot resume; start from them with `--init`.
- **Old checkpoints:** a checkpoint without `"arch"` (Phase 2) loads as `PolicyV1`, which rebuilds its 22 input
  channels from today's 49 (`v1_spatial`).
- **Simulation facts:** structures of all players stay ≥ `structureMinDist()` = 15 tiles apart, so a cell (≤ 8 × 8
  tiles) holds at most one. The building channels and `v1_spatial` rely on it; recheck it on upgrades.

`docs/PLAN.md`: add under 3A a dated "**3A progress (code)**" note:
- what was built;
- the rulings from the ledger (49 channels; the scoring set; ~12 h per pass);
- the measured speed and memory;
- the runs that remain.

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md docs/PLAN.md
git commit -m "Phase 3A code done: speed and memory checks, commands and notes"
```

---

## After the code: the runs (each starts only after the user's OK)

1. **BC plateau probe** (the user, ~1 h).
   - Command:
     `.venv/Scripts/python -m trainer.bc --run bc-probe --init checkpoints/bc/update0060000.pt --workers 14 --games-per-worker 6 --updates 2000 --warmup 100`.
   - Then score it (~10 min, in session):
     `.venv/Scripts/python -m trainer.bc --score checkpoints/bc/update0060000.pt checkpoints/bc-probe/latest.pt`.
   - A clear NLL drop below the Phase 2 clone's number on the same rows means the plateau was the optimizer.
2. **Purge and crawl before the refresh nights, never between them.** The two A/B runs must see the same data in the
   same order. The purge is due by about 2026-10-10.
3. **BC refresh, two nights** (the user, ~12 h each). Each night runs one arm in the restart loop, in PowerShell from
   the repo root (the `bc3-dense` arm with `--core dense`, then the `bc3-moe` arm with `--core moe`):

```powershell
.venv/Scripts/python -m trainer.bc --run bc3-moe --core moe --workers 14 --games-per-worker 6
for ($i = 1; ($LASTEXITCODE -ne 0) -and ($i -le 5); $i++) { Start-Sleep 60; .venv/Scripts/python -m trainer.bc --run bc3-moe --resume }
```

4. **A/B scoring** (~15 min, in session):
   - Command: `trainer.bc --score` on the Phase 2 clone and both arms' `latest.pt` plus their last two kept
     snapshots.
   - The MoE rule: keep the MoE unless its overall NLL is above the dense arm's by more than 0.02 (Phase 2's
     snapshots spread over 0.035 at a constant learning rate), or it was more than 5% slower in Task 8.
5. **Gate 3A**, on the kept arm:
   - lower NLL and higher attack-target accuracy than the Phase 2 clone in the scoring table;
   - ≥ 0.545 against the Phase 2 clone over 512 games (the user, ~45 min):
     `.venv/Scripts/python -m trainer.eval --a checkpoints/bc3-<arm>/latest.pt --b checkpoints/bc/update0060000.pt --episodes 512 --workers 16 --games-per-worker 2 --record 1`;
   - decisions/s ≥ 75% of Phase 2's (Task 8).
6. **Record the results** in `docs/PLAN.md`, then plan 3B.

## Verification (whole stage)

- **Tests:** `.venv/Scripts/python -m unittest discover -s trainer -t .` is green, with 76 tests. The bridge suite and
  typecheck are green and unchanged.
- **Every PLAN 3A item has a test:**

  | PLAN 3A item | Test |
  |---|---|
  | per-owner channels equal a per-cell lookup; each building lands in its type-and-relation channel with its value | `test_owner_and_building_channels_follow_their_definitions` |
  | the rebuilt old channels match the Phase 2 code | `test_old_networks_get_the_phase2_inputs`, `test_a_structure_counts_once_whatever_its_level` |
  | the transformer ignores absent players and row order | `test_absent_players_change_nothing`, `test_players_can_come_in_any_order` |
  | the router's weights sum to 1; the output is the weighted sum of the experts; every expert gets a gradient | `SoftMoETest` |
  | the A/B arms differ only in the core | `test_the_two_cores_start_alike_everywhere_else` |
  | rounded inputs survive the 8-bit store exactly | `test_values_are_rounded_to_64ths_and_saturate`, `test_64ths_are_exact_in_bf16`; uint8 `"spatial"` in `test_observations_during_random_play` |
  | averaging windows gives the gradient of one big batch | `test_accumulated_windows_give_the_gradient_of_one_batch` |
  | the learning-rate schedule's shape | `test_warmup_then_cosine_to_zero` |
  | scoring twice gives the same numbers | `test_scores_are_repeatable_and_per_head` |
  | `--resume` refuses ignored flags | `ResumeArgsTest`, `test_a_resume_refuses_flags_it_would_ignore` |
  | recordings per matchup | `test_recordings_go_to_a_folder_per_matchup`, `test_games_are_served_from_the_given_folder` |

- **Speed and memory:** measured in Task 8, before any overnight run.
