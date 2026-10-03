# Phase 3A — Attention over the 64 Most Relevant Players: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Phase 3 network passes Gate 3A's speed bar (PPO ≥ 75% of Phase 2's ~190 decisions/s) by limiting the
player transformer's attention to the 64 most relevant players. Every player keeps its token, so the attack pointer,
the attention summary and the relation means still see everyone.

**Architecture:**
- `relevance(players)` scores each player from its features alone: the seat and unowned land first, then the other
  non-bot players, then those the seat can attack or that are fighting it, then the biggest. Absent players come last.
- `Policy.forward` embeds every player as before. It gathers the top `ATTENDED = 64` tokens, runs the 2-layer
  transformer on them, scatters the results back, and applies the final norm to all tokens.
- With 64 players or fewer, every player is attended, which is the network of the main 3A plan.

**Tech Stack:** as the main 3A plan (torch 2.14, `unittest`).

**Spec:**
- `docs/PLAN.md` 3A, player encoder: "The ≤64-token selection comes in only if the speed check fails". It failed:
  120–128 decisions/s against a bar of ~142 ("3A progress").
- The user's decision (2026-10-03): attention over 64, every player keeping a token.

## Global Constraints

As `docs/superpowers/plans/2026-10-03-phase3a-encoder.md`:
- no edits in the game submodule, no Claude attribution, no new dependencies;
- GPU checks ≤ 15 min in session, run with the PC idle;
- `phase3` is pushed (the user's OK, 2026-10-03), and later pushes are asked again.

## Review Focus

- **More than 64 players:** a player outside the attended 64 still gets a pointer logit that depends on its own
  features. Test: `test_players_beyond_the_attended_64_keep_their_token`.
- **The selection order:** the seat, unowned land and the rival player are always attended; absent players never
  take a slot from a present one. Test: `test_relevance_puts_the_seat_unowned_land_and_rivals_first`.
- **64 players or fewer:** behaviour is unchanged. The existing absent-player and permutation tests (5 players) pass
  unchanged.

---

### Task 1: Attention over the 64 most relevant players

**Files:**
- Modify: `trainer/model.py` (new `ATTENDED`, `relevance`; `Policy.forward`'s transformer step; the docstring)
- Test: `trainer/test_model.py`

**Interfaces:**
- Produces:
  - `trainer.model.ATTENDED = 64`;
  - `trainer.model.relevance(players [N, P, F]) -> [N, P]` scores; absent players score -1;
  - `Policy` outputs are unchanged in shape.

- [ ] **Step 1: Write the failing tests** (`trainer/test_model.py`; import `ATTENDED, relevance` from `trainer.model`)

```python
class AttentionTest(unittest.TestCase):
    def test_relevance_puts_the_seat_unowned_land_and_rivals_first(self):
        f = torch.zeros(1, 7, 25)
        f[0, :6, 0] = 1  # rows 0-5 exist, row 6 is absent
        f[0, 0, 12] = 1  # unowned land, attackable
        f[0, 1, 2] = 1  # the seat
        f[0, 2, 4] = 1  # an enemy
        f[0, 3, 5] = f[0, 3, 12] = 1  # a bot the seat can attack
        f[0, 4, 5], f[0, 4, 7] = 1, 0.2  # a big bot
        f[0, 5, 5], f[0, 5, 7] = 1, 0.1  # a smaller bot
        self.assertEqual(relevance(f)[0].argsort(descending=True).tolist(), [0, 1, 2, 3, 4, 5, 6])

    def test_players_beyond_the_attended_64_keep_their_token(self):
        exact_fp32(self)
        obs = fake_obs()
        extra = 70 - P  # bots beyond the 64 slots, each smaller than the last
        bots = torch.zeros(N, extra, 25, device="cuda")
        bots[..., 0] = bots[..., 5] = 1
        bots[..., 7] = torch.linspace(0.05, 0.001, extra, device="cuda")
        obs["players"] = torch.cat([obs["players"], bots], 1)
        obs["attack_mask"] = torch.cat([obs["attack_mask"], torch.ones(N, extra, dtype=torch.bool, device="cuda")], 1)
        net = randomized(Policy(DIMS)).cuda().eval()
        attended = []
        net.encoder[0].register_forward_pre_hook(lambda module, args: attended.append(args[0].shape[1]))
        changed = dict(obs, players=obs["players"].clone())
        changed["players"][:, -1, 8] = 1.0  # the least relevant bot gets troops
        with torch.no_grad():
            a, b = net(obs), net(changed)
        self.assertEqual(attended, [ATTENDED, ATTENDED])
        self.assertEqual(tuple(a["ptr"].shape), (N, 70))
        self.assertTrue(torch.isfinite(a["ptr"]).all())
        self.assertGreater(float((a["ptr"][:, -1] - b["ptr"][:, -1]).abs().min()), 1e-4, "its own card counts")
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v`
Expected: ERROR, `ImportError: cannot import name 'ATTENDED' from 'trainer.model'`.

- [ ] **Step 3: Implement** in `trainer/model.py`

After `D = 128 ...`, add:

```python
ATTENDED = 64  # players the transformer attends to (the most relevant ones); every player keeps a token


def relevance(players):
    """How much each player matters to the seat, from its features (bridge/obs.ts), for the transformer's ATTENDED
    slots, in tiers:
    - the seat and unowned land (row 0);
    - the other non-bot players (friends, enemies);
    - the players the seat can attack, or that are attacking it or being attacked by it;
    - then by share of the land.
    Absent players come last."""
    f = players
    score = (8 * f[..., 2] + 4 * (f[..., 3] + f[..., 4]) + 2 * f[..., 12] + 2 * ((f[..., 13] > 0) | (f[..., 14] > 0))
             + f[..., 7])
    score[:, 0] += 8
    return torch.where(f[..., 0] > 0, score, -1.0)
```

In `Policy.forward`, replace

```python
        t = self.embed(torch.cat([players, land_features(obs, f2).to(players.dtype)], 2))
        for layer in self.encoder:
            t = layer(t, src_key_padding_mask=~exists)
        tokens = self.encoder_norm(t)
```

with

```python
        t = self.embed(torch.cat([players, land_features(obs, f2).to(players.dtype)], 2))
        top = relevance(players).topk(min(ATTENDED, players.shape[1]), dim=1).indices  # [N, ATTENDED]
        index = top.unsqueeze(-1).expand(-1, -1, D)
        attended = t.gather(1, index)
        for layer in self.encoder:
            attended = layer(attended, src_key_padding_mask=~exists.gather(1, top))
        tokens = self.encoder_norm(t.scatter(1, index, attended.to(t.dtype)))  # back in place; the rest unattended
```

In the `Policy` docstring, change the transformer line to
`- A 2-layer transformer runs over the ATTENDED most relevant players (relevance); absent players are masked out. The
  other players keep their tokens, so the pointer, the attention summary and the means still see everyone.`

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest trainer.test_model -v`
Expected: all pass. The 5-player absent and permutation tests are unchanged.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/Scripts/python -m unittest discover -s trainer -t .`
Expected: OK (79 tests).

```bash
git add trainer/model.py trainer/test_model.py
git commit -m "The player transformer attends to the 64 most relevant players; every player keeps its token"
```

---

### Task 2: Speed and memory check, then the docs

**Files:**
- Modify: `docs/PLAN.md` (the player encoder line, the Architecture table row, "3A progress"), `CLAUDE.md` (the
  GPU-memory note)

- [ ] **Step 1: Speed check, with the PC idle** (~10 min each)

Run `.venv/Scripts/python scripts/speed_check.py --skip-bc` (8 minibatches), watching "Shared GPU memory" as in the
main plan. If it spills, run `--ppo-minibatches 12 --skip-bc` instead.

Expected:
- the Phase 3 networks keep ≥ 75% of Phase 2's 190 (≥ 142 decisions/s) at a minibatch count that fits;
- moe stays within 5% of dense.

If not, report the numbers to the user before any overnight run.

- [ ] **Step 2: Docs**

`docs/PLAN.md`:
- Architecture table, player encoder row: "2-layer transformer d=128 attending to the 64 most relevant players; every
  player keeps a token (features + building counts + spatial features pooled over their cells)".
- 3A player encoder bullet: replace "The ≤64-token selection comes in only if the speed check fails." with "The speed
  check failed with all 403 tokens (120–128 decisions/s), so the transformer attends to the 64 most relevant players
  (user decision 2026-10-03) and every player keeps its token."
- "3A progress": the new numbers, and the minibatch count the Phase 3 PPO runs use.

`CLAUDE.md`: update the GPU-memory note with the minibatch count that fits.

- [ ] **Step 3: Commit**

```bash
git add docs/PLAN.md CLAUDE.md
git commit -m "Speed check with attention over 64: <numbers>; docs"
```

(The commit message carries the measured numbers.)
