import contextlib
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from trainer.env import BUILDINGS, DST, KINDS, N_CHANNELS, N_SCALARS, ROOT
from trainer.model import (ATTACK, ATTENDED, BOAT, BUILD, CORES, DIPLOMACY, DONATE, MAP_OF, MASK_VALUE, MOVE, MOVE_MAP,
                           OLD_CHANNELS, OLD_SCALARS, REQUEST, RETREAT, SHARES, SIZE_OF, SIZES, SPAWN, V4_CHANNELS,
                           V4_FEATURES, V4_SCALARS, ActionDist, Policy, PolicyV1, RandomPolicy, SoftMoE, describe,
                           load_net, new_net, relevance, v1_spatial)

DIMS = {"channels": N_CHANNELS, "playerFeatures": 25, "scalars": 26, "acts": 5}
DIMS_V5 = {"channels": N_CHANNELS, "playerFeatures": 31, "scalars": N_SCALARS}
N, P, H, W = 6, 5, 32, 32


def fake_obs(device="cuda", seed=0):
    """Rows as trainer/env.py builds them. Player 0 is unowned land, 1 the seat, 2 an enemy, 3 a bot, 4 absent."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cell_mask = torch.rand(N, 3, H * W, generator=g) < 0.1
    cell_mask[0] = False  # a row without any legal cell must not produce NaNs
    type_mask = torch.tensor([[1, 1] + [0] * 13] * 3 + [[1, 0, 1, 0, 1] + [0] * 10] * 3, dtype=torch.bool)
    attack_mask = torch.zeros(N, P, dtype=torch.bool)
    attack_mask[3:, 2] = True
    players = torch.rand(N, P, 25, generator=g)
    players[..., :6] = 0
    players[:, :4, 0] = 1  # exists
    players[:, 1, 2] = players[:, 2, 4] = players[:, 3, 5] = 1  # self, enemy, bot
    players[:, 4] = 0
    owners = torch.randint(0, 4, (N, 2, H * W), generator=g)
    obs = {
        "spatial": torch.randint(0, 65, (N, OLD_CHANNELS, H, W), generator=g).to(torch.uint8),
        "scalars": torch.rand(N, 16, generator=g),
        "players": players,
        "owners": owners.to(torch.int16),
        "owner_land": (torch.randint(1, 33, (N, 2, H * W), generator=g) * (owners > 0)).to(torch.uint8),
        "type_mask": type_mask, "attack_mask": attack_mask, "cell_mask": cell_mask,
    }
    # drawn after the old fields, so their values stay as they were
    obs["cell_mask"] = torch.cat([cell_mask, torch.rand(N, 9, H * W, generator=g) < 0.1], 1)  # the 9 later heat maps
    obs["cell_mask"][0] = False
    obs["retreat_mask"] = torch.zeros(N, P, dtype=torch.bool)
    obs["scalars"] = torch.cat([obs["scalars"], torch.rand(N, 10, generator=g)], 1)  # the 10 scalars of 3B part 2
    new = torch.randint(0, 65, (N, N_CHANNELS - OLD_CHANNELS, H, W), generator=g).to(torch.uint8)
    obs["spatial"] = torch.cat([obs["spatial"], new], 1)  # the channels of 3B part 2 and step 6
    # step 6's inputs and masks: 6 player features, 5 types (none legal), the warship source and destination maps, and
    # who each diplomatic type could be aimed at
    obs["players"] = torch.cat([obs["players"], torch.rand(N, P, 6, generator=g) * (obs["players"][..., :1] > 0)], 2)
    obs["type_mask"] = F.pad(obs["type_mask"], (0, 5))
    obs["cell_mask"] = torch.cat([obs["cell_mask"], torch.rand(N, 2, H * W, generator=g) < 0.1], 1)
    obs["cell_mask"][0] = False
    obs["diplomacy_mask"] = (torch.rand(N, 4, P, generator=g) < 0.5) & (obs["players"][:, None, :, 0] > 0)
    # step 7's mode input: the lobby kind (one-hot, row % len(KINDS)), then 3 values in [0, 1]
    kind = F.one_hot(torch.arange(N) % len(KINDS), len(KINDS)).float()
    obs["scalars"] = torch.cat([obs["scalars"], kind, torch.rand(N, 3, generator=g)], 1)
    return {k: v.to(device) for k, v in obs.items()}


def nets(dims=DIMS):
    """One network of every kind."""
    return {"v1": PolicyV1(dims), "dense": Policy(dims, core="dense"), "moe": Policy(dims, core="moe"),
            "v3": Policy(dims, shares=True), "swiglu": Policy(dims, core="moe-swiglu", shares=True),
            "v4": Policy(dims, shares=True, full=True), "v5": Policy(DIMS_V5, v5=True)}


def randomized(net):
    """net with random weights everywhere, scaled by fan-in: the zero-initialised heads would hide what the inputs
    do."""
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() > 1:
                fan_in = p.shape[-1] if p.dim() == 3 else p[0].numel()  # SoftMoE's experts: [experts, out, in]
                p.normal_(0, fan_in ** -0.5)
    return net


def exact_fp32(test):
    """Turns TF32 off for the rest of a test. By default this GPU's fp32 convolutions round to 10 mantissa bits, which
    turns last-bit differences between two equivalent inputs into errors far above a tight tolerance (up to 12x)."""
    flags = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    test.addCleanup(lambda: (setattr(torch.backends.cudnn, "allow_tf32", flags[0]),
                             setattr(torch.backends.cuda.matmul, "allow_tf32", flags[1])))


class FakeMove:
    """A "move" head: the destination logits and the fleet logits of each row, plus the source cell's index on one
    destination's logit and on the fleet's, so the destination depends on the source."""

    def __init__(self, dst, fleet):
        self.dst, self.fleet = dst, fleet
        self.dst_bump = torch.zeros(dst.shape[-1])
        self.dst_bump[5] = 0.3
        self.fleet_bump = torch.tensor([0.0, 0.2])

    def __call__(self, src):
        return self.dst + src[:, None] * self.dst_bump, self.fleet + src[:, None] * self.fleet_bump

    def row(self, i):
        return FakeMove(self.dst[i:i + 1], self.fleet[i:i + 1])


def joint_actions(n_types, hw, players, troops):
    """Every joint action of a network with n_types types, a hw-cell grid and `players` player rows."""
    acts = [[0, 0, 0, 0]]
    for t in range(1, n_types):
        if t == ATTACK or t == DONATE:
            acts += [[t, 0, p, s] for p in range(players) for s in troops]
        elif t == BOAT:
            acts += [[t, c, 0, s] for c in range(hw) for s in troops]
        elif t == RETREAT or t in DIPLOMACY:
            acts += [[t, 0, p, 0] for p in range(players)]
        elif t == MOVE:
            acts += [[t, c, d, f] for c in range(hw) for d in range(hw) for f in (0, 1)]
        else:
            acts += [[t, c, 0, 0] for c in range(hw)]
    return torch.tensor(acts)


def repeated(d, i, n):
    """Row i of an output or observation dict, repeated n times."""
    return {k: v.row(i) if isinstance(v, FakeMove) else v[i:i + 1].expand(n, *v.shape[1:]) for k, v in d.items()}


def brute_kl(p_out, q_out, obs, troops):
    """Per row, the sum over every joint action of p log(p / q); `troops` are the percents a troop head can send."""
    acts = joint_actions(p_out["type"].shape[-1], obs["cell_mask"].shape[-1], obs["attack_mask"].shape[-1], troops)
    rows = []
    for i in range(len(obs["type_mask"])):
        lp = ActionDist(repeated(p_out, i, len(acts)), repeated(obs, i, len(acts))).log_prob(acts)
        lq = ActionDist(repeated(q_out, i, len(acts)), repeated(obs, i, len(acts))).log_prob(acts)
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


def full_obs(seed=0):
    """small_obs with the masks of a v4 network: 15 types, 12 heat maps and the retreat pointer."""
    g = torch.Generator().manual_seed(seed + 100)
    obs = small_obs(seed)
    obs.update(type_mask=torch.ones(3, 15, dtype=torch.bool), cell_mask=torch.rand(3, 12, 16, generator=g) < 0.6,
               retreat_mask=torch.tensor([[True, False, True, False]] * 3))
    return obs


def full_out(seed=0):
    """Random outputs of a v4 network for full_obs."""
    g = torch.Generator().manual_seed(seed + 100)
    out = share_out(seed)
    out.update(type=torch.randn(3, 15, generator=g), maps=torch.randn(3, 12, 16, generator=g) * 3,
               retreat=torch.randn(3, 4, generator=g))
    return out


def v5_obs(seed=0):
    """full_obs with step 6's masks: 20 types, 14 cell masks (the 13 heat maps, then the destinations) and which of
    the 4 players each diplomatic type could be aimed at."""
    g = torch.Generator().manual_seed(seed + 200)
    obs = full_obs(seed)
    obs.update(type_mask=torch.ones(3, 20, dtype=torch.bool), cell_mask=torch.rand(3, 14, 16, generator=g) < 0.6,
               diplomacy_mask=torch.rand(3, 4, 4, generator=g) < 0.6)
    return obs


def v5_out(seed=0):
    """Random outputs of a v5 network for v5_obs."""
    g = torch.Generator().manual_seed(seed + 200)
    out = full_out(seed)
    out.update(type=torch.randn(3, 20, generator=g), maps=torch.randn(3, 13, 16, generator=g) * 3,
               shares=torch.randn(3, 3, 4, SHARES, generator=g) * 2, diplomacy=torch.randn(3, 4, 4, generator=g),
               move=FakeMove(torch.randn(3, 16, generator=g) * 2, torch.randn(3, 2, generator=g)))
    return out


def log_softmax(logits, mask):
    """The masked log-probabilities ActionDist keeps (a row without a legal entry gets entry 0)."""
    mask = mask.clone()
    mask[..., 0] |= ~mask.any(-1)
    return torch.where(mask, logits, MASK_VALUE).log_softmax(-1)


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.obs = fake_obs()

    def test_output_shapes(self):
        for name, net in nets().items():
            with self.subTest(name):
                with torch.autocast("cuda", torch.bfloat16):
                    out = net.cuda()(self.obs)
                full = name in ("v4", "v5")
                self.assertEqual(out["type"].shape, (N, {"v4": 15, "v5": 20}.get(name, 5)))
                self.assertEqual(out["maps"].shape, (N, {"v4": 12, "v5": 13}.get(name, 3), H * W))
                if name in ("v3", "swiglu", "v4", "v5"):
                    self.assertEqual(out["shares"].shape, (N, 3 if name == "v5" else 2, P, SHARES))
                else:
                    self.assertEqual(out["bins"].shape, (N, 2, 5))
                self.assertEqual("retreat" in out, full)
                self.assertEqual("move" in out, name == "v5")
                if name == "v5":
                    self.assertEqual(out["diplomacy"].shape, (N, 4, P))
                    dst, fleet = out["move"](torch.zeros(N, dtype=torch.long, device="cuda"))
                    self.assertEqual((dst.shape, fleet.shape), ((N, H * W), (N, 2)))
                self.assertEqual(out["ptr"].shape, (N, P))
                self.assertEqual(out["value"].shape, (N,))

    def test_v4_keeps_15_types_and_12_maps(self):
        """Step 6 adds types and heat maps to the env, but a v4 network keeps its sizes, so v4 checkpoints still load."""
        net = nets()["v4"].cuda()
        with torch.autocast("cuda", torch.bfloat16):
            out = net(self.obs)
        self.assertEqual((out["type"].shape, out["maps"].shape), ((N, 15), (N, 12, H * W)))
        path = ROOT / "checkpoints" / "league4a2" / "avg004000.pt"
        if not path.exists():
            self.skipTest(f"{path} is not on this machine")
        net = load_net(path, DIMS)  # strict: the network its args describe
        self.assertEqual((net.type_head.out_features, net.n_maps), (15, 12))

    def test_samples_respect_masks_and_log_probs_are_finite(self):
        for name, net in nets().items():
            with self.subTest(name):
                with torch.autocast("cuda", torch.bfloat16):
                    out = net.cuda()(self.obs)
                for _ in range(20):
                    dist = ActionDist(out, self.obs)
                    a = dist.sample()
                    t = a[:, 0]
                    self.assertTrue(self.obs["type_mask"].gather(1, t[:, None]).all())
                    for i in range(N):
                        if t[i] == SPAWN and self.obs["cell_mask"][i, 0].any():
                            self.assertTrue(self.obs["cell_mask"][i, 0, a[i, 1]])
                        if t[i] == ATTACK:
                            self.assertEqual(int(a[i, 2]), 2)
                        if t[i] == BUILD and self.obs["cell_mask"][i, 2].any():
                            self.assertTrue(self.obs["cell_mask"][i, 2, a[i, 1]])
                    lp = dist.log_prob(a)
                    self.assertTrue(torch.isfinite(lp).all())
                    self.assertTrue(torch.isfinite(dist.entropy()).all())

    def test_gradients_are_finite(self):
        for name, net in nets().items():
            with self.subTest(name):
                net = net.cuda()
                with torch.autocast("cuda", torch.bfloat16):
                    out = net(self.obs)
                dist = ActionDist(out, self.obs)
                a = dist.sample()
                (-(dist.log_prob(a).mean()) - dist.entropy().mean() + out["value"].pow(2).mean()).backward()
                for p in net.parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all())

    def test_random_policy_is_uniform_over_legal_types(self):
        dist = ActionDist(RandomPolicy(DIMS)(self.obs), self.obs)
        probs = dist.type.probs
        self.assertAlmostEqual(float(probs[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(probs[3, 4]), 1 / 3, places=5)
        noop = torch.zeros(N, 4, dtype=torch.long, device="cuda")
        self.assertAlmostEqual(float(dist.log_prob(noop)[0]), math.log(0.5), places=5)

    def test_kl_is_the_exact_kl_of_the_joint_action(self):
        g = torch.Generator(device="cpu").manual_seed(1)
        logits = lambda: {"type": torch.randn(N, 5, generator=g), "maps": torch.randn(N, 3, H * W, generator=g) * 3,
                          "ptr": torch.randn(N, P, generator=g), "bins": torch.randn(N, 2, 5, generator=g)}
        obs = {k: v.cpu() for k, v in self.obs.items()}
        p_out, q_out = logits(), logits()
        kl = ActionDist(p_out, obs).kl(ActionDist(q_out, obs))
        for i, want in enumerate(brute_kl(p_out, q_out, obs, SIZES)):
            self.assertAlmostEqual(float(kl[i]), want, places=4)
        self.assertTrue(torch.allclose(ActionDist(p_out, obs).kl(ActionDist(p_out, obs)), torch.zeros(N), atol=1e-6))
        q_out["maps"][:, :, 7] += 300.0  # a reference sure of one cell: the others' probabilities underflow to 0
        self.assertTrue(torch.isfinite(ActionDist(p_out, obs).kl(ActionDist(q_out, obs))).all())

    def test_a_new_share_head_starts_uniform(self):
        with torch.no_grad():
            out = Policy(DIMS, shares=True).cuda()(self.obs)
        self.assertEqual(int(torch.count_nonzero(out["shares"])), 0)

    def test_five_size_networks_send_their_sizes_as_percents(self):
        self.assertEqual(SIZE_OF[list(SIZES)].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(SIZE_OF[[1, 15, 16, 35, 36, 61, 62, 86, 87, 100]].tolist(), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        obs = {k: v for k, v in self.obs.items() if k != "diplomacy_mask"}  # an observation without step 6's masks
        out = RandomPolicy(DIMS)(obs)
        out["type"][:, ATTACK] = 30.0  # rows 3-5 attack (their only target is player 2); rows 0-2 cannot
        out["bins"][:, 0, 1] = 30.0  # and send 25%
        dist = ActionDist(out, obs)
        self.assertEqual(dist.sample()[3:, [0, 2, 3]].tolist(), [[ATTACK, 2, 25]] * 3)
        lp = dist.log_prob(torch.tensor([[ATTACK, 0, 2, 20]] * N, device="cuda"))  # 20% lies in 25%'s range
        self.assertTrue(torch.allclose(lp[3:], torch.zeros(3, device="cuda"), atol=1e-4))

    def test_entropy_bonus_does_not_favour_types_with_more_arguments(self):
        out = RandomPolicy(DIMS)(self.obs)
        out["type"].requires_grad_()
        ActionDist(out, self.obs).entropy().sum().backward()
        # Uniform over the legal types maximises the type entropy: only the argument terms could push the type head.
        self.assertLess(float(out["type"].grad.abs().max()), 1e-6)


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



class PolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.obs = fake_obs()

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

    def test_the_cores_start_alike_everywhere_else(self):
        states = {}
        for core in CORES:
            torch.manual_seed(3)
            states[core] = Policy(DIMS, core=core, shares=True).state_dict()
        for core in CORES[1:]:
            for k, v in states["dense"].items():
                if not k.startswith("core."):
                    self.assertTrue(torch.equal(v, states[core][k]), (core, k))

    def test_64ths_are_exact_in_bf16(self):
        k = torch.arange(256)
        self.assertTrue(torch.equal(k.to(torch.uint8).to(torch.bfloat16).mul(1 / 64).float(), k.float() / 64))


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
        bots = torch.zeros(N, extra, obs["players"].shape[-1], device="cuda")
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

    def test_every_expert_learns(self):
        torch.manual_seed(0)
        moe = SoftMoE(16, experts=4)
        y, _ = moe(torch.randn(5, 16))
        y.pow(2).sum().backward()
        self.assertTrue(all(float(moe.w.grad[e].abs().sum()) > 0 for e in range(4)))
        self.assertGreater(float(moe.router.weight.grad.abs().sum()), 0)



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

    def test_the_ppo_loss_keeps_three_heat_map_sized_tensors_for_the_backward_pass(self):
        """The taken action's log-probability and the exact KL, as PPO learns from them. On 12 heat maps of 49,152
        cells every map-sized float tensor kept for the backward pass weighs 0.3 GB in a 128-row minibatch, and
        torch's Categorical kept four (3B step 3, part 2)."""
        obs, out, ref = full_obs(), full_out(1), full_out(2)
        with torch.no_grad():
            act = ActionDist(out, obs).sample()
        out["maps"].requires_grad_()
        n, kept = out["maps"].numel(), set()

        def pack(t):
            if t.is_floating_point() and t.numel() == n:
                kept.add(t.untyped_storage().data_ptr())
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            d = ActionDist(out, obs)
            (d.log_prob(act) + d.kl(ActionDist(ref, obs))).sum().backward()
        self.assertLessEqual(len(kept), 3)

    def test_the_heat_maps_score_and_learn_as_torchs_categoricals_did(self):
        obs, out, ref = full_obs(), full_out(1), full_out(2)
        with torch.no_grad():
            act = ActionDist(out, obs).sample()
        out["maps"].requires_grad_()

        def as_torch(d, maps):
            mask = obs["cell_mask"].clone()
            mask[..., 0] |= ~mask.any(-1)
            d.cells = Categorical(logits=torch.where(mask, maps.float(), MASK_VALUE))
            return d

        results = []
        for build in (lambda o: ActionDist(o, obs), lambda o: as_torch(ActionDist(o, obs), o["maps"])):
            d = build(out)
            terms = torch.stack([d.log_prob(act), d.kl(build(ref)), d.entropy()])
            grad, = torch.autograd.grad(terms.sum(), out["maps"])
            results.append((terms.detach(), grad))
        (mine, my_grad), (torchs, torchs_grad) = results
        self.assertTrue(torch.allclose(mine, torchs, atol=1e-5), (mine, torchs))
        self.assertTrue(torch.allclose(my_grad, torchs_grad, atol=1e-6), (my_grad - torchs_grad).abs().max())

    def test_kl_covers_every_action_of_a_v4_network(self):
        obs, p_out, q_out = full_obs(), full_out(1), full_out(2)
        kl = ActionDist(p_out, obs).kl(ActionDist(q_out, obs))
        for i, want in enumerate(brute_kl(p_out, q_out, obs, range(1, SHARES + 1))):
            self.assertAlmostEqual(float(kl[i]), want, delta=1e-4 * max(1.0, want))

    def test_each_type_takes_its_own_heat_map_and_retreat_its_pointer(self):
        obs, out = full_obs(), full_out()
        dist = ActionDist(out, obs)
        for t in range(15):
            a = dist.complete(torch.full((3,), t))
            m = int(MAP_OF[t])
            for i in range(3):
                if m >= 0:
                    self.assertTrue(obs["cell_mask"][i, m, a[i, 1]], (t, i))
                if t == RETREAT:
                    self.assertTrue(obs["retreat_mask"][i, a[i, 2]])
            self.assertTrue(torch.isfinite(dist.log_prob(a)).all(), t)
        self.assertTrue(torch.isfinite(dist.entropy()).all())

    def test_old_networks_see_only_their_own_actions(self):
        obs, old = full_obs(), share_out()  # a 5-type, 3-map network in a game with 15 types
        dist = ActionDist(old, obs)
        self.assertEqual((dist.type.logits.shape[-1], dist.cells.logits.shape[1], dist.retreat), (5, 3, None))
        for _ in range(20):
            self.assertTrue((dist.sample()[:, 0] < 5).all())

    def test_the_kl_needs_the_same_troop_head(self):
        obs, out = small_obs(), share_out()
        sizes = {k: v for k, v in out.items() if k != "shares"} | {"bins": torch.zeros(3, 2, 5)}
        with self.assertRaisesRegex(ValueError, "same troop head"):
            ActionDist(out, obs).kl(ActionDist(sizes, obs))


class V5Test(unittest.TestCase):
    """Step 6: diplomacy pointers, donation shares and warship moves (a destination and a fleet choice that read the
    source)."""

    def setUp(self):
        torch.manual_seed(0)

    def test_v5_joint_probabilities_sum_to_one(self):
        obs, out = v5_obs(), v5_out()
        acts = joint_actions(20, 16, 4, range(1, SHARES + 1))
        for i in range(3):
            lp = ActionDist(repeated(out, i, len(acts)), repeated(obs, i, len(acts))).log_prob(acts)
            self.assertAlmostEqual(float(lp.exp().sum()), 1.0, places=4, msg=i)

    def test_v5_samples_are_legal_and_the_destination_reads_the_source(self):
        obs, out = v5_obs(), v5_out()
        obs["cell_mask"][:, DST] = True
        legal = lambda mask, i, k, v: bool(mask[i, k, v]) or not mask[i, k].any()  # a row without any gets entry 0
        for _ in range(30):
            dist = ActionDist(out, obs)
            a = dist.sample()
            for i, (t, c, d, f) in enumerate(a.tolist()):
                self.assertTrue(obs["type_mask"][i, t])
                if t in (SPAWN, BOAT) or t >= BUILD and t < RETREAT or t == MOVE:
                    self.assertTrue(legal(obs["cell_mask"], i, int(MAP_OF[t]), c), (t, c))
                if t in DIPLOMACY:
                    self.assertTrue(legal(obs["diplomacy_mask"], i, t - REQUEST, d), (t, d))
                if t == DONATE:
                    self.assertTrue(1 <= f <= SHARES)
                if t == MOVE:
                    self.assertTrue(legal(obs["cell_mask"], i, DST, d))
                    self.assertIn(f, (0, 1))
                    self.assertEqual(c, int(dist.src[i]))
            self.assertTrue(torch.isfinite(dist.log_prob(a)).all())
        # the same move from another source: only the source's, the destination's and the fleet's terms change
        a = dist.complete(torch.full((3,), MOVE))
        other = a.clone()
        other[:, 1] = (a[:, 1] + 1) % 16
        gap = dist.log_prob(other) - dist.log_prob(a)
        for i in range(3):
            lc = log_softmax(out["maps"][i, MOVE_MAP], obs["cell_mask"][i, MOVE_MAP])
            ld = lambda src: log_softmax(out["move"].row(i)(torch.tensor([src]))[0][0], obs["cell_mask"][i, DST])[a[i, 2]]
            lf = lambda src: log_softmax(out["move"].row(i)(torch.tensor([src]))[1][0], torch.ones(2, dtype=torch.bool))[a[i, 3]]
            before, after = int(a[i, 1]), int(other[i, 1])
            dst_gap = ld(after) - ld(before)
            self.assertGreater(abs(float(dst_gap)), 1e-4, "the destination does not read the source")
            want = lc[after] - lc[before] + dst_gap + lf(after) - lf(before)
            self.assertAlmostEqual(float(gap[i]), float(want), places=4)

    def test_v5_kl_is_exact_when_the_source_is_certain(self):
        obs, p_out, q_out = v5_obs(), v5_out(1), v5_out(2)
        only = torch.tensor([3, 7, 11])
        obs["cell_mask"][:, MOVE_MAP] = False
        obs["cell_mask"][torch.arange(3), MOVE_MAP, only] = True
        kl = ActionDist(p_out, obs).kl(ActionDist(q_out, obs))
        for i, want in enumerate(brute_kl(p_out, q_out, obs, range(1, SHARES + 1))):
            self.assertAlmostEqual(float(kl[i]), want, delta=1e-4 * max(1.0, want))
        self.assertTrue(torch.allclose(kl, ActionDist(p_out, obs).kl(ActionDist(q_out, obs), only), atol=1e-6))
        self.assertTrue(torch.allclose(ActionDist(p_out, obs).kl(ActionDist(p_out, obs)), torch.zeros(3), atol=1e-6))

    def test_the_kl_and_entropy_use_the_source_they_are_given(self):
        obs, out, ref = v5_obs(), v5_out(1), v5_out(2)
        dist, other = ActionDist(out, obs), ActionDist(ref, obs)
        a, b = torch.tensor([0, 1, 2]), torch.tensor([9, 10, 15])
        self.assertFalse(torch.allclose(dist.kl(other, a), dist.kl(other, b)))
        self.assertFalse(torch.allclose(dist.entropy(a), dist.entropy(b)))
        # logits computed at a known source stand in for the move head
        given = {k: v for k, v in ref.items() if k != "move"} | dict(zip(("dst_logits", "fleet_logits"), ref["move"](a)))
        self.assertTrue(torch.allclose(dist.kl(ActionDist(given, obs), a), dist.kl(other, a), atol=1e-6))
        self.assertTrue(torch.isfinite(dist.entropy()).all())

    def test_the_move_head_is_evaluated_once_for_the_stored_source(self):
        obs, out = v5_obs(), v5_out()
        calls = []
        move = out["move"]
        out["move"] = lambda src: calls.append(src) or move(src)
        dist = ActionDist(out, obs)
        a = dist.sample()
        calls.clear()
        dist = ActionDist(out, obs)
        src = a[:, 1]
        lp = dist.log_prob(a, src=src)
        dist.entropy(src)
        dist.kl(ActionDist(v5_out(1), obs), src)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.allclose(lp, ActionDist(out, obs).log_prob(a)))

    def test_a_donation_reads_its_recipients_share(self):
        obs, out = v5_obs(), v5_out()
        obs["diplomacy_mask"][:, DONATE - REQUEST] = True
        out["shares"].zero_()
        for who in range(4):
            out["shares"][:, 2, who, 10 * who + 4] = 50.0  # a donation to player `who` sends 10 * who + 5 %
            out["shares"][:, 0, who, 99] = 50.0  # an attack on anyone sends 100 %: the donation must not read this
        dist = ActionDist(out, obs)
        a = dist.complete(torch.full((3,), DONATE))
        self.assertEqual(a[:, 3].tolist(), (10 * a[:, 2] + 5).tolist())
        self.assertTrue(torch.allclose(dist.log_prob(a), dist.type.log_prob(a[:, 0]) + dist.diplo.log_prob(
            a[:, 2, None].expand(-1, 4))[:, DONATE - REQUEST], atol=1e-4))

    def test_the_moves_entropy_term_is_the_destinations_and_the_fleets(self):
        obs, out = v5_obs(), v5_out()
        src = torch.tensor([1, 6, 14])
        with_move = ActionDist(out, obs).entropy(src)
        no_move = ActionDist({k: v for k, v in out.items() if k != "move"}, obs).entropy()
        dst, fleet = out["move"](src)
        h = (Categorical(logits=log_softmax(dst, obs["cell_mask"][:, DST])).entropy() / math.log(16)
             + Categorical(logits=fleet).entropy() / math.log(2))
        pt = ActionDist(out, obs).type.probs
        self.assertTrue(torch.allclose(with_move - no_move, pt[:, MOVE] * h, atol=1e-5))

    def test_a_pointer_reaches_players_outside_the_attended_64(self):
        exact_fp32(self)
        obs = fake_obs()
        extra = 80 - P  # bots beyond the 64 slots, each smaller than the last
        bots = torch.zeros(N, extra, 31, device="cuda")
        bots[..., 0] = bots[..., 5] = 1
        bots[..., 7] = torch.linspace(0.05, 0.001, extra, device="cuda")
        obs["players"] = torch.cat([obs["players"], bots], 1)
        obs["attack_mask"] = torch.cat([obs["attack_mask"], torch.zeros(N, extra, dtype=torch.bool, device="cuda")], 1)
        obs["retreat_mask"] = torch.cat([obs["retreat_mask"], torch.zeros(N, extra, dtype=torch.bool, device="cuda")], 1)
        obs["diplomacy_mask"] = torch.zeros(N, 4, 80, dtype=torch.bool, device="cuda")
        obs["diplomacy_mask"][:, DONATE - REQUEST, 79] = True  # the least relevant of all is the only legal recipient
        obs["type_mask"] = torch.zeros_like(obs["type_mask"])
        obs["type_mask"][:, DONATE] = True
        net = randomized(Policy(DIMS_V5, v5=True)).cuda().eval()
        with torch.no_grad():
            out = net(obs)
            self.assertEqual(tuple(out["diplomacy"].shape), (N, 4, 80))
            for _ in range(10):
                a = ActionDist(out, obs).sample()
                self.assertEqual(a[:, [0, 1, 2]].tolist(), [[DONATE, 0, 79]] * N)
                self.assertTrue(((a[:, 3] >= 1) & (a[:, 3] <= SHARES)).all())

    def test_the_move_head_is_finite_with_and_without_autocast_and_without_warships(self):
        net = randomized(Policy(DIMS_V5, v5=True)).cuda().eval()
        obs = fake_obs()  # row 0 has no legal source cell
        no_ships = dict(obs, spatial=torch.zeros_like(obs["spatial"]))
        for cast in (True, False):
            for o in (obs, no_ships):
                with self.subTest(autocast=cast, ships=o is obs), torch.no_grad():
                    with torch.autocast("cuda", torch.bfloat16) if cast else contextlib.nullcontext():
                        out = net(o)
                    dist = ActionDist(out, o)
                    dist.complete(torch.full((N,), MOVE, device="cuda"))  # every row's source, row 0's included
                    self.assertFalse(o["cell_mask"][0, MOVE_MAP].any())
                    for src in (dist.src, torch.zeros_like(dist.src), torch.full_like(dist.src, H * W - 1)):
                        dst, fleet = out["move"](src)
                        self.assertEqual((tuple(dst.shape), tuple(fleet.shape)), ((N, H * W), (N, 2)))
                        self.assertTrue(torch.isfinite(dst.float()).all() and torch.isfinite(fleet.float()).all())
                    self.assertTrue(torch.isfinite(dist.entropy()).all())

    def test_the_move_head_gives_gradients_to_its_layers_and_the_shared_features(self):
        net = randomized(Policy(DIMS_V5, v5=True)).cuda()
        obs = fake_obs()
        with torch.autocast("cuda", torch.bfloat16):
            out = net(obs)
        dist = ActionDist(out, obs)
        a = dist.complete(torch.full((N,), MOVE, device="cuda"))
        (dist.log_prob(a) + dist.entropy(dist.src)).sum().backward()
        for layer in (net.dst_film, net.dst_shuffle, net.dst_skip, net.fleet_head, net.s1[0].c1, net.core):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                                for p in layer.parameters()), layer)

    def test_v5_starts_uniform_over_legal_actions(self):
        net = Policy(DIMS_V5, v5=True).cuda()
        obs = fake_obs()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = net(obs)
        dist = ActionDist(out, obs)
        src = torch.arange(N, device="cuda")
        dst, fleet = out["move"](src)
        for name, x in (("type", out["type"]), ("maps", out["maps"]), ("diplomacy", out["diplomacy"]),
                        ("shares", out["shares"]), ("ptr", out["ptr"]), ("retreat", out["retreat"]), ("dst", dst),
                        ("fleet", fleet)):
            self.assertEqual(int(torch.count_nonzero(x)), 0, name)
        legal = obs["diplomacy_mask"].float()
        self.assertTrue(torch.allclose(dist.diplo.probs * (legal.sum(-1, keepdim=True) > 0), legal / legal.sum(-1, keepdim=True).clamp(min=1), atol=1e-6))
        dist.complete(torch.full((N,), MOVE, device="cuda"))
        self.assertTrue(torch.allclose(dist._move_at(dist.src)[1].probs, torch.full((N, 2), 0.5, device="cuda")))
        want = obs["cell_mask"][:, DST].float()
        self.assertTrue(torch.allclose(dist._move_at(dist.src)[0].probs[1:], (want / want.sum(-1, keepdim=True))[1:], atol=1e-6))

    def test_a_v4_network_acts_the_same_on_a_v5_observation(self):
        exact_fp32(self)
        obs = fake_obs()
        v4 = dict(obs, spatial=obs["spatial"][:, :V4_CHANNELS].contiguous(), players=obs["players"][..., :V4_FEATURES].contiguous(),
                  type_mask=obs["type_mask"][:, :15].contiguous(), cell_mask=obs["cell_mask"][:, :12].contiguous())
        del v4["diplomacy_mask"]
        net = randomized(Policy(DIMS, shares=True, full=True)).cuda().eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            a, b = net(obs), net(v4)
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        torch.manual_seed(7)
        acts_a = ActionDist(a, obs).sample()
        torch.manual_seed(7)
        acts_b = ActionDist(b, v4).sample()
        self.assertTrue(torch.equal(acts_a, acts_b))
        self.assertTrue((acts_a[:, 0] < 15).all())

    def test_random_policy_covers_a_v5_observation(self):
        obs = fake_obs()
        out = RandomPolicy(DIMS_V5)(obs)
        self.assertEqual((tuple(out["type"].shape), tuple(out["maps"].shape), tuple(out["shares"].shape)),
                         ((N, 20), (N, 13, H * W), (N, 3, P, SHARES)))
        dist = ActionDist(out, dict(obs, type_mask=torch.ones_like(obs["type_mask"])))
        a = dist.complete(torch.full((N,), MOVE, device="cuda"))
        self.assertTrue(torch.isfinite(dist.log_prob(a)).all())
        self.assertTrue(torch.isfinite(dist.entropy()).all())

    def test_describe_says_20_actions(self):
        self.assertIn("20 actions", describe({"arch": "v5", "core": "moe", "widths": "64,128,192"}))


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

    def test_older_networks_read_only_their_own_inputs(self):
        """v5 appends channels, player features and scalars: every older network loads into a model built for the wider
        inputs and gives the same outputs on them."""
        wide = {**DIMS, "channels": DIMS["channels"] + 7, "playerFeatures": DIMS["playerFeatures"] + 6,
                "scalars": DIMS["scalars"] + 4}
        obs = fake_obs()
        wider = dict(obs, spatial=torch.cat([obs["spatial"], torch.full_like(obs["spatial"][:, :7], 40)], 1),
                     players=torch.cat([obs["players"], torch.rand(N, P, 6, device="cuda")], 2),
                     scalars=torch.cat([obs["scalars"], torch.rand(N, 4, device="cuda")], 1))
        for (name, net), twin in zip(nets().items(), nets(wide).values()):
            if name == "v5":  # it reads every input the dims give: not an older network
                continue
            net = randomized(net).cuda().eval()
            twin.load_state_dict(net.state_dict())
            twin = twin.cuda().eval()
            with torch.no_grad():
                a, b = net(obs), twin(wider)
            for k, v in a.items():
                self.assertTrue(torch.allclose(v, b[k], atol=1e-5), (name, k))

    def test_v4_networks_read_every_input(self):
        net = randomized(Policy(self.wide, shares=True, full=True)).cuda().eval()
        self.assertEqual((net.stem.in_channels, net.scalars[0].in_features),
                         (V4_CHANNELS, V4_SCALARS))
        with torch.no_grad():
            a, b = net(self.more), net(dict(self.more, scalars=self.more["scalars"] * 0))
        self.assertFalse(torch.allclose(a["value"], b["value"]))


class LoadTest(unittest.TestCase):
    def test_networks_saved_before_part_2_load_with_todays_wider_inputs(self):
        old = {**DIMS, "channels": OLD_CHANNELS, "scalars": OLD_SCALARS}
        wide = {**DIMS, "channels": OLD_CHANNELS + 8, "scalars": OLD_SCALARS + 10}
        with tempfile.TemporaryDirectory() as d:
            for args in ({"widths": "64,128,192"}, {"arch": "v3", "core": "moe-swiglu", "widths": "64,128,192"}):
                with self.subTest(args.get("arch", "v1")):
                    path = Path(d) / "net.pt"
                    torch.save({"model": new_net(old, args).state_dict(), "args": args}, path)
                    load_net(path, wide)  # a shape mismatch raises

    def test_checkpoints_load_as_the_network_they_hold(self):
        obs = fake_obs()
        with tempfile.TemporaryDirectory() as d:
            for args, kind in (({"widths": "64,128,192"}, PolicyV1),  # Phase 2 checkpoints have no "arch"
                               ({"arch": "v2", "core": "dense", "widths": "64,128,192"}, Policy),
                               ({"arch": "v2", "core": "moe", "widths": "64,128,192"}, Policy),
                               ({"arch": "v3", "core": "moe", "widths": "64,128,192"}, Policy),
                               ({"arch": "v3", "core": "moe-swiglu", "widths": "64,128,192"}, Policy),
                               ({"arch": "v4", "core": "moe", "widths": "64,128,192"}, Policy),
                               ({"arch": "v5", "core": "moe", "widths": "64,128,192"}, Policy)):
                dims = DIMS_V5 if args.get("arch") == "v5" else DIMS
                path = Path(d) / "net.pt"
                torch.save({"model": new_net(dims, args).state_dict(), "args": args}, path)
                net = load_net(path, dims)
                self.assertIsInstance(net, kind)
                self.assertEqual(getattr(net, "moe", False), args.get("core", "dense") != "dense")
                self.assertEqual(getattr(net, "shares", False), args.get("arch") in ("v3", "v4", "v5"))
                self.assertEqual(getattr(net, "full", False), args.get("arch") in ("v4", "v5"))
                self.assertEqual(getattr(net, "v5", False), args.get("arch") == "v5")
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    self.assertEqual(tuple(net(obs)["value"].shape), (N,))


if __name__ == "__main__":
    unittest.main()
