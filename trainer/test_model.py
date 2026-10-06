import math
import tempfile
import unittest
from pathlib import Path

import torch

from trainer.env import BUILDINGS, N_CHANNELS
from trainer.model import (ATTACK, ATTENDED, BOAT, BUILD, SIZE_OF, SIZES, SPAWN, ActionDist, Policy, PolicyV1,
                           RandomPolicy, SoftMoE, load_net, new_net, relevance, v1_spatial)

DIMS = {"channels": N_CHANNELS, "playerFeatures": 25, "scalars": 16, "acts": 5}
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
    return {"v1": PolicyV1(DIMS), "dense": Policy(DIMS, core="dense"), "moe": Policy(DIMS, core="moe")}


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


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.obs = fake_obs()

    def test_output_shapes(self):
        for name, net in nets().items():
            with self.subTest(name):
                with torch.autocast("cuda", torch.bfloat16):
                    out = net.cuda()(self.obs)
                self.assertEqual(out["type"].shape, (N, 5))
                self.assertEqual(out["maps"].shape, (N, 3, H * W))
                self.assertEqual(out["bins"].shape, (N, 2, 5))
                self.assertEqual(out["ptr"].shape, (N, P))
                self.assertEqual(out["value"].shape, (N,))

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
        hw = range(H * W)
        acts = torch.tensor([[0, 0, 0, 0]] + [[SPAWN, c, 0, 0] for c in hw]
                            + [[ATTACK, 0, t, b] for t in range(P) for b in SIZES]
                            + [[BOAT, c, 0, b] for c in hw for b in SIZES] + [[BUILD, c, 0, 0] for c in hw])
        for i in range(N):  # sum over every joint action of p log(p / q)
            one = lambda d: {k: v[i:i + 1].expand(len(acts), *v.shape[1:]) for k, v in d.items()}
            lp = ActionDist(one(p_out), one(obs)).log_prob(acts)
            lq = ActionDist(one(q_out), one(obs)).log_prob(acts)
            self.assertAlmostEqual(float(kl[i]), float((lp.exp() * (lp - lq)).sum()), places=4)
        self.assertTrue(torch.allclose(ActionDist(p_out, obs).kl(ActionDist(p_out, obs)), torch.zeros(N), atol=1e-6))
        q_out["maps"][:, :, 7] += 300.0  # a reference sure of one cell: the others' probabilities underflow to 0
        self.assertTrue(torch.isfinite(ActionDist(p_out, obs).kl(ActionDist(q_out, obs))).all())

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
        exact_fp32(self)
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


if __name__ == "__main__":
    unittest.main()
