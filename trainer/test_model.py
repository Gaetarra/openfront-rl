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
                            + [[ATTACK, 0, t, b] for t in range(P) for b in range(5)]
                            + [[BOAT, c, 0, b] for c in hw for b in range(5)] + [[BUILD, c, 0, 0] for c in hw])
        for i in range(N):  # sum over every joint action of p log(p / q)
            one = lambda d: {k: v[i:i + 1].expand(len(acts), *v.shape[1:]) for k, v in d.items()}
            lp = ActionDist(one(p_out), one(obs)).log_prob(acts)
            lq = ActionDist(one(q_out), one(obs)).log_prob(acts)
            self.assertAlmostEqual(float(kl[i]), float((lp.exp() * (lp - lq)).sum()), places=4)
        self.assertTrue(torch.allclose(ActionDist(p_out, obs).kl(ActionDist(p_out, obs)), torch.zeros(N), atol=1e-6))
        q_out["maps"][:, :, 7] += 300.0  # a reference sure of one cell: the others' probabilities underflow to 0
        self.assertTrue(torch.isfinite(ActionDist(p_out, obs).kl(ActionDist(q_out, obs))).all())

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


if __name__ == "__main__":
    unittest.main()
