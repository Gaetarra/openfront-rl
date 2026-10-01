import math
import unittest

import torch

from trainer.model import ATTACK, BOAT, BUILD, SPAWN, ActionDist, Policy, RandomPolicy

DIMS = {"channels": 22, "playerFeatures": 17, "scalars": 16, "acts": 5, "troopBins": [0.1, 0.25, 0.5, 0.75, 1.0]}
N, P, H, W = 6, 5, 32, 32


def fake_obs(device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(0)
    cell_mask = torch.rand(N, 3, H * W, generator=g) < 0.1
    cell_mask[0] = False  # a row without any legal cell must not produce NaNs
    type_mask = torch.tensor([[1, 1, 0, 0, 0]] * 3 + [[1, 0, 1, 0, 1]] * 3, dtype=torch.bool)
    attack_mask = torch.zeros(N, P, dtype=torch.bool)
    attack_mask[3:, 2] = True
    obs = {
        "spatial": torch.rand(N, 22, H, W, generator=g).to(torch.bfloat16),
        "scalars": torch.rand(N, 16, generator=g),
        "players": torch.rand(N, P, 17, generator=g),
        "type_mask": type_mask, "attack_mask": attack_mask, "cell_mask": cell_mask,
    }
    return {k: v.to(device) for k, v in obs.items()}


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.obs = fake_obs()
        self.model = Policy(DIMS).cuda()

    def test_output_shapes(self):
        with torch.autocast("cuda", torch.bfloat16):
            out = self.model(self.obs)
        self.assertEqual(out["type"].shape, (N, 5))
        self.assertEqual(out["maps"].shape, (N, 3, H * W))
        self.assertEqual(out["bins"].shape, (N, 2, 5))
        self.assertEqual(out["ptr"].shape, (N, P))
        self.assertEqual(out["value"].shape, (N,))

    def test_samples_respect_masks_and_log_probs_are_finite(self):
        with torch.autocast("cuda", torch.bfloat16):
            out = self.model(self.obs)
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
        with torch.autocast("cuda", torch.bfloat16):
            out = self.model(self.obs)
        dist = ActionDist(out, self.obs)
        a = dist.sample()
        (-(dist.log_prob(a).mean()) - dist.entropy().mean() + out["value"].pow(2).mean()).backward()
        for p in self.model.parameters():
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


if __name__ == "__main__":
    unittest.main()
