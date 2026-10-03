"""Policy networks (PLAN "Architecture"): Policy, the Phase 3 network (ResNet map encoder, U-Net-lite heat maps, a
transformer over the players, a dense or Soft-MoE core), and PolicyV1, the Phase 2 network, kept for its checkpoints."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from trainer.env import BUILDING_TYPES, BUILDINGS, OWNER

ACT_NAMES = ["noop", "spawn", "attack", "boat", "build_city"]
SPAWN, ATTACK, BOAT, BUILD = 1, 2, 3, 4
N_MAPS = 3  # heat maps: spawn, boat landing, city
MASK_VALUE = -1e8  # finite, so masked entries never make NaN gradients
V1_CHANNELS, V1_FEATURES = 22, 17  # the Phase 2 network's inputs
D = 128  # width of the player tokens


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


class Res(nn.Module):
    """IMPALA-style residual block: two 3x3 convs, no norm, zero-init residual branch."""

    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


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


class PolicyV1(nn.Module):
    """The Phase 2 network, kept for its checkpoints (those without "arch"). It reads its 22 input channels and 17
    player features, rebuilt from today's observations."""

    def __init__(self, dims, widths=(64, 128, 192)):
        super().__init__()
        n_scal = dims["scalars"]
        self.n_bins = len(dims["troopBins"])
        c1, c2, c3 = widths
        self.stem = nn.Conv2d(V1_CHANNELS, c1, 4, stride=4)
        self.s1 = nn.Sequential(Res(c1), Res(c1))
        self.d2, self.s2 = nn.Conv2d(c1, c2, 2, stride=2), nn.Sequential(Res(c2), Res(c2))
        self.d3, self.s3 = nn.Conv2d(c2, c3, 2, stride=2), nn.Sequential(Res(c3), Res(c3))
        self.players = nn.Sequential(nn.Linear(V1_FEATURES, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.scalars = nn.Sequential(nn.Linear(n_scal, 128), nn.ReLU())
        self.core = nn.Sequential(nn.Linear(2 * c3 + 256, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.film = nn.Linear(512, 2 * c1)
        self.u3, self.u2 = nn.Conv2d(c3, c2, 1), nn.Conv2d(c2, c1, 1)
        self.shuffle = nn.Conv2d(c1, N_MAPS * 16, 1)
        self.skip = nn.Conv2d(V1_CHANNELS, N_MAPS, 1)
        self.type_head = nn.Linear(512, dims["acts"])
        self.bin_head = nn.Linear(512, 2 * self.n_bins)
        self.query, self.key = nn.Linear(512, 128), nn.Linear(128, 128)
        self.value_head = nn.Linear(512, 1)
        # Policy outputs start at zero: a uniform policy over legal actions.
        for layer in (self.type_head, self.bin_head, self.shuffle, self.skip, self.query):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, obs):
        x = _spatial_input(v1_spatial(obs["spatial"]), self.stem.weight)
        players = obs["players"][..., :V1_FEATURES]
        f1 = self.s1(self.stem(x))
        f2 = self.s2(self.d2(F.relu(f1)))
        f3 = self.s3(self.d3(F.relu(f2)))
        pooled = torch.cat([f3.mean((2, 3)), f3.amax((2, 3))], 1)
        emb = self.players(players)
        exists = players[..., :1]
        p_pool = (emb * exists).sum(1) / exists.sum(1).clamp(min=1)
        h = self.core(torch.cat([pooled, p_pool, self.scalars(obs["scalars"])], 1))
        g = f2 + F.interpolate(self.u3(F.relu(f3)), scale_factor=2)
        g = f1 + F.interpolate(self.u2(F.relu(g)), scale_factor=2)
        gamma, beta = self.film(h).chunk(2, 1)
        g = g * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        maps = F.pixel_shuffle(self.shuffle(F.relu(g)), 4) + self.skip(x)
        ptr = torch.einsum("nd,npd->np", self.query(h), self.key(emb)) / math.sqrt(128)
        return {"type": self.type_head(h), "maps": maps.flatten(2), "bins": self.bin_head(h).view(-1, 2, self.n_bins),
                "ptr": ptr, "value": self.value_head(h).squeeze(1)}


class RandomPolicy(nn.Module):
    """Uniform over legal actions: zero logits everywhere, the masks do the rest (the Phase 1 baseline)."""

    def __init__(self, dims):
        super().__init__()
        self.n_bins = len(dims["troopBins"])

    def forward(self, obs):
        n, dev = obs["type_mask"].shape[0], obs["type_mask"].device
        return {"type": torch.zeros(obs["type_mask"].shape, device=dev),
                "maps": torch.zeros(obs["cell_mask"].shape, device=dev),
                "bins": torch.zeros(n, 2, self.n_bins, device=dev),
                "ptr": torch.zeros(obs["attack_mask"].shape, device=dev),
                "value": torch.zeros(n, device=dev)}


def _masked(logits, mask):
    mask = mask.clone()
    mask[..., 0] |= ~mask.any(-1)  # rows without a legal entry get a dummy one; their head is never used
    return Categorical(logits=torch.where(mask, logits, MASK_VALUE))


class ActionDist:
    """The joint action [type, cell, target, bin]: the type picks which argument heads are used.

    spawn: spawn map cell; attack: player pointer + attack troops; boat: boat map cell + boat troops;
    build city: city map cell."""

    def __init__(self, out, obs):
        self.type = _masked(out["type"].float(), obs["type_mask"])
        self.cells = _masked(out["maps"].float(), obs["cell_mask"])  # batch [N, 3]
        self.ptr = _masked(out["ptr"].float(), obs["attack_mask"])
        self.bins = Categorical(logits=out["bins"].float())  # batch [N, 2]: attack, boat

    def sample(self):
        t, c, p, b = self.type.sample(), self.cells.sample(), self.ptr.sample(), self.bins.sample()
        zero = torch.zeros_like(t)
        cell = torch.where(t == SPAWN, c[:, 0], torch.where(t == BOAT, c[:, 1], torch.where(t == BUILD, c[:, 2], zero)))
        target = torch.where(t == ATTACK, p, zero)
        troops = torch.where(t == ATTACK, b[:, 0], torch.where(t == BOAT, b[:, 1], zero))
        return torch.stack([t, cell, target, troops], 1)

    def log_prob(self, actions):
        t, cell, target, troops = actions.unbind(1)
        lc = self.cells.log_prob(cell[:, None].expand(-1, N_MAPS))
        lb = self.bins.log_prob(troops[:, None].expand(-1, 2))
        zero = torch.zeros_like(lc[:, 0])
        return (self.type.log_prob(t)
                + torch.where(t == SPAWN, lc[:, 0], zero)
                + torch.where(t == ATTACK, self.ptr.log_prob(target) + lb[:, 0], zero)
                + torch.where(t == BOAT, lc[:, 1] + lb[:, 1], zero)
                + torch.where(t == BUILD, lc[:, 2], zero))

    def entropy(self):
        """Exact joint entropy, H(type) + sum_k p(k) H(args | k), with every head divided by the log of its size so
        the 4k-cell maps do not drown out the type head. p(k) is detached: otherwise the bonus pushes the type head
        toward the types with the most arguments (attack, boat) and away from no-op, as it did in Phase 1."""
        hc = self.cells.entropy() / math.log(self.cells.logits.shape[-1])
        hb = self.bins.entropy() / math.log(self.bins.logits.shape[-1])
        hp = self.ptr.entropy() / math.log(max(self.ptr.logits.shape[-1], 2))
        pt = self.type.probs.detach()
        return (self.type.entropy() / math.log(pt.shape[-1]) + pt[:, SPAWN] * hc[:, 0]
                + pt[:, ATTACK] * (hp + hb[:, 0]) + pt[:, BOAT] * (hc[:, 1] + hb[:, 1]) + pt[:, BUILD] * hc[:, 2])

    def kl(self, ref):
        """Exact KL(self || ref) of the joint action, KL(type) + sum_k p(k) KL(args | k). The one-sample estimate
        from the taken action is not usable: it explodes when that action is one this policy all but rules out and
        ref does not, which a spread-out 49k-cell map makes common."""
        kc, kb = _kl(self.cells, ref.cells), _kl(self.bins, ref.bins)  # [N, 3], [N, 2]
        pt = self.type.probs
        return (_kl(self.type, ref.type) + pt[:, SPAWN] * kc[:, 0] + pt[:, ATTACK] * (_kl(self.ptr, ref.ptr) + kb[:, 0])
                + pt[:, BOAT] * (kc[:, 1] + kb[:, 1]) + pt[:, BUILD] * kc[:, 2])


def _kl(p, q):
    """KL(p || q) of categoricals over the same legal entries. torch's kl_divergence returns inf where q's probability
    underflows to 0 (a sure policy on a 49k-cell map); the log-probabilities stay finite, and masked entries add 0."""
    return (p.probs * (p.logits - q.logits)).sum(-1)


def load_net(path, dims, device="cuda"):
    """A frozen policy from a trainer.bc or trainer.ppo checkpoint, in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyV1(dims, tuple(int(w) for w in ckpt["args"]["widths"].split(",")))
    net.load_state_dict(ckpt["model"])
    return net.to(device).to(memory_format=torch.channels_last).eval().requires_grad_(False)
