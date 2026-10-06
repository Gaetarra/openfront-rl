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
SIZES = (10, 25, 50, 75, 100)  # the troop sizes of the Phase 2 and 3A networks, in percent of current troops
SHARES = 100  # the Phase 3B network's troop shares: 1-100% (PLAN 3B)
# Each percent's nearest size in log scale, as the Phase 2-3A labels were made: 1-15% -> 10, 16-35 -> 25, 36-61 -> 50,
# 62-86 -> 75, 87-100 -> 100.
SIZE_OF = torch.tensor([min(range(len(SIZES)), key=lambda i: abs(math.log(max(p, SIZES[0]) / SIZES[i])))
                        for p in range(SHARES + 1)])
SIZE_PERCENT = torch.tensor(SIZES)
# [5, SHARES]: 0 on the percents of each size, -inf elsewhere
SIZE_RANGES = torch.where(SIZE_OF[1:] == torch.arange(len(SIZES))[:, None], 0.0, -math.inf)
N_MAPS = 3  # heat maps: spawn, boat landing, city
MASK_VALUE = -1e8  # finite, so masked entries never make NaN gradients
V1_CHANNELS, V1_FEATURES = 22, 17  # the Phase 2 network's inputs
D = 128  # width of the player tokens
ATTENDED = 64  # players the transformer attends to (the most relevant ones); every player keeps a token
DS = 64  # hidden width of the troop-share head, per kind (attack, boat)
CORES = ("dense", "moe", "moe-swiglu")  # Policy cores: one 512 x 512 layer, 8 such experts, 8 SwiGLU experts


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
    - A 2-layer transformer runs over the ATTENDED most relevant players (relevance); absent players are masked out.
      The other players keep their tokens, so the pointer, the attention summary and the means still see everyone.
    - The core reads an attention summary queried by the map and the scalars, the seat's own token, and the means over
      its friends, its enemies and the bots.
    - The attack pointer reads the tokens.
    - core: "dense" (one 512 x 512 layer), "moe" (a Soft-MoE layer of 8 experts in its place) or "moe-swiglu" (8
      SwiGLU experts, 512 -> 1024 -> 512).
    - troops: with shares=True (arch v3, PLAN 3B), per kind a hidden layer reads the core and one player's token, then
      100 shares; otherwise the 3A network's 5 sizes from the core alone."""

    def __init__(self, dims, widths=(64, 128, 192), core="moe", shares=False):
        super().__init__()
        c_in, n_feat, n_scal = dims["channels"], dims["playerFeatures"], dims["scalars"]
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
        # Built last, so for a given seed both cores start every other layer from the same weights (the A/B).
        self.moe = core != "dense"
        self.core = SoftMoE(512, hidden=1024 if core == "moe-swiglu" else None) if self.moe else nn.Linear(512, 512)

    def forward(self, obs):
        x = _spatial_input(obs["spatial"], self.stem.weight)
        f1 = self.s1(self.stem(x))
        f2 = self.s2(self.d2(F.relu(f1)))
        f3 = self.s3(self.d3(F.relu(f2)))
        pooled = torch.cat([f3.mean((2, 3)), f3.amax((2, 3))], 1)
        players = obs["players"]
        exists = players[..., 0] > 0
        t = self.embed(torch.cat([players, land_features(obs, f2).to(players.dtype)], 2))
        top = relevance(players).topk(min(ATTENDED, players.shape[1]), dim=1).indices  # [N, ATTENDED]
        index = top.unsqueeze(-1).expand(-1, -1, D)
        attended = t.gather(1, index)
        for layer in self.encoder:
            attended = layer(attended, src_key_padding_mask=~exists.gather(1, top))
        tokens = self.encoder_norm(t.scatter(1, index, attended.to(t.dtype)))  # back in place; the rest unattended
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
        out.update(type=self.type_head(h), maps=maps.flatten(2), ptr=ptr, value=self.value_head(h).squeeze(1))
        if self.shares:
            n, p = tokens.shape[:2]
            z = F.relu(self.share_h(h).view(n, 1, 2, DS) + self.share_t(tokens).view(n, p, 2, DS))  # [N, P, kind, DS]
            out["shares"] = torch.stack([head(z[:, :, k]) for k, head in enumerate(self.share_out)], 1)
        else:
            out["bins"] = self.bin_head(h).view(-1, 2, len(SIZES))
        return out


class PolicyV1(nn.Module):
    """The Phase 2 network, kept for its checkpoints (those without "arch"). It reads its 22 input channels and 17
    player features, rebuilt from today's observations."""

    def __init__(self, dims, widths=(64, 128, 192)):
        super().__init__()
        n_scal = dims["scalars"]
        self.n_bins = len(SIZES)
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
        self.n_bins = len(SIZES)

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


def _kl(p, q):
    """KL(p || q) of categoricals over the same legal entries. torch's kl_divergence returns inf where q's probability
    underflows to 0 (a sure policy on a 49k-cell map); the log-probabilities stay finite, and masked entries add 0."""
    return (p.probs * (p.logits - q.logits)).sum(-1)


def arch_of(args):
    """The network a run's settings describe: arch v1 (Phase 2), v2 (Phase 3A, 5 troop sizes) or v3 (Phase 3B, 100
    troop shares). Phase 2 checkpoints predate "arch": they are PolicyV1."""
    return {"arch": args.get("arch", "v1"), "core": args.get("core", "dense"), "widths": args["widths"]}


def new_net(dims, args):
    """A fresh network of the kind a run's settings (a dict like a checkpoint's "args") describe."""
    spec = arch_of(args)
    widths = tuple(int(w) for w in spec["widths"].split(","))
    if spec["arch"] == "v1":
        return PolicyV1(dims, widths)
    return Policy(dims, widths, spec["core"], shares=spec["arch"] == "v3")


CORE_NAMES = {"dense": "dense core", "moe": "Soft-MoE core (8 experts)",
              "moe-swiglu": "Soft-MoE core (8 SwiGLU experts)"}


def describe(args):
    """A network in words, from a run's settings (the control page)."""
    spec = arch_of(args)
    if spec["arch"] == "v1":
        return "the Phase 2 network"
    troops = "100 troop shares" if spec["arch"] == "v3" else "5 troop sizes"
    return f"player transformer, {CORE_NAMES[spec['core']]}, {troops}"


def load_net(path, dims, device="cuda"):
    """A frozen policy from a trainer.bc or trainer.ppo checkpoint, in eval mode: the network it holds."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = new_net(dims, ckpt["args"])
    net.load_state_dict(ckpt["model"])
    return net.to(device).to(memory_format=torch.channels_last).eval().requires_grad_(False)
