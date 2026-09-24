"""Phase 0 spike S3: GPU throughput of the map encoder + heat-map decoder (eager, bf16 autocast, channels-last).

Grid = World at 8x8-tile cells (250x125), padded to 256x128. Run: .venv/Scripts/python scripts/bench_gpu.py
IMPALA-style ResNet blocks won over ConvNeXt (depthwise 7x7 + LayerNorm are memory-bound in eager mode and
torch.compile has no Triton on Windows): see docs/PLAN.md, Phase 0 results.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CH, H, W, N_MAPS = 48, 128, 256, 8


class Res(nn.Module):  # residual block, no norm, zero-init last conv
    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class Net(nn.Module):
    def __init__(self, widths, depths=(2, 2, 2)):
        super().__init__()
        c1, c2, c3 = widths
        self.stem = nn.Conv2d(IN_CH, c1, 4, stride=4)
        self.s1 = nn.Sequential(*[Res(c1) for _ in range(depths[0])])
        self.d2, self.s2 = nn.Conv2d(c1, c2, 2, stride=2), nn.Sequential(*[Res(c2) for _ in range(depths[1])])
        self.d3, self.s3 = nn.Conv2d(c2, c3, 2, stride=2), nn.Sequential(*[Res(c3) for _ in range(depths[2])])
        # U-Net-lite decoder: back to 1/4 res, pixel shuffle to cell res, plus a per-cell skip from the input
        self.u3, self.u2 = nn.Conv2d(c3, c2, 1), nn.Conv2d(c2, c1, 1)
        self.shuffle = nn.Conv2d(c1, N_MAPS * 16, 1)
        self.skip = nn.Conv2d(IN_CH, N_MAPS, 1)
        self.value = nn.Linear(2 * c3, 1)

    def forward(self, x):
        f1 = self.s1(self.stem(x))
        f2 = self.s2(self.d2(F.relu(f1)))
        f3 = self.s3(self.d3(F.relu(f2)))
        pooled = torch.cat([f3.mean((2, 3)), f3.amax((2, 3))], 1)
        g = f2 + F.interpolate(self.u3(F.relu(f3)), scale_factor=2)
        g = f1 + F.interpolate(self.u2(F.relu(g)), scale_factor=2)
        maps = F.pixel_shuffle(self.shuffle(F.relu(g)), 4) + self.skip(x)
        return maps, self.value(pooled)


def bench(net, batch, train, steps=20):
    x = torch.randn(batch, IN_CH, H, W, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, fused=True) if train else None
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16), torch.set_grad_enabled(train):
            maps, v = net(x)
            if train:
                loss = maps.float().logsumexp((1, 2, 3)).mean() + v.float().pow(2).mean()
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.cuda.synchronize()
    return batch * steps / (time.perf_counter() - t0), torch.cuda.max_memory_allocated() / 2**30


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(), torch.__version__)
    for name, widths in (("S", (64, 128, 192)), ("M", (96, 192, 256)), ("L", (128, 256, 384))):
        net = Net(widths).cuda().to(memory_format=torch.channels_last)
        params = sum(p.numel() for p in net.parameters()) / 1e6
        for batch in (64, 128):
            inf, mem_i = bench(net.eval(), batch, train=False)
            trn, mem_t = bench(net.train(), batch, train=True)
            print(f"{name} {params:4.1f}M  batch {batch:3d}: inference {inf:6.0f}/s ({mem_i:3.1f} GB)"
                  f"  train fwd+bwd {trn:5.0f}/s ({mem_t:3.1f} GB)")
        del net
        torch.cuda.empty_cache()
