import os

# Expandable allocator segments: memory freed by one tensor serves a larger one, where PyTorch's default kept both
# blocks reserved. v4's PPO peak went from 10.5 to 9.9 GB reserved and stopped spilling into system RAM (docs/PLAN.md
# 3B step 3). Pinned buffers above 1 MB keep their size: PyTorch rounds them up to a power of two, and PPO's rollout
# buffer took 12.9 GiB of locked RAM for 7.6 GiB of rows (PLAN 4A, "First league run"). Set on import, before any
# module touches CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,pinned_max_round_threshold_mb:1")
