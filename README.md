# openfront-rl

A deep reinforcement learning bot for [OpenFront.io](https://openfront.io), the territorial real-time strategy game.
Work in progress.

The goal is a bot that plays every mode (Ranked 1v1/2v2, FFA, Team) well: first learning from human replays,
then improving through self-play. The full design, phases and decisions are in [docs/PLAN.md](docs/PLAN.md).

**Status:** Phase 2. A clone of ranked 1v1 players, trained by behaviour cloning on 16k public replays, beats random
play 98% of the time and a scripted expander 95% of the time. PPO from the clone is in progress: the first run peaked,
then collapsed, and the second runs with the fixes (see [docs/PLAN.md](docs/PLAN.md), "Phase 2 results so far").
Training: `.venv/Scripts/python -m trainer.ppo --run <name>`, then open the control page at http://127.0.0.1:8787.

## Layout

| Path | What |
|---|---|
| `OpenFront_Source_Code/openfrontio/` | The official game, as an unmodified git submodule pinned to the production release |
| `bridge/` | TypeScript: runs games headlessly and exchanges observations/actions with the trainer |
| `trainer/` | Python: model, training (behaviour cloning + PPO self-play), evaluation |
| `scripts/` | Benchmarks and maintenance tools |
| `docs/PLAN.md` | Project plan |

## Setup

```bash
git clone --recursive https://github.com/Gaetarra/openfront-rl
cd openfront-rl/OpenFront_Source_Code/openfrontio
npx -y npm@12.1.0 ci --ignore-scripts   # the game needs npm >= 12.1
cd ../..
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Credits

OpenFront is © OpenFront and Contributors, licensed under AGPL-3.0 (code) and CC BY-SA 4.0 (assets).
The game is included unmodified as a submodule.
