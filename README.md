# openfront-rl

A deep reinforcement learning bot for [OpenFront.io](https://openfront.io), the territorial real-time strategy game.
Work in progress.

The goal is a bot that plays every mode (Ranked 1v1/2v2, FFA, Team) well: first learning from human replays,
then improving through self-play. The full design, phases and decisions are in [docs/PLAN.md](docs/PLAN.md).

**Status:** Phase 1 (training pipeline: headless games, observations, model, PPO, pause/resume control) is built. Its
learning test stalled without behaviour cloning, as a from-scratch agent rarely beats even the easiest opponents, and a
re-plan that moves behaviour cloning earlier is under review (see [docs/PLAN.md](docs/PLAN.md)).
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
npm run inst
cd ../..
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Credits

OpenFront is © OpenFront and Contributors, licensed under AGPL-3.0 (code) and CC BY-SA 4.0 (assets).
The game is included unmodified as a submodule.
