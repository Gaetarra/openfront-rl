# openfront-rl

A deep reinforcement learning bot for [OpenFront.io](https://openfront.io), the territorial real-time strategy game.
Work in progress.

The goal is a bot that plays every mode (Ranked 1v1/2v2, FFA, Team) well: first learning from human replays,
then improving through self-play. The full design, phases and decisions are in [docs/PLAN.md](docs/PLAN.md).

**Status:** Phase 3A done. The Phase 2 clone of ranked 1v1 players beat random play 98% of the time and a scripted
expander 95%, and PPO from it beat it 77% (see [docs/PLAN.md](docs/PLAN.md), "Phase 2 results"). The Phase 3A clone,
with a richer view of the map and the other players (a player transformer and a mixture-of-experts core) and a fixed
imitation training on 20k replays, beats the Phase 2 clone 63% of the time from imitation alone ("3A done"). Next:
every human action (3B), starting with troop shares chosen per target, then ranked 2v2 (3C).
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
