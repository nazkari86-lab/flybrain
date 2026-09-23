# Game-learning reuse audit — 2026-09-24

## Environment

- Machine: Apple Silicon (`arm64`)
- Python: 3.12.13
- Package manager: uv 0.12.1
- Existing project: NumPy 2.x, Pydantic 2.x, Typer, pytest, Ruff, mypy

## Selected maintained components

| Component | Checked release | License | Exact reusable capability |
| --- | ---: | --- | --- |
| Gymnasium | 1.3.0 | MIT | Standard seeded environment API and spaces |
| Stable-Baselines3 | 2.9.0 | MIT | Maintained DQN implementation, prediction, model and replay persistence |
| PyTorch | 2.14.0 | BSD-style upstream license | Apple Silicon-compatible tensor and neural-network runtime |
| pygame-ce | 2.x | LGPL-2.1-or-later | Local rendering, keyboard/mouse input, and headless surfaces |
| python-chess | 1.999 | GPL-3.0-or-later | Complete chess rules, legal moves, FEN/PGN, and UCI integration |
| Stockfish | 19, Homebrew ARM64 | GPL-3.0 | Optional visible teacher and fixed-strength benchmark |

All selected packages declare Python 3.12-compatible version ranges. Stable-Baselines3 2.9.0 declares `gymnasium>=0.29.1,<2`, `torch>=2.8,<3`, and `numpy>=1.20,<3`, which is compatible with this project.

Upstream sources:

- <https://github.com/Farama-Foundation/Gymnasium>
- <https://github.com/DLR-RM/stable-baselines3>
- <https://github.com/pytorch/pytorch>
- <https://github.com/pygame-community/pygame-ce>
- <https://github.com/niklasf/python-chess>
- <https://github.com/official-stockfish/Stockfish>

`python-chess` and Stockfish are optional game-layer GPL dependencies. Their versions and any
teacher limits are recorded in chess artifacts. Stockfish output may bootstrap a model, but the
engine is neither hidden in the learned policy nor evidence that the policy independently reached
Stockfish strength.

The installed Stockfish 19 binary resolved to
`/opt/homebrew/Cellar/stockfish/19/bin/stockfish` with SHA-256
`dc2f18c34ae962dff591b66147d220ec06e61d756b93d8a4f5e04fd8e55c251f`.

## Geometry-Dash-style prior art

A bounded GitHub search was run for `geometry dash reinforcement learning`, `geometry dash gym`, and `geometry dash ai`. The closest maintained licensed candidate was [Vaibhav0PS/DashRL](https://github.com/Vaibhav0PS/DashRL), an MIT-licensed CPU DQN demonstration using a two-action Gymnasium environment, fixed-timestep Pygame physics, a `7 → 128 → 128 → 2` network, replay memory, greedy evaluation, and visual playback.

The project validates the selected mechanism but is not an end-to-end fit for FlyBrain:

- it trains and evaluates on one hand-authored level rather than disjoint procedural seed sets;
- it does not provide immutable train/validation/final-holdout manifests;
- its checkpoints do not satisfy FlyBrain's atomic full-state resume and provenance contract;
- it has no shared chess adapter or FlyBrain Typer integration;
- its README is evidence of design intent, not independent proof of unseen-level generalization.

Therefore FlyBrain reuses the maintained Gymnasium/SB3/PyTorch stack and the published high-level DQN/physics pattern. It implements only the missing procedural environment, artifact contract, evaluation isolation, CLI, and viewer. No DashRL source file is copied into this repository.

## Rejected candidates

- `GaspardCulis/GDGym`: last pushed in 2023 and no detected repository license.
- Several 2025–2026 Geometry Dash agents: no detected license, direct commercial-client automation, or no reproducible holdout and checkpoint contract.
- A single pixel-only generalist: materially higher compute and data requirements with weaker evidence for both one-button control and chess.

## Claim boundary

The selected game learner is an engineered RL system. It is separate from the MaleCNS biological pathway and cannot be cited as evidence of connectome-generated learning, biological realism, general intelligence, or direct control of commercial Geometry Dash.
