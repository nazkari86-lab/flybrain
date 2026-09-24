# Practical Autonomy Design

## Goal

Deliver a fast, visibly autonomous virtual fly that reaches food, avoids threats and walls, learns online, and saves a replayable result. Scientific proof of full biological intelligence is explicitly out of scope.

## Architecture

Use the existing deterministic `ArenaWorld` as the fast body/environment loop. A hybrid controller combines hard safety reflexes, food/threat potential-field navigation, and a tabular Q-learning residual over discretized egocentric sensory state. The heuristic makes the first run useful; Q-learning updates action values from progress, contacts, and wall penalties.

Each selected planar command is translated into left/right descending drive for FlyGym's official `HybridTurningController`. The controller drives 42 position-controlled leg degrees of freedom plus six adhesion channels through MuJoCo while applying its CPG, stumbling, and retraction corrections. Alternating six-leg phase records remain in the fast trace, but the default CLI result must also contain measured physical displacement and stability.

## Product Surface

`flybrain experiment practical-autonomy --output RESULT.json` trains in randomized arenas, evaluates on held-out layouts, and writes one atomic JSON artifact containing success metrics, learning state counts, a representative trajectory, commands, and six-leg gait phases.

`flybrain interactive` opens a persistent MuJoCo viewer for the same official physical controller. Keyboard presses adjust persistent forward and turn commands, stop, reset, toggle an autonomous demonstration mode, or quit. A viewer overlay always shows the controls and current command. On macOS the CLI relaunches the viewer through the environment's `mjpython` executable automatically.

## Success Criteria

- Deterministic for a fixed seed.
- Held-out episodes reach food while avoiding threat contact in the bundled default configuration.
- Q values are updated and reported.
- Every command has a six-leg alternating-tripod phase record.
- The physical run controls all 42 official locomotion DOFs and six adhesion channels.
- The physical body produces finite nonzero horizontal displacement without ground collapse.
- Output collisions fail rather than overwrite data.
- The artifact labels itself an engineering demonstration, not proof of full biological intelligence.
