# modular_robots-IsaacLab

Isaac Lab port of [modular-robots-mujoco](https://github.com/Psynge/modular-robots-mujoco):
RL for 3-module HSS-SRMR reconfiguration. See CONTINUATION_PROMPT.md for full
project state and the working contract.

## Layout (mirrors the MuJoCo repo)
- `common/` — geometry + magnet constants ported from MuJoCo (single source of truth)
- `envs/` — magnetic force model + RL env (steps 2 and 5)
- `models/meshes/` — source STL; `models/usd/` — generated, gitignored
- `scripts/` — one-time utilities (STL→USD conversion)
- `tests/` — probes, one per verification step
- `training/` — rsl_rl PPO configs/runners (step 5)

## Setup on a new machine
Prereqs: Isaac Lab installed (tested with source install at /opt/isaac/IsaacLab,
venv at /opt/isaac/env_isaaclab).

    source /opt/isaac/env_isaaclab/bin/activate
    git clone https://github.com/Psynge/modular_robots-IsaacLab.git
    cd modular_robots-IsaacLab
    python scripts/convert_stl_to_usd.py --headless   # regenerates models/usd/
    python tests/probe_spawn.py --headless            # step-1 verification

## Current step
Step 1 (spawn stability) — files written, awaiting probe verification.
