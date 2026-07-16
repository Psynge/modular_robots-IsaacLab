# Continuation Prompt — HSS-SRMR Modular Robot RL (Isaac Lab port)

I'm John. This repo (`modular_robots-IsaacLab`, github.com/Psynge/modular_robots-IsaacLab)
is the Isaac Lab port of my MuJoCo project `modular-robots-mujoco`. Target: CoRL 2026.

**Machines:** HunterKiller — Ubuntu 24.04 laptop, RTX 4070 (8GB), driver 580.159,
Isaac Lab installed from source at `/opt/isaac/IsaacLab`, venv at
`/opt/isaac/env_isaaclab` (Python 3.11, torch 2.7.0+cu128, isaacsim pip).
Repo working copy at `/opt/isaac/modular_robots-IsaacLab`. Second machine:
T-1000 — Threadripper, dual RTX 3090 (24GB each), pulls the same repo.
Run scripts with the venv active: `python tests/<probe>.py --headless`.

**Working contract (unchanged from MuJoCo phase): one targeted change at a
time, explained before coding. Verify every change with a throwaway probe
whose output I paste back — nothing is "fixed" until a probe or the viewer
confirms it. Confirm geometry/formulas with me before coding. Don't
over-claim: a failed probe of one recipe does not prove a general
impossibility. Re-derive from what the data actually shows.**

## Physics model (ported from MuJoCo — see common/)
6 face magnets per hex module (faces 0-5), polarity -1/0/+1 + strength.
F = -(M_k·Sa·Sb)·Ia·Ib/X², opposite poles attract, CUTOFF 30mm site-to-site,
CONTACT/latch 4mm, M_HOLD=25, M_DEFAULT=55, FORCE_CONSTANT=1.332e-4, module
mass 34g. Modules B=0, G=1, R=2; connection strings lower-module-first
(e.g. `B.3-G.0`). Verified spawn: line along y (face-normal axis), spacing
39.7912mm, connecting faces 0 and 3. Gravity is 0.1g (0,0,-0.981) — copied
from the MuJoCo model, intentional. Constants live in `common/geometry.py`
and `common/magnet_constants.py`; the MuJoCo values are the starting point
but PhysX contact/friction behavior differs — every threshold-sensitive
result (latching, rolling, chamfer-lift) must be re-probed here, MuJoCo
results are not ground truth for this engine.

## Port plan (agreed order — do not skip ahead)
1. **Asset + spawn probe** — STL→USD (`scripts/convert_stl_to_usd.py`),
   3 modules spawn flush and stable 2000 steps (`tests/probe_spawn.py`).
2. **Batched MagneticForceModel** (`envs/magnetic_force.py`, torch, (num_envs,…)
   tensors, forces via set_external_force_and_torque) — verify with a
   two-module attraction probe against the analytical force curve.
3. **Seed-hold probe** — B3N50/G0S50/G3N50/R0S50, line stays latched.
4. **Rotation oscillation probe** (Isaac version of watch_rotation.py):
   hold R.0-G.3, alternate B.3-G.0 ↔ B.2-G.1 at M100, classify with
   ShapingReward logic. Answers "can modules rotate in PhysX" BEFORE any RL.
5. **Full direct-workflow env + ShapingReward + rsl_rl PPO.** Reward design
   carries over from MuJoCo: ROTATED +5 sole positive (net-new detents only),
   SLIDE -1, DISCONNECT -2 (grace 2), OUTRANGE -5, ALLOUT -10. Obs: 27 inputs,
   CoM-relative positions ÷ MODULE_DIAMETER, signed yaw diff from center
   module ÷ 180. Known MuJoCo-phase trap to avoid: training loop CLEARed the
   seed hold every step so the assembly collapsed at step 0 — decide
   deliberately how the hold interacts with the policy's actions.

## Status
Step 1 files written, NOT yet verified on hardware. Nothing runs until the
spawn probe passes. MuJoCo-phase open thread (does B roll between detents)
remains open — it becomes step 4 here.

## Deferred (carried over from MuJoCo phase)
- M_HOLD/M_DEFAULT (25/55) vs older notes (15/15) — reconcile.
- Locomotion (command-frame alignment idea) — shelved until reconfiguration learns.
- SUCCESS_REWARD calibration — meaningless until the new reward learns.
- Live-viewer polarity arrows / force lines — port from watch_policy_rl.py later.
