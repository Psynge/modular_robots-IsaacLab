# Continuation Prompt — HSS-SRMR Modular Robot RL (Isaac Lab port, PAUSED)

I'm John. This repo (`modular_robots-IsaacLab`, github.com/Psynge/modular_robots-IsaacLab)
is the Isaac Lab port of my MuJoCo project `modular-robots-mujoco`. Target: CoRL 2026.
**Status: paused July 18, 2026 after completing verification steps 1-4. This
prompt is the full restart state — read it before touching anything.**

**Machines:** HunterKiller — Ubuntu 24.04 laptop, RTX 4070 (8GB), driver 580.159,
Isaac Sim 5.1 (pip) + Isaac Lab source at `/opt/isaac/IsaacLab`, venv at
`/opt/isaac/env_isaaclab` (Py3.11, torch 2.7.0+cu128). Working copy:
`/opt/isaac/modular_robots-IsaacLab`. T-1000 — Threadripper, dual RTX 3090,
pulls the same repo. Activate env first: `source /opt/isaac/env_isaaclab/bin/activate`.
Container is fresh each session — re-upload files Claude needs, or point it at
the public repo to clone.

**Working contract (unchanged): one targeted change at a time, explained before
coding. Verify every change with a throwaway probe whose output I paste back —
nothing is "fixed" until a probe or the viewer confirms it. Confirm
geometry/formulas with me before coding. Don't over-claim: a failed probe of
one recipe does not prove a general impossibility (this rule PAID OFF — see
finding below). Re-derive from what the data actually shows.**

## Architecture
Constants are the single source of truth, ported verbatim from MuJoCo and
shared by everything: `common/geometry.py` (hex module, face-site offsets,
verified y-axis spawn line, 0.1g gravity — intentional, friction 0.4/0.2) and
`common/magnet_constants.py` (F = -k_use·Ia·Ib/X_use², k = FORCE_CONSTANT·M/1000,
CUTOFF 30mm, CONTACT/latch 4mm with force capped AND evaluated at 4mm,
MIN_DISTANCE 1mm floor, M_HOLD=25, M_DEFAULT=55, seed hold B3N50/G0S50/G3N50/R0S50).
`envs/magnetic_force.py` is the batched torch MagneticForceModel: pure math,
(num_envs, 3, 6) state, world site positions + COMs in → world-frame
force/torque wrenches about COM out; caller applies them via
`set_external_force_and_torque` each step. Modules B=0,G=1,R=2, connection
strings lower-module-first (`B.3-G.0`).

## Verified (steps 1-4, all probe outputs in the conversation record / git log)
1. **Spawn** (`tests/probe_spawn.py`): 3 modules seat flush on the y-line,
   frozen 2000 steps, seams ~2.2mm. PASS.
2. **Force model** (`tests/probe_magnet_force.py`): model == analytic == PhysX
   measured (m·Δv/dt) to all digits, 29→3mm, attract/repel/both cap branches. PASS.
3. **Seed hold** (`tests/probe_seed_hold.py`): line stays latched 2000 steps,
   hold forces exactly the capped 0.1145N. Settles at ~2.4° hold-torque pitch
   (≈ the 2° chamfer — suggestive, unconfirmed). The MuJoCo collapse-at-step-0
   failure does NOT occur when the hold is maintained → training-loop CLEAR
   remains prime suspect for the MuJoCo plateau. PASS.
4. **Rotation** (`tests/watch_rotation.py`, `tests/probe_force_threshold.py`):
   **THE OPEN THREAD IS ANSWERED: modules CAN rotate between detents in PhysX.**
   - At M100, the de-energize/re-energize recipe stalls: latch breaks, B
     settles at a genuine static equilibrium 1.4-8.2° into the 60° detent.
   - **Threshold is between M100 and M150.** At M150: full detent roll,
     latched (2,1) at yaw +59.4° — but underdamped, overshot to +128° (>1
     extra detent) before falling back. Matcher/grace must tolerate this.
   - M≥200 destroys the assembly (spins free / torn apart). Usable band is
     roughly M120-180, UNREFINED.
   - KNOWN PROBE DEFECT: ramp trials alternated seat=(3,0)/seat=None — state
     leaks across trials despite teleport (suspects: applied wrench not zeroed
     before teleport; PhysX contact warm-start). M100/M150 rows had clean
     seats and stand; M≥200 rows are directional only. FIX BEFORE TRUSTING:
     zero wrench + ~200 no-magnet settle steps post-teleport, finer sweep
     (110-200), log G-R hold state at trial end.

## Build quirks (cost days — do not rediscover)
- This Isaac build's MeshConverter emits BARE GEOMETRY (no physics APIs).
  After ANY re-conversion run `scripts/apply_physics_to_usd.py --headless`
  (stamps RigidBodyAPI + MassAPI 34g on root, convexHull collision on mesh).
  Verify with `tests/probe_usd_inspect.py --headless`.
- API skew vs docs: MeshConverterCfg wants `mesh_collision_props=
  MeshCollisionPropertiesCfg(mesh_approximation_name=...)`; UsdFileCfg has NO
  `physics_material` (spawn a RigidBodyMaterialCfg prim + `bind_physics_material`
  per module — see any probe); `pxr` imports only inside a running AppLauncher.
  When an API arg is rejected, fetch the source from the IsaacLab repo — don't guess.
- `simulation_app.close()` wedges routinely: Ctrl+Z then `kill -9 %1`
  IMMEDIATELY — suspended kits hold VRAM and starved PhysX once (PxgCuda
  alloc fail → no physics scene). `pkill -9 -f "python tests/"` cleans up.
- `set_external_force_and_torque` is deprecation-warned. Migration to
  `permanent_wrench_composer.set_forces_and_torques(forces, torques, positions)`
  is DEFERRED deliberately: the new API applies at link frame and can compose
  torque from positions itself (could replace our manual r×F) — migrate as its
  own step WITH probes 2+3 re-run as regression.
- Viewer: use RTX Real-Time renderer, NOT RTX Interactive/path tracing
  (VRAM). `tests/watch_modules.py --speed 1` = live view, colored B/G/R.

## Where to resume
1. Fix the ramp-probe seat defect (above), re-run, get the clean M map
   (threshold + destruction edge + hold integrity).
2. Decide env force design from that map (M range the policy commands;
   matcher gap/JUMP tolerance for the underdamped roll).
3. Step 5: direct-workflow env + ShapingReward port + rsl_rl PPO.
   Reward: ROTATED +5 sole positive (net-new detents), SLIDE -1,
   DISCONNECT -2 (grace 2), OUTRANGE -5, ALLOUT -10; obs 27 (CoM-relative
   ÷ MODULE_DIAMETER, signed yaw from center module ÷180). CRITICAL design
   decision from the MuJoCo post-mortem: the policy must NOT wipe the seed
   hold every step (the old loop's CLEAR did; prime plateau suspect).
   "Latched" ≠ "settled-by-velocity" in PhysX (A-phases jitter forever at
   contact) — don't build terminators on velocity thresholds alone.

## Deferred (carried from MuJoCo phase + new)
- M_HOLD/M_DEFAULT (25/55) vs older notes (15/15) — reconcile.
- Chamfer-lift never verified working; step-3's 2.4° pitch is circumstantial.
- Locomotion (command-frame alignment idea) — shelved until reconfiguration learns.
- Viewer polarity arrows/force lines — port from MuJoCo watch_policy_rl.py.
- `docs/PROGRESS_REPORT.md` — professor-facing summary as of the pause.
