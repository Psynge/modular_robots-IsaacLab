"""FORCE-THRESHOLD RAMP PROBE — at what M does the detent complete? NO NN.

Step-4 follow-up. The oscillation probe showed the B-G roll stalls at a
genuine static equilibrium ~1.4-8.2 deg into the 60 deg detent at M100.
This probe runs INDEPENDENT trials at increasing global M to discriminate:

  - completes at some M       -> force-scale wall; the multiplier is the answer
  - pinned at ~8 deg at any M -> geometric wall (chamfer/contact) -> mesh probe next
  - overshoots (JUMP) at high M -> matcher/settle design info for the env

Each trial: teleport to the spawn line -> seat the A latch (B.3-G.0) at
M100 for SEAT_STEPS -> switch to the B recipe (B.2-G.1) at the trial M ->
run to settle/timeout, tracking MAX yaw reached (not just final: an
overshoot-and-fall-back has a different signature than never-exceeding).

NOTE: global M scales ALL pairs including the G-R hold — the hold gets
stronger with the trial M, which is what we want (hold must not be the
weak link while we probe the B-G seam).

    python tests/probe_force_threshold.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Force-threshold M-ramp probe.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.geometry import (  # noqa: E402
    FACE_SITE_OFFSETS, GRAVITY, GROUND_FRICTION, MODULE_FRICTION,
    PHYSICS_DT, SPAWN_POSITIONS,
)
from envs.magnetic_force import MagneticForceModel  # noqa: E402

USD_PATH = os.path.join(REPO_ROOT, "models", "usd", "hex_module.usd")

# ── tunables ─────────────────────────────────────────────────────────
M_LEVELS = [100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0]
SEAT_STEPS = 1000         # A-latch seating at M100 before each trial
SETTLE_LIN = 0.001
SETTLE_ANG = 0.05
SETTLE_STEPS = 50
SETTLE_TIMEOUT = 4000
STOP_AT_FIRST_SUCCESS = False   # run all levels; set True to stop on ROTATED
# ─────────────────────────────────────────────────────────────────────

HOLD = [(2, 0, +1.0), (1, 3, -1.0)]            # R0N / G3S
RECIPE_A = [(1, 0, +1.0), (0, 3, -1.0)]        # G0N / B3S  (seat)
RECIPE_B = [(1, 1, +1.0), (0, 2, -1.0)]        # G1N / B2S  (target)

sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, gravity=GRAVITY, device=args.device)
sim = SimulationContext(sim_cfg)

ground_cfg = sim_utils.GroundPlaneCfg(
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION
    )
)
ground_cfg.func("/World/ground", ground_cfg)

modules = []
for i, pos in enumerate(SPAWN_POSITIONS):
    cfg = RigidObjectCfg(
        prim_path=f"/World/module_{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
    modules.append(RigidObject(cfg))

module_mat_cfg = sim_utils.RigidBodyMaterialCfg(
    static_friction=MODULE_FRICTION, dynamic_friction=MODULE_FRICTION
)
module_mat_cfg.func("/World/Materials/module_material", module_mat_cfg)
for i in range(len(modules)):
    sim_utils.bind_physics_material(f"/World/module_{i}", "/World/Materials/module_material")

sim.reset()
device = sim.device
site_offsets = torch.tensor(FACE_SITE_OFFSETS, device=device)
mag = MagneticForceModel(num_envs=1, n_modules=3, device=device)

warned_com = False


def com_pos(obj):
    global warned_com
    if hasattr(obj.data, "root_com_pos_w"):
        return obj.data.root_com_pos_w[0]
    if not warned_com:
        print("[warn] root_com_pos_w unavailable; using root_pos_w")
        warned_com = True
    return obj.data.root_pos_w[0]


def gather():
    sp = torch.zeros((1, 3, 6, 3), device=device)
    for i, obj in enumerate(modules):
        pos = obj.data.root_pos_w[0]
        quat = obj.data.root_quat_w[0]
        sp[0, i] = pos.unsqueeze(0) + quat_apply(quat.expand(6, 4), site_offsets)
    com = torch.stack([com_pos(o) for o in modules]).unsqueeze(0)
    return sp, com


def teleport_line() -> None:
    for obj, pos in zip(modules, SPAWN_POSITIONS):
        root = obj.data.default_root_state.clone()
        root[0, 0:3] = torch.tensor(pos, device=device)
        root[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        root[0, 7:13] = 0.0
        obj.write_root_pose_to_sim(root[:, 0:7])
        obj.write_root_velocity_to_sim(root[:, 7:13])


def set_recipe(recipe, m_global: float) -> None:
    mag.clear_all()
    mag.set_strength_pct(m_global)
    for module, face, pol in HOLD + recipe:
        mag.set_magnet(module, face, pol, 100.0)


def b_yaw_deg() -> float:
    w, x, y, z = modules[0].data.root_quat_w[0].tolist()
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def bg_connection(sp):
    for a, fa, b, fb, X, F in mag.get_active_pairs(sp):
        if a == 0 and b == 1 and X < 0.004 and F > 0:
            return (fa, fb)
    return None


def step_once() -> None:
    sp, com = gather()
    forces, torques = mag.compute(sp, com)
    for i, obj in enumerate(modules):
        obj.set_external_force_and_torque(
            forces[:, i].unsqueeze(1), torques[:, i].unsqueeze(1)
        )
        obj.write_data_to_sim()
    sim.step()
    for obj in modules:
        obj.update(PHYSICS_DT)


def vels():
    vmax = wmax = 0.0
    for obj in modules:
        vmax = max(vmax, torch.norm(obj.data.root_lin_vel_w[0]).item())
        wmax = max(wmax, torch.norm(obj.data.root_ang_vel_w[0]).item())
    return vmax, wmax


print(f"device={device}  dt={PHYSICS_DT}  gravity={GRAVITY}")
print(f"seat: A recipe @M100 x{SEAT_STEPS} | trial: B recipe, settle "
      f"v<{SETTLE_LIN} w<{SETTLE_ANG} x{SETTLE_STEPS}, timeout {SETTLE_TIMEOUT}\n")

for M in M_LEVELS:
    # independent trial: fresh line, seat the A latch at M100
    teleport_line()
    set_recipe(RECIPE_A, 100.0)
    for _ in range(SEAT_STEPS):
        step_once()
    sp, _ = gather()
    seat_conn = bg_connection(sp)

    # switch to the target recipe at the trial M
    set_recipe(RECIPE_B, M)
    yaw_max, quiet, steps_run, settled = 0.0, 0, 0, False
    for step in range(1, SETTLE_TIMEOUT + 1):
        step_once()
        steps_run = step
        yaw = b_yaw_deg()
        if abs(yaw) > abs(yaw_max):
            yaw_max = yaw
        vmax, wmax = vels()
        quiet = quiet + 1 if (vmax < SETTLE_LIN and wmax < SETTLE_ANG) else 0
        if quiet >= SETTLE_STEPS:
            settled = True
            break

    sp, _ = gather()
    conn = bg_connection(sp)
    outcome = ("ROTATED" if conn == (2, 1)
               else "no-latch" if conn is None
               else f"latch{conn}")
    print(f"[M={M:5.0f}] seat={seat_conn} | {'settled' if settled else 'TIMEOUT'} "
          f"@{steps_run:4d} | end={conn} {outcome:10s} | "
          f"yaw end={b_yaw_deg():+7.2f} max={yaw_max:+7.2f} deg")
    if STOP_AT_FIRST_SUCCESS and conn == (2, 1):
        break

print("\nInterpretation: ROTATED at some M -> force-scale wall (note the "
      "multiplier). Pinned yaw at all M -> geometric wall (chamfer/contact). "
      "max>>end -> overshoot-and-return (energy/damping signature).")
simulation_app.close()
