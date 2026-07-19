"""STEP 4 PROBE — rotation oscillation (Isaac watch_rotation). NO NN.

The open-thread test, MuJoCo recipe reproduced with settle-triggered phases:
hold R.0-G.3 fixed (R0N/G3S), alternate the B-G latch forever between

    phase A:  B.3-G.0   (G0N / B3S)
    phase B:  B.2-G.1   (G1N / B2S)

— one detent apart — everything at M100, full strength. Each phase runs
until the assembly settles (all module speeds below thresholds for
SETTLE_STEPS consecutive steps) or SETTLE_TIMEOUT steps elapse, then the
B-G connection is classified against the previous phase end.

    python tests/watch_rotation.py --headless               # numbers only
    python tests/watch_rotation.py --speed 1                # watch it live

Classification (rotation matcher brought forward from ShapingReward —
the full reward class stays in step 5):
    ROTATE+ / ROTATE-  B-G faces shifted (fa±1, fb∓1) mod 6
    JUMP a->b          shifted more than one detent
    hold               same faces latched
    break              was latched, now isn't
    form               newly latched (no prior to compare)
    (none)             not latched at either end

Expected if B physically rolls: alternating ROTATE- / ROTATE+ every phase.
The four outcome branches (continuation prompt) interpret anything else.

Tunable constants below — edit freely.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step-4 rotation oscillation probe.")
parser.add_argument("--num_phases", type=int, default=12)
parser.add_argument("--speed", type=float, default=0.0,
                    help="Real-time factor for viewing (0 = uncapped).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

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
M_GLOBAL = 100.0          # everything at M100 per the recipe
SETTLE_LIN = 0.001        # m/s   — "settled" linear speed ceiling
SETTLE_ANG = 0.05         # rad/s — "settled" angular speed ceiling
SETTLE_STEPS = 50         # consecutive quiet steps to call it settled
SETTLE_TIMEOUT = 4000     # steps (8s) — give up waiting and classify anyway
# ─────────────────────────────────────────────────────────────────────

# phase -> list of (module, face, polarity); strength 100 implied
HOLD = [(2, 0, +1.0), (1, 3, -1.0)]            # R0N / G3S — never changes
PHASES = {
    "A: B.3-G.0": [(1, 0, +1.0), (0, 3, -1.0)],   # G0N / B3S
    "B: B.2-G.1": [(1, 1, +1.0), (0, 2, -1.0)],   # G1N / B2S
}

sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, gravity=GRAVITY, device=args.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=(0.15, 0.10, 0.10), target=(0.0, 0.0, 0.005))

ground_cfg = sim_utils.GroundPlaneCfg(
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION
    )
)
ground_cfg.func("/World/ground", ground_cfg)
if not args.headless:
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/light", light_cfg)

COLORS = [(0.2, 0.4, 1.0), (0.2, 0.9, 0.3), (1.0, 0.25, 0.2)]
modules = []
for i, pos in enumerate(SPAWN_POSITIONS):
    cfg = RigidObjectCfg(
        prim_path=f"/World/module_{i}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[i]),
        ),
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


def set_phase(phase_magnets) -> None:
    """Hold + phase pair, all at 100%, global M100."""
    mag.clear_all()
    mag.set_strength_pct(M_GLOBAL)
    for module, face, pol in HOLD + phase_magnets:
        mag.set_magnet(module, face, pol, 100.0)


def bg_connection(sp):
    """(fa, fb) of the latched attracting B-G pair, or None."""
    for a, fa, b, fb, X, F in mag.get_active_pairs(sp):
        if a == 0 and b == 1 and X < 0.004 and F > 0:
            return (fa, fb)
    return None


def classify(prev, curr) -> str:
    if prev is None and curr is None:
        return "(none)"
    if prev is None:
        return f"form {curr}"
    if curr is None:
        return f"break (was {prev})"
    if prev == curr:
        return "hold"
    da = ((curr[0] - prev[0] + 3) % 6) - 3   # wrap to -3..2
    db = ((curr[1] - prev[1] + 3) % 6) - 3
    if da == -1 and db == +1:
        return "ROTATE-"
    if da == +1 and db == -1:
        return "ROTATE+"
    return f"JUMP {prev}->{curr}"


def b_yaw_deg() -> float:
    w, x, y, z = modules[0].data.root_quat_w[0].tolist()
    import math
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_phase():
    """Step until settled or timeout. Returns (steps_run, settled)."""
    quiet = 0
    for step in range(1, SETTLE_TIMEOUT + 1):
        t0 = time.perf_counter()
        sp, com = gather()
        forces, torques = mag.compute(sp, com)
        for i, obj in enumerate(modules):
            obj.set_external_force_and_torque(
                forces[:, i].unsqueeze(1), torques[:, i].unsqueeze(1)
            )
            obj.write_data_to_sim()
        sim.step()
        vmax = wmax = 0.0
        for obj in modules:
            obj.update(PHYSICS_DT)
            vmax = max(vmax, torch.norm(obj.data.root_lin_vel_w[0]).item())
            wmax = max(wmax, torch.norm(obj.data.root_ang_vel_w[0]).item())
        quiet = quiet + 1 if (vmax < SETTLE_LIN and wmax < SETTLE_ANG) else 0
        if quiet >= SETTLE_STEPS:
            return step, True
        if args.speed > 0:
            budget = PHYSICS_DT / args.speed
            el = time.perf_counter() - t0
            if el < budget:
                time.sleep(budget - el)
    return SETTLE_TIMEOUT, False


print(f"device={device}  dt={PHYSICS_DT}  gravity={GRAVITY}  M={M_GLOBAL}")
print(f"hold: R0N/G3S | settle: v<{SETTLE_LIN} w<{SETTLE_ANG} x{SETTLE_STEPS}, "
      f"timeout {SETTLE_TIMEOUT}\n")

phase_names = list(PHASES.keys())
prev_conn = None
for n in range(args.num_phases):
    name = phase_names[n % 2]
    set_phase(PHASES[name])
    steps, settled = run_phase()
    sp, _ = gather()
    conn = bg_connection(sp)
    label = classify(prev_conn, conn)
    bp = modules[0].data.root_pos_w[0]
    dmin = mag.min_face_pair_distance(sp).item()
    print(f"[phase {n:2d} {name}] {'settled' if settled else 'TIMEOUT'} "
          f"@{steps:4d} steps | B-G={conn} -> {label:18s} | "
          f"B pos=({bp[0].item()*1e3:+7.2f},{bp[1].item()*1e3:+7.2f})mm "
          f"yaw={b_yaw_deg():+7.2f}deg | min pair={dmin*1e3:5.2f}mm")
    prev_conn = conn

print("\nInterpretation: alternating ROTATE-/ROTATE+ = rolls & detection OK; "
      "hold = no roll (force/friction wall); break/form churn = gap > matcher; "
      "JUMP = multi-detent per settle.")
simulation_app.close()
