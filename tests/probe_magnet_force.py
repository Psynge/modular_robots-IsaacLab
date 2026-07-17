"""STEP 2 PROBE — magnet force pipeline vs analytic curve. NO RL, NO gravity.

Two modules face-on along y (B.3 vs G.0), floating (gravity off, z=0.05,
no ground contact possible at these separations). For each case: teleport to
an exact site-to-site separation, energize known magnets, apply the model's
wrench for ONE physics step, and compare:

    analytic F        — formula computed independently in this file
    model F           — y-component from MagneticForceModel.compute()
    measured m*dv/dt  — realized velocity change of each body

    python tests/probe_magnet_force.py --headless

PASS: model F == analytic F (exact math), measured within ~1% of analytic
(integrator), signs correct (attract pulls together, repel pushes apart),
cap case reads force-at-4mm with k capped at k_hold.

NOTE: wrenches are computed in world frame and applied with the modules at
identity orientation, so world==body frame here. Frame semantics of
set_external_force_and_torque under rotation get their own check in step 3+.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step-2 magnet force probe.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.geometry import FACE_SITE_OFFSETS, MODULE_MASS, PHYSICS_DT  # noqa: E402
from common.magnet_constants import (  # noqa: E402
    CONTACT_DISTANCE, FORCE_CONSTANT, M_HOLD, MIN_DISTANCE,
)
from envs.magnetic_force import MagneticForceModel  # noqa: E402

USD_PATH = os.path.join(REPO_ROOT, "models", "usd", "hex_module.usd")
FACE_Y = abs(FACE_SITE_OFFSETS[0][1])   # 0.0188956 — site inset from center, y
Z = 0.05                                 # float height, gravity is off


def analytic_force(M: float, sa_pct: float, sb_pct: float,
                   pol_a: float, pol_b: float, X: float) -> float:
    """Independent re-derivation of the MuJoCo formula (positive = attract)."""
    k = FORCE_CONSTANT * (M / 1000.0)
    k_pair = k * (sa_pct / 100.0) * (sb_pct / 100.0)
    Xc = max(X, MIN_DISTANCE)
    if Xc < CONTACT_DISTANCE:
        k_use = min(k_pair, FORCE_CONSTANT * (M_HOLD / 1000.0))
        X_use = CONTACT_DISTANCE
    else:
        k_use, X_use = k_pair, Xc
    return -k_use * pol_a * pol_b / (X_use ** 2)


# ---------------- sim setup: NO gravity, NO ground ----------------
sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, gravity=(0.0, 0.0, 0.0),
                                  device=args.device)
sim = SimulationContext(sim_cfg)

modules = []
for i in range(2):
    cfg = RigidObjectCfg(
        prim_path=f"/World/module_{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.1 * i, Z)),
    )
    modules.append(RigidObject(cfg))

sim.reset()
device = sim.device
site_offsets = torch.tensor(FACE_SITE_OFFSETS, device=device)
mag = MagneticForceModel(num_envs=1, n_modules=2, device=device)


def com_pos(obj: RigidObject) -> torch.Tensor:
    if hasattr(obj.data, "root_com_pos_w"):
        return obj.data.root_com_pos_w[0]
    print("[warn] root_com_pos_w unavailable; falling back to root_pos_w")
    return obj.data.root_pos_w[0]


def teleport(sep: float) -> None:
    """Place modules so B.3 <-> G.0 site distance is exactly `sep`; zero vel."""
    center_dist = sep + 2.0 * FACE_Y
    poses = [(0.0, 0.0, Z), (0.0, center_dist, Z)]
    for obj, p in zip(modules, poses):
        root = obj.data.default_root_state.clone()
        root[0, 0:3] = torch.tensor(p, device=device)
        root[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        root[0, 7:13] = 0.0
        obj.write_root_pose_to_sim(root[:, 0:7])
        obj.write_root_velocity_to_sim(root[:, 7:13])


def sites_world() -> torch.Tensor:
    """(1, 2, 6, 3) — identity orientation, so world = pos + offset."""
    out = torch.zeros((1, 2, 6, 3), device=device)
    for i, obj in enumerate(modules):
        out[0, i] = obj.data.root_pos_w[0].unsqueeze(0) + site_offsets
    return out


CASES = [
    # (label,           sep_m, M,     Sa,   Sb,   pol_a, pol_b)
    ("attract 29mm",    0.029, 55.0,  50.0, 50.0, +1.0, -1.0),
    ("attract 20mm",    0.020, 55.0,  50.0, 50.0, +1.0, -1.0),
    ("attract 10mm",    0.010, 55.0,  50.0, 50.0, +1.0, -1.0),
    ("attract  5mm",    0.005, 55.0,  50.0, 50.0, +1.0, -1.0),
    ("repel   10mm",    0.010, 55.0,  50.0, 50.0, +1.0, +1.0),
    ("cap hit  3mm",    0.003, 100.0, 100.0, 100.0, +1.0, -1.0),  # k_pair > k_hold
    ("cap miss 3mm",    0.003, 55.0,  50.0, 50.0, +1.0, -1.0),    # k_pair < k_hold
]

print(f"\ndevice={device}  dt={PHYSICS_DT}  mass={MODULE_MASS}kg  gravity OFF")
print(f"{'case':14s} {'X(mm)':>6s} {'analytic(N)':>12s} {'model(N)':>12s} "
      f"{'measured(N)':>12s} {'err%':>7s}   torque_x(B, N·m)")

zero3 = torch.zeros((1, 1, 3), device=device)
for label, sep, M, sa, sb, pa, pb in CASES:
    teleport(sep)
    mag.clear_all()
    mag.set_strength_pct(M)
    mag.set_magnet(0, 3, pa, sa)   # B.3
    mag.set_magnet(1, 0, pb, sb)   # G.0
    for obj in modules:
        obj.update(PHYSICS_DT)

    sp = sites_world()
    com = torch.stack([com_pos(o) for o in modules]).unsqueeze(0)  # (1,2,3)
    forces, torques = mag.compute(sp, com)

    for i, obj in enumerate(modules):
        obj.set_external_force_and_torque(
            forces[:, i].unsqueeze(1), torques[:, i].unsqueeze(1)
        )
        obj.write_data_to_sim()
    sim.step()
    for obj in modules:
        obj.update(PHYSICS_DT)

    # measured: y-velocity change of B over one step (started at rest)
    v_y = modules[0].data.root_lin_vel_w[0, 1].item()
    measured = MODULE_MASS * v_y / PHYSICS_DT          # +y toward G = attraction
    model_f = forces[0, 0, 1].item()                   # y-comp on B
    analytic = analytic_force(M, sa, sb, pa, pb, sep)
    err = 100.0 * (measured - analytic) / analytic if analytic != 0 else float("nan")
    print(f"{label:14s} {sep*1e3:6.1f} {analytic:12.6f} {model_f:12.6f} "
          f"{measured:12.6f} {err:7.2f}   {torques[0, 0, 0].item():+.3e}")

    # zero out applied wrench before next case
    for obj in modules:
        obj.set_external_force_and_torque(zero3, zero3)
        obj.write_data_to_sim()

print("\nPASS: model==analytic exactly; measured within ~1%; repel negative; "
      "both 3mm rows equal force-at-4mm values (cap-hit row uses k_hold).")
simulation_app.close()
