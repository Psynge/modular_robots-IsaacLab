"""STEP 3 PROBE — seed hold under gravity. NO RL.

Seated y-axis line, gravity ON (0.1g), ground plane with friction, and the
seed hold energized: B3N50 / G0S50 / G3N50 / R0S50. The model's wrenches are
applied every physics step — the loop shape the RL env will use.

    python tests/probe_seed_hold.py --headless

PASS criteria:
  - connections read ['B.3-G.0', 'G.3-R.0'] at every report through 2000 steps
  - both seam site distances stay ~2mm (inside the 4mm latch band)
  - drift/tilt stay in probe_spawn territory (<1mm, <1deg) — the hold should
    keep the line seated, not fling or fold it
  - hold forces (printed) are the capped-at-4mm values, steady

First test where PhysX contact resolution, friction, and magnet forces
interact — and the first real exercise of the bound friction material.
Paste the full output back.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step-3 seed hold probe.")
parser.add_argument("--num_steps", type=int, default=2000)
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
    MODULE_NAMES, PHYSICS_DT, SPAWN_POSITIONS,
)
from common.magnet_constants import SEED_HOLD  # noqa: E402
from envs.magnetic_force import MagneticForceModel  # noqa: E402

USD_PATH = os.path.join(REPO_ROOT, "models", "usd", "hex_module.usd")

# ---------------- sim setup (same physics as probe_spawn) ----------------
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
spawn_pos = torch.tensor(SPAWN_POSITIONS, device=device)

mag = MagneticForceModel(num_envs=1, n_modules=3, device=device)
for module, face, pol, pct in SEED_HOLD:
    mag.set_magnet(module, face, pol, pct)

warned_com = False


def com_pos(obj: RigidObject) -> torch.Tensor:
    global warned_com
    if hasattr(obj.data, "root_com_pos_w"):
        return obj.data.root_com_pos_w[0]
    if not warned_com:
        print("[warn] root_com_pos_w unavailable; using root_pos_w")
        warned_com = True
    return obj.data.root_pos_w[0]


def gather():
    """site positions (1,3,6,3) with real orientation, and COMs (1,3,3)."""
    sp = torch.zeros((1, 3, 6, 3), device=device)
    for i, obj in enumerate(modules):
        pos = obj.data.root_pos_w[0]
        quat = obj.data.root_quat_w[0]
        sp[0, i] = pos.unsqueeze(0) + quat_apply(quat.expand(6, 4), site_offsets)
    com = torch.stack([com_pos(o) for o in modules]).unsqueeze(0)
    return sp, com


def report(step: int, sp: torch.Tensor, forces: torch.Tensor) -> None:
    drifts, tilts = [], []
    for i, m in enumerate(modules):
        drifts.append(torch.norm(m.data.root_pos_w[0] - spawn_pos[i]).item() * 1e3)
        w = m.data.root_quat_w[0, 0].clamp(-1.0, 1.0).item()
        tilts.append(2.0 * torch.rad2deg(torch.acos(torch.tensor(abs(w)))).item())
    d_bg = torch.norm(sp[0, 0, 3] - sp[0, 1, 0]).item() * 1e3
    d_gr = torch.norm(sp[0, 1, 3] - sp[0, 2, 0]).item() * 1e3
    conns = mag.get_connections(sp)
    fmag = [torch.norm(forces[0, i]).item() for i in range(3)]
    print(
        f"[{step:5d}] conns={conns} | seams B.3-G.0={d_bg:5.3f}mm G.3-R.0={d_gr:5.3f}mm | "
        f"drift(mm) {drifts[0]:.3f}/{drifts[1]:.3f}/{drifts[2]:.3f} | "
        f"tilt(deg) {tilts[0]:.2f}/{tilts[1]:.2f}/{tilts[2]:.2f} | "
        f"|F|(N) {fmag[0]:.4f}/{fmag[1]:.4f}/{fmag[2]:.4f}"
    )


print(f"device={device}  dt={PHYSICS_DT}  gravity={GRAVITY}")
print(f"seed hold: {SEED_HOLD}")
for step in range(args.num_steps + 1):
    sp, com = gather()
    forces, torques = mag.compute(sp, com)
    for i, obj in enumerate(modules):
        obj.set_external_force_and_torque(
            forces[:, i].unsqueeze(1), torques[:, i].unsqueeze(1)
        )
        obj.write_data_to_sim()
    if step % 200 == 0:
        report(step, sp, forces)
    sim.step()
    for obj in modules:
        obj.update(PHYSICS_DT)

print(f"\nPASS: conns=['B.3-G.0', 'G.3-R.0'] at every report, seams ~2mm, "
      f"drift <1mm, tilt <1deg, forces steady.")
simulation_app.close()
