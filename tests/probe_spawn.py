"""STEP 1 PROBE — spawn stability, no magnets, no RL.

Isaac equivalent of the MuJoCo "seated line stable 2000 steps" test.
Spawns B, G, R in the verified y-axis line and just steps physics.

    python tests/probe_spawn.py --headless            # numbers only
    python tests/probe_spawn.py --num_steps 2000      # with viewer

PASS criteria (same as the MuJoCo verification):
  - no ejection: max per-module drift stays sub-millimeter over 2000 steps
  - faces meet flush: B.3-G.0 and G.3-R.0 site-pair distances ~2mm
    (sites are 1mm inward from each face -> flush contact reads ~2mm)
  - modules stay level: quaternion stays ~identity (no tipping)

Paste the full output back into the chat.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step-1 spawn stability probe.")
parser.add_argument("--num_steps", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below needs the app running ----
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.geometry import (  # noqa: E402
    FACE_SITE_OFFSETS, GRAVITY, GROUND_FRICTION, MODULE_FRICTION,
    MODULE_NAMES, PHYSICS_DT, SPAWN_POSITIONS,
)

USD_PATH = os.path.join(REPO_ROOT, "models", "usd", "hex_module.usd")

# ---------------- sim setup ----------------
sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, gravity=GRAVITY, device=args.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=(0.15, 0.0, 0.12), target=(0.0, 0.0, 0.0))

# ground plane with the MuJoCo ground friction
ground_cfg = sim_utils.GroundPlaneCfg(
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION
    )
)
ground_cfg.func("/World/ground", ground_cfg)

light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
light_cfg.func("/World/light", light_cfg)

# three modules at the verified y-axis line positions
modules = []
for i, (name, pos) in enumerate(zip(MODULE_NAMES, SPAWN_POSITIONS)):
    cfg = RigidObjectCfg(
        prim_path=f"/World/module_{i}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=MODULE_FRICTION, dynamic_friction=MODULE_FRICTION
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
    modules.append(RigidObject(cfg))

sim.reset()

site_offsets = torch.tensor(FACE_SITE_OFFSETS, device=sim.device)  # (6, 3)
spawn_pos = torch.tensor(SPAWN_POSITIONS, device=sim.device)       # (3, 3)


def world_sites(module: RigidObject) -> torch.Tensor:
    """World positions of the module's 6 face sites. Returns (6, 3)."""
    pos = module.data.root_pos_w[0]      # (3,)
    quat = module.data.root_quat_w[0]    # (4,) wxyz
    return pos.unsqueeze(0) + quat_apply(quat.expand(6, 4), site_offsets)


def report(step: int) -> None:
    drifts, tilts = [], []
    for i, m in enumerate(modules):
        drift = torch.norm(m.data.root_pos_w[0] - spawn_pos[i]).item()
        # w-component of quat -> tilt angle from identity
        w = m.data.root_quat_w[0, 0].clamp(-1.0, 1.0).item()
        tilt_deg = 2.0 * torch.rad2deg(torch.acos(torch.tensor(abs(w)))).item()
        drifts.append(drift)
        tilts.append(tilt_deg)
    sB, sG, sR = (world_sites(m) for m in modules)
    d_bg = torch.norm(sB[3] - sG[0]).item()   # B.3 - G.0
    d_gr = torch.norm(sG[3] - sR[0]).item()   # G.3 - R.0
    print(
        f"[{step:5d}] drift(mm) B={drifts[0]*1e3:6.3f} G={drifts[1]*1e3:6.3f} "
        f"R={drifts[2]*1e3:6.3f} | tilt(deg) B={tilts[0]:5.2f} G={tilts[1]:5.2f} "
        f"R={tilts[2]:5.2f} | B.3-G.0={d_bg*1e3:6.3f}mm  G.3-R.0={d_gr*1e3:6.3f}mm"
    )


print(f"device={sim.device}  dt={PHYSICS_DT}  gravity={GRAVITY}")
report(0)
for step in range(1, args.num_steps + 1):
    sim.step()
    for m in modules:
        m.update(PHYSICS_DT)
    if step % 200 == 0 or step == args.num_steps:
        report(step)

print("\nPASS criteria: drift < 1mm, tilt < 1deg, site pairs ~2mm and steady.")
simulation_app.close()
