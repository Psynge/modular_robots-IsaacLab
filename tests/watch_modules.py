"""Live viewer — watch the modules behave in real time. NO NN, no magnets (yet).

Isaac counterpart of the MuJoCo watch scripts. Runs until you close the
viewer window (or Ctrl+C).

    python tests/watch_modules.py                 # real time (1x)
    python tests/watch_modules.py --speed 0.25    # slow motion
    python tests/watch_modules.py --speed 0       # uncapped, as fast as possible

Prints a status line (drift / tilt / connecting-face site distances) once
per simulated second, same quantities as probe_spawn.py, so what you see
and what the numbers say can be compared directly.

Controls in the viewer: right-drag orbits, middle-drag pans, scroll zooms.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Live module viewer.")
parser.add_argument(
    "--speed", type=float, default=1.0,
    help="Real-time factor. 1.0 = real time, 0.25 = quarter speed, 0 = uncapped.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.headless:
    print("[watch_modules] --headless makes no sense for a viewer script; ignoring.")
    args.headless = False

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

sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, gravity=GRAVITY, device=args.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=(0.15, 0.10, 0.10), target=(0.0, 0.0, 0.005))

ground_cfg = sim_utils.GroundPlaneCfg(
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION
    )
)
ground_cfg.func("/World/ground", ground_cfg)
light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
light_cfg.func("/World/light", light_cfg)

# distinct colors so B/G/R are tellable apart in the viewer
COLORS = [(0.2, 0.4, 1.0), (0.2, 0.9, 0.3), (1.0, 0.25, 0.2)]
modules = []
for i, (name, pos) in enumerate(zip(MODULE_NAMES, SPAWN_POSITIONS)):
    cfg = RigidObjectCfg(
        prim_path=f"/World/module_{i}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[i]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
    modules.append(RigidObject(cfg))

# module friction: spawn one shared material prim and bind it to each module
# (UsdFileCfg no longer takes physics_material directly).
module_mat_cfg = sim_utils.RigidBodyMaterialCfg(
    static_friction=MODULE_FRICTION, dynamic_friction=MODULE_FRICTION
)
module_mat_cfg.func("/World/Materials/module_material", module_mat_cfg)
for i in range(len(modules)):
    sim_utils.bind_physics_material(f"/World/module_{i}", "/World/Materials/module_material")

sim.reset()

site_offsets = torch.tensor(FACE_SITE_OFFSETS, device=sim.device)
spawn_pos = torch.tensor(SPAWN_POSITIONS, device=sim.device)


def world_sites(module: RigidObject) -> torch.Tensor:
    pos = module.data.root_pos_w[0]
    quat = module.data.root_quat_w[0]
    return pos.unsqueeze(0) + quat_apply(quat.expand(6, 4), site_offsets)


def status(step: int) -> None:
    drifts, tilts = [], []
    for i, m in enumerate(modules):
        drifts.append(torch.norm(m.data.root_pos_w[0] - spawn_pos[i]).item() * 1e3)
        w = m.data.root_quat_w[0, 0].clamp(-1.0, 1.0).item()
        tilts.append(2.0 * torch.rad2deg(torch.acos(torch.tensor(abs(w)))).item())
    sB, sG, sR = (world_sites(m) for m in modules)
    d_bg = torch.norm(sB[3] - sG[0]).item() * 1e3
    d_gr = torch.norm(sG[3] - sR[0]).item() * 1e3
    print(
        f"[t={step * PHYSICS_DT:7.2f}s] drift(mm) "
        f"B={drifts[0]:6.3f} G={drifts[1]:6.3f} R={drifts[2]:6.3f} | "
        f"tilt(deg) B={tilts[0]:5.2f} G={tilts[1]:5.2f} R={tilts[2]:5.2f} | "
        f"B.3-G.0={d_bg:6.3f}mm  G.3-R.0={d_gr:6.3f}mm"
    )


steps_per_status = max(1, int(round(1.0 / PHYSICS_DT)))  # once per sim-second
step = 0
status(step)
try:
    while simulation_app.is_running():
        t0 = time.perf_counter()
        sim.step()
        for m in modules:
            m.update(PHYSICS_DT)
        step += 1
        if step % steps_per_status == 0:
            status(step)
        if args.speed > 0:
            budget = PHYSICS_DT / args.speed
            elapsed = time.perf_counter() - t0
            if elapsed < budget:
                time.sleep(budget - elapsed)
except KeyboardInterrupt:
    pass

simulation_app.close()
