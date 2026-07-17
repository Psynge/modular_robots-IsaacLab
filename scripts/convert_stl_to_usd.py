"""One-time conversion: models/meshes/hex_module_centered.stl -> models/usd/hex_module.usd

Run from the repo root, inside the env_isaaclab environment:

    python scripts/convert_stl_to_usd.py --headless

Collision approximation is convexHull: the hex module is a convex prism
(the 2-degree face chamfer keeps it convex), so the hull is essentially
exact — no decomposition error to worry about.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert hex module STL to USD.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below needs the app running ----
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STL_PATH = os.path.join(REPO_ROOT, "models", "meshes", "hex_module_centered.stl")
USD_DIR = os.path.join(REPO_ROOT, "models", "usd")

cfg = MeshConverterCfg(
    asset_path=STL_PATH,
    usd_dir=USD_DIR,
    usd_file_name="hex_module.usd",
    force_usd_conversion=True,
    make_instanceable=True,
    mesh_collision_props=schemas_cfg.MeshCollisionPropertiesCfg(mesh_approximation_name="convexHull"),
    mass_props=schemas_cfg.MassPropertiesCfg(mass=0.034),
    rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
    collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
)

converter = MeshConverter(cfg)
print(f"[OK] wrote {converter.usd_path}")

simulation_app.close()
