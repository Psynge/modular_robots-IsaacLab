"""Apply physics APIs to models/usd/hex_module.usd in place.

The MeshConverter on this Isaac build emits pure geometry (verified with
probe_usd_inspect.py — no physics APIs applied), so we stamp them on
directly: RigidBodyAPI + MassAPI (34g) on the root, CollisionAPI +
MeshCollisionAPI (convexHull) on the mesh. Idempotent — safe to re-run.

Run after every re-conversion:

    python scripts/apply_physics_to_usd.py --headless

Then verify with: python tests/probe_usd_inspect.py --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stamp physics APIs onto hex_module.usd.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_PATH = os.path.join(REPO_ROOT, "models", "usd", "hex_module.usd")

MODULE_MASS = 0.034  # kg — keep in sync with common/geometry.py

stage = Usd.Stage.Open(USD_PATH)
root = stage.GetDefaultPrim()
assert root and root.IsValid(), f"no default prim in {USD_PATH}"

mesh = stage.GetPrimAtPath(f"{root.GetPath().pathString}/geometry/mesh")
assert mesh and mesh.IsValid(), "expected mesh at <root>/geometry/mesh (run probe_usd_inspect.py)"

# root: dynamic rigid body with explicit mass
UsdPhysics.RigidBodyAPI.Apply(root)
mass_api = UsdPhysics.MassAPI.Apply(root)
mass_api.CreateMassAttr(MODULE_MASS)

# mesh: collider with convex-hull approximation (exact for our convex hex prism)
UsdPhysics.CollisionAPI.Apply(mesh)
mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(mesh)
mesh_col_api.CreateApproximationAttr("convexHull")

stage.Save()

print(f"[OK] saved {USD_PATH}")
print(f"root  {root.GetPath()}: {list(root.GetAppliedSchemas())}")
print(f"mesh  {mesh.GetPath()}: {list(mesh.GetAppliedSchemas())}")

os._exit(0) # skip kit's wedge-prone shutdown; all output/saves are done above
