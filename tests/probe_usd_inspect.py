"""THROWAWAY PROBE — inspect models/usd/hex_module.usd.

pxr ships inside Isaac's runtime, so this needs the app (headless is fine):

    python tests/probe_usd_inspect.py --headless

Prints the on-disk files, the prim tree with applied physics APIs, and
whether the geometry reference resolves. Paste the full output back.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="USD inspector probe.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_DIR = os.path.join(REPO_ROOT, "models", "usd")
USD_PATH = os.path.join(USD_DIR, "hex_module.usd")

print("=== files under models/usd/ ===")
for root, _dirs, files in os.walk(USD_DIR):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, USD_DIR):40s} {os.path.getsize(p):>10d} bytes")

print("\n=== stage ===")
stage = Usd.Stage.Open(USD_PATH)
default_prim = stage.GetDefaultPrim()
print(f"default prim: {default_prim.GetPath() if default_prim else 'NONE'}")
print(f"up axis: {UsdGeom.GetStageUpAxis(stage)}   meters/unit: {UsdGeom.GetStageMetersPerUnit(stage)}")

print("\n=== prim tree (instance proxies included) ===")
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if prim.IsPseudoRoot():
        continue
    indent = "  " * (len(prim.GetPath().pathString.split("/")) - 1)
    flags = []
    if prim.IsInstanceable():
        flags.append("instanceable")
    if prim.IsInstance():
        flags.append("instance")
    if prim.IsInstanceProxy():
        flags.append("proxy")
    if prim.HasAuthoredReferences():
        flags.append("has-references")
    schemas = list(prim.GetAppliedSchemas())
    print(f"{indent}{prim.GetPath()}  <{prim.GetTypeName() or 'over'}>"
          f"{'  [' + ', '.join(flags) + ']' if flags else ''}")
    if schemas:
        print(f"{indent}    applied: {schemas}")

print("\n=== composition check on geometry prim ===")
geom = stage.GetPrimAtPath(f"{default_prim.GetPath()}/geometry") if default_prim else None
if geom and geom.IsValid():
    print(f"geometry prim valid: True, children: "
          f"{[c.GetName() for c in Usd.PrimRange(geom, Usd.TraverseInstanceProxies()) if c != geom]}")
else:
    print("geometry prim valid: False  <-- reference likely broken")

os._exit(0) # skip kit's wedge-prone shutdown; all output/saves are done above
