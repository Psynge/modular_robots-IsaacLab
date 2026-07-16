"""Geometry constants for the HSS-SRMR hex modules.

Ported verbatim from the MuJoCo model (models/three_module_rl.xml in the
MuJoCo repo). All units are meters. Values were hand-verified in the MuJoCo
phase of the project — do not change without re-deriving.

Module frame convention (matches hex_module_centered.stl):
  - Flat-top hexagon. +x is a VERTEX axis (43.4mm vertex-to-vertex).
  - +/-y are FACE-NORMAL axes (40mm across flats) -> faces 0 (-y) and 3 (+y).
  - Mesh base sits at z=0, top at z=0.010.
"""

# --- module ---
MODULE_MASS = 0.034          # kg
MODULE_DIAMETER = 0.040      # m, across flats (used for obs normalization)
MODULE_HEIGHT = 0.010        # m

# --- magnet site offsets in module frame (faces 0..5) ---
# 1mm inward from each face surface, at magnet height z=0.0069894.
# Copied from the act_*_{0..5} sites in three_module_rl.xml.
FACE_SITE_OFFSETS = [
    ( 0.0000000, -0.0188956, 0.0069894),   # face 0  (-y)
    ( 0.0163643, -0.0094478, 0.0069894),   # face 1
    ( 0.0163643,  0.0094478, 0.0069894),   # face 2
    ( 0.0000000,  0.0188956, 0.0069894),   # face 3  (+y)
    (-0.0163643,  0.0094478, 0.0069894),   # face 4
    (-0.0163643, -0.0094478, 0.0069894),   # face 5
]

# --- verified spawn layout (Option A from the MuJoCo geometry fix) ---
# Line along y (a face-normal axis) so faces meet flush. Spacing 39.7912mm.
# Connecting faces are 0 and 3 (+/-y). z=0.0001 puts the base 0.1mm above
# the ground plane, exactly as in MuJoCo.
MODULE_SPACING = 0.0397912
SPAWN_POSITIONS = [
    (0.0, -MODULE_SPACING, 0.0001),   # module 0 = B (blue)
    (0.0,  0.0,            0.0001),   # module 1 = G (green)
    (0.0,  MODULE_SPACING, 0.0001),   # module 2 = R (red)
]
MODULE_NAMES = ["B", "G", "R"]

# --- physics options replicated from the MuJoCo <option>/<default> ---
# NOTE: the MuJoCo model runs at 0.1g on purpose (gravity="0 0 -0.981").
PHYSICS_DT = 0.002           # s
GRAVITY = (0.0, 0.0, -0.981)  # m/s^2  (0.1g — intentional, keep it)
MODULE_FRICTION = 0.4        # MuJoCo sliding friction (geom default)
GROUND_FRICTION = 0.2        # MuJoCo ground plane sliding friction
