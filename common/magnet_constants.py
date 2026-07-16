"""Electromagnet model constants — ported unchanged from envs/magnetic_force.py
in the MuJoCo repo. Force law:

    F = -(M_k * Sa * Sb) * Ia * Ib / X^2      (opposite poles attract)

with k = FORCE_CONSTANT * M / 1000, applied site-to-site within
CUTOFF_DISTANCE, contact-capped at M_HOLD once latched.

These are the values to re-probe in PhysX (step 2/3) — the MuJoCo-calibrated
scale may behave differently under a different contact solver. Change only
with a probe result in hand.
"""

FORCE_CONSTANT = 1.332e-4
M_DEFAULT = 55.0
M_HOLD = 25.0
M_MAX = 100.0

CUTOFF_DISTANCE = 0.030      # m — site-to-site magnetic interaction range
CONTACT_DISTANCE = 0.004     # m — latch threshold

# Seed hold for the y-axis line (re-derived after the geometry fix):
# B3N50 / G0S50 / G3N50 / R0S50
SEED_HOLD = [
    # (module, face, polarity, strength_pct)
    (0, 3, +1, 50.0),
    (1, 0, -1, 50.0),
    (1, 3, +1, 50.0),
    (2, 0, -1, 50.0),
]
