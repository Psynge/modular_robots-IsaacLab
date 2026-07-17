"""Batched electromagnet force model — Isaac/torch port of the MuJoCo
MagneticForceModel (envs/magnetic_force.py in the MuJoCo repo).

Semantics ported verbatim (verified against the original source, not the
summary):
  - per-magnet polarity I in {-1, 0, +1}, per-magnet strength S in [0, 1]
  - global k = FORCE_CONSTANT * (M / 1000), pair k_pair = k * Sa * Sb
  - F = -k_use * Ia * Ib / X_use**2 ; opposite poles -> F > 0 -> attraction
  - X floored at MIN_DISTANCE
  - contact cap: X < CONTACT_DISTANCE -> k_use = min(k_pair, k_hold) AND
    X_use = CONTACT_DISTANCE (force evaluated at 4mm, not the true X)
  - wrench per body: force summed over sites, torque = sum r x F about COM

Differences from MuJoCo version (deliberate):
  - batched: every state/output tensor has a leading (num_envs,) dim
  - pure math: the caller supplies world site positions and COM positions and
    applies the returned wrenches; no simulator handles inside the model.
    World-frame in, world-frame out.

Shapes: E = num_envs, N = n_modules.
  polarity, strength : (E, N, 6)
  site_pos_w         : (E, N, 6, 3)
  com_pos_w          : (E, N, 3)
  forces, torques    : (E, N, 3)
"""

import torch

from common.magnet_constants import (
    CONTACT_DISTANCE, CUTOFF_DISTANCE, FORCE_CONSTANT, M_DEFAULT, M_HOLD,
    MIN_DISTANCE,
)

MODULE_LETTERS = {0: "B", 1: "G", 2: "R"}


class MagneticForceModel:
    def __init__(self, num_envs: int, n_modules: int, device: str):
        self.num_envs = num_envs
        self.n_modules = n_modules
        self.device = device

        self.polarity = torch.zeros((num_envs, n_modules, 6), device=device)
        self.strength = torch.ones((num_envs, n_modules, 6), device=device)

        self._k = FORCE_CONSTANT * (M_DEFAULT / 1000.0)
        self._k_hold = FORCE_CONSTANT * (M_HOLD / 1000.0)

        mods = torch.arange(n_modules, device=device)
        # (N, N) upper-triangular module-pair mask (a < b)
        self._pair_mask = mods[:, None] < mods[None, :]

    # ── global M control ──────────────────────────────────────────────
    def set_strength_pct(self, pct: float) -> None:
        """Set global M (1-100). k = FORCE_CONSTANT * (M/1000)."""
        self._k = FORCE_CONSTANT * (float(pct) / 1000.0)

    def get_strength_pct(self) -> float:
        return self._k / (FORCE_CONSTANT / 1000.0)

    # ── per-magnet control ────────────────────────────────────────────
    def set_magnet(self, module: int, face: int, polarity: float,
                   strength_pct: float = 100.0, env_ids=None) -> None:
        """Set one magnet. env_ids: tensor/slice of envs (default: all)."""
        if env_ids is None:
            env_ids = slice(None)
        self.polarity[env_ids, module, face] = max(-1.0, min(1.0, float(polarity)))
        self.strength[env_ids, module, face] = max(0.0, min(100.0, float(strength_pct))) / 100.0

    def clear_module(self, module: int, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.polarity[env_ids, module] = 0.0
        self.strength[env_ids, module] = 1.0

    def clear_all(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.polarity[env_ids] = 0.0
        self.strength[env_ids] = 1.0

    # ── core pairwise computation ─────────────────────────────────────
    def _pair_terms(self, site_pos_w: torch.Tensor):
        """Shared pairwise terms. Returns (diff, X, Fmag, active) with
        pair-axis shapes (E, N, 6, N, 6[, 3]); only a<b module pairs active."""
        pa = site_pos_w[:, :, :, None, None, :]
        pb = site_pos_w[:, None, None, :, :, :]
        diff = pb - pa                                   # (E,N,6,N,6,3)
        X = torch.linalg.vector_norm(diff, dim=-1)       # (E,N,6,N,6)

        Ia = self.polarity[:, :, :, None, None]
        Ib = self.polarity[:, None, None, :, :]
        Sa = self.strength[:, :, :, None, None]
        Sb = self.strength[:, None, None, :, :]

        mask = self._pair_mask[None, :, None, :, None]   # a<b modules only
        active = mask & (Ia != 0.0) & (Ib != 0.0) & (X <= CUTOFF_DISTANCE)

        Xc = X.clamp_min(MIN_DISTANCE)
        k_pair = self._k * Sa * Sb
        in_contact = Xc < CONTACT_DISTANCE
        k_use = torch.where(
            in_contact, torch.clamp(k_pair, max=self._k_hold), k_pair
        )
        X_use = torch.where(in_contact, torch.full_like(Xc, CONTACT_DISTANCE), Xc)

        Fmag = -k_use * Ia * Ib / (X_use ** 2)
        Fmag = torch.where(active, Fmag, torch.zeros_like(Fmag))
        return diff, Xc, Fmag, active

    def compute(self, site_pos_w: torch.Tensor, com_pos_w: torch.Tensor):
        """World-frame wrenches about each module's COM.
        Returns forces (E, N, 3), torques (E, N, 3)."""
        diff, Xc, Fmag, _ = self._pair_terms(site_pos_w)
        dirn = diff / Xc.unsqueeze(-1)
        F_ab = Fmag.unsqueeze(-1) * dirn                 # force ON a's site, toward b if +

        # per-site accumulation (Newton's third law: b gets -F_ab)
        site_force = F_ab.sum(dim=(3, 4)) + (-F_ab).sum(dim=(1, 2))   # (E,N,6,3)

        forces = site_force.sum(dim=2)                                # (E,N,3)
        r = site_pos_w - com_pos_w[:, :, None, :]                     # (E,N,6,3)
        torques = torch.cross(r, site_force, dim=-1).sum(dim=2)       # (E,N,3)
        return forces, torques

    # ── batched diagnostics (geometry-only unless noted) ──────────────
    def min_face_pair_distance(self, site_pos_w: torch.Tensor) -> torch.Tensor:
        """(E,) smallest cross-module site-to-site distance."""
        pa = site_pos_w[:, :, :, None, None, :]
        pb = site_pos_w[:, None, None, :, :, :]
        X = torch.linalg.vector_norm(pb - pa, dim=-1)
        big = torch.full_like(X, float("inf"))
        X = torch.where(self._pair_mask[None, :, None, :, None], X, big)
        return X.amin(dim=(1, 2, 3, 4))

    def any_faces_within(self, site_pos_w: torch.Tensor,
                         cutoff: float = CUTOFF_DISTANCE) -> torch.Tensor:
        """(E,) bool — any cross-module site pair within cutoff."""
        return self.min_face_pair_distance(site_pos_w) <= cutoff

    def modules_in_range(self, site_pos_w: torch.Tensor,
                         cutoff: float = CUTOFF_DISTANCE) -> torch.Tensor:
        """(E, N) bool — module has any site within cutoff of another module's."""
        pa = site_pos_w[:, :, :, None, None, :]
        pb = site_pos_w[:, None, None, :, :, :]
        X = torch.linalg.vector_norm(pb - pa, dim=-1)
        cross = self._pair_mask | self._pair_mask.T                   # a != b
        near = (X <= cutoff) & cross[None, :, None, :, None]
        return near.any(dim=(3, 4)).any(dim=2)

    def get_active_pairs(self, site_pos_w: torch.Tensor, env: int = 0):
        """List of (mod_a, face_a, mod_b, face_b, X, F) for one env — the
        MuJoCo diagnostic, for probes and watch scripts."""
        diff, _Xc, Fmag, active = self._pair_terms(site_pos_w[env:env + 1])
        X = torch.linalg.vector_norm(diff, dim=-1)[0]
        out = []
        idx = active[0].nonzero(as_tuple=False)
        for a, fa, b, fb in idx.tolist():
            out.append((a, fa, b, fb, float(X[a, fa, b, fb]),
                        float(Fmag[0, a, fa, b, fb])))
        return out

    def get_connections(self, site_pos_w: torch.Tensor, env: int = 0):
        """Connection strings for attracting pairs within CONTACT_DISTANCE,
        e.g. ['B.3-G.0'] — same canonical form as MuJoCo."""
        result = []
        for a, fa, b, fb, X, F in self.get_active_pairs(site_pos_w, env):
            if X < CONTACT_DISTANCE and F > 0:
                result.append(f"{MODULE_LETTERS[a]}.{fa}-{MODULE_LETTERS[b]}.{fb}")
        return result
