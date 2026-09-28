"""Moving / fixed harmonic restraints on d_ee, shared by SMD, the umbrella reference
and SAMS. Energies in kJ/mol and nm inside OpenMM; everything returned is in
Angstrom and kcal/mol."""
import numpy as np
import openmm

from common import cv_atoms, cvs, langevin, load_prmtop, make_system, platform, platform_props

KCAL_A2 = 4.184 * 100  # kcal/mol/A^2 -> kJ/mol/nm^2


def restrained_context(k_kcal_A2, r0_A, seed):
    prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
    system = make_system(prmtop)
    f = openmm.CustomBondForce("0.5*k_umb*(r-r0)^2")
    f.addGlobalParameter("k_umb", k_kcal_A2 * KCAL_A2); f.addGlobalParameter("r0", r0_A / 10)
    f.addBond(*d_pair, []); f.setForceGroup(3)
    system.addForce(f)
    ctx = openmm.Context(system, langevin(seed), platform(), platform_props())
    return ctx, d_pair, hb


def pos_A(ctx):
    return ctx.getState(getPositions=True).getPositions(asNumpy=True)._value.astype(np.float32) * 10


def pull(state, r_start, r_end, speed_A_per_ns, k=7.2, seed=0, frame_every_A=0.25, update_steps=10):
    """Steered MD: move r0 linearly; accumulate work W = sum dU/dr0 * dr0 at fixed x.

    Returns dict with r0, d, W (kcal/mol) sampled every ``frame_every_A`` and frames.
    """
    ctx, d_pair, hb = restrained_context(k, r_start, seed)
    ctx.setState(state)
    ctx.setParameter("r0", r_start / 10); ctx.setParameter("k_umb", k * KCAL_A2)
    n_updates = int(round(abs(r_end - r_start) / speed_A_per_ns * 1000 / 0.002 / update_steps))
    dr = (r_end - r_start) / n_updates
    every = max(1, int(round(frame_every_A / abs(dr))))
    W, r0 = 0.0, r_start
    out = dict(r0=[r_start], d=[], W=[0.0], frames=[])
    p = pos_A(ctx); d, _ = cvs(p, d_pair, hb); out["d"].append(float(d)); out["frames"].append(p)
    ig = ctx.getIntegrator()
    for i in range(n_updates):
        ig.step(update_steps)
        p = pos_A(ctx); d, _ = cvs(p, d_pair, hb)
        W += k * ((d - (r0 + dr)) ** 2 - (d - r0) ** 2) / 2      # U(r0+dr) - U(r0) at fixed x
        r0 += dr; ctx.setParameter("r0", r0 / 10)
        if (i + 1) % every == 0:
            out["r0"].append(r0); out["d"].append(float(d)); out["W"].append(float(W)); out["frames"].append(p)
    out = {k_: np.array(v) for k_, v in out.items()}
    out["state"] = ctx.getState(getPositions=True, getVelocities=True)
    return out
