#!/usr/bin/env python
"""Dual-boost Gaussian accelerated MD (GaMD) with a hand-written CustomIntegrator.

Stages: cMD (collect Vmax/Vmin/Vavg/sigma of total and dihedral energy) ->
GaMD equilibration (boost parameters re-derived every 200 ps from running
stats) -> GaMD production with fixed parameters. Every frame records the boost
dV = dV_P + dV_D so analysis can reweight (exp average vs 2nd-order cumulant).

Force groups: 1 = PeriodicTorsionForce (dihedral boost), 0 = everything else.
"""
import argparse
import math

import numpy as np
import openmm
import openmm.app as app

from common import BUILD, RES, Lineage, Recorder, cv_atoms, load_prmtop, load_state, log, make_system, platform, platform_props, positions_A

ap = argparse.ArgumentParser()
ap.add_argument("--cmd-ns", type=float, default=2); ap.add_argument("--eq-ns", type=float, default=4)
ap.add_argument("--prod-ns", type=float, default=14); ap.add_argument("--sigma0", type=float, default=6.0)  # kcal/mol
a = ap.parse_args()
SIG0 = a.sigma0 * 4.184
UPPER = False
KT = 0.0083144626 * 300
DT = 0.002


def integrator(seed):
    ig = openmm.CustomIntegrator(DT)
    g = 1.0
    ig.addGlobalVariable("a", math.exp(-g * DT / 2)); ig.addGlobalVariable("b", math.sqrt(1 - math.exp(-g * DT)))
    ig.addGlobalVariable("kT", KT)
    for name, val in (("EP", 0), ("kP", 0), ("ED", 0), ("kD", 0), ("sP", 1), ("sD", 1), ("V0", 0), ("VD", 0)):
        ig.addGlobalVariable(name, val)
    ig.addPerDofVariable("x1", 0)

    def scales():
        ig.addComputeGlobal("V0", "energy0")
        ig.addComputeGlobal("VD", "energy1")
        ig.addComputeGlobal("sP", "1 - kP*max(EP-(V0+VD), 0)")
        ig.addComputeGlobal("sD", "1 - kD*max(ED-VD, 0)")

    ig.addUpdateContextState()
    ig.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian"); ig.addConstrainVelocities()
    scales()
    ig.addComputePerDof("v", "v + 0.5*dt*sP*f0/m")
    ig.addComputePerDof("v", "v + 0.5*dt*(sP+sD-1)*f1/m")
    ig.addComputePerDof("x", "x + dt*v")
    ig.addComputePerDof("x1", "x"); ig.addConstrainPositions()
    scales()
    ig.addComputePerDof("v", "v + 0.5*dt*sP*f0/m + (x-x1)/dt")
    ig.addComputePerDof("v", "v + 0.5*dt*(sP+sD-1)*f1/m"); ig.addConstrainVelocities()
    ig.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian"); ig.addConstrainVelocities()
    ig.setRandomNumberSeed(seed)
    return ig


def params(V):
    """GaMD boost parameters (E, k) from energy samples V (kJ/mol)."""
    vmax, vmin, vavg, sig = V.max(), V.min(), V.mean(), V.std()
    # upper bound (iE=2) runs away in vacuum: the boosted run samples higher V,
    # which raises E on the next update. Lower bound (E = Vmax) is stable.
    k0 = (1 - SIG0 / sig) * (vmax - vmin) / (vavg - vmin) if (sig > 0 and UPPER) else 0
    if 0 < k0 <= 1:
        E = vmin + (vmax - vmin) / k0; bound = "upper"
    else:
        k0 = min(1.0, SIG0 / sig * (vmax - vmin) / (vmax - vavg)); E = vmax; bound = "lower"
    return dict(E=float(E), k=float(k0 / (vmax - vmin)), k0=float(k0), bound=bound,
                vmax=float(vmax), vmin=float(vmin), vavg=float(vavg), sigma=float(sig))


def energies(ctx):
    vd = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value
    v0 = ctx.getState(getEnergy=True, groups={0}).getPotentialEnergy()._value
    return v0 + vd, vd


def boost(VP, VD, p):
    dp = 0.5 * p["P"]["k"] * (p["P"]["E"] - VP) ** 2 if VP < p["P"]["E"] else 0.0
    dd = 0.5 * p["D"]["k"] * (p["D"]["E"] - VD) ** 2 if VD < p["D"]["E"] else 0.0
    return dp + dd


def set_params(ig, p):
    ig.setGlobalVariableByName("EP", p["P"]["E"]); ig.setGlobalVariableByName("kP", p["P"]["k"])
    ig.setGlobalVariableByName("ED", p["D"]["E"]); ig.setGlobalVariableByName("kD", p["D"]["k"])


prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
system = make_system(prmtop, dihedral_group=True)
ig = integrator(31)
ctx = openmm.Context(system, ig, platform(), platform_props())
ctx.setState(load_state(BUILD / "equil_state.xml"))
out = RES / "gamd"; lin = Lineage("gamd")
history = []   # boost parameters over time, for the dashboard

# ---- stage 1: cMD statistics (boost off) -----------------------------------
VP, VD = [], []
rec = Recorder(d_pair, hb)
n = int(a.cmd_ns * 1000 / 0.1)
for i in range(n):
    ig.step(50)
    vp, vd = energies(ctx); VP.append(vp); VD.append(vd)
    if (i + 1) % 50 == 0:
        rec.add(positions_A(ctx), (i + 1) * 0.1, dV=0.0, VP=vp, VD=vd)
rec.save(out / "cmd.npz")
p = {"P": params(np.array(VP)), "D": params(np.array(VD))}
history.append(dict(t_ps=a.cmd_ns * 1000, stage="cmd", **{f"{k}_{x}": v for k in "PD" for x, v in p[k].items()}))
lin.node("gamd_cmd", "cmd", f"cMD {a.cmd_ns:g} ns: collect V stats", ["equil"], t_ps=a.cmd_ns * 1000, traj="gamd/cmd.npz",
         **{k: p[k] for k in p})
log("gamd", f"cMD params {p}")

# ---- stage 2: GaMD equilibration, parameters updated every 200 ps ----------
set_params(ig, p)
rec = Recorder(d_pair, hb)
n = int(a.eq_ns * 1000 / 0.1)
for i in range(n):
    ig.step(50)
    vp, vd = energies(ctx); VP.append(vp); VD.append(vd)
    if (i + 1) % 50 == 0:
        rec.add(positions_A(ctx), a.cmd_ns * 1000 + (i + 1) * 0.1, dV=boost(vp, vd, p), VP=vp, VD=vd)
    if (i + 1) % 2000 == 0:
        p = {"P": params(np.array(VP)), "D": params(np.array(VD))}; set_params(ig, p)
        history.append(dict(t_ps=a.cmd_ns * 1000 + (i + 1) * 0.1, stage="equil",
                            **{f"{k}_{x}": v for k in "PD" for x, v in p[k].items()}))
rec.save(out / "equil.npz")
lin.node("gamd_equil", "equil", f"GaMD equilibration {a.eq_ns:g} ns (params adapt)", ["gamd_cmd"],
         t_ps=a.eq_ns * 1000, traj="gamd/equil.npz", **{k: p[k] for k in p})
log("gamd", f"equil params {p}")

# ---- stage 3: production, fixed parameters ---------------------------------
rec = Recorder(d_pair, hb)
t0 = (a.cmd_ns + a.eq_ns) * 1000
n = int(a.prod_ns * 1000 / 2)
for i in range(n):
    ig.step(1000)
    vp, vd = energies(ctx)
    rec.add(positions_A(ctx), t0 + (i + 1) * 2.0, dV=boost(vp, vd, p), VP=vp, VD=vd)
    if (i + 1) % 500 == 0:
        rec.save(out / "prod.npz"); log("gamd", f"prod {(i+1)*2/1000:.1f} ns")
rec.save(out / "prod.npz", params=str(p))
np.save(out / "history.npy", np.array(history, dtype=object), allow_pickle=True)
import json; (out / "params.json").write_text(json.dumps(dict(final=p, history=history)))
lin.node("gamd_prod", "prod", f"GaMD production {a.prod_ns:g} ns", ["gamd_equil"], t_ps=a.prod_ns * 1000,
         traj="gamd/prod.npz", **{k: p[k] for k in p})
log("gamd", "done")
