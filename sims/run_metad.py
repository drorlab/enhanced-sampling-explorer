#!/usr/bin/env python
"""Well-tempered metadynamics on d_ee (openmm.app.Metadynamics).

Records every 1 ps the CV and the bias at the walker (-> Gaussian height
h*exp(-V/kdT)), frames every 5 ps, and the full free-energy grid every 250 ps
for the convergence plot.
"""
import argparse

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u

from common import BUILD, RES, TEMP, Lineage, Recorder, add_upper_wall, cv_atoms, cvs, dee_force, langevin, load_prmtop, load_state, log, make_system, platform, platform_props, positions_A, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--ns", type=float, default=20); ap.add_argument("--gamma", type=float, default=10)
ap.add_argument("--height", type=float, default=1.2)  # kJ/mol
a = ap.parse_args()

prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
system = make_system(prmtop); add_upper_wall(system, d_pair)
cv = openmm.CustomCVForce("d"); cv.addCollectiveVariable("d", dee_force(d_pair))
LO, HI, NG = 0.3, 3.6, 331
bv = app.BiasVariable(cv, LO, HI, 0.05, False, gridWidth=NG)
meta = app.Metadynamics(system, [bv], TEMP, a.gamma, a.height * u.kilojoules_per_mole, 500)
sim = app.Simulation(prmtop.topology, system, langevin(21), platform(), platform_props())
sim.context.setState(load_state(BUILD / "equil_state.xml"))

grid_A = np.linspace(LO, HI, NG) * 10
rec = Recorder(d_pair, hb)
cv_t, cv_d, cv_bias, fes, fes_t = [], [], [], [], []
steps_ps = 500
n_ps = int(a.ns * 1000)
for ps in range(1, n_ps + 1):
    meta.step(sim, steps_ps)
    pos = positions_A(sim.context)
    d, _ = cvs(pos, d_pair, hb)
    fe = np.asarray(meta.getFreeEnergy().value_in_unit(u.kilojoules_per_mole))
    bias = -fe * (a.gamma - 1) / a.gamma                   # V(s) = -(dT/(T+dT)) F(s)
    cv_t.append(ps); cv_d.append(float(d)); cv_bias.append(float(np.interp(d, grid_A, bias)))
    if ps % 5 == 0:
        rec.add(pos, ps)
    if ps % 250 == 0:
        fes.append((fe - fe.min()) / 4.184); fes_t.append(ps)
        rec.save(RES / "metad" / "traj.npz")
        np.savez(RES / "metad" / "metad.npz", grid_A=grid_A, fes=np.array(fes), fes_t_ps=np.array(fes_t),
                 cv_t_ps=np.array(cv_t), cv_d=np.array(cv_d), bias_at_walker=np.array(cv_bias),
                 gamma=a.gamma, height=a.height)
        log("metad", f"{ps/1000:.2f} ns  d={float(d):.1f}")
Lineage("metad").node("metad", "prod", f"WT-metaD on d_ee, gamma={a.gamma:g}, {a.ns:g} ns", ["equil"],
                      t_ps=n_ps, traj="metad/traj.npz")
log("metad", "done")
