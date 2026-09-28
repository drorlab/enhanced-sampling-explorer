#!/usr/bin/env python
"""Plain MD baseline: equil -> one unbiased trajectory."""
import argparse

import openmm.app as app

from common import BUILD, RES, Lineage, Recorder, cv_atoms, langevin, load_prmtop, load_state, log, make_system, platform, platform_props, positions_A

ap = argparse.ArgumentParser(); ap.add_argument("--ns", type=float, default=20); ap.add_argument("--frame-ps", type=float, default=5)
a = ap.parse_args()
prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
sim = app.Simulation(prmtop.topology, make_system(prmtop), langevin(11), platform(), platform_props())
sim.context.setState(load_state(BUILD / "equil_state.xml"))
rec = Recorder(d_pair, hb)
chunk = int(a.frame_ps / 0.002); n = int(a.ns * 1000 / a.frame_ps)
for i in range(n):
    sim.step(chunk)
    rec.add(positions_A(sim.context), (i + 1) * a.frame_ps)
    if (i + 1) % 400 == 0:
        rec.save(RES / "md" / "traj.npz"); log("md", f"{(i+1)*a.frame_ps/1000:.1f} ns")
rec.save(RES / "md" / "traj.npz")
Lineage("md").node("md", "prod", f"unbiased MD, {a.ns:g} ns", ["equil"], t_ps=a.ns * 1000, traj="md/traj.npz")
log("md", "done")
