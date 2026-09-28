#!/usr/bin/env python
"""Build Ace-(Ala)10-Nme as an alpha helix, minimize, equilibrate 500 ps at 300 K.

Writes results/build/{deca.prmtop, deca.inpcrd, helix.pdb, equil_state.xml} and
the three shared lineage nodes (build -> minimize -> equil). ``--bench`` also
prints CPU ns/day at 1 and 4 threads.
"""
import argparse
import subprocess
import time

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u

from common import BUILD, Lineage, Recorder, cv_atoms, cvs, langevin, load_prmtop, log, make_system, positions_A, save_state

LEAP = """source leaprc.protein.ff14SB
m = sequence {{ ACE {alas} NME }}
impose m {{ {resids} }} {{ {{ N CA C N -47.0 }} {{ C N CA C -57.0 }} }}
saveamberparm m deca.prmtop deca.inpcrd
savepdb m helix_leap.pdb
quit
"""


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    script = LEAP.format(alas=" ".join(["ALA"] * 10), resids=" ".join(str(i) for i in range(2, 12)))
    (BUILD / "leap.in").write_text(script)
    subprocess.run(["tleap", "-f", "leap.in"], cwd=BUILD, check=True, capture_output=True)


def bench(prmtop, pos):
    for threads in ("1", "4"):
        system = make_system(prmtop)
        ctx = openmm.Context(system, langevin(), openmm.Platform.getPlatformByName("CPU"), {"Threads": threads})
        ctx.setPositions(pos)
        ctx.getIntegrator().step(500)
        t0 = time.time(); ctx.getIntegrator().step(10000); dt = time.time() - t0
        log("bench", f"threads={threads}: {10000*0.002/1000/dt*86400:.0f} ns/day")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", action="store_true"); a = ap.parse_args()
    build()
    prmtop = load_prmtop()
    inpcrd = app.AmberInpcrdFile(str(BUILD / "deca.inpcrd"))
    d_pair, hb = cv_atoms(prmtop.topology)
    lin = Lineage("shared")
    lin.node("build", "build", "tleap: Ace-(Ala)10-Nme helix", n_atoms=prmtop.topology.getNumAtoms())

    system = make_system(prmtop)
    sim = app.Simulation(prmtop.topology, system, langevin(1), openmm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(inpcrd.positions)
    sim.minimizeEnergy()
    lin.node("minimize", "minimize", "energy minimization", ["build"])
    with open(BUILD / "helix.pdb", "w") as fh:
        app.PDBFile.writeFile(prmtop.topology, sim.context.getState(getPositions=True).getPositions(), fh)

    sim.context.setVelocitiesToTemperature(300 * u.kelvin, 1)
    rec = Recorder(d_pair, hb)
    for i in range(250):                       # 500 ps, frame every 2 ps
        sim.step(1000)
        rec.add(positions_A(sim.context), (i + 1) * 2.0)
    d, n = rec.save(BUILD / "equil.npz")
    save_state(sim.context, BUILD / "equil_state.xml")
    lin.node("equil", "equil", "300 K equilibration, 500 ps", ["minimize"], t_ps=500, traj="build/equil.npz")
    log("build", f"equil d_ee {d[-50:].mean():.1f} A, n_hb {n[-50:].mean():.1f}")
    if a.bench:
        bench(prmtop, sim.context.getState(getPositions=True).getPositions())


if __name__ == "__main__":
    main()
