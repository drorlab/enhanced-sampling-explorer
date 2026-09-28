#!/usr/bin/env python
"""Steered MD + Jarzynski: seed sampler -> N seeds -> N forward pulls, at two speeds.

Park & Schulten (2003) setup: k = 7.2 kcal/mol/A^2, pulled from the start
d_ee to 34 A. The slow set (10 A/ns) is the one used for the free energy; the
fast set (100 A/ns) is there to show how dissipation biases the estimate.
"""
import argparse
from multiprocessing import Pool

import numpy as np
import openmm
import openmm.app as app

from common import BUILD, RES, Lineage, cv_atoms, langevin, load_prmtop, load_state, log, make_system, platform, platform_props, positions_A, save_state
from pulling import pull

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=32); ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--end", type=float, default=34.0)
a = ap.parse_args()
OUT = RES / "smd"; OUT.mkdir(parents=True, exist_ok=True)
SPEEDS = {"slow": 10.0, "fast": 100.0}


def job(args):
    tag, i, speed = args
    st = load_state(OUT / f"seed_{i:02d}.xml")
    prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
    from common import cvs
    p0 = st.getPositions(asNumpy=True)._value * 10
    d0 = float(cvs(p0, d_pair, hb)[0])
    r = pull(st, d0, a.end, speed, seed=1000 * (tag == "fast") + i + 1)
    r.pop("state")
    np.savez_compressed(OUT / f"pull_{tag}_{i:02d}.npz", **r, speed=speed)
    return tag, i, float(r["W"][-1])


if __name__ == "__main__":
    prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
    lin = Lineage("smd")
    # seed sampler: unbiased 300 K run, one seed every 50 ps
    sim = app.Simulation(prmtop.topology, make_system(prmtop), langevin(41), platform(), platform_props())
    sim.context.setState(load_state(BUILD / "equil_state.xml"))
    for i in range(a.n):
        sim.step(25000)
        save_state(sim.context, OUT / f"seed_{i:02d}.xml")
    lin.node("smd_seeds", "sampler", f"unbiased seed sampler, {a.n * 50} ps; seed every 50 ps", ["equil"], t_ps=a.n * 50)
    log("smd", "seeds done")
    jobs = [(tag, i, s) for tag, s in SPEEDS.items() for i in range(a.n)]
    with Pool(a.workers) as pool:
        for tag, i, w in pool.imap_unordered(job, jobs):
            lin.node(f"smd_{tag}_{i:02d}", f"pull_{tag}", f"{tag} pull #{i} ({SPEEDS[tag]:g} A/ns)", ["smd_seeds"],
                     t_ps=(a.end - 16) / SPEEDS[tag] * 1000, traj=f"smd/pull_{tag}_{i:02d}.npz", W=w, seed=i)
            log("smd", f"{tag} {i} W={w:.1f}")
    log("smd", "done")
