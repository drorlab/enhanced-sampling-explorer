#!/usr/bin/env python
"""Temperature replica exchange (openmmtools ParallelTemperingSampler).

Runs one iteration at a time so we can record, per iteration, which state
(temperature) each replica is in, the full replica x state reduced-potential
matrix (for MBAR), and periodically every replica's coordinates.
"""
import argparse

import numpy as np
import openmm.unit as u
from openmmtools import cache, mcmc, multistate, states

from common import BUILD, RES, Lineage, cv_atoms, cvs, load_prmtop, load_state, log, make_system, platform

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=2500); ap.add_argument("--n-temps", type=int, default=10)
ap.add_argument("--tmax", type=float, default=700.0); ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--frame-every", type=int, default=5)
a = ap.parse_args()
OUT = RES / "remd"; OUT.mkdir(parents=True, exist_ok=True)

cache.global_context_cache.platform = platform()
prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
system = make_system(prmtop)
st = load_state(BUILD / "equil_state.xml")
thermo = states.ThermodynamicState(system, 300 * u.kelvin)
sstate = states.SamplerState(st.getPositions(), velocities=st.getVelocities())
move = mcmc.LangevinDynamicsMove(timestep=2 * u.femtoseconds, collision_rate=1 / u.picosecond,
                                 n_steps=a.steps, reassign_velocities=False)
sampler = multistate.ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=a.iters)
nc = OUT / "remd.nc"
for f in (nc, OUT / "remd_checkpoint.nc"):
    f.unlink(missing_ok=True)
reporter = multistate.MultiStateReporter(str(nc), checkpoint_interval=1000)
sampler.create(thermo, sstate, reporter, min_temperature=300 * u.kelvin, max_temperature=a.tmax * u.kelvin,
               n_temperatures=a.n_temps)
temps = np.array([s.temperature.value_in_unit(u.kelvin) for s in sampler._thermodynamic_states])
R = a.n_temps
rep_state, u_kn, d_rep, frames, frame_it = [], [], [], [], []


def save():
    np.savez_compressed(OUT / "remd.npz", temps=temps, rep_state=np.array(rep_state), u_rs=np.array(u_kn),
                        d_rep=np.array(d_rep), frames=np.array(frames, dtype=np.float32), frame_it=np.array(frame_it),
                        n_acc=sampler._n_accepted_matrix, n_prop=sampler._n_proposed_matrix,
                        ps_per_iter=a.steps * 0.002)


lin = Lineage("remd")
for it in range(a.iters):
    sampler.run(1)
    rep_state.append(np.array(sampler._replica_thermodynamic_states).copy())
    u_kn.append(np.array(sampler._energy_thermodynamic_states).copy())      # [replica, state]
    pos = np.array([s.positions.value_in_unit(u.angstrom) for s in sampler._sampler_states], dtype=np.float32)
    d_rep.append(cvs(pos, d_pair, hb)[0])
    if it % a.frame_every == 0:
        frames.append(pos); frame_it.append(it)
    if (it + 1) % 250 == 0:
        save()
        acc = sampler._n_accepted_matrix.sum() / max(1, sampler._n_proposed_matrix.sum())
        log("remd", f"iter {it+1}  acceptance {acc:.2f}  d@300K {d_rep[-1][list(rep_state[-1]).index(0)]:.1f}")
save()
for r in range(R):
    lin.node(f"remd_rep{r:02d}", "replica", f"replica {r} ({a.iters * a.steps * 0.002 / 1000:g} ns)", ["equil"],
             t_ps=a.iters * a.steps * 0.002, traj="remd/remd.npz", replica=r)
log("remd", f"done; temps {np.round(temps, 1).tolist()}")
