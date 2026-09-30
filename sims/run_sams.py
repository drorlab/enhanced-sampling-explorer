#!/usr/bin/env python
"""SAMS (self-adjusting mixture sampling, openmmtools SAMSSampler) over umbrella
states along d_ee: one walker that jumps between restraint centres while the
sampler learns each state's log-partition function (logZ) online. Two stages:
fast burn-in updates, then the asymptotically optimal 1/t schedule.

--mode temperature swaps the umbrella ladder for a temperature ladder
(simulated tempering). With --init-ps > 0, the initial weights come from a short
unbiased run at each temperature: f_k = f_0 + integral of <U> d(beta) (thermodynamic
integration), passed to SAMS as logZ_guess. Without it every state starts at
logZ = 0, which for temperatures is off by ~75 kT and the walker stays hot.
"""
import argparse
import copy

import numpy as np
import openmm
import openmm.unit as u
from openmmtools import cache, mcmc, multistate, states

from common import BUILD, RES, Lineage, cv_atoms, cvs, load_prmtop, load_state, log, make_system, platform
from pulling import KCAL_A2

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=15000); ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--k", type=float, default=2.0); ap.add_argument("--mode", default="umbrella")
ap.add_argument("--frame-every", type=int, default=5); ap.add_argument("--name", default="sams")
ap.add_argument("--init-ps", type=float, default=0.0, help="temperature mode: ps of sampling per temperature for initial weights")
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
OUT = RES / a.name; OUT.mkdir(parents=True, exist_ok=True)
CENTERS = np.round(np.arange(4.0, 35.01, 0.75), 2)


class Umbrella(states.GlobalParameterState):
    r0 = states.GlobalParameterState.GlobalParameter("r0", standard_value=2.4)


cache.global_context_cache.platform = platform()
prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
system = make_system(prmtop)
st = load_state(BUILD / "equil_state.xml")
if a.mode == "umbrella":
    f = openmm.CustomBondForce(f"0.5*{a.k * KCAL_A2}*(r-r0)^2")
    f.addGlobalParameter("r0", 2.4); f.addBond(*d_pair, [])
    system.addForce(f)
    base = states.CompoundThermodynamicState(states.ThermodynamicState(system, 300 * u.kelvin),
                                             composable_states=[Umbrella(r0=2.4)])
    tstates = []
    for c in CENTERS:
        s = copy.deepcopy(base); s.r0 = float(c) / 10; tstates.append(s)
    labels = CENTERS
else:
    labels = np.geomspace(300, 700, 12)
    tstates = [states.ThermodynamicState(system, T * u.kelvin) for T in labels]

def short_run(args):
    """Unbiased Langevin at temperature T from the equilibrated helix: 20 ps settle, then init_ps of <U> samples."""
    T, seed = args
    ig = openmm.LangevinMiddleIntegrator(T * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds)
    ig.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, ig, platform())
    ctx.setPositions(st.getPositions()); ctx.setVelocitiesToTemperature(T * u.kelvin, seed)
    ig.step(10000)
    U = []
    for _ in range(int(a.init_ps / 0.2)):
        ig.step(100)
        U.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
    return np.array(U)


logZ_guess, init = None, {}
if a.mode != "umbrella" and a.init_ps > 0:
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        Us = pool.map(short_run, [(float(T), 900 + i) for i, T in enumerate(labels)])
    beta = 1 / (0.0083144626 * labels)                      # mol/kJ
    meanU = np.array([x.mean() for x in Us])
    f = np.concatenate([[0], np.cumsum(0.5 * (meanU[1:] + meanU[:-1]) * np.diff(beta))])   # dimensionless
    logZ_guess = -f
    init = dict(init_meanU=meanU, init_sdU=np.array([x.std() for x in Us]), init_f=f, init_ps=a.init_ps + 20.0)
    log(a.name, "initial f_k (kT): " + " ".join(f"{x:.1f}" for x in f))

sstate = states.SamplerState(st.getPositions(), velocities=st.getVelocities())
move = mcmc.LangevinDynamicsMove(timestep=2 * u.femtoseconds, collision_rate=1 / u.picosecond,
                                 n_steps=a.steps, reassign_velocities=False)
sampler = multistate.SAMSSampler(mcmc_moves=move, number_of_iterations=a.iters, state_update_scheme="global-jump",
                                 update_stages="two-stage", weight_update_method="rao-blackwellized",
                                 gamma0=1.0, flatness_threshold=0.2, logZ_guess=logZ_guess)
nc = OUT / "sams.nc"
for p in (nc, OUT / "sams_checkpoint.nc"):
    p.unlink(missing_ok=True)
sampler.create(tstates, [sstate], multistate.MultiStateReporter(str(nc), checkpoint_interval=1000))

state_t, logZ_t, stage_t, d_t, u_t, frames, frame_it = [], [], [], [], [], [], []


def save():
    np.savez_compressed(OUT / "sams.npz", mode=a.mode, labels=labels, k=a.k, state=np.array(state_t),
                        logZ=np.array(logZ_t), stage=np.array(stage_t), d=np.array(d_t), u=np.array(u_t),
                        frames=np.array(frames, dtype=np.float32), frame_it=np.array(frame_it),
                        ps_per_iter=a.steps * 0.002, **init)


stage1_end = None
for it in range(a.iters):
    sampler.run(1)
    state_t.append(int(sampler._replica_thermodynamic_states[0]))
    logZ_t.append(np.array(sampler._logZ).copy())
    stage_t.append(int(sampler._stage))
    u_t.append(np.array(sampler._energy_thermodynamic_states[0]).copy())
    pos = sampler._sampler_states[0].positions.value_in_unit(u.angstrom).astype(np.float32)
    d_t.append(float(cvs(pos, d_pair, hb)[0]))
    if stage1_end is None and stage_t[-1] == 1:
        stage1_end = it
    if it % a.frame_every == 0:
        frames.append(pos); frame_it.append(it)
    if (it + 1) % 500 == 0:
        save(); log(a.name, f"iter {it+1} stage {stage_t[-1]} state {state_t[-1]} d {d_t[-1]:.1f} "
                           f"visited {len(set(state_t))}/{len(labels)}")
save()
lin = Lineage(a.name); ps = a.steps * 0.002
s1 = stage1_end if stage1_end is not None else a.iters
first = ["equil"]
if init:
    lin.node(f"{a.name}_init", "cmd", f"initial weights: {len(labels)} × {init['init_ps']:g} ps unbiased runs", ["equil"],
             t_ps=len(labels) * init["init_ps"])
    first = [f"{a.name}_init"]
lin.node(f"{a.name}_stage1", "stage1", f"SAMS stage 1 (burn-in), {s1 * ps / 1000:.1f} ns", first, t_ps=s1 * ps,
         traj=f"{a.name}/sams.npz", iter_range=[0, s1])
if stage1_end is not None:
    lin.node(f"{a.name}_stage2", "stage2", f"SAMS stage 2 (1/t updates), {(a.iters - s1) * ps / 1000:.1f} ns",
             [f"{a.name}_stage1"], t_ps=(a.iters - s1) * ps, traj=f"{a.name}/sams.npz", iter_range=[s1, a.iters])
log(a.name, "done")
