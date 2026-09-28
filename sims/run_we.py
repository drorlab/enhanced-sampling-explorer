#!/usr/bin/env python
"""Weighted ensemble (equilibrium mode, Huber-Kim split/merge) on d_ee bins.

Each iteration: every walker runs tau = 10 ps of plain Langevin MD (fresh
Context per segment so each child of a split gets its own random seed), then
walkers are resampled to ``--per-bin`` per occupied 1-A bin. Weights are
conserved exactly; FE(bin) = -kT ln P(bin).

Output per iteration i: results/we/iter_{i:04d}.npz with walker ids, parent
ids, weights, bins, d_ee per ps along the segment and the end frame. Merges
are recorded as extra parents (``merged_from``) so the tree shows them.
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import openmm

from common import BUILD, RES, Lineage, cv_atoms, cvs, langevin, load_prmtop, load_state, log, make_system, platform, platform_props

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=250); ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--per-bin", type=int, default=4); ap.add_argument("--tau-ps", type=float, default=10.0)
ap.add_argument("--bins2d", action="store_true", help="bin on (d_ee, n_hb) instead of d_ee alone")
ap.add_argument("--name", default="we")
a = ap.parse_args()
OUT = RES / a.name; OUT.mkdir(parents=True, exist_ok=True)
EDGES = np.arange(3.0, 36.01, 1.0)
G = {}


def init():
    prmtop = load_prmtop()
    G["system"] = make_system(prmtop)
    G["d_pair"], G["hb"] = cv_atoms(prmtop.topology)


def propagate(args):
    pos, vel, seed = args
    ctx = openmm.Context(G["system"], langevin(seed), platform(), platform_props())
    ctx.setPositions(pos); ctx.setVelocities(vel)
    ig = ctx.getIntegrator(); d = []
    n = int(round(a.tau_ps))
    for _ in range(n):
        ig.step(500)
        st = ctx.getState(getPositions=True)
        d.append(float(cvs(st.getPositions(asNumpy=True)._value * 10, G["d_pair"], G["hb"])[0]))
    st = ctx.getState(getPositions=True, getVelocities=True)
    p = st.getPositions(asNumpy=True)._value
    n_end = int(cvs(p * 10, G["d_pair"], G["hb"])[1])
    return p, st.getVelocities(asNumpy=True)._value, np.array(d), n_end


def bin_of(d, n_hb=0):
    b = int(np.clip(np.searchsorted(EDGES, d) - 1, -1, len(EDGES) - 1))
    return b * 10 + min(int(n_hb), 8) if a.bins2d else b


def resample(walkers, rng):
    """Huber-Kim: split heaviest / merge lightest until each occupied bin holds per_bin walkers."""
    out = []
    bins = {}
    for w in walkers:
        bins.setdefault(w["bin"], []).append(w)
    for b, ws in bins.items():
        ws = [dict(w) for w in ws]
        while len(ws) < a.per_bin:
            ws.sort(key=lambda w: w["weight"]); h = ws.pop()
            ws += [dict(h, weight=h["weight"] / 2), dict(h, weight=h["weight"] / 2)]
        while len(ws) > a.per_bin:
            ws.sort(key=lambda w: w["weight"]); x, y = ws.pop(0), ws.pop(0)
            keep, lose = (x, y) if rng.random() < x["weight"] / (x["weight"] + y["weight"]) else (y, x)
            keep = dict(keep, weight=x["weight"] + y["weight"],
                        merged=keep.get("merged", []) + [lose["parent"]] + lose.get("merged", []))
            ws.append(keep)
        out += ws
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    prmtop = load_prmtop(); d_pair, hb = cv_atoms(prmtop.topology)
    st = load_state(BUILD / "equil_state.xml")
    pos0 = st.getPositions(asNumpy=True)._value; vel0 = st.getVelocities(asNumpy=True)._value
    d0 = float(cvs(pos0 * 10, d_pair, hb)[0])
    walkers = [dict(pos=pos0, vel=vel0, weight=1.0 / a.per_bin, parent="equil", bin=bin_of(d0, cvs(pos0 * 10, d_pair, hb)[1])) for _ in range(a.per_bin)]
    lin = Lineage(a.name); prev_node = "equil"; seed = 1; start = 0
    done = sorted(OUT.glob("iter_*.npz"))
    if done:   # resume: rebuild the walker set from the last saved iteration, then resample
        z = np.load(done[-1]); start = len(done); prev_node = f"we_it{start-1:04d}"; seed = 10_000_000 + start * 1000
        new = [dict(pos=z["end_frames"][k].astype(float) / 10, vel=z["end_vel"][k].astype(float), weight=float(z["weights"][k]),
                    parent=str(z["ids"][k]), bin=int(z["bins"][k]), d=z["d"][k]) for k in range(len(z["ids"]))]
        walkers = resample(new, rng); log("we", f"resumed at iteration {start}")
    with Pool(a.workers, initializer=init) as pool:
        for it in range(start, a.iters):
            jobs = []
            for w in walkers:
                jobs.append((w["pos"], w["vel"], seed)); seed += 1
            res = pool.map(propagate, jobs, chunksize=1)
            ids = [f"{it}:{k}" for k in range(len(walkers))]
            new = []
            for k, (w, (p, v, d, n_end)) in enumerate(zip(walkers, res)):
                new.append(dict(pos=p, vel=v, weight=w["weight"], parent=ids[k], bin=bin_of(d[-1], n_end), d=d))
            np.savez_compressed(OUT / f"iter_{it:04d}.npz",
                                ids=np.array(ids), parents=np.array([w["parent"] for w in walkers]),
                                merged_from=np.array([json.dumps(w.get("merged", [])) for w in walkers]),
                                weights=np.array([w["weight"] for w in walkers]),
                                d=np.array([n["d"] for n in new]), bins=np.array([n["bin"] for n in new]),
                                end_frames=np.array([n["pos"] * 10 for n in new], dtype=np.float32),
                                end_vel=np.array([n["vel"] for n in new], dtype=np.float32))
            node = f"we_it{it:04d}"
            lin.node(node, "iteration", f"iteration {it}: {len(walkers)} walkers", [prev_node], t_ps=a.tau_ps * len(walkers),
                     n_walkers=len(walkers), n_bins=len({n['bin'] for n in new}), d_max=float(max(n["d"].max() for n in new)))
            prev_node = node
            walkers = resample(new, rng)
            if it % 10 == 0:
                log("we", f"iter {it}: {len(new)} walkers, {len({n['bin'] for n in new})} bins, "
                          f"max d {max(n['d'].max() for n in new):.1f}, sum w {sum(n['weight'] for n in new):.6f}")
    log("we", "done")
