#!/usr/bin/env python
"""Reference PMF: umbrella sampling on d_ee + MBAR (analysis does the MBAR).

One fast pull provides seed structures along d_ee; each window then runs
200 ps equilibration + ``--ns`` production, recording d_ee every 0.5 ps.
"""
import argparse
from multiprocessing import Pool

import numpy as np

from common import BUILD, RES, Lineage, cvs, load_state, log
from pulling import pos_A, pull, restrained_context

ap = argparse.ArgumentParser()
ap.add_argument("--ns", type=float, default=3.0); ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--k", type=float, default=2.0)
ap.add_argument("--low", action="store_true", help="windows 4-12.25 A, seeded by a compressing pull")
a = ap.parse_args()
OUT = RES / "reference"; OUT.mkdir(parents=True, exist_ok=True)
CENTERS = np.round(np.arange(13.0, 35.01, 0.75), 2)
if a.low:
    CENTERS = np.round(np.arange(4.0, 12.26, 0.75), 2)
OFF = 100 if a.low else 0
SEED = "seed_pull_low.npz" if a.low else "seed_pull.npz"


def window(args):
    i, r0 = args
    i += OFF
    seeds = np.load(OUT / SEED)
    j = int(np.argmin(np.abs(seeds["r0"] - r0)))
    ctx, d_pair, hb = restrained_context(a.k, r0, seed=500 + i)
    ctx.setPositions(seeds["frames"][j] / 10)
    ctx.setVelocitiesToTemperature(300)
    ig = ctx.getIntegrator(); ig.step(100000)
    d, frames = [], []
    n = int(a.ns * 1000 / 0.5)
    for s in range(n):
        ig.step(250)
        p = pos_A(ctx); d.append(float(cvs(p, d_pair, hb)[0]))
        if s % 40 == 0:
            frames.append(p)
    fr = np.array(frames); dd, nn = cvs(fr, d_pair, hb)
    np.savez_compressed(OUT / f"win_{i:02d}.npz", r0=r0, k=a.k, d_all=np.array(d), frames=fr, d_ee=dd, n_hb=nn,
                        t_ps=np.arange(len(fr)) * 20.0)
    return i, r0, float(np.mean(d))


if __name__ == "__main__":
    lin = Lineage("reference")
    st = load_state(BUILD / "equil_state.xml")
    lo, hi, pid = (15.5, 3.5, "ref_pull_low") if a.low else (15.5, 35.5, "ref_pull")
    r = pull(st, lo, hi, 50.0, k=5.0, seed=77 + OFF, frame_every_A=0.25)
    r.pop("state"); np.savez_compressed(OUT / SEED, **r)
    lin.node(pid, "pull", f"fast pull (50 A/ns) {lo:g}->{hi:g} A to seed windows", ["equil"], t_ps=abs(hi - lo) / 50 * 1000,
             traj=f"reference/{SEED}")
    log("reference", "seed pull done")
    with Pool(a.workers) as pool:
        for i, r0, m in pool.imap_unordered(window, list(enumerate(CENTERS))):
            lin.node(f"ref_win_{i:02d}", "window", f"window r0={r0:.2f} A", ["ref_pull_low" if a.low else "ref_pull"], t_ps=a.ns * 1000 + 200,
                     traj=f"reference/win_{i:02d}.npz", r0=r0, mean=m)
            log("reference", f"window {i} r0={r0} <d>={m:.2f}")
    log("reference", "done")
