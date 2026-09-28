"""Shared pieces for every deca-alanine run: system, CVs, recording, lineage.

System: Ace-(Ala)10-Nme in vacuum, amber ff14SB from tleap, NoCutoff, HBonds,
300 K, 2 fs. Two CVs are reported by every method:

* ``d_ee`` - distance ACE:C to NME:N (Angstrom). About 15 A as a helix and
  about 33 A fully extended. This is the pulling / biasing coordinate.
* ``n_hb`` - number of i -> i+4 backbone O...H-N contacts closer than 2.6 A
  (0-8). This is the helicity readout.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("ESDEMO_RES", ROOT / "results"))
BUILD = ROOT / "results" / "build"
TEMP = 300.0 * u.kelvin
DT = 2.0 * u.femtoseconds
FRICTION = 1.0 / u.picosecond
KT = (u.MOLAR_GAS_CONSTANT_R * TEMP).value_in_unit(u.kilocalorie_per_mole)  # kcal/mol
D_MIN, D_MAX = 3.0, 36.0          # Angstrom; every PMF is reported on this range
WALL_HI = 35.0                     # Angstrom; flat-bottom wall for the biased methods


def platform():
    return openmm.Platform.getPlatformByName("CPU")


def platform_props():
    return {"Threads": os.environ.get("OPENMM_CPU_THREADS", "1")}


def load_prmtop():
    return app.AmberPrmtopFile(str(BUILD / "deca.prmtop"))


def make_system(prmtop=None, dihedral_group=False):
    prmtop = prmtop or load_prmtop()
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    if dihedral_group:
        for f in system.getForces():
            f.setForceGroup(1 if isinstance(f, openmm.PeriodicTorsionForce) else 0)
    return system


def cv_atoms(topology):
    """(i, j) of the d_ee pair and the list of (O, H) i->i+4 pairs."""
    res = list(topology.residues())
    atom = lambda r, name: next(a.index for a in r.atoms() if a.name == name)
    d_pair = (atom(res[0], "C"), atom(res[-1], "N"))
    hb = []
    for i in range(len(res) - 4):
        hb.append((atom(res[i], "O"), atom(res[i + 4], "H")))
    return d_pair, hb


def cvs(pos_A, d_pair, hb):
    """pos_A: (..., n_atoms, 3) in Angstrom -> (d_ee, n_hb) arrays."""
    pos_A = np.asarray(pos_A)
    d = np.linalg.norm(pos_A[..., d_pair[0], :] - pos_A[..., d_pair[1], :], axis=-1)
    o = pos_A[..., [p[0] for p in hb], :]
    h = pos_A[..., [p[1] for p in hb], :]
    n = (np.linalg.norm(o - h, axis=-1) < 2.6).sum(axis=-1)
    return d, n


def dee_force(d_pair):
    """CustomBondForce whose energy is the d_ee distance (nm); for CustomCVForce."""
    f = openmm.CustomBondForce("r")
    f.addBond(*d_pair, [])
    return f


def add_upper_wall(system, d_pair, at_A=WALL_HI, k=10.0):
    """Flat-bottom harmonic wall on d_ee (k in kcal/mol/A^2)."""
    k_kj_nm = k * 4.184 * 100
    f = openmm.CustomBondForce(f"step(r-{at_A/10})*0.5*{k_kj_nm}*(r-{at_A/10})^2")
    f.addBond(*d_pair, [])
    f.setForceGroup(2)
    system.addForce(f)
    return f


def langevin(seed=None):
    integ = openmm.LangevinMiddleIntegrator(TEMP, FRICTION, DT)
    if seed is not None:
        integ.setRandomNumberSeed(int(seed))
    return integ


def load_state(path):
    with open(path) as fh:
        return openmm.XmlSerializer.deserialize(fh.read())


def save_state(ctx, path):
    st = ctx.getState(getPositions=True, getVelocities=True)
    Path(path).write_text(openmm.XmlSerializer.serialize(st))


def positions_A(ctx):
    return ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.angstrom).astype(np.float32)


class Recorder:
    """Accumulates frames (Angstrom, float32) + CVs + arbitrary scalars; saves npz."""

    def __init__(self, d_pair, hb):
        self.d_pair, self.hb = d_pair, hb
        self.frames, self.t, self.extra = [], [], {}

    def add(self, pos_A, t_ps, **scalars):
        self.frames.append(np.asarray(pos_A, dtype=np.float32))
        self.t.append(t_ps)
        for k, v in scalars.items():
            self.extra.setdefault(k, []).append(v)

    def save(self, path, **more):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fr = np.array(self.frames, dtype=np.float32)
        d, n = cvs(fr, self.d_pair, self.hb) if len(fr) else (np.zeros(0), np.zeros(0))
        np.savez_compressed(path, frames=fr, t_ps=np.array(self.t), d_ee=d, n_hb=n,
                            **{k: np.array(v) for k, v in self.extra.items()}, **more)
        return d, n


class Lineage:
    """Append-only node log: results/<method>/lineage.jsonl."""

    def __init__(self, method):
        self.dir = RES / method
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "lineage.jsonl"

    def node(self, id, kind, label, parents=(), t_ps=0.0, traj=None, **meta):
        rec = dict(id=id, kind=kind, label=label, parents=list(parents), t_ps=float(t_ps),
                   traj=traj, meta=meta, wall=time.time())
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return id


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj))


def log(method, msg):
    print(f"[{time.strftime('%H:%M:%S')}] {method}: {msg}", flush=True)
