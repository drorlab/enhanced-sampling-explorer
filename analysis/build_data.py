#!/usr/bin/env python
"""Turn results/ into dashboard/data/ (see dashboard/SCHEMA.md).

Safe to run while simulations are still going: every method is optional and
uses whatever has been written so far.

    python analysis/build_data.py
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sims")); sys.path.insert(0, str(ROOT / "analysis"))
from common import KT, RES, cv_atoms, cvs, load_prmtop  # noqa: E402

DATA = ROOT / "dashboard" / "data"
TRAJ = DATA / "traj"
BETA = 1.0 / KT
EDGES = np.arange(3.0, 36.01, 0.5)
CENT = 0.5 * (EDGES[1:] + EDGES[:-1])
MAX_FRAMES = 400
FRACS = tuple(np.round(np.geomspace(0.03, 1.0, 14), 4))   # sampling checkpoints for convergence

COLORS = dict(metad="#2a78d6", smd="#eb6834", we="#1baf7a", remd="#eda100", sams="#e87ba4",
              gamd="#008300", md="#4a3aa7", reference="#52514e", we2d="#e34948", samsT="#e87ba4")
TEXT2, MUTED, RAMP = "#52514e", "#9a9893", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NAMES = dict(md="Plain MD", metad="Well-tempered metadynamics", smd="Steered MD + Jarzynski",
             we="Weighted ensemble (bins on d_ee)", we2d="Weighted ensemble (bins on d_ee × H-bonds)", remd="Temperature REMD", sams="SAMS (adaptive umbrellas)", samsT="SAMS over temperature (simulated tempering)",
             gamd="Gaussian accelerated MD", reference="Umbrella sampling + MBAR (reference)")
BLURB = dict(
    md='Unbiased Langevin dynamics at 300 K for 20 ns. The baseline: it samples the helix and the compact states but never reaches the extended chain, so it says nothing about the uphill side of the profile.',
    metad="Every 1 ps a Gaussian (1.2 kJ/mol high, 0.5 Å wide) is added to a bias on d_ee at the walker's position, pushing it away from where it has already been. Well-tempered: heights shrink as exp(−V/kΔT) with γ = 10, so the bias converges instead of overfilling.",
    gamd='Dual-boost Gaussian accelerated MD: whenever the total or dihedral energy V is below a threshold E, a harmonic boost ½k(E−V)² is added, lifting every energy basin without choosing a CV. E and k come from a short conventional run and keep the boost near-Gaussian (σ₀ = 6 kcal/mol).',
    remd='Temperature replica exchange: 10 copies at 300–700 K run side by side, and every 1 ps neighbouring temperatures attempt a Metropolis swap. Hot copies cross barriers and hand their configurations down to 300 K. No CV.',
    samsT="Self-adjusting mixture sampling over temperature (simulated tempering): one copy hops among 12 temperatures (300–700 K) while SAMS learns each temperature's free energy on the fly, so all are visited equally. No CV.",
    sams="Self-adjusting mixture sampling over umbrellas: one copy hops among 42 harmonic restraints on d_ee (4–35 Å) while SAMS learns each restraint's free energy on the fly, so all are visited equally. Adaptive umbrella sampling with a single walker.",
    we='Weighted ensemble: many short (10 ps) unbiased walkers, each carrying a probability weight. After each segment, walkers are split in under-occupied 1 Å d_ee bins and merged in over-occupied ones (4 per bin), conserving total weight exactly. It stalls near 21 Å, where going further means breaking helical H-bonds that d_ee bins cannot see.',
    we2d='The same weighted ensemble, binned on (d_ee, number of helical H-bonds), so walkers that start to lose H-bonds get their own bins and are split. More bins means more walkers per iteration, and so fewer iterations for the same cost.',
    smd="Steered MD: a spring (7.2 kcal/mol/Å²) drags d_ee from the helix to 34 Å at constant speed, far from equilibrium; 32 pulls at 10 Å/ns and 32 at 100 Å/ns. Jarzynski's equality recovers the equilibrium free energy from the work distribution.",
    reference='Umbrella sampling: 42 independent windows, each restrained near a fixed d_ee (4–35 Å, 2 kcal/mol/Å²) for 3 ns, seeded from one fast pull. Windows never exchange; MBAR combines them afterwards. This is the reference curve on every page.',
)
FE_HOW = dict(
    md='F(d_ee) = −kT ln P(d_ee), from the trajectory histogram.',
    metad='F(s) = −γ/(γ−1) · V(s) + const, read directly from the bias.',
    gamd='Frames reweighted by exp(βΔV); per bin via a 2nd-order cumulant expansion.',
    remd='MBAR over all replicas and temperatures, evaluated at 300 K.',
    samsT="MBAR over the walker's samples at all temperatures, evaluated at 300 K.",
    sams="MBAR over all umbrella states; SAMS's online log Z is a smoothed estimate of the same profile.",
    we='F(bin) = −kT ln(total walker weight in the bin).',
    we2d='F(bin) = −kT ln(total walker weight), summed over H-bond bins onto d_ee.',
    smd='F(r₀) = −kT ln⟨exp(−W(r₀)/kT)⟩ over pulls (Jarzynski).',
    reference='MBAR over all windows.',
)
NEEDS_CV = dict(md=False, metad=True, smd=True, we=True, we2d=True, remd=False, sams=True, samsT=False, gamd=False, reference=True)
ORDER = ["md", "metad", "gamd", "remd", "samsT", "sams", "we", "we2d", "smd", "reference"]


# ----------------------------------------------------------------- helpers
def jdump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), default=_default))


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(type(o))


def clean(a, nd=3):
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(x) else round(float(x), nd) for x in a.ravel()]


def pmf_hist(d, w=None, logw=None):
    """F (kcal/mol) on CENT from samples d with weights w or log-weights logw; min = 0."""
    d = np.asarray(d).ravel()
    if logw is not None:
        logw = np.asarray(logw).ravel()
        idx = np.digitize(d, EDGES) - 1
        F = np.full(len(CENT), np.nan)
        for b in np.unique(idx):
            if 0 <= b < len(CENT):
                F[b] = -logsumexp(logw[idx == b])
        F *= KT
    else:
        h, _ = np.histogram(d, EDGES, weights=w)
        with np.errstate(divide="ignore"):
            F = -KT * np.log(h)
        F[~np.isfinite(F)] = np.nan
    return F - np.nanmin(F) if np.isfinite(F).any() else F


def dF(F):
    """F(30 A) - F(15 A) (helix -> extended)."""
    f = lambda x: F[np.argmin(np.abs(CENT - x))]
    v = f(30.0) - f(15.0)
    return None if not np.isfinite(v) else float(v)


def align_to(ref, frames, idx):
    """Kabsch-superpose each frame onto ref using atoms idx."""
    out = np.empty_like(frames)
    rc = ref[idx].mean(0); R0 = ref[idx] - rc
    for i, fr in enumerate(frames):
        c = fr[idx].mean(0); X = fr[idx] - c
        U, _, Vt = np.linalg.svd(X.T @ R0)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        out[i] = (fr - c) @ (U @ D @ Vt) + rc
    return out


G = {}

_np_load = np.load


def _load_retry(f, *a, **k):
    for i in range(5):
        try:
            z = _np_load(f, *a, **k)
            if hasattr(z, "files"):
                z = {n: z[n] for n in z.files}   # force full read now, inside the retry
            return z
        except Exception:
            if i == 4:
                raise
            time.sleep(2)


np.load = _load_retry


def write_traj(key, frames, t_ns, title, labels=None):
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 0:
        return None
    sel = np.unique(np.linspace(0, len(frames) - 1, min(MAX_FRAMES, len(frames))).astype(int))
    fr = align_to(G["ref"], frames[sel], G["bb"]).astype("<f4")
    d, n = cvs(fr, G["d_pair"], G["hb"])
    (TRAJ / f"{key}.bin").write_bytes(fr.tobytes())
    jdump(TRAJ / f"{key}.json", dict(n_frames=len(fr), n_atoms=fr.shape[1], t_ns=clean(np.asarray(t_ns)[sel], 4),
                                     d_ee=clean(d, 2), n_hb=[int(x) for x in n], title=title,
                                     frame_label=None if labels is None else [labels[i] for i in sel], bin=f"{key}.bin"))
    return key


def fe_sync(key, snaps, hint, color=None):
    """Live panel: F(d_ee) estimate as of the movie's current time. snaps = [(t_ns on the trajectory's clock, F)]."""
    snaps = [(float(t), F) for t, F in snaps if F is not None and np.isfinite(F).any()]
    snaps.sort(key=lambda x: x[0])
    return dict(kind="pmf", title="Free-energy estimate so far", hint=hint, x=clean(CENT, 2), t_ns=[round(t, 4) for t, _ in snaps],
                V=[clean(F, 2) for _, F in snaps], ref=clean(G["refF"], 2) if G.get("refF") is not None else None,
                color=color or COLORS[key], ylabel="F (kcal/mol)", yrange=[0, 30], xrange=[3, 36])


def disp_sel(n):
    """Frame indices write_traj keeps for an n-frame trajectory."""
    return np.unique(np.linspace(0, n - 1, min(MAX_FRAMES, n)).astype(int))


def ensemble(title, hint, members, events=None, tint=None, exchange=False):
    """Grid view: members = [(traj key, short label)], all with the same frame count.
    events[f] = list of [a, b] member pairs that exchanged between displayed frame f-1 and f.
    tint[m][f] in 0..1 colours cell m at frame f (e.g. its temperature)."""
    return dict(title=title, hint=hint, members=[dict(key=k, label=l) for k, l in members if k],
                events=events, tint=tint, exchange=exchange)


def mbar_logw_target(u_kn, N_k, u_target):
    """log weights of each sample for the target state (unnormalized), via pymbar f_k."""
    from pymbar import MBAR
    m = MBAR(u_kn, N_k, verbose=False, solver_protocol="robust")
    f_k = m.f_k
    denom = logsumexp(np.log(np.maximum(N_k, 1e-300))[:, None] + f_k[:, None] - u_kn, axis=0)
    return -u_target - denom, m


# Plotly spec builders ----------------------------------------------------
def line(x, y, name, color, dash=None, width=2, **kw):
    t = dict(type="scatter", mode="lines", x=clean(x), y=clean(y), name=name,
             line=dict(color=color, width=width, **({"dash": dash} if dash else {})))
    t.update(kw)
    return t


def ref_trace():
    return line(CENT, G["refF"], "reference (US+MBAR)", TEXT2, dash="dash", width=2) if G.get("refF") is not None else None


def panel(id, title, hint, data, layout=None, wide=False):
    return dict(id=id, title=title, hint=hint, wide=wide, data=[t for t in data if t], layout=layout or {})


def pmf_layout(extra=None):
    lay = dict(xaxis=dict(title="d_ee (Å)", range=[3, 36]), yaxis=dict(title="F (kcal/mol)", rangemode="tozero"))
    lay.update(extra or {})
    return lay


def conv_panel(key, curves, color, hint):
    """curves: list of (label, ns, F). Older = lighter. At most 6 are drawn."""
    if len(curves) > 6:
        curves = [curves[i] for i in np.unique(np.linspace(0, len(curves) - 1, 6).astype(int))]
    data = [ref_trace()]
    n = len(curves)
    for i, (lab, ns, F) in enumerate(curves):
        op = 0.25 + 0.75 * (i + 1) / n
        data.append(line(CENT, F, lab, color, width=2 if i == n - 1 else 1.5, opacity=op, showlegend=(i in (0, n - 1))))
    return panel(f"{key}-conv", "Free-energy estimate as sampling accumulates", hint, data,
                 pmf_layout(dict(yaxis=dict(title="F (kcal/mol)", range=[0, 30]))))


def dF_panel(key, ns, vals, color, errs=None):
    t = dict(type="scatter", mode="lines+markers", x=clean(ns), y=clean(vals), name=NAMES[key],
             line=dict(color=color, width=2), marker=dict(size=8, color=color))
    if errs is not None:
        t["error_y"] = dict(type="data", array=clean(errs), color=color, thickness=1.5)
    data = [t]
    if G.get("refdF") is not None:
        data.append(dict(type="scatter", mode="lines", x=[0, max(clean(ns) or [1]) or 1], y=[G["refdF"]] * 2,
                         name="reference", line=dict(color=TEXT2, dash="dash", width=2)))
    return panel(f"{key}-dF", "ΔF(helix 15 Å → extended 30 Å) vs cost", "x = total simulated ns across all walkers/replicas",
                 data, dict(xaxis=dict(title="simulated ns (all copies)"), yaxis=dict(title="ΔF (kcal/mol)")))


def hist2d_panel(key, d, n, w=None):
    H, _, _ = np.histogram2d(np.asarray(n).ravel(), np.asarray(d).ravel(), bins=[np.arange(-0.5, 9.5, 1), EDGES], weights=w)
    with np.errstate(divide="ignore"):
        F = -KT * np.log(H / H.sum())
    F[~np.isfinite(F)] = np.nan; F -= np.nanmin(F)
    data = [dict(type="heatmap", x=clean(CENT), y=list(range(9)), z=[clean(r, 2) for r in F], zmin=0, zmax=15,
                 colorscale=[[i / 6, c] for i, c in enumerate(RAMP[::-1])], colorbar=dict(title="F", thickness=12),
                 hovertemplate="d_ee %{x:.1f} Å<br>%{y} H-bonds<br>F %{z:.1f}<extra></extra>")]
    return panel(f"{key}-2d", "States sampled", "free energy over d_ee × i→i+4 H-bonds (dark = low F); blank = never visited",
                 data, dict(xaxis=dict(title="d_ee (Å)", range=[3, 36]), yaxis=dict(title="helical H-bonds", dtick=1)))


def stats_list(**kv):
    return [dict(value=v, label=k.replace("_", " ")) for k, v in kv.items()]


def lineage(key, extra_nodes=None, collapse=None):
    nodes, edges = {}, []
    for f in [RES / "shared" / "lineage.jsonl", RES / key / "lineage.jsonl"]:
        if f.exists():
            for line_ in f.read_text().splitlines():
                r = json.loads(line_)
                nodes[r["id"]] = dict(id=r["id"], label=r["label"], kind=r["kind"], t_ps=r["t_ps"],
                                      traj=None, strip=None, _parents=r["parents"])
    if collapse:
        nodes = collapse(nodes)
    for nd in extra_nodes or []:
        nodes[nd["id"]] = nd
    for nd in nodes.values():
        for p in nd.pop("_parents", []):
            if p in nodes:
                edges.append([p, nd["id"]])
    return dict(nodes=list(nodes.values()), edges=edges)


def ensure(lin, id, label, kind, parents, t_ps=0.0):
    """Add a placeholder for a segment that is still running (nodes are logged on completion)."""
    if not any(n["id"] == id for n in lin["nodes"]):
        lin["nodes"].append(dict(id=id, label=label + " (running)", kind=kind, t_ps=t_ps, traj=None, strip=None))
        lin["edges"] += [[p, id] for p in parents]


def set_traj(lin, node_id, traj_key):
    for nd in lin["nodes"]:
        if nd["id"] == node_id:
            nd["traj"] = traj_key


# ----------------------------------------------------------------- methods
def do_reference():
    files = sorted(glob.glob(str(RES / "reference" / "win_*.npz")))
    if len(files) < 3:
        return None
    wins = [np.load(f) for f in files]
    r0 = np.array([float(w["r0"]) for w in wins]); k = float(wins[0]["k"])
    order = np.argsort(r0); wins = [wins[i] for i in order]; r0 = r0[order]

    def estimate(frac):
        ds = [w["d_all"][: max(10, int(len(w["d_all"]) * frac))][::4] for w in wins]
        x = np.concatenate(ds); N = np.array([len(d) for d in ds])
        u = BETA * 0.5 * k * (x[None, :] - r0[:, None]) ** 2
        logw, m = mbar_logw_target(u, N, np.zeros(len(x)))
        return pmf_hist(x, logw=logw), m, x
    curves = []
    for frac in FRACS:
        F, m, x = estimate(frac)
        ns = sum(len(w["d_all"]) for w in wins) * 0.5e-3 * frac
        curves.append((f"{frac:.0%} of each window", ns, F))
    F = curves[-1][2]
    G["refF"] = F; G["refdF"] = dF(F)
    win_ns = len(wins[0]["d_all"]) * 0.5e-3
    sync_fe = fe_sync("reference", [(fr_ * win_ns, estimate(fr_)[0]) for fr_ in np.linspace(0.05, 1, 20)],
                      "MBAR over all 42 windows, each truncated at the time shown")
    data = [line(CENT, F, "reference", TEXT2, width=2.5)]
    hist = []
    for i, w in enumerate(wins):
        h, _ = np.histogram(w["d_all"], EDGES, density=True)
        hist.append(dict(type="scatter", mode="lines", x=clean(CENT), y=clean(h), line=dict(width=1.5, color=RAMP[2 + i % 5]),
                         fill="tozeroy", opacity=0.5, name=f"r0={w['r0']:.2f}", showlegend=False))
    ov = m.compute_overlap()["matrix"]
    tkeys = []
    for i in (0, len(wins) // 3, 2 * len(wins) // 3, len(wins) - 1):
        w = wins[i]
        tkeys.append(write_traj(f"reference_w{i:02d}", w["frames"], w["t_ps"] / 1000, f"window r0 = {float(w['r0']):.2f} Å"))
    members = []
    nmin = min(len(w["frames"]) for w in wins)
    for i in np.unique(np.linspace(0, len(wins) - 1, 12).astype(int)):
        w = wins[i]
        kk = write_traj(f"reference_grid{i:02d}", w["frames"][:nmin], w["t_ps"][:nmin] / 1000, f"window r0 = {float(w['r0']):.2f} Å",
                        labels=[f"r0 {float(w['r0']):.1f} Å · d_ee {x:.1f} Å" for x in w["d_ee"][:nmin]])
        members.append((kk, f"r0 {float(w['r0']):.1f} Å"))
    ens = ensemble("12 of the 42 umbrella windows side by side", "independent restrained runs: windows never exchange; MBAR combines them afterwards", members)
    total = sum(len(w["d_all"]) for w in wins) * 0.5e-3 + len(wins) * 0.2
    lin = lineage("reference")
    ids = sorted([n["id"] for n in lin["nodes"] if n["kind"] == "window"])
    for i, tk in zip((0, len(wins) // 3, 2 * len(wins) // 3, len(wins) - 1), tkeys):
        if i < len(ids):
            set_traj(lin, ids[i], tk)
    method = dict(key="reference", syncs=[sync_fe], ensemble=ens, stats=stats_list(windows=str(len(wins)), simulated=f"{total:.0f} ns",
                                                   **{"ΔF 15→30 Å": f"{G['refdF']:.1f} kcal/mol" if G['refdF'] else "–"}),
                  trajectories=[dict(key=t, label=t.split("_w")[1], group="window") for t in tkeys if t],
                  default_traj=tkeys[0],
                  panels=[panel("ref-pmf", "Reference PMF", "MBAR over all windows", data, pmf_layout()),
                          panel("ref-hist", "Window histograms (overlap is what MBAR needs)", "one curve per umbrella window",
                                hist, dict(xaxis=dict(title="d_ee (Å)"), yaxis=dict(title="density"), showlegend=False)),
                          panel("ref-ov", "MBAR overlap matrix", "neighbouring windows must overlap",
                                [dict(type="heatmap", z=[clean(r, 3) for r in ov], x=clean(r0, 2), y=clean(r0, 2),
                                      colorscale=[[0, "#ffffff"], [1, RAMP[5]]], zmin=0, zmax=0.5, colorbar=dict(thickness=12))],
                                dict(xaxis=dict(title="window r0 (Å)"), yaxis=dict(title="window r0 (Å)"))),
                          conv_panel("reference", curves, TEXT2, "MBAR on the first 10/25/50/100% of each window")])
    return method, lin, dict(total_ns=total, dF=G["refdF"], F=F, cost=[(c[1], dF(c[2])) for c in curves], curves=[(c[1], c[2]) for c in curves])


def do_md():
    f = RES / "md" / "traj.npz"
    if not f.exists():
        return None
    z = np.load(f); d, n, t = z["d_ee"], z["n_hb"], z["t_ps"] / 1000
    tk = write_traj("md", z["frames"], t, "unbiased MD")
    curves = []
    for frac in FRACS:
        m = max(2, int(len(d) * frac)); curves.append((f"{t[m-1]:.1f} ns", t[m - 1], pmf_hist(d[:m])))
    snaps = [(t[m - 1], pmf_hist(d[:m])) for m in np.unique(np.linspace(20, len(d), 40).astype(int))]
    sync_fe = fe_sync("md", snaps, "histogram of everything sampled up to the frame shown")
    lin = lineage("md"); ensure(lin, "md", f"unbiased MD, {t[-1]:.1f} ns", "prod", ["equil"], t[-1] * 1000); set_traj(lin, "md", tk); set_traj(lin, "equil", write_traj("equil", np.load(RES / "build" / "equil.npz")["frames"], np.load(RES / "build" / "equil.npz")["t_ps"] / 1000, "equilibration"))
    F = curves[-1][2]
    method = dict(key="md", syncs=[sync_fe], stats=stats_list(simulated=f"{t[-1]:.1f} ns", **{"max d_ee": f"{d.max():.1f} Å",
                                             "ΔF 15→30 Å": "never reached" if dF(F) is None else f"{dF(F):.1f}"}),
                  trajectories=[dict(key=tk, label="production", group="trajectory"), dict(key="equil", label="equilibration", group="trajectory")],
                  default_traj=tk,
                  panels=[hist2d_panel("md", d, n), conv_panel("md", curves, COLORS["md"], "histogram of the trajectory so far"),
                          panel("md-hist", "Where the trajectory spends its time", "the extended state (> 25 Å) is never visited",
                                [dict(type="histogram", x=clean(d, 2), xbins=dict(start=3, end=36, size=0.5),
                                      marker=dict(color=COLORS["md"]), name="MD")],
                                dict(xaxis=dict(title="d_ee (Å)", range=[3, 36]), yaxis=dict(title="frames")))])
    return method, lin, dict(total_ns=float(t[-1]), dF=dF(F), F=F, cost=[(c[1], dF(c[2])) for c in curves], curves=[(c[1], c[2]) for c in curves])


def do_metad():
    f = RES / "metad" / "metad.npz"
    if not f.exists():
        return None
    z = np.load(f); tr = np.load(RES / "metad" / "traj.npz")
    grid, fes, ft = z["grid_A"], z["fes"], z["fes_t_ps"] / 1000
    gamma = float(z["gamma"]); h0 = float(z["height"]) / 4.184
    Fs = [np.interp(CENT, grid, fe) for fe in fes]
    Fs = [F - F[(CENT > 13) & (CENT < 17)].min() + (G["refF"][(CENT > 13) & (CENT < 17)].min() if G.get("refF") is not None else 0) for F in Fs]
    Fs = [F - np.nanmin(F) for F in Fs]
    pick = np.unique(np.linspace(0, len(Fs) - 1, min(6, len(Fs))).astype(int))
    curves = [(f"{ft[i]:.1f} ns", ft[i], Fs[i]) for i in pick]
    tk = write_traj("metad", tr["frames"], tr["t_ps"] / 1000, "WT-metaD walker")
    # bias filling: V = -(gamma-1)/gamma F_est
    fill = [ref_trace()]
    for j, i in enumerate(pick):
        V = -(gamma - 1) / gamma * (fes[i] - fes[i].max()) / 1.0
        fill.append(line(grid, (V - V.min()), f"bias V at {ft[i]:.1f} ns", COLORS["metad"], width=1.5,
                         opacity=0.25 + 0.75 * (j + 1) / len(pick)))
    dT = (gamma - 1) * 300.0
    heights = h0 * np.exp(-np.array(z["bias_at_walker"]) / 4.184 / (0.0019872 * dT))
    ct = z["cv_t_ps"] / 1000
    s = slice(None, None, max(1, len(ct) // 3000))
    sync = dict(title="Bias potential as the walker samples", hint="bias V(s,t) at the time shown in the movie; dot = walker",
                x=clean(grid[::2], 2), t_ns=clean(ft, 3), gamma=gamma,
                V=[clean((-(gamma - 1) / gamma * (fe - fe.max()))[::2] / 1.0, 2) for fe in fes],
                ref=clean(np.interp(grid[::2], CENT, G["refF"], left=np.nan, right=np.nan), 2) if G.get("refF") is not None else None,
                color=COLORS["metad"])
    lin = lineage("metad"); ensure(lin, "metad", f"WT-metaD, {ct[-1]:.1f} ns", "prod", ["equil"], ct[-1] * 1000); set_traj(lin, "metad", tk)
    F = Fs[-1]
    method = dict(key="metad", syncs=[sync, fe_sync("metad", list(zip(ft, Fs)), "F = −(T+ΔT)/ΔT · V at the frame shown; dashed = reference")], stats=stats_list(simulated=f"{ct[-1]:.1f} ns", γ=f"{gamma:g}", **{"ΔF 15→30 Å": f"{dF(F):.1f} kcal/mol" if dF(F) else "–"}),
                  trajectories=[dict(key=tk, label="walker", group="trajectory")], default_traj=tk,
                  panels=[panel("metad-cv", "Walker d_ee and deposited Gaussian height", "heights shrink as the bias fills (well-tempered)",
                                [line(ct[s], z["cv_d"][s], "d_ee", COLORS["metad"], width=1),
                                 line(ct[s], heights[s], "Gaussian height", TEXT2, width=1.5, xaxis="x", yaxis="y2")],
                                dict(grid=dict(rows=2, columns=1, pattern="independent"),
                                     xaxis=dict(title="t (ns)"), yaxis=dict(title="d_ee (Å)"),
                                     xaxis2=dict(title="t (ns)", matches="x"), yaxis2=dict(title="height (kcal/mol)", domain=[0, 0.4]),
                                     showlegend=False), wide=False),
                          panel("metad-fill", "The bias fills the free-energy wells", "bias V(s,t) grows until it mirrors (γ-1)/γ · F", fill,
                                pmf_layout(dict(yaxis=dict(title="V (kcal/mol)")))),
                          conv_panel("metad", curves, COLORS["metad"], "F = -(T+ΔT)/ΔT · V at increasing times"),
                          hist2d_panel("metad", tr["d_ee"], tr["n_hb"])])
    # fix the metad-cv panel: use simple two-trace overlay with separate axes via subplot domains
    method["panels"][0]["data"][0].update(xaxis="x", yaxis="y")
    method["panels"][0]["data"][1].update(xaxis="x2", yaxis="y2")
    method["panels"][0]["layout"] = dict(xaxis=dict(title="", anchor="y", showticklabels=False), yaxis=dict(title="d_ee (Å)", domain=[0.45, 1]),
                                         xaxis2=dict(title="t (ns)", anchor="y2", matches="x"), yaxis2=dict(title="height", domain=[0, 0.35]),
                                         showlegend=False)
    allc = [(ft[i], Fs[i]) for i in range(len(Fs))]
    return method, lin, dict(total_ns=float(ct[-1]), dF=dF(F), F=F, cost=[(n_, dF(F_)) for n_, F_ in allc], curves=allc)


def do_gamd():
    fp = RES / "gamd" / "prod.npz"
    if not fp.exists():
        return None
    parts = {s: np.load(RES / "gamd" / f"{s}.npz") for s in ("cmd", "equil", "prod") if (RES / "gamd" / f"{s}.npz").exists()}
    z = parts["prod"]; d, dV = z["d_ee"], z["dV"] / 4.184
    bdV = BETA * dV
    raw = pmf_hist(d)
    expw = pmf_hist(d, logw=bdV)

    def cumul(d, x):
        idx = np.digitize(d, EDGES) - 1; F = np.full(len(CENT), np.nan)
        for b in np.unique(idx):
            if 0 <= b < len(CENT):
                xb = x[idx == b]; p = len(xb) / len(x)
                F[b] = -KT * (np.log(p) + xb.mean() + 0.5 * xb.var())
        return F - np.nanmin(F)
    cum = cumul(d, bdV)
    t = z["t_ps"] / 1000
    curves = []
    for frac in FRACS:
        m = max(10, int(len(d) * frac)); curves.append((f"{t[m-1]-t[0]:.1f} ns prod", t[m - 1], cumul(d[:m], bdV[:m])))
    snaps = [(t[m - 1], cumul(d[:m], bdV[:m])) for m in np.unique(np.linspace(50, len(d), 30).astype(int))]
    sync_fe = fe_sync("gamd", snaps, "cumulant-reweighted PMF from production up to the frame shown (play the prod trajectory)")
    lin = lineage("gamd")
    ensure(lin, "gamd_cmd", "cMD: collect V stats", "cmd", ["equil"]); ensure(lin, "gamd_equil", "GaMD equilibration", "equil", ["gamd_cmd"])
    ensure(lin, "gamd_prod", "GaMD production", "prod", ["gamd_equil"])
    keys = {}
    for s, zz in parts.items():
        keys[s] = write_traj(f"gamd_{s}", zz["frames"], zz["t_ps"] / 1000, f"GaMD {s}",
                             labels=[f"ΔV {x/4.184:.1f} kcal/mol" for x in zz["dV"]])
        set_traj(lin, f"gamd_{s}", keys[s])
    params = json.loads((RES / "gamd" / "params.json").read_text()) if (RES / "gamd" / "params.json").exists() else None
    h, e = np.histogram(dV, 60, density=True); c = 0.5 * (e[1:] + e[:-1])
    mu, sd = dV.mean(), dV.std()
    gauss = np.exp(-0.5 * ((c - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    from scipy.stats import skew
    panels = [panel("gamd-rew", "Reweighting the boosted ensemble", "raw = biased histogram; exp = exact but noisy; cumulant = GaMD's estimator",
                    [ref_trace(), line(CENT, raw, "raw (no reweighting)", MUTED, width=1.5, dash="dot"),
                     line(CENT, expw, "exp(βΔV) average", COLORS["gamd"], width=1.5, opacity=0.5),
                     line(CENT, cum, "2nd-order cumulant", COLORS["gamd"], width=2.5)], pmf_layout(dict(yaxis=dict(title="F (kcal/mol)", range=[0, 25])))),
              panel("gamd-dv", "Boost potential distribution", f"near-Gaussian ΔV keeps the cumulant expansion accurate; skew {skew(dV):.2f}, σ {sd:.1f} kcal/mol",
                    [dict(type="bar", x=clean(c), y=clean(h), marker=dict(color=COLORS["gamd"]), name="ΔV", opacity=0.7),
                     line(c, gauss, "Gaussian fit", TEXT2, width=2)],
                    dict(xaxis=dict(title="ΔV (kcal/mol)"), yaxis=dict(title="density"), bargap=0.05))]
    if params:
        P = params["final"]["P"]
        V = np.linspace(P["vmin"] - 20, P["vmax"] + 40, 200)
        Vb = V + np.where(V < P["E"], 0.5 * P["k"] * (P["E"] - V) ** 2, 0)
        panels.append(panel("gamd-flat", "How the boost flattens the energy (total-potential term)",
                            "V* = V + ½k(E−V)² below the threshold E; relative ordering of energies is preserved",
                            [line((V - P["vmin"]) / 4.184, (V - P["vmin"]) / 4.184, "V (original)", MUTED, dash="dot"),
                             line((V - P["vmin"]) / 4.184, (Vb - P["vmin"]) / 4.184, "V + ΔV (boosted)", COLORS["gamd"], width=2.5)],
                            dict(xaxis=dict(title="V − Vmin (kcal/mol)"), yaxis=dict(title="energy felt (kcal/mol)"))))
    panels += [conv_panel("gamd", curves, COLORS["gamd"], "cumulant-reweighted PMF from growing slices of production"),
               hist2d_panel("gamd", d, z["n_hb"])]
    total = sum(float(zz["t_ps"][-1] - (zz["t_ps"][0] if s != "cmd" else 0)) for s, zz in parts.items()) / 1000
    F = cum
    method = dict(key="gamd", syncs=[sync_fe], stats=stats_list(simulated=f"{total:.1f} ns", **{"mean boost": f"{mu:.1f} kcal/mol", "ΔF 15→30 Å": f"{dF(F):.1f}" if dF(F) else "not reached"}),
                  trajectories=[dict(key=v, label=k, group="stage") for k, v in keys.items()], default_traj=keys.get("prod"),
                  panels=panels)
    return method, lin, dict(total_ns=total, dF=dF(F), F=F, cost=[(c_[1], dF(c_[2])) for c_ in curves], curves=[(c_[1], c_[2]) for c_ in curves])


def do_smd():
    files = {tag: sorted(glob.glob(str(RES / "smd" / f"pull_{tag}_*.npz"))) for tag in ("slow", "fast")}
    if not files["slow"]:
        return None
    rgrid = np.arange(17.0, 34.01, 0.25)

    def load(tag):
        Ws, ds, frs = [], [], []
        for f in files[tag]:
            z = np.load(f)
            Ws.append(np.interp(rgrid, z["r0"], z["W"], left=0.0)); ds.append(z)
        return np.array(Ws), ds
    out = {}
    for tag in ("slow", "fast"):
        if files[tag]:
            W, zs = load(tag)
            W = W - W[:, :1]
            out[tag] = (W, zs)

    def jar(W):
        return -KT * (logsumexp(-BETA * W, axis=0) - np.log(len(W)))
    panels = []
    spag = []
    for tag, col in (("slow", COLORS["smd"]), ("fast", MUTED)):
        if tag in out:
            W = out[tag][0]
            for i, w in enumerate(W):
                spag.append(line(rgrid, w, f"{tag} pulls", col, width=1, opacity=0.35, showlegend=(i == 0), legendgroup=tag))
    Fj = jar(out["slow"][0])
    off = 0.0
    if G.get("refF") is not None:
        off = np.interp(17.0, CENT, G["refF"])
    spag.append(line(rgrid, Fj, "Jarzynski (slow)", "#0b0b0b", width=2.5))
    panels.append(panel("smd-work", "Work along every pull", "each thin line is one pull; the Jarzynski average sits below the mean work",
                        spag, dict(xaxis=dict(title="spring position r0 (Å)"), yaxis=dict(title="W (kcal/mol)"))))
    est = [ref_trace()]
    for tag, dash in (("slow", None), ("fast", "dot")):
        if tag in out:
            W = out[tag][0]
            est.append(line(rgrid, jar(W) + off, f"Jarzynski ({tag})", COLORS["smd"], dash=dash, width=2.5 if tag == "slow" else 1.5))
            est.append(line(rgrid, W.mean(0) + off, f"mean work ({tag})", MUTED, dash=dash, width=1.5))
            est.append(line(rgrid, W.mean(0) - 0.5 * BETA * W.var(0) + off, f"cumulant ({tag})", COLORS["smd"], dash=dash, width=1, opacity=0.5))
    panels.append(panel("smd-est", "Estimators vs the reference", "mean work over-estimates by the dissipated work; faster pulls dissipate more",
                        est, pmf_layout(dict(xaxis=dict(title="d_ee / r0 (Å)", range=[3, 36]), yaxis=dict(title="F (kcal/mol)", range=[0, 40])))))
    hist = []
    for tag, col in (("slow", COLORS["smd"]), ("fast", MUTED)):
        if tag in out:
            wf = out[tag][0][:, np.argmin(np.abs(rgrid - 30))]
            hist.append(dict(type="histogram", x=clean(wf, 2), name=f"{tag} pulls", marker=dict(color=col), opacity=0.7, nbinsx=15))
    if G.get("refdF") is not None:
        hist.append(dict(type="scatter", mode="lines", x=[G["refdF"]] * 2, y=[0, 8], name="reference ΔF", line=dict(color=TEXT2, dash="dash", width=2)))
    panels.append(panel("smd-hist", "Work distribution at r0 = 30 Å", "Jarzynski is dominated by the rare low-work pulls in the left tail",
                        hist, dict(barmode="overlay", xaxis=dict(title="W (kcal/mol)"), yaxis=dict(title="pulls"))))
    # convergence vs number of pulls
    W = out["slow"][0]; j30 = np.argmin(np.abs(rgrid - 30)); j15 = 0
    Ns = np.arange(1, len(W) + 1)
    rng = np.random.default_rng(0)
    vals, errs = [], []
    for N in Ns:
        bs = [jar(W[rng.choice(len(W), N, replace=True)])[j30] for _ in range(100)]
        vals.append(jar(W[:N])[j30]); errs.append(np.std(bs))
    ns_per = (34 - 17) / 10.0
    curves = [(f"{N} pulls", N * ns_per, np.interp(CENT, rgrid, jar(W[:N]) + off, left=np.nan, right=np.nan)) for N in sorted({1, 2, 3, 4, 6, 8, 12, 16, 24, len(W)}) if N <= len(W)]
    offF = off
    panels.append(dF_panel("smd", Ns * ns_per, np.array(vals) + offF - (np.interp(15, CENT, G["refF"]) if G.get("refF") is not None else 0), COLORS["smd"], errs))
    panels.append(conv_panel("smd", curves, COLORS["smd"], "Jarzynski estimate with increasing numbers of slow pulls"))
    lin = lineage("smd")
    ensure(lin, "smd_seeds", "unbiased seed sampler", "sampler", ["equil"])
    for tag in ("slow", "fast"):
        for f in files[tag]:
            i = int(f[-6:-4]); ensure(lin, f"smd_{tag}_{i:02d}", f"{tag} pull #{i}", f"pull_{tag}", ["smd_seeds"])
    trajs = []
    for tag in ("slow", "fast"):
        if tag in out:
            order = np.argsort(out[tag][0][:, j30])
            for rank, i in enumerate([order[0], order[len(order) // 2], order[-1]]):
                z = out[tag][1][i]
                lab = ["lowest", "median", "highest"][rank] + " work"
                tk = write_traj(f"smd_{tag}_{i:02d}", z["frames"], (z["r0"] - z["r0"][0]) / float(z["speed"]), f"{tag} pull #{i} ({lab})",
                                labels=[f"r0 {r:.1f} Å, W {w:.1f} kcal/mol" for r, w in zip(z["r0"], z["W"])])
                trajs.append(dict(key=tk, label=f"{lab} #{i}", group=f"{tag} pulls ({float(z['speed']):g} Å/ns)"))
                set_traj(lin, f"smd_{tag}_{i:02d}", tk)
    members = []
    nmin = min(len(z["frames"]) for z in out["slow"][1])
    for i, z in enumerate(out["slow"][1][:16]):
        kk = write_traj(f"smd_grid{i:02d}", z["frames"][:nmin], ((z["r0"] - z["r0"][0]) / float(z["speed"]))[:nmin], f"slow pull #{i}",
                        labels=[f"#{i} · W {w:.1f} kcal/mol" for w in z["W"][:nmin]])
        members.append((kk, f"pull #{i}"))
    ens = ensemble("16 slow pulls side by side", "independent pulls (no exchange); each cell's label shows its accumulated work", members)
    Fcomp = np.interp(CENT, rgrid, Fj + off, left=np.nan, right=np.nan)
    total = sum(len(v) for v in files.values()) and (len(files["slow"]) * 1.8 + len(files["fast"]) * 0.18 + 1.6)
    method = dict(key="smd", ensemble=ens, stats=stats_list(pulls=f"{len(files['slow'])} slow + {len(files['fast'])} fast", simulated=f"{total:.0f} ns",
                                             **{"ΔF 15→30 Å": f"{vals[-1] + offF - (np.interp(15, CENT, G['refF']) if G.get('refF') is not None else 0):.1f}"}),
                  trajectories=trajs, default_traj=trajs[0]["key"], panels=panels)
    dFv = float(vals[-1] + offF - (np.interp(15, CENT, G["refF"]) if G.get("refF") is not None else 0))
    return method, lin, dict(total_ns=total, dF=dFv, F=Fcomp, cost=[(float(n), float(v + offF - (np.interp(15, CENT, G["refF"]) if G.get("refF") is not None else 0))) for n, v in zip(Ns * ns_per, vals)], curves=[(c[1], c[2]) for c in curves])


def do_we(name="we"):
    files = sorted(glob.glob(str(RES / name / "iter_*.npz")))
    if len(files) < 3:
        return None
    its = [np.load(f) for f in files]
    tau_ns = 0.010
    nw = np.array([len(z["weights"]) for z in its])
    # PMF from the second half of iterations (and convergence)
    def pmf_from(a, b):
        d = np.concatenate([z["d"].ravel() for z in its[a:b]])
        w = np.concatenate([np.repeat(z["weights"], z["d"].shape[1]) for z in its[a:b]])
        return pmf_hist(d, logw=np.log(np.maximum(w, 1e-300)))
    curves, cost = [], []
    for b in np.unique(np.geomspace(10, len(its), 10).astype(int)):
        F = pmf_from(b // 2, b); ns = nw[:b].sum() * tau_ns
        curves.append((f"iter {b}", ns, F)); cost.append((ns, dF(F)))
    F = curves[-1][2]
    sync_fe = fe_sync(name, [(b * tau_ns, pmf_from(b // 2, b)) for b in np.unique(np.linspace(6, len(its), 40).astype(int))],
                      "−kT ln P(bin) from the second half of iterations up to the frame shown")
    # bin population heatmap
    P = np.full((len(its), len(CENT)), np.nan)
    for i, z in enumerate(its):
        idx = np.digitize(z["d"][:, -1], EDGES) - 1
        for b in np.unique(idx):
            if 0 <= b < len(CENT):
                P[i, b] = np.log10(z["weights"][idx == b].sum())
    step = max(1, len(its) // 150)
    heat = dict(type="heatmap", x=clean(CENT), y=list(range(0, len(its), step)), z=[clean(r, 2) for r in P[::step]],
                zmin=-15, zmax=0, colorscale=[[i / 6, c] for i, c in enumerate(RAMP)], colorbar=dict(title="log₁₀ P", thickness=12),
                hovertemplate="iter %{y}<br>d_ee %{x:.1f} Å<br>log10 P %{z:.1f}<extra></extra>")
    # walker tree: x = iteration, y = end d_ee; segment from parent end to child end
    idmap = {}
    for i, z in enumerate(its):
        for k, wid in enumerate(z["ids"]):
            idmap[str(wid)] = (i, float(z["d"][k, -1]), float(z["weights"][k]))
    # Thinned for the browser: one segment per walker every `step` iterations, from its ancestor
    # `step` iterations back (a ~130k-segment WebGL plot made the movie player crawl).
    parent_of = {str(wid): str(par) for z in its for wid, par in zip(z["ids"], z["parents"])}
    n_seg = sum(len(z["ids"]) for z in its)
    step = max(1, int(np.ceil(n_seg / 15000)))
    segs = {c: ([], []) for c in range(4)}
    for i in range(step, len(its), step):
        z = its[i]; seen = set()
        for k, wid in enumerate(z["ids"]):
            anc = str(wid)
            for _ in range(step):
                anc = parent_of.get(anc)
                if anc is None:
                    break
            if anc is None or anc not in idmap:
                continue
            pi, pd, _ = idmap[anc]
            key_ = (anc, round(float(z["d"][k, -1]), 1))
            if key_ in seen:
                continue
            seen.add(key_)
            lw = np.log10(max(z["weights"][k], 1e-300))
            cls = 0 if lw < -12 else 1 if lw < -8 else 2 if lw < -4 else 3
            xs, ys = segs[cls]; xs += [pi, i, None]; ys += [round(pd, 2), round(float(z["d"][k, -1]), 2), None]
    tree = []
    for cls, lab in enumerate(["w < 1e-12", "1e-12 – 1e-8", "1e-8 – 1e-4", "w > 1e-4"]):
        xs, ys = segs[cls]
        tree.append(dict(type="scattergl", mode="lines", x=xs, y=ys, name=lab,
                         line=dict(color=RAMP[1 + 2 * cls] if cls < 3 else RAMP[6], width=0.6 + 0.5 * cls)))
    # ancestry trajectories
    last = its[-1]
    picks = {"furthest walker": int(np.argmax(last["d"][:, -1])), "heaviest walker": int(np.argmax(last["weights"])),
             "most compact walker": int(np.argmin(last["d"][:, -1]))}
    frames_by_id = {}
    trajs = []
    lin = lineage(name, collapse=lambda nodes: _collapse_we(nodes, name))
    for lab, k in picks.items():
        chain = []; wid = str(last["ids"][k])
        while wid in idmap:
            chain.append(wid); i, _, _ = idmap[wid]
            z = its[i]; kk = list(map(str, z["ids"])).index(wid)
            wid = str(z["parents"][kk])
        chain = chain[::-1]
        fr, tt, labs = [], [], []
        for wid in chain:
            i, dd, ww = idmap[wid]; z = its[i]; kk = list(map(str, z["ids"])).index(wid)
            fr.append(z["end_frames"][kk]); tt.append((i + 1) * tau_ns); labs.append(f"iter {i}, weight {ww:.1e}")
        xs = [idmap[w][0] for w in chain]; ys = [idmap[w][1] for w in chain]
        tree.append(dict(type="scatter", mode="lines", x=xs, y=ys, name=lab, line=dict(color="#0b0b0b" if lab == "furthest walker" else COLORS["smd"] if lab == "heaviest walker" else COLORS["md"], width=2.5)))
        key = f"{name}_" + lab.split()[0]
        trajs.append(dict(key=write_traj(key, fr, tt, f"WE ancestry: {lab}", labels=labs), label=lab, group="ancestral lines (final iteration)"))
    lin["nodes"][-1]["traj"] = trajs[0]["key"]
    total = nw.sum() * tau_ns
    method = dict(key=name, syncs=[sync_fe], stats=stats_list(iterations=str(len(its)), walkers=f"{nw[-1]} now", simulated=f"{total:.0f} ns",
                                            **{"ΔF 15→30 Å": f"{dF(F):.1f}" if dF(F) else "not yet"}),
                  trajectories=trajs, default_traj=trajs[0]["key"],
                  panels=[panel(f"{name}-tree", "Walker family tree", f"each line joins a walker to its ancestor {step * 10} ps earlier (thinned to every {step} iterations); splits fan out, merges end lines. Bold = ancestry of 3 final walkers",
                                tree, dict(xaxis=dict(title="iteration"), yaxis=dict(title="d_ee at end of segment (Å)")), wide=True),
                          panel(f"{name}-pop", "Probability in each bin, per iteration", "WE resolves probabilities of 10⁻¹⁰ and below without bias",
                                [heat], dict(xaxis=dict(title="d_ee (Å)"), yaxis=dict(title="iteration"))),
                          conv_panel(name, curves, COLORS[name], "F = -kT ln P(bin), averaged over the second half of iterations so far"),
                          dF_panel(name, [c[0] for c in cost], [c[1] for c in cost], COLORS[name]),
                          panel(f"{name}-n", "Walkers per iteration", "splitting keeps 4 walkers in every occupied bin",
                                [dict(type="bar", x=list(range(len(nw))), y=nw.tolist(), marker=dict(color=COLORS[name]), name="walkers")],
                                dict(xaxis=dict(title="iteration"), yaxis=dict(title="walkers"), bargap=0))])
    return method, lin, dict(total_ns=total, dF=dF(F), F=F, cost=cost, curves=[(c[1], c[2]) for c in curves])


def _collapse_we(nodes, name="we"):
    it = sorted([n for n in nodes.values() if n["kind"] == "iteration"], key=lambda n: n["id"])
    keep = {k: v for k, v in nodes.items() if v["kind"] != "iteration"}
    if it:
        import re
        strip_n = [int(re.search(r": (\d+) walkers", n["label"]).group(1)) for n in it]
        dmax = [None] * len(it)
        f = RES / name / "lineage.jsonl"
        recs = {json.loads(l)["id"]: json.loads(l) for l in f.read_text().splitlines()}
        dmax = [round(recs[n["id"]]["meta"].get("d_max", 0), 1) for n in it]
        keep[f"{name}_iters"] = dict(id=f"{name}_iters", label=f"{len(it)} WE iterations (split/merge every 10 ps)", kind="iterations",
                                t_ps=sum(n["t_ps"] for n in it), traj=None, strip=dict(n=strip_n, dmax=dmax), _parents=["equil"])
    return keep


def do_remd():
    f = RES / "remd" / "remd.npz"
    if not f.exists():
        return None
    z = np.load(f)
    temps, rs, u, drep = z["temps"], z["rep_state"], z["u_rs"], z["d_rep"]
    ps = float(z["ps_per_iter"]); n_it, R = rs.shape
    # d at each state: invert replica->state
    d_state = np.empty_like(drep)
    for i in range(n_it):
        d_state[i, rs[i]] = drep[i]
    skip = max(1, n_it // 5000)

    def mbar_pmf(m):
        uu = u[:m:skip]; dd = drep[:m:skip]
        u_kn = uu.transpose(2, 0, 1).reshape(R, -1)          # [state, (iter, replica)]
        x = dd.reshape(-1); N_k = np.full(R, uu.shape[0])
        logw, mb = mbar_logw_target(u_kn, N_k, u_kn[0])
        return pmf_hist(x, logw=logw), mb
    curves = []
    for frac in FRACS:
        m = max(20, int(n_it * frac))
        F, mb = mbar_pmf(m); curves.append((f"{m*ps/1000:.1f} ns/replica", m * ps * R / 1000, F))
    F = curves[-1][2]
    snaps = [(m * ps / 1000, mbar_pmf(m)[0]) for m in np.unique(np.linspace(max(20, n_it // 25), n_it, 25).astype(int))]
    sync_fe = fe_sync("remd", snaps, "MBAR at 300 K using all replicas up to the frame shown")
    F300 = pmf_hist(d_state[:, 0])
    # observed neighbour swaps per iteration, counted from the replica->state record
    moves = np.zeros(R - 1)
    for i in range(1, n_it):
        for r in range(R):
            a_, b_ = rs[i - 1, r], rs[i, r]
            if abs(int(a_) - int(b_)) == 1:
                moves[min(a_, b_)] += 0.5
    nb = moves / max(1, n_it - 1)
    t = np.arange(n_it) * ps / 1000
    s = slice(None, None, max(1, n_it // 1500))
    braid = []
    for r in range(R):
        hl = r in (0, R // 2, R - 1)
        braid.append(dict(type="scattergl", mode="lines", x=clean(t[s], 3), y=[float(temps[k]) for k in rs[s, r]],
                          name=f"replica {r}", line=dict(width=2 if hl else 0.7, color=[COLORS["remd"], COLORS["metad"], COLORS["smd"]][[0, R // 2, R - 1].index(r)] if hl else "#d6d5d0"),
                          showlegend=hl, line_shape="hv"))
    th = []
    for k in range(R):
        h, _ = np.histogram(d_state[:, k], EDGES, density=True)
        th.append(line(CENT, h, f"{temps[k]:.0f} K", RAMP[min(6, int(k * 7 / R))], width=1.5 if k else 2.5, showlegend=k in (0, R - 1)))
    ov = mb.compute_overlap()["matrix"]
    fr, fi = z["frames"], z["frame_it"]
    lin = lineage("remd")
    for r in range(R):
        ensure(lin, f"remd_rep{r:02d}", f"replica {r}", "replica", ["equil"], n_it * ps)
    trajs = []
    rs_f = rs[fi]
    for k in (0, R - 1):
        sel = [np.where(rs_f[j] == k)[0][0] for j in range(len(fi))]
        key = write_traj(f"remd_T{int(temps[k])}", fr[np.arange(len(fi)), sel], fi * ps / 1000, f"{temps[k]:.0f} K (demuxed: whichever replica holds it)",
                         labels=[f"{temps[k]:.0f} K ← replica {r}" for r in sel])
        trajs.append(dict(key=key, label=f"{temps[k]:.0f} K", group="by temperature"))
    for r in (0, R // 2):
        key = write_traj(f"remd_rep{r:02d}", fr[:, r], fi * ps / 1000, f"replica {r} (wanders in temperature)",
                         labels=[f"replica {r} at {temps[k]:.0f} K" for k in rs_f[:, r]])
        trajs.append(dict(key=key, label=f"replica {r}", group="by replica"))
        set_traj(lin, f"remd_rep{r:02d}", key)
    # grid cells are TEMPERATURES; each shows whichever replica currently holds that temperature
    sel = disp_sel(len(fi))
    holder = np.argsort(rs_f, axis=1)            # holder[j, k] = replica in state k at saved frame j
    members = []
    for k in range(R):
        kk = write_traj(f"remd_grid{k:02d}", fr[np.arange(len(fi)), holder[:, k]], fi * ps / 1000, f"{temps[k]:.0f} K",
                        labels=[f"{temps[k]:.0f} K · replica {r}" for r in holder[:, k]])
        members.append((kk, f"{temps[k]:.0f} K"))
    hd = holder[sel]
    events = [[]]
    for f in range(1, len(sel)):
        prev, now = hd[f - 1], hd[f]
        events.append([[a, b] for a in range(R) for b in range(a + 1, R)
                       if prev[a] != prev[b] and now[a] == prev[b] and now[b] == prev[a]])
    tint = [[round(k / (R - 1), 3)] * len(sel) for k in range(R)]
    ens = ensemble("All 10 temperatures at once", "each cell is one temperature (light = 300 K, dark = 700 K) showing whichever replica holds it; "
                   "two cells flash in a matching colour when their replicas swap", members, events, tint, exchange=True)
    total = n_it * ps * R / 1000
    method = dict(key="remd", syncs=[sync_fe], ensemble=ens, stats=stats_list(replicas=f"{R} ({temps[0]:.0f}–{temps[-1]:.0f} K)", simulated=f"{total:.0f} ns",
                                              **{"swaps / iteration / pair": f"{np.mean(nb):.2f}"}, **{"ΔF 15→30 Å": f"{dF(F):.1f}" if dF(F) else "not reached"}),
                  trajectories=trajs, default_traj=trajs[0]["key"],
                  panels=[panel("remd-braid", "Replicas random-walk in temperature", "3 replicas highlighted; each swap is a Metropolis move between neighbours",
                                braid, dict(xaxis=dict(title="t (ns)"), yaxis=dict(title="temperature (K)")), wide=True),
                          panel("remd-T", "d_ee distribution at each temperature", "hot replicas unfold; 300 K stays helical/compact",
                                th, dict(xaxis=dict(title="d_ee (Å)", range=[3, 36]), yaxis=dict(title="density"))),
                          panel("remd-mbar", "300 K PMF: 300 K replica alone vs MBAR over all temperatures", "MBAR borrows information from hot replicas",
                                [ref_trace(), line(CENT, F300, "300 K samples only", MUTED, dash="dot"), line(CENT, F, "MBAR (all T)", COLORS["remd"], width=2.5)],
                                pmf_layout(dict(yaxis=dict(title="F (kcal/mol)", range=[0, 30])))),
                          panel("remd-acc", "Neighbour swaps per iteration", "counted from which replica holds each temperature; no zero = no bottleneck", [dict(type="bar", x=[f"{temps[i]:.0f}–{temps[i+1]:.0f}" for i in range(R - 1)], y=clean(nb),
                                marker=dict(color=COLORS["remd"]), name="acceptance")], dict(yaxis=dict(title="swaps per iteration", rangemode="tozero"), xaxis=dict(title="pair (K)"))),
                          panel("remd-ov", "MBAR overlap between temperatures", "", [dict(type="heatmap", z=[clean(r, 3) for r in ov], x=clean(temps, 0), y=clean(temps, 0),
                                colorscale=[[0, "#ffffff"], [1, RAMP[5]]], zmin=0, zmax=0.5, colorbar=dict(thickness=12))],
                                dict(xaxis=dict(title="T (K)"), yaxis=dict(title="T (K)"))),
                          conv_panel("remd", curves, COLORS["remd"], "MBAR at 300 K from growing slices")])
    return method, lin, dict(total_ns=total, dF=dF(F), F=F, cost=[(c[1], dF(c[2])) for c in curves], curves=[(c[1], c[2]) for c in curves])


def do_sams(name="sams"):
    f = RES / name / "sams.npz"
    if not f.exists():
        return None
    z = np.load(f)
    labels, st, logZ, stage, d, u = z["labels"], z["state"], z["logZ"], z["stage"], z["d"], z["u"]
    k = float(z["k"]); ps = float(z["ps_per_iter"]); n = len(st); K = len(labels)
    temp = str(z["mode"]) == "temperature"
    t = np.arange(n) * ps / 1000
    if temp:
        uk_all = u.T; u0 = u[:, 0]                   # u[n, k] = beta_k U(x_n); target = 300 K
        unit, what, lab = "K", "temperature", lambda x: f"{x:.0f} K"
    else:
        bias = BETA * 0.5 * k * (d[:, None] - labels[None, :]) ** 2          # [n, K]
        u0 = u[np.arange(n), st] - bias[np.arange(n), st]                    # unbiased reduced potential
        uk_all = (u0[:, None] + bias).T
        unit, what, lab = "Å", "umbrella", lambda x: f"r0={x:.1f} Å"

    def pmf(m, ret_mbar=False):
        N_k = np.bincount(st[:m], minlength=K).astype(float)
        logw, mb = mbar_logw_target(uk_all[:, :m], N_k, u0[:m])
        return (pmf_hist(d[:m], logw=logw), mb) if ret_mbar else pmf_hist(d[:m], logw=logw)
    curves = []
    for frac in FRACS:
        m = max(50, int(n * frac)); curves.append((f"{m*ps/1000:.1f} ns", m * ps / 1000, pmf(m)))
    F = curves[-1][2]
    sync_fe = fe_sync(name, [(m * ps / 1000, pmf(m)) for m in np.unique(np.linspace(max(50, n // 25), n, 25).astype(int))],
                      "MBAR over the walker's history up to the frame shown")
    s = slice(None, None, max(1, n // 2000))
    fk = -logZ * KT
    fk = fk - fk[:, :1]
    lz = []
    for j, kk in enumerate(np.linspace(0, K - 1, 7).astype(int)):
        lz.append(line(t[s], fk[s, kk], f"state {lab(labels[kk])}", RAMP[j], width=1.5))
    s1 = int(np.argmax(stage == 1)) if (stage == 1).any() else None
    shapes = [dict(type="line", x0=t[s1], x1=t[s1], y0=0, y1=1, yref="paper", line=dict(color=TEXT2, dash="dash", width=1.5))] if s1 else []
    ann = [dict(x=t[s1], y=1, yref="paper", text="stage 1 → 2", showarrow=False, xanchor="left", font=dict(color=TEXT2))] if s1 else []
    visits = [dict(type="bar", x=clean(labels, 2), y=np.bincount(st[:s1 or n], minlength=K).tolist(), name="stage 1", marker=dict(color=MUTED))]
    if s1:
        visits.append(dict(type="bar", x=clean(labels, 2), y=np.bincount(st[s1:], minlength=K).tolist(), name="stage 2", marker=dict(color=COLORS[name])))
    fin = fk[-1] - fk[-1].min()
    fr, fi = z["frames"], z["frame_it"]
    key = write_traj(name, fr, fi * ps / 1000, "SAMS walker", labels=[f"in {what} {lab(labels[st[i]])}" for i in fi])
    lin = lineage(name)
    ensure(lin, f"{name}_stage1", "SAMS stage 1 (burn-in)", "stage1", ["equil"], (s1 or n) * ps)
    if s1:
        ensure(lin, f"{name}_stage2", "SAMS stage 2 (1/t updates)", "stage2", [f"{name}_stage1"], (n - s1) * ps)
    set_traj(lin, f"{name}_stage1", key); set_traj(lin, f"{name}_stage2", key)
    if temp:
        _, mb = pmf(n, ret_mbar=True)
        fk_mbar = (mb.f_k - mb.f_k[0]) * KT
        fk_panel = panel(f"{name}-fk", "Online estimate vs MBAR: free energy of each temperature state",
                         "dimensionless f_k = −ln Z_k (×kT at 300 K); SAMS learns these on the fly, MBAR recomputes them afterwards",
                         [line(labels, -(logZ[-1] - logZ[-1][0]) * KT, "SAMS f_k (online)", COLORS[name], dash="dot", width=2, mode="lines+markers"),
                          line(labels, fk_mbar, "MBAR f_k", TEXT2, width=2, mode="lines+markers")],
                         dict(xaxis=dict(title="temperature (K)"), yaxis=dict(title="f_k (kcal/mol-equivalent)")))
        pmf_panel = panel(f"{name}-pmf", "300 K PMF: MBAR over all temperatures", "like T-REMD, the extended state is out of reach",
                          [ref_trace(), line(CENT, F, "MBAR at 300 K", COLORS[name], width=2.5)], pmf_layout(dict(yaxis=dict(title="F (kcal/mol)", range=[0, 30]))))
        th = []
        for kk in range(K):
            h, _ = np.histogram(d[st == kk], EDGES, density=True)
            th.append(line(CENT, h, f"{labels[kk]:.0f} K", RAMP[min(6, int(kk * 7 / K))], width=1.5 if kk else 2.5, showlegend=kk in (0, K - 1)))
        extra = [fk_panel, pmf_panel, panel(f"{name}-T", "d_ee distribution at each temperature", "", th,
                                            dict(xaxis=dict(title="d_ee (Å)", range=[3, 36]), yaxis=dict(title="density")))]
    else:
        extra = [panel(f"{name}-fk", "SAMS online estimate vs MBAR", "f_k is F smoothed by the umbrella; MBAR unsmooths it",
                       [ref_trace(), line(labels, fin, "SAMS f_k (online)", COLORS[name], dash="dot", width=2, mode="lines+markers"),
                        line(CENT, F, "MBAR", COLORS[name], width=2.5)], pmf_layout(dict(yaxis=dict(title="F (kcal/mol)", range=[0, 30]))))]
    total = n * ps / 1000
    method = dict(key=name, syncs=[sync_fe], stats=stats_list(states=str(K), simulated=f"{total:.1f} ns", stage=("2 since " + f"{t[s1]:.1f} ns") if s1 else "1 (burn-in)",
                                              **{"ΔF 15→30 Å": f"{dF(F):.1f}" if dF(F) else "–"}),
                  trajectories=[dict(key=key, label="walker", group="trajectory")], default_traj=key,
                  panels=[panel(f"{name}-state", f"The walker jumps between {what} states", f"state visited each 1 ps iteration (y = {what})",
                                [dict(type="scattergl", mode="markers", x=clean(t[s], 3), y=clean(labels[st[s]], 2), marker=dict(size=3, color=COLORS[name]), name="state")]
                                + ([] if temp else [dict(type="scattergl", mode="lines", x=clean(t[s], 3), y=clean(d[s], 2), line=dict(width=1, color=TEXT2), name="d_ee", opacity=0.6)]),
                                dict(xaxis=dict(title="t (ns)"), yaxis=dict(title=f"{what} ({unit})"), shapes=shapes, annotations=ann), wide=True),
                          panel(f"{name}-logz", f"Online free energy of each {what} state", "−kT log Z_k relative to the first state; flattens as SAMS converges",
                                lz, dict(xaxis=dict(title="t (ns)"), yaxis=dict(title="f_k (kcal/mol)"), shapes=shapes)),
                          panel(f"{name}-visits", f"Visits per {what} state", "SAMS drives visits toward uniform", visits,
                                dict(barmode="stack", xaxis=dict(title=f"{what} ({unit})"), yaxis=dict(title="iterations")))]
                         + extra + [conv_panel(name, curves, COLORS[name], "MBAR on growing slices")])
    return method, lin, dict(total_ns=total, dF=dF(F), F=F, cost=[(c[1], dF(c[2])) for c in curves], curves=[(c[1], c[2]) for c in curves])


def build_intro():
    """Three representative structures (helix / extended / compact) + facts + annotated reference PMF."""
    picks = []
    eq = np.load(RES / "build" / "equil.npz")
    picks.append(("helix", "α-helix (starting structure)", eq["frames"][-1:]))
    for label, target in (("extended", 33.5), ("compact", 5.5)):
        best = None
        for f in glob.glob(str(RES / "reference" / "win_*.npz")):
            z = np.load(f)
            j = int(np.argmin(np.abs(z["d_ee"] - target)))
            if best is None or abs(z["d_ee"][j] - target) < best[0]:
                best = (abs(z["d_ee"][j] - target), z["frames"][j:j + 1])
        if best is not None:
            picks.append((label, "fully extended" if label == "extended" else "compact, folded over", best[1]))
    structs = []
    for key, label, fr in picks:
        tk = write_traj(f"intro_{key}", fr, [0.0], label)
        meta = json.loads((TRAJ / f"{tk}.json").read_text())
        structs.append(dict(key=key, label=label, traj=tk, d_ee=meta["d_ee"][0], n_hb=meta["n_hb"][0]))
    F = G.get("refF")
    facts = [
        "<b>System.</b> Ace-(Ala)<sub>10</sub>-Nme: ten alanines with capped ends, 112 atoms, in vacuum. AMBER ff14SB, 300 K, Langevin dynamics, 2 fs steps. "
        "Simulated with OpenMM 8.6.1 (CPU platform), openmmtools 0.26.0 and pymbar 4.2.0.",
        "<b>Why vacuum.</b> With no water the system is tiny, so it is very fast to simulate (≈ 830 ns/day per CPU core), which makes it a good demo of enhanced sampling. "
        "But its specific behavior differs from the solvated peptide: water competes for the backbone H-bonds, so in solution the helix is far less stable and the free-energy profile much flatter.",
        "<b>The coordinate.</b> d<sub>ee</sub> is the distance from the ACE carbonyl carbon to the NME nitrogen, so it spans all ten residues. "
        "An α-helix rises 1.5 Å per residue (d<sub>ee</sub> ≈ 15 Å); a fully stretched chain gives ≈ 3.4 Å per residue (≈ 33 Å).",
        "<b>A classic benchmark.</b> Park, Khalili-Araghi, Tajkhorshid &amp; Schulten (J. Chem. Phys. 2003) used this exact helix→coil stretch "
        "in vacuum to test free energies from steered MD and Jarzynski's equality.",
        "<b>Why the helix is so stable here.</b> In vacuum no water competes for the backbone's i→i+4 H-bonds (up to 8 here), so pulling the helix "
        "apart means breaking them one by one." + (f" Our reference puts F(30 Å) − F(15 Å) at ≈ {G['refdF']:.0f} kcal/mol." if G.get("refdF") else ""),
        "<b>A third state.</b> In this vacuum model the chain can also fold back on itself into compact states (d<sub>ee</sub> < 8 Å) with roughly the helix's free energy. This is a prediction of the model, not an established structure.",
        "<b>What makes it a good test.</b> d<sub>ee</sub> is an intuitive slow coordinate, but it hides a second barrier (breaking H-bonds) that it cannot see directly. "
        "Methods that bias or bin on d<sub>ee</sub> alone feel that hidden barrier; methods that need no coordinate do not, but pay in other ways.",
    ]
    data = []
    if F is not None:
        data.append(line(CENT, F, "reference PMF (umbrella sampling + MBAR)", TEXT2, width=2.5))
        for s in structs:
            y = float(np.interp(s["d_ee"], CENT[np.isfinite(F)], F[np.isfinite(F)]))
            data.append(dict(type="scatter", mode="markers+text", x=[s["d_ee"]], y=[y], text=[s["label"]], textposition="top center",
                             marker=dict(size=12, color="#0b0b0b"), showlegend=False, textfont=dict(color=TEXT2, size=13)))
    pmf = panel("intro-pmf", "Free energy along d_ee", "reference: 42 umbrella windows + MBAR", data,
                pmf_layout(dict(yaxis=dict(title="F (kcal/mol)", range=[0, 28]), showlegend=False)), wide=True)
    return dict(title="The system: deca-alanine", d_pair=list(G["d_pair"]), structures=structs, facts=facts, panels=[pmf])


LO_M, HI_M = 5.0, 30.0   # region the convergence metrics are scored on


def pmf_error(F):
    """RMSE (kcal/mol) vs reference over bins both have sampled in [5, 30] A, best constant offset removed."""
    R = G.get("refF")
    if R is None or F is None:
        return None
    m = np.isfinite(F) & np.isfinite(R) & (CENT >= LO_M) & (CENT <= HI_M) & (F < 40)
    if m.sum() < 3:
        return None
    dlt = F[m] - R[m]
    return float(np.sqrt(np.mean((dlt - dlt.mean()) ** 2)))


def coverage(F):
    """Fraction of the [5, 30] A range where the method has an estimate at all."""
    m = (CENT >= LO_M) & (CENT <= HI_M)
    return float(np.mean(np.isfinite(F[m]) & (F[m] < 40)))


def metric_traces(key, curves, what):
    xs, ys = [], []
    for ns, F in curves:
        v = pmf_error(F) if what == "err" else coverage(F)
        if v is not None and ns and ns > 0:
            xs.append(float(ns)); ys.append(v if what == "err" else 100 * v)
    return dict(type="scatter", mode="lines+markers", x=clean(xs, 4), y=clean(ys, 3), name=NAMES[key],
                line=dict(color=COLORS[key], width=2, **({"dash": "dash"} if key == "samsT" else {})),
                marker=dict(size=7, color=COLORS[key]))


ERR_HINT = f"RMSE of F(d_ee) against the reference over {LO_M:.0f}–{HI_M:.0f} Å, only where both have data (best constant offset removed)"
COV_HINT = f"share of {LO_M:.0f}–{HI_M:.0f} Å where the method has any estimate; low error with low coverage means it only saw the basins"


def method_metric_panels(key, curves):
    return [panel(f"{key}-err", "Error vs the reference as sampling accumulates", ERR_HINT, [metric_traces(key, curves, "err")],
                  dict(xaxis=dict(title="simulated ns (all copies)", type="log"), yaxis=dict(title="PMF RMSE (kcal/mol)", rangemode="tozero"), showlegend=False)),
            panel(f"{key}-cov", "How much of d_ee it has reached", COV_HINT, [metric_traces(key, curves, "cov")],
                  dict(xaxis=dict(title="simulated ns (all copies)", type="log"), yaxis=dict(title="coverage of 5–30 Å (%)", range=[0, 105]), showlegend=False))]


# ----------------------------------------------------------------- main
def main():
    prmtop = load_prmtop()
    G["d_pair"], G["hb"] = cv_atoms(prmtop.topology)
    G["bb"] = [a.index for a in prmtop.topology.atoms() if a.name in ("N", "CA", "C")]
    import openmm.app as app
    pdb = app.PDBFile(str(RES / "build" / "helix.pdb"))
    G["ref"] = np.array(pdb.getPositions(asNumpy=True)._value * 10, dtype=np.float32)
    DATA.mkdir(parents=True, exist_ok=True); TRAJ.mkdir(parents=True, exist_ok=True)
    (DATA / "topology.pdb").write_text((RES / "build" / "helix.pdb").read_text())

    results = {}
    for key, fn in [("reference", do_reference), ("md", do_md), ("metad", do_metad), ("gamd", do_gamd),
                    ("remd", do_remd), ("sams", do_sams), ("samsT", lambda: do_sams("samsT")), ("we", do_we), ("we2d", lambda: do_we("we2d")), ("smd", do_smd)]:
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:  # keep going: partial data is expected mid-run
            import traceback; traceback.print_exc()
            print(f"{key}: FAILED {e}"); r = None
        if r is None:
            print(f"{key}: no data yet"); continue
        method, lin, summ = r
        method.update(name=NAMES[key], color=COLORS[key])
        if key != "reference" and summ.get("curves"):
            method["panels"] += method_metric_panels(key, summ["curves"])
        jdump(DATA / f"{key}.json", method)
        results[key] = (lin, summ)
        print(f"{key}: ok ({time.time()-t0:.1f}s) total {summ['total_ns']:.1f} ns dF {summ['dF']}")

    methods, lineages = [], {}
    for key in ORDER:
        lin, summ = results.get(key, (dict(nodes=[], edges=[]), dict(total_ns=0, dF=None)))
        methods.append(dict(key=key, name=NAMES[key], color=COLORS[key], blurb=BLURB[key], fe_how=FE_HOW[key],
                            total_ns=round(summ["total_ns"], 1), dF=summ["dF"], dF_err=None, needs_cv=NEEDS_CV[key],
                            available=key in results))
        lineages[key] = lin
    # comparison
    comp = [ref_trace()]
    for key in ORDER:
        if key in results and key != "reference":
            comp.append(line(CENT, results[key][1]["F"], NAMES[key], COLORS[key], width=2, dash="dash" if key == "samsT" else None))
    cost = []
    for key in ORDER:
        if key in results and key != "reference":
            c = [(x, y) for x, y in results[key][1]["cost"] if y is not None]
            if c:
                cost.append(dict(type="scatter", mode="lines+markers", x=[x for x, _ in c], y=[y for _, y in c], name=NAMES[key],
                                 line=dict(color=COLORS[key], width=2), marker=dict(size=8, color=COLORS[key])))
    if G.get("refdF") is not None:
        cost.append(dict(type="scatter", mode="lines", x=[0, max([max(t["x"]) for t in cost] + [1])], y=[G["refdF"]] * 2,
                         name="reference", line=dict(color=TEXT2, dash="dash", width=2)))
    err = [metric_traces(k, results[k][1]["curves"], "err") for k in ORDER if k in results and k != "reference" and results[k][1].get("curves")]
    cov = [metric_traces(k, results[k][1]["curves"], "cov") for k in ORDER if k in results and k != "reference" and results[k][1].get("curves")]
    bars = [dict(type="bar", x=[NAMES[k] for k in ORDER if k in results], y=[results[k][1]["total_ns"] for k in ORDER if k in results],
                 marker=dict(color=[COLORS[k] for k in ORDER if k in results]), name="simulated ns")]
    timeline = dict(x=clean(CENT, 2), ref=clean(G["refF"], 2) if G.get("refF") is not None else None, methods=[])
    for key in ORDER:
        if key in results and key != "reference" and results[key][1].get("curves"):
            cv = sorted([(float(n_), F_) for n_, F_ in results[key][1]["curves"] if n_ and n_ > 0 and np.isfinite(F_).any()], key=lambda c: c[0])
            timeline["methods"].append(dict(key=key, name=NAMES[key], color=COLORS[key], dash="dash" if key == "samsT" else None,
                                            ns=[round(n_, 4) for n_, _ in cv], F=[clean(F_, 2) for _, F_ in cv]))
    allns = [n_ for m_ in timeline["methods"] for n_ in m_["ns"]]
    timeline.update(ns_min=min(allns) if allns else 0.1, ns_max=max(allns) if allns else 1)
    comparison = dict(timeline=timeline, panels=[
        panel("cmp-pmf", "Every method's free-energy profile", "dashed = reference (umbrella sampling + MBAR)", comp,
              pmf_layout(dict(yaxis=dict(title="F (kcal/mol)", range=[0, 30]))), wide=True),
        panel("cmp-err", "Convergence: PMF error vs simulation cost", ERR_HINT, err,
              dict(xaxis=dict(title="simulated ns (all copies)", type="log"), yaxis=dict(title="PMF RMSE (kcal/mol)", rangemode="tozero")), wide=True),
        panel("cmp-cov", "Convergence: coverage of d_ee vs simulation cost", COV_HINT, cov,
              dict(xaxis=dict(title="simulated ns (all copies)", type="log"), yaxis=dict(title="coverage of 5–30 Å (%)", range=[0, 105])), wide=True),
        panel("cmp-cost", "ΔF(15 → 30 Å) vs simulation cost", "only methods that reach 30 Å have a value; dashed = reference", cost,
              dict(xaxis=dict(title="simulated ns (all copies)", type="log"), yaxis=dict(title="ΔF (kcal/mol)"))),
        panel("cmp-ns", "Total simulated time", "all walkers / replicas / pulls counted", bars,
              dict(yaxis=dict(title="ns"), showlegend=False))])
    intro = build_intro()
    try:
        import estimators
        est = estimators.build(RES, BETA, EDGES, line, panel, pmf_layout, clean, TEXT2)
    except Exception:
        import traceback; traceback.print_exc(); est = None
    jdump(DATA / "index.json", dict(intro=intro, estimators=est, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                                    system=dict(name="Ace-(Ala)10-Nme, vacuum, ff14SB, 300 K", n_atoms=len(G["ref"]),
                                                blurb="Ten enhanced-sampling methods on one small peptide. Deca-alanine in vacuum: a stable α-helix (d_ee ≈ 15 Å) that can unravel to an extended chain (≈ 33 Å) or fold over into compact states. Plain MD stays near the helix."),
                                    methods=methods, lineage=lineages, comparison=comparison))
    print("wrote", DATA)


if __name__ == "__main__":
    main()
