"""Free-energy estimators compared on the same umbrella-sampling windows.

MBAR, WHAM, umbrella integration (Kästner & Thiel 2005) and naive histogram
stitching, all applied to results/reference/win_*.npz (restraint 0.5 k (d - r0)^2).

Three comparisons:
  * error vs data per window, against full-data MBAR (the natural "truth" here);
  * spread across 4 independent blocks of each window (reference-free statistical error);
  * robustness when windows are dropped (less overlap between neighbours).
"""
import glob

import numpy as np
from scipy.special import logsumexp

EST_COLORS = {"MBAR": "#2a78d6", "WHAM": "#1baf7a", "WHAM, 2 Å bins": "#e87ba4", "umbrella integration": "#eb6834",
              "histogram stitching": "#4a3aa7"}


def load_windows(res):
    ws = [np.load(f) for f in sorted(glob.glob(str(res / "reference" / "win_*.npz")))]
    ws.sort(key=lambda w: float(w["r0"]))
    return [np.asarray(w["d_all"]) for w in ws], np.array([float(w["r0"]) for w in ws]), float(ws[0]["k"])


def _finish(F):
    F = np.asarray(F, dtype=float)
    return F - np.nanmin(F) if np.isfinite(F).any() else F


def mbar(xs, r0, k, beta, edges):
    from pymbar import MBAR
    x = np.concatenate(xs); N = np.array([len(v) for v in xs], dtype=float)
    u = beta * 0.5 * k * (x[None, :] - r0[:, None]) ** 2
    f = MBAR(u, N, verbose=False, solver_protocol="robust").f_k
    logw = -logsumexp(np.log(N)[:, None] + f[:, None] - u, axis=0)
    idx = np.digitize(x, edges) - 1
    F = np.full(len(edges) - 1, np.nan)
    for b in np.unique(idx):
        if 0 <= b < len(F):
            F[b] = -logsumexp(logw[idx == b])
    return _finish(F / beta)


def wham(xs, r0, k, beta, edges, tol=1e-8, maxit=20000):
    c = 0.5 * (edges[1:] + edges[:-1])
    n = np.array([np.histogram(v, edges)[0] for v in xs], dtype=float)      # [K, B]
    N = n.sum(1); tot = n.sum(0)
    u = beta * 0.5 * k * (c[None, :] - r0[:, None]) ** 2                     # [K, B]
    f = np.zeros(len(xs))
    for _ in range(maxit):
        with np.errstate(divide="ignore"):
            logP = np.log(tot) - logsumexp(np.log(N)[:, None] + f[:, None] - u, axis=0)
        f_new = -logsumexp(logP[None, :] - u, axis=1)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new; break
        f = f_new
    F = -logP / beta
    F[~np.isfinite(F)] = np.nan
    return _finish(F)


def wham_coarse(xs, r0, k, beta, edges, width=2.0):
    """WHAM on 2 A bins (the bias is assumed constant across each bin), interpolated back to the fine grid."""
    ce = np.arange(edges[0], edges[-1] + 1e-9, width)
    Fc = wham(xs, r0, k, beta, ce)
    cc = 0.5 * (ce[1:] + ce[:-1]); c = 0.5 * (edges[1:] + edges[:-1])
    ok = np.isfinite(Fc)
    F = np.interp(c, cc[ok], Fc[ok], left=np.nan, right=np.nan)
    return _finish(F)


def umbrella_integration(xs, r0, k, beta, edges):
    """dA/dxi = sum_i p_i(xi) [kT (xi - mu_i)/s_i^2 - k (xi - r0_i)] / sum_i p_i(xi), integrated."""
    kT = 1 / beta
    mu = np.array([v.mean() for v in xs]); s2 = np.array([v.var() for v in xs]); N = np.array([len(v) for v in xs])
    g = np.linspace(edges[0], edges[-1], 1321)
    p = N[:, None] * np.exp(-0.5 * (g[None, :] - mu[:, None]) ** 2 / s2[:, None]) / np.sqrt(2 * np.pi * s2[:, None])
    dA = (kT * (g[None, :] - mu[:, None]) / s2[:, None] - k * (g[None, :] - r0[:, None]))
    w = p.sum(0)
    grad = np.where(w > 1e-12 * w.max(), (p * dA).sum(0) / np.maximum(w, 1e-300), np.nan)
    ok = np.isfinite(grad)
    A = np.full_like(g, np.nan)
    gi = grad[ok]; xi = g[ok]
    A[ok] = np.concatenate([[0], np.cumsum(0.5 * (gi[1:] + gi[:-1]) * np.diff(xi))])
    c = 0.5 * (edges[1:] + edges[:-1])
    lo, hi = mu.min() - 2 * np.sqrt(s2.min()), mu.max() + 2 * np.sqrt(s2.max())
    F = np.interp(c, g[ok], A[ok])
    F[(c < lo) | (c > hi)] = np.nan
    return _finish(F)


def stitching(xs, r0, k, beta, edges, min_count=10):
    """Unbias each window's histogram, then chain additive offsets from overlaps with the previous window."""
    c = 0.5 * (edges[1:] + edges[:-1])
    F = np.full(len(c), np.nan); acc = None
    for v, r in zip(xs, r0):
        h, _ = np.histogram(v, edges)
        with np.errstate(divide="ignore"):
            Fi = -np.log(h) / beta - 0.5 * k * (c - r) ** 2
        Fi[h < min_count] = np.nan
        if acc is None:
            acc = Fi
        else:
            ov = np.isfinite(acc) & np.isfinite(Fi)
            if ov.any():
                Fi = Fi + np.mean(acc[ov] - Fi[ov])
            else:   # no overlap: the chain breaks - carry on with an arbitrary (wrong) offset
                last = np.where(np.isfinite(acc))[0]
                Fi = Fi - np.nanmin(Fi) + (acc[last[-1]] if len(last) else 0)
            acc = np.where(np.isfinite(acc) & np.isfinite(Fi), 0.5 * (acc + Fi), np.where(np.isfinite(acc), acc, Fi))
    return _finish(acc)


ESTIMATORS = {"MBAR": mbar, "WHAM": wham, "WHAM, 2 Å bins": wham_coarse, "umbrella integration": umbrella_integration,
              "histogram stitching": stitching}


def rmse(F, R, c, lo=5.0, hi=33.0):
    m = np.isfinite(F) & np.isfinite(R) & (c >= lo) & (c <= hi)
    if m.sum() < 3:
        return np.nan
    d = F[m] - R[m]
    return float(np.sqrt(np.mean((d - d.mean()) ** 2)))


def build(res, beta, edges, line, panel, pmf_layout, clean, text2):
    xs_full, r0, k = load_windows(res)
    if len(xs_full) < 5:
        return None
    c = 0.5 * (edges[1:] + edges[:-1])
    dt_ns = 0.5e-3
    stride = 4                                   # 2 ps between samples used
    xs = [v[::stride] for v in xs_full]
    n_full = min(len(v) for v in xs)
    ref = mbar(xs, r0, k, beta, edges)

    # 1) full-data PMFs, and with very little data
    full = {name: fn(xs, r0, k, beta, edges) for name, fn in ESTIMATORS.items()}
    few_n = 25                                   # 25 samples = 50 ps per window
    few = {name: fn([v[:few_n] for v in xs], r0, k, beta, edges) for name, fn in ESTIMATORS.items()}

    # 2) error vs data, with spread over 4 independent blocks
    nblk = 4; blk = n_full // nblk
    sizes = np.unique(np.geomspace(10, blk, 7).astype(int))
    err = {name: [] for name in ESTIMATORS}; spread = {name: [] for name in ESTIMATORS}
    for n in sizes:
        for name, fn in ESTIMATORS.items():
            ests = [fn([v[b * blk: b * blk + n] for v in xs], r0, k, beta, edges) for b in range(nblk)]
            e = [rmse(F, ref, c) for F in ests]
            err[name].append((np.nanmean(e), np.nanstd(e)))
            stack = np.array(ests)
            m = np.all(np.isfinite(stack), axis=0) & (c >= 5) & (c <= 33)
            if m.sum() >= 3:
                aligned = stack[:, m] - stack[:, m].mean(1, keepdims=True)
                spread[name].append(float(np.sqrt(np.mean(aligned.std(0) ** 2))))
            else:
                spread[name].append(np.nan)
    ns_per_window = sizes * stride * dt_ns

    # 3) drop windows: every 1st, 2nd, 3rd, 4th, 5th
    keeps = [1, 2, 3, 4, 5]
    spacing = [np.mean(np.diff(r0[::s])) for s in keeps]
    drop = {name: [rmse(fn([xs[i] for i in range(0, len(xs), s)], r0[::s], k, beta, edges), ref, c) for s in keeps]
            for name, fn in ESTIMATORS.items()}

    def pm(Fs, title, hint, pid):
        data = [line(c, Fs[n], n, EST_COLORS[n], width=2.5 if n == "MBAR" else 2,
                     dash="dot" if n == "histogram stitching" else None) for n in ESTIMATORS]
        return panel(pid, title, hint, data, pmf_layout(dict(yaxis=dict(title="F (kcal/mol)", range=[0, 30]))))

    def curve(name, x, y, err=None):
        t = dict(type="scatter", mode="lines+markers", x=clean(x, 4), y=clean(y, 3), name=name,
                 line=dict(color=EST_COLORS[name], width=2.5 if name == "MBAR" else 2), marker=dict(size=8, color=EST_COLORS[name]))
        if err is not None:
            t["error_y"] = dict(type="data", array=clean(err, 3), color=EST_COLORS[name], thickness=1.5)
        return t

    panels = [
        pm(full, "All estimators, full data (3 ns per window)", "with this much data and overlap they agree, except coarse-binned WHAM", "est-full"),
        pm(few, "All estimators, 50 ps per window", "with little data the estimators separate", "est-few"),
        panel("est-err", "Error vs data per window", "RMSE vs full-data MBAR over 5–33 Å; points = mean of 4 independent blocks, bars = their spread",
              [curve(n, ns_per_window, [e[0] for e in err[n]], [e[1] for e in err[n]]) for n in ESTIMATORS],
              dict(xaxis=dict(title="ns per window (42 windows)", type="log"), yaxis=dict(title="PMF RMSE (kcal/mol)", rangemode="tozero")), wide=True),
        panel("est-spread", "Statistical error without any reference", "RMS spread of the PMF across 4 independent blocks (lower = more precise)",
              [curve(n, ns_per_window, spread[n]) for n in ESTIMATORS],
              dict(xaxis=dict(title="ns per window", type="log"), yaxis=dict(title="block spread (kcal/mol)", rangemode="tozero"))),
        panel("est-drop", "Robustness to fewer windows", "full data, keeping every 1st…5th window; stitching needs neighbour overlap, MBAR/WHAM/UI degrade gracefully",
              [curve(n, spacing, drop[n]) for n in ESTIMATORS],
              dict(xaxis=dict(title="window spacing (Å)"), yaxis=dict(title="PMF RMSE vs all-window MBAR (kcal/mol)", rangemode="tozero"))),
    ]
    blurb = ("Same 42 umbrella windows, five ways to turn them into a free-energy profile. MBAR weights every sample optimally "
             "and needs no bins; WHAM is its binned approximation (shown at 0.5 Å and 2 Å bins, to expose the binning error MBAR avoids); umbrella integration uses only each window's mean and "
             "variance; stitching unbiases each window's histogram separately and glues neighbours by their overlap.")
    return dict(title="Estimators on the umbrella-sampling data", blurb=blurb,
                fe_how="all estimators see exactly the same samples (every 2 ps); error is measured against full-data MBAR",
                stats=[dict(value=str(len(xs)), label="windows"), dict(value=f"{n_full * stride * dt_ns:.1f} ns", label="per window"),
                       dict(value=f"{k:g} kcal/mol/Å²", label="restraint k")],
                panels=panels)
