import numpy as np
import matplotlib.pyplot as plt
import bin_convert
# made with the help of Claude
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

import glob, os, re, subprocess
res = subprocess.run(["bash", "-lc", "module load ffmpeg >/dev/null 2>&1 && echo $PATH"],
                     capture_output=True, text=True)
lines = res.stdout.strip().splitlines()
if res.returncode == 0 and lines:
    os.environ["PATH"] = lines[-1]
else:
    print("andrew_tools: couldn't load the ffmpeg module; pngs_to_mp4 won't work")
desiredfontsize=18
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no minus sign, so tick labels need this
})


def mb_cell_centers(mb_geometry, n, shape):
    """Cell-center coords for meshblock n, (Nz, Ny, Nx) order, sparse for broadcasting."""
    xmin, xmax, ymin, ymax, zmin, zmax = mb_geometry[n]
    Nz, Ny, Nx = shape
    x = xmin + (np.arange(Nx) + 0.5)*(xmax - xmin)/Nx
    y = ymin + (np.arange(Ny) + 0.5)*(ymax - ymin)/Ny
    z = zmin + (np.arange(Nz) + 0.5)*(zmax - zmin)/Nz
    return np.meshgrid(z, y, x, indexing='ij', sparse=True)

# Ordinal blue ramp (light -> dark = small -> large r); validated light end 2.06:1 vs white.
R_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"

QUANTITIES = {
    "sigma": (r"$\sigma = b^2/\rho$", True),
    "beta":  (r"$\beta = 2p/b^2$", True),
    "gamma": (r"$\Gamma$", False),
    "h":     (r"$h - 1$", True),
}


def band_stats(binary_fname, radii, thetas, dr=0.3, dtheta=0.05, fold=True,
               gamma_ad=4.0/3.0, stat="mean"):
    """sigma, beta, Gamma, h-1 averaged over the band |r - r_i| < dr/2, |theta - theta_j| < dtheta/2.

    Uses comoving b^2 = B^2/Gamma^2 + (v.B)^2.  fold=True also includes the mirror band
    pi - theta (north/south symmetric wind).  Works block by block, so 512^3 never gets
    stitched into one global array.  Returns {q: array (n_r, n_theta)}.
    """
    radii, thetas = np.atleast_1d(radii), np.atleast_1d(thetas)
    fd = bin_convert.read_binary(binary_fname)
    geo, data = fd["mb_geometry"], fd["mb_data"]
    vals = {q: [[[] for _ in thetas] for _ in radii] for q in QUANTITIES}
    wts = [[[] for _ in thetas] for _ in radii]

    for n in range(fd["n_mbs"]):
        Z, Y, X = mb_cell_centers(geo, n, np.shape(data["dens"][n]))
        R = np.sqrt(X**2 + Y**2 + Z**2)
        if R.max() < radii.min() - dr or R.min() > radii.max() + dr:
            continue
        TH = np.arccos(np.clip(Z/R, -1.0, 1.0))
        if fold:
            TH = np.minimum(TH, np.pi - TH)
        near = np.zeros(R.shape, bool)
        for r in radii:
            near |= np.abs(R - r) < 0.5*dr
        if not near.any():
            continue

        rho, eint = data["dens"][n], data["eint"][n]
        ux, uy, uz = data["velx"][n], data["vely"][n], data["velz"][n]
        bx, by, bz = data["bcc1"][n], data["bcc2"][n], data["bcc3"][n]
        G = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
        vB = (ux*bx + uy*by + uz*bz)/G
        b2 = (bx**2 + by**2 + bz**2)/G**2 + vB**2
        p = (gamma_ad - 1.0)*eint
        q_all = {"sigma": b2/rho, "beta": 2.0*p/b2, "gamma": G,
                 "h": gamma_ad/(gamma_ad - 1.0)*p/rho}
        xmin, xmax, ymin, ymax, zmin, zmax = geo[n]
        dV = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)/rho.size

        for i, r in enumerate(radii):
            shell = np.abs(R - r) < 0.5*dr
            if not shell.any():
                continue
            for j, th in enumerate(thetas):
                th_f = min(th, np.pi - th) if fold else th
                m = shell & (np.abs(TH - th_f) < 0.5*dtheta)
                if not m.any():
                    continue
                wts[i][j].append(np.full(m.sum(), dV))
                for q in QUANTITIES:
                    vals[q][i][j].append(np.broadcast_to(q_all[q], R.shape)[m])

    out = {q: np.full((radii.size, thetas.size), np.nan) for q in QUANTITIES}
    for q in QUANTITIES:
        for i in range(radii.size):
            for j in range(thetas.size):
                if not vals[q][i][j]:
                    continue
                v, w = np.concatenate(vals[q][i][j]), np.concatenate(wts[i][j])
                out[q][i, j] = np.average(v, weights=w) if stat == "mean" else np.median(v)
    return out


def resolution_study(runs, radii, thetas, snaps, name="farzone_wind", **kw):
    """runs: {N: run_directory}.  Averages band_stats over the snapshot list snaps
    (pick a steady-state window).  Returns resolutions (sorted) and
    {q: array (n_res, n_r, n_theta)}."""
    res = sorted(runs)
    stats = {q: np.full((len(res), len(radii), len(thetas)), np.nan) for q in QUANTITIES}
    for k, N in enumerate(res):
        per_snap = [band_stats(f"{runs[N]}/bin/{name}.mhd_w_bcc.{s:05d}.bin", radii, thetas, **kw)
                    for s in snaps]
        for q in QUANTITIES:
            stats[q][k] = np.nanmean([ps[q] for ps in per_snap], axis=0)
    return res, stats


def plot_resolution_study(res, stats, radii, thetas, quantities=("sigma", "beta", "gamma"),
                          refs=None, title=None, save=None, dpi=150):
    """Rows = quantities, columns = thetas, one line per radius.
    refs: optional {q: value} drawn as a dashed reference (e.g. injected Gamma, sigma0)."""
    if len(radii) > len(R_RAMP):
        raise ValueError(f"at most {len(R_RAMP)} radii per figure -- split into two figures")
    colors = R_RAMP if len(radii) == len(R_RAMP) else \
        [R_RAMP[int(round(i))] for i in np.linspace(0, len(R_RAMP) - 1, len(radii))]
    nq, nt = len(quantities), len(thetas)
    fig, axes = plt.subplots(nq, nt, figsize=(3.4*nt + 0.8, 2.6*nq + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    x = np.asarray(res, float)

    for a, q in enumerate(quantities):
        label, logy = QUANTITIES[q]
        for b, th in enumerate(thetas):
            ax = axes[a, b]
            if refs and q in refs:
                ax.axhline(refs[q], color=MUTED, lw=1, ls="--", zorder=1)
            for i, (r, c) in enumerate(zip(radii, colors)):
                y = stats[q][:, i, b]
                ax.plot(x, y, color=c, lw=2, marker="o", ms=6, mec="white", mew=1.5,
                        zorder=3, label=rf"$r = {r:g}$")
                if len(radii) <= 4 and np.isfinite(y[-1]):   # direct label at the right end
                    ax.annotate(rf"$r={r:g}$", (x[-1], y[-1]), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=9, color=INK)
            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(N)}" for N in x])
            #ax.minorticks_off()
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=10)
            if a == 0:
                ax.set_title(rf"$\theta = {np.degrees(th):.0f}^\circ$", fontsize=12, color=INK)
            if b == 0:
                ax.set_ylabel(label, fontsize=13, color=INK)
            if a == nq - 1:
                ax.set_xlabel(r"resolution $N$ (cells per side)", fontsize=11, color=INK)
            if len(radii) <= 4:
                ax.set_xlim(x[0]/1.25, x[-1]*1.6)   # room for the direct labels

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(radii), frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, fontsize=12, color=INK, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig, axes
