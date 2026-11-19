#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the quantitative results in the manuscript from the small
processed-statistics files in ./data.  No full 3D field march is required:
the four .npz files below already hold the statistics reported in the paper,
and this script simply recomputes and prints them so they can be checked
against Tables 2-3 and the validation/convergence text.

Manuscript:
  "Three-Dimensional Wide-Angle Split-Step Parabolic-Equation Modelling of
   Urban Microcells: A Component-Wise Error Budget over Leontovich Ground and
   Knife Screens" (IET Microwaves, Antennas & Propagation,
   manuscript ID MAP-2026-10-0176).

Requirements: Python 3.9+ with NumPy.
Run:          python reproduce_tables.py

Data files used (in ./data):
  urban_multiline.npz   -> Table 2  (12 receiver-line PE-vs-ray errors)
  urban_param_sweep.npz -> Table 3  (frequency / source-height / layout sweep)
  fdtd_errors.npz       -> independent FDTD validation paragraph
  urban_grid.npz        -> cross-range grid convergence, angular spectrum, run time
"""
import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def sc(xc, t=3.0):
    """Building span for a centre coordinate xc and 3 m display thickness."""
    return (xc - t / 2.0, xc + t / 2.0)


# Five staggered knife-screen buildings (x0,x1,y1,y2,h); identical to
# Table 1 and to code/urban_lib.py.  Used only to mask in-building points
# when recomputing the grid-convergence curve difference.
BLDS = [
    (*sc(205.0), -30.0, 30.0, 10.0),
    (*sc(375.0), -30.0,  5.0, 25.0),
    (*sc(545.0),  -5.0, 30.0, 16.0),
    (*sc(705.0), -30.0, 30.0, 12.0),
    (*sc(875.0), -20.0, 30.0, 30.0),
]


def theta_nyq(f_mhz, dy=0.625):
    """Lateral Nyquist angle (deg) for cross-range spacing dy at f."""
    lam = 3.0e8 / (f_mhz * 1.0e6)
    return np.degrees(np.arcsin(lam / (2.0 * dy)))


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Table 2: PE vs coherent-ray path-loss error over twelve receiver lines
# --------------------------------------------------------------------------
hr("Table 2.  PE vs coherent-ray error over 12 receiver lines (dB)")
d = np.load(os.path.join(DATA, "urban_multiline.npz"))
R, zs, ys = d["R"], d["zs"], d["ys"]
print("receiver heights z (m) :", list(zs.astype(int)))
print("cross positions  y (m) :", list(ys.astype(int)))
print("columns: z, y, mean, RMS, p95, LOS-rms, rooftop-rms, deep-rms, n")
print("-" * 78)
for r in R:
    print("%3d %4d | %5.2f %5.2f %5.2f | %5.2f %6.2f %6.2f | %4d"
          % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
pooled_mean = np.nanmean(R[:, 2])
pooled_rms = np.sqrt(np.nanmean(R[:, 3] ** 2))
pooled_p95 = np.nanmean(R[:, 4])
print("-" * 78)
print("POOLED: mean = %.2f   RMS = %.2f   95th = %.2f dB"
      % (pooled_mean, pooled_rms, pooled_p95))
print("(manuscript reports pooled 3.1 / 5.2 dB; LOS pooled %.2f dB, "
      "rooftop %.2f dB, deep %.2f dB)"
      % (np.nanmean(R[:, 5]), np.nanmean(R[:, 6]), np.nanmean(R[:, 7])))


# --------------------------------------------------------------------------
# Table 3: parameter-sensitivity sweep
# --------------------------------------------------------------------------
hr("Table 3.  Parameter-sensitivity sweep, pooled 12-line errors (dB)")
s = np.load(os.path.join(DATA, "urban_param_sweep.npz"))
print("array column order:", list(s["keys"]))
print("= mean, RMS, p95, LOS, rooftop, deep (all pooled RMS except first 3)")
print("-" * 78)
sweep_rows = [
    ("frequency 700 MHz",            "freq_700",  theta_nyq(700.0)),
    ("frequency 900 MHz (reference)","freq_900",  theta_nyq(900.0)),
    ("frequency 1800 MHz",           "freq_1800", theta_nyq(1800.0)),
    ("source height 10 m",           "txz_10",    theta_nyq(900.0)),
    ("source height 20 m (reference)","txz_20",   theta_nyq(900.0)),
    ("source height 30 m",           "txz_30",    theta_nyq(900.0)),
    ("layout regular uniform",       "layout_regular", theta_nyq(900.0)),
]
print("%-32s %5s %5s %6s %5s %5s %5s  Nyq" %
      ("case", "mean", "RMS", "p95", "LOS", "roof", "deep"))
for name, key, ang in sweep_rows:
    v = s[key]
    print("%-32s %5.2f %5.2f %6.2f %5.2f %5.2f %5.2f  %4.1f°"
          % (name, v[0], v[1], v[2], v[3], v[4], v[5], ang))


# --------------------------------------------------------------------------
# Independent full-wave FDTD validation (2D reference cases)
# --------------------------------------------------------------------------
hr("Independent FDTD validation (RMS dB, from fdtd_errors.npz)")
fe = np.load(os.path.join(DATA, "fdtd_errors.npz"))
print("Free-space cylindrical spreading:")
print("  PE   vs analytic Hankel      = %.3f dB" % fe["eFree"])
print("  FDTD vs analytic Hankel      = %.3f dB" % fe["eFreeFD"])
print("PEC ground two-ray interference at z = 3 m:")
print("  FDTD vs PE                   = %.3f dB" % fe["ePecFP"])
print("  FDTD vs image method         = %.3f dB" % fe["ePec"])
print("  PE   vs image method         = %.3f dB" % fe["ePecP"])
print("PEC half-plane knife-edge diffraction:")
print("  FDTD vs PE                   = %.3f dB" % fe["eSc"])
print("  FDTD vs Fresnel knife-edge   = %.3f dB" % fe["eScF"])


# --------------------------------------------------------------------------
# Grid convergence, lateral angular spectrum, and run time
# --------------------------------------------------------------------------
hr("Grid convergence, angular spectrum and cost (urban_grid.npz)")
g = np.load(os.path.join(DATA, "urban_grid.npz"))
X, l192, l384 = g["X"], g["l192"], g["l384"]
valid = X > 300.0
for x0, x1, y1, y2, h in BLDS:
    if y1 <= 0.0 <= y2 and h >= 15.0:
        valid[(X >= x0 - 2.0) & (X <= x1 + 5.0)] = False
dd = np.abs(l192 - l384)[valid]
print("y=0,z=15 path-loss curve, Ny=192 vs Ny=384 (in-building points excluded):")
print("  mean difference = %.2f dB,  RMS = %.2f dB" % (dd.mean(),
      np.sqrt((dd ** 2).mean())))
fracs = np.asarray(g["fracs"]) * 100.0
print("\nLateral power beyond 15.5 deg at the four cross-planes (%):")
for i, fr in enumerate(fracs):
    print("  plane %d: %.4f%%" % (i + 1, fr))
print("  range %.3f%%-%.3f%%, mean %.4f%% (manuscript: 0.003%%-0.08%%, mean 0.036%%)"
      % (fracs.min(), fracs.max(), fracs.mean()))
print("\nDesktop CPU wall-clock for one 1 km march (800 steps, Nz=4096):")
print("  Ny=192 (dy=0.625 m): %.1f s   (reported ~66 s)" % g["t192"])
print("  Ny=384 (dy=0.3125 m): %.1f s  (reported ~131 s; single-thread timing"
      % g["t384"])
print("                                         varies by a few seconds between runs)")

print("\nAll statistics were read directly from the provided .npz files;")
print("the core SSFT-PE / ray / FDTD programs that generated them are in ./code.")
