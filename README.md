# Reproducible Dataset — 3D Wide-Angle SSFT Parabolic-Equation Urban Microcell

**Manuscript:** *Three-Dimensional Wide-Angle Split-Step Parabolic-Equation
Modelling of Urban Microcells: A Component-Wise Error Budget over Leontovich
Ground and Knife Screens*
**Manuscript ID:** MAP-2026-10-0176
**Journal:** *IET Microwaves, Antennas & Propagation*

**Authors:** Ruidong Wang, Chao Han, Meiru Wang, Zhibo Wang
(School of Electrical and Information Engineering, North Minzu University,
Yinchuan 750021, China)
**Corresponding author:** Ruidong Wang — <2017018@nmu.edu.cn>

This repository is the **reproducible dataset that supports the central
findings** of the study. It contains the complete solver and reference-method
source code together with the processed statistical data from which every
quantitative statement in the results and validation sections is derived. A
single self-contained script recomputes the reported tables and validation
numbers without re-running the full-wave simulations.

## 1. Contents

```
urban-3dpe-dataset/
├── README.md                 This file
├── LICENSE                   MIT licence (source code)
├── requirements.txt          Pinned scientific-Python dependencies
├── reproduce_tables.py       Recomputes Tables 2–3 and the validation numbers
│                              from the small data files (seconds, NumPy only)
├── reproduce_output.txt      Captured output of reproduce_tables.py for reference
├── code/                     Solver and reference-method source code
│   ├── kuttler_core3d.py     3D wide-angle SSFT-PE: spectral square-root
│   │                          propagator, DST/DMFT Leontovich impedance ground
│   │                          (H & V polarisations), Gaussian absorbing layers
│   ├── kuttler_core.py       2D Kuttler SSFT-PE solver (used for the 2D FDTD
│   │                          comparison)
│   ├── rt3d.py               Coherent 3D edge-diffraction ray reference
│   │                          (Fresnel integrals; direct/rooftop/deep-shadow rays)
│   ├── fdtd_engine.py        Independent 2D Yee FDTD full-wave reference
│   ├── scenario_config.py    Central scenario and model parameters
│   ├── urban_lib.py          Shared urban-block geometry, concrete half-space
│   │                          parameters and the 3D forward-march wrapper
│   ├── urban_multiline.py    Generates urban_multiline.npz  (Table 2)
│   ├── urban_param_sweep.py  Generates urban_param_sweep.npz (Table 3)
│   ├── urban_grid.py         Generates urban_grid.npz (grid convergence,
│   │                          lateral angular spectrum, run time)
│   ├── grid_convergence.py   Cross-range convergence and absorbing-window checks
│   └── fdtd_compare.py       Generates fdtd_errors.npz (independent FDTD
│                              validation)
└── data/                     Processed statistics reported in the manuscript
    ├── urban_multiline.npz   12 receiver-line PE-vs-ray errors  -> Table 2
    ├── urban_param_sweep.npz Frequency / source-height / layout sweep -> Table 3
    ├── urban_grid.npz        Grid-convergence curves, angular fractions, timings
    └── fdtd_errors.npz       Free-space / PEC-ground / knife-edge FDTD errors
```

## 2. Quick verification of the reported numbers

Only Python 3.9+ with NumPy is required. Install the dependencies and run the
reproduction script from the repository root:

```bash
pip install -r requirements.txt
python reproduce_tables.py
```

This reads the four `.npz` files in `data/` and prints, for direct comparison
with the manuscript:

- **Table 2** — per-line mean / RMS / 95th-percentile path-loss error over 12
  receiver lines (heights z = 10, 15, 20, 25 m; cross positions y = −10, 0, 10 m)
  and the pooled 3.13 / 5.18 dB (reported as 3.1 / 5.2 dB), together with the
  line-of-sight / rooftop / deep-shadow regime RMS values.
- **Table 3** — pooled errors for the frequency sweep (700 / 900 / 1800 MHz),
  source-height sweep (10 / 20 / 30 m) and regular-versus-staggered layout,
  including the analytic lateral Nyquist angles (20.1° / 15.5° / 7.7°).
- **Independent FDTD validation** — RMS errors against the analytic Hankel,
  image and Fresnel solutions (free space 0.04 / 0.46 dB; PEC two-ray
  2.29 / 2.44 / 3.12 dB; knife-edge 0.43 / 0.42 dB).
- **Grid convergence and cost** — Ny=192 vs Ny=384 curve difference
  (0.97 / 1.88 dB), the fraction of lateral power beyond 15.5°
  (0.003%–0.08%, mean 0.036%), and the single-thread 1 km march time
  (~66 s for Ny=192, ~131 s for Ny=384 at Nz=4096).

The supplied `reproduce_output.txt` is the captured output of this script on
the authors' machine; a fresh run reproduces it.

## 3. Requirements and full reproduction

The solvers depend only on the standard scientific Python stack:

- Python 3.9+
- NumPy
- SciPy (`scipy.fft`, `scipy.special`, `scipy.signal`, `scipy.integrate`)
- Matplotlib (for the plotting scripts; not needed by `reproduce_tables.py`)

The scripts in `code/` are the exact programs used to generate the results.
Run from inside the `code/` directory, e.g. `python urban_grid.py` or
`python fdtd_compare.py`. The parameter-sweep and grid studies execute the
full 3D forward march themselves (roughly one minute to several minutes per
case on a desktop CPU). The 12-line statistic (`urban_multiline.py`) reads the
stored full 3D field history.

## 4. Data availability

The processed statistics that reproduce the tables and figures, together with
the solver, the reference implementations and the reproduction scripts, are
publicly available in this repository and archived with a persistent DOI on
Zenodo:

- **Repository:** https://github.com/wrd3306/urban-3dpe-dataset
- **Zenodo archive (DOI):** https://doi.org/10.5281/zenodo.24357394

The complete three-dimensional field-history arrays (10–20 GB per case, which
are regenerable intermediates produced by running the supplied solvers) are not
archived because of their size; they are available from the corresponding
author on reasonable request.

## 5. Licence

- The **source code** in the repository root and `code/` is licensed under the
  MIT Licence (see `LICENSE`).
- The **processed data** in `data/` are licensed under the Creative Commons
  Attribution 4.0 International Licence (CC BY 4.0):
  <https://creativecommons.org/licenses/by/4.0/>.

## 6. Citation

Please cite both the journal article and the archived dataset. The dataset
citation (with the final DOI) is generated by Zenodo; a template is:

> R. Wang, C. Han, M. Wang and Z. Wang, "Reproducible dataset for
> *Three-Dimensional Wide-Angle Split-Step Parabolic-Equation Modelling of
> Urban Microcells*," Zenodo, 2026. https://doi.org/10.5281/zenodo.24357394
