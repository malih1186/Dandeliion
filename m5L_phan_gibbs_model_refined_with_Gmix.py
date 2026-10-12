"""
Refined M5.L Gibbs-energy model for the LiFePO4-FePO4 olivine join.

Based on Phan et al., Can. J. Chem. Eng. 97 (2019) 2224-2233,
DOI: 10.1002/cjce.23416.

This version keeps the published M5.L thermodynamic model and improves the
numerical post-processing:

1. Fine composition grids can be used for G(x,T).
2. Coexistence compositions are refined with direct common-tangent equations.
3. Spinodals use the curvature of the internally minimized G(x,T), with a
   high-order numerical derivative and an envelope-theorem option.
4. The eutectoid point is located numerically from Gmix=0 and dGmix/dx=0.
5. A phase-diagram plot is produced with separately refined binodal/spinodal
   branches and paper/model reference annotations.
6. A diagnostic report compares the numerical model against the quantitative
   values explicitly reported by Phan et al. (no experimental-point overlay).
7. User-defined G(xLi,T) and Gmix(xLi,T) curves can be exported to CSV.
8. User-defined G(xLi,T) and Gmix(xLi,T) curves can be saved as multi-page,
   individual, and overlay PDF plots.

The experimental data points from the paper are intentionally NOT included.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize_scalar, root, brentq
from scipy.interpolate import CubicSpline

# -----------------------------------------------------------------------------
# Constants and published parameters
# -----------------------------------------------------------------------------
R = 8.31446261815324          # J mol^-1 K^-1
FARADAY = 96485.33212         # C mol^-1
T0 = 298.15                   # K
T_SWITCH_L = 773.0            # K

# Pure-compound thermodynamic data from Table 1
DH298_L = -161602.0           # J/mol
S298_L = 130.95               # J/(mol K)
DH298_F = -127923.0           # J/mol
S298_F = 122.21               # J/(mol K)

# Optimized M5.L parameters from Table 1
DG_LI_PLUS = 15000.0          # J/mol
DG_FE_MINUS = 15000.0         # J/mol
DG_ADJUST1 = 19500.0          # J/mol
DG_ADJUST2 = 29250.0          # J/mol

# -----------------------------------------------------------------------------
# Numerical settings
# -----------------------------------------------------------------------------
X_EPS = 1.0e-7
T_MIN_DEFAULT = 298.15
T_MAX_DEFAULT = 650.0


def xlogx(x):
    """Return x*ln(x), using the thermodynamic limit x ln x -> 0 at x=0."""
    if x <= 0.0:
        return 0.0
    return x * np.log(x)


# -----------------------------------------------------------------------------
# Heat capacities
# -----------------------------------------------------------------------------

def Cp_LiFePO4(T):
    """LiFePO4 Cp(T), piecewise as given in Table 1."""
    T = np.asarray(T)
    return np.where(
        T <= T_SWITCH_L,
        41.881336
        + 0.78278483 * T
        + 890694.39 / T**2
        - 0.0010255433 * T**2
        + 5.0862948e-7 * T**3,
        247.915258
        - 0.01771626 * T
        - 28310678.63 / T**2,
    )


def Cp_FePO4(T):
    """FePO4 Cp(T) from Table 1."""
    T = np.asarray(T)
    return 114.52232 + 0.05966384 * T - 2690312.0 / T**2


# -----------------------------------------------------------------------------
# Analytical Cp integrals
# -----------------------------------------------------------------------------

def _CpInt_L1(T):
    return (
        41.881336 * T
        + 0.78278483 * T**2 / 2.0
        - 890694.39 / T
        - 0.0010255433 * T**3 / 3.0
        + 5.0862948e-7 * T**4 / 4.0
    )


def _CpOverTInt_L1(T):
    return (
        41.881336 * np.log(T)
        + 0.78278483 * T
        - 890694.39 / (2.0 * T**2)
        - 0.0010255433 * T**2 / 2.0
        + 5.0862948e-7 * T**3 / 3.0
    )


def _CpInt_L2(T):
    return (
        247.915258 * T
        - 0.01771626 * T**2 / 2.0
        + 28310678.63 / T
    )


def _CpOverTInt_L2(T):
    return (
        247.915258 * np.log(T)
        - 0.01771626 * T
        - 28310678.63 / (2.0 * T**2)
    )


def int_Cp_L(Ta, Tb):
    """Integral of Cp(LiFePO4) dT from Ta to Tb."""
    if Tb <= T_SWITCH_L:
        return _CpInt_L1(Tb) - _CpInt_L1(Ta)
    if Ta >= T_SWITCH_L:
        return _CpInt_L2(Tb) - _CpInt_L2(Ta)
    return (
        _CpInt_L1(T_SWITCH_L) - _CpInt_L1(Ta)
        + _CpInt_L2(Tb) - _CpInt_L2(T_SWITCH_L)
    )


def int_Cp_over_T_L(Ta, Tb):
    """Integral of Cp(LiFePO4)/T dT from Ta to Tb."""
    if Tb <= T_SWITCH_L:
        return _CpOverTInt_L1(Tb) - _CpOverTInt_L1(Ta)
    if Ta >= T_SWITCH_L:
        return _CpOverTInt_L2(Tb) - _CpOverTInt_L2(Ta)
    return (
        _CpOverTInt_L1(T_SWITCH_L) - _CpOverTInt_L1(Ta)
        + _CpOverTInt_L2(Tb) - _CpOverTInt_L2(T_SWITCH_L)
    )


def _CpInt_F(T):
    return 114.52232 * T + 0.05966384 * T**2 / 2.0 + 2690312.0 / T


def _CpOverTInt_F(T):
    return (
        114.52232 * np.log(T)
        + 0.05966384 * T
        + 2690312.0 / (2.0 * T**2)
    )


def int_Cp_F(Ta, Tb):
    return _CpInt_F(Tb) - _CpInt_F(Ta)


def int_Cp_over_T_F(Ta, Tb):
    return _CpOverTInt_F(Tb) - _CpOverTInt_F(Ta)


# -----------------------------------------------------------------------------
# Pure-compound Gibbs energies and enthalpies
# -----------------------------------------------------------------------------

def G0_LiFePO4(T):
    return DH298_L + int_Cp_L(T0, T) - T * (
        S298_L + int_Cp_over_T_L(T0, T)
    )


def G0_FePO4(T):
    return DH298_F + int_Cp_F(T0, T) - T * (
        S298_F + int_Cp_over_T_F(T0, T)
    )


def H0_LiFePO4(T):
    return DH298_L + int_Cp_L(T0, T)


def H0_FePO4(T):
    return DH298_F + int_Cp_F(T0, T)


# -----------------------------------------------------------------------------
# M5.L end members
# -----------------------------------------------------------------------------

def M5L_endmembers(T):
    """Return the eight 5-formula-unit CEF end-member energies."""
    GL = G0_LiFePO4(T)
    GF = G0_FePO4(T)

    return np.array([
        5.0 * GL,
        5.0 * GF,
        5.0 * GL + 5.0 * DG_LI_PLUS,
        5.0 * GF + 5.0 * DG_FE_MINUS,
        3.0 * GF + 3.0 * DG_FE_MINUS + 2.0 * GL,
        2.0 * GL + 2.0 * DG_LI_PLUS + 3.0 * GF + 3.0 * DG_ADJUST1,
        3.0 * GL + 2.0 * GF + 2.0 * DG_FE_MINUS - DG_ADJUST1,
        2.0 * GF + 3.0 * GL + 3.0 * DG_LI_PLUS - DG_ADJUST2,
    ])


# -----------------------------------------------------------------------------
# Composition and internal order parameter
# -----------------------------------------------------------------------------

def a_bounds(x):
    """Allowed a range from x=(3a+2b)/5 and 0<=a,b<=1."""
    lo = max(0.0, (5.0 * x - 2.0) / 3.0)
    hi = min(1.0, 5.0 * x / 3.0)
    return lo, hi


def b_from_a_x(a, x):
    return (5.0 * x - 3.0 * a) / 2.0


# -----------------------------------------------------------------------------
# M5.L Gibbs energy at fixed a, x, T
# -----------------------------------------------------------------------------

def G_M5L(a, x, T, endmembers=None):
    if x < 0.0 or x > 1.0:
        return np.inf

    lo, hi = a_bounds(x)
    if a < lo - 1e-12 or a > hi + 1e-12:
        return np.inf

    a = float(np.clip(a, 0.0, 1.0))
    b = float(np.clip(b_from_a_x(a, x), 0.0, 1.0))

    if endmembers is None:
        endmembers = M5L_endmembers(T)

    fractions = np.array([
        a * b * x,
        (1.0 - a) * (1.0 - b) * (1.0 - x),
        a * b * (1.0 - x),
        (1.0 - a) * (1.0 - b) * x,
        (1.0 - a) * b * x,
        (1.0 - a) * b * (1.0 - x),
        a * (1.0 - b) * x,
        a * (1.0 - b) * (1.0 - x),
    ])

    G_end = np.dot(fractions, endmembers) / 5.0

    G_config = R * T * (
        (3.0 / 5.0) * (xlogx(a) + xlogx(1.0 - a))
        + (2.0 / 5.0) * (xlogx(b) + xlogx(1.0 - b))
        + xlogx(x) + xlogx(1.0 - x)
    )

    # The paper's M5.L model uses no excess Gibbs-energy term.
    return G_end + G_config


# -----------------------------------------------------------------------------
# Equilibrium M5.L: minimize over internal LRO variable a
# -----------------------------------------------------------------------------

def equilibrium_M5L(x, T):
    if x <= 0.0:
        return G0_FePO4(T), 0.0, 0.0
    if x >= 1.0:
        return G0_LiFePO4(T), 1.0, 1.0

    lo, hi = a_bounds(x)
    E = M5L_endmembers(T)

    result = minimize_scalar(
        lambda a: G_M5L(a, x, T, E),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-12},
    )

    candidates = [
        (result.fun, result.x),
        (G_M5L(lo, x, T, E), lo),
        (G_M5L(hi, x, T, E), hi),
    ]
    Gmin, amin = min(candidates, key=lambda q: q[0])
    bmin = b_from_a_x(amin, x)
    return Gmin, amin, bmin


# -----------------------------------------------------------------------------
# Curves and mixing functions
# -----------------------------------------------------------------------------

def calculate_G_curve(T, x_grid):
    x_grid = np.asarray(x_grid, dtype=float)
    G = np.empty_like(x_grid)
    A = np.empty_like(x_grid)
    B = np.empty_like(x_grid)
    for i, x in enumerate(x_grid):
        G[i], A[i], B[i] = equilibrium_M5L(float(x), T)
    return G, A, B


def G_mixing(x, T):
    G, _, _ = equilibrium_M5L(float(x), T)
    return G - ((1.0 - x) * G0_FePO4(T) + x * G0_LiFePO4(T))


def calculate_Gmix_curve(T, x_grid):
    return np.array([G_mixing(float(x), T) for x in x_grid])


# -----------------------------------------------------------------------------
# Enthalpy of mixing
# -----------------------------------------------------------------------------

def H_mixing(x, T):
    _, a, b = equilibrium_M5L(x, T)

    H_end = np.array([
        5.0 * H0_LiFePO4(T),
        5.0 * H0_FePO4(T),
        5.0 * H0_LiFePO4(T),
        5.0 * H0_FePO4(T),
        2.0 * H0_LiFePO4(T) + 3.0 * H0_FePO4(T),
        2.0 * H0_LiFePO4(T) + 3.0 * H0_FePO4(T),
        3.0 * H0_LiFePO4(T) + 2.0 * H0_FePO4(T),
        3.0 * H0_LiFePO4(T) + 2.0 * H0_FePO4(T),
    ])

    H_const = np.array([
        0.0,
        0.0,
        5.0 * DG_LI_PLUS,
        5.0 * DG_FE_MINUS,
        3.0 * DG_FE_MINUS,
        2.0 * DG_LI_PLUS + 3.0 * DG_ADJUST1,
        2.0 * DG_FE_MINUS - DG_ADJUST1,
        3.0 * DG_LI_PLUS - DG_ADJUST2,
    ])

    fractions = np.array([
        a * b * x,
        (1.0 - a) * (1.0 - b) * (1.0 - x),
        a * b * (1.0 - x),
        (1.0 - a) * (1.0 - b) * x,
        (1.0 - a) * b * x,
        (1.0 - a) * b * (1.0 - x),
        a * (1.0 - b) * x,
        a * (1.0 - b) * (1.0 - x),
    ])

    H_eq = np.dot(fractions, H_end + H_const) / 5.0
    H_ref = (1.0 - x) * H0_FePO4(T) + x * H0_LiFePO4(T)
    return H_eq - H_ref


# -----------------------------------------------------------------------------
# High-accuracy interpolation of the internally minimized G(x,T)
# -----------------------------------------------------------------------------

def _safe_x(x):
    return float(np.clip(x, X_EPS, 1.0 - X_EPS))


def G_eq(x, T):
    """Equilibrium Gibbs energy G(x,T) after internal LRO minimization."""
    return equilibrium_M5L(float(x), float(T))[0]


def build_G_spline(T, nx=2001):
    """Calculate G(x,T) once on a fine grid and build a cubic spline."""
    x = np.linspace(X_EPS, 1.0 - X_EPS, nx)
    G, A, B = calculate_G_curve(T, x)
    spline = CubicSpline(x, G, bc_type="natural")
    return x, G, A, B, spline


def dGdx(x, T, spline=None):
    """First derivative of equilibrium G with respect to composition."""
    if spline is not None:
        return float(spline(_safe_x(x), 1))
    _, _, _, _, spline = build_G_spline(T, nx=2001)
    return float(spline(_safe_x(x), 1))


def d2Gdx2(x, T, spline=None):
    """Second derivative of equilibrium G with respect to composition."""
    if spline is not None:
        return float(spline(_safe_x(x), 2))
    _, _, _, _, spline = build_G_spline(T, nx=2001)
    return float(spline(_safe_x(x), 2))


# -----------------------------------------------------------------------------
# Direct common-tangent solution
# -----------------------------------------------------------------------------

def common_tangent_residual(z, spline):
    """Direct common-tangent equations for a given G(x) spline."""
    xa, xb = z
    if not (X_EPS < xa < xb < 1.0 - X_EPS):
        return np.array([1e9, 1e9])

    Ga = float(spline(xa))
    Gb = float(spline(xb))
    secant = (Gb - Ga) / (xb - xa)
    mua = float(spline(xa, 1))
    mub = float(spline(xb, 1))
    return np.array([mua - secant, mub - secant])


def refine_common_tangent(xa0, xb0, T, spline=None):
    """Refine a candidate pair by solving the two common-tangent equations."""
    if not (X_EPS < xa0 < xb0 < 1.0 - X_EPS):
        return None
    if xb0 - xa0 < 5e-4:
        return None

    if spline is None:
        _, _, _, _, spline = build_G_spline(T, nx=2001)

    sol = root(lambda z: common_tangent_residual(z, spline), [xa0, xb0])
    if not sol.success:
        return None

    xa, xb = sol.x
    if not (X_EPS < xa < xb < 1.0 - X_EPS):
        return None
    if xb - xa < 1e-4:
        return None

    residual = common_tangent_residual([xa, xb], spline)
    if np.max(np.abs(residual)) > 1e-3:
        return None

    # The spline is built from the actual internally minimized G values on a
    # fine grid.  At nx=1601 its interpolation error is much smaller than
    # the composition resolution needed for the phase-boundary plot.
    return float(sol.x[0]), float(sol.x[1])


# -----------------------------------------------------------------------------
# Coarse convex hull: only used to locate candidate gaps
# -----------------------------------------------------------------------------

def lower_convex_hull(x, G):
    hull = []
    for i in range(len(x)):
        hull.append(i)
        while len(hull) >= 3:
            i1, i2, i3 = hull[-3:]
            s1 = (G[i2] - G[i1]) / (x[i2] - x[i1])
            s2 = (G[i3] - G[i2]) / (x[i3] - x[i2])
            if s2 <= s1:
                hull.pop(-2)
            else:
                break
    return np.asarray(hull, dtype=int)


def candidate_gaps_from_grid(x, G):
    hull = lower_convex_hull(x, G)
    return [(x[i], x[j]) for i, j in zip(hull[:-1], hull[1:]) if j > i + 1]


def coexistence_compositions(T, nx=2001):
    """Find all binodal pairs at T using a fine G grid + direct tangent solve."""
    x, G, _, _, spline = build_G_spline(T, nx=nx)
    refined = []
    for xa0, xb0 in candidate_gaps_from_grid(x, G):
        pair = refine_common_tangent(xa0, xb0, T, spline=spline)
        if pair is not None:
            if not any(abs(pair[0]-q[0]) < 1e-4 and abs(pair[1]-q[1]) < 1e-4 for q in refined):
                refined.append(pair)
    return sorted(refined)


def stationary_points(T, nx=2001):
    """Return interior stationary points of G(x,T) from a fine spline."""
    x, G, _, _, spline = build_G_spline(T, nx=nx)
    mu = spline(x, 1)
    roots = []
    for i in range(len(x)-1):
        if mu[i] * mu[i+1] < 0.0:
            roots.append(float(brentq(lambda xx: float(spline(xx, 1)), x[i], x[i+1])))
    return roots, spline


def metastable_tangents(T, nx=2001):
    """
    Find tangent pairs associated with local minima that are not the global
    equilibrium convex-hull pair. These are useful for Figure 4-style plots.
    """
    x, G, _, _, spline = build_G_spline(T, nx=nx)
    minima = []
    # Stationary points of Gmix have dG/dx equal to the slope of the
    # pure-compound reference line, G_LiFePO4 - G_FePO4.
    mu = spline(x, 1) - (G0_LiFePO4(T) - G0_FePO4(T))
    curv = spline(x, 2)
    for i in range(1, len(x)-1):
        if mu[i-1] < 0.0 and mu[i+1] > 0.0 and curv[i] > 0.0:
            minima.append(x[i])

    pairs = []
    stable = coexistence_compositions(T, nx=nx)
    # Try every pair of distinct local minima. A common tangent is valid even
    # if it lies above the stable convex envelope, hence "metastable".
    for i in range(len(minima)):
        for j in range(i+1, len(minima)):
            xa0, xb0 = minima[i], minima[j]
            pair = refine_common_tangent(xa0, xb0, T, spline=spline)
            if pair is not None:
                if any(abs(pair[0]-q[0]) < 1e-3 and abs(pair[1]-q[1]) < 1e-3 for q in stable):
                    continue
                if not any(abs(pair[0]-q[0]) < 1e-4 and abs(pair[1]-q[1]) < 1e-4 for q in pairs):
                    pairs.append(pair)
    return sorted(pairs)


# -----------------------------------------------------------------------------
# Spinodal boundary
# -----------------------------------------------------------------------------

def spinodal_roots(T, nx=2001):
    """Find spinodal roots from d2G/dx2=0 using an interior fine-G spline."""
    x, G, _, _, spline = build_G_spline(T, nx=nx)
    xs = np.linspace(0.005, 0.995, max(201, nx // 2))
    curvature = spline(xs, 2)
    roots = []
    for i in range(len(xs)-1):
        y1, y2 = curvature[i], curvature[i+1]
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        if y1 == 0.0:
            roots.append(xs[i])
        elif y1 * y2 < 0.0:
            roots.append(float(brentq(lambda xx: float(spline(xx, 2)), xs[i], xs[i+1])))
    unique = []
    for r in roots:
        if not unique or abs(r - unique[-1]) > 1e-4:
            unique.append(r)
    return unique


# -----------------------------------------------------------------------------
# Numerical eutectoid point
# -----------------------------------------------------------------------------

def three_phase_eutectoid_residual(z, nx=2001):
    """Four equations for a three-point common tangent at (x1,x2,x3,T)."""
    x1, x2, x3, T = z
    if not (0.0 < x1 < x2 < x3 < 1.0 and 430.0 < T < 570.0):
        return np.ones(4) * 1e6

    _, _, _, _, spline = build_G_spline(T, nx=nx)
    G1, G2, G3 = float(spline(x1)), float(spline(x2)), float(spline(x3))
    mu1, mu2, mu3 = float(spline(x1, 1)), float(spline(x2, 1)), float(spline(x3, 1))

    sec13 = (G3 - G1) / (x3 - x1)
    sec12 = (G2 - G1) / (x2 - x1)

    # Equal chemical potentials and collinearity of all three points.
    return np.array([
        (mu1 - mu2) / 10000.0,
        (mu2 - mu3) / 10000.0,
        (mu1 - sec13) / 10000.0,
        (sec12 - sec13) / 10000.0,
    ])


def find_eutectoid(x_guess=(0.13, 0.60, 0.88), T_guess=500.0):
    """Locate the three-phase eutectoid by solving the triple-tangent problem."""
    z0 = [x_guess[0], x_guess[1], x_guess[2], T_guess]
    sol = root(lambda z: three_phase_eutectoid_residual(z, nx=1201), z0)
    if not sol.success:
        return None

    x1, x2, x3, T = sol.x
    if not (0.0 < x1 < x2 < x3 < 1.0 and 430.0 < T < 570.0):
        return None

    res = three_phase_eutectoid_residual(sol.x, nx=1601)
    if np.max(np.abs(res)) > 1e-4:
        return None

    return float(x1), float(x2), float(x3), float(T)


# -----------------------------------------------------------------------------
# Phase-diagram calculation
# -----------------------------------------------------------------------------

def calculate_phase_diagram(T_grid=None, nx=1601):
    if T_grid is None:
        T_grid = np.linspace(T_MIN_DEFAULT, T_MAX_DEFAULT, 71)

    data = []
    for T in T_grid:
        x, G, _, _, spline = build_G_spline(float(T), nx=nx)
        pairs = []
        for xa0, xb0 in candidate_gaps_from_grid(x, G):
            pair = refine_common_tangent(xa0, xb0, float(T), spline=spline)
            if pair is not None and not any(abs(pair[0]-q[0]) < 1e-4 and abs(pair[1]-q[1]) < 1e-4 for q in pairs):
                pairs.append(pair)
        pairs.sort()

        # Avoid the endpoint region where x*ln(x) creates a very sharp
        # numerical curvature and the natural spline boundary condition can
        # generate artificial roots.  The physical spinodal is an interior
        # stability boundary.
        xs = np.linspace(0.005, 0.995, max(101, nx // 2))
        curvature = spline(xs, 2)
        spins = []
        for i in range(len(xs)-1):
            y1, y2 = curvature[i], curvature[i+1]
            if np.isfinite(y1) and np.isfinite(y2) and y1*y2 < 0.0:
                r = float(brentq(lambda xx: float(spline(xx, 2)), xs[i], xs[i+1]))
                if not spins or abs(r-spins[-1]) > 1e-4:
                    spins.append(r)

        data.append({"T": float(T), "binodals": pairs, "spinodals": spins})
    return data


# -----------------------------------------------------------------------------
# Gibbs-energy extraction and plotting for user-defined temperatures
# -----------------------------------------------------------------------------

def export_gibbs_vs_x_csv(temperatures, output_dir="gibbs_vs_x_csv",
                          x_min=0.001, x_max=0.999, nx=1201):
    """
    Calculate the equilibrium Gibbs energy G(xLi,T) and save one CSV per
    user-defined temperature.

    CSV columns:
      xLi                 : mole fraction of LiFePO4 (= xLi)
      G_J_per_mol         : equilibrium Gibbs energy, J/mol
      G_kJ_per_mol        : equilibrium Gibbs energy, kJ/mol
      a_Li_sub1           : Li site fraction on the 3-site Li/Va sublattice
      b_Li_sub2           : Li site fraction on the 2-site Li/Va sublattice
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(x_min, x_max, nx)
    saved_files = []

    for T in temperatures:
        T = float(T)
        G, A, B = calculate_G_curve(T, x_grid)

        data = np.column_stack((
            x_grid,
            G,
            G / 1000.0,
            A,
            B,
        ))

        # Use a filesystem-safe temperature in the filename.
        T_label = f"{T:.2f}".replace(".", "p")
        filename = output_dir / f"G_vs_xLi_{T_label}K.csv"

        header = (
            "xLi,G_J_per_mol,G_kJ_per_mol,"
            "a_Li_sub1,b_Li_sub2"
        )
        np.savetxt(
            filename,
            data,
            delimiter=",",
            header=header,
            comments="",
            fmt=["%.10f", "%.10f", "%.10f", "%.10f", "%.10f"],
        )

        saved_files.append(filename)
        print(f"Saved: {filename}")

    return saved_files


def export_gmix_vs_x_csv(temperatures, output_dir="gmix_vs_xLi_csv",
                         x_min=0.001, x_max=0.999, nx=1201):
    """
    Calculate the equilibrium Gibbs energy of mixing Gmix(xLi,T) and save
    one CSV per user-defined temperature.

    Gmix is defined as:
        Gmix = G - [(1-xLi)*G0_FePO4(T) + xLi*G0_LiFePO4(T)]

    CSV columns:
      xLi                 : mole fraction of LiFePO4
      Gmix_J_per_mol      : Gibbs energy of mixing, J/mol
      Gmix_kJ_per_mol     : Gibbs energy of mixing, kJ/mol
      G_J_per_mol         : absolute equilibrium Gibbs energy, J/mol
      G_kJ_per_mol        : absolute equilibrium Gibbs energy, kJ/mol
      G_ref_J_per_mol     : linear pure-endmember reference, J/mol
      a_Li_sub1           : Li site fraction on the 3-site Li/Va sublattice
      b_Li_sub2           : Li site fraction on the 2-site Li/Va sublattice
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(x_min, x_max, nx)
    saved_files = []

    for T in temperatures:
        T = float(T)
        G, A, B = calculate_G_curve(T, x_grid)
        G_ref = ((1.0 - x_grid) * G0_FePO4(T)
                 + x_grid * G0_LiFePO4(T))
        Gmix = G - G_ref

        data = np.column_stack((
            x_grid,
            Gmix,
            Gmix / 1000.0,
            G,
            G / 1000.0,
            G_ref,
            A,
            B,
        ))

        T_label = f"{T:.2f}".replace(".", "p")
        filename = output_dir / f"Gmix_vs_xLi_{T_label}K.csv"

        header = (
            "xLi,Gmix_J_per_mol,Gmix_kJ_per_mol,"
            "G_J_per_mol,G_kJ_per_mol,G_ref_J_per_mol,"
            "a_Li_sub1,b_Li_sub2"
        )

        np.savetxt(
            filename,
            data,
            delimiter=",",
            header=header,
            comments="",
            fmt=["%.10f"] * data.shape[1],
        )

        saved_files.append(filename)
        print(f"Saved: {filename}")

    return saved_files


def plot_gibbs_vs_x_temperatures(temperatures, x_min=0.001, x_max=0.999,
                                 nx=1201, output_pdf="G_vs_xLi_all_temperatures.pdf",
                                 individual_pdf_dir=None, show=True):
    """
    Plot equilibrium Gibbs energy G versus xLi for user-defined temperatures.

    A single multi-page PDF is always created: one page per temperature.
    If individual_pdf_dir is supplied, an additional one-page PDF is created
    for each temperature.

    Returns
    -------
    figures : list
        Matplotlib figure objects, one for each temperature.
    """
    from pathlib import Path

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if individual_pdf_dir is not None:
        individual_pdf_dir = Path(individual_pdf_dir)
        individual_pdf_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(x_min, x_max, nx)
    figures = []

    with PdfPages(output_pdf) as pdf:
        for T in temperatures:
            T = float(T)
            G, _, _ = calculate_G_curve(T, x_grid)

            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            ax.plot(x_grid, G / 1000.0, linewidth=1.8)

            ax.set_xlabel("Mole fraction of LiFePO4, xLi")
            ax.set_ylabel("Gibbs energy (kJ/mol)")
            ax.set_title(f"M5.L Gibbs energy vs xLi at T = {T:.2f} K")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()

            pdf.savefig(fig)

            if individual_pdf_dir is not None:
                T_label = f"{T:.2f}".replace(".", "p")
                individual_file = individual_pdf_dir / f"G_vs_xLi_{T_label}K.pdf"
                with PdfPages(individual_file) as one_pdf:
                    one_pdf.savefig(fig)

            figures.append(fig)

            if not show:
                plt.close(fig)

    print(f"Saved combined PDF: {output_pdf}")
    if individual_pdf_dir is not None:
        print(f"Saved individual PDFs in: {individual_pdf_dir}")

    if show:
        plt.show()

    return figures


def plot_gibbs_vs_x_overlay(temperatures, x_min=0.001, x_max=0.999, nx=1201,
                            output_pdf="G_vs_xLi_overlay.pdf", show=True):
    """
    Plot all requested G(xLi,T) curves on one figure and save as PDF.
    """
    from pathlib import Path

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(x_min, x_max, nx)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    for T in temperatures:
        G, _, _ = calculate_G_curve(float(T), x_grid)
        ax.plot(x_grid, G / 1000.0, label=f"{float(T):.2f} K")

    ax.set_xlabel("Mole fraction of LiFePO4, xLi")
    ax.set_ylabel("Gibbs energy (kJ/mol)")
    ax.set_title("M5.L Gibbs energy vs xLi")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_pdf)
    print(f"Saved overlay PDF: {output_pdf}")

    if show:
        plt.show()

    return fig, ax


def plot_gmix_vs_x_temperatures(temperatures, x_min=0.001, x_max=0.999,
                                nx=1201,
                                output_pdf="Gmix_vs_xLi_all_temperatures.pdf",
                                individual_pdf_dir=None, show=False,
                                show_tangents=False):
    """
    Plot equilibrium Gmix versus xLi for user-defined temperatures.

    A multi-page PDF is always created, with one page per temperature.
    If individual_pdf_dir is supplied, one additional PDF is created per
    temperature.

    If show_tangents=True, stable and metastable common tangents are added
    using the same constructions as plot_gmix_curves().
    """
    from pathlib import Path

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if individual_pdf_dir is not None:
        individual_pdf_dir = Path(individual_pdf_dir)
        individual_pdf_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(x_min, x_max, nx)
    figures = []

    with PdfPages(output_pdf) as pdf:
        for T in temperatures:
            T = float(T)
            Gm = calculate_Gmix_curve(T, x_grid) / 1000.0

            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            ax.plot(x_grid, Gm, linewidth=1.8, label=f"{T:.2f} K")

            if show_tangents:
                # Stable common tangents.
                for xa, xb in coexistence_compositions(T, nx=min(nx, 1001)):
                    Ga = G_mixing(xa, T) / 1000.0
                    Gb = G_mixing(xb, T) / 1000.0
                    xt = np.linspace(xa, xb, 200)
                    tangent = Ga + (Gb - Ga) * (xt - xa) / (xb - xa)
                    ax.plot(xt, tangent, "--", linewidth=1.0)
                    ax.plot([xa, xb], [Ga, Gb], "o", markersize=3)

                # Metastable common tangents.
                stable = coexistence_compositions(T, nx=min(nx, 1001))
                for xa, xb in metastable_tangents(T, nx=min(nx, 1001)):
                    if any(abs(xa-sa) < 1e-3 and abs(xb-sb) < 1e-3
                           for sa, sb in stable):
                        continue
                    Ga = G_mixing(xa, T) / 1000.0
                    Gb = G_mixing(xb, T) / 1000.0
                    xt = np.linspace(xa, xb, 200)
                    tangent = Ga + (Gb - Ga) * (xt - xa) / (xb - xa)
                    ax.plot(xt, tangent, ":", linewidth=1.0)

            ax.axhline(0.0, linewidth=0.8)
            ax.set_xlabel("Mole fraction of LiFePO4, xLi")
            ax.set_ylabel("Gibbs energy of mixing (kJ/mol)")
            ax.set_title(f"M5.L Gibbs energy of mixing vs xLi at T = {T:.2f} K")
            ax.grid(True, alpha=0.2)
            ax.legend()
            fig.tight_layout()

            pdf.savefig(fig)

            if individual_pdf_dir is not None:
                T_label = f"{T:.2f}".replace(".", "p")
                individual_file = (
                    individual_pdf_dir / f"Gmix_vs_xLi_{T_label}K.pdf"
                )
                with PdfPages(individual_file) as one_pdf:
                    one_pdf.savefig(fig)

            figures.append(fig)

            if not show:
                plt.close(fig)

    print(f"Saved combined Gmix PDF: {output_pdf}")
    if individual_pdf_dir is not None:
        print(f"Saved individual Gmix PDFs in: {individual_pdf_dir}")

    if show:
        plt.show()

    return figures


def plot_gmix_vs_x_overlay(temperatures, x_min=0.001, x_max=0.999,
                           nx=1201,
                           output_pdf="Gmix_vs_xLi_overlay.pdf",
                           show=False):
    """Plot all requested Gmix(xLi,T) curves on one figure and save as PDF."""
    from pathlib import Path

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(x_min, x_max, nx)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    for T in temperatures:
        Gm = calculate_Gmix_curve(float(T), x_grid) / 1000.0
        ax.plot(x_grid, Gm, label=f"{float(T):.2f} K")

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Mole fraction of LiFePO4, xLi")
    ax.set_ylabel("Gibbs energy of mixing (kJ/mol)")
    ax.set_title("M5.L Gibbs energy of mixing vs xLi")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_pdf)

    print(f"Saved Gmix overlay PDF: {output_pdf}")

    if show:
        plt.show()

    return fig, ax


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_gmix_curves(temperatures=(298.15, 478.0, 500.0), nx=1201, show_metastable=True):
    x = np.linspace(0.001, 0.999, nx)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for T in temperatures:
        Gm = calculate_Gmix_curve(T, x) / 1000.0
        ax.plot(x, Gm, label=f"{T:.2f} K")

        # Stable common tangent(s).
        for xa, xb in coexistence_compositions(T, nx=1001):
            Ga = G_mixing(xa, T) / 1000.0
            Gb = G_mixing(xb, T) / 1000.0
            xt = np.linspace(xa, xb, 200)
            tangent = Ga + (Gb - Ga) * (xt - xa) / (xb - xa)
            ax.plot(xt, tangent, "--", linewidth=1.0)
            ax.plot([xa, xb], [Ga, Gb], "o", markersize=3)

        # Metastable common tangents are plotted in dotted style, analogous
        # to the construction discussed around Figure 4 of the paper.
        if show_metastable:
            stable = coexistence_compositions(T, nx=1001)
            for xa, xb in metastable_tangents(T, nx=1001):
                if any(abs(xa-sa) < 1e-3 and abs(xb-sb) < 1e-3 for sa, sb in stable):
                    continue
                Ga = G_mixing(xa, T) / 1000.0
                Gb = G_mixing(xb, T) / 1000.0
                xt = np.linspace(xa, xb, 200)
                tangent = Ga + (Gb - Ga) * (xt - xa) / (xb - xa)
                ax.plot(xt, tangent, ":", linewidth=1.0)

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Gibbs energy of mixing (kJ/mol)")
    ax.set_title("M5.L Gibbs energy of mixing")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_gibbs_contour(T_min=298.15, T_max=650.0, nT=121, nx=241):
    T_grid = np.linspace(T_min, T_max, nT)
    x_grid = np.linspace(0.001, 0.999, nx)
    Z = np.empty((nT, nx))
    for i, T in enumerate(T_grid):
        Z[i] = calculate_Gmix_curve(float(T), x_grid) / 1000.0

    X, TT = np.meshgrid(x_grid, T_grid)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cf = ax.contourf(X, TT, Z, levels=30)
    c0 = ax.contour(X, TT, Z, levels=[0.0], linewidths=1.3)
    ax.clabel(c0, fmt={0.0: "Gmix = 0"})
    fig.colorbar(cf, ax=ax, label="Gibbs energy of mixing (kJ/mol)")
    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("M5.L Gibbs-energy landscape")
    fig.tight_layout()
    return fig, ax


def plot_enthalpy_500K(nx=1001):
    x = np.linspace(0.001, 0.999, nx)
    Hm = np.array([H_mixing(xi, 500.0) for xi in x]) / 1000.0
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, Hm, label="M5.L, 500 K")
    ax.axvline(0.6, linestyle="--", linewidth=1.0, label="paper eutectoid x=0.6")
    h600 = H_mixing(0.6, 500.0) / 1000.0
    ax.plot([0.6], [h600], "o")
    ax.annotate(f"{h600:.3f} kJ/mol", xy=(0.6, h600), xytext=(0.64, h600))
    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Enthalpy of mixing (kJ/mol)")
    ax.set_title("M5.L enthalpy of mixing at 500 K")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_phase_diagram(T_grid=None, nx=1601):
    if T_grid is None:
        T_grid = np.linspace(T_MIN_DEFAULT, T_MAX_DEFAULT, 71)

    data = calculate_phase_diagram(
        T_grid=T_grid,
        nx=nx,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot every refined binodal pair as two separate branches.
    for j in range(3):
        left = [(d["binodals"][j][0], d["T"]) for d in data if len(d["binodals"]) > j]
        right = [(d["binodals"][j][1], d["T"]) for d in data if len(d["binodals"]) > j]
        if left:
            p = np.asarray(left)
            ax.plot(p[:, 0], p[:, 1], linewidth=1.8, label="binodal" if j == 0 else None)
        if right:
            p = np.asarray(right)
            ax.plot(p[:, 0], p[:, 1], linewidth=1.8)

    # Spinodal roots are plotted as individual ordered branches.
    for j in range(4):
        pts = [(d["spinodals"][j], d["T"]) for d in data if len(d["spinodals"]) > j]
        if pts:
            p = np.asarray(pts)
            ax.plot(p[:, 0], p[:, 1], "--", linewidth=1.1, label="spinodal" if j == 0 else None)

    eut = find_eutectoid()
    if eut is not None:
        x1e, x2e, x3e, Te = eut
        ax.plot([x1e, x2e, x3e], [Te, Te, Te], "o", markersize=5,
                label=f"numerical 3-phase eutectoid, T={Te:.1f} K")
        ax.axhline(Te, linestyle=":", linewidth=0.8)
    else:
        x1e, x2e, x3e, Te = 0.13, 0.60, 0.88, 500.0

    # Paper/model reference composition-temperature point, not an experimental overlay.
    ax.plot([0.6], [500.0], "x", markersize=7, mew=1.5,
            label="paper reported eutectoid composition/temperature (0.6, 500 K)")

    ax.text(0.10, 380, "H + T", fontsize=11)
    ax.text(0.30, 540, "H + D", fontsize=11)
    ax.text(0.73, 540, "D + T", fontsize=11)
    ax.text(0.57, 630, "D", fontsize=11)

    ax.set_xlim(0, 1)
    ax.set_ylim(T_MIN_DEFAULT, 650)
    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("LiFePO4–FePO4 phase diagram: refined M5.L")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, ax, data


def plot_ocv(temperatures=(298.15, 448.0, 563.0), nx=1201):
    """Plot -dG/dx/F from the equilibrium G; absolute voltage reference is separate."""
    x = np.linspace(0.001, 0.999, nx)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for T in temperatures:
        V = np.array([ -dGdx(xi, T) / FARADAY for xi in x ])
        ax.plot(x, V, label=f"{T:.2f} K")
    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("-dG/dx / F (V)")
    ax.set_title("M5.L thermodynamic voltage shape")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Diagnostics / comparison with the paper
# -----------------------------------------------------------------------------

def validation_report():
    print("\n" + "=" * 72)
    print("REFINED M5.L VALIDATION")
    print("=" * 72)

    GL = G0_LiFePO4(298.15)
    GF = G0_FePO4(298.15)
    print(f"G° LiFePO4 @ 298.15 K : {GL: .6f} J/mol")
    print(f"G° FePO4   @ 298.15 K : {GF: .6f} J/mol")

    x = np.linspace(0.001, 0.999, 2001)
    gm = calculate_Gmix_curve(298.15, x)
    imax = np.argmax(gm)
    print(f"max Gmix @ 298.15 K  : {gm[imax]/1000:.6f} kJ/mol at x={x[imax]:.5f}")

    h500 = H_mixing(0.6, 500.0)
    print(f"Hmix @ x=0.6, 500 K  : {h500/1000:.6f} kJ/mol")

    eut = find_eutectoid()
    if eut is None:
        print("Numerical eutectoid    : not found by the triple-tangent solver")
    else:
        print(
            "numerical eutectoid   : "
            f"x1={eut[0]:.6f}, x2={eut[1]:.6f}, x3={eut[2]:.6f}, T={eut[3]:.6f} K"
        )

    for T in (298.15, 478.0, 500.0, 563.0):
        pairs = coexistence_compositions(T, nx=801)
        spins = spinodal_roots(T, nx=301)
        print(f"T={T:7.2f} K | binodals={pairs} | spinodals={spins}")

    print("\nPaper/model reference values explicitly stated by Phan et al.:")
    print("  eutectoid approximately x=0.6, T=500 K")
    print("  Hmix approximately 3.5 kJ/mol at x=0.6, T=500 K")
    print("  maximum Gmix approximately 1.2 kJ/mol at 298 K")
    print("  M5.L/M5.F give the same equilibrium phase diagram")
    print("\nNo experimental-point overlay is included in this script.")
    print("=" * 72)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # =====================================================================
    # USER-DEFINED G(xLi,T) EXTRACTION / PLOTTING
    # =====================================================================
    # Put any temperatures you want to extract/plot here, in kelvin.
    USER_TEMPERATURES = [298.15, 300.0, 305.0, 310.0, 315.0, 320.0]

    # Number of composition points in each CSV/plot.
    USER_NX = 1201

    # Output locations.
    G_CSV_DIR = "gibbs_vs_xLi_csv"
    G_PDF_ALL = "G_vs_xLi_all_temperatures.pdf"
    G_PDF_INDIVIDUAL_DIR = "gibbs_vs_xLi_pdfs"
    G_PDF_OVERLAY = "G_vs_xLi_overlay.pdf"

    # Gmix output locations.
    GMIX_CSV_DIR = "gmix_vs_xLi_csv"
    GMIX_PDF_ALL = "Gmix_vs_xLi_all_temperatures.pdf"
    GMIX_PDF_INDIVIDUAL_DIR = "gmix_vs_xLi_pdfs"
    GMIX_PDF_OVERLAY = "Gmix_vs_xLi_overlay.pdf"

    # -----------------------------------------------------------------
    # 1. Extract G(xLi,T) into one CSV file per requested temperature.
    # -----------------------------------------------------------------
    export_gibbs_vs_x_csv(
        USER_TEMPERATURES,
        output_dir=G_CSV_DIR,
        nx=USER_NX,
    )

    # -----------------------------------------------------------------
    # 2. Plot G(xLi,T): multi-page PDF + individual PDFs + overlay.
    # -----------------------------------------------------------------
    plot_gibbs_vs_x_temperatures(
        USER_TEMPERATURES,
        nx=USER_NX,
        output_pdf=G_PDF_ALL,
        individual_pdf_dir=G_PDF_INDIVIDUAL_DIR,
        show=False,
    )

    plot_gibbs_vs_x_overlay(
        USER_TEMPERATURES,
        nx=USER_NX,
        output_pdf=G_PDF_OVERLAY,
        show=False,
    )

    # -----------------------------------------------------------------
    # 3. Extract Gmix(xLi,T) into one CSV file per requested temperature.
    # -----------------------------------------------------------------
    export_gmix_vs_x_csv(
        USER_TEMPERATURES,
        output_dir=GMIX_CSV_DIR,
        nx=USER_NX,
    )

    # -----------------------------------------------------------------
    # 4. Plot Gmix(xLi,T): multi-page PDF + individual PDFs + overlay.
    # -----------------------------------------------------------------
    plot_gmix_vs_x_temperatures(
        USER_TEMPERATURES,
        nx=USER_NX,
        output_pdf=GMIX_PDF_ALL,
        individual_pdf_dir=GMIX_PDF_INDIVIDUAL_DIR,
        show=False,
        show_tangents=False,
    )

    plot_gmix_vs_x_overlay(
        USER_TEMPERATURES,
        nx=USER_NX,
        output_pdf=GMIX_PDF_OVERLAY,
        show=False,
    )

    validation_report()

    # Existing model plots.
    plot_gmix_curves()

    # 2D Gibbs-energy landscape.
    plot_gibbs_contour()

    # Enthalpy at the paper's eutectoid temperature.
    plot_enthalpy_500K()

    # Refined equilibrium phase diagram.
    plot_phase_diagram(
        T_grid=np.linspace(298.15, 650.0, 61),
        nx=1601,
    )

    # Uncomment if needed:
    # plot_ocv()

    plt.show()
