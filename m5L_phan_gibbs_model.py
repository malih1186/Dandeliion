
"""
M5.L Gibbs-energy model for the LiFePO4-FePO4 olivine join.

Based on:
Phan et al., "Modelling of phase equilibria of LiFePO4-FePO4 olivine join",
Canadian Journal of Chemical Engineering, 97 (2019) 2224-2233,
DOI: 10.1002/cjce.23416

The model is the 5-stoichiometry M5.L CEF model:
    (Li+,Va)3 (Li+,Va)2 (Fe2+,Fe3+)5 (P5+)5 (O2-)20

The first Li/vacancy sublattice is split 3:2, giving one internal
long-range-order variable after imposing the overall composition.

This script produces:
  1. Gmix(x) curves at selected temperatures
  2. Gmix(x,T) contour plot
  3. Hmix(x) at 500 K
  4. equilibrium phase diagram from the lower convex envelope
  5. spinodal boundaries from d2G/dx2 = 0
  6. optional OCV curves (relative voltage; see note in plot_ocv_curves)

Dependencies:
    numpy
    scipy
    matplotlib

Install:
    python -m pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

R = 8.31446261815324          # J mol^-1 K^-1
FARADAY = 96485.33212         # C mol^-1
T0 = 298.15                   # K
T_SWITCH_L = 773.0            # K

# Pure-compound thermodynamic data from Table 1
DH298_L = -161602.0           # J/mol
S298_L = 130.95               # J/(mol K)

DH298_F = -127923.0           # J/mol
S298_F = 122.21                # J/(mol K)

# Optimized Gibbs-energy parameters from Table 1
DG_LI_PLUS = 15000.0          # J/mol
DG_FE_MINUS = 15000.0         # J/mol
DG_ADJUST1 = 19500.0           # J/mol
DG_ADJUST2 = 29250.0           # J/mol


# ---------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------

def xlogx(x):
    """Return x*ln(x), with the thermodynamic limit 0*ln(0)=0."""
    if x <= 0.0:
        return 0.0
    return x * np.log(x)


# ---------------------------------------------------------------------
# Heat capacities
# ---------------------------------------------------------------------

def Cp_LiFePO4(T):
    """
    Heat capacity of LiFePO4 from Table 1.

    250 <= T <= 773 K:
        Cp = 41.881336 + 0.78278483*T
             + 890694.39/T^2
             - 0.0010255433*T^2
             + 5.0862948e-7*T^3

    773 <= T <= 1473 K:
        Cp = 247.915258 - 0.01771626*T
             - 28310678.63/T^2
    """
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
    """
    Heat capacity of FePO4 from Table 1:
        Cp = 114.52232 + 0.05966384*T - 2690312/T^2
    """
    T = np.asarray(T)
    return 114.52232 + 0.05966384 * T - 2690312.0 / T**2


# ---------------------------------------------------------------------
# Analytical Cp integrals
# ---------------------------------------------------------------------

def _CpInt_L_segment1(T):
    return (
        41.881336 * T
        + 0.78278483 * T**2 / 2.0
        - 890694.39 / T
        - 0.0010255433 * T**3 / 3.0
        + 5.0862948e-7 * T**4 / 4.0
    )


def _CpOverTInt_L_segment1(T):
    return (
        41.881336 * np.log(T)
        + 0.78278483 * T
        - 890694.39 / (2.0 * T**2)
        - 0.0010255433 * T**2 / 2.0
        + 5.0862948e-7 * T**3 / 3.0
    )


def _CpInt_L_segment2(T):
    return (
        247.915258 * T
        - 0.01771626 * T**2 / 2.0
        + 28310678.63 / T
    )


def _CpOverTInt_L_segment2(T):
    return (
        247.915258 * np.log(T)
        - 0.01771626 * T
        - 28310678.63 / (2.0 * T**2)
    )


def int_Cp_L(Ta, Tb):
    """Integral of Cp(LiFePO4) dT from Ta to Tb."""
    if Tb <= T_SWITCH_L:
        return _CpInt_L_segment1(Tb) - _CpInt_L_segment1(Ta)
    if Ta >= T_SWITCH_L:
        return _CpInt_L_segment2(Tb) - _CpInt_L_segment2(Ta)

    return (
        _CpInt_L_segment1(T_SWITCH_L)
        - _CpInt_L_segment1(Ta)
        + _CpInt_L_segment2(Tb)
        - _CpInt_L_segment2(T_SWITCH_L)
    )


def int_Cp_over_T_L(Ta, Tb):
    """Integral of Cp(LiFePO4)/T dT from Ta to Tb."""
    if Tb <= T_SWITCH_L:
        return _CpOverTInt_L_segment1(Tb) - _CpOverTInt_L_segment1(Ta)
    if Ta >= T_SWITCH_L:
        return _CpOverTInt_L_segment2(Tb) - _CpOverTInt_L_segment2(Ta)

    return (
        _CpOverTInt_L_segment1(T_SWITCH_L)
        - _CpOverTInt_L_segment1(Ta)
        + _CpOverTInt_L_segment2(Tb)
        - _CpOverTInt_L_segment2(T_SWITCH_L)
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
    """Integral of Cp(FePO4) dT from Ta to Tb."""
    return _CpInt_F(Tb) - _CpInt_F(Ta)


def int_Cp_over_T_F(Ta, Tb):
    """Integral of Cp(FePO4)/T dT from Ta to Tb."""
    return _CpOverTInt_F(Tb) - _CpOverTInt_F(Ta)


# ---------------------------------------------------------------------
# Pure-compound Gibbs energies and enthalpies
# ---------------------------------------------------------------------

def G0_LiFePO4(T):
    """
    Standard Gibbs energy of pure LiFePO4.

    G(T) = H298 + integral(Cp dT)
           - T*[S298 + integral(Cp/T dT)]
    """
    return (
        DH298_L
        + int_Cp_L(T0, T)
        - T * (S298_L + int_Cp_over_T_L(T0, T))
    )


def G0_FePO4(T):
    """Standard Gibbs energy of pure FePO4."""
    return (
        DH298_F
        + int_Cp_F(T0, T)
        - T * (S298_F + int_Cp_over_T_F(T0, T))
    )


def H0_LiFePO4(T):
    """Standard enthalpy of LiFePO4 at T."""
    return DH298_L + int_Cp_L(T0, T)


def H0_FePO4(T):
    """Standard enthalpy of FePO4 at T."""
    return DH298_F + int_Cp_F(T0, T)


# ---------------------------------------------------------------------
# M5.L end-member Gibbs energies
# ---------------------------------------------------------------------

def M5L_endmembers(T):
    """
    Return the 8 M5.L end-member Gibbs energies.

    Ordering:
      0 Li:Li:Fe2
      1 Va:Va:Fe3
      2 Li:Li:Fe3
      3 Va:Va:Fe2
      4 Va:Li:Fe2
      5 Va:Li:Fe3
      6 Li:Va:Fe2
      7 Li:Va:Fe3

    Values are the 5-formula-unit CEF end-member energies from Table 1.
    """
    GL = G0_LiFePO4(T)
    GF = G0_FePO4(T)

    return np.array([
        5.0 * GL,
        5.0 * GF,
        5.0 * GL + 5.0 * DG_LI_PLUS,
        5.0 * GF + 5.0 * DG_FE_MINUS,
        3.0 * GF + 3.0 * DG_FE_MINUS + 2.0 * GL,
        2.0 * GL + 2.0 * DG_LI_PLUS + 3.0 * GF + DG_ADJUST1 * 3.0,
        3.0 * GL + 2.0 * GF + 2.0 * DG_FE_MINUS - DG_ADJUST1,
        2.0 * GF + 3.0 * GL + 3.0 * DG_LI_PLUS - DG_ADJUST2,
    ])


# ---------------------------------------------------------------------
# Composition relations for M5.L
# ---------------------------------------------------------------------

def a_bounds(x):
    """
    Allowed range of a = y_Li on the 3-site sublattice.

    Overall composition:
        x = (3*a + 2*b)/5

    Therefore:
        b = (5*x - 3*a)/2
    """
    lo = max(0.0, (5.0 * x - 2.0) / 3.0)
    hi = min(1.0, 5.0 * x / 3.0)
    return lo, hi


def b_from_a_x(a, x):
    """Return b = y_Li on the 2-site sublattice."""
    return (5.0 * x - 3.0 * a) / 2.0


# ---------------------------------------------------------------------
# M5.L Gibbs energy
# ---------------------------------------------------------------------

def G_M5L(a, x, T, endmembers=None):
    """
    M5.L Gibbs energy per mole of LixFePO4 formula unit.

    The CEF end-member energies in Table 1 contain 5 formula units,
    so the end-member interpolation is divided by 5.

    The configurational entropy is:
        RT [
          3/5 * sum(y_Li, y_Va)*ln(y)
        + 2/5 * sum(y_Li, y_Va)*ln(y)
        +       sum(y_Fe2, y_Fe3)*ln(y)
        ]

    There is no excess Gibbs-energy term in the paper's M5.L model.
    """
    if x < 0.0 or x > 1.0:
        return np.inf

    lo, hi = a_bounds(x)
    if a < lo - 1e-12 or a > hi + 1e-12:
        return np.inf

    a = min(max(a, 0.0), 1.0)
    b = b_from_a_x(a, x)
    b = min(max(b, 0.0), 1.0)

    if endmembers is None:
        endmembers = M5L_endmembers(T)

    # End-member fractions
    f = np.array([
        a * b * x,
        (1.0 - a) * (1.0 - b) * (1.0 - x),
        a * b * (1.0 - x),
        (1.0 - a) * (1.0 - b) * x,
        (1.0 - a) * b * x,
        (1.0 - a) * b * (1.0 - x),
        a * (1.0 - b) * x,
        a * (1.0 - b) * (1.0 - x),
    ])

    # Table-1 end-member energies are for the 5-stoichiometry model.
    G_end = np.dot(f, endmembers) / 5.0

    G_config = R * T * (
        (3.0 / 5.0) * (
            xlogx(a) + xlogx(1.0 - a)
        )
        + (2.0 / 5.0) * (
            xlogx(b) + xlogx(1.0 - b)
        )
        + xlogx(x)
        + xlogx(1.0 - x)
    )

    # GE = 0 in the paper's M5.L model.
    return G_end + G_config


def equilibrium_M5L(x, T):
    """
    Minimize M5.L with respect to the internal LRO variable a.

    Returns:
        G_eq, a_eq, b_eq
    """
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
        options={"xatol": 1e-10},
    )

    # Explicitly compare the two boundaries as well.
    candidates = [
        (result.fun, result.x),
        (G_M5L(lo, x, T, E), lo),
        (G_M5L(hi, x, T, E), hi),
    ]

    Gmin, amin = min(candidates, key=lambda q: q[0])
    bmin = b_from_a_x(amin, x)

    return Gmin, amin, bmin


# ---------------------------------------------------------------------
# Composition curves
# ---------------------------------------------------------------------

def calculate_G_curve(T, x_grid):
    """Equilibrium G(x,T) over a supplied composition grid."""
    G = np.empty_like(x_grid, dtype=float)
    A = np.empty_like(x_grid, dtype=float)
    B = np.empty_like(x_grid, dtype=float)

    for i, x in enumerate(x_grid):
        G[i], A[i], B[i] = equilibrium_M5L(float(x), T)

    return G, A, B


def G_mixing(x, T):
    """
    Gibbs energy of mixing relative to:
        (1-x) FePO4 + x LiFePO4
    """
    G, _, _ = equilibrium_M5L(x, T)
    return G - ((1.0 - x) * G0_FePO4(T) + x * G0_LiFePO4(T))


def calculate_Gmix_curve(T, x_grid):
    """Vectorized-style wrapper for Gmix over a composition grid."""
    return np.array([G_mixing(float(x), T) for x in x_grid])


# ---------------------------------------------------------------------
# Enthalpy of mixing
# ---------------------------------------------------------------------

def H_mixing(x, T):
    """
    Enthalpy of mixing of the equilibrium M5.L solution.

    Because all optimized DG parameters in the paper are temperature
    independent and the configurational term is proportional to T,
    the configurational contribution has zero enthalpy. Therefore Hmix
    can be evaluated directly from the optimized end-member mixture.
    """
    G, a, b = equilibrium_M5L(x, T)

    E_H = np.array([
        5.0 * H0_LiFePO4(T),
        5.0 * H0_FePO4(T),
        5.0 * H0_LiFePO4(T),
        5.0 * H0_FePO4(T),
        2.0 * H0_LiFePO4(T) + 3.0 * H0_FePO4(T),
        2.0 * H0_LiFePO4(T) + 3.0 * H0_FePO4(T),
        3.0 * H0_LiFePO4(T) + 2.0 * H0_FePO4(T),
        3.0 * H0_LiFePO4(T) + 2.0 * H0_FePO4(T),
    ])

    # Temperature-independent Gibbs parameters contribute directly
    # to enthalpy.
    E_const = np.array([
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

    H_eq = np.dot(fractions, E_H + E_const) / 5.0

    return H_eq - (
        (1.0 - x) * H0_FePO4(T)
        + x * H0_LiFePO4(T)
    )


# ---------------------------------------------------------------------
# Lower convex envelope = equilibrium phase separation
# ---------------------------------------------------------------------

def lower_convex_hull(x, G):
    """
    Return indices of the lower convex envelope.

    A missing set of points between two hull indices means those
    compositions lie inside a two-phase equilibrium region.
    """
    hull = []

    for i in range(len(x)):
        hull.append(i)

        while len(hull) >= 3:
            i1, i2, i3 = hull[-3:]

            slope1 = (G[i2] - G[i1]) / (x[i2] - x[i1])
            slope2 = (G[i3] - G[i2]) / (x[i3] - x[i2])

            # Convex envelope requires monotonically increasing slopes.
            if slope2 <= slope1:
                hull.pop(-2)
            else:
                break

    return np.asarray(hull, dtype=int)


def coexistence_gaps(x, G):
    """
    Find two-phase intervals from the lower convex hull.

    Returns a list [(x_left, x_right), ...].
    """
    hull = lower_convex_hull(x, G)
    gaps = []

    for j in range(len(hull) - 1):
        i1 = hull[j]
        i2 = hull[j + 1]

        if i2 > i1 + 1:
            gaps.append((x[i1], x[i2]))

    return gaps


def convexified_G(x, G):
    """Return the piecewise-linear lower convex envelope."""
    hull = lower_convex_hull(x, G)
    Gh = np.empty_like(G)

    for j in range(len(hull) - 1):
        i1 = hull[j]
        i2 = hull[j + 1]

        m = (G[i2] - G[i1]) / (x[i2] - x[i1])
        Gh[i1:i2 + 1] = G[i1] + m * (x[i1:i2 + 1] - x[i1])

    return Gh, hull


# ---------------------------------------------------------------------
# Spinodal boundary
# ---------------------------------------------------------------------

def spinodal_roots(T, x_grid):
    """
    Estimate spinodal compositions from d2G/dx2 = 0.

    Finite differences are used because the equilibrium G already
    contains an internal minimization with respect to the LRO variable.
    """
    G, _, _ = calculate_G_curve(T, x_grid)

    dGdx = np.gradient(G, x_grid)
    d2Gdx2 = np.gradient(dGdx, x_grid)

    roots = []

    for i in range(len(x_grid) - 1):
        y1 = d2Gdx2[i]
        y2 = d2Gdx2[i + 1]

        if np.isfinite(y1) and np.isfinite(y2):
            if y1 == 0.0:
                roots.append(x_grid[i])
            elif y1 * y2 < 0.0:
                # Linear interpolation of the zero crossing.
                xr = x_grid[i] - y1 * (
                    x_grid[i + 1] - x_grid[i]
                ) / (y2 - y1)
                roots.append(xr)

    return roots


# ---------------------------------------------------------------------
# Plot 1: Gibbs-energy curves
# ---------------------------------------------------------------------

def plot_gibbs_curve(T, x_grid=None, show_tangent=True):
    """
    Plot Gmix(x) for one temperature.

    If show_tangent=True, the equilibrium common tangent is plotted
    wherever the lower convex hull contains a two-phase gap.
    """
    if x_grid is None:
        x_grid = np.linspace(0.001, 0.999, 401)

    Gmix = calculate_Gmix_curve(T, x_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_grid, Gmix / 1000.0, label=f"{T:.0f} K")

    if show_tangent:
        G_total, _, _ = calculate_G_curve(T, x_grid)
        Gh, hull = convexified_G(x_grid, G_total)

        for j in range(len(hull) - 1):
            i1 = hull[j]
            i2 = hull[j + 1]

            if i2 > i1 + 1:
                ax.plot(
                    x_grid[i1:i2 + 1],
                    (
                        Gh[i1:i2 + 1]
                        - ((1.0 - x_grid[i1:i2 + 1]) * G0_FePO4(T)
                           + x_grid[i1:i2 + 1] * G0_LiFePO4(T))
                    ) / 1000.0,
                    "--",
                    label="common tangent",
                )

    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Gibbs energy of mixing (kJ/mol)")
    ax.set_title(f"M5.L Gibbs energy of mixing at {T:.0f} K")
    ax.legend()
    ax.grid(True, alpha=0.25)

    return fig, ax


# ---------------------------------------------------------------------
# Plot 2: Contour map
# ---------------------------------------------------------------------

def plot_gibbs_contour(
    T_min=298.0,
    T_max=650.0,
    nT=101,
    nx=151,
):
    """
    Contour plot of Gmix(x,T).

    This is an additional visualization rather than a reproduction of
    a specific figure in the paper.
    """
    temperatures = np.linspace(T_min, T_max, nT)
    x_grid = np.linspace(0.001, 0.999, nx)

    Z = np.empty((nT, nx))

    for i, T in enumerate(temperatures):
        Z[i, :] = calculate_Gmix_curve(T, x_grid) / 1000.0

    X, TT = np.meshgrid(x_grid, temperatures)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    levels = 25
    cf = ax.contourf(X, TT, Z, levels=levels)
    c0 = ax.contour(X, TT, Z, levels=[0.0], linewidths=1.5)

    ax.clabel(c0, fmt={0.0: "Gmix = 0"}, inline=True)

    ax.axhline(500.0, linestyle="--", linewidth=1.0, label="500 K")
    ax.axvline(0.6, linestyle="--", linewidth=1.0, label="x = 0.6")

    fig.colorbar(cf, ax=ax, label="Gibbs energy of mixing (kJ/mol)")

    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("M5.L Gibbs-energy landscape")
    ax.legend()

    return fig, ax


# ---------------------------------------------------------------------
# Plot 3: Enthalpy of mixing
# ---------------------------------------------------------------------

def plot_enthalpy_mixing_500K(x_grid=None):
    """Reproduce the calculated M5.L enthalpy-of-mixing curve at 500 K."""
    if x_grid is None:
        x_grid = np.linspace(0.001, 0.999, 401)

    Hmix = np.array([H_mixing(float(x), 500.0) for x in x_grid])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_grid, Hmix / 1000.0, label="M5.L, 500 K")
    ax.axvline(0.6, linestyle="--", linewidth=1.0, label="x = 0.6")

    x0 = 0.6
    h0 = H_mixing(x0, 500.0) / 1000.0
    ax.plot([x0], [h0], "o")

    ax.annotate(
        f"{h0:.2f} kJ/mol",
        xy=(x0, h0),
        xytext=(x0 + 0.04, h0 + 0.15),
        arrowprops=dict(arrowstyle="->"),
    )

    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Enthalpy of mixing (kJ/mol)")
    ax.set_title("M5.L enthalpy of mixing at 500 K")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return fig, ax


# ---------------------------------------------------------------------
# Plot 4: Phase diagram
# ---------------------------------------------------------------------

def calculate_phase_diagram(
    T_grid=None,
    nx=301,
):
    """
    Calculate equilibrium miscibility-gap boundaries from the lower
    convex envelope.

    Returns:
        results = list of dictionaries with temperature and gaps
    """
    if T_grid is None:
        T_grid = np.linspace(298.0, 650.0, 101)

    x_grid = np.linspace(0.001, 0.999, nx)

    results = []

    for T in T_grid:
        G, _, _ = calculate_G_curve(T, x_grid)
        gaps = coexistence_gaps(x_grid, G)

        results.append({
            "T": float(T),
            "gaps": gaps,
        })

    return results


def plot_phase_diagram(
    T_grid=None,
    nx=301,
    show_spinodal=True,
):
    """
    Plot the calculated M5.L equilibrium phase diagram.

    The solid curves are binodal/coexistence boundaries obtained from
    the lower convex envelope.

    Dashed curves are spinodal boundaries obtained from d2G/dx2 = 0.
    """
    if T_grid is None:
        T_grid = np.linspace(298.0, 650.0, 101)

    results = calculate_phase_diagram(T_grid=T_grid, nx=nx)

    # Branch containers
    low_left = []
    low_right = []
    left_gap_left = []
    left_gap_right = []
    right_gap_left = []
    right_gap_right = []

    for item in results:
        T = item["T"]
        gaps = item["gaps"]

        if len(gaps) == 1:
            xl, xr = gaps[0]

            # One broad low-temperature gap contains the eutectoid
            # composition. A single remaining gap on the left/right
            # is treated as the corresponding high-temperature branch.
            if xl < 0.6 < xr:
                low_left.append((xl, T))
                low_right.append((xr, T))
            elif xr <= 0.6:
                left_gap_left.append((xl, T))
                left_gap_right.append((xr, T))
            else:
                right_gap_left.append((xl, T))
                right_gap_right.append((xr, T))

        elif len(gaps) >= 2:
            xl1, xr1 = gaps[0]
            xl2, xr2 = gaps[1]

            left_gap_left.append((xl1, T))
            left_gap_right.append((xr1, T))
            right_gap_left.append((xl2, T))
            right_gap_right.append((xr2, T))

    fig, ax = plt.subplots(figsize=(8, 6))

    def plot_branch(points, label=None, linestyle="-"):
        if points:
            p = np.asarray(points)
            ax.plot(p[:, 0], p[:, 1], linestyle, label=label)

    plot_branch(low_left, "binodal")
    plot_branch(low_right)

    plot_branch(left_gap_left, "binodal")
    plot_branch(left_gap_right)

    plot_branch(right_gap_left)
    plot_branch(right_gap_right)

    # Eutectoid reported by the paper/model optimization.
    ax.plot(
        [0.6],
        [500.0],
        "o",
        markersize=7,
        label="eutectoid (x = 0.6, T = 500 K)",
    )
    ax.axhline(500.0, linestyle=":", linewidth=1.0)

    if show_spinodal:
        x_spin = np.linspace(0.01, 0.99, 301)
        spinodal_branches = {}

        for T in T_grid:
            roots = spinodal_roots(float(T), x_spin)
            for j, root in enumerate(roots):
                spinodal_branches.setdefault(j, []).append((root, T))

        first = True
        for points in spinodal_branches.values():
            p = np.asarray(points)
            if len(p) >= 2:
                ax.plot(
                    p[:, 0],
                    p[:, 1],
                    "--",
                    linewidth=1.0,
                    label="spinodal" if first else None,
                )
                first = False

    # Region labels, following the terminology used in Figure 1.
    ax.text(0.10, 390, "H + T", fontsize=11)
    ax.text(0.30, 545, "H + D", fontsize=11)
    ax.text(0.73, 545, "D + T", fontsize=11)
    ax.text(0.57, 640, "D", fontsize=11)

    ax.set_xlim(0, 1)
    ax.set_ylim(298, 650)
    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("LiFePO4–FePO4 phase diagram: M5.L")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    return fig, ax, results


# ---------------------------------------------------------------------
# Plot 5: Relative OCV curves
# ---------------------------------------------------------------------

def equilibrium_voltage_relative(T, x_grid):
    """
    Calculate the OCV shape from the convexified Gibbs energy.

    V = -dG/dx / F

    IMPORTANT:
    The paper's Figure 3 reports an absolute voltage around 3.5 V.
    The tabulated formation Gibbs energies alone do not uniquely provide
    the same absolute electrochemical reference. Therefore this function
    returns the thermodynamic voltage shape directly from the M5.L model,
    not a claimed absolute reproduction of Figure 3.
    """
    G, _, _ = calculate_G_curve(T, x_grid)
    Gh, _ = convexified_G(x_grid, G)

    dGdx = np.gradient(Gh, x_grid)
    V = -dGdx / FARADAY

    return V


def plot_ocv_curves(
    temperatures=(298.0, 448.0, 563.0),
    x_grid=None,
):
    """
    Plot relative OCV curves.

    A constant vertical shift can be applied afterwards if you want to
    compare the *shape* with the paper's approximately 3.5 V plateau.
    """
    if x_grid is None:
        x_grid = np.linspace(0.001, 0.999, 1001)

    fig, ax = plt.subplots(figsize=(8, 5))

    for T in temperatures:
        V = equilibrium_voltage_relative(T, x_grid)
        ax.plot(x_grid, V, label=f"{T:.0f} K")

    ax.set_xlabel("Mole fraction of LiFePO4, x")
    ax.set_ylabel("Thermodynamic voltage from -dG/dx / F (V)")
    ax.set_title("M5.L equilibrium OCV shape")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return fig, ax


# ---------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------

def print_validation():
    """
    Print a few useful checks against values discussed in the paper.
    """
    print("\n--- M5.L validation checks ---")

    print(
        "G°(LiFePO4, 298.15 K) = "
        f"{G0_LiFePO4(298.15):.3f} J/mol"
    )
    print(
        "G°(FePO4, 298.15 K)   = "
        f"{G0_FePO4(298.15):.3f} J/mol"
    )

    x_grid = np.linspace(0.001, 0.999, 401)
    gm298 = calculate_Gmix_curve(298.15, x_grid)

    imax = np.argmax(gm298)
    print(
        "Maximum Gmix at 298.15 K = "
        f"{gm298[imax] / 1000.0:.3f} kJ/mol "
        f"at x = {x_grid[imax]:.3f}"
    )

    h600 = H_mixing(0.6, 500.0)
    print(
        "Hmix at x=0.6, T=500 K = "
        f"{h600 / 1000.0:.3f} kJ/mol"
    )

    print(
        "The paper reports approximately 1.2 kJ/mol maximum Gmix "
        "at 298 K and approximately 3.5 kJ/mol Hmix at x=0.6, "
        "T=500 K."
    )
    print(
        "Small differences are expected because the paper does not "
        "publish its original FactSage/MATLAB numerical implementation."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print_validation()

    # Figure-like Gibbs curves
    plot_gibbs_curve(298.0)
    plot_gibbs_curve(478.0)

    # Additional curve at the eutectoid temperature
    plot_gibbs_curve(500.0)

    # Gibbs-energy contour map
    plot_gibbs_contour(
        T_min=298.0,
        T_max=650.0,
        nT=81,
        nx=121,
    )

    # Enthalpy of mixing, analogous to Figure 2
    plot_enthalpy_mixing_500K()

    # Phase diagram, analogous to Figure 1(c)
    plot_phase_diagram(
        T_grid=np.linspace(298.0, 650.0, 81),
        nx=251,
        show_spinodal=True,
    )

    # OCV shape. Uncomment if desired.
    # plot_ocv_curves()

    plt.show()
