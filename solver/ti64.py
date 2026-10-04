import numpy as np
RHO = 4420.0
L_FUSION = 286e3
T_S, T_L = 1878.0, 1928.0
T0 = 300.0
def k(T):
    T = np.minimum(T, T_L)
    return np.where(T < 1268.0, 1.25 + 0.015 * T, 13.15 + 0.012 * T)
def cp(T):
    T = np.minimum(T, T_L)
    return np.where(T < 1268.0, 483.04 + 0.215 * T, 412.7 + 0.180 * T)
def liquid_fraction(T):
    return np.clip((T - T_S) / (T_L - T_S), 0.0, 1.0)
_T = np.linspace(T0, 6000.0, 30000)   # well above boiling (3660 K) so nothing is clipped
_cp_mid = 0.5 * (cp(_T[1:]) + cp(_T[:-1]))
_H = np.concatenate([[0.0], np.cumsum(_cp_mid * np.diff(_T))]) + L_FUSION * liquid_fraction(_T)
def H_from_T(T): return np.interp(T, _T, _H)
def T_from_H(H): return np.interp(H, _H, _T)
R_SPOT = 50e-6
ETA = 0.35

# Constant properties without melting: week-1 pipeline data and the Rosenthal check
K_CONST, CP_CONST = 20.0, 600.0
CONST_PROPS = dict(
    k=lambda T: np.full_like(np.asarray(T, float), K_CONST),
    cp=lambda T: np.full_like(np.asarray(T, float), CP_CONST),
    H_from_T=lambda T: CP_CONST * (np.asarray(T, float) - T0),
    T_from_H=lambda H: T0 + np.asarray(H, float) / CP_CONST,
)
