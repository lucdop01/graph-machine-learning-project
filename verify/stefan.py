"""Check 3: latent heat. 1D melting (Neumann's one-phase Stefan problem) on the 2D solver with a 3-cell-high strip.
Solid starts at the melting temperature, the left column is held at T_hot, and the melt front must follow
s(t) = 2*lambda*sqrt(alpha*t), with lambda*exp(lambda^2)*erf(lambda) = St/sqrt(pi), St = c*(T_hot - T_melt)/L.
Usage: python -m verify.stefan     Pass: front within 2% once it has moved >= 20 cells."""
import numpy as np
from scipy.special import erf
from scipy.optimize import brentq
from solver.solver import simulate

RHO, K, C, L = 4420.0, 30.0, 750.0, 286e3          # constant liquid-like properties, Ti-6Al-4V latent heat
T_S, DT_M, T_HOT = 1900.0, 2.0, 2400.0             # narrow 2 K melting range, close to Neumann's sharp front
T_MELT = T_S + DT_M / 2
_T = np.linspace(300.0, 6000.0, 60001)
def _phi(T):
    return np.clip((T - T_S) / DT_M, 0.0, 1.0)

_H = C * (_T - 300.0) + L * _phi(_T)
props = dict(k=lambda T: np.full_like(np.asarray(T, float), K), cp=lambda T: np.full_like(np.asarray(T, float), C),
             H_from_T=lambda T: np.interp(T, _T, _H), T_from_H=lambda H: np.interp(H, _H, _T))
alpha = K / (RHO * C)
St = C * (T_HOT - T_MELT) / L
lam = brentq(lambda x: x * np.exp(x * x) * erf(x) - St / np.sqrt(np.pi), 1e-6, 5.0)

dx, nx, ny = 2e-6, 400, 3
mask = np.zeros((nx, ny), bool)
mask[0, :] = True
r = simulate(0.0, 1.0, [(0.0, 0.0), (1e-9, 0.0)], nx=nx, ny=ny, dx=dx, t_cool=9e-3, save_every=4000,
             props=props, T_init=T_S, fixed_mask=mask, fixed_T=T_HOT)
print(f"St = {St:.3f}, lambda = {lam:.4f}, alpha = {alpha:.3e} m^2/s, steps = {r['n_steps']}")
worst = 0.0
for t, T in zip(r["t"][1:], r["T"][1:]):
    phi = _phi(T[:, 1].astype(float))
    s_num = dx / 2 + phi[1:].sum() * dx              # melted length measured from the held cell's centre
    s_ex = 2 * lam * np.sqrt(alpha * t)
    err = 100 * (s_num - s_ex) / s_ex
    if s_ex >= 20 * dx:
        worst = max(worst, abs(err))
    print(f"t = {t*1e3:5.2f} ms: front {s_num*1e6:6.1f} um, exact {s_ex*1e6:6.1f} um, error {err:+.2f}%")
print(f"worst error once the front passed 20 cells: {worst:.2f}%  {'PASS' if worst < 2 else 'FAIL'}")
if worst >= 2:
    raise SystemExit(1)
