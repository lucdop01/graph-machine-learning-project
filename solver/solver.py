"""2D thin-plate conduction model with melting (enthalpy method), explicit finite differences.
Insulated edges. Laser: Gaussian spot moving along a Path, switched off for t_cool at the end."""
import numpy as np
from .ti64 import RHO, T0, T_L, k, cp, H_from_T, T_from_H, R_SPOT, ETA
from .paths import Path

def simulate(P, v, waypoints, nx=200, ny=100, dx=10e-6, d=100e-6,
             save_every=10, t_cool=0.2e-3, props=None, cfl=0.2,
             T_init=None, fixed_mask=None, fixed_T=None):
    """props: optional dict overriding k, cp, H_from_T, T_from_H (constant-property runs, verification).
    cfl: time step as a fraction of dx^2/alpha_max (explicit stability limit is 0.25).
    T_init: initial temperature (scalar or nx x ny array), default T0.
    fixed_mask, fixed_T: cells held at a fixed temperature (Dirichlet), used by the Stefan check."""
    kf, cpf, HfT, TfH = k, cp, H_from_T, T_from_H
    if props:
        kf, cpf = props.get("k", kf), props.get("cp", cpf)
        HfT, TfH = props.get("H_from_T", HfT), props.get("T_from_H", TfH)

    path = Path(waypoints, v)
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")

    alpha_max = kf(np.array(T_L)) / (RHO * cpf(np.array(T0)))
    dt = float(cfl * dx**2 / alpha_max)                    # explicit stability limit is 0.25
    n_steps = int(np.ceil((path.duration + t_cool) / dt))
    q0 = 2 * ETA * P / (np.pi * R_SPOT**2 * d)             # peak power density, W/m^3
    hm = lambda a, b: 2 * a * b / (a + b)                  # conductivity between two cells

    T = np.full((nx, ny), T0, dtype=float) if T_init is None else np.broadcast_to(np.asarray(T_init, float), (nx, ny)).copy()
    if fixed_mask is not None:
        T[fixed_mask] = fixed_T
    H = HfT(T)
    out_t, out_T, out_laser = [0.0], [T.astype(np.float32)], [path.pos(0.0)]
    T_max = float(T.max())                                  # tracked every step, not only on saved frames
    for i in range(n_steps):
        t = i * dt
        if t <= path.duration:
            xl, yl = path.pos(t)
            q = q0 * np.exp(-2 * ((X - xl) ** 2 + (Y - yl) ** 2) / R_SPOT**2)
        else:
            q = 0.0
        kc = kf(T)
        Tp, kp = np.pad(T, 1, mode="edge"), np.pad(kc, 1, mode="edge")
        flux = (hm(kc, kp[2:, 1:-1]) * (Tp[2:, 1:-1] - T) + hm(kc, kp[:-2, 1:-1]) * (Tp[:-2, 1:-1] - T)
              + hm(kc, kp[1:-1, 2:]) * (Tp[1:-1, 2:] - T) + hm(kc, kp[1:-1, :-2]) * (Tp[1:-1, :-2] - T)) / dx**2
        H = H + dt * (flux + q) / RHO
        T = TfH(H)
        if fixed_mask is not None:
            T[fixed_mask] = fixed_T
            H[fixed_mask] = HfT(np.asarray(fixed_T, float))
        T_max = max(T_max, float(T.max()))
        if (i + 1) % save_every == 0:
            out_t.append((i + 1) * dt); out_T.append(T.astype(np.float32)); out_laser.append(path.pos((i + 1) * dt))
    return dict(t=np.array(out_t), T=np.stack(out_T), laser=np.array(out_laser),
                dt=dt, dx=dx, d=d, P=P, v=v, waypoints=np.asarray(waypoints),
                laser_off=path.duration, n_steps=n_steps, T_max=T_max)
