"""2D thin-plate conduction model with melting (enthalpy method), explicit finite differences.
Insulated edges. Laser: Gaussian spot moving along a Path, then a cooling phase with the laser off."""
import numpy as np
from .ti64 import RHO, T0, T_S, T_L, k, cp, H_from_T, T_from_H, R_SPOT, ETA
from .paths import Path
from .events import PhaseEvents, bilinear_sampler

def simulate(P, v, waypoints, nx=200, ny=100, dx=10e-6, d=100e-6,
             save_every=10, t_cool=0.2e-3, t_cool_max=None, props=None, cfl=0.2,
             T_init=None, fixed_mask=None, fixed_T=None, events=None):
    """props: optional dict overriding k, cp, H_from_T, T_from_H (constant-property runs, verification).
    cfl: time step as a fraction of dx^2/alpha_max (explicit stability limit is 0.25).
    T_init: initial temperature (scalar or nx x ny array), default T0.
    fixed_mask, fixed_T: cells held at a fixed temperature (Dirichlet), used by the Stefan check.
    t_cool, t_cool_max: cooling after the laser switches off.
      t_cool_max None (default): cool for exactly t_cool, as before.
      t_cool_max set: cool for at least t_cool, then stop at the first saved frame where every cell is below
      the solidus. Stop at t_cool_max (rounded up to a saved frame) if that never happens, and report the run
      as incomplete. The last state is then always a saved frame.
    events: None (default), "grid" (every cell) or an (m, 2) array of points in metres (bilinear from the grid).
      Melting and solidification events are recorded at every integration step (solver/events.py).

    Returned beyond the fields: t_end (time of the last state), cooling_complete (every cell below the solidus
    at t_end), t_cool_actual (t_end - laser_off), terminal_frame (index of the last saved frame),
    terminal_saved (whether the last state is that frame), events (dict or None)."""
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
    n_min = int(np.ceil((path.duration + t_cool) / dt))
    if t_cool_max is None:
        n_steps = n_min
    else:
        n_cap = int(np.ceil((path.duration + max(t_cool, t_cool_max)) / dt))
        n_steps = -(-n_cap // save_every) * save_every     # round up so the last state is a saved frame
    q0 = 2 * ETA * P / (np.pi * R_SPOT**2 * d)             # peak power density, W/m^3

    def hm(a, b):
        return 2 * a * b / (a + b)                  # conductivity between two cells

    T = np.full((nx, ny), T0, dtype=float) if T_init is None else np.broadcast_to(np.asarray(T_init, float), (nx, ny)).copy()
    if fixed_mask is not None:
        T[fixed_mask] = fixed_T
    H = HfT(T)
    out_t, out_T, out_laser = [0.0], [T.astype(np.float32)], [path.pos(0.0)]
    T_max = float(T.max())                                  # tracked every step, not only on saved frames
    rec = sample = None
    if events is not None:
        sample = (lambda F: F) if isinstance(events, str) and events == "grid" else bilinear_sampler(events, nx, ny, dx)
        rec = PhaseEvents(sample(T), 0.0)
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
        if rec is not None:
            rec.update(sample(T), (i + 1) * dt)
        if (i + 1) % save_every == 0:
            out_t.append((i + 1) * dt)
            out_T.append(T.astype(np.float32))
            out_laser.append(path.pos((i + 1) * dt))
            if t_cool_max is not None and i + 1 >= n_min and T.max() < T_S:   # fully solid: stop here
                n_steps = i + 1
                break
    t_end = n_steps * dt
    return dict(t=np.array(out_t), T=np.stack(out_T), laser=np.array(out_laser),
                dt=dt, dx=dx, d=d, P=P, v=v, waypoints=np.asarray(waypoints),
                laser_off=path.duration, n_steps=n_steps, T_max=T_max,
                t_end=t_end, cooling_complete=bool(T.max() < T_S), t_cool_actual=t_end - path.duration,
                terminal_frame=len(out_t) - 1, terminal_saved=n_steps % save_every == 0,
                events=None if rec is None else rec.result())
