import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---- auto_LiRPA imports ----
try:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
except Exception:
    # Fallback for older auto_LiRPA versions where PerturbationLpNorm
    # may be exposed directly from auto_LiRPA
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

import math

# ===========================================
# 1. Plant parameters & Lyapunov-based controller
# ===========================================

L, C, R0 = 10e-6, 47e-6, 5.0    # H, F, Ohm
Vin0, Vref = 12.0, 5.0          # V
fs = 200e3                      # Hz
Ts = 1.0 / fs                   # s

eps = 0.02                      # duty guard
alpha = 0.03                    # CCM inner-loop gain (original value)

dn = Vref / Vin0                # nominal duty for CCM
i_star = Vref / R0              # 1 A at 5V / 5Ω

# ===========================================
# 2. ADT-based λ_c, λ_d and minimal dwell times
# ===========================================

# DCM design shrink factor: 0 < mu_dcm < 2
mu_dcm = 0.2
lambda_d = -(1.0 / Ts) * math.log(abs(1.0 - mu_dcm))  # DCM decay rate

# Jump factor & margin
mu_max = 1.03
sigma = 0.95  # extra contraction margin

# CCM decay rate from Vdot <= -lambda_c V (matches report formula)
lambda_c = 2.0 * min(alpha * Vin0**2, 1.0 / R0) / max(L, C)

# ADT-based minimum dwell times (with margin sigma)
num = math.log(mu_max / sigma)
tau_c2d_min = num / lambda_c              # seconds in CCM
n_d2c_min = num / (2.0 * lambda_d * Ts)   # periods in DCM

N_CCM_DWELL = max(1, int(math.ceil(tau_c2d_min / Ts)))
N_DCM_DWELL = max(1, int(math.ceil(n_d2c_min)))

print("[ADT] lambda_c     = %.3e 1/s" % lambda_c)
print("[ADT] lambda_d     = %.3e 1/s" % lambda_d)
print("[ADT] mu_dcm       = %.3f" % mu_dcm)
print("[ADT] tau_c2d_min  = %.3e s  (~ %.2f Ts)" %
      (tau_c2d_min, tau_c2d_min / Ts))
print("[ADT] n_d2c_min    = %.3f periods" % n_d2c_min)
print("[ADT] Using N_CCM_DWELL = %d periods" % N_CCM_DWELL)
print("[ADT] Using N_DCM_DWELL = %d periods" % N_DCM_DWELL)

Nsub_per_Ts = 10  # Euler sub-steps per Ts for continuous simulations

# ===========================================
# 3. Lyapunov-based duty for CCM
# ===========================================

def lyapunov_duty(vC: float) -> float:
    """
    CCM inner-loop law (voltage-based in this implementation):
        d = dn - alpha * (vC - Vref),
    with saturation to [eps, 1-eps].
    """
    y = vC - Vref
    d_unsat = dn - alpha * y
    return float(np.clip(d_unsat, eps, 1.0 - eps))

# ===========================================
# 4. Continuous-time CCM/DCM averaged ODEs (for plotting only)
# ===========================================

def buck_ode_ccm(state: np.ndarray) -> np.ndarray:
    """
    CCM averaged model with Lyapunov-based duty.
    state = [iL, vC].
    """
    iL, vC = state
    d = lyapunov_duty(vC)
    diL = (Vin0 * d - vC) / L
    dvC = (iL - vC / R0) / C
    return np.array([diL, dvC], dtype=float)


def buck_ode_dcm_avg(state: np.ndarray) -> np.ndarray:
    """
    Averaged DCM behaviour in continuous time, consistent with
        e_{k+1} = (1 - mu_dcm) e_k
    and the choice
        e_dot = -lambda_d * e,
    with e = vC - Vref.

    Here we keep iL as a state but assume diL/dt ≈ 0 in DCM average.
    This avoids artificial resets in the hybrid model.
    """
    iL, vC = state
    diL = 0.0
    dvC = -lambda_d * (vC - Vref)
    return np.array([diL, dvC], dtype=float)

# ===========================================
# 5. Euler integrator
# ===========================================

def euler_integrate(state: np.ndarray, ode_fun, total_time: float) -> np.ndarray:
    """
    Explicit Euler integration of ODE 'ode_fun' over [0, total_time].
    """
    steps_per_Ts = Nsub_per_Ts
    dt = Ts / steps_per_Ts
    N_steps = int(math.ceil(total_time / dt))

    s = np.array(state, dtype=float)
    for _ in range(N_steps):
        deriv = ode_fun(s)
        s = s + dt * deriv
    return s

# ===========================================
# 6. Discrete one-period maps (matching the proofs)
# ===========================================

def ccm_one_period_step(x: np.ndarray) -> np.ndarray:
    """
    One full CCM switching period (Ts), using the averaged CCM ODE.
    """
    return euler_integrate(x, buck_ode_ccm, Ts)


def dcm_one_period_step(x: np.ndarray) -> np.ndarray:
    """
    One full DCM switching period (Ts), using the averaged DCM ODE.

    This is obtained by integrating
        e_dot = -lambda_d * e
    for one Ts, so that e(Ts) ≈ (1 - mu_dcm) e(0),
    but WITHOUT resetting iL.
    """
    return euler_integrate(x, buck_ode_dcm_avg, Ts)


def hybrid_ccm_to_dcm_step(x: np.ndarray) -> np.ndarray:
    """
    One ADT-compliant CCM→DCM dwell cycle:

      1. Stay in CCM for N_CCM_DWELL periods (Ts each) using the CCM averaged model.
      2. Switch to DCM *without state reset* (continuous state).
      3. Stay in DCM for N_DCM_DWELL periods using the DCM averaged map.
    """
    # 1) CCM dwell
    x_ccm = np.array(x, dtype=float)
    for _ in range(N_CCM_DWELL):
        x_ccm = ccm_one_period_step(x_ccm)

    # 2) Continuous switch: no projection/reset.
    x_dcm = np.array(x_ccm, dtype=float)

    # 3) DCM dwell
    for _ in range(N_DCM_DWELL):
        x_dcm = dcm_one_period_step(x_dcm)

    return x_dcm


def hybrid_cycle_time() -> float:
    """
    Total time of one CCM→DCM dwell cycle.
    """
    return (N_CCM_DWELL + N_DCM_DWELL) * Ts

# ===========================================
# 7. Simulation helpers for plotting
# ===========================================

def simulate_mode_time_response(x0, ode_fun, t_total, steps_per_Ts=50):
    """
    From initial state x0, simulate ODE 'ode_fun' for t_total seconds.
    Returns time array t and state trajectory X (N+1, 2).
    """
    dt = Ts / steps_per_Ts
    N_steps = int(math.ceil(t_total / dt))
    t = np.linspace(0.0, N_steps * dt, N_steps + 1)

    X = np.zeros((N_steps + 1, 2))
    s = np.array(x0, dtype=float)
    X[0] = s
    for k in range(N_steps):
        ds = ode_fun(s)
        s = s + dt * ds
        X[k + 1] = s

    return t, X


def simulate_hybrid_cycles_continuous(x0, t_total, steps_per_Ts=50):
    """
    Continuous-time plotting version of CCM↔DCM hybrid switching:
    within t_total, alternate CCM dwell and DCM dwell according to ADT.
    **No state reset** at CCM↔DCM switching.
    """
    t_list = []
    X_list = []

    t_offset = 0.0
    x = np.array(x0, dtype=float)

    while t_offset < t_total:
        # CCM dwell
        t_ccm, X_ccm = simulate_mode_time_response(
            x, buck_ode_ccm, N_CCM_DWELL * Ts, steps_per_Ts=steps_per_Ts
        )
        t_ccm += t_offset
        t_offset = t_ccm[-1]
        x_ccm_end = X_ccm[-1]

        # DCM dwell starting from the same state (continuous switch)
        x_dcm_start = np.array(x_ccm_end, dtype=float)

        t_dcm, X_dcm = simulate_mode_time_response(
            x_dcm_start, buck_ode_dcm_avg, N_DCM_DWELL * Ts, steps_per_Ts=steps_per_Ts
        )
        t_dcm += t_offset
        t_offset = t_dcm[-1]
        x = X_dcm[-1]

        # Concatenate, avoiding duplicate endpoints
        if not t_list:
            t_list.append(t_ccm)
            X_list.append(X_ccm)
        else:
            t_list.append(t_ccm[1:])
            X_list.append(X_ccm[1:])
        t_list.append(t_dcm[1:])
        X_list.append(X_dcm[1:])

        if t_offset >= t_total:
            break

    t_full = np.concatenate(t_list, axis=0)
    X_full = np.concatenate(X_list, axis=0)
    return t_full, X_full

# ---------- DCM discrete per-period simulation for plotting ----------

kappa = C * mu_dcm / Ts  # from report: kappa := C*mu/Ts

def simulate_dcm_discrete(V0: float, T_total: float):
    """
    Use the discrete-time DCM design:
        e_{k+1} = (1 - mu_dcm) e_k,
        V_k = Vref + e_k,
        i_avg_k = V_k/R0 - kappa e_k.

    Returns:
        t      - time stamps k*Ts
        I_avg  - average inductor/output current per period
        V      - output voltage per period
    """
    N = int(math.ceil(T_total / Ts))
    t = np.arange(N + 1) * Ts
    V = np.zeros(N + 1)
    I_avg = np.zeros(N + 1)

    V[0] = V0
    e = V[0] - Vref

    for k in range(N + 1):
        # commanded average current for period k
        I_avg[k] = V[k] / R0 - kappa * (V[k] - Vref)
        if k < N:
            e = (1.0 - mu_dcm) * e
            V[k + 1] = Vref + e

    return t, I_avg, V

# ---------- NEW: average current from a continuous-time trajectory -----

def compute_period_average_current(t, iL, Ts):
    """
    Given a time grid t (uniform) and inductor current samples iL(t),
    compute the average current over each switching period Ts:
        Ī[k] = (1/Ts) ∫_{kTs}^{(k+1)Ts} iL(t) dt
    using simple mean over samples.
    Returns:
        t_avg  - one time stamp per period (center of each period)
        I_avg  - average current per period
    """
    t = np.asarray(t)
    iL = np.asarray(iL)

    if len(t) < 2:
        return np.array([]), np.array([])

    dt = t[1] - t[0]
    steps_per_period = int(round(Ts / dt))
    if steps_per_period <= 0:
        raise ValueError("steps_per_period must be positive")

    # number of full periods we can form
    N_steps = len(t) - 1
    N_periods = N_steps // steps_per_period

    I_avg = np.zeros(N_periods)
    t_avg = np.zeros(N_periods)

    for k in range(N_periods):
        start = k * steps_per_period
        end = start + steps_per_period
        segment_i = iL[start:end]
        segment_t = t[start:end]
        # average current in this period (dt uniform)
        I_avg[k] = np.mean(segment_i)
        # time stamp: center of the period
        t_avg[k] = 0.5 * (segment_t[0] + segment_t[-1])

    return t_avg, I_avg

# ===========================================
# 8. NN surrogates
# ===========================================

class BuckStepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


def generate_data_mode(step_fun, domain_box, N_samples=8000):
    """
    Sample x in 'domain_box' and generate (x, step_fun(x)) pairs.
    domain_box = ((iL_min, iL_max), (vC_min, vC_max))
    """
    (iL_min, iL_max), (vC_min, vC_max) = domain_box
    X_list, Y_list = [], []
    for _ in range(N_samples):
        if iL_max == iL_min:
            iL = iL_min
        else:
            iL = np.random.uniform(iL_min, iL_max)
        if vC_max == vC_min:
            vC = vC_min
        else:
            vC = np.random.uniform(vC_min, vC_max)
        x = np.array([iL, vC], dtype=float)
        x_next = step_fun(x)
        X_list.append(x)
        Y_list.append(x_next)
    X = np.stack(X_list)
    Y = np.stack(Y_list)
    return X, Y


def train_surrogate(X, Y, label="MODE", epochs=200, lr=1e-3):
    """
    Train a small ReLU NN surrogate for given one-step map.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BuckStepNet().to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"[Train-{label}] Epoch {epoch+1:3d}/{epochs}, loss = {loss.item():.3e}")

    return model

# ===========================================
# 9. auto_LiRPA helpers
# ===========================================

def make_lirpa_input_box(X_box, device):
    """
    X_box: ((iL_min, iL_max), (vC_min, vC_max))
    Return a BoundedTensor representing the whole box.
    """
    (iL_min, iL_max), (vC_min, vC_max) = X_box
    lower = torch.tensor([[iL_min, vC_min]], dtype=torch.float32, device=device)
    upper = torch.tensor([[iL_max, vC_max]], dtype=torch.float32, device=device)
    center = 0.5 * (lower + upper)
    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=lower, x_U=upper)
    x = BoundedTensor(center, ptb)
    return x


def multi_step_bounds_iter(model, X0_box, K, mode_label, dwell_time):
    """
    Use auto_LiRPA to iterate K times the surrogate map F:
        X^{(0)} = X0_box
        X^{(k+1)} = F(X^{(k)})
    and print the final iL, vC intervals after K steps.
    """
    device = next(model.parameters()).device
    model.eval()
    dummy_input = torch.zeros(1, 2, device=device)
    lirpa_model = BoundedModule(model, dummy_input, device=device)

    (iL_min, iL_max), (vC_min, vC_max) = X0_box

    for k in range(K):
        X_box_k = ((iL_min, iL_max), (vC_min, vC_max))
        x0 = make_lirpa_input_box(X_box_k, device)
        try:
            lb, ub = lirpa_model.compute_bounds(x=(x0,), method="CROWN-IBP")
        except Exception as e:
            print(f"[Verify-{mode_label}-iter] step {k+1}: CROWN-IBP failed ({e}), fallback IBP")
            lb, ub = lirpa_model.compute_bounds(x=(x0,), method="IBP")

        lb_np = lb.detach().cpu().numpy()[0]
        ub_np = ub.detach().cpu().numpy()[0]
        iL_min, vC_min = lb_np[0], lb_np[1]
        iL_max, vC_max = ub_np[0], ub_np[1]

    final_box = ((iL_min, iL_max), (vC_min, vC_max))
    T_total = K * dwell_time

    print(f"\n[Verify-{mode_label}-2ms] K = {K} steps, total time ≈ {T_total*1e3:.3f} ms")
    print(f"[Verify-{mode_label}-2ms] iL range ≈ [{iL_min:.6f}, {iL_max:.6f}] A")
    print(f"[Verify-{mode_label}-2ms] vC range ≈ [{vC_min:.6f}, {vC_max:.6f}] V")

    return final_box


def multi_step_bounds_hybrid(model_ccm, model_dcm, X0_box, K_cycles, mode_label, dwell_time):
    """
    ADT-compliant hybrid verification using CCM and DCM surrogates
    (no separate hybrid NN). Each cycle does:
       - N_CCM_DWELL CCM periods
       - N_DCM_DWELL DCM periods
    **Without manual state reset**; state continuity is enforced by the maps.
    """
    device = next(model_ccm.parameters()).device
    model_ccm.eval()
    model_dcm.eval()

    dummy_input = torch.zeros(1, 2, device=device)
    lirpa_ccm = BoundedModule(model_ccm, dummy_input, device=device)
    lirpa_dcm = BoundedModule(model_dcm, dummy_input, device=device)

    (iL_min, iL_max), (vC_min, vC_max) = X0_box

    for k in range(K_cycles):
        # CCM dwell: N_CCM_DWELL periods
        for j in range(N_CCM_DWELL):
            X_box_k = ((iL_min, iL_max), (vC_min, vC_max))
            x0 = make_lirpa_input_box(X_box_k, device)
            try:
                lb, ub = lirpa_ccm.compute_bounds(x=(x0,), method="CROWN-IBP")
            except Exception as e:
                print(f"[Verify-{mode_label}-CCM] cycle {k+1}, dwell {j+1}: CROWN-IBP failed ({e}), fallback IBP")
                lb, ub = lirpa_ccm.compute_bounds(x=(x0,), method="IBP")

            lb_np = lb.detach().cpu().numpy()[0]
            ub_np = ub.detach().cpu().numpy()[0]
            iL_min, vC_min = lb_np[0], lb_np[1]
            iL_max, vC_max = ub_np[0], ub_np[1]

        # DCM dwell: N_DCM_DWELL periods (no manual projection/reset)
        for j in range(N_DCM_DWELL):
            X_box_k = ((iL_min, iL_max), (vC_min, vC_max))
            x0 = make_lirpa_input_box(X_box_k, device)
            try:
                lb, ub = lirpa_dcm.compute_bounds(x=(x0,), method="CROWN-IBP")
            except Exception as e:
                print(f"[Verify-{mode_label}-DCM] cycle {k+1}, dwell {j+1}: CROWN-IBP failed ({e}), fallback IBP")
                lb, ub = lirpa_dcm.compute_bounds(x=(x0,), method="IBP")

            lb_np = lb.detach().cpu().numpy()[0]
            ub_np = ub.detach().cpu().numpy()[0]
            iL_min, vC_min = lb_np[0], lb_np[1]
            iL_max, vC_max = ub_np[0], ub_np[1]

    final_box = ((iL_min, iL_max), (vC_min, vC_max))
    T_total = K_cycles * dwell_time

    print(f"\n[Verify-{mode_label}-2ms] K = {K_cycles} cycles, total time ≈ {T_total*1e3:.3f} ms")
    print(f"[Verify-{mode_label}-2ms] iL range ≈ [{iL_min:.6f}, {iL_max:.6f}] A")
    print(f"[Verify-{mode_label}-2ms] vC range ≈ [{vC_min:.6f}, {vC_max:.6f}] V")

    return final_box

# ===========================================
# 10. Main
# ===========================================

if __name__ == "__main__":
    # Use the same equilibrium for CCM and DCM (no reset)
    x_eq_ccm = np.array([i_star, Vref])
    x_eq_dcm = np.array([i_star, Vref])

    print("\nSanity CCM one-period step:  x_eq_ccm ->", ccm_one_period_step(x_eq_ccm))
    print("Sanity DCM one-period step:  x_eq_dcm ->", dcm_one_period_step(x_eq_dcm))
    print("Sanity HYBRID one-cycle step from CCM eq:",
          hybrid_ccm_to_dcm_step(x_eq_ccm))

    # ---------- Case 1: pure CCM ----------
    domain_CCM_wide = ((0.3, 1.7), (4.0, 6.0))
    domain_CCM_local = ((0.8, 1.2), (4.8, 5.2))

    print("\n[Data-CCM] wide region samples...")
    X_ccm_wide, Y_ccm_wide = generate_data_mode(
        ccm_one_period_step, domain_CCM_wide, N_samples=4000
    )
    print("[Data-CCM] local region samples...")
    X_ccm_local, Y_ccm_local = generate_data_mode(
        ccm_one_period_step, domain_CCM_local, N_samples=4000
    )

    print("[Data-CCM] equilibrium reinforcement samples...")
    N_eq = 1000
    X_ccm_eq = np.tile(x_eq_ccm, (N_eq, 1))
    Y_ccm_eq = np.stack([ccm_one_period_step(x_eq_ccm) for _ in range(N_eq)], axis=0)

    X_ccm = np.concatenate([X_ccm_wide, X_ccm_local, X_ccm_eq], axis=0)
    Y_ccm = np.concatenate([Y_ccm_wide, Y_ccm_local, Y_ccm_eq], axis=0)

    print("[Train-CCM] Training CCM surrogate NN (one-period map)...")
    model_ccm = train_surrogate(X_ccm, Y_ccm, label="CCM", epochs=300, lr=1e-3)

    X0_CCM_box = ((0.8, 1.2), (4.8, 5.2))
    T_target = 2e-3
    step_CCM = Ts
    K_ccm = int(math.ceil(T_target / step_CCM))

    print("\n[Verify-CCM-2ms] target time ≈ %.3f ms, step = Ts = %.3f ms, K = %d"
          % (T_target * 1e3, step_CCM * 1e3, K_ccm))

    final_box_ccm = multi_step_bounds_iter(
        model_ccm, X0_CCM_box, K_ccm, mode_label="CCM", dwell_time=step_CCM
    )

    # ---------- Case 2: pure DCM ----------
    domain_DCM_wide = domain_CCM_wide
    domain_DCM_local = domain_CCM_local

    print("\n[Data-DCM] wide region samples...")
    X_dcm_wide, Y_dcm_wide = generate_data_mode(
        dcm_one_period_step, domain_DCM_wide, N_samples=4000
    )
    print("[Data-DCM] local region samples...")
    X_dcm_local, Y_dcm_local = generate_data_mode(
        dcm_one_period_step, domain_DCM_local, N_samples=4000
    )

    print("[Data-DCM] equilibrium reinforcement samples...")
    X_dcm_eq = np.tile(x_eq_dcm, (N_eq, 1))
    Y_dcm_eq = np.stack([dcm_one_period_step(x_eq_dcm) for _ in range(N_eq)], axis=0)

    X_dcm = np.concatenate([X_dcm_wide, X_dcm_local, X_dcm_eq], axis=0)
    Y_dcm = np.concatenate([Y_dcm_wide, Y_dcm_local, Y_dcm_eq], axis=0)

    print("[Train-DCM] Training DCM surrogate NN...")
    model_dcm = train_surrogate(X_dcm, Y_dcm, label="DCM", epochs=200, lr=1e-3)

    X0_DCM_box = ((0.8, 1.2), (4.8, 5.2))
    step_DCM = Ts
    K_dcm = int(math.ceil(T_target / step_DCM))

    print("\n[Verify-DCM-2ms] target time ≈ %.3f ms, step = Ts = %.3f ms, K = %d"
          % (T_target * 1e3, step_DCM * 1e3, K_dcm))

    final_box_dcm = multi_step_bounds_iter(
        model_dcm, X0_DCM_box, K_dcm, mode_label="DCM", dwell_time=step_DCM
    )

    # ---------- Case 3: hybrid CCM→DCM with ADT ----------
    dwell_HYB = hybrid_cycle_time()
    K_hyb = int(math.ceil(T_target / dwell_HYB))

    print("\n[Verify-HYBRID-2ms] target time ≈ %.3f ms, dwell per cycle ≈ %.3f ms, K = %d"
          % (T_target * 1e3, dwell_HYB * 1e3, K_hyb))

    final_box_hyb = multi_step_bounds_hybrid(
        model_ccm, model_dcm, X0_CCM_box,
        K_cycles=K_hyb,
        mode_label="HYBRID",
        dwell_time=dwell_HYB
    )

    # ---------- Continuous-time / discrete-time plots ----------
    x0_ccm = np.array([0.8, 5.2])   # off-nominal initial state for CCM
    x0_hyb = np.array([0.8, 5.2])   # same for hybrid
    V0_dcm = 5.2                    # off-nominal initial output voltage for DCM

    # extend plotting horizon a bit so convergence is visually clear
    T_plot = 5e-3

    # Case 1: pure CCM (continuous-time averaged model)
    t_ccm, X_ccm_traj = simulate_mode_time_response(
        x0_ccm, buck_ode_ccm, T_plot, steps_per_Ts=50
    )

    # Case 2: pure DCM (discrete per-period average current and voltage)
    t_dcm, I_dcm_traj, V_dcm_traj = simulate_dcm_discrete(V0_dcm, T_plot)

    # Case 3: hybrid CCM↔DCM (continuous-time averaged hybrid model)
    t_hyb, X_hyb_traj = simulate_hybrid_cycles_continuous(
        x0_hyb, T_plot, steps_per_Ts=50
    )

    # NEW: compute average current per switching period for the hybrid case
    t_hyb_avg, I_hyb_avg = compute_period_average_current(
        t_hyb, X_hyb_traj[:, 0], Ts
    )

    t_ccm_ms = t_ccm * 1e3
    t_dcm_ms = t_dcm * 1e3
    t_hyb_ms = t_hyb * 1e3
    t_hyb_avg_ms = t_hyb_avg * 1e3

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    # Row 1: pure CCM
    axes[0, 0].plot(t_ccm_ms, X_ccm_traj[:, 0])
    axes[0, 0].axhline(1.0, linestyle="--")   # reference 1 A
    axes[0, 0].set_title("Case 1: pure CCM - iL(t)")
    axes[0, 0].set_xlabel("t [ms]")
    axes[0, 0].set_ylabel("iL [A]")

    axes[0, 1].plot(t_ccm_ms, X_ccm_traj[:, 1])
    axes[0, 1].axhline(5.0, linestyle="--")   # reference 5 V
    axes[0, 1].set_title("Case 1: pure CCM - vC(t)")
    axes[0, 1].set_xlabel("t [ms]")
    axes[0, 1].set_ylabel("vC [V]")

    # Row 2: pure DCM (discrete average behaviour)
    axes[1, 0].plot(t_dcm_ms, I_dcm_traj)
    axes[1, 0].axhline(1.0, linestyle="--")   # reference 1 A
    axes[1, 0].set_title("Case 2: pure DCM - average current")
    axes[1, 0].set_xlabel("t [ms]")
    axes[1, 0].set_ylabel("ī [A]")

    axes[1, 1].plot(t_dcm_ms, V_dcm_traj)
    axes[1, 1].axhline(5.0, linestyle="--")   # reference 5 V
    axes[1, 1].set_title("Case 2: pure DCM - V_k")
    axes[1, 1].set_xlabel("t [ms]")
    axes[1, 1].set_ylabel("vC [V]")

    # Row 3: hybrid CCM↔DCM
    # Left: **average current per period** (NEW)
    axes[2, 0].plot(t_hyb_avg_ms, I_hyb_avg)
    axes[2, 0].axhline(1.0, linestyle="--")   # reference 1 A
    axes[2, 0].set_title("Case 3: hybrid CCM↔DCM - average current")
    axes[2, 0].set_xlabel("t [ms]")
    axes[2, 0].set_ylabel("ī [A]")

    # Right: instantaneous vC(t)
    axes[2, 1].plot(t_hyb_ms, X_hyb_traj[:, 1])
    axes[2, 1].axhline(5.0, linestyle="--")
    axes[2, 1].set_title("Case 3: hybrid CCM↔DCM - vC(t)")
    axes[2, 1].set_xlabel("t [ms]")
    axes[2, 1].set_ylabel("vC [V]")

    plt.tight_layout()
    plt.show()

    print("\nDone.")
