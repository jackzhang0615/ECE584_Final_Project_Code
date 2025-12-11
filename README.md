# Lyapunov-Based Stability Verification of a Buck Converter (ECE 584 Final Project)

This repository contains the code for my ECE 584 final project:  
**Lyapunov-based stability analysis and neural verification of a buck converter that switches between CCM and DCM under an average dwell-time (ADT) constraint.**

All experiments are implemented in a single Python script:

- `buck_verification.py`

The script defines averaged models for CCM, DCM, and hybrid CCM↔DCM operation, trains small neural-network surrogates for the one-period maps, and uses **auto_LiRPA** to over-approximate reachable sets around the desired equilibrium.

---

## 1. Requirements

- Python ≥ 3.8  
- [PyTorch](https://pytorch.org) (CPU or GPU build is fine)
- `auto_LiRPA` library for LiRPA-based bounds
- Python packages:
  - `numpy`
  - `matplotlib`
  - `torch`, `torchvision` (installed with PyTorch)
  - `auto_LiRPA` (installed separately)

---

## 2. Setup

It is recommended to use a virtual environment.

```bash
git clone <this-repo-url>
cd ECE584_Final_Project_Code

# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
```

### 2.1 Install PyTorch

Install a version of PyTorch compatible with your system (CPU or GPU).  
Example (CPU-only):

```bash
pip install torch torchvision torchaudio
```

You can also follow the official “Get Started” instructions on the PyTorch website for a CUDA-enabled install.

### 2.2 Install auto_LiRPA

Install `auto_LiRPA` in the same environment:

**Option A – from GitHub (recommended)**

```bash
git clone https://github.com/Verified-Intelligence/auto_LiRPA
cd auto_LiRPA
pip install .
cd ..
```

**Option B – from PyPI (if available for your platform)**

```bash
pip install auto-LiRPA
```

The script has a small `try/except` around the imports so it works with both “new” and “old” versions of auto_LiRPA (where `PerturbationLpNorm` may live in different modules).

### 2.3 Install remaining dependencies

If they are not already installed:

```bash
pip install numpy matplotlib
```

---

## 3. How to Run

From the repository root:

```bash
python buck_verification.py
```

The script will:

1. Print **ADT-related parameters**:

   - CCM and DCM decay rates `lambda_c` and `lambda_d`
   - DCM shrink factor `mu_dcm`
   - Minimum CCM and DCM dwell times (in seconds and number of periods)
   - Discretized dwell counts `N_CCM_DWELL` and `N_DCM_DWELL`

2. Train two small ReLU networks:

   - `model_ccm`: one-period CCM map  
   - `model_dcm`: one-period DCM map  

   Training progress is printed as:

   - `[Train-CCM] Epoch ... loss = ...`
   - `[Train-DCM] Epoch ... loss = ...`

3. Run **LiRPA-based verification** for three benchmark cases (about 2 ms of operation each):

   - **Case 1 – pure CCM**
   - **Case 2 – pure DCM**
   - **Case 3 – hybrid CCM↔DCM with ADT**

   For each case, the script prints final over-approximate ranges of the inductor current and capacitor voltage, e.g.:

   ```text
   [Verify-CCM-2ms] iL range ≈ [...]
   [Verify-CCM-2ms] vC range ≈ [...]
   ```

4. Generate a figure with six plots:

   - Row 1: pure CCM trajectories \(i_L(t)\) and \(v_C(t)\)
   - Row 2: pure DCM trajectories (average current per period and sampled voltage)
   - Row 3: hybrid CCM↔DCM trajectories (average current and continuous voltage)

A Matplotlib window will pop up with the plots. In a headless environment you can modify the script to save figures instead of showing them.

---

## 4. What the Script Verifies

All benchmarks use the same buck converter and incremental-energy Lyapunov function:

- Inductance \(L = 10~\mu\text{H}\)
- Capacitance \(C = 47~\mu\text{F}\)
- Load resistance \(R_0 = 5~\Omega\)
- Input voltage \(V_{\text{in}} = 12~\text{V}\)
- Reference voltage \(V_{\text{ref}} = 5~\text{V}\)
- PWM frequency \(f_s = 200~\text{kHz}\) (\(T_s = 1/f_s\))

The target operating point is \((i_L^\*, v_C^\*) = (1~\text{A}, 5~\text{V})\).

Key design parameters:

- CCM inner-loop gain: `alpha = 0.03`
- DCM error contraction per cycle: `mu_dcm = 0.2`
- Lyapunov jump bound at mode switches: `mu_max = 1.03`
- Additional contraction margin: `sigma = 0.95`

### Case 1 – Pure CCM

- Model: averaged CCM ODE with duty law  
  \( d = d_n - \alpha (v_C - V_{\text{ref}}) \), with saturation to \([\varepsilon, 1-\varepsilon]\).
- Initial box used for verification:  
  \( i_L \in [0.8, 1.2]~\text{A},\ v_C \in [4.8, 5.2]~\text{V} \).
- The script constructs a one-period map \(F_{\text{CCM}}\), trains a neural surrogate, and uses auto_LiRPA to show that after ~2 ms all states in this box have contracted into a tight neighborhood of \((1~\text{A}, 5~\text{V})\).

### Case 2 – Pure DCM

- Model: sampled DCM dynamics with per-cycle error recursion  
  \( e_{k+1} = (1 - \mu_{\text{dcm}}) e_k\), \(e_k = V_k - V_{\text{ref}}\).
- Initial voltage box: \( V_k \in [4.8, 5.2]~\text{V}\) with \(i_L = 0\) at sampling.
- The script trains a surrogate for the DCM one-period map \(F_{\text{DCM}}\), and auto_LiRPA shows that both the sampled voltage and average inductor current converge rapidly to the desired values.

### Case 3 – Hybrid CCM↔DCM with ADT

- Switching law: alternate between
  - \(N_{\text{CCM\_DWELL}}\) consecutive CCM periods, and
  - \(N_{\text{DCM\_DWELL}}\) consecutive DCM periods,
- where `N_CCM_DWELL` and `N_DCM_DWELL` are chosen from the ADT bounds derived from \(\lambda_c\), \(\lambda_d\), and \(\mu_{\max}\).
- Initial box is again  
  \( i_L \in [0.8, 1.2]~\text{A},\ v_C \in [4.8, 5.2]~\text{V} \).
- Using the CCM and DCM surrogates in sequence, the script shows that under these dwell-time constraints the hybrid system still contracts toward the equilibrium and the reachable set after ~2 ms remains small.

---

## 5. Modifying the Experiment

You can change behavior by editing the constants near the top of `buck_verification.py`:

- `alpha` – CCM inner-loop gain (affects CCM damping).
- `mu_dcm` – per-cycle shrink factor in DCM.
- `mu_max`, `sigma` – margin assumptions in the ADT analysis.
- Initial boxes:
  - `domain_CCM_wide`, `domain_CCM_local`, `domain_DCM`
  - `X0_CCM_box`, `X0_DCM_box`
- Target verification horizon:
  - `T_target` (for 2 ms verification)
  - `T_plot` (for longer visual simulations)

After editing, simply re-run:

```bash
python buck_verification.py
```

---

## 6. Troubleshooting

**auto_LiRPA import error**

- Make sure `auto_LiRPA` is installed in the same environment.  
- Re-run `pip install .` from inside the cloned `auto_LiRPA` folder, or reinstall from PyPI.

**CUDA / GPU issues**

- The script does not require a GPU; CPU-only PyTorch is fine.  
- If CUDA versions conflict, install a CPU-only build of PyTorch.

**Plots do not show**

- On some remote/terminal-only setups, you may need to change the Matplotlib backend at the top of the script:

  ```python
  import matplotlib
  matplotlib.use("Agg")  # then call plt.savefig(...) instead of plt.show()
  ```

---

If you use this code or adapt it, feel free to add a citation or short note in your own work.
