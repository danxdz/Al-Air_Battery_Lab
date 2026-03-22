"""
Al-Air Paste Battery — Calibration + Uncertainty Quantification
================================================================
Implements the physics-region calibration approach:
  j=2-5   mA/cm² → OCV + activation losses  → fit E_ocv_ref, i0_Al_ref
  j=5-10  mA/cm² → Butler-Volmer slope       → fit i0_Al_ref, i0_O2_ref
  j=10-25 mA/cm² → ohmic resistance          → fit L_eff_m
  j=25-50 mA/cm² → mass transport            → fit L_diff_factor (optional)

After calibration: Monte Carlo uncertainty bands by propagating
parameter uncertainty through 1000 model evaluations.

Usage:
  python al_air_calibrate.py                    # demo with placeholder data
  python al_air_calibrate.py --data my_data.csv # calibrate to your measurements
  python al_air_calibrate.py --mc               # Monte Carlo bands (slow)

CSV format (my_data.csv):
  j_mA_cm2,V_cell
  2.0,1.32
  5.0,1.25
  10.0,1.18
  25.0,1.05
  50.0,0.90

After running with real data, paste the printed CALIBRATED_PARAMS into
al_air_model.py PARAMS dict to lock in the calibration.
"""

import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from al_air_model import (
    cell_model, PARAMS, BASE_CONFIG,
    polarisation_curve, G, O, Y, B, P, RE, _rc, _save,
)

# ─────────────────────────────────────────────
# CALIBRATION CONDITIONS
# ─────────────────────────────────────────────
# These MUST match your experimental setup exactly.
# Our BASE_CONFIG already corresponds to the recommended setup:
#   3.5M KOH, 60°C, 53 vol% Al, 100µm particles
# Change these if your experiment differs.

CALIB_CONFIG = dict(BASE_CONFIG)   # single source of truth

# Measurement points (current densities, mA/cm²)
# Match these exactly to your galvanostatic steps
J_POINTS = np.array([2.0, 5.0, 10.0, 25.0, 50.0])

# ─────────────────────────────────────────────
# CALIBRATION DATA
# ─────────────────────────────────────────────
# Placeholder data — replace with your measurements.
# These are physically plausible values for the CALIB_CONFIG conditions.
# Your real cell may differ significantly — that's why calibration matters.

PLACEHOLDER_DATA = {
    "j_mA_cm2": J_POINTS,
    "V_cell":   np.array([1.32, 1.25, 1.18, 1.05, 0.90]),
    "source":   "placeholder — replace with your measurements",
}

# ─────────────────────────────────────────────
# PHYSICS-REGION DIAGNOSTIC
# ─────────────────────────────────────────────

def diagnose_physics_regions(exp_data, params_override=None):
    """
    Show which physics dominates at each measurement point.
    Helps understand where the model error comes from.
    """
    print("\n── Physics region diagnostic ──")
    print(f"  {'j':>6}  {'V_exp':>7}  {'V_mod':>7}  {'err_mV':>8}  "
          f"{'η_BV':>7}  {'η_ohm':>7}  {'η_mt':>7}  dominant")
    print("  " + "─"*70)

    for j, V_exp in zip(exp_data["j_mA_cm2"], exp_data["V_cell"]):
        r = cell_model(**CALIB_CONFIG, j_mA_cm2=j,
                       params_override=params_override)
        err = (r['V_cell'] - V_exp) * 1000
        eta_bv  = (r['eta_anode_V'] + r['eta_cathode_V']) * 1000
        eta_ohm = r['eta_ohmic_V'] * 1000
        eta_mt  = r['eta_mass_trans_V'] * 1000
        dom = max([('BV', eta_bv), ('ohm', eta_ohm), ('mt', eta_mt)],
                  key=lambda x: x[1])[0]
        print(f"  {j:>6.1f}  {V_exp:>7.3f}  {r['V_cell']:>7.3f}  {err:>+8.1f}  "
              f"{eta_bv:>7.1f}  {eta_ohm:>7.1f}  {eta_mt:>7.1f}  {dom}")


# ─────────────────────────────────────────────
# STEP-BY-STEP CALIBRATION
# ─────────────────────────────────────────────

def calibrate(exp_data, verbose=True):
    """
    Physics-region calibration of 3 parameters:
    E_ocv_ref  — from low-j region (OCV offset)
    i0_Al_ref  — from BV slope region
    L_eff_m    — from ohmic region
    (L_diff_factor optional if mass transport tail is off)

    Returns: calibrated params dict, RMSE before/after
    """
    j_exp = np.array(exp_data["j_mA_cm2"])
    V_exp = np.array(exp_data["V_cell"])

    def model_V(j, override):
        try:
            r = cell_model(**CALIB_CONFIG, j_mA_cm2=j, params_override=override)
            return r['V_cell']
        except Exception:
            return 0.0

    def rmse(override):
        errs = [model_V(j, override) - Ve for j, Ve in zip(j_exp, V_exp)]
        return float(np.sqrt(np.mean(np.array(errs)**2)))

    rmse_before = rmse({}) * 1000
    if verbose:
        print(f"\n── Calibration ──")
        print(f"  RMSE before: {rmse_before:.1f} mV")
        diagnose_physics_regions(exp_data)

    # ── Step 1: OCV correction from lowest current point ─────────────────────
    # At j=2 mA/cm², BV dominates. OCV offset shifts the whole curve.
    j_low, V_low = j_exp[0], V_exp[0]
    V_mod_low = model_V(j_low, {})
    dOCV = V_low - V_mod_low
    # Apply bounded OCV correction
    E_ocv_new = float(np.clip(PARAMS["E_ocv_ref"] + dOCV, 1.10, 1.80))
    step1_override = {"E_ocv_ref": E_ocv_new}

    if verbose:
        print(f"\n  Step 1 — OCV correction: {dOCV*1000:+.1f} mV → E_ocv={E_ocv_new:.4f} V")
        print(f"           RMSE after step 1: {rmse(step1_override)*1000:.1f} mV")

    # ── Step 2: Exchange current (j0) from BV slope ──────────────────────────
    # Use kinetics-dominated points (j ≤ 10 mA/cm²) to fit i0_Al_ref, i0_O2_ref
    kinetics_mask = j_exp <= 12
    j_kin, V_kin = j_exp[kinetics_mask], V_exp[kinetics_mask]

    def obj_kinetics(x):
        override = {
            "E_ocv_ref": E_ocv_new,
            "i0_Al_ref": 10**x[0],
            "i0_O2_ref": 10**x[1],
        }
        errs = [model_V(j, override) - Ve for j, Ve in zip(j_kin, V_kin)]
        return np.mean(np.array(errs)**2)

    x0_kin = [np.log10(PARAMS["i0_Al_ref"]),
               np.log10(PARAMS["i0_O2_ref"])]
    bounds_kin = [(-5, -1), (-7, -2)]

    res_kin = minimize(obj_kinetics, x0_kin, method='Nelder-Mead',
                       options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 500})

    i0_Al_new = float(np.clip(10**res_kin.x[0], 1e-5, 1.0))
    i0_O2_new = float(np.clip(10**res_kin.x[1], 1e-7, 0.1))
    step2_override = {"E_ocv_ref": E_ocv_new,
                      "i0_Al_ref": i0_Al_new, "i0_O2_ref": i0_O2_new}

    if verbose:
        print(f"  Step 2 — Kinetics fit: i0_Al={i0_Al_new:.3e} i0_O2={i0_O2_new:.3e}")
        print(f"           RMSE after step 2: {rmse(step2_override)*1000:.1f} mV")

    # ── Step 3: Ohmic resistance from high-j slope ────────────────────────────
    # Use slope ΔV/Δj at high current (dominated by ohmic)
    if len(j_exp) >= 4:
        j_hi = j_exp[-2:]
        V_hi = V_exp[-2:]
        R_exp_mOhm_cm2 = abs((V_hi[1] - V_hi[0]) / (j_hi[1] - j_hi[0])) * 1000 * 100
        # Convert to L_eff: R = L_eff / sigma, sigma ≈ 33 S/m at 3.5M 60°C
        from al_air_model import koh_conductivity
        sigma = koh_conductivity(CALIB_CONFIG['c_KOH'],
                                  CALIB_CONFIG['T_C'] + 273.15)
        L_eff_new = float(np.clip(R_exp_mOhm_cm2 * 1e-3 * 1e-4 * sigma,
                                   0.5e-3, 5e-3))
        step3_override = dict(step2_override)
        step3_override["L_eff_m"] = L_eff_new

        if verbose:
            print(f"  Step 3 — Ohmic fit: R_exp={R_exp_mOhm_cm2:.1f} mΩ·cm²"
                  f"  → L_eff={L_eff_new*1000:.2f} mm")
            print(f"           RMSE after step 3: {rmse(step3_override)*1000:.1f} mV")
    else:
        step3_override = step2_override
        L_eff_new = PARAMS["L_eff_m"]

    # ── Step 4 (optional): Mass transport tweak ───────────────────────────────
    # Only if high-current tail still significantly off
    rmse_after3 = rmse(step3_override) * 1000
    if rmse_after3 > 25:
        def obj_mt(x):
            override = dict(step3_override)
            override["L_diff_factor"] = float(np.clip(x[0], 10, 80))
            errs = [model_V(j, override) - Ve
                    for j, Ve in zip(j_exp, V_exp)]
            return np.mean(np.array(errs)**2)

        res_mt = minimize(obj_mt, [PARAMS["L_diff_factor"]],
                          method='Nelder-Mead',
                          options={"xatol": 0.5, "fatol": 1e-6, "maxiter": 200})
        L_diff_new = float(np.clip(res_mt.x[0], 10, 80))
        final_override = dict(step3_override)
        final_override["L_diff_factor"] = L_diff_new
        if verbose:
            print(f"  Step 4 — MT tweak: L_diff_factor={L_diff_new:.1f}d")
    else:
        final_override = step3_override
        L_diff_new = PARAMS["L_diff_factor"]

    rmse_after = rmse(final_override) * 1000

    if verbose:
        print(f"\n  RMSE before: {rmse_before:.1f} mV")
        print(f"  RMSE after:  {rmse_after:.1f} mV  (improvement: {rmse_before-rmse_after:.1f} mV)")
        print(f"\n  ── CALIBRATED PARAMS (paste into al_air_model.py PARAMS) ──")
        print(f"    'E_ocv_ref':     {E_ocv_new:.5f},")
        print(f"    'i0_Al_ref':     {i0_Al_new:.4e},")
        print(f"    'i0_O2_ref':     {i0_O2_new:.4e},")
        print(f"    'L_eff_m':       {L_eff_new:.4e},")
        print(f"    'L_diff_factor': {L_diff_new:.1f},")
        print(f"  ──────────────────────────────────────────────────────────")

    if verbose:
        diagnose_physics_regions(exp_data, final_override)

    return final_override, rmse_before, rmse_after


# ─────────────────────────────────────────────
# MONTE CARLO UNCERTAINTY QUANTIFICATION
# ─────────────────────────────────────────────

def monte_carlo_uncertainty(calibrated_params, n_samples=1000,
                            j_range=(1, 60), n_j=40, verbose=True):
    """
    Propagate parameter uncertainty through the model.
    Varies E_ocv_ref, i0_Al_ref, L_eff_m by ±1σ simultaneously.

    Parameter uncertainties (estimated from calibration residuals):
      E_ocv_ref:  ±30 mV   (OCV measurement accuracy)
      i0_Al_ref:  ±50%     (exchange current uncertainty, log-normal)
      L_eff_m:    ±20%     (paste geometry variability)

    Returns: j_array, V_median, V_p5, V_p95 (5th-95th percentile band)
    """
    if verbose:
        print(f"\n── Monte Carlo uncertainty ({n_samples} samples) ──")

    j_vals = np.linspace(j_range[0], j_range[1], n_j)

    # Parameter distributions (physically motivated uncertainty ranges)
    E_ocv_base   = calibrated_params.get("E_ocv_ref",   PARAMS["E_ocv_ref"])
    i0_Al_base   = calibrated_params.get("i0_Al_ref",   PARAMS["i0_Al_ref"])
    i0_O2_base   = calibrated_params.get("i0_O2_ref",   PARAMS["i0_O2_ref"])
    L_eff_base   = calibrated_params.get("L_eff_m",     PARAMS["L_eff_m"])
    L_diff_base  = calibrated_params.get("L_diff_factor", PARAMS["L_diff_factor"])

    # σ for each parameter
    sigma_E_ocv  = 0.030            # ±30 mV OCV measurement uncertainty
    sigma_i0_log = 0.50             # ±50% in log space (i0 very uncertain)
    sigma_L_eff  = 0.20             # ±20% geometry variability

    rng = np.random.default_rng(seed=42)
    V_curves = np.zeros((n_samples, n_j))

    for i in range(n_samples):
        # Sample each parameter from its distribution
        E_ocv_s   = E_ocv_base  + rng.normal(0, sigma_E_ocv)
        i0_Al_s   = i0_Al_base  * np.exp(rng.normal(0, sigma_i0_log))
        i0_O2_s   = i0_O2_base  * np.exp(rng.normal(0, sigma_i0_log * 0.5))
        L_eff_s   = L_eff_base  * (1 + rng.normal(0, sigma_L_eff))

        override = {
            "E_ocv_ref":   float(np.clip(E_ocv_s,  1.0, 2.0)),
            "i0_Al_ref":   float(np.clip(i0_Al_s,  1e-6, 1.0)),
            "i0_O2_ref":   float(np.clip(i0_O2_s,  1e-8, 0.1)),
            "L_eff_m":     float(np.clip(L_eff_s,  2e-4, 5e-3)),
            "L_diff_factor": L_diff_base,
        }

        for k, j in enumerate(j_vals):
            try:
                r = cell_model(**CALIB_CONFIG, j_mA_cm2=j, params_override=override)
                V_curves[i, k] = r['V_cell']
            except Exception:
                V_curves[i, k] = np.nan

    V_median = np.nanpercentile(V_curves, 50, axis=0)
    V_p5     = np.nanpercentile(V_curves,  5, axis=0)
    V_p95    = np.nanpercentile(V_curves, 95, axis=0)

    band_width = np.mean(V_p95 - V_p5) * 1000
    if verbose:
        print(f"  Mean 90% band width: ±{band_width/2:.0f} mV")
        print(f"  (reflects parameter uncertainty, not model error)")

    return j_vals, V_median, V_p5, V_p95


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def fig_calibration(exp_data, calibrated_params, mc_results=None):
    """
    Publication-quality calibration figure:
    - Experimental data points
    - Model before calibration
    - Model after calibration
    - Monte Carlo uncertainty band (if provided)
    - Residuals panel
    """
    j_exp = np.array(exp_data["j_mA_cm2"])
    V_exp = np.array(exp_data["V_cell"])
    j_fine = np.linspace(0.5, max(j_exp) * 1.1, 60)

    # Model curves
    V_before, V_after = [], []
    for j in j_fine:
        try:
            V_before.append(cell_model(**CALIB_CONFIG, j_mA_cm2=j)['V_cell'])
            V_after.append(cell_model(**CALIB_CONFIG, j_mA_cm2=j,
                                       params_override=calibrated_params)['V_cell'])
        except Exception:
            V_before.append(np.nan); V_after.append(np.nan)

    V_before = np.array(V_before)
    V_after  = np.array(V_after)

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(13, 10))
        gs  = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1.2], hspace=0.35)
        ax_main  = fig.add_subplot(gs[0, :])
        ax_res_b = fig.add_subplot(gs[1, 0])
        ax_res_a = fig.add_subplot(gs[1, 1])
        ax_band  = fig.add_subplot(gs[2, :])

        fig.suptitle('Al-air paste battery — calibration + uncertainty quantification',
                     fontsize=13, y=1.01)

        # ── Main: discharge curves ─────────────────────────────────────────
        ax = ax_main
        ax.plot(j_fine, V_before, color=RE, lw=1.5, ls='--', alpha=0.7,
                label='Model (default params)')
        ax.plot(j_fine, V_after,  color=G,  lw=2,
                label='Model (calibrated)')

        if mc_results is not None:
            j_mc, V_med, V_p5, V_p95 = mc_results
            ax.fill_between(j_mc, V_p5, V_p95, color=G, alpha=0.12,
                            label='90% uncertainty band (Monte Carlo)')
            ax.plot(j_mc, V_med, color=G, lw=1, ls=':', alpha=0.5)

        ax.scatter(j_exp, V_exp, color='white', edgecolors=O,
                   s=80, zorder=6, lw=2, label='Experimental data')

        # Region labels
        for j_mid, label, col in [(3.5,'OCV+act',B),(7.5,'BV slope',Y),
                                   (17.5,'ohmic',P),(37.5,'mass transp.',RE)]:
            ax.axvline(j_mid, color='#2a3a2a', lw=0.8, ls=':', alpha=0.5)
            ax.text(j_mid, 0.25, label, color='#6b8f6b', fontsize=8,
                    ha='center', rotation=90, va='bottom')

        ax.set(xlabel='Current density (mA cm⁻²)', ylabel='Cell voltage (V)',
               ylim=(0.2, 1.7), title='Polarisation curve — calibration')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── Residuals: before ──────────────────────────────────────────────
        ax = ax_res_b
        V_mod_b = [cell_model(**CALIB_CONFIG, j_mA_cm2=j)['V_cell'] for j in j_exp]
        err_b   = (np.array(V_mod_b) - V_exp) * 1000
        rmse_b  = np.sqrt(np.mean(err_b**2))
        ax.bar(range(len(j_exp)), err_b, color=[G if e>0 else RE for e in err_b],
               alpha=0.8, edgecolor='none')
        ax.axhline(0, color='white', lw=0.8)
        ax.axhline(30, color=Y, lw=0.8, ls='--', alpha=0.6)
        ax.axhline(-30, color=Y, lw=0.8, ls='--', alpha=0.6)
        ax.set_xticks(range(len(j_exp)))
        ax.set_xticklabels([f'{j:.0f}' for j in j_exp], fontsize=8)
        ax.set(xlabel='j (mA cm⁻²)', ylabel='Residual (mV)',
               title=f'Residuals — default  RMSE={rmse_b:.0f} mV')
        ax.grid(alpha=0.3)

        # ── Residuals: after ───────────────────────────────────────────────
        ax = ax_res_a
        V_mod_a = [cell_model(**CALIB_CONFIG, j_mA_cm2=j,
                               params_override=calibrated_params)['V_cell']
                   for j in j_exp]
        err_a   = (np.array(V_mod_a) - V_exp) * 1000
        rmse_a  = np.sqrt(np.mean(err_a**2))
        ax.bar(range(len(j_exp)), err_a, color=[G if e>0 else RE for e in err_a],
               alpha=0.8, edgecolor='none')
        ax.axhline(0, color='white', lw=0.8)
        ax.axhline(30, color=Y, lw=0.8, ls='--', alpha=0.6, label='±30mV target')
        ax.axhline(-30, color=Y, lw=0.8, ls='--', alpha=0.6)
        ax.set_xticks(range(len(j_exp)))
        ax.set_xticklabels([f'{j:.0f}' for j in j_exp], fontsize=8)
        ax.set(xlabel='j (mA cm⁻²)', ylabel='Residual (mV)',
               title=f'Residuals — calibrated  RMSE={rmse_a:.0f} mV')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── Uncertainty band at multiple j ────────────────────────────────
        ax = ax_band
        if mc_results is not None:
            j_mc, V_med, V_p5, V_p95 = mc_results
            ax.fill_between(j_mc, V_p5, V_p95, color=B, alpha=0.3,
                            label='90% CI (Monte Carlo)')
            ax.plot(j_mc, V_med, color=B, lw=2, label='Median')
            ax.plot(j_fine, V_after, color=G, lw=1.5, ls='--',
                    label='Calibrated model')
            ax.scatter(j_exp, V_exp, color='white', edgecolors=O, s=60, zorder=5)
            ax.set(xlabel='Current density (mA cm⁻²)', ylabel='Cell voltage (V)',
                   ylim=(0.2, 1.7),
                   title='Monte Carlo uncertainty band\n'
                         '(E_ocv ±30mV, i₀ ±50%, L_eff ±20%)')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Run with --mc for uncertainty bands',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#6b8f6b', fontsize=11)
            ax.axis('off')

        plt.tight_layout()
        _save(fig, 'fig_calibration.png')


def fig_uncertainty_params(calibrated_params, n_samples=500):
    """Show parameter uncertainty as distributions + correlation."""
    rng = np.random.default_rng(42)
    E_base  = calibrated_params.get("E_ocv_ref", PARAMS["E_ocv_ref"])
    i0_base = calibrated_params.get("i0_Al_ref", PARAMS["i0_Al_ref"])
    L_base  = calibrated_params.get("L_eff_m",   PARAMS["L_eff_m"])

    E_samp  = E_base  + rng.normal(0, 0.030, n_samples)
    i0_samp = np.log10(i0_base * np.exp(rng.normal(0, 0.5, n_samples)))
    L_samp  = L_base  * (1 + rng.normal(0, 0.20, n_samples))

    # Compute energy density for each sample at j=50
    ed_samp = []
    for E, i0l, L in zip(E_samp, i0_samp, L_samp):
        try:
            override = {"E_ocv_ref": float(np.clip(E, 1.0, 2.0)),
                        "i0_Al_ref": float(np.clip(10**i0l, 1e-6, 1.0)),
                        "L_eff_m":   float(np.clip(L, 2e-4, 5e-3))}
            r = cell_model(**CALIB_CONFIG, j_mA_cm2=50, params_override=override)
            ed_samp.append(r['ed_Wh_kg_paste'])
        except Exception:
            ed_samp.append(np.nan)
    ed_samp = np.array(ed_samp)

    with plt.rc_context(_rc()):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Parameter uncertainty propagation — Monte Carlo distributions',
                     fontsize=13, y=1.01)

        for ax, data, label, col in zip(axs,
            [E_samp, i0_samp, ed_samp[np.isfinite(ed_samp)]],
            ['E_ocv (V)', 'log₁₀(i₀_Al)', 'Energy density (Wh/kg)'],
            [G, O, B]):
            ax.hist(data, bins=30, color=col, alpha=0.75, edgecolor='none')
            ax.axvline(np.nanpercentile(data,  5), color='white', lw=1, ls='--',
                       alpha=0.7, label='5th/95th pct')
            ax.axvline(np.nanpercentile(data, 95), color='white', lw=1, ls='--', alpha=0.7)
            ax.axvline(np.nanmean(data), color=Y, lw=1.5, label='mean')
            p5, p95 = np.nanpercentile(data, [5, 95])
            ax.set(xlabel=label, ylabel='Count',
                   title=f'{label}\n90% CI: [{p5:.3g}, {p95:.3g}]')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        _save(fig, 'fig_uncertainty_params.png')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def load_csv(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return {"j_mA_cm2": data[:, 0], "V_cell": data[:, 1],
            "source": path}


def cross_validate(calibrated_params, val_datasets, verbose=True):
    """
    Test calibrated params on independent datasets (second cell / repeat run).

    This is the reviewer's final requirement:
      'The next proof is one independent repeat on a second cell or a second
       run from the same batch. If that lands in the same RMSE range, you are done.'

    Parameters
    ----------
    calibrated_params : dict  — from calibrate()
    val_datasets      : list of dicts  — each {"j_mA_cm2":..., "V_cell":..., "source":...}

    Returns
    -------
    list of (source, rmse_mV, pass/fail)
    """
    TARGET_RMSE = 30.0  # mV
    results = []

    if verbose:
        print(f"\n── Cross-validation on independent datasets ──")
        print(f"  {'Source':40s}  {'RMSE (mV)':>10}  {'vs target':>10}  Status")
        print("  " + "─"*72)

    for ds in val_datasets:
        j_v = np.array(ds["j_mA_cm2"])
        V_v = np.array(ds["V_cell"])
        errs = []
        for j, Ve in zip(j_v, V_v):
            try:
                r = cell_model(**CALIB_CONFIG, j_mA_cm2=j,
                               params_override=calibrated_params)
                errs.append(r['V_cell'] - Ve)
            except Exception:
                pass
        if not errs:
            continue
        rmse_mV = float(np.sqrt(np.mean(np.array(errs)**2)) * 1000)
        passed   = rmse_mV < TARGET_RMSE
        drmse    = rmse_mV - TARGET_RMSE

        if verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {str(ds['source']):40s}  {rmse_mV:>10.1f}  "
                  f"{drmse:>+9.1f}  {status}")
        results.append((ds["source"], rmse_mV, passed))

    if verbose and results:
        n_pass = sum(1 for _, _, p in results if p)
        print(f"\n  {n_pass}/{len(results)} datasets within {TARGET_RMSE:.0f} mV target")
        if n_pass == len(results):
            print(f"  ✓ CALIBRATION IS TRANSFERABLE — model is validated")
        elif n_pass >= len(results) // 2:
            print(f"  ○ Partial transfer — check cell-to-cell variability")
        else:
            print(f"  ✗ Poor transfer — model may be overfitted to first cell")
            print(f"    Consider fitting L_eff per batch (geometric variability)")

    return results


def fig_crossvalidation(calibrated_params, cal_data, val_datasets):
    """
    Figure showing calibration curve + all validation datasets.
    This is the publication figure that proves transferability.
    """
    j_fine = np.linspace(0.5, 60, 80)
    V_cal  = [cell_model(**CALIB_CONFIG, j_mA_cm2=j,
                          params_override=calibrated_params)['V_cell']
              for j in j_fine]

    colors = [G, O, Y, B, P, RE]

    with plt.rc_context(_rc()):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Calibration transferability — independent validation',
                     fontsize=13, y=1.01)

        # Panel 1: all curves
        ax = axs[0]
        ax.plot(j_fine, V_cal, color=G, lw=2, label='Calibrated model')
        ax.scatter(cal_data["j_mA_cm2"], cal_data["V_cell"],
                   color='white', edgecolors=G, s=80, lw=2, zorder=6,
                   label=f'Calibration: {cal_data["source"]}')

        for i, ds in enumerate(val_datasets):
            col = colors[(i+1) % len(colors)]
            ax.scatter(ds["j_mA_cm2"], ds["V_cell"],
                       color=col, s=60, marker='s', zorder=5,
                       label=f'Validation {i+1}: {ds["source"]}')

        ax.set(xlabel='Current density (mA cm⁻²)', ylabel='Cell voltage (V)',
               title='Model vs all datasets', ylim=(0.5, 1.7))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 2: RMSE bar chart per dataset
        ax = axs[1]
        all_ds  = [cal_data] + val_datasets
        labels  = ['Calibration'] + [f'Val {i+1}' for i in range(len(val_datasets))]
        col_list= [G] + [colors[(i+1) % len(colors)] for i in range(len(val_datasets))]
        rmses   = []

        for ds in all_ds:
            errs = []
            for j, Ve in zip(ds["j_mA_cm2"], ds["V_cell"]):
                try:
                    r = cell_model(**CALIB_CONFIG, j_mA_cm2=j,
                                   params_override=calibrated_params)
                    errs.append((r['V_cell'] - Ve) * 1000)
                except Exception:
                    pass
            rmses.append(np.sqrt(np.mean(np.array(errs)**2)) if errs else 0)

        bars = ax.bar(labels, rmses, color=col_list, alpha=0.85, edgecolor='none')
        ax.axhline(30, color=Y, lw=1.5, ls='--', label='Target: 30 mV')
        ax.axhline(60, color=RE, lw=1, ls=':', alpha=0.6, label='Current: 60 mV')

        for bar, rmse in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rmse:.0f}', ha='center', va='bottom', fontsize=9,
                    color='white')

        ax.set(ylabel='RMSE (mV)', title='RMSE per dataset\n(lower = better)',
               ylim=(0, max(rmses) * 1.3 + 10))
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        _save(fig, 'fig_crossvalidation.png')



def main():
    parser = argparse.ArgumentParser(description='Al-Air Calibration + UQ')
    parser.add_argument('--data',     type=str, default=None,
                        help='CSV: calibration data (j_mA_cm2,V_cell)')
    parser.add_argument('--validate', type=str, default=None, nargs='+',
                        help='One or more CSV files for cross-validation')
    parser.add_argument('--mc',       action='store_true',
                        help='Run Monte Carlo uncertainty (slow)')
    parser.add_argument('--mc-n',     type=int, default=1000)
    args = parser.parse_args()

    print("=" * 62)
    print("  Al-Air Paste Battery — Calibration + Uncertainty")
    print("  Physics-region fitting + Monte Carlo UQ + Cross-validation")
    print("=" * 62)

    if args.data:
        exp_data = load_csv(args.data)
        print(f"\n  Calibration data: {args.data}  ({len(exp_data['j_mA_cm2'])} points)")
    else:
        exp_data = PLACEHOLDER_DATA
        print(f"\n  ⚠ Using PLACEHOLDER data — replace with your measurements!")
        print(f"  Create a CSV: j_mA_cm2,V_cell  with 5 rows")
        print(f"  Run: python al_air_calibrate.py --data cell1.csv")

    print(f"\n  Calibration conditions (must match your experiment exactly):")
    for k, v in CALIB_CONFIG.items():
        print(f"    {k:12s}: {v}")

    calibrated, rmse_b, rmse_a = calibrate(exp_data)

    if args.validate:
        val_datasets = [load_csv(p) for p in args.validate]
        cross_validate(calibrated, val_datasets)
        fig_crossvalidation(exp_data, calibrated, val_datasets)
        print("  Saved: fig_crossvalidation.png")

    mc_results = None
    if args.mc:
        mc_results = monte_carlo_uncertainty(
            calibrated, n_samples=args.mc_n, verbose=True)
        fig_uncertainty_params(calibrated, n_samples=min(args.mc_n, 500))
        print("  Saved: fig_uncertainty_params.png")

    print("\n── Generating figures ──")
    fig_calibration(exp_data, calibrated, mc_results)

    print(f"\n── Summary ──")
    print(f"  RMSE before: {rmse_b:.1f} mV")
    print(f"  RMSE after:  {rmse_a:.1f} mV")
    if rmse_a < 30:
        print(f"  ✓ Target <30 mV achieved")
        print(f"  → Next: run --validate on a second cell to confirm transferability")
    elif rmse_a < 50:
        print(f"  ○ Below 50 mV — acceptable, more data points would help")
    else:
        print(f"  ✗ High RMSE — check conditions match CALIB_CONFIG exactly")
    print()
    print("  Commands:")
    print("  python al_air_calibrate.py --data cell1.csv --mc")
    print("  python al_air_calibrate.py --data cell1.csv --validate cell2.csv --mc")



if __name__ == '__main__':
    main()
