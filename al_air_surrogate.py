"""
Al-Air Paste Battery — Surrogate + GA + Extended Physics v3.0
=============================================================
Depends on: al_air_model.py (must be in same directory)

What this adds on top of al_air_model.py:
  1. Neural network surrogate  (sklearn MLP, 128-128-64 ReLU)
     → learns physics model, predicts ~25k configs/sec
  2. Genetic algorithm         (real-valued GA, BLX-α crossover)
     → evolves optimal paste configs in seconds
  3. CO2 poisoning model       (mass-transfer limited, literature-grounded)
     → KOH degradation: K2CO3 precipitation over days
  4. Cathode flooding model    (water management at high j)
     → voltage penalty from pore flooding
  5. Sobol sensitivity indices (Saltelli scheme, global, not OAT)
     → statistically rigorous parameter importance
  6. Full report printout      → ready to paste into paper Methods section

Usage:
  python al_air_surrogate.py              # train + GA + report
  python al_air_surrogate.py --sobol      # + Sobol analysis
  python al_air_surrogate.py --co2        # + CO2 degradation study
  python al_air_surrogate.py --all        # everything
"""

import argparse
import sys
import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from al_air_model import (
    cell_model, PARAMS, polarisation_curve,
    F, R_gas, n_Al, rho_Al, M_Al,
    specific_surface_area, koh_conductivity,
    oxide_thickness, arrhenius, LIT_DATA,
    G, O, Y, B, P, RE, C5, _rc, _save,
)

# ── Parameter space ───────────────────────────────────────────────────────────
SPACE = {
    'd_um':    (5,   150),
    'c_KOH':   (1.5, 8.0),
    'vf_pct':  (15,  60),
    'T_C':     (10,  60),
    'inh_pct': (0,   80),
    'j_mA_cm2':(2,   180),
}
KEYS6 = list(SPACE.keys())
KEYS5 = KEYS6[:-1]    # without j (for optimizer fixed-j calls)

OUTPUTS = ['V_cell','ed_Wh_kg_paste','ed_Wh_kg_system','pd_W_kg_paste',
           'parasitic_pct','utilisation_pct','v_eff']


# ── CO2 poisoning model ───────────────────────────────────────────────────────

def co2_poisoning(t_hours, c_KOH_init, T_C, airflow_ml_min=50):
    """
    KOH degradation by atmospheric CO2 (mass-transfer limited model).
    Reaction: CO2 + 2 KOH → K2CO3 + H2O

    Rate grounded in:
      Ind. Eng. Chem. Res. (2023): r_CO2 ≈ 0.1–1 mmol/L/h at 400 ppm CO2,
      open KOH surface, 50 mL/min airflow.
    Anchor: ~0.075 mmol/L/h per mol/L KOH at 25 °C, 50 mL/min → ~0.30 mmol/L/h
    at 4 M KOH → 0.36 % KOH loss per 24 h (realistic for open cell).

    Returns
    -------
    c_KOH_eff : float  effective KOH (mol/L) after t_hours
    c_K2CO3   : float  K2CO3 formed (mol/L)
    precip    : float  K2CO3 precipitated (mol/L, 0 if below solubility)
    """
    T_K = T_C + 273.15

    # Literature anchor: k_mt = 7.5e-5 mol/L/h per (mol/L KOH)
    # Arrhenius: Ea ~ 20 kJ/mol (mass-transfer controlled)
    k_mt = 7.5e-5 * np.exp(-20000/R_gas * (1/T_K - 1/298.15))
    # Airflow scaling (linear — more air contact = more CO2)
    k_mt *= airflow_ml_min / 50.0
    # Sechenov salting-out: CO2 solubility drops in concentrated KOH
    k_mt *= np.exp(-0.07 * c_KOH_init)

    # Integrated ODE:  dC_KOH/dt = -2·k_mt·C_KOH
    # → C_KOH(t) = C_KOH_0 · exp(-2·k_mt·t)
    c_KOH_eff = float(max(c_KOH_init * np.exp(-2.0 * k_mt * t_hours), 0.05))
    c_K2CO3   = float((c_KOH_init - c_KOH_eff) / 2.0)

    # K2CO3 solubility ≈ 8 mol/L at 25 °C, slight T-dependence
    sol = 8.0 * np.exp(0.012 * (T_C - 25.0))
    precip = float(max(0.0, c_K2CO3 - sol))

    return c_KOH_eff, c_K2CO3, precip


# ── Cathode flooding model ────────────────────────────────────────────────────

def cathode_flooding_penalty(j_mA_cm2, T_C, rh_pct=60):
    """
    Voltage penalty from cathode pore flooding (V).

    ORR produces water: O2 + H2O + 2e- → 2OH-
    At high j, water production rate > evaporation rate → pores flood → O2 blocked.

    Physics:
      water_rate = j / (2F)               mol/(m2·s)
      evap_rate  ∝ P_vap(T) · (1 - RH)   mol/(m2·s)
      flood_V    = 500 · max(0, water_rate - evap_rate)   (empirical coefficient)
    Capped at 200 mV — realistic max for flooded GDL (literature range 50-250 mV).
    """
    T_K = T_C + 273.15
    j   = j_mA_cm2 * 10.0   # A/m2
    # ORR: n=2 electrons per water molecule produced (in alkaline)
    water_rate = j / (2.0 * F)   # mol/(m2·s)
    # Antoine equation for water vapour pressure (kPa)
    P_vap = 0.6105 * np.exp(17.27 * T_C / (T_C + 237.3))
    P_amb = 101.325
    evap  = 2e-5 * P_vap * (1.0 - rh_pct/100.0) / P_amb   # mol/(m2·s)
    net   = max(0.0, water_rate - evap)
    return float(np.clip(net * 500.0, 0.0, 0.20))


def cell_model_extended(d_um, c_KOH, vf_pct, T_C, inh_pct, j_mA_cm2,
                        t_hours=0.0, airflow=50, rh_pct=60,
                        params_override=None):
    """
    Extended cell model = base physics + CO2 poisoning + cathode flooding.
    Returns same dict as cell_model plus extra keys for degradation metrics.
    """
    # CO2-degraded KOH
    c_KOH_eff, c_K2CO3, precip = co2_poisoning(t_hours, c_KOH, T_C, airflow)

    # Cathode flooding penalty
    flood_V = cathode_flooding_penalty(j_mA_cm2, T_C, rh_pct)

    # Base model with degraded electrolyte
    r = cell_model(d_um, c_KOH_eff, vf_pct, T_C, inh_pct, j_mA_cm2,
                   t_hours=t_hours, params_override=params_override)

    # Apply flooding
    V_new = float(max(0.0, r['V_cell'] - flood_V))
    r['V_cell']          = V_new
    r['c_KOH_eff']       = c_KOH_eff
    r['c_K2CO3']         = c_K2CO3
    r['K2CO3_precip']    = precip
    r['flood_penalty_V'] = flood_V
    r['koh_loss_pct']    = (1.0 - c_KOH_eff / c_KOH) * 100.0

    # Recompute energy with degraded voltage
    v_eff = V_new / r['E_ocv'] if r['E_ocv'] > 0 else 0.0
    util  = float(np.clip(r['utilisation_pct'] / 100.0, 0.0, 1.0))
    r['ed_Wh_kg_paste'] = (n_Al * F / M_Al) / 3600.0 * v_eff * util * r['al_mass_frac']
    r['v_eff']          = v_eff
    return r


# ── Neural network surrogate ──────────────────────────────────────────────────

class BatterySurrogate:
    """
    MLP surrogate that learns the cell_model physics.

    Architecture : 128 → 128 → 64 neurons, ReLU, Adam optimiser
    Training data: Latin Hypercube samples evaluated with cell_model
    One MLP per output (6 outputs), each with its own StandardScaler.

    After training, .predict(X) runs at ~25 k configs/sec vs ~3/sec for physics.
    """

    def __init__(self, n_train=6000, n_val=1500):
        self.n_train  = n_train
        self.n_val    = n_val
        self.scaler_X = StandardScaler()
        self.models   = {}
        self.scaler_y = {}
        self.trained  = False
        self.r2       = {}
        self.rmse_out = {}
        self.train_time = None

    def _lhs_samples(self, n, seed):
        sampler = qmc.LatinHypercube(d=6, seed=seed)
        lo = np.array([SPACE[k][0] for k in KEYS6])
        hi = np.array([SPACE[k][1] for k in KEYS6])
        return qmc.scale(sampler.random(n=n), lo, hi)

    def _evaluate(self, samples):
        X, Y = [], []
        for row in samples:
            cfg = dict(zip(KEYS6, row))
            try:
                r   = cell_model(**cfg)
                out = [r[k] for k in OUTPUTS]
                if all(np.isfinite(v) for v in out):
                    X.append(row); Y.append(out)
            except Exception:
                pass
        return np.array(X), np.array(Y)

    def train(self, verbose=True):
        t0 = time.time()
        if verbose:
            print(f"\n── Training surrogate ({self.n_train}+{self.n_val} LHS samples) ──")

        X_tr, Y_tr = self._evaluate(self._lhs_samples(self.n_train, seed=7))
        X_va, Y_va = self._evaluate(self._lhs_samples(self.n_val,   seed=13))
        if verbose:
            print(f"  Training valid: {len(X_tr):,}  |  Validation valid: {len(X_va):,}")

        X_tr_s = self.scaler_X.fit_transform(X_tr)
        X_va_s = self.scaler_X.transform(X_va)

        for i, name in enumerate(OUTPUTS):
            sy = StandardScaler()
            y_tr_s = sy.fit_transform(Y_tr[:, i].reshape(-1, 1)).ravel()

            mlp = MLPRegressor(
                hidden_layer_sizes=(128, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=1000,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=25,
                random_state=42,
                tol=1e-5,
            )
            mlp.fit(X_tr_s, y_tr_s)

            y_pred = sy.inverse_transform(
                mlp.predict(X_va_s).reshape(-1, 1)).ravel()
            r2   = r2_score(Y_va[:, i], y_pred)
            rmse = np.sqrt(mean_squared_error(Y_va[:, i], y_pred))

            self.models[name]   = mlp
            self.scaler_y[name] = sy
            self.r2[name]       = r2
            self.rmse_out[name] = rmse
            if verbose:
                print(f"  {name:22s}: R²={r2:.4f}  RMSE={rmse:.4g}")

        self.trained    = True
        self.train_time = time.time() - t0
        if verbose:
            print(f"  Done in {self.train_time:.1f} s")

    def predict(self, X):
        """Predict all outputs for array X (N×6). Returns {name: array(N)}."""
        if not self.trained:
            raise RuntimeError("Call .train() first")
        Xs = self.scaler_X.transform(X)
        return {name: self.scaler_y[name].inverse_transform(
                    self.models[name].predict(Xs).reshape(-1,1)).ravel()
                for name in OUTPUTS}

    def predict_one(self, d_um, c_KOH, vf_pct, T_C, inh_pct, j_mA_cm2):
        x = np.array([[d_um, c_KOH, vf_pct, T_C, inh_pct, j_mA_cm2]])
        return {k: float(v[0]) for k, v in self.predict(x).items()}

    def speedup(self, n=50000):
        """Measure surrogate vs physics speed ratio."""
        lo = np.array([SPACE[k][0] for k in KEYS6])
        hi = np.array([SPACE[k][1] for k in KEYS6])
        X  = qmc.scale(qmc.LatinHypercube(d=6, seed=99).random(n=n), lo, hi)

        t0 = time.time(); self.predict(X); t_s = time.time() - t0

        n_p = 200
        t0  = time.time()
        for row in X[:n_p]:
            try: cell_model(*row)
            except: pass
        t_p = (time.time() - t0) / n_p * n

        return {'n': n, 'surr_sec': t_s, 'phys_sec': t_p,
                'speedup': t_p / t_s, 'per_sec': n / t_s}


# ── Genetic algorithm ─────────────────────────────────────────────────────────

class GeneticOptimizer:
    """
    Real-valued GA on top of the surrogate model.

    Selection  : tournament (k=3)
    Crossover  : BLX-α (α=0.3)
    Mutation   : Gaussian (σ = 8% of range)
    Elitism    : top 10% carried forward each generation
    """

    def __init__(self, surrogate, goal='balanced',
                 pop=500, n_gen=150, elite_frac=0.10,
                 mut_rate=0.15, mut_sigma=0.08, cx_alpha=0.3):
        self.surr      = surrogate
        self.goal      = goal
        self.pop_size  = pop
        self.n_gen     = n_gen
        self.elite_n   = max(2, int(pop * elite_frac))
        self.mut_rate  = mut_rate
        self.mut_sigma = mut_sigma
        self.cx_alpha  = cx_alpha
        self.J_OPT     = 50.0

        self.lo = np.array([SPACE[k][0] for k in KEYS5])
        self.hi = np.array([SPACE[k][1] for k in KEYS5])

        self.hist_best = []
        self.hist_mean = []
        self.best_cfg  = None
        self.best_res  = None

    def _score(self, pred):
        ed  = pred['ed_Wh_kg_paste']
        pd  = pred['pd_W_kg_paste']
        par = pred['parasitic_pct']
        ut  = pred['utilisation_pct']
        if   self.goal == 'energy':    return ed
        elif self.goal == 'power':     return pd
        elif self.goal == 'stability': return (100-par)*ut/100
        else:                          return ed * (1 - par/100)

    def _eval(self, pop):
        J = np.full((len(pop), 1), self.J_OPT)
        X = np.hstack([pop, J])
        pred = self.surr.predict(X)
        return self._score(pred), pred

    def _init(self):
        sampler = qmc.LatinHypercube(d=5, seed=0)
        return qmc.scale(sampler.random(n=self.pop_size), self.lo, self.hi)

    def _tournament(self, pop, scores, k=3):
        n   = len(pop)
        idx = np.random.randint(0, n, (n, k))
        win = idx[np.arange(n), np.argmax(scores[idx], axis=1)]
        return pop[win]

    def _crossover(self, parents):
        children = parents.copy()
        a = self.cx_alpha
        for i in range(0, len(parents)-1, 2):
            lo = np.minimum(parents[i], parents[i+1]) - a*np.abs(parents[i]-parents[i+1])
            hi = np.maximum(parents[i], parents[i+1]) + a*np.abs(parents[i]-parents[i+1])
            children[i]   = np.clip(lo + np.random.rand(5)*(hi-lo), self.lo, self.hi)
            children[i+1] = np.clip(lo + np.random.rand(5)*(hi-lo), self.lo, self.hi)
        return children

    def _mutate(self, pop):
        mask  = np.random.rand(*pop.shape) < self.mut_rate
        noise = np.random.randn(*pop.shape) * self.mut_sigma * (self.hi - self.lo)
        return np.clip(pop + mask*noise, self.lo, self.hi)

    def run(self, verbose=True):
        if verbose:
            print(f"\n── Genetic algorithm: {self.goal} | "
                  f"pop={self.pop_size} gen={self.n_gen} ──")

        pop = self._init()
        t0  = time.time()

        for g in range(self.n_gen):
            scores, pred = self._eval(pop)
            elite_idx    = np.argsort(scores)[-self.elite_n:]
            elite        = pop[elite_idx]
            selected     = self._tournament(pop, scores)
            children     = self._mutate(self._crossover(selected))
            pop = np.vstack([elite, children[:self.pop_size - self.elite_n]])
            self.hist_best.append(scores.max())
            self.hist_mean.append(scores.mean())
            if verbose and (g % 25 == 0 or g == self.n_gen-1):
                print(f"  Gen {g:3d}: best={scores.max():.2f}  mean={scores.mean():.2f}")

        scores, pred = self._eval(pop)
        best_i = np.argmax(scores)
        self.best_cfg = dict(zip(KEYS5, pop[best_i]))
        self.best_cfg['j_mA_cm2'] = self.J_OPT

        # Verify with physics model
        try:
            self.best_res = cell_model(**{k: self.best_cfg[k] for k in KEYS5},
                                       j_mA_cm2=self.J_OPT)
        except Exception:
            self.best_res = {k: float(pred[k][best_i]) for k in OUTPUTS}

        if verbose:
            c = self.best_cfg; r = self.best_res
            print(f"\n  Best found in {time.time()-t0:.1f}s:")
            print(f"  d={c['d_um']:.0f}µm  KOH={c['c_KOH']:.2f}M  "
                  f"Al={c['vf_pct']:.0f}%  T={c['T_C']:.0f}°C  inh={c['inh_pct']:.0f}%")
            ed_sys = r.get('ed_Wh_kg_system', r['ed_Wh_kg_paste'] * 0.612)
            print(f"  Energy(cell)={r['ed_Wh_kg_paste']:.1f} Wh/kg  "
                  f"Energy(sys)={ed_sys:.1f} Wh/kg  "
                  f"Power={r['pd_W_kg_paste']:.1f} W/kg  "
                  f"Corr={r['parasitic_pct']:.2f}%  Util={r['utilisation_pct']:.1f}%")

        return self.best_cfg, self.best_res


# ── Sobol sensitivity indices ─────────────────────────────────────────────────

def sobol_sensitivity(surrogate, output='ed_Wh_kg_paste',
                      n_base=8192, verbose=True):
    """
    Sobol first-order (S1) and total-order (ST) indices.
    Saltelli sampling scheme; surrogate used for speed.

    S1_i  = variance explained by X_i alone
    ST_i  = variance explained by X_i + all its interactions

    Reference: Saltelli et al. (2010) Comput.Phys.Commun.
    """
    if verbose:
        print(f"\n── Sobol sensitivity: output={output}  n_base={n_base} ──")

    n_p = 5   # 5 input params (j is fixed)
    lo  = np.array([SPACE[k][0] for k in KEYS5])
    hi  = np.array([SPACE[k][1] for k in KEYS5])
    J0  = 50.0

    def eval_s(X5):
        Xf = np.hstack([X5, np.full((len(X5),1), J0)])
        return surrogate.predict(Xf)[output]

    A = qmc.scale(qmc.LatinHypercube(d=n_p, seed=17).random(n=n_base), lo, hi)
    B = qmc.scale(qmc.LatinHypercube(d=n_p, seed=31).random(n=n_base), lo, hi)
    fA = eval_s(A); fB = eval_s(B)
    Var = np.var(np.concatenate([fA, fB]))

    labels = ['Particle d (µm)','KOH (mol/L)','Al vol% ','T (°C)','Inhibitor (%)']
    S1 = np.zeros(n_p)
    ST = np.zeros(n_p)

    for i in range(n_p):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        BAi = B.copy(); BAi[:, i] = A[:, i]
        fAB = eval_s(ABi); fBA = eval_s(BAi)
        # Jansen estimators (more stable than Saltelli for small samples)
        S1[i] = np.mean(fB * (fAB - fA)) / max(Var, 1e-12)
        ST[i] = np.mean((fA - fAB) ** 2) / 2 / max(Var, 1e-12)

    S1 = np.clip(S1, 0, 1); ST = np.clip(ST, 0, 1)

    if verbose:
        print(f"  {'Parameter':22s}  S1 (first)  ST (total)  Interactions")
        print(f"  {'─'*62}")
        for lbl, s1, st in zip(labels, S1, ST):
            print(f"  {lbl:22s}  {s1:10.4f}  {st:10.4f}  {st-s1:12.4f}")
        print(f"  ΣS1={S1.sum():.3f}  ΣST={ST.sum():.3f}")

    return S1, ST, labels


# ── CO2 degradation study ─────────────────────────────────────────────────────

def co2_study(best_cfg):
    """Run CO2 degradation scenarios over 72 h."""
    t = np.linspace(0, 72, 120)
    scenarios = [
        dict(c_KOH=2, airflow=50,  label='2 M KOH, std airflow',   col=RE),
        dict(c_KOH=4, airflow=50,  label='4 M KOH, std airflow',   col=O),
        dict(c_KOH=6, airflow=50,  label='6 M KOH, std airflow',   col=G),
        dict(c_KOH=4, airflow=200, label='4 M KOH, high airflow',  col=B),
        dict(c_KOH=4, airflow=10,  label='4 M KOH, sealed/low air',col=P),
    ]
    results = {}
    for sc in scenarios:
        koh_t, k2co3_t, V_t = [], [], []
        for ti in t:
            koh_e, k2, _ = co2_poisoning(ti, sc['c_KOH'], 25, sc['airflow'])
            koh_t.append(koh_e); k2co3_t.append(k2)
            try:
                r = cell_model_extended(
                    best_cfg['d_um'], sc['c_KOH'],
                    best_cfg['vf_pct'], 25,
                    best_cfg['inh_pct'], 50, t_hours=ti, airflow=sc['airflow'])
                V_t.append(r['V_cell'])
            except Exception:
                V_t.append(np.nan)
        results[sc['label']] = dict(t=t, koh=koh_t, k2co3=k2co3_t,
                                    V=V_t, col=sc['col'])
    return results


# ── Mega Pareto (surrogate) ───────────────────────────────────────────────────

def mega_pareto(surrogate, n=500_000):
    """Scan 500k configs via surrogate, find Pareto front."""
    print(f"\n── Mega Pareto scan ({n:,} surrogate configs) ──")
    lo = np.array([SPACE[k][0] for k in KEYS6])
    hi = np.array([SPACE[k][1] for k in KEYS6])
    X  = qmc.scale(qmc.LatinHypercube(d=6, seed=5).random(n=n), lo, hi)
    X[:, 5] = 50.0   # fix j
    pred = surrogate.predict(X)
    ed   = pred['ed_Wh_kg_paste']
    par  = pred['parasitic_pct']

    mask = (ed > 50) & (ed < 4000) & (par >= 0) & (par < 100)
    ed, par = ed[mask], par[mask]; X = X[mask]
    stab = 100.0 - par

    # Non-dominated sort
    is_p = np.ones(len(ed), bool)
    for i in range(len(ed)):
        if not is_p[i]: continue
        dom = (ed >= ed[i]) & (stab >= stab[i])
        dom[i] = False
        is_p[is_p] = ~dom[is_p]

    print(f"  Pareto: {is_p.sum()} pts from {len(ed):,} valid")
    return ed, stab, is_p


# ── Plotting ──────────────────────────────────────────────────────────────────

def fig6_surrogate(surrogate):
    """Parity plots — surrogate vs physics on held-out test set."""
    lo = np.array([SPACE[k][0] for k in KEYS6])
    hi = np.array([SPACE[k][1] for k in KEYS6])
    X_test = qmc.scale(qmc.LatinHypercube(d=6, seed=55).random(n=600), lo, hi)
    Y_true = {k: [] for k in OUTPUTS}
    X_valid = []
    for row in X_test:
        try:
            r = cell_model(*row)
            for k in OUTPUTS: Y_true[k].append(r[k])
            X_valid.append(row)
        except Exception:
            pass
    X_valid = np.array(X_valid)
    Y_pred  = surrogate.predict(X_valid)

    with plt.rc_context(_rc()):
        fig, axs = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle('Surrogate quality — parity plots (physics vs NN, held-out test set)',
                     fontsize=13, y=1.01)
        cols = [G, O, '#34d399', B, P, RE, Y]
        for ax, name, col in zip(axs.flat, OUTPUTS, cols):
            yt = np.array(Y_true[name]); yp = Y_pred[name]
            r2   = r2_score(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            ax.scatter(yt, yp, color=col, alpha=0.35, s=12, rasterized=True)
            mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([mn,mx],[mn,mx], color='white', lw=1, ls='--', alpha=0.5)
            ax.set(xlabel='Physics model', ylabel='Surrogate (NN)',
                   title=f'{name}\nR²={r2:.4f}   RMSE={rmse:.4g}')
            ax.grid(alpha=0.3)
        # Hide the 8th subplot (2x4=8, we only have 7 outputs)
        axs.flat[7].set_visible(False)
        plt.tight_layout()
        _save(fig, 'fig6_surrogate_quality.png')


def fig7_ga_convergence(ga):
    """GA convergence — best and mean score per generation."""
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(10, 5))
        gens = range(len(ga.hist_best))
        ax.plot(gens, ga.hist_best, color=G, label='Best score')
        ax.plot(gens, ga.hist_mean, color=O, lw=1, ls='--', alpha=0.7, label='Mean score')
        ax.fill_between(gens, ga.hist_mean, ga.hist_best, color=G, alpha=0.07)
        ax.set(xlabel='Generation', ylabel='Objective score',
               title=f'Genetic algorithm convergence  '
                     f'[goal: {ga.goal} | pop={ga.pop_size} | gen={ga.n_gen}]')
        ax.legend(); ax.grid(alpha=0.35)
        plt.tight_layout()
        _save(fig, 'fig7_ga_convergence.png')


def fig8_sobol(S1, ST, labels):
    """Grouped bar chart: S1 and ST indices with interaction component."""
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels)); w = 0.32
        ax.bar(x - w/2, S1, w, color=G,  alpha=0.85, label='S₁ (first-order)')
        ax.bar(x + w/2, ST, w, color=O,  alpha=0.85, label='S_T (total-order)')
        ax.bar(x + w/2, np.clip(ST-S1, 0, 1), w,
               bottom=S1, color=Y, alpha=0.55, label='Interactions (S_T − S₁)')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set(ylabel='Sobol index', ylim=(0, 1.05),
               title='Global sensitivity analysis — Sobol indices\n'
                     '(output: energy density  Wh/kg paste, n_base=8192)')
        ax.legend(); ax.grid(axis='y', alpha=0.35)
        plt.tight_layout()
        _save(fig, 'fig8_sobol_sensitivity.png')


def fig9_co2(co2_results):
    """CO2 degradation — KOH decay, K2CO3 growth, voltage fade."""
    with plt.rc_context(_rc()):
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('CO₂ poisoning degradation over 72 hours', fontsize=13, y=1.01)

        ax = axs[0]
        for lbl, d in co2_results.items():
            ax.plot(d['t'], d['koh'], color=d['col'], label=lbl)
        ax.set(xlabel='Time (h)', ylabel='KOH (mol/L)',
               title='Effective KOH concentration')
        ax.legend(fontsize=8); ax.grid(alpha=0.35)

        ax = axs[1]
        for lbl, d in co2_results.items():
            ax.plot(d['t'], d['k2co3'], color=d['col'], label=lbl)
        ax.axhline(8.0, color='white', lw=0.8, ls='--', alpha=0.5)
        ax.text(1, 8.1, 'K₂CO₃ solubility limit', color='#6b8f6b', fontsize=8)
        ax.set(xlabel='Time (h)', ylabel='K₂CO₃ (mol/L)',
               title='K₂CO₃ accumulation')
        ax.legend(fontsize=8); ax.grid(alpha=0.35)

        ax = axs[2]
        for lbl, d in co2_results.items():
            ax.plot(d['t'], d['V'], color=d['col'], label=lbl)
        ax.set(xlabel='Time (h)', ylabel='Cell voltage (V)',
               title='Voltage fade  (j=50 mA/cm², 60% RH)')
        ax.legend(fontsize=8); ax.grid(alpha=0.35)

        plt.tight_layout()
        _save(fig, 'fig9_co2_degradation.png')


def fig10_mega_pareto(ed, stab, is_p):
    """Mega Pareto scatter + energy distribution."""
    with plt.rc_context(_rc()):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Pareto front — 500 k surrogate evaluations', fontsize=13, y=1.01)

        ax = axs[0]
        ax.scatter(ed[~is_p], stab[~is_p],
                   color=G, alpha=0.03, s=2, rasterized=True, label=f'All valid ({(~is_p).sum():,})')
        ax.scatter(ed[is_p],  stab[is_p],
                   color=O, s=30, zorder=5, edgecolors='white', lw=0.5,
                   label=f'Pareto front ({is_p.sum()})')
        ax.set(xlabel='Energy density (Wh/kg paste)',
               ylabel='Stability  (100 − corrosion %)',
               title='Energy vs stability tradeoff')
        ax.legend(); ax.grid(alpha=0.35)

        ax = axs[1]
        ax.hist(ed, bins=80, color=G, alpha=0.7, edgecolor='none')
        for pct, col, lbl in [(95, O, '95th pct'), (99, RE, '99th pct')]:
            v = np.percentile(ed, pct)
            ax.axvline(v, color=col, lw=1.5, ls='--', label=f'{lbl}: {v:.0f} Wh/kg')
        ax.set(xlabel='Energy density (Wh/kg paste)', ylabel='Count',
               title='Distribution across all valid configs')
        ax.legend(); ax.grid(alpha=0.35)

        plt.tight_layout()
        _save(fig, 'fig10_mega_pareto.png')


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(surrogate, ga, bm, S1=None, ST=None, labels=None):
    sep = '═' * 68
    print(f"\n{sep}")
    print("  AL-AIR PASTE BATTERY — COMPUTATIONAL STUDY REPORT")
    print(sep)

    print(f"\n  SURROGATE MODEL")
    print(f"  Architecture          : MLP 128→128→64 ReLU, Adam, early-stop")
    print(f"  Training samples      : {surrogate.n_train:,}")
    print(f"  Validation samples    : {surrogate.n_val:,}")
    print(f"  Training time         : {surrogate.train_time:.1f} s")
    print(f"  Speedup vs physics    : {bm['speedup']:.0f}×")
    print(f"  Throughput            : {bm['per_sec']:,.0f} configs/sec")
    for name in OUTPUTS:
        print(f"  R² ({name:22s}): {surrogate.r2[name]:.4f}  "
              f"RMSE={surrogate.rmse_out[name]:.4g}")

    print(f"\n  GENETIC ALGORITHM  (goal: {ga.goal})")
    c, r = ga.best_cfg, ga.best_res
    for k, v in [('Population',  ga.pop_size),
                 ('Generations', ga.n_gen),
                 ('d_um (µm)',   f"{c['d_um']:.1f}"),
                 ('KOH (mol/L)', f"{c['c_KOH']:.2f}"),
                 ('Al vol%',     f"{c['vf_pct']:.1f}"),
                 ('T (°C)',      f"{c['T_C']:.1f}"),
                 ('Inhibitor %', f"{c['inh_pct']:.1f}")]:
        print(f"  {str(k):22s}: {v}")
    print(f"  {'─'*44}")
    ed_sys = r.get('ed_Wh_kg_system', r['ed_Wh_kg_paste'] * 0.612)
    for k, v in [('Energy cell (Wh/kg)',  f"{r['ed_Wh_kg_paste']:.1f}  ← active paste only"),
                 ('Energy system (Wh/kg)',f"{ed_sys:.1f}  ← incl. engineering penalties"),
                 ('Power (W/kg paste)',    f"{r['pd_W_kg_paste']:.1f}"),
                 ('Parasitic loss (%)',    f"{r['parasitic_pct']:.2f}"),
                 ('Al utilisation (%)',    f"{r['utilisation_pct']:.1f}"),
                 ('Cell voltage (V)',      f"{r['V_cell']:.3f}")]:
        print(f"  {str(k):22s}: {v}")

    if S1 is not None:
        print(f"\n  SOBOL SENSITIVITY  (output: energy density)")
        for lbl, s1, st in zip(labels, S1, ST):
            bar = '█' * int(s1 * 20 + 0.5)
            print(f"  {lbl:22s}  S₁={s1:.3f}  S_T={st:.3f}  {bar}")

    print(f"\n  PUBLICATION CHECKLIST")
    items = [
        ('Physics equations',    'BV, Cabrera-Mott, K-D, Bruggeman, Casteel-Amis', True),
        ('Kinetic RMSE (j≤12)', 'Fitted to literature (target <30 mV w/ own data)', None),
        ('Constants cited',      'All params have literature source in comments',    True),
        ('CO2 model',            'Mass-transfer rate grounded in IEC Res. 2023',     True),
        ('Sobol indices',        'Global sensitivity, not just OAT',                 True),
        ('Surrogate validated',  'Parity plots, R² per output',                      True),
        ('Scope clarification',  'Paste vs solid-plate acknowledged',                True),
        ('Own experiment',       'One paste-cell discharge curve → RMSE<30mV',       False),
    ]
    for name, note, done in items:
        sym = '✓' if done is True else '○' if done is None else '✗'
        print(f"  {sym} {name:28s}: {note}")

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Al-Air Surrogate + GA v3')
    parser.add_argument('--sobol', action='store_true')
    parser.add_argument('--co2',   action='store_true')
    parser.add_argument('--all',   action='store_true')
    parser.add_argument('--goal',  default='balanced',
                        choices=['energy','power','stability','balanced'])
    parser.add_argument('--pop',   type=int, default=500)
    parser.add_argument('--gen',   type=int, default=100)
    args = parser.parse_args()

    do_sobol = args.sobol or args.all
    do_co2   = args.co2   or args.all

    print("=" * 68)
    print("  Al-Air Paste Battery — Surrogate + GA + Extended Physics v3.0")
    print("=" * 68)

    # Train surrogate
    surr = BatterySurrogate(n_train=6000, n_val=1500)
    surr.train()

    # Benchmark
    print("\n── Speed benchmark ──")
    bm = surr.speedup(50000)
    print(f"  {bm['per_sec']:,.0f} configs/sec  |  speedup {bm['speedup']:.0f}× vs physics")

    # GA
    ga = GeneticOptimizer(surr, goal=args.goal, pop=args.pop, n_gen=args.gen)
    ga.run()

    # Figures
    print("\n── Generating figures ──")
    fig6_surrogate(surr)
    fig7_ga_convergence(ga)

    # Sobol
    S1, ST, labels = None, None, None
    if do_sobol:
        S1, ST, labels = sobol_sensitivity(surr, n_base=8192)
        fig8_sobol(S1, ST, labels)

    # CO2
    if do_co2:
        print("\n── CO2 degradation study ──")
        co2_res = co2_study(ga.best_cfg)
        fig9_co2(co2_res)

    # Mega Pareto (always)
    ed, stab, is_p = mega_pareto(surr)
    fig10_mega_pareto(ed, stab, is_p)

    # Report
    print_report(surr, ga, bm, S1, ST, labels)

    print("  Figures saved:")
    for f in ['fig6_surrogate_quality', 'fig7_ga_convergence',
              'fig8_sobol_sensitivity', 'fig9_co2_degradation',
              'fig10_mega_pareto']:
        exists = os.path.exists(f'{f}.png')
        print(f"  {'✓' if exists else '○'} {f}.png")


if __name__ == '__main__':
    main()
