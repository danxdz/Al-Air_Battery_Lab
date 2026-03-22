"""
Al-Air Alloy Explorer v2.0
===========================
Atomic-level alloy analysis using the unified al_air_model.py physics.
All physics now lives in al_air_model.py — this file handles CLI + plots only.

Physics inside:
  - Vegard's law linear mixing (E0, M, rho, chi)
  - Brønsted-Evans-Polanyi (BEP) activation energy scaling
  - Miedema binary interaction model
  - Independent-pathway inhibitor combination

Usage:
  python al_air_alloy.py                             # metal comparison table
  python al_air_alloy.py --alloy Al:0.95,In:0.03,Sn:0.02
  python al_air_alloy.py --sweep                     # binary Al-X sweeps
  python al_air_alloy.py --opt --goal balanced        # optimise alloy
  python al_air_alloy.py --opt --goal stability
"""

import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc

warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from al_air_model import (
    cell_model, alloy_properties, ATOMS, PARAMS,
    BASE_CONFIG, F, n_Al, M_Al,
    G, O, Y, B, P, RE, _rc, _save,
)

PRACTICAL_METALS = ["Al","Mg","In","Sn","Zn","Ga","Mn","Ti","Ce","Si"]


# ─────────────────────────────────────────────
# BINARY SWEEP
# ─────────────────────────────────────────────

def binary_sweep(base_cfg, additive, fractions=None, j=50):
    if fractions is None:
        fractions = np.linspace(0, 0.10, 20)
    results = []
    for f in fractions:
        comp = {"Al": 1.0-f, additive: f} if f > 1e-4 else {"Al": 1.0}
        try:
            r = cell_model(**base_cfg, j_mA_cm2=j, composition=comp)
            results.append({
                "f":         f * 100,
                "V":         r["V_cell"],
                "ed_cell":   r["ed_Wh_kg_paste"],
                "ed_sys":    r["ed_Wh_kg_system"],
                "net_ed":    r["net_useful_ed"],
                "pd":        r["pd_W_kg_paste"],
                "parasitic": r["parasitic_pct"],
                "util":      r["utilisation_pct"],
                "inh_total": r["inh_total_pct"],
            })
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────
# ALLOY OPTIMIZER
# ─────────────────────────────────────────────

def optimize_alloy(base_cfg, additives, goal="balanced", n_samples=4000, j=50):
    """
    Multi-objective alloy optimizer with:
    - Literature-grounded composition hard limits
    - True non-dominated Pareto front (NSGA-II style sorting)
    - Three objectives: net energy, corrosion stability, power

    Composition limits (literature):
      Mg  ≤ 3%  (Yoo 2014: NDE worsens above this)
      Ga  ≤ 2%  (Ga is expensive; liquid phase issues above 2%)
      In  ≤ 3%  (Mokhtar 2015: diminishing returns above 3%)
      Sn  ≤ 3%  (Fan 2014)
      Zn  ≤ 5%  (Doche 2002)
      Total additives ≤ 8% (structural integrity)
    """
    print(f"\n── Alloy optimizer: true Pareto | {n_samples:,} samples ──")
    print(f"  Additives: {additives}")
    print(f"  Composition limits: Mg≤3% Ga≤2% In/Sn≤3% Zn≤5% total≤8%")

    # Literature-grounded max fractions per element
    MAX_FRAC = {"Mg":0.03, "Ga":0.02, "In":0.03, "Sn":0.03,
                "Zn":0.05, "Ti":0.05, "Mn":0.03, "Ce":0.02,
                "Si":0.05}
    MAX_TOTAL = 0.08

    n_add = len(additives)
    hi_bounds = np.array([MAX_FRAC.get(a, 0.05) for a in additives])
    sampler = qmc.LatinHypercube(d=n_add, seed=42)
    samples = qmc.scale(sampler.random(n=n_samples),
                        np.zeros(n_add), hi_bounds)

    results = []
    for row in samples:
        # Enforce total additive limit
        if row.sum() > MAX_TOTAL:
            row = row / row.sum() * MAX_TOTAL * 0.95

        comp = {"Al": float(1.0 - row.sum())}
        for sym, f in zip(additives, row):
            if f > 5e-4:
                comp[sym] = float(f)

        try:
            r = cell_model(**base_cfg, j_mA_cm2=j, composition=comp)
            results.append({
                "score":     r["net_useful_ed"] * (1 - r["parasitic_pct"]/100),
                "net_ed":    r["net_useful_ed"],
                "corr":      r["parasitic_pct"],
                "power":     r["pd_W_kg_paste"],
                "voltage":   r["V_cell"],
                "ed_cell":   r["ed_Wh_kg_paste"],
                "ed_sys":    r["ed_Wh_kg_system"],
                "util":      r["utilisation_pct"],
                "synergy":   r["alloy"]["has_synergy"] if r["alloy"] else False,
                "comp":      comp,
                "result":    r,
            })
        except Exception:
            pass

    # ── True Pareto non-dominated sort ────────────────────────────────────────
    def dominates(a, b):
        """a dominates b if better/equal on all, strictly better on one."""
        return (a["net_ed"]  >= b["net_ed"]  and
                a["corr"]    <= b["corr"]     and
                a["power"]   >= b["power"]    and
                (a["net_ed"] > b["net_ed"] or
                 a["corr"]   < b["corr"]  or
                 a["power"]  > b["power"]))

    pareto = []
    for r in results:
        dominated = any(dominates(f, r) for f in pareto)
        if not dominated:
            pareto = [f for f in pareto if not dominates(r, f)]
            pareto.append(r)

    # Sort Pareto by net energy descending
    pareto.sort(key=lambda x: x["net_ed"], reverse=True)
    # Sort all results by balanced score for top-10 table
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"  Valid: {len(results):,}  |  Pareto front: {len(pareto)} configs")
    best = results[0]
    print(f"  Best balanced composition:")
    for sym, f in best["comp"].items():
        print(f"    {sym}: {f*100:.2f}%")
    print(f"  Net energy={best['net_ed']:.1f} Wh/kg  Corr={best['corr']:.2f}%  "
          f"Synergy={'Yes' if best['synergy'] else 'No'}")

    return results, pareto



# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def fig_metal_table(base_cfg, j=50):
    """Bar chart comparing pure metals."""
    syms, roles, ed_th, inh_effs, Ea_vals = [], [], [], [], []
    for sym in PRACTICAL_METALS:
        a = ATOMS[sym]
        syms.append(sym)
        roles.append(a["role"])
        ed_th.append((a["n"] * F / a["M"]) / 3600.0)
        inh_effs.append(a["inh_eff"] * 100)
        Ea_vals.append(a["Ea_corr"])

    role_col = {"base": G, "activator": O, "inhibitor": B, "modifier": Y}
    colors = [role_col.get(r, '#6b8f6b') for r in roles]

    with plt.rc_context(_rc()):
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Anode metal comparison — atomic properties (CRC Handbook + NIST)',
                     fontsize=13, y=1.01)

        for ax, vals, ylabel, title in zip(axs,
            [ed_th, inh_effs, Ea_vals],
            ['Theoretical Ed (Wh/kg)', 'Inhibition of Al corrosion (%)',
             'Corrosion activation energy (kJ/mol)'],
            ['Specific energy per kg anode metal',
             'Inhibitor effectiveness when added\nto Al anode or KOH electrolyte',
             'Self-corrosion stability\n(higher Ea = slower corrosion)']):
            ax.bar(syms, vals, color=colors, alpha=0.85, edgecolor='none')
            ax.set(ylabel=ylabel, title=title)
            ax.grid(axis='y', alpha=0.3)

        # Legend on first plot
        for role, col in role_col.items():
            axs[0].bar([], [], color=col, alpha=0.85, label=role)
        axs[0].legend(fontsize=8)
        axs[0].axhline(ed_th[syms.index("Al")], color='white', lw=1, ls='--', alpha=0.4)

        plt.tight_layout()
        _save(fig, 'fig_alloy_metal_comparison.png')


def fig_binary_sweeps(base_cfg, additives, j=50):
    """Binary Al-X sweep plots using net useful energy."""
    add_cols = [O, B, G, Y, P, RE]

    with plt.rc_context(_rc()):
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Binary Al-X alloy sweeps (0–10 mol% additive)\n'
                     'Net useful energy = energy density × utilisation',
                     fontsize=12, y=1.03)

        for add, col in zip(additives, add_cols):
            data = binary_sweep(base_cfg, add, j=j)
            if not data: continue
            fs   = [d["f"]         for d in data]
            nets = [d["net_ed"]    for d in data]
            prs  = [d["parasitic"] for d in data]
            pds  = [d["pd"]        for d in data]
            axs[0].plot(fs, nets, color=col, label=add)
            axs[1].plot(fs, prs,  color=col, label=add)
            axs[2].plot(fs, pds,  color=col, label=add)

        for ax, ylabel, title in zip(axs,
            ['Net useful energy (Wh/kg)', 'Parasitic corrosion (%)', 'Power (W/kg)'],
            ['Net useful energy vs additive %\n(ed × utilisation)',
             'Corrosion loss vs additive %',
             'Power density vs additive %']):
            ax.set(xlabel='Additive content (mol%)', ylabel=ylabel, title=title)
            ax.legend(fontsize=9); ax.grid(alpha=0.3)

        plt.tight_layout()
        _save(fig, 'fig_alloy_binary_sweep.png')


def fig_design_space(results):
    """
    Corrosion vs net energy scatter, coloured by voltage.
    This is the single most useful plot for engineering decisions:
    the optimal region is top-left (low corrosion, high energy).
    Inverted x-axis: better stability is visually on the right.
    Stars = synergy configurations.
    """
    corr    = [r["corr"]    for r in results]
    energy  = [r["net_ed"]  for r in results]
    voltage = [r["voltage"] for r in results]
    synergy = [r["synergy"] for r in results]

    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(9, 7))

        # All points, coloured by voltage
        sc = ax.scatter(corr, energy, c=voltage,
                        cmap='plasma', alpha=0.4, s=10, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Cell voltage (V)', fontsize=10)

        # Synergy configs highlighted
        syn_c = [c for c, s in zip(corr,   synergy) if s]
        syn_e = [e for e, s in zip(energy, synergy) if s]
        syn_v = [v for v, s in zip(voltage,synergy) if s]
        if syn_c:
            ax.scatter(syn_c, syn_e, c=syn_v, cmap='plasma',
                       marker='*', s=120, zorder=5, edgecolors='white',
                       lw=0.5, label=f'Mg+In/Sn synergy ({len(syn_c)})')

        # Optimal region annotation
        opt_e = np.percentile(energy, 80)
        opt_c = np.percentile(corr,   25)
        ax.axhline(opt_e, color=G, lw=1, ls=':', alpha=0.6)
        ax.axvline(opt_c, color=O, lw=1, ls=':', alpha=0.6)
        ax.text(opt_c * 0.98, opt_e * 1.005,
                'optimal\nregion', color=G, fontsize=8, ha='right', va='bottom')

        ax.invert_xaxis()   # low corrosion = better → right side
        ax.set(xlabel='Corrosion loss (%)  ← better stability',
               ylabel='Net useful energy (Wh/kg)',
               title='Design space: net energy vs corrosion\n'
                     '(colour = voltage, ★ = synergy configs, x-axis inverted)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(fig, 'fig_design_space.png')


def fig_alloy_pareto(results, pareto, additives):
    """True Pareto front + composition profiles + design space scatter."""
    top10_balanced = results[:10]

    with plt.rc_context(_rc()):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Multi-objective alloy optimisation — true Pareto front\n'
                     '(constrained: Mg≤3% Ga≤2% In/Sn≤3% total≤8%)',
                     fontsize=12, y=1.02)

        # Panel 1: Pareto front — net energy vs corrosion
        ax = axs[0]
        # All results as background
        ax.scatter([r["net_ed"] for r in results],
                   [100-r["corr"] for r in results],
                   c=[r["voltage"] for r in results],
                   cmap='viridis', alpha=0.2, s=8, rasterized=True)
        # Pareto front highlighted
        px = [r["net_ed"] for r in pareto]
        py = [100-r["corr"]  for r in pareto]
        sc = ax.scatter(px, py, c=[r["voltage"] for r in pareto],
                        cmap='viridis', s=80, zorder=5,
                        edgecolors='white', lw=0.8)
        plt.colorbar(sc, ax=ax, label='Cell voltage (V)')
        # Connect Pareto front
        p_sorted = sorted(pareto, key=lambda x: x["net_ed"])
        ax.plot([r["net_ed"] for r in p_sorted],
                [100-r["corr"] for r in p_sorted],
                color='white', lw=0.8, ls='--', alpha=0.5)
        # Mark synergy configs
        syn = [r for r in pareto if r["synergy"]]
        if syn:
            ax.scatter([r["net_ed"] for r in syn],
                       [100-r["corr"] for r in syn],
                       marker='*', s=200, color=Y, zorder=6,
                       label=f'Mg+In/Sn synergy ({len(syn)})')
            ax.legend(fontsize=8)
        ax.set(xlabel='Net useful energy (Wh/kg)',
               ylabel='Stability (100 − corrosion %)',
               title=f'True Pareto front\n({len(pareto)} non-dominated configs)')
        ax.grid(alpha=0.3)

        # Panel 2: Design space — corrosion vs energy, colour=voltage
        ax = axs[1]
        sc2 = ax.scatter([r["corr"]   for r in results],
                         [r["net_ed"] for r in results],
                         c=[r["voltage"] for r in results],
                         cmap='plasma', alpha=0.3, s=8, rasterized=True)
        plt.colorbar(sc2, ax=ax, label='Voltage (V)')
        # Highlight optimal region
        opt_ed  = np.percentile([r["net_ed"] for r in results], 85)
        opt_cor = np.percentile([r["corr"]   for r in results], 20)
        ax.axhline(opt_ed,  color=G, lw=1, ls=':', alpha=0.7, label=f'85th pct energy')
        ax.axvline(opt_cor, color=O, lw=1, ls=':', alpha=0.7, label=f'20th pct corrosion')
        ax.set(xlabel='Corrosion loss (%)',
               ylabel='Net useful energy (Wh/kg)',
               title='Design space\n(optimal region: top-left)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 3: Top 10 compositions
        ax = axs[2]
        x = np.arange(len(additives)); w = 0.07
        cmap = plt.cm.viridis(np.linspace(0.15, 0.9, 10))
        for i, r in enumerate(top10_balanced):
            fracs = [r["comp"].get(a, 0)*100 for a in additives]
            ax.bar(x + i*w - 4.5*w, fracs, w*0.9,
                   color=cmap[i], alpha=0.85, label=f'#{i+1}')
        ax.set_xticks(x); ax.set_xticklabels(additives)
        ax.set(ylabel='Additive content (mol%)',
               title='Top 10 balanced compositions\n(Al = remainder)',
               ylim=(0, 3.5))
        ax.legend(fontsize=7, ncol=2); ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        _save(fig, 'fig_alloy_pareto.png')



# ─────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────

def print_binary_table(base_cfg, j=50):
    """Binary Al-5%X comparison with theoretical vs effective energy split."""
    r_Al = cell_model(**base_cfg, j_mA_cm2=j, composition={"Al": 1.0})

    print(f"\n── Binary Al-X at 5 mol% additive ──")
    print(f"  {'Alloy':>10}  {'ΔEd_th':>8}  {'ΔEd_eff':>9}  "
          f"{'ΔNet':>7}  {'ΔCorr%':>8}  {'Mechanism'}")
    print("  " + "─"*80)

    for sym in ["Mg","In","Sn","Zn","Ga","Ce","Ti","Si"]:
        comp5 = {"Al": 0.95, sym: 0.05}
        try:
            r5 = cell_model(**base_cfg, j_mA_cm2=j, composition=comp5)
            dEd_th  = r5['ed_theoretical_paste'] - r_Al['ed_theoretical_paste']
            dEd_eff = r5['ed_Wh_kg_paste']       - r_Al['ed_Wh_kg_paste']
            dNet    = r5['net_useful_ed']         - r_Al['net_useful_ed']
            dCorr   = r5['parasitic_pct']         - r_Al['parasitic_pct']
            mech    = r5['dominant_mechanism']
            feas    = "⚠" if not r5['feasibility']['feasible'] else " "

            print(f"  {feas}Al+5%{sym:>3}:  {dEd_th:>+7.0f}  {dEd_eff:>+8.0f}  "
                  f"{dNet:>+6.0f}  {dCorr:>+7.2f}  {mech}")
        except Exception as e:
            print(f"  Al+5%{sym:>3}:  ERROR {e}")

    print(f"\n  ΔEd_th  = theoretical energy change (mass dilution only)")
    print(f"  ΔEd_eff = effective energy change (incl. efficiency)")
    print(f"  ΔNet    = net useful energy change (ed × util)")
    print(f"  ⚠       = exceeds practical composition limit")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_comp(s):
    comp = {}
    for part in s.split(","):
        sym, frac = part.strip().split(":")
        comp[sym.strip()] = float(frac.strip())
    return comp


def main():
    parser = argparse.ArgumentParser(description='Al-Air Alloy Explorer v2.0')
    parser.add_argument('--alloy',  type=str,   default=None,
                        help='e.g. Al:0.95,In:0.03,Sn:0.02')
    parser.add_argument('--sweep',  action='store_true')
    parser.add_argument('--opt',    action='store_true')
    parser.add_argument('--goal',   default='balanced',
                        choices=['energy','power','stability','net','balanced'])
    parser.add_argument('--additives', type=str, default='Mg,In,Sn,Zn,Ga',
                        help='Comma-separated additive elements for sweep/opt')
    args = parser.parse_args()

    additives = args.additives.split(",")

    print("=" * 62)
    print("  Al-Air Paste Battery — Atomic Alloy Explorer v2.0")
    print("  BEP · Miedema · Vegard — unified with al_air_model.py")
    print("=" * 62)

    # Use v3 optimum as base
    base = dict(BASE_CONFIG)

    # Atomic database summary
    print(f"\n── Atomic database ({len(ATOMS)} metals) ──")
    print(f"  {'Metal':>5}  {'E0(V)':>7}  {'Ed(Wh/kg)':>10}  "
          f"{'Ea(kJ)':>7}  {'Inh%':>5}  {'Role':>12}  Note")
    print("  " + "─"*88)
    for sym in PRACTICAL_METALS:
        a = ATOMS[sym]
        ed = (a["n"] * F / a["M"]) / 3600.0
        print(f"  {sym:>5}  {a['E0']:>7.3f}  {ed:>10.0f}  "
              f"{a['Ea_corr']:>7.1f}  {a['inh_eff']*100:>4.0f}%  "
              f"{a['role']:>12}  {a['note'][:42]}")

    # Pure Al baseline
    print("\n── Pure Al baseline (v3 optimum conditions) ──")
    r_Al = cell_model(**base, j_mA_cm2=50, composition={"Al": 1.0})
    print(f"  Cell voltage:     {r_Al['V_cell']:.3f} V")
    print(f"  Energy (cell):    {r_Al['ed_Wh_kg_paste']:.1f} Wh/kg")
    print(f"  Energy (system):  {r_Al['ed_Wh_kg_system']:.1f} Wh/kg")
    print(f"  Net useful:       {r_Al['net_useful_ed']:.1f} Wh/kg  (ed × util)")
    print(f"  Parasitic:        {r_Al['parasitic_pct']:.2f} %")
    print(f"  Utilisation:      {r_Al['utilisation_pct']:.1f} %")

    # Specific alloy
    if args.alloy:
        comp = parse_comp(args.alloy)
        print(f"\n── Specific alloy: {comp} ──")
        r = cell_model(**base, j_mA_cm2=50, composition=comp)
        ap = r["alloy"]
        feas = r["feasibility"]

        # Feasibility warnings
        if not feas["feasible"]:
            print(f"  ⚠ FEASIBILITY WARNINGS:")
            for w in feas["warnings"]:
                print(f"    ! {w}")
        else:
            print(f"  ✓ Feasible (total additives: {feas['total_add_pct']:.1f}%)")

        print(f"  Dominant mechanism: {r['dominant_mechanism']}")
        print(f"  E0_mix:            {ap['E0_mix']:.3f} V  (Al: -1.662)")
        print(f"  ΔOCV:              {ap['dOCV']*1000:+.1f} mV")
        print(f"  Ea_alloy:          {ap['Ea_mix_kJ']:.1f} kJ/mol  (Al: 48.0)")
        print(f"  Inh total:         {r['inh_total_pct']:.1f} %")
        print(f"  Synergy active:    {'Yes (Mg+In/Sn)' if ap.get('has_synergy') else 'No'}")
        print(f"  ─────────────────────────────────────────")
        print(f"  Voltage:           {r['V_cell']:.3f} V")
        print(f"  Parasitic:         {r['parasitic_pct']:.2f} %")
        print(f"  Ed theoretical:    {r['ed_theoretical_paste']:.1f} Wh/kg  ← dilution only")
        print(f"  Ed effective:      {r['ed_Wh_kg_paste']:.1f} Wh/kg  ← incl. efficiency")
        print(f"  Ed system:         {r['ed_Wh_kg_system']:.1f} Wh/kg  ← incl. engineering")
        print(f"  Net useful:        {r['net_useful_ed']:.1f} Wh/kg  ← ed × util")
        print(f"  Power:             {r['pd_W_kg_paste']:.1f} W/kg")
        print(f"  Utilisation:       {r['utilisation_pct']:.1f} %")
        print(f"\n  Δ vs pure Al:")
        print(f"    ΔEd theoretical: {r['ed_theoretical_paste'] - r_Al['ed_theoretical_paste']:+.1f} Wh/kg  (mass dilution)")
        print(f"    ΔEd effective:   {r['ed_Wh_kg_paste'] - r_Al['ed_Wh_kg_paste']:+.1f} Wh/kg")
        print(f"    ΔNet useful:     {r['net_useful_ed'] - r_Al['net_useful_ed']:+.1f} Wh/kg")
        print(f"    ΔCorrosion:      {r['parasitic_pct'] - r_Al['parasitic_pct']:+.2f} %")
        print(f"    ΔVoltage:        {(r['V_cell'] - r_Al['V_cell'])*1000:+.1f} mV")

    # Binary comparison table
    print_binary_table(base)

    # Figures
    print("\n── Generating figures ──")
    fig_metal_table(base)

    if args.sweep:
        fig_binary_sweeps(base, additives)

    if args.opt:
        opt_res, pareto = optimize_alloy(base, additives, goal=args.goal)
        fig_design_space(opt_res)
        fig_alloy_pareto(opt_res, pareto, additives)

    print("\n── Summary ──")
    print("  fig_alloy_metal_comparison.png")
    if args.sweep: print("  fig_alloy_binary_sweep.png")
    if args.opt:   print("  fig_alloy_pareto.png")
    print()
    print("  Tip: This script reads all physics from al_air_model.py")
    print("  Any change to PARAMS or cell_model() automatically propagates here")
    print()
    print("  Commands:")
    print("  python al_air_alloy.py --alloy Al:0.95,In:0.03,Sn:0.02")
    print("  python al_air_alloy.py --sweep")
    print("  python al_air_alloy.py --opt --goal net")


if __name__ == '__main__':
    main()
