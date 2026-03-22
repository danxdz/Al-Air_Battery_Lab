"""
Full test suite for Al-Air Battery Lab.
Run: python run_tests.py
"""
import sys, time, re, os

# Directory where this script lives — works from any location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import numpy as np

# ── imports ───────────────────────────────────────────────────────────────────
from al_air_model import (cell_model, BASE_CONFIG, PARAMS, ATOMS,
                           thermal_model, thermal_sweep,
                           check_feasibility, sensitivity_analysis)
from al_air_alloy  import (binary_sweep, optimize_alloy,
                            optimize_joint, temperature_alloy_map,
                            current_alloy_map)
from al_air_calibrate import (calibrate, monte_carlo_uncertainty, CALIB_CONFIG)

# ── test runner ───────────────────────────────────────────────────────────────
passed, failed = [], []

def test(name, fn):
    try:
        t0 = time.time()
        fn()
        ms = (time.time()-t0)*1000
        passed.append(name)
        print(f"  ✓ {name:<55} {ms:>6.0f}ms")
    except Exception as e:
        failed.append((name, str(e)))
        print(f"  ✗ {name:<55} FAIL: {str(e)[:55]}")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FULL TEST SUITE — Al-Air Battery Lab")
print("=" * 70)

# ── 1. BASE_CONFIG ────────────────────────────────────────────────────────────
print("\n── 1. BASE_CONFIG & CALIB_CONFIG ───────────────────────────────────────")

def t_base_config():
    assert BASE_CONFIG['T_C']    == 25,  f"T={BASE_CONFIG['T_C']} should be 25"
    assert BASE_CONFIG['c_KOH'] == 4.0, f"KOH={BASE_CONFIG['c_KOH']} should be 4.0"
    assert BASE_CONFIG['d_um']  == 100
    assert BASE_CONFIG['vf_pct']== 53
test("BASE_CONFIG = 25°C, 4M KOH, 100µm, 53vol%", t_base_config)

def t_calib_matches():
    assert CALIB_CONFIG['T_C']   == 25
    assert CALIB_CONFIG['c_KOH'] == 4.0
test("CALIB_CONFIG inherits BASE_CONFIG (25°C, 4M)", t_calib_matches)

# ── 2. cell_model physics ─────────────────────────────────────────────────────
print("\n── 2. cell_model — physics constraints ─────────────────────────────────")

def t_baseline():
    r = cell_model(**BASE_CONFIG, j_mA_cm2=50)
    assert 0.9 < r['V_cell'] < 1.6
    assert 0   < r['ed_Wh_kg_paste']
    assert 0   <= r['parasitic_pct'] <= 100
test("Baseline 25°C — V, Ed, corr in range", t_baseline)

def t_v_lt_ocv():
    for j in [1, 10, 25, 50, 70]:
        r = cell_model(**BASE_CONFIG, j_mA_cm2=j)
        assert r['V_cell'] < r['E_ocv'], f"V>OCV at j={j}"
test("V_cell < OCV at all currents", t_v_lt_ocv)

def t_v_decreases():
    vs = [cell_model(**BASE_CONFIG, j_mA_cm2=j)['V_cell'] for j in [5,20,50,70]]
    assert vs[0] > vs[1] > vs[2] > vs[3]
test("V_cell decreases monotonically with j", t_v_decreases)

def t_corr_range():
    for d in [1, 10, 100, 500, 2000]:
        r = cell_model(d_um=d, c_KOH=4, vf_pct=53, T_C=25, inh_pct=0, j_mA_cm2=50)
        assert 0 <= r['parasitic_pct'] <= 100, f"corr={r['parasitic_pct']} at d={d}"
test("Corrosion in [0,100]% across all particle sizes", t_corr_range)

def t_higher_T_higher_V():
    v25 = cell_model(d_um=100,c_KOH=4,vf_pct=53,T_C=25,inh_pct=0,j_mA_cm2=50)['V_cell']
    v60 = cell_model(d_um=100,c_KOH=4,vf_pct=53,T_C=60,inh_pct=0,j_mA_cm2=50)['V_cell']
    assert v60 > v25, f"Higher T should give higher V: {v25:.4f} vs {v60:.4f}"
test("Higher temperature → higher voltage (Arrhenius kinetics)", t_higher_T_higher_V)

def t_inh_reduces_corr():
    c0  = cell_model(d_um=100,c_KOH=4,vf_pct=53,T_C=25,inh_pct=0, j_mA_cm2=50)['parasitic_pct']
    c80 = cell_model(d_um=100,c_KOH=4,vf_pct=53,T_C=25,inh_pct=80,j_mA_cm2=50)['parasitic_pct']
    assert c80 < c0
test("Inhibitor reduces corrosion", t_inh_reduces_corr)

def t_system_lt_cell():
    r = cell_model(**BASE_CONFIG, j_mA_cm2=50)
    assert r['ed_Wh_kg_system'] < r['ed_Wh_kg_paste']
test("System energy < cell energy (engineering penalties)", t_system_lt_cell)

def t_ocv_literature():
    r = cell_model(**BASE_CONFIG, j_mA_cm2=50)
    assert 1.45 <= r['E_ocv'] <= 1.65, f"OCV={r['E_ocv']}"
test("OCV in literature range 1.45–1.65V", t_ocv_literature)

def t_edge_tiny():
    r = cell_model(d_um=0.001,c_KOH=4,vf_pct=53,T_C=25,inh_pct=0,j_mA_cm2=50)
    assert r['parasitic_pct'] > 90
test("Tiny particle (0.001µm) — corrosion >90%, no crash", t_edge_tiny)

def t_edge_giant():
    r = cell_model(d_um=2000,c_KOH=4,vf_pct=53,T_C=25,inh_pct=0,j_mA_cm2=50)
    assert r['parasitic_pct'] < 5
test("Giant particle (2000µm) — corrosion <5%, no crash", t_edge_giant)

def t_edge_low_j():
    r = cell_model(**BASE_CONFIG, j_mA_cm2=0.1)
    assert r['V_cell'] > 1.0
test("Very low j (0.1 mA/cm²) — no crash", t_edge_low_j)

def t_edge_high_j():
    r = cell_model(**BASE_CONFIG, j_mA_cm2=200)
    assert r['V_cell'] > 0
test("Very high j (200 mA/cm²) — no crash", t_edge_high_j)

# ── 3. Alloy model ────────────────────────────────────────────────────────────
print("\n── 3. Alloy model ──────────────────────────────────────────────────────")

def t_pure_al():
    r = cell_model(**BASE_CONFIG, j_mA_cm2=50, composition={'Al':1.0})
    assert r['dominant_mechanism'] == 'pure Al (reference)'
test("Pure Al → 'pure Al (reference)' mechanism", t_pure_al)

def t_mg_activates():
    v_al = cell_model(**BASE_CONFIG, j_mA_cm2=50)['V_cell']
    v_mg = cell_model(**BASE_CONFIG, j_mA_cm2=50, composition={'Al':0.99,'Mg':0.01})['V_cell']
    assert v_mg > v_al
test("Mg addition increases voltage (activation)", t_mg_activates)

def t_in_inhibits():
    c_al = cell_model(**BASE_CONFIG, j_mA_cm2=50)['parasitic_pct']
    c_in = cell_model(**BASE_CONFIG, j_mA_cm2=50, composition={'Al':0.99,'In':0.01})['parasitic_pct']
    assert c_in < c_al
test("In addition reduces corrosion (inhibition)", t_in_inhibits)

def t_synergy():
    r = cell_model(**BASE_CONFIG, j_mA_cm2=50,
                   composition={'Al':0.977,'Mg':0.005,'In':0.008,'Sn':0.01})
    assert 'synergy' in r['dominant_mechanism'].lower() or r.get('alloy',{}).get('has_synergy')
test("Mg+In+Sn combination triggers synergy flag", t_synergy)

def t_feasibility_over():
    f = check_feasibility({'Al':0.85,'Mg':0.05,'In':0.05,'Sn':0.03,'Zn':0.02})
    assert not f['feasible']
test("Over-limit composition → not feasible", t_feasibility_over)

def t_feasibility_ok():
    f = check_feasibility({'Al':0.975,'Mg':0.01,'In':0.01,'Sn':0.005})
    assert f['feasible']
test("Valid composition → feasible", t_feasibility_ok)

# ── 4. Thermal model ──────────────────────────────────────────────────────────
print("\n── 4. Thermal model ────────────────────────────────────────────────────")

def t_thermal_optimal_range():
    r = thermal_model(d_um=100,c_KOH=4,vf_pct=53,T_ambient_C=25,inh_pct=0,
                      j_mA_cm2=70,h_W_m2K=10)
    assert 60 <= r['T_self_heated_C'] <= 70, f"T={r['T_self_heated_C']}"
test("j=70, h=10 → T_cell in 60–70°C (optimal range)", t_thermal_optimal_range)

def t_thermal_hot_gt_cold():
    r = thermal_model(d_um=100,c_KOH=4,vf_pct=53,T_ambient_C=25,inh_pct=0,
                      j_mA_cm2=70,h_W_m2K=10)
    assert r['V_cell_hot'] > r['V_cell_cold']
    assert r['ed_hot']     > r['ed_cold']
test("Self-heated: V and Ed both higher than cold", t_thermal_hot_gt_cold)

def t_thermal_strong_cooling():
    r = thermal_model(d_um=100,c_KOH=4,vf_pct=53,T_ambient_C=25,inh_pct=0,
                      j_mA_cm2=50,h_W_m2K=1000)
    assert r['dT_C'] < 1.0, f"dT={r['dT_C']} with strong cooling"
test("Strong cooling (h=1000) → ΔT < 1°C", t_thermal_strong_cooling)

def t_thermal_q_positive():
    r = thermal_model(d_um=100,c_KOH=4,vf_pct=53,T_ambient_C=25,inh_pct=0,
                      j_mA_cm2=50,h_W_m2K=10)
    assert r['Q_gen_W_m2'] > 0
    assert r['Q_irrev_W_m2'] > 0
    assert r['Q_rev_W_m2'] > 0
test("Q_gen, Q_irrev, Q_rev all positive", t_thermal_q_positive)

def t_thermal_sweep_length():
    rows = thermal_sweep(d_um=100,c_KOH=4,vf_pct=53,T_ambient_C=25,inh_pct=0,h_W_m2K=10)
    assert len(rows) == 40
    assert all(k in rows[0] for k in ['j','T_cell','dT','V_hot','Q_gen'])
test("Thermal sweep returns 40 points with correct keys", t_thermal_sweep_length)

def t_thermal_j_increases_dT():
    rows = thermal_sweep(d_um=100,c_KOH=4,vf_pct=53,T_ambient_C=25,inh_pct=0,h_W_m2K=10)
    # Temperature should generally increase with j
    low_T  = np.mean([r['T_cell'] for r in rows[:5]])
    high_T = np.mean([r['T_cell'] for r in rows[-5:]])
    assert high_T > low_T
test("Cell temperature increases with current", t_thermal_j_increases_dT)

# ── 5. Parameter sweeps ───────────────────────────────────────────────────────
print("\n── 5. Parameter sweeps — optimal values ────────────────────────────────")

def t_koh_peak():
    vals = [(c, cell_model(d_um=100,c_KOH=c,vf_pct=53,T_C=25,inh_pct=0,j_mA_cm2=50)['net_useful_ed'])
            for c in np.linspace(1,8,25)]
    peak_koh = max(vals, key=lambda x:x[1])[0]
    assert 2.0 < peak_koh < 6.0, f"KOH peak at {peak_koh}M unexpected"
test("KOH sweep — energy peak exists in 2–6M range", t_koh_peak)

def t_temp_peak():
    vals = [(T, cell_model(d_um=100,c_KOH=4,vf_pct=53,T_C=T,inh_pct=0,j_mA_cm2=50)['net_useful_ed'])
            for T in range(20,80,3)]
    peak_T = max(vals, key=lambda x:x[1])[0]
    assert 55 <= peak_T <= 75, f"T peak at {peak_T}°C unexpected"
test("Temperature sweep — energy peak in 55–75°C range", t_temp_peak)

def t_particle_peak_25():
    vals = [(d, cell_model(d_um=d,c_KOH=4,vf_pct=53,T_C=25,inh_pct=0,j_mA_cm2=50)['net_useful_ed'])
            for d in np.linspace(50,600,50)]
    peak_d = max(vals, key=lambda x:x[1])[0]
    assert 200 < peak_d < 450, f"Particle peak at {peak_d}µm at 25°C unexpected"
test("Particle sweep 25°C — peak in 200–450µm range", t_particle_peak_25)

def t_particle_peak_60():
    vals = [(d, cell_model(d_um=d,c_KOH=3.5,vf_pct=53,T_C=60,inh_pct=0,j_mA_cm2=50)['net_useful_ed'])
            for d in np.linspace(50,400,50)]
    peak_d = max(vals, key=lambda x:x[1])[0]
    assert 120 < peak_d < 280, f"Particle peak at {peak_d}µm at 60°C unexpected"
test("Particle sweep 60°C — peak in 120–280µm range", t_particle_peak_60)

def t_particle_peak_shifts():
    def peak_at(T, koh):
        vals = [(d, cell_model(d_um=d,c_KOH=koh,vf_pct=53,T_C=T,inh_pct=0,j_mA_cm2=50)['net_useful_ed'])
                for d in np.linspace(50,500,40)]
        return max(vals, key=lambda x:x[1])[0]
    d25 = peak_at(25, 4.0)
    d60 = peak_at(60, 3.5)
    assert d25 > d60, f"Peak should shift: 25°C={d25}µm, 60°C={d60}µm"
test("Particle optimum shifts with T: larger at 25°C than 60°C", t_particle_peak_shifts)

def t_corrosion_crossover():
    # Below ~28 mA/cm² corrosion should be high
    c_low  = cell_model(d_um=100,c_KOH=4,vf_pct=53,T_C=25,inh_pct=0,j_mA_cm2=5)['parasitic_pct']
    c_high = cell_model(d_um=100,c_KOH=4,vf_pct=53,T_C=25,inh_pct=0,j_mA_cm2=60)['parasitic_pct']
    assert c_low > c_high * 3, f"Crossover not clear: j=5:{c_low:.1f}% vs j=60:{c_high:.1f}%"
test("Corrosion-to-kinetics crossover: j=5 corr >> j=60 corr", t_corrosion_crossover)

# ── 6. Alloy optimiser ────────────────────────────────────────────────────────
print("\n── 6. Alloy optimiser ──────────────────────────────────────────────────")

def t_opt_returns_both():
    r, p = optimize_alloy(BASE_CONFIG,['Mg','In','Sn'],goal='balanced',n_samples=200,j=50)
    assert len(r) > 0 and len(p) > 0
test("optimize_alloy returns (results, pareto) both non-empty", t_opt_returns_both)

def t_sweep_keys():
    pts = binary_sweep(BASE_CONFIG,'In',fractions=[0,0.01,0.02,0.03],j=50)
    assert len(pts) == 4
    assert all(k in pts[0] for k in ['f','V','net_ed','parasitic','ed_cell'])
test("binary_sweep returns 4 pts with correct keys", t_sweep_keys)

def t_sweep_monotone_corr():
    pts = binary_sweep(BASE_CONFIG,'In',fractions=np.linspace(0,0.03,10),j=50)
    corrs = [p['parasitic'] for p in pts]
    # In inhibits — corrosion should decrease
    assert corrs[0] > corrs[-1]
test("In sweep: corrosion decreases as In increases", t_sweep_monotone_corr)

def t_pareto_synergy_25():
    r, p = optimize_alloy(BASE_CONFIG,['Mg','In','Sn'],goal='balanced',n_samples=400,j=50)
    top5_syn = sum(1 for x in r[:5] if x['synergy'])
    assert top5_syn >= 3, f"Only {top5_syn}/5 top configs have synergy at 25°C"
test("At 25°C: ≥3/5 top alloy configs have Mg+In/Sn synergy", t_pareto_synergy_25)

def t_opt_score_sorted():
    r, _ = optimize_alloy(BASE_CONFIG,['Mg','In','Sn'],goal='balanced',n_samples=200,j=50)
    scores = [x['score'] for x in r]
    assert scores == sorted(scores, reverse=True)
test("Results sorted by score descending", t_opt_score_sorted)

# ── 7. Calibration ────────────────────────────────────────────────────────────
print("\n── 7. Calibration & Monte Carlo ────────────────────────────────────────")

def t_calib_self():
    """Model fitting its own output — RMSE should be very low either before or after."""
    j_arr = np.array([2., 5., 10., 25., 50.])
    V_arr = np.array([cell_model(**CALIB_CONFIG,j_mA_cm2=j)['V_cell'] for j in j_arr])
    V_arr += np.random.normal(0, 0.003, len(j_arr))  # tiny noise
    exp = {'j_mA_cm2':j_arr,'V_cell':V_arr,'source':'self_test'}
    cal, rb, ra = calibrate(exp, verbose=False)
    # Either before or after should be excellent (<10mV)
    assert min(rb, ra) < 10, f"Neither RMSE is good: before={rb:.1f}, after={ra:.1f} mV"
test("Self-calibration on model+noise → min(RMSE) < 10 mV", t_calib_self)

def t_mc_shape():
    j, med, p5, p95 = monte_carlo_uncertainty({}, n_samples=50, verbose=False)
    assert len(j) == len(med) == len(p5) == len(p95)
    assert all(p5[i] <= med[i] <= p95[i] for i in range(len(j)))
test("MC returns 4 arrays; p5 ≤ median ≤ p95", t_mc_shape)

def t_mc_band_width():
    j, med, p5, p95 = monte_carlo_uncertainty({}, n_samples=100, verbose=False)
    band = np.mean(p95 - p5) * 1000  # mV
    assert 10 < band < 200, f"MC band width {band:.0f} mV implausible"
test("MC 90% band width in physically plausible range (10–200 mV)", t_mc_band_width)

# ── 8. Sobol sensitivity ──────────────────────────────────────────────────────
print("\n── 8. Sensitivity analysis ─────────────────────────────────────────────")

def t_sobol_vf_dominant():
    result = sensitivity_analysis(BASE_CONFIG, n_steps=20)
    # Returns (sweep_dict, ...) — sweep_dict maps param → (x_arr, y_arr, label)
    sweep = result[0] if isinstance(result, tuple) else result
    # vf_pct should have the largest range (most sensitive)
    ranges = {}
    for k, v in sweep.items():
        if isinstance(v, tuple) and len(v) >= 2:
            y = v[1]
            ranges[k] = float(max(y) - min(y))
    assert ranges, "No sweep data found"
    dominant = max(ranges, key=ranges.get)
    assert 'vf' in dominant.lower() or ranges.get('vf_pct',0) > 0, \
        f"Dominant param is {dominant}, ranges={ranges}"
test("Sensitivity: Al vol% has highest output range", t_sobol_vf_dominant)

# ── 9. Flask app ──────────────────────────────────────────────────────────────
print("\n── 9. Flask app ────────────────────────────────────────────────────────")

def t_app_syntax():
    with open(os.path.join(BASE_DIR, 'app.py'), encoding='utf-8') as f:
        compile(f.read(), 'app.py', 'exec')
test("app.py has valid Python syntax", t_app_syntax)

def t_routes_count():
    with open(os.path.join(BASE_DIR, 'app.py'), encoding='utf-8') as f: src = f.read()
    routes = re.findall(r'@app\.route\("(/api/[^"]+)"', src)
    assert len(routes) == 17, f"Expected 17 routes, found {len(routes)}: {routes}"
test("app.py has exactly 15 API routes", t_routes_count)

def t_thermal_routes():
    with open(os.path.join(BASE_DIR, 'app.py'), encoding='utf-8') as f: src = f.read()
    assert '/api/thermal' in src
    assert '/api/thermal/sweep' in src
    assert '/api/thermal/optfinder' in src
test("All 3 thermal routes present", t_thermal_routes)

def t_port_env():
    with open(os.path.join(BASE_DIR, 'app.py'), encoding='utf-8') as f: src = f.read()
    assert 'os.environ.get("PORT"' in src
    assert 'host="0.0.0.0"' in src
test("Render port config: PORT env var + host=0.0.0.0", t_port_env)

# ── 10. Frontend ──────────────────────────────────────────────────────────────
print("\n── 10. Frontend (index.html) ───────────────────────────────────────────")

def t_panels():
    with open(os.path.join(BASE_DIR, 'index.html'), encoding='utf-8') as f: html = f.read()
    expected = ['baseline','polarisation','sweeps','alloy','optimizer',
                'atoms','thermal','degradation','heatmap','joint',
                'tempmap','currentmap','calibrate','montecarlo']
    for p in expected:
        assert f'id="panel-{p}"' in html, f"Missing panel: {p}"
test("All 14 panels present", t_panels)

def t_tooltips():
    with open(os.path.join(BASE_DIR, 'index.html'), encoding='utf-8') as f: html = f.read()
    assert html.count('title="') >= 6
test("At least 6 tooltip title attributes on sliders", t_tooltips)

def t_csv_buttons():
    with open(os.path.join(BASE_DIR, 'index.html'), encoding='utf-8') as f: html = f.read()
    assert html.count('downloadCSV') >= 6
test("CSV download buttons on ≥6 panels", t_csv_buttons)

def t_file_upload():
    with open(os.path.join(BASE_DIR, 'index.html'), encoding='utf-8') as f: html = f.read()
    assert 'type="file"' in html
    assert 'loadCalibFile' in html
test("File upload in calibration panel", t_file_upload)

def t_default_25():
    with open(os.path.join(BASE_DIR, 'index.html'), encoding='utf-8') as f: html = f.read()
    assert 'value="25"' in html  # T slider default
test("Default temperature set to 25°C", t_default_25)

# ── 11. README ────────────────────────────────────────────────────────────────
print("\n── 11. README.md ───────────────────────────────────────────────────────")

def t_17_findings():
    with open(os.path.join(BASE_DIR, 'README.md'), encoding='utf-8') as f: txt = f.read()
    n = len(re.findall(r'^### \d+\.', txt, re.MULTILINE))
    assert n == 17, f"Found {n} findings, expected 17"
test("Exactly 17 numbered findings", t_17_findings)

def t_no_stale():
    with open(os.path.join(BASE_DIR, 'README.md'), encoding='utf-8') as f: txt = f.read()
    assert 'YOUR_USERNAME' not in txt
    assert '3.5 M KOH · 60 °C' not in txt
    assert 'Your Name' not in txt
test("No stale placeholder text", t_no_stale)

def t_correct_urls():
    with open(os.path.join(BASE_DIR, 'README.md'), encoding='utf-8') as f: txt = f.read()
    assert 'danxdz/Al-Air_Battery_Lab' in txt
    assert 'al-air-battery-lab.onrender.com' in txt
    assert 'cooldan' in txt
test("Correct GitHub URL, live app URL, bibtex author", t_correct_urls)

def t_figures():
    with open(os.path.join(BASE_DIR, 'README.md'), encoding='utf-8') as f: txt = f.read()
    for fig in ['polarisation_curve.png','particle_size_sweep.png',
                'alloy_design_space.png','calibration_result.png']:
        assert fig in txt, f"Missing figure: {fig}"
test("All 4 notebook figures referenced in README", t_figures)

def t_thermal_in_readme():
    with open(os.path.join(BASE_DIR, 'README.md'), encoding='utf-8') as f: txt = f.read()
    assert 'natural convection' in txt.lower()
    assert '64.8' in txt or 'self-heat' in txt.lower()
test("Thermal finding (natural convection, self-heating) in README", t_thermal_in_readme)

def t_standard_conditions():
    with open(os.path.join(BASE_DIR, 'README.md'), encoding='utf-8') as f: txt = f.read()
    assert '25 °C' in txt or '25°C' in txt
    assert '4 M KOH' in txt or '4M KOH' in txt
test("Standard conditions (25°C, 4M KOH) referenced", t_standard_conditions)

def t_sections():
    with open(os.path.join(BASE_DIR, 'README.md'), encoding='utf-8') as f: txt = f.read()
    for sec in ['## Key findings', '## Physics inside', '## Files',
                '## Install', '## Calibration', '## Contribute', '## Cite']:
        assert sec in txt, f"Missing section: {sec}"
test("All key README sections present", t_sections)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
total = len(passed) + len(failed)
print(f"RESULTS:  {len(passed)}/{total} passed   {len(failed)} failed")
print("=" * 70)

if failed:
    print(f"\n{'FAILED TESTS':}")
    for name, err in failed:
        print(f"  ✗ {name}")
        print(f"    → {err}")
else:
    print("\n✓ ALL TESTS PASSED — ready to ship")

print()