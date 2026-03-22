"""
Al-Air Battery Lab — Flask App
==============================
Serves a browser UI that calls al_air_model.py, al_air_alloy.py,
and al_air_calibrate.py directly on every request.
No pre-computed lookup tables — every result is live.

Run:
    python app.py
Then open: http://localhost:5000
"""

import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from al_air_model import (
    cell_model, BASE_CONFIG, ATOMS, PARAMS,
    COMPOSITION_LIMITS, MAX_TOTAL_ADDITIVE,
    check_feasibility, polarisation_curve,
    fit_parameters, compute_rmse,
)
from al_air_alloy  import (binary_sweep, optimize_alloy,
                            optimize_joint, temperature_alloy_map,
                            current_alloy_map)
from al_air_calibrate import (
    calibrate, monte_carlo_uncertainty,
    CALIB_CONFIG, J_POINTS,
)

app = Flask(__name__, static_folder='static')

# ── helpers ───────────────────────────────────────────────────────────────────

def npsafe(obj):
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):
        return {k: npsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [npsafe(i) for i in obj]
    return obj

def ok(data):
    return jsonify({"ok": True,  "data": npsafe(data)})

def err(msg, code=400):
    return jsonify({"ok": False, "error": str(msg)}), code

# ── API: baseline ─────────────────────────────────────────────────────────────

@app.route("/api/baseline", methods=["POST"])
def api_baseline():
    """Run cell_model at given conditions."""
    try:
        body = request.get_json(force=True) or {}
        d    = float(body.get("d_um",   BASE_CONFIG["d_um"]))
        c    = float(body.get("c_KOH",  BASE_CONFIG["c_KOH"]))
        vf   = float(body.get("vf_pct", BASE_CONFIG["vf_pct"]))
        T    = float(body.get("T_C",    BASE_CONFIG["T_C"]))
        inh  = float(body.get("inh_pct",BASE_CONFIG["inh_pct"]))
        j    = float(body.get("j",      50))

        r = cell_model(d, c, vf, T, inh, j)
        # strip matplotlib objects and alloy dict for JSON
        out = {k: v for k, v in r.items()
               if k not in ("alloy",) and not callable(v)}
        out["dominant_mechanism"] = r.get("dominant_mechanism","—")
        out["feasibility"] = r.get("feasibility", {})
        return ok(out)
    except Exception as e:
        return err(traceback.format_exc())


# ── API: polarisation curve ───────────────────────────────────────────────────

@app.route("/api/polarisation", methods=["POST"])
def api_polarisation():
    """Return full polarisation curve for given conditions."""
    try:
        body = request.get_json(force=True) or {}
        cfg  = {
            "d_um":    float(body.get("d_um",   BASE_CONFIG["d_um"])),
            "c_KOH":   float(body.get("c_KOH",  BASE_CONFIG["c_KOH"])),
            "vf_pct":  float(body.get("vf_pct", BASE_CONFIG["vf_pct"])),
            "T_C":     float(body.get("T_C",    BASE_CONFIG["T_C"])),
            "inh_pct": float(body.get("inh_pct",BASE_CONFIG["inh_pct"])),
        }
        comp = body.get("composition", None)
        if comp:
            comp = {k: float(v) for k, v in comp.items()}

        j_vals = np.linspace(1, 70, 50)
        pts = []
        for j in j_vals:
            try:
                r = cell_model(**cfg, j_mA_cm2=float(j), composition=comp)
                pts.append({
                    "j":       round(float(j),    2),
                    "V":       round(r["V_cell"],  4),
                    "eta_bv":  round((r["eta_anode_V"]+r["eta_cathode_V"])*1000, 1),
                    "eta_ohm": round(r["eta_ohmic_V"]*1000, 1),
                    "eta_mt":  round(r["eta_mass_trans_V"]*1000, 1),
                    "ed":      round(r["ed_Wh_kg_paste"], 1),
                    "pd":      round(r["pd_W_kg_paste"],  1),
                })
            except Exception:
                pass
        return ok(pts)
    except Exception as e:
        return err(traceback.format_exc())


# ── API: parameter sweep ──────────────────────────────────────────────────────

@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    """Sweep one parameter while holding others fixed."""
    try:
        body  = request.get_json(force=True) or {}
        param = body.get("param", "c_KOH")
        lo    = float(body.get("lo", 1.5))
        hi    = float(body.get("hi", 8.0))
        n     = int(body.get("n",   30))
        j     = float(body.get("j", 50))
        base  = {
            "d_um":    float(body.get("d_um",   BASE_CONFIG["d_um"])),
            "c_KOH":   float(body.get("c_KOH",  BASE_CONFIG["c_KOH"])),
            "vf_pct":  float(body.get("vf_pct", BASE_CONFIG["vf_pct"])),
            "T_C":     float(body.get("T_C",    BASE_CONFIG["T_C"])),
            "inh_pct": float(body.get("inh_pct",BASE_CONFIG["inh_pct"])),
        }

        pts = []
        for val in np.linspace(lo, hi, n):
            cfg = dict(base); cfg[param] = float(val)
            try:
                r = cell_model(**cfg, j_mA_cm2=j)
                pts.append({
                    "x":    round(float(val), 3),
                    "V":    round(r["V_cell"],            4),
                    "ed":   round(r["ed_Wh_kg_paste"],    1),
                    "pd":   round(r["pd_W_kg_paste"],     1),
                    "corr": round(r["parasitic_pct"],     2),
                    "util": round(r["utilisation_pct"],   1),
                })
            except Exception:
                pass
        return ok({"param": param, "points": pts})
    except Exception as e:
        return err(traceback.format_exc())


# ── API: alloy — single composition ─────────────────────────────────────────

@app.route("/api/alloy/eval", methods=["POST"])
def api_alloy_eval():
    """Evaluate a specific alloy composition at given j."""
    try:
        body = request.get_json(force=True) or {}
        comp = {k: float(v) for k, v in body.get("composition", {"Al":1.0}).items()}
        j    = float(body.get("j", 50))
        cfg  = {
            "d_um":    float(body.get("d_um",   BASE_CONFIG["d_um"])),
            "c_KOH":   float(body.get("c_KOH",  BASE_CONFIG["c_KOH"])),
            "vf_pct":  float(body.get("vf_pct", BASE_CONFIG["vf_pct"])),
            "T_C":     float(body.get("T_C",    BASE_CONFIG["T_C"])),
            "inh_pct": 0,  # alloy model handles inhibition internally
        }

        r   = cell_model(**cfg, j_mA_cm2=j, composition=comp)
        ap  = r.get("alloy") or {}
        feas = check_feasibility(comp)

        return ok({
            "V":               round(r["V_cell"],              4),
            "ed_cell":         round(r["ed_Wh_kg_paste"],      1),
            "ed_system":       round(r["ed_Wh_kg_system"],     1),
            "ed_theoretical":  round(r.get("ed_theoretical_paste", 0), 1),
            "net_useful_ed":   round(r["net_useful_ed"],       1),
            "pd":              round(r["pd_W_kg_paste"],       1),
            "corr":            round(r["parasitic_pct"],       2),
            "util":            round(r["utilisation_pct"],     1),
            "dominant_mechanism": r.get("dominant_mechanism","—"),
            "inh_total_pct":   round(r.get("inh_total_pct",0), 1),
            "has_synergy":     bool(ap.get("has_synergy", False)),
            "E0_mix":          round(float(ap.get("E0_mix",-1.662)),4) if ap else -1.662,
            "dOCV_mV":         round(float(ap.get("dOCV",0))*1000, 1) if ap else 0,
            "Ea_kJ":           round(float(ap.get("Ea_mix_kJ",48)),1) if ap else 48,
            "feasible":        feas["feasible"],
            "warnings":        feas["warnings"],
            "total_add_pct":   round(feas["total_add_pct"], 2),
        })
    except Exception as e:
        return err(traceback.format_exc())


# ── API: alloy sweep ──────────────────────────────────────────────────────────

@app.route("/api/alloy/sweep", methods=["POST"])
def api_alloy_sweep():
    """Binary sweep of one additive element 0→max%."""
    try:
        body    = request.get_json(force=True) or {}
        element = body.get("element", "In")
        max_f   = float(body.get("max_pct", 3.0)) / 100
        n       = int(body.get("n", 20))
        j       = float(body.get("j", 50))
        cfg = {
            "d_um":    float(body.get("d_um",   BASE_CONFIG["d_um"])),
            "c_KOH":   float(body.get("c_KOH",  BASE_CONFIG["c_KOH"])),
            "vf_pct":  float(body.get("vf_pct", BASE_CONFIG["vf_pct"])),
            "T_C":     float(body.get("T_C",    BASE_CONFIG["T_C"])),
            "inh_pct": 0,
        }
        fracs = np.linspace(0, max_f, n)
        data  = binary_sweep(cfg, element, fractions=fracs, j=j)
        return ok(data)
    except Exception as e:
        return err(traceback.format_exc())


# ── API: alloy optimize ───────────────────────────────────────────────────────

@app.route("/api/alloy/optimize", methods=["POST"])
def api_alloy_optimize():
    """Run Pareto optimizer — may take 10-30s."""
    try:
        body      = request.get_json(force=True) or {}
        additives = body.get("additives", ["Mg","In","Sn","Zn","Ga"])
        goal      = body.get("goal", "balanced")
        n_samples = int(body.get("n_samples", 2000))
        j         = float(body.get("j", 50))
        cfg = {
            "d_um":    float(body.get("d_um",   BASE_CONFIG["d_um"])),
            "c_KOH":   float(body.get("c_KOH",  BASE_CONFIG["c_KOH"])),
            "vf_pct":  float(body.get("vf_pct", BASE_CONFIG["vf_pct"])),
            "T_C":     float(body.get("T_C",    BASE_CONFIG["T_C"])),
            "inh_pct": 0,
        }

        results, pareto = optimize_alloy(cfg, additives,
                                         goal=goal, n_samples=n_samples, j=j)

        def fmt(r):
            return {
                "score":   round(float(r["score"]),   1),
                "net_ed":  round(float(r["net_ed"]),  1),
                "corr":    round(float(r["corr"]),    2),
                "power":   round(float(r["power"]),   1),
                "voltage": round(float(r["voltage"]), 4),
                "synergy": bool(r["synergy"]),
                "comp":    {k: round(float(v)*100, 2) for k,v in r["comp"].items()},
            }

        return ok({
            "n_valid":  len(results),
            "n_pareto": len(pareto),
            "all":      [fmt(r) for r in results[:500]],   # cap for JSON size
            "pareto":   [fmt(r) for r in pareto],
            "top10":    [fmt(r) for r in results[:10]],
        })
    except Exception as e:
        return err(traceback.format_exc())


# ── API: calibrate ────────────────────────────────────────────────────────────

@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    """Run physics-region calibration on user-supplied data."""
    try:
        body = request.get_json(force=True) or {}
        pts  = body.get("points", [])   # [{j:..., V:...}, ...]
        if len(pts) < 3:
            return err("Need at least 3 data points")

        exp = {
            "j_mA_cm2": np.array([float(p["j"]) for p in pts]),
            "V_cell":   np.array([float(p["V"]) for p in pts]),
            "source":   "browser upload",
        }

        cal, rmse_b, rmse_a = calibrate(exp, verbose=False)

        # Model curve before and after
        j_fine = np.linspace(0.5, 65, 60)
        before, after = [], []
        for j in j_fine:
            try:
                before.append({"j": round(j,1),
                                "V": round(cell_model(**CALIB_CONFIG, j_mA_cm2=j)["V_cell"],4)})
                after.append( {"j": round(j,1),
                                "V": round(cell_model(**CALIB_CONFIG, j_mA_cm2=j,
                                                      params_override=cal)["V_cell"],4)})
            except Exception:
                pass

        return ok({
            "rmse_before_mV": round(rmse_b, 1),
            "rmse_after_mV":  round(rmse_a, 1),
            "target_mV":      30.0,
            "achieved":       rmse_a < 30.0,
            "params": {
                "E_ocv_ref":     round(cal.get("E_ocv_ref", PARAMS["E_ocv_ref"]), 5),
                "i0_Al_ref":     cal.get("i0_Al_ref",  PARAMS["i0_Al_ref"]),
                "i0_O2_ref":     cal.get("i0_O2_ref",  PARAMS["i0_O2_ref"]),
                "L_eff_m":       cal.get("L_eff_m",    PARAMS["L_eff_m"]),
                "L_diff_factor": cal.get("L_diff_factor", PARAMS["L_diff_factor"]),
            },
            "curve_before": before,
            "curve_after":  after,
        })
    except Exception as e:
        return err(traceback.format_exc())


# ── API: Monte Carlo ──────────────────────────────────────────────────────────

@app.route("/api/montecarlo", methods=["POST"])
def api_montecarlo():
    """Run Monte Carlo uncertainty (n=200 for speed)."""
    try:
        body   = request.get_json(force=True) or {}
        cal_params = body.get("params", {})
        n      = int(body.get("n", 200))

        j_arr, V_med, V_p5, V_p95 = monte_carlo_uncertainty(
            cal_params, n_samples=n, verbose=False)

        return ok({
            "j":     j_arr.tolist(),
            "med":   [round(float(v),4) for v in V_med],
            "p5":    [round(float(v),4) for v in V_p5],
            "p95":   [round(float(v),4) for v in V_p95],
            "band_mV": round(float(np.mean(V_p95-V_p5))*1000/2, 0),
        })
    except Exception as e:
        return err(traceback.format_exc())


# ── API: atom database ────────────────────────────────────────────────────────

@app.route("/api/atoms", methods=["GET"])
def api_atoms():
    F = 96485
    rows = []
    for sym, a in ATOMS.items():
        rows.append({
            "sym":      sym,
            "E0":       a["E0"],
            "n":        a["n"],
            "M":        a["M"],
            "ed_theory":round((a["n"]*F/a["M"])/3600),
            "Ea_kJ":    a["Ea_corr"],
            "inh_pct":  round(a["inh_eff"]*100),
            "role":     a["role"],
            "note":     a["note"],
        })
    return ok(rows)


# ── API: joint optimisation ───────────────────────────────────────────────────

@app.route("/api/alloy/joint", methods=["POST"])
def api_joint():
    """Joint optimisation over paste conditions + alloy — 7D search."""
    try:
        body      = request.get_json(force=True) or {}
        n_samples = int(body.get("n_samples", 2000))
        j         = float(body.get("j", 50))

        results = optimize_joint(n_samples=n_samples, j=j, verbose=True)

        def fmt(r):
            return {
                "score":   round(float(r["score"]),   1),
                "net_ed":  round(float(r["net_ed"]),  1),
                "corr":    round(float(r["corr"]),    2),
                "voltage": round(float(r["voltage"]), 4),
                "power":   round(float(r["power"]),   1),
                "synergy": bool(r["synergy"]),
                "d_um":    round(float(r["d_um"]),    1),
                "c_KOH":   round(float(r["c_KOH"]),  2),
                "T_C":     round(float(r["T_C"]),     1),
                "comp":    {k: round(float(v)*100, 2) for k, v in r["comp"].items()},
            }

        return ok({
            "n_valid": len(results),
            "top20":   [fmt(r) for r in results[:20]],
            "all":     [fmt(r) for r in results[:300]],
        })
    except Exception as e:
        return err(traceback.format_exc())


# ── API: temperature alloy map ─────────────────────────────────────────────────

@app.route("/api/alloy/tempmap", methods=["POST"])
def api_tempmap():
    """How optimal alloy shifts with temperature."""
    try:
        body   = request.get_json(force=True) or {}
        temps  = body.get("temperatures", [25, 40, 55, 60, 70, 75])
        n      = int(body.get("n_samples", 600))

        res_by_T = temperature_alloy_map(temperatures=temps, n_samples=n,
                                          verbose=True)
        out = []
        for T, r in res_by_T.items():
            out.append({
                "T":       T,
                "net_ed":  round(float(r["net_ed"]),  1),
                "corr":    round(float(r["corr"]),    2),
                "voltage": round(float(r["voltage"]), 4),
                "synergy": bool(r["synergy"]),
                "comp":    {k: round(float(v)*100, 2) for k, v in r["comp"].items()},
            })
        return ok(out)
    except Exception as e:
        return err(traceback.format_exc())


# ── API: current alloy map ─────────────────────────────────────────────────────

@app.route("/api/alloy/currentmap", methods=["POST"])
def api_currentmap():
    """How optimal alloy shifts with operating current density."""
    try:
        body     = request.get_json(force=True) or {}
        currents = body.get("currents", [5, 10, 20, 50, 70])
        n        = int(body.get("n_samples", 600))

        res_by_j = current_alloy_map(currents=currents, n_samples=n, verbose=True)
        out = []
        for j, r in res_by_j.items():
            total_add = sum(v for k, v in r["comp"].items() if k != "Al") * 100
            out.append({
                "j":        j,
                "net_ed":   round(float(r["net_ed"]),  1),
                "corr":     round(float(r["corr"]),    2),
                "voltage":  round(float(r["voltage"]), 4),
                "synergy":  bool(r["synergy"]),
                "total_add_pct": round(float(total_add), 2),
                "comp":     {k: round(float(v)*100, 2) for k, v in r["comp"].items()},
            })
        return ok(out)
    except Exception as e:
        return err(traceback.format_exc())


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 54)
    print("  Al-Air Battery Lab")
    print(f"  Open: http://localhost:{port}")
    print("=" * 54)
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
