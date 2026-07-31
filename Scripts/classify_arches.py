"""
Arch classification pipeline (Step 4 of the methodology).

Reads all arch-run CSVs exported from Grasshopper/Kangaroo2, fits three
candidate geometric models (catenary, parabolic, elliptical) to each arch's
(x, z) profile, picks the best-fitting model via BIC (with an extra
tolerance rule to guard against the elliptical model's known tendency to
win on shallow arcs purely from its extra free parameter), and writes a
summary results table.

USAGE:
    1. Put all your arch_run{N}_{loadtype}_g{val}_k{val}.csv files in one folder.
    2. Edit FOLDER below to point to that folder.
    3. Run:  python classify_arches.py
    4. Look for results_summary.csv in the same folder.

Expected filename format:
    arch_run24.0_0_g-0.1_k10.0.csv
    loadtype code: 0 = uniform, 1 = central_point, 2 = asymmetric

Expected CSV format (as produced by your Grasshopper Python export):
    x,y,z
    0,0,0
    0.416667,0,-0.xxxx
    ...
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ------------------------------------------------------------------
# 1. CONFIG — change this to your actual folder path
# ------------------------------------------------------------------
FOLDER = r"D:\UT_IT\Conferenca\2026\Architecture\Computational_Form-Finding_of_Arches\testImages"

# Loadtype code -> human-readable label
LOADTYPE_MAP = {
    "0": "uniform",
    "1": "central_point",
    "2": "asymmetric",
}

# How much extra BIC "cushion" the elliptical model must clear over the
# next-best model before it's allowed to win. This guards against the
# known shallow-arc degeneracy where ellipse's extra scale parameter lets
# it out-fit catenary/parabola even when they're the "true" generator.
# Tune this based on your own sensitivity checks; start conservative.
ELLIPSE_BIC_MARGIN = 6.0


# ------------------------------------------------------------------
# 2. CANDIDATE MODELS
# ------------------------------------------------------------------
def catenary(x, a, x0, d):
    """y = a*(cosh((x-x0)/a) - 1) + d
    Reparametrized (peak height d instead of raw offset c) to avoid the
    large-cancellation ill-conditioning that occurs for shallow arches,
    where a and c both blow up but nearly cancel."""
    return a * (np.cosh((x - x0) / a) - 1) + d


def parabola(x, A, x0, c):
    """y = A*(x-x0)^2 + c"""
    return A * (x - x0) ** 2 + c


def ellipse_arc(x, a, b, x0, c):
    """Upper half of an ellipse centered at (x0, c): y = c - b*sqrt(1-((x-x0)/a)^2)"""
    val = 1 - ((x - x0) / a) ** 2
    val = np.clip(val, 0, None)  # guard against tiny negative values from float error
    return c - b * np.sqrt(val)


MODELS = {
    "catenary": (catenary, 3),
    "parabolic": (parabola, 3),
    "elliptical": (ellipse_arc, 4),
}


# ------------------------------------------------------------------
# 3. FIT HELPERS
# ------------------------------------------------------------------
def aic(n, rss, k):
    """Akaike Information Criterion for least-squares fits."""
    if rss <= 0:
        rss = 1e-12
    return n * np.log(rss / n) + 2 * k


def bic(n, rss, k):
    """Bayesian Information Criterion — penalizes extra parameters harder
    than AIC as n grows (k * ln(n) vs a flat 2k), which is what we want
    given the ellipse's extra free scale parameter."""
    if rss <= 0:
        rss = 1e-12
    return n * np.log(rss / n) + k * np.log(n)


def fit_model(name, func, k, x, z):
    span = x.max() - x.min()
    depth = max(z.max() - z.min(), 1e-3)
    x0_guess = (x.max() + x.min()) / 2
    n = len(x)

    ss_tot = np.sum((z - z.mean()) ** 2)

    if name == "catenary":
        # Shallow-arch approximation: a ~ span^2/(8*depth) is the textbook
        # relation between a catenary's shape parameter and its rise. For
        # shallow sags this parameter is nearly non-identifiable (catenary
        # -> parabola as a -> infinity), so a single start can diverge onto
        # a flat ridge. Try several starting magnitudes and keep the best.
        a_base = max(span ** 2 / (8 * depth), span)
        candidates = [a_base * m for m in (0.3, 1, 3, 10, 30)]
        best = None
        for a0 in candidates:
            p0 = [a0, x0_guess, z.max()]
            try:
                popt, _ = curve_fit(func, x, z, p0=p0, maxfev=20000)
                pred = func(x, *popt)
                rss = np.sum((z - pred) ** 2)
                if best is None or rss < best[0]:
                    best = (rss, popt)
            except Exception:
                continue
        if best is None:
            return {"params": None, "rss": np.inf, "r2": float("nan"),
                     "aic": np.inf, "bic": np.inf}
        rss, popt = best
        r2 = 1 - rss / ss_tot if ss_tot > 0 else float("nan")
        return {"params": popt, "rss": rss, "r2": r2,
                 "aic": aic(n, rss, k), "bic": bic(n, rss, k)}

    elif name == "parabolic":
        p0 = [-depth / (span / 2) ** 2, x0_guess, z.max()]
    else:  # elliptical
        p0 = [span / 2, depth, x0_guess, z.max()]

    try:
        popt, _ = curve_fit(func, x, z, p0=p0, maxfev=20000)
        pred = func(x, *popt)
        rss = np.sum((z - pred) ** 2)
        r2 = 1 - rss / ss_tot if ss_tot > 0 else float("nan")
        return {
            "params": popt,
            "rss": rss,
            "r2": r2,
            "aic": aic(n, rss, k),
            "bic": bic(n, rss, k),
        }
    except Exception as e:
        return {"params": None, "rss": np.inf, "r2": float("nan"),
                 "aic": np.inf, "bic": np.inf, "error": str(e)}


def parse_filename(filename):
    """Pull N, loadtype, g, k out of a filename like
    arch_run24.0_0_g-0.1_k10.0.csv
    Returns (N, loadtype_code, loadtype_label, g_val, k_val).
    Any field that fails to parse comes back as None.
    """
    m = re.search(
        r"arch_run(?P<N>-?\d+\.?\d*)_(?P<loadtype>\d+)_g(?P<g>-?\d+\.?\d*)_k(?P<k>-?\d+\.?\d*)",
        filename,
    )
    if not m:
        return None, None, None, None, None

    N = float(m.group("N"))
    loadtype_code = m.group("loadtype")
    loadtype_label = LOADTYPE_MAP.get(loadtype_code, f"unknown({loadtype_code})")
    g_val = float(m.group("g"))
    k_val = float(m.group("k"))
    return N, loadtype_code, loadtype_label, g_val, k_val


def pick_best_model(fits):
    """Select best-fitting model by BIC, with an extra guard: the
    elliptical model only wins if its BIC beats the best non-elliptical
    model by more than ELLIPSE_BIC_MARGIN. This exists because ellipse's
    extra free scale parameter lets it out-fit catenary/parabola on
    shallow arcs even when it isn't the true generator (see discussion
    in chat / paper §6.3 limitations).
    """
    sorted_names = sorted(fits, key=lambda n: fits[n]["bic"])
    best_name = sorted_names[0]
    best_bic = fits[best_name]["bic"]
    second_name = sorted_names[1]
    second_bic = fits[second_name]["bic"]

    # Decisive-ness check (conventional rule of thumb: delta < 2 is
    # "not worth more than a bare mention")
    decisive = (second_bic - best_bic) >= 2

    # Ellipse guard: only let ellipse win if it clears the tolerance
    # margin over the best non-elliptical model.
    non_ellipse_best = min(
        (n for n in fits if n != "elliptical"),
        key=lambda n: fits[n]["bic"],
    )
    if best_name == "elliptical":
        margin = fits["elliptical"]["bic"] - fits[non_ellipse_best]["bic"]
        # margin is negative if ellipse's bic is lower (better); we want
        # ellipse to be at least ELLIPSE_BIC_MARGIN points better before
        # trusting it over the guard.
        if -margin < ELLIPSE_BIC_MARGIN:
            best_name = non_ellipse_best
            # Recompute decisiveness against whichever is now second
            remaining = sorted(
                (n for n in fits if n != best_name), key=lambda n: fits[n]["bic"]
            )
            decisive = (fits[remaining[0]]["bic"] - fits[best_name]["bic"]) >= 2

    return best_name, decisive


# ------------------------------------------------------------------
# 4. MAIN LOOP
# ------------------------------------------------------------------
def main():
    csv_paths = sorted(glob.glob(os.path.join(FOLDER, "*.csv")))
    csv_paths = [p for p in csv_paths if os.path.basename(p) != "results_summary.csv"]

    if not csv_paths:
        print(f"No CSV files found in {FOLDER}")
        return

    rows = []
    for path in csv_paths:
        fname = os.path.basename(path)
        df = pd.read_csv(path)
        x = df["x"].to_numpy(dtype=float)
        z = df["z"].to_numpy(dtype=float)

        order = np.argsort(x)
        x, z = x[order], z[order]

        N, loadtype_code, loadtype_label, g_val, k_val = parse_filename(fname)

        fits = {name: fit_model(name, func, k, x, z) for name, (func, k) in MODELS.items()}
        best_name, decisive = pick_best_model(fits)

        span = x.max() - x.min()
        depth = max(z.max() - z.min(), 1e-9)
        shallow_flag = (depth / span) < 0.15  # flag known-degenerate shallow arcs

        row = {
            "file": fname,
            "N": N,
            "loadtype_code": loadtype_code,
            "loadtype": loadtype_label,
            "g_val": g_val,
            "k_val": k_val,
            "n_points": len(x),
            "span": round(span, 4),
            "depth": round(depth, 4),
            "shallow_flag": shallow_flag,
            "best_fit": best_name,
            "decisive": decisive,
        }
        for name in MODELS:
            row[f"{name}_r2"] = round(fits[name]["r2"], 5) if not np.isnan(fits[name]["r2"]) else None
            row[f"{name}_aic"] = round(fits[name]["aic"], 3) if np.isfinite(fits[name]["aic"]) else None
            row[f"{name}_bic"] = round(fits[name]["bic"], 3) if np.isfinite(fits[name]["bic"]) else None

        rows.append(row)
        tag = "" if decisive else "  [NOT DECISIVE - models near-indistinguishable]"
        shallow_tag = "  [SHALLOW ARC - classification less reliable]" if shallow_flag else ""
        print(f"{fname}: best fit = {best_name}{tag}{shallow_tag}  "
              f"(catenary R2={row['catenary_r2']}, parabolic R2={row['parabolic_r2']}, "
              f"elliptical R2={row['elliptical_r2']})")

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(FOLDER, "results_summary.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved summary table: {out_path}")


if __name__ == "__main__":
    main()
