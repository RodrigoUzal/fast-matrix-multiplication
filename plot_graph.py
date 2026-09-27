"""Figures and summary numbers for the empirical study (report, Section 2).
 
(AI-assisted) Written with Claude (Anthropic), following the design we agreed on.
 
Reads the CSV written by strassen.exe and, for each objective of Section 2.1, saves its figures
to the figures/ folder as PDF and prints the numbers the report text quotes. Run it from the
project folder, after the measurements have finished:
 
    python plot_graph.py                     # results.csv, plus results_large.csv if it exists
    python plot_graph.py results_quick.csv   # any other CSV(s), e.g. to test the script
 
CSV columns (see main.cpp): algorithm, n, cutoff, run, seconds, max_diff, ref_max.
"""

import sys
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")  # draw straigth to files so no plot windows open
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
 

# Every number the analysis depends on is defined here.

MAIN_CSV = Path("results.csv")
LARGE_CSV = Path("results_large.csv")
FIG_DIR = Path("figures")
 
FIXED_CUTOFF = 64                # cut-off used for O1, O3 and O4 (FIXED_CUTOFF in main.cpp)
SWEEP_SIZES = [512, 1024, 2048]  # sizes at which O2 sweeps te cut-off (SWEEP_SIZES in main.cpp)
FIT_MIN_N = 512                  # O3 fits the exponents over n >= this (report, Section 2.2)
 
H1_RANGE = (256, 1024)           # H1: predicted crossover
H2_RANGE = (32, 128)             # H2: predicted optimal cut-off
H4_BOUND = 1e-12                 # H4: E(n) stays below this
NEAR_BEST = 0.05                 # O2: cut-offs within 5% of the best count as equally good
 
# One colour per algorithm, identical in every figure (checked for colour-blind separation).
COLOR = {"standard": "#2a78d6", "strassen": "#eb6834"}
LABEL = {"standard": "standard", "strassen": f"Strassen (cut-off {FIXED_CUTOFF})"}
# Figures 3 and 6 show Strassen at three sizes: shades of Strassen's orange, light to dark as n
# grows, plus a different marker per size so they can be told apart without colour.
SIZE_STYLE = {512: ("#f57e4d", "o"), 1024: ("#c54a05", "s"), 2048: ("#921300", "^")}
INK_2 = "#52514e"   # axis and tick labels
MUTED = "#898781"   # reference lines and their labels
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"    # shading for the range a hypothesis predicts
 
FIG_SIZE = (6.3, 3.0)  # inches; 6.3 in is the text width of A4 with 2.5 cm margins
 
plt.rcParams.update({
    "font.size": 9,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,        # embed real fonts, so text in the PDF stays sharp and selectable
    "savefig.bbox": "tight",
})
 
# Loading and shared helpers

def load_data(paths):
    """Read one or more CSVs into a single table and add E(n) for every row."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    expected = {"algorithm", "n", "cutoff", "run", "seconds", "max_diff", "ref_max"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"Missing columns {sorted(missing)}: is this a CSV written by strassen.exe?")
    df["rel_err"] = df["max_diff"] / df["ref_max"]   # E(n), equation (1) of the report
    return df
 
def fixed_cutoff_rows(df):
    """Rows used by O1, O3 and O4: every standard row plus Strassen at the fixed cut-off."""
    return df[(df.algorithm == "standard") | (df.cutoff == FIXED_CUTOFF)]

def paired(df):
    """Every Strassen measurement next to the standard one of the same size and run.
 
    Both multiplied the same matrices moments apart, so dividing one time by the other cancels
    most of the run-to-run noise (clock speed, background load)."""
    std = df[df.algorithm == "standard"][["n", "run", "seconds"]].rename(columns={"seconds": "t_std"})
    stra = df[df.algorithm == "strassen"][["n", "cutoff", "run", "seconds"]].rename(
        columns={"seconds": "t_str"})
    pairs = stra.merge(std, on=["n", "run"])
    pairs["speedup"] = pairs.t_std / pairs.t_str   # > 1 means Strassen is faster (O1)
    pairs["ratio"] = pairs.t_str / pairs.t_std     # < 1 means Strassen is faster (O2)
    return pairs
 
 
def med_range(df, by, column):
    """Median over the repetitions, and the fastest and slowest of them (for the error bars)."""
    return df.groupby(by)[column].agg(median="median", low="min", high="max").reset_index()
 
 
def fit_exponent(n, t):
    """Least-squares slope of log t against log n, and its standard error.
 
    The base of the logarithm does not matter: it cancels in the slope. With m sizes the fit
    has m - 2 degrees of freedom, so with only three sizes the standard error is a rough guide."""
    x, y = np.log2(n), np.log2(t)
    slope, intercept = np.polyfit(x, y, 1)
    if len(x) < 3:
        return slope, float("nan")   # two points fit exactly: no error estimate
    residuals = y - (slope * x + intercept)
    se = np.sqrt((residuals @ residuals) / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean())))
    return slope, se
 
 
def plot_med_range(ax, x, s, color, marker="o", label=None):
    """Median as a line with markers; the fastest-to-slowest range as error bars."""
    yerr = [s["median"] - s["low"], s["high"] - s["median"]]
    ax.errorbar(np.asarray(x), s["median"].to_numpy(), yerr=np.asarray(yerr), color=color,
                marker=marker, capsize=2, elinewidth=0.8, label=label)
 
 
def log2_x(ax, ticks, labels=None):
    """Logarithmic x axis in base 2, with a labelled tick at each value given."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if labels is not None else [str(int(t)) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
 
 
def shade(ax, low, high, text):
    """Shade the x range a hypothesis predicted, labelled at the top."""
    ax.axvspan(low, high, color=BAND, lw=0, zorder=0)
    ax.text(np.sqrt(low * high), 0.97, text, transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=MUTED, fontsize=8)
 
 
def ref_hline(ax, y, text, in_legend=False):
    """Horizontal reference line (a theoretical value or a threshold). Its label goes at the
    right-hand end, or into the legend when data points would collide with it there."""
    ax.axhline(y, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1, label=text if in_legend else None)
    if not in_legend:
        ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=8)
 
 
def save(fig, name):
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}")
 
 
def header(title):
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))
 
# Printed text sticks to plain ASCII (-, ~, >=): the Windows console can fail on symbols such as
# the approximately-equal sign when the output is redirected to a file.
 
# ---------------------------------------------------------------------------------------------
# Sanity checks (no figure)
# ---------------------------------------------------------------------------------------------
 
def check_data(df):
    header("Checks")
    counts = df.groupby(["algorithm", "n", "cutoff"]).size()
    print(f"{len(df)} rows, {len(counts)} configurations, sizes {sorted(df.n.unique().tolist())}")
    print(f"repetitions per configuration: {counts.min()} to {counts.max()}")
    print(f"duplicated measurements: {df.duplicated(['algorithm', 'n', 'cutoff', 'run']).sum()}")
 
    # Control runs: with cut-off >= n, strassen_mul calls standard_mul directly, so the paired
    # ratio should be close to 1. A value far from 1 means the timings are unreliable.
    pairs = paired(df)
    control = pairs[pairs.cutoff >= pairs.n]
    for n, g in control.groupby("n"):
        print(f"control n = {n}: median t(cut-off >= n) / t_std = {g.ratio.median():.3f} (should be ~1)")
 
# O1: crossover

def o1_crossover(df):
    header(f"O1  Crossover (cut-off {FIXED_CUTOFF})")
    s = med_range(fixed_cutoff_rows(df), ["algorithm", "n"], "seconds")
    sizes = sorted(s.n.unique())
 
    # Figure 1: running time of both algorithms.
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for algo in ["standard", "strassen"]:
        g = s[s.algorithm == algo]
        plot_med_range(ax, g.n, g, COLOR[algo], label=LABEL[algo])
    ax.set_yscale("log")
    log2_x(ax, sizes)
    ax.set_xlabel("matrix size $n$")
    ax.set_ylabel("time (s)")
    ax.legend()
    save(fig, "o1_time_vs_n")
 
    # Figure 2: speedup, computed within each run and then summarised over runs.
    pairs = paired(df)
    sp = med_range(pairs[pairs.cutoff == FIXED_CUTOFF], "n", "speedup")
    # A size is a clear result only when every run agrees: Strassen faster in all of them, or
    # slower in all of them. Otherwise the difference is within the run-to-run noise ("mixed").
    sp["every run"] = np.where(sp.low > 1, "faster", np.where(sp.high < 1, "slower", "mixed"))
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    shade(ax, *H1_RANGE, "H1 prediction")
    ref_hline(ax, 1.0, "equal time")
    plot_med_range(ax, sp.n, sp, COLOR["strassen"])
    log2_x(ax, sizes)
    ax.set_xlabel("matrix size $n$")
    ax.set_ylabel(r"speedup $t_{\mathrm{std}}\,/\,t_{\mathrm{str}}$")
    save(fig, "o1_speedup")
 
    # Table: median times and speedup per size.
    t = s.pivot(index="n", columns="algorithm", values="median")
    table = pd.DataFrame({"t_std (s)": t["standard"], "t_str (s)": t["strassen"]})
    by_n = sp.set_index("n")
    table["speedup"] = by_n["median"]
    table["speedup min"] = by_n["low"]
    table["speedup max"] = by_n["high"]
    table["Strassen, every run"] = by_n["every run"]
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))
 
    # Crossover. Sizes with n <= cut-off are excluded: there Strassen does not recurse, so both
    # algorithms run the same code and any difference is noise.
    rec = sp[sp.n > FIXED_CUTOFF].sort_values("n").reset_index(drop=True)
    ns, v = rec.n.to_numpy(), rec["median"].to_numpy()
    wins = (rec["every run"] == "faster").to_numpy()
    if len(ns) == 0 or not wins[-1]:
        print("No clear crossover: at the largest size, Strassen is not faster in every run.")
    else:
        # The smallest size from which Strassen is faster in every run at every larger size.
        i = len(ns) - 1
        while i > 0 and wins[i - 1]:
            i -= 1
        if i == 0:
            print(f"Strassen is faster in every run from n = {ns[0]}, the smallest size where it"
                  " recurses.")
        else:
            print(f"Strassen is faster in every run from n = {ns[i]}; at n = {ns[i - 1]} the result"
                  f" is {rec['every run'][i - 1]} (speedup {rec.low[i - 1]:.3f} to {rec.high[i - 1]:.3f}).")
    # Where the median speedup crosses 1, interpolated linearly in log2 n.
    crossings = [j for j in range(1, len(v)) if v[j - 1] <= 1 < v[j]]
    for j in crossings:
        x0, x1 = np.log2(ns[j - 1]), np.log2(ns[j])
        n_star = 2 ** (x0 + (1 - v[j - 1]) / (v[j] - v[j - 1]) * (x1 - x0))
        print(f"The median speedup crosses 1 between n = {ns[j - 1]} and n = {ns[j]};"
              f" interpolated n ~ {n_star:.0f}.")
    if len(v) and not crossings and v[0] > 1:
        print(f"The median speedup is already above 1 at n = {ns[0]}.")
    print(f"H1 predicted a crossover between n = {H1_RANGE[0]} and n = {H1_RANGE[1]}.")
 
# O2: optimal cut-off

def o2_cutoff(df):
    header("O2  Cut-off")
    pairs = paired(df)
    sweep = pairs[pairs.n.isin(SWEEP_SIZES)]
    if sweep.empty:
        print("No cut-off sweep in this data.")
        return
    rel = med_range(sweep, ["n", "cutoff"], "ratio")
 
    # Figure 3: Strassen's time relative to standard at the same n. Dividing by standard's time
    # puts all sizes on one axis, and the control run (cut-off = n) lands at exactly 1.
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    shade(ax, *H2_RANGE, "H2 prediction")
    ref_hline(ax, 1.0, "standard algorithm", in_legend=True)
    for n, g in rel.groupby("n"):
        color, marker = SIZE_STYLE.get(n, (COLOR["strassen"], "o"))
        plot_med_range(ax, g.cutoff, g, color, marker, label=f"$n = {n}$")
    log2_x(ax, sorted(rel.cutoff.unique()))
    ax.set_xlabel("cut-off $n_0$")
    ax.set_ylabel(r"relative time $t_{\mathrm{str}}\,/\,t_{\mathrm{std}}$")
    # Legend above the plot: inside it, it would cover data at some cut-off or other.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(SWEEP_SIZES) + 1)
    save(fig, "o2_cutoff_sweep")
 
    secs = sweep.groupby(["cutoff", "n"]).t_str.median().unstack("n")
    print("Median Strassen time (s), cut-off in rows, n in columns:")
    print(secs.to_string(float_format=lambda v: f"{v:.4f}", na_rep="-"))
    for n, g in rel.groupby("n"):
        best = g.loc[g["median"].idxmin()]
        near = g[g["median"] <= best["median"] * (1 + NEAR_BEST)].cutoff.astype(int).tolist()
        print(f"n = {n}: best cut-off {int(best.cutoff)} (t_str/t_std = {best['median']:.3f});"
              f" within {NEAR_BEST:.0%} of the best: {near}")
    print(f"H2 predicted a minimum between n0 = {H2_RANGE[0]} and n0 = {H2_RANGE[1]},"
          " roughly constant in n.")
 
# O3: growth exponents

def o3_exponents(df):
    header("O3  Growth exponents")
    s = med_range(fixed_cutoff_rows(df), ["algorithm", "n"], "seconds")
 
    fitted = {}
    for algo in ["standard", "strassen"]:
        g = s[(s.algorithm == algo) & (s.n >= FIT_MIN_N)]
        if len(g) < 2:
            print(f"{algo}: fewer than two sizes with n >= {FIT_MIN_N}, no fit.")
            continue
        slope, se = fit_exponent(g.n.to_numpy(), g["median"].to_numpy())
        fitted[algo] = (slope, se)
        print(f"{algo:9s} fitted exponent {slope:.3f} +/- {se:.3f} (standard error),"
              f" over n = {g.n.tolist()}")
    if len(fitted) == 2:
        (a, sa), (b, sb) = fitted["standard"], fitted["strassen"]
        print(f"difference (standard - Strassen) {a - b:.3f} +/- {np.hypot(sa, sb):.3f}")
    print("H3 predicted ~3.0 (standard) and ~2.85 (Strassen), a difference of at least 0.1;"
          " theory: 3 and log2(7) = 2.807.")
 
    # Figure 4: the exponent between each pair of consecutive sizes, log(t2/t1) / log(n2/n1).
    # This shows how the growth rate changes with n, which a single fitted value hides.
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    print("Local exponents between consecutive sizes:")
    for algo in ["standard", "strassen"]:
        g = s[s.algorithm == algo].sort_values("n")
        n, t = g.n.to_numpy(), g["median"].to_numpy()
        local = np.log2(t[1:] / t[:-1]) / np.log2(n[1:] / n[:-1])
        mid = np.sqrt(n[1:] * n[:-1])   # midway between the two sizes on a log axis
        ax.plot(mid, local, marker="o", color=COLOR[algo], label=LABEL[algo])
        print(f"  {algo:9s} " + "  ".join(f"{a}->{b}: {e:.3f}" for a, b, e in zip(n[:-1], n[1:], local)))
    ref_hline(ax, 3.0, "3")
    ref_hline(ax, np.log2(7), r"$\log_2 7 \approx 2.807$")
    # No horizontal gridlines here: the two reference lines are the guides that matter, and a
    # gridline at 2.8 would sit on top of the one at 2.807.
    ax.grid(axis="y", visible=False)
    pairs_n = s.n.sort_values().unique()
    log2_x(ax, np.sqrt(pairs_n[1:] * pairs_n[:-1]),
           [f"{a}→{b}" for a, b in zip(pairs_n[:-1], pairs_n[1:])])
    ax.set_xlabel("between sizes")
    ax.set_ylabel("local exponent")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "o3_local_exponents")
 
# O4: accuracy

def o4_accuracy(df):
    header("O4  Accuracy")
    # At n <= cut-off Strassen calls standard_mul directly, so E is exactly 0 there, which a
    # logarithmic axis cannot show: those sizes are left out.
    rows = df[(df.algorithm == "strassen") & (df.cutoff == FIXED_CUTOFF) & (df.n > FIXED_CUTOFF)]
    e = med_range(rows, "n", "rel_err")
 
    # Figure 5: E(n), with a reference line proportional to n (doubling when n doubles).
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ref_hline(ax, H4_BOUND, r"H4 bound $10^{-12}$")
    n0, e0 = e.n.iloc[0], e["median"].iloc[0]
    ax.plot(e.n, e0 * e.n / n0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.annotate(r"$\propto n$", xy=(e.n.iloc[-1], e0 * e.n.iloc[-1] / n0), xytext=(4, 0),
                textcoords="offset points", va="center", color=MUTED, fontsize=8)
    plot_med_range(ax, e.n, e, COLOR["strassen"])
    ax.set_yscale("log")
    log2_x(ax, e.n.tolist())
    ax.set_xlabel("matrix size $n$")
    ax.set_ylabel("relative discrepancy $E(n)$")
    save(fig, "o4_accuracy")
 
    table = e.set_index("n").rename(columns={"median": "E median", "low": "E min", "high": "E max"})
    table["ratio to previous n"] = table["E median"] / table["E median"].shift(1)
    print(table.to_string(float_format=lambda v: f"{v:.3g}", na_rep="-"))
    if len(e) >= 2:
        slope, se = fit_exponent(e.n.to_numpy(), e["median"].to_numpy())
        print(f"E grows like n^{slope:.2f} (+/- {se:.2f}); H4 predicted roughly doubling per"
              " doubling, i.e. an exponent near 1.")
    worst = rows.rel_err.max()
    print(f"largest E over all runs: {worst:.3g} ({'below' if worst < H4_BOUND else 'NOT below'}"
          f" the H4 bound {H4_BOUND:g})")
 
    # Figure 6 (further discussion): E against the cut-off at the sweep sizes. Each halving of
    # the cut-off adds one level of recursion, and Strassen's error bound grows with the number
    # of levels. Cut-off = n is left out: no recursion, so E is exactly 0 there.
    sweep = df[(df.algorithm == "strassen") & df.n.isin(SWEEP_SIZES) & (df.cutoff < df.n)]
    if sweep.empty:
        return
    es = med_range(sweep, ["n", "cutoff"], "rel_err")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for n, g in es.groupby("n"):
        color, marker = SIZE_STYLE.get(n, (COLOR["strassen"], "o"))
        plot_med_range(ax, g.cutoff, g, color, marker, label=f"$n = {n}$")
    ax.set_yscale("log")
    log2_x(ax, sorted(es.cutoff.unique()))
    ax.set_xlabel("cut-off $n_0$")
    ax.set_ylabel("relative discrepancy $E$")
    ax.legend(loc="upper right")
    save(fig, "o4_accuracy_vs_cutoff")
 
    es["levels"] = np.log2(es.n / es.cutoff).astype(int)   # levels of recursion
    print("Median E by cut-off (rows) and n (columns):")
    print(es.pivot(index="cutoff", columns="n", values="median")
            .to_string(float_format=lambda v: f"{v:.3g}", na_rep="-"))
    print("Median E by levels of recursion (rows) and n (columns):")
    print(es.pivot(index="levels", columns="n", values="median")
            .to_string(float_format=lambda v: f"{v:.3g}", na_rep="-"))
 
 
def main():
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = [MAIN_CSV] + ([LARGE_CSV] if LARGE_CSV.exists() else [])
    for p in paths:
        if not p.exists():
            sys.exit(f"{p} not found: run strassen.exe first, from the project folder.")
    print("Reading " + ", ".join(str(p) for p in paths))
    df = load_data(paths)
    FIG_DIR.mkdir(exist_ok=True)
 
    check_data(df)
    o1_crossover(df)
    o2_cutoff(df)
    o3_exponents(df)
    o4_accuracy(df)
 
 
if __name__ == "__main__":
    main()