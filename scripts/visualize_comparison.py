"""
scripts/visualize_comparison.py
================================
Create an indexed comparison chart: Mount Rainier (MORA) visitation growth
vs. the median of 162 "destination parks" (>500K peak TRV), both indexed
so that 1990 = 100.

Indexed time series let you compare parks with very different absolute visitor
counts on a single y-axis — the question shifts from "who is bigger?" to
"who has grown faster since 1990?"

Both series use the NPS IRMA data package (TRV statistic) for methodological
consistency: the same counting rules apply to MORA and every other park.

Usage:
    python scripts/visualize_comparison.py

Output:
    output/rainier_comparison.png  (1080 × 1080 px, 150 DPI)

─────────────────────────────────────────────────────────────────────────────
Key R → matplotlib/pandas translation notes
─────────────────────────────────────────────────────────────────────────────
  R / tidyverse                      Python / pandas + matplotlib equivalent
  ──────────────────────────────     ─────────────────────────────────────────
  inner_join(mora, nat, by="year")   pd.merge(mora, nat, on="year", how="inner")
  mutate(idx = val / base * 100)     df["idx"] = df["val"] / base * 100
  group_by(unit, year) |>            df.groupby(["unit","year"])["val"].sum()
    summarise(trv=sum(val))
  group_by(year) |>                  df.groupby("year")["trv"].median()
    summarise(med=median(trv))
  geom_ribbon(aes(fill=mora>nat))    ax.fill_between(x, a, b, where=(a>=b))
  geom_line(aes(linetype=series))    ax.plot(..., linestyle="-" / "--")
  annotate("text", ...)              ax.text(x, y, label, ...)
  scale_y_continuous(labels=...)     ax.yaxis.set_major_formatter(FuncFormatter)

  Key design note: we intentionally avoid a dual y-axis (twinx) because
  both series share the same units (index, base = 100), making dual axes
  misleading and unnecessary.  In R, ggplot2's sec_axis() is tempting but
  often abused — the indexed approach sidesteps that trap entirely.
─────────────────────────────────────────────────────────────────────────────
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# Use the non-interactive Agg backend (renders to file; no display needed).
# R equivalent: png(filename) before your plot commands.
matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
RAW_PACKAGE    = PROJECT_ROOT / "data" / "raw" / "nps_visitation_all_parks.csv"
CLEANED_DIR    = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_DIR     = PROJECT_ROOT / "output"
OUTPUT_FILE    = OUTPUT_DIR  / "rainier_comparison.png"
COMPARISON_CSV = CLEANED_DIR / "comparison_indexed.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Design system — Pacific Northwest palette (matches visualize.py)
# ─────────────────────────────────────────────────────────────────────────────
PAL = {
    # Backgrounds
    "bg":          "#F4F1EC",   # warm morning-fog parchment
    # Data ink
    "mora_line":   "#2D6A4F",   # forest green — MORA series
    "nat_line":    "#2C5282",   # deep slate blue — national median
    # Gap fills (echoes the line colors at low opacity)
    "fill_above":  "#52B788",   # spruce green — MORA outperforms
    "fill_below":  "#8BA7C7",   # muted slate — MORA underperforms
    "fill_alpha":   0.22,
    # Structural
    "grid":        "#D5CFC8",
    "spine":       "#C5BFB8",
    "text":        "#1E2832",
    "subtext":     "#5E6E7E",
    "handle":      "#2D6A4F",
    # Event annotations
    "ev_covid":    "#6B48A0",   # muted purple
}

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_YEAR            = 1990
DEST_PARK_THRESHOLD  = 500_000   # peak TRV to qualify as a "destination park"


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions (adapted from fetch_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def _find_column(columns: pd.Index, keywords: list[str]) -> str | None:
    """
    Case-insensitive column search by keyword list.
    R equivalent: dplyr::select(df, contains("keyword"))
    Returns the original column name (original casing) if found, else None.
    """
    # Build a lookup: normalised_name → original_name
    normalised = {
        c.lower().replace(" ", "_").replace("-", "_"): c
        for c in columns
    }
    for kw in keywords:
        if kw in normalised:
            return normalised[kw]
    # Substring fallback
    for kw in keywords:
        for norm, orig in normalised.items():
            if kw in norm:
                return orig
    return None


def parse_number(series: pd.Series) -> pd.Series:
    """
    Strip numeric formatting characters and coerce to float.
    R equivalent: readr::parse_number()
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[,+%\s]", "", regex=True)
        .str.replace("—", "", regex=False)
        .str.replace("N/A", "", regex=False)
        .replace("", float("nan"))
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare() -> pd.DataFrame:
    """
    Load the full NPS IRMA package CSV, derive annual TRV per park,
    identify destination parks, compute the national median series,
    merge with the MORA series, and index both to BASE_YEAR = 100.

    Returns
    -------
    pd.DataFrame with columns:
        year        – calendar year (int)
        mora_trv    – MORA TRV (raw visitor count)
        median_trv  – median TRV across destination parks
        mora_idx    – MORA indexed (BASE_YEAR = 100)
        nat_idx     – national-median indexed (BASE_YEAR = 100)
    """

    if not RAW_PACKAGE.exists():
        print(
            f"\n[error] Full NPS package not found at:\n"
            f"        {RAW_PACKAGE}\n\n"
            f"        Run 'python scripts/fetch_data.py' first to download it.\n"
            f"        (The file is ~70 MB and may take a minute to download.)"
        )
        sys.exit(1)

    # ── 1. Load the full package ──────────────────────────────────────────────
    # The NPS IRMA package is a large CSV with all parks and all statistics.
    # low_memory=False forces pandas to infer column dtypes from the whole file
    # rather than sampling, which prevents mixed-type warnings on big files.
    # R equivalent: readr::read_csv(path) — which also reads the whole file.
    print(f"[read]  Loading {RAW_PACKAGE.name}  (may take a moment)…")
    try:
        df_all = pd.read_csv(RAW_PACKAGE, low_memory=False)
    except UnicodeDecodeError:
        # Government datasets often use Latin-1 encoding (ISO-8859-1).
        # R handles this with: read_csv(path, locale = locale(encoding="latin1"))
        df_all = pd.read_csv(RAW_PACKAGE, encoding="latin1", low_memory=False)

    print(f"[info]  Full dataset — {df_all.shape[0]:,} rows × {df_all.shape[1]} cols")

    # ── 2. Locate key columns (flexible naming across NPS data releases) ───────
    code_col      = _find_column(df_all.columns, ["unitcode", "unit_code", "parkcode"])
    year_col      = _find_column(df_all.columns, ["year"])
    statistic_col = _find_column(df_all.columns, ["statistic"])
    value_col     = _find_column(df_all.columns, ["value"])

    for label, col in [("park code", code_col), ("year", year_col),
                        ("statistic", statistic_col), ("value", value_col)]:
        if col is None:
            print(f"[error] Cannot find '{label}' column. Available: {list(df_all.columns)}")
            sys.exit(1)

    # ── 3. Filter to TRV (Total Recreation Visitors) ─────────────────────────
    # TRV is the NPS standard for recreation visitor counts.
    # Using TRV for all parks ensures MORA and peers share the same methodology.
    #
    # R: df_all |> filter(Statistic == "TRV")
    # pandas: boolean indexing ≈ dplyr::filter()
    trv = df_all[df_all[statistic_col].astype(str).str.upper() == "TRV"].copy()

    if trv.empty:
        # Fallback if TRV code not found in this data release
        print("[warn]  TRV not found — trying TV (Total Visitors) instead")
        trv = df_all[df_all[statistic_col].astype(str).str.upper() == "TV"].copy()

    if trv.empty:
        codes = df_all[statistic_col].dropna().unique().tolist()
        print(f"[error] No TRV or TV rows found. Statistic codes present: {codes}")
        sys.exit(1)

    print(f"[info]  TRV rows: {len(trv):,}")

    # Parse numerics in-place — these columns may be strings with commas
    trv = trv.copy()
    trv[year_col]  = parse_number(trv[year_col])
    trv[value_col] = parse_number(trv[value_col])

    # ── 4. Annual TRV per park ────────────────────────────────────────────────
    # The package stores monthly counts; sum to annual per (park, year).
    #
    # R equivalent:
    #   trv |> group_by(UnitCode, Year) |> summarise(trv = sum(Value, na.rm=TRUE))
    #
    # pd.groupby() returns a GroupBy object — calling .sum() on it is like
    # summarise(sum(.)) in dplyr.  as_index=False keeps the group keys as
    # regular columns rather than as the DataFrame index.
    annual = (
        trv
        .groupby([code_col, year_col], as_index=False)[value_col]
        .sum()
    )
    annual.columns = ["unit_code", "year", "trv"]
    annual = annual[annual["year"] > 1900].copy()
    annual["year"] = annual["year"].astype(int)

    n_parks = annual["unit_code"].nunique()
    n_years = annual["year"].nunique()
    print(f"[info]  Annual TRV computed: {n_parks} parks × {n_years} years")

    # ── 5. Identify destination parks ─────────────────────────────────────────
    # A "destination park" is one that ever exceeded DEST_PARK_THRESHOLD TRV.
    # This excludes heritage corridors, historic sites, and other NPS units
    # that attract far fewer visitors than a classic "park" experience.
    #
    # R: annual |> group_by(unit_code) |>
    #              summarise(peak = max(trv)) |>
    #              filter(peak > 500000) |>
    #              pull(unit_code)
    peak_by_park = annual.groupby("unit_code")["trv"].max()
    dest_codes   = peak_by_park[peak_by_park > DEST_PARK_THRESHOLD].index
    n_dest = len(dest_codes)
    print(f"[info]  Destination parks (peak TRV > {DEST_PARK_THRESHOLD:,}): {n_dest}")

    # ── 6. National median per year ───────────────────────────────────────────
    # We use the *median* rather than the mean because the distribution is
    # heavily right-skewed: Blue Ridge Parkway (~16M TRV) and Great Smoky
    # (~13M) would pull the mean far above a "typical" destination park.
    # The median is robust to outliers — R's median() behaves identically.
    #
    # R: annual |>
    #      filter(unit_code %in% dest_codes) |>
    #      group_by(year) |>
    #      summarise(median_trv = median(trv, na.rm = TRUE))
    dest_annual = annual[annual["unit_code"].isin(dest_codes)].copy()
    nat_median  = (
        dest_annual
        .groupby("year", as_index=False)["trv"]
        .median()
        .rename(columns={"trv": "median_trv"})
    )
    print(f"[info]  National median series: {nat_median['year'].min()}–{nat_median['year'].max()}")

    # ── 7. MORA series ────────────────────────────────────────────────────────
    # Extract just MORA rows and rename for clarity.
    # R: filter(annual, unit_code == "MORA") |> select(year, mora_trv = trv)
    mora = (
        annual[annual["unit_code"] == "MORA"]
        .copy()[["year", "trv"]]
        .rename(columns={"trv": "mora_trv"})
    )
    print(f"[info]  MORA years in package: {mora['year'].min()}–{mora['year'].max()}")
    mora_2023 = mora.loc[mora["year"] == 2023, "mora_trv"]
    if not mora_2023.empty:
        print(f"[info]  MORA 2023 package TRV: {mora_2023.iloc[0]:,.0f}")

    # ── 8. Merge ──────────────────────────────────────────────────────────────
    # pd.merge() is the pandas equivalent of dplyr's join functions:
    #   how="inner"  ≈  inner_join()  — keeps only years in BOTH series
    #   how="left"   ≈  left_join()   — keeps all MORA years
    #   how="outer"  ≈  full_join()   — keeps all years from either series
    #
    # We use inner so both lines share the exact same x-axis extent.
    # R: mora |> inner_join(nat_median, by = "year")
    merged = (
        pd.merge(mora, nat_median, on="year", how="inner")
        .sort_values("year")
        .reset_index(drop=True)
    )
    print(f"[info]  Merged range: {merged['year'].min()}–{merged['year'].max()} "
          f"({len(merged)} years)")

    # ── 9. Index to base year = 100 ───────────────────────────────────────────
    # Index formula: (value / base_year_value) × 100
    # A value of 125 means "25% higher than in BASE_YEAR".
    #
    # R equivalent:
    #   base_mora <- merged$mora_trv[merged$year == BASE_YEAR]
    #   base_nat  <- merged$median_trv[merged$year == BASE_YEAR]
    #   merged <- mutate(merged,
    #                    mora_idx = mora_trv / base_mora * 100,
    #                    nat_idx  = median_trv / base_nat * 100)
    base_row = merged[merged["year"] == BASE_YEAR]
    if base_row.empty:
        print(f"[error] Base year {BASE_YEAR} not found in merged data. "
              f"Years available: {merged['year'].tolist()}")
        sys.exit(1)

    base_mora_val = base_row["mora_trv"].iloc[0]
    base_nat_val  = base_row["median_trv"].iloc[0]

    # R: mutate(mora_idx = mora_trv / base_mora * 100, ...)
    merged["mora_idx"] = merged["mora_trv"] / base_mora_val * 100
    merged["nat_idx"]  = merged["median_trv"] / base_nat_val  * 100

    print(f"\n[index] Base year: {BASE_YEAR} = 100")
    print(f"        MORA {BASE_YEAR} TRV:        {base_mora_val:,.0f}")
    print(f"        Median {BASE_YEAR} TRV:      {base_nat_val:,.0f}")
    last = merged.iloc[-1]
    print(f"        MORA {int(last['year'])} index:       {last['mora_idx']:.1f}")
    print(f"        National {int(last['year'])} index:   {last['nat_idx']:.1f}")
    print(f"        Gap (MORA − median):          {last['mora_idx'] - last['nat_idx']:.1f}")

    # Save the processed comparison data for inspection / reuse
    # R equivalent: readr::write_csv(merged, path)
    merged.to_csv(COMPARISON_CSV, index=False)
    print(f"\n[save]  Comparison data → {COMPARISON_CSV.name}")

    return merged, n_dest


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def _idx_fmt(x: float, _pos) -> str:
    """
    Y-axis tick formatter for index values.
    Returns "100", "125", etc. — no decimal for whole numbers.
    R equivalent: scales::label_number()
    """
    return f"{x:.0f}"


def create_chart(df: pd.DataFrame, n_dest: int) -> None:
    """
    Build and save the Instagram-ready 1080 × 1080 indexed comparison chart.

    Layout (figure-normalised coordinates, 0=bottom, 1=top):
        0.97–1.00   Decorative accent bar
        0.88–0.97   Title text
        0.80–0.88   Subtitle text
        0.80–0.80   Thin rule
        0.12–0.80   Plot axes
        0.02–0.10   Footer
    """

    # ── Style ─────────────────────────────────────────────────────────────────
    # R equivalent: theme_set(theme_minimal()) + theme(text=element_text(...))
    plt.rcParams.update({
        "font.family":            "sans-serif",
        "font.sans-serif":        ["Helvetica Neue", "Helvetica", "Arial",
                                   "Liberation Sans", "DejaVu Sans"],
        "figure.facecolor":       PAL["bg"],
        "axes.facecolor":         PAL["bg"],
        "axes.edgecolor":         PAL["spine"],
        "axes.grid":              True,
        "grid.color":             PAL["grid"],
        "grid.linewidth":         0.6,
        "grid.linestyle":         "--",
        "xtick.color":            PAL["subtext"],
        "ytick.color":            PAL["subtext"],
        "text.color":             PAL["text"],
        "axes.spines.top":        False,
        "axes.spines.right":      False,
    })

    # ── Figure + axes ─────────────────────────────────────────────────────────
    DPI = 150
    PX  = 1080
    # 1080 / 150 = 7.2 inches → exactly 1080 × 1080 px before bbox adjustments.
    # R equivalent: png("output.png", width=1080, height=1080, res=150)
    fig = plt.figure(figsize=(PX / DPI, PX / DPI), dpi=DPI, facecolor=PAL["bg"])

    # [left, bottom, width, height] as fractions of the figure (0–1).
    # fig.add_axes() gives precise layout control — more explicit than ggplot2's
    # automatic layout engine, but more flexible for custom designs.
    ax = fig.add_axes([0.12, 0.12, 0.82, 0.65])
    ax.set_facecolor(PAL["bg"])

    # ── Extract arrays ────────────────────────────────────────────────────────
    df_s      = df.sort_values("year").reset_index(drop=True)
    years     = df_s["year"].astype(int).values
    mora_idx  = df_s["mora_idx"].values
    nat_idx   = df_s["nat_idx"].values
    last_year = int(years[-1])
    last_mora = mora_idx[-1]
    last_nat  = nat_idx[-1]

    # ── Axis limits ───────────────────────────────────────────────────────────
    # Leave 12% padding above and a little below; right margin for end labels.
    y_min = min(mora_idx.min(), nat_idx.min()) * 0.88
    y_max = max(mora_idx.max(), nat_idx.max()) * 1.13
    # Round y_min down to nearest 25, y_max up to nearest 25
    y_min = np.floor(y_min / 25) * 25
    y_max = np.ceil( y_max / 25) * 25

    x_lo = int(years.min()) - 1       # one year before first data
    x_hi = last_year + 9              # extra room on the right for end labels

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_min, y_max)

    # ── Gap shading (fill_between) ────────────────────────────────────────────
    # fill_between with a `where=` condition shades regions conditionally.
    # This is the pandas/matplotlib equivalent of ggplot2's geom_ribbon():
    #
    #   geom_ribbon(
    #     aes(ymin = pmin(mora, nat), ymax = pmax(mora, nat),
    #         fill = mora > nat),
    #     alpha = 0.22
    #   )
    #
    # interpolate=True smooths the fill at crossover points so there are
    # no abrupt gaps where one series dips below the other.
    ax.fill_between(
        years, mora_idx, nat_idx,
        where=(mora_idx >= nat_idx),   # MORA outperforms → green
        color=PAL["fill_above"],
        alpha=PAL["fill_alpha"],
        interpolate=True,
        zorder=1,
    )
    ax.fill_between(
        years, mora_idx, nat_idx,
        where=(mora_idx < nat_idx),    # MORA underperforms → slate
        color=PAL["fill_below"],
        alpha=PAL["fill_alpha"],
        interpolate=True,
        zorder=1,
    )

    # ── Lines ─────────────────────────────────────────────────────────────────
    # Draw national median first (lower z-order) so MORA sits on top.
    # Dashed line for the national median signals it's a reference / benchmark.
    # R: geom_line(aes(linetype = series, colour = series))
    ax.plot(
        years, nat_idx,
        color=PAL["nat_line"],
        linewidth=2.0,
        linestyle="--",
        dash_capstyle="round",
        zorder=3,
    )
    ax.plot(
        years, mora_idx,
        color=PAL["mora_line"],
        linewidth=2.6,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
    )

    # ── Base-year reference line (1990 = 100) ─────────────────────────────────
    # A subtle horizontal line at y=100 anchors the viewer's eye to the
    # base year. This is cleaner than gridlines at every tick.
    ax.axhline(
        100,
        color=PAL["subtext"], linewidth=0.85,
        linestyle=":", alpha=0.65, zorder=2,
    )
    ax.text(
        x_lo + 0.8, 102,
        f"{BASE_YEAR} = 100",
        color=PAL["subtext"],
        fontsize=7.5, va="bottom", ha="left",
        zorder=6,
    )

    # ── COVID-19 annotation ───────────────────────────────────────────────────
    # Mark 2020 on both series — both MORA and the national median dipped.
    ax.axvline(
        2020,
        color=PAL["ev_covid"], linewidth=0.9,
        linestyle=":", alpha=0.50, zorder=2,
    )

    # Dots at the 2020 data points on each line
    for idx_arr in [mora_idx, nat_idx]:
        mask = years == 2020
        if mask.any():
            ax.scatter(
                2020, idx_arr[mask][0],
                color=PAL["ev_covid"], s=52, zorder=7,
                edgecolors="white", linewidths=1.2,
            )

    # Label — placed at a fixed y fraction, shifted right to clear the line
    label_y_covid = y_min + 0.30 * (y_max - y_min)
    ax.text(
        2021.6, label_y_covid,
        "COVID-19\nclosure",
        color=PAL["ev_covid"],
        fontsize=8.0,
        ha="left", va="center",
        linespacing=1.35,
        zorder=8,
    )

    # ── Direct end labels ─────────────────────────────────────────────────────
    # Direct labelling (right-aligned to each line) replaces a legend.
    # This is the FiveThirtyEight / The Pudding aesthetic.
    # R equivalent: geom_text() or ggrepel::geom_text_repel()
    label_x = last_year + 1.0

    # Prevent overlap: if the two series end within 8 index points, nudge apart
    sep = last_mora - last_nat
    mora_label_y = last_mora
    nat_label_y  = last_nat
    MIN_GAP = 8.0
    if abs(sep) < MIN_GAP:
        half = MIN_GAP / 2
        mora_label_y = last_mora + half * np.sign(sep if sep != 0 else 1)
        nat_label_y  = last_nat  - half * np.sign(sep if sep != 0 else 1)

    ax.text(
        label_x, mora_label_y,
        f"Rainier\n{last_mora:.0f}",
        color=PAL["mora_line"],
        fontsize=8.5, fontweight="bold",
        va="center", ha="left",
        linespacing=1.3, zorder=9,
    )
    ax.text(
        label_x, nat_label_y,
        f"Nat. median\n{last_nat:.0f}",
        color=PAL["nat_line"],
        fontsize=8.5, fontweight="bold",
        va="center", ha="left",
        linespacing=1.3, zorder=9,
    )

    # ── Legend for gap shading ────────────────────────────────────────────────
    # Small patch legend in the lower-left — explains the fill colors.
    above_patch = mpatches.Patch(
        facecolor=PAL["fill_above"], alpha=0.55,
        label="Rainier above median",
    )
    below_patch = mpatches.Patch(
        facecolor=PAL["fill_below"], alpha=0.55,
        label="Rainier below median",
    )
    ax.legend(
        handles=[above_patch, below_patch],
        loc="lower left",
        fontsize=7.5,
        framealpha=0.88,
        edgecolor=PAL["spine"],
        handlelength=1.0,
        handletextpad=0.5,
        borderpad=0.6,
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    # X-axis: decade ticks + final year
    # R: scale_x_continuous(breaks = c(seq(1980, 2020, 10), last_year))
    first_decade = (int(years.min()) // 10 + 1) * 10   # e.g. 1980
    x_ticks = sorted(set(range(first_decade, last_year, 10)) | {last_year})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(y) for y in x_ticks], fontsize=9, color=PAL["subtext"])

    # Y-axis: 25-point intervals, formatted as integers
    # R: scale_y_continuous(labels = scales::label_number(), breaks = seq(y_min, y_max, 25))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_idx_fmt))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.tick_params(axis="y", labelsize=9, colors=PAL["subtext"])

    # Horizontal grid only (FiveThirtyEight aesthetic)
    # R: theme(panel.grid.major.x = element_blank())
    ax.grid(axis="y", color=PAL["grid"], linewidth=0.6, linestyle="--", zorder=0)
    ax.grid(axis="x", visible=False)

    ax.spines["left"].set_color(PAL["spine"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(PAL["spine"])
    ax.spines["bottom"].set_linewidth(0.8)

    # Y-axis label
    # R: labs(y = "Visitor index (1990 = 100)")
    ax.set_ylabel(
        "Visitor index (1990 = 100)",
        color=PAL["subtext"], fontsize=9, labelpad=7,
    )

    # ── Title block ───────────────────────────────────────────────────────────
    # Decorative green accent bar — editorial header pattern (NYT, FiveThirtyEight)
    fig.add_artist(
        plt.Line2D(
            [0.12, 0.17], [0.975, 0.975],
            transform=fig.transFigure,
            color=PAL["mora_line"], linewidth=4, solid_capstyle="butt",
        )
    )

    # Compute headline dynamically based on final index values
    if last_mora > last_nat:
        headline = "RAINIER HAS GROWN FASTER THAN MOST DESTINATION PARKS"
    else:
        headline = "MOUNT RAINIER COMPARED TO THE TYPICAL DESTINATION PARK"

    fig.text(
        0.12, 0.955,
        headline,
        fontsize=13.8, fontweight="bold",
        color=PAL["text"],
        ha="left", va="top",
        transform=fig.transFigure,
    )

    # Subtitle
    fig.text(
        0.12, 0.907,
        f"Recreation visitors indexed to {BASE_YEAR} = 100.  "
        f"Mount Rainier (MORA) vs. median of {n_dest} destination parks\n"
        f"(NPS units with >500K peak annual TRV, 1979–{last_year}).  "
        f"Source: NPS IRMA data package, TRV statistic.",
        fontsize=8.8,
        color=PAL["subtext"],
        ha="left", va="top",
        linespacing=1.45,
        transform=fig.transFigure,
    )

    # Thin rule between header and chart
    fig.add_artist(
        plt.Line2D(
            [0.12, 0.955], [0.808, 0.808],
            transform=fig.transFigure,
            color=PAL["spine"], linewidth=0.7,
        )
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.12, 0.048,
        "Source: NPS Visitor Use Statistics · irma.nps.gov  ·  Statistic: TRV (Total Recreation Visitors)",
        fontsize=7.2, color=PAL["subtext"],
        ha="left", va="bottom",
        transform=fig.transFigure,
    )
    fig.text(
        0.955, 0.048,
        "@readingatabar",
        fontsize=8.5, color=PAL["handle"], fontweight="bold",
        ha="right", va="bottom",
        transform=fig.transFigure,
    )

    # ── Save (Pillow pipeline → exactly 1080 × 1080 px) ───────────────────────
    # bbox_inches="tight" expands the bounding box to include all text,
    # which can make pixel dimensions unpredictable.  We save to a buffer,
    # pad the shorter dimension with the background colour, then resize.
    # R equivalent:
    #   ggsave(file, plot, width=7.2, height=7.2, dpi=150)
    #   then magick::image_resize(img, "1080x1080!") if exact dims are needed.
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=DPI,
        bbox_inches="tight",
        facecolor=PAL["bg"], edgecolor="none",
    )
    plt.close(fig)   # Free memory — R equivalent: dev.off()
    buf.seek(0)

    rendered = Image.open(buf)
    w, h     = rendered.size
    side     = max(w, h)

    # Convert background hex → RGB tuple for Pillow
    bg_r = int(PAL["bg"][1:3], 16)
    bg_g = int(PAL["bg"][3:5], 16)
    bg_b = int(PAL["bg"][5:7], 16)

    square = Image.new("RGB", (side, side), (bg_r, bg_g, bg_b))
    square.paste(rendered.convert("RGB"), ((side - w) // 2, (side - h) // 2))

    # High-quality Lanczos downsampling to exactly 1080 × 1080
    final = square.resize((PX, PX), Image.LANCZOS)
    final.save(OUTPUT_FILE, dpi=(DPI, DPI))

    print(f"\n[done]  Chart saved → {OUTPUT_FILE}")
    img = Image.open(OUTPUT_FILE)
    print(f"        {OUTPUT_FILE.stat().st_size / 1024:.0f} KB  "
          f"|  {DPI} DPI  |  {img.size[0]} × {img.size[1]} px")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Mount Rainier — Indexed Comparison Visualization")
    print("=" * 60)

    df, n_dest = load_and_prepare()

    last = df.iloc[-1]
    gap  = last["mora_idx"] - last["nat_idx"]

    print(f"\n[story] MORA final-year index:          {last['mora_idx']:.1f}")
    print(f"[story] National median final-year idx: {last['nat_idx']:.1f}")
    if gap > 0:
        print(f"[story] MORA outperforms by:            {gap:.1f} index points")
    else:
        print(f"[story] National median outperforms by: {-gap:.1f} index points")

    create_chart(df, n_dest)


if __name__ == "__main__":
    main()
