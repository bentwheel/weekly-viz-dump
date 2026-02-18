"""
scripts/visualize_top10.py
==========================
Create a multi-line time series of the 10 most-visited NPS units by average
annual TRV (Total Recreation Visitors, 1979–2024), with Mount Rainier always
included and highlighted even if it falls outside the top 10.

Each park gets a distinct color within a Pacific Northwest earthy-green palette.
MORA always uses its signature forest green and a heavier line weight.

Usage:
    python scripts/visualize_top10.py

Output:
    output/rainier_top10.png  (1080 × 1080 px, 150 DPI)

─────────────────────────────────────────────────────────────────────────────
Key R → matplotlib/pandas translation notes
─────────────────────────────────────────────────────────────────────────────
  R / tidyverse                      Python / pandas + matplotlib equivalent
  ──────────────────────────────     ─────────────────────────────────────────
  group_by(unit) |>                  df.groupby("unit")["trv"].mean()
    summarise(avg = mean(trv))
  slice_max(avg, n=10)               series.nlargest(10).index
  pivot_wider(names_from=unit,       df.pivot(index="year",
    values_from=trv)                   columns="unit", values="trv")
  bind_rows(top10, mora_row)         pd.concat([top10_df, mora_row])
  map(df_list, ~ggplot(...) +        for code, series in ...:
    geom_line())                         ax.plot(years, series, ...)
  ggrepel::geom_label_repel()        custom greedy label-spread algorithm
  scales::label_number(              mticker.FuncFormatter(millions_fmt)
    scale=1e-6, suffix="M")
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

matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PACKAGE  = PROJECT_ROOT / "data" / "raw" / "nps_visitation_all_parks.csv"
OUTPUT_DIR   = PROJECT_ROOT / "output"
OUTPUT_FILE  = OUTPUT_DIR / "rainier_top10.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Design system
# ─────────────────────────────────────────────────────────────────────────────
PAL = {
    "bg":      "#F4F1EC",   # warm parchment
    "grid":    "#D5CFC8",
    "spine":   "#C5BFB8",
    "text":    "#1E2832",
    "subtext": "#5E6E7E",
    "handle":  "#2D6A4F",
    "mora":    "#2D6A4F",   # signature forest green — always for MORA
}

# 10 earthy/green colors for the comparison parks.
# Arranged to be visually distinct while staying in the PNW natural palette.
# R equivalent: a named character vector of hex codes, e.g.
#   park_colors <- c(park1="#1B4332", park2="#40916C", ...)
OTHER_COLORS = [
    "#1B4332",   # midnight forest
    "#40916C",   # medium green
    "#74C69D",   # sage green
    "#8B5E3C",   # earth brown
    "#B5886E",   # terracotta
    "#6B7C5A",   # olive
    "#A07D50",   # sandy amber
    "#4A7C6F",   # teal sage
    "#7D9B6A",   # meadow green
    "#9A8560",   # driftwood
]

# ─────────────────────────────────────────────────────────────────────────────
# Park code → friendly display name
# These are the most common NPS unit codes that appear at the top of the
# TRV rankings.  Unknown codes fall back to the code itself.
# ─────────────────────────────────────────────────────────────────────────────
PARK_NAMES = {
    "BLRI": "Blue Ridge Pkwy",
    "GRSM": "Gr. Smoky Mtns",
    "GCNP": "Grand Canyon",
    "GRCA": "Grand Canyon",
    "ZION": "Zion",
    "ROMO": "Rocky Mountain",
    "YELL": "Yellowstone",
    "OLYM": "Olympic",
    "ACAD": "Acadia",
    "JOTR": "Joshua Tree",
    "GATE": "Gateway NRA",
    "GOGA": "Golden Gate NRA",
    "LAME": "Lake Mead NRA",
    "CAHA": "Cape Hatteras NS",
    "DEVA": "Death Valley",
    "GLAC": "Glacier",
    "SHEN": "Shenandoah",
    "CUVA": "Cuyahoga Valley",
    "INDU": "Indiana Dunes",
    "BRCA": "Bryce Canyon",
    "ARCH": "Arches",
    "MEVE": "Mesa Verde",
    "MORA": "Mt. Rainier",
    "NOCA": "N. Cascades",
    "GRBA": "Great Basin",
    "MOJA": "Mojave NP",
    "CHOH": "C&O Canal",
    "GWMP": "Geo. Wash. Pkwy",
    "NACE": "Natl Capital Parks",
    "PRWI": "Prince William FS",
    "CHAT": "Chattahoochee NRA",
    "HAFE": "Harpers Ferry",
    "PETE": "Petersburg NB",
    "COLO": "Colorado NM",
    "FOPU": "Fort Pulaski NM",
    # Additional codes that commonly appear at the top of TRV rankings
    "LAKE": "Lake Mead NRA",
    "NATR": "Natchez Trace Pkwy",
    "NACA": "Natl Capital Parks",
    "GUIS": "Gulf Islands NS",
    "CACO": "Cape Cod NS",
    "DEWA": "Delaware Water Gap",
    "PINE": "Pinelands NS",
    "NERI": "New River Gorge",
    "BIBE": "Big Bend",
    "GUMO": "Guadalupe Mtns",
    "BADL": "Badlands",
    "THRO": "Theodore Roosevelt",
    "ISRO": "Isle Royale",
    "VOYA": "Voyageurs",
    "DENA": "Denali",
    "KATM": "Katmai",
    "KEFJ": "Kenai Fjords",
    "WRST": "Wrangell-St. Elias",
    "GLBA": "Glacier Bay",
    "LACL": "Lake Clark",
    "GAAR": "Gates of the Arctic",
    "KOVA": "Kobuk Valley",
    "HALE": "Haleakala",
    "HAVO": "Hawaii Volcanoes",
    "BISC": "Biscayne",
    "DRTO": "Dry Tortugas",
    "EVER": "Everglades",
    "WICA": "Wind Cave",
    "CAVE": "Carlsbad Caverns",
    "PINN": "Pinnacles",
    "CHIS": "Channel Islands",
    "CRLA": "Crater Lake",
    "LAVO": "Lassen Volcanic",
    "SEQU": "Sequoia",
    "KICA": "Kings Canyon",
    "YOSE": "Yosemite",
    "CANY": "Canyonlands",
    "CARE": "Capitol Reef",
    "PEFO": "Petrified Forest",
    "SAGU": "Saguaro",
    "MOJA": "Mojave NP",
    "GRBA": "Great Basin",
    "GRTE": "Grand Teton",
}

# ─────────────────────────────────────────────────────────────────────────────
# Backcountry / wilderness park allowlist
# ─────────────────────────────────────────────────────────────────────────────
# These are NPS units with substantial backcountry access — parks where
# dispersed camping, permit systems, or multi-day trail networks exist.
# Excluded: parkways (BLRI, NATR, GWMP), urban recreation areas (GATE, GOGA),
# national seashores/lakeshores (CACO, GUIS), and other non-wilderness units.
BACKCOUNTRY_PARKS = {
    # Pacific Northwest / Cascades
    "MORA", "OLYM", "NOCA", "CRLA", "LAVO", "REDW",
    # California
    "YOSE", "SEQU", "KICA", "PINN", "CHIS",
    # Southwest desert / canyon country
    "GRCA", "GCNP", "ZION", "BRCA", "ARCH", "CANY", "CARE", "MEVE",
    "PEFO", "SAGU", "CAVE",
    # Great Basin / Mojave
    "DEVA", "JOTR", "GRBA", "MOJA",
    # Rocky Mountains
    "ROMO", "GRTE", "YELL", "GLAC", "WICA", "BADL", "THRO",
    # Upper Midwest
    "ISRO", "VOYA",
    # East / Appalachian
    "GRSM", "SHEN", "ACAD", "NERI",
    # Southeast / Gulf / Florida
    "EVER", "BISC", "DRTO", "BIBE", "GUMO",
    # Alaska
    "DENA", "KATM", "KEFJ", "WRST", "GLBA", "LACL", "GAAR", "KOVA",
    # Hawaii
    "HALE", "HAVO",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions (shared with other scripts in this project)
# ─────────────────────────────────────────────────────────────────────────────

def _find_column(columns: pd.Index, keywords: list[str]) -> str | None:
    """Case-insensitive column search. R: dplyr::select(df, contains(...))"""
    normalised = {c.lower().replace(" ", "_").replace("-", "_"): c for c in columns}
    for kw in keywords:
        if kw in normalised:
            return normalised[kw]
    for kw in keywords:
        for norm, orig in normalised.items():
            if kw in norm:
                return orig
    return None


def parse_number(series: pd.Series) -> pd.Series:
    """Strip formatting and coerce to float. R: readr::parse_number()"""
    cleaned = (
        series.astype(str)
        .str.replace(r"[,+%\s]", "", regex=True)
        .str.replace("—", "", regex=False)
        .str.replace("N/A", "", regex=False)
        .replace("", float("nan"))
    )
    return pd.to_numeric(cleaned, errors="coerce")


def millions_fmt(x: float, _pos) -> str:
    """Y-axis formatter: 1500000 → '1.5M'. R: scales::label_number(scale=1e-6, suffix='M')"""
    if x == 0:
        return "0"
    m = x / 1_000_000
    return f"{m:.1f}M" if m != int(m) else f"{int(m)}M"


# ─────────────────────────────────────────────────────────────────────────────
# Label-spreading: a greedy algorithm to prevent overlapping end labels
# ─────────────────────────────────────────────────────────────────────────────

def spread_labels(
    codes: list[str],
    final_vals: dict[str, float],
    min_gap: float,
) -> dict[str, float]:
    """
    Given a set of final data values, compute non-overlapping label y-positions
    using a two-pass greedy nudge (top-down then bottom-up).

    This is the matplotlib equivalent of ggrepel::geom_text_repel().
    R's ggrepel handles this automatically; in matplotlib we implement it
    manually — a useful exercise in understanding what ggrepel does internally.

    Parameters
    ----------
    codes      : list of park codes
    final_vals : dict mapping code → final-year TRV value
    min_gap    : minimum vertical distance between label centres (in data units)

    Returns
    -------
    dict mapping code → nudged y-position for the label
    """
    # Sort codes by final value, descending (highest park label at the top)
    # R: arrange(desc(final_val))
    ordered = sorted(codes, key=lambda c: -final_vals[c])

    # Pass 1: top-down — push labels down when too close
    pos = {c: final_vals[c] for c in ordered}
    prev = float("inf")
    for c in ordered:
        if prev - pos[c] < min_gap:
            pos[c] = prev - min_gap
        prev = pos[c]

    # Pass 2: bottom-up — correct any over-nudge at the bottom
    prev = float("-inf")
    for c in reversed(ordered):
        if pos[c] - prev < min_gap:
            pos[c] = prev + min_gap
        prev = pos[c]

    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_top_parks(n: int = 10) -> tuple[dict, pd.DataFrame, bool]:
    """
    Load the NPS package, identify the top-n backcountry/wilderness parks
    by 2024 TRV, add MORA if not present, and return per-park annual time series.

    Only parks in BACKCOUNTRY_PARKS are considered — parkways, urban recreation
    areas, and national seashores are excluded so the comparison is meaningful
    for a wilderness park like Mount Rainier.

    Returns
    -------
    series_dict : dict  {unit_code: (years_array, trv_array)}
    meta        : DataFrame  {unit_code, trv_2024, display_name, color}
    mora_added  : bool — True if MORA was not in the top-n and was appended
    """

    if not RAW_PACKAGE.exists():
        print(
            f"\n[error] NPS package not found at:\n  {RAW_PACKAGE}\n\n"
            f"  Run 'python scripts/fetch_data.py' first to download it."
        )
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[read]  Loading {RAW_PACKAGE.name}…")
    try:
        df_all = pd.read_csv(RAW_PACKAGE, low_memory=False)
    except UnicodeDecodeError:
        df_all = pd.read_csv(RAW_PACKAGE, encoding="latin1", low_memory=False)
    print(f"[info]  {df_all.shape[0]:,} rows × {df_all.shape[1]} cols")

    # ── Locate columns ────────────────────────────────────────────────────────
    code_col  = _find_column(df_all.columns, ["unitcode", "unit_code", "parkcode"])
    year_col  = _find_column(df_all.columns, ["year"])
    stat_col  = _find_column(df_all.columns, ["statistic"])
    value_col = _find_column(df_all.columns, ["value"])

    for label, col in [("park code", code_col), ("year", year_col),
                        ("statistic", stat_col), ("value", value_col)]:
        if col is None:
            print(f"[error] Cannot find '{label}' column. Got: {list(df_all.columns)}")
            sys.exit(1)

    # ── Filter to TRV ─────────────────────────────────────────────────────────
    # R: filter(df_all, Statistic == "TRV")
    trv = df_all[df_all[stat_col].astype(str).str.upper() == "TRV"].copy()
    if trv.empty:
        trv = df_all[df_all[stat_col].astype(str).str.upper() == "TV"].copy()
    if trv.empty:
        print(f"[error] No TRV/TV rows. Codes: {df_all[stat_col].unique().tolist()}")
        sys.exit(1)
    print(f"[info]  TRV rows: {len(trv):,}")

    trv[year_col]  = parse_number(trv[year_col])
    trv[value_col] = parse_number(trv[value_col])

    # ── Annual TRV per park ───────────────────────────────────────────────────
    # Sum monthly → annual, then keep year > 1900.
    # R: trv |> group_by(UnitCode, Year) |> summarise(trv = sum(Value, na.rm=TRUE))
    annual = (
        trv
        .groupby([code_col, year_col], as_index=False)[value_col]
        .sum()
    )
    annual.columns = ["unit_code", "year", "trv"]
    annual = annual[annual["year"] > 1900].copy()
    annual["year"] = annual["year"].astype(int)

    # ── Filter to backcountry/wilderness parks ────────────────────────────────
    # Restricts the comparison pool to parks where backpacking is a core
    # activity, excluding parkways, urban NRAs, seashores, etc.
    # R: annual |> filter(unit_code %in% BACKCOUNTRY_PARKS)
    bc_annual = annual[annual["unit_code"].isin(BACKCOUNTRY_PARKS)].copy()
    print(f"[info]  Parks in backcountry allowlist with data: "
          f"{bc_annual['unit_code'].nunique()}")

    # ── Rank by 2024 TRV ─────────────────────────────────────────────────────
    # Use a single year rather than a mean so the ranking reflects current
    # popularity and isn't skewed by parks that were closed or newly opened.
    # R: bc_annual |> filter(year == 2024) |> arrange(desc(trv))
    rank_year = bc_annual["year"].max()   # use latest available year
    trv_ranked = (
        bc_annual[bc_annual["year"] == rank_year]
        .copy()
        .sort_values("trv", ascending=False)
        .reset_index(drop=True)
    )
    print(f"[info]  Ranking by {rank_year} TRV across "
          f"{len(trv_ranked)} backcountry parks")

    # ── Select top-n parks ────────────────────────────────────────────────────
    # slice_max(trv, n=n) in R; .head(n) in pandas after sort_values
    top_codes   = trv_ranked.head(n)["unit_code"].tolist()
    mora_in_top = "MORA" in top_codes

    if mora_in_top:
        selected   = top_codes
        mora_added = False
        print(f"[info]  MORA is in the top {n} backcountry parks — no need to add separately.")
    else:
        selected   = top_codes + ["MORA"]
        mora_added = True
        mora_row   = trv_ranked[trv_ranked["unit_code"] == "MORA"]
        mora_trv   = mora_row["trv"].iloc[0] if not mora_row.empty else 0
        mora_rank  = mora_row.index[0] + 1   if not mora_row.empty else "?"
        print(f"[info]  MORA is NOT in top {n} backcountry parks  "
              f"({rank_year} TRV {mora_trv:,.0f}; rank #{mora_rank} among backcountry parks)")

    print(f"[info]  Parks to plot: {selected}")

    # ── Assign display names and colors ───────────────────────────────────────
    # Colors: MORA always gets PAL["mora"]; top-n parks cycle through OTHER_COLORS
    # in their rank order (rank 1 = first color, etc.).
    color_map  = {}
    color_iter = iter(OTHER_COLORS)

    for code in top_codes:        # top 10 in rank order
        if code == "MORA":
            color_map[code] = PAL["mora"]
        else:
            color_map[code] = next(color_iter)

    if not mora_in_top:
        color_map["MORA"] = PAL["mora"]

    name_map = {c: PARK_NAMES.get(c, c) for c in selected}

    # ── Build per-park time series ────────────────────────────────────────────
    # Filter annual data to only the selected parks, then pivot to wide format.
    # pd.pivot() ≈ tidyr::pivot_wider(names_from="unit_code", values_from="trv")
    subset = annual[annual["unit_code"].isin(selected)].copy()
    wide   = subset.pivot(index="year", columns="unit_code", values="trv")
    # fill NaN with 0 for missing year/park combinations (rare in this dataset)
    wide   = wide.fillna(0)

    years = wide.index.values.astype(int)

    # Build dict: code → (years_array, trv_array)
    series_dict = {code: (years, wide[code].values) for code in wide.columns}

    # Summary metadata DataFrame for the chart
    # trv_2024: the ranking value; used for label ordering and the ranks printout.
    # R: left_join(tibble(unit_code=selected), trv_ranked, by="unit_code")
    trv_lookup = trv_ranked.set_index("unit_code")["trv"].to_dict()
    meta = pd.DataFrame({
        "unit_code":    selected,
        "display_name": [name_map[c] for c in selected],
        "trv_2024":     [trv_lookup.get(c, 0) for c in selected],
        "color":        [color_map[c] for c in selected],
    }).sort_values("trv_2024", ascending=False).reset_index(drop=True)

    # Print the rankings
    print(f"\n[ranks] Backcountry park rankings by {rank_year} TRV:")
    for _, row in meta.iterrows():
        tag = "  ← MORA" if row["unit_code"] == "MORA" and mora_added else ""
        print(f"        {row['display_name']:<22}  {rank_year} TRV {row['trv_2024']:>10,.0f}{tag}")

    return series_dict, meta, mora_added


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def create_chart(
    series_dict: dict,
    meta: pd.DataFrame,
    mora_added: bool,
) -> None:
    """
    Build and save the 1080 × 1080 multi-line time series chart.
    """

    DPI = 150
    PX  = 1080

    # ── Style ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial",
                               "Liberation Sans", "DejaVu Sans"],
        "figure.facecolor":   PAL["bg"],
        "axes.facecolor":     PAL["bg"],
        "axes.edgecolor":     PAL["spine"],
        "axes.grid":          True,
        "grid.color":         PAL["grid"],
        "grid.linewidth":     0.55,
        "grid.linestyle":     "--",
        "xtick.color":        PAL["subtext"],
        "ytick.color":        PAL["subtext"],
        "text.color":         PAL["text"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

    fig = plt.figure(figsize=(PX / DPI, PX / DPI), dpi=DPI, facecolor=PAL["bg"])
    # Narrower axes to give ample room for single-line end labels on the right.
    # The right margin (0.72–0.955) is ~1.7 inches at 7.2" total — enough for
    # "Great Smoky Mtns  12.4M" at fontsize 7.5 without wrapping.
    ax  = fig.add_axes([0.12, 0.12, 0.60, 0.65])
    ax.set_facecolor(PAL["bg"])

    # ── Determine axis bounds ─────────────────────────────────────────────────
    # Gather global x range and y ceiling across all series.
    all_years = sorted({y for years, _ in series_dict.values() for y in years})
    all_vals  = [v for _, trv in series_dict.values() for v in trv if v > 0]

    x_lo = min(all_years) - 1
    x_hi = max(all_years) + 6   # extra room for gap + labels
    y_max = max(all_vals) * 1.10   # 10% headroom above the top park
    y_max = np.ceil(y_max / 1_000_000) * 1_000_000   # round up to nearest million

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_max)

    # ── Draw lines ────────────────────────────────────────────────────────────
    # Plot all comparison parks first (lower zorder), then MORA on top.
    # R equivalent: a ggplot call with geom_line(aes(colour=unit_code)) +
    #   scale_colour_manual(values=color_map)
    final_vals = {}   # code → last-year TRV (used for label spreading)

    for _, row in meta.iterrows():
        code  = row["unit_code"]
        color = row["color"]
        is_mora = (code == "MORA")

        if code not in series_dict:
            continue

        years, trv = series_dict[code]
        # Mask zeros for cleaner line (don't draw segments through missing years)
        mask  = trv > 0
        y_arr = trv.copy().astype(float)
        y_arr[~mask] = np.nan

        ax.plot(
            years, y_arr,
            color=color,
            linewidth=2.5  if is_mora else 1.6,
            alpha=1.0      if is_mora else 0.82,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5       if is_mora else 3,
        )

        # Record the final non-zero value for label placement
        valid = trv[mask]
        final_vals[code] = float(valid[-1]) if len(valid) > 0 else 0.0

    # ── Spread end labels ─────────────────────────────────────────────────────
    # A greedy algorithm (implemented in spread_labels()) mimics ggrepel.
    # Minimum gap: 4% of y_max so labels don't touch each other.
    min_gap = y_max * 0.038
    label_ys = spread_labels(list(final_vals.keys()), final_vals, min_gap)

    label_x  = max(all_years) + 2.5   # gap between line end and label

    for _, row in meta.iterrows():
        code    = row["unit_code"]
        color   = row["color"]
        name    = row["display_name"]
        is_mora = (code == "MORA")

        if code not in final_vals or final_vals[code] == 0:
            continue

        label_y  = label_ys[code]
        actual_y = final_vals[code]

        # If the label was nudged significantly from the line end, draw a
        # thin connector — like ggrepel's leader line.
        # R: ggrepel handles this automatically with segment.* aesthetics.
        if abs(label_y - actual_y) > min_gap * 0.3:
            ax.plot(
                [max(all_years) + 0.4, label_x - 0.3],
                [actual_y, label_y],
                color=color, linewidth=0.7, alpha=0.5, zorder=4,
            )

        # End label: park name and final TRV on a single line.
        # e.g.  "Gr. Smoky Mtns  12.4M"  or  "[R] Mt. Rainier  1.6M"
        final_m = final_vals[code] / 1_000_000
        val_str = f"{final_m:.1f}M" if final_m != int(final_m) else f"{int(final_m)}M"
        prefix  = "[R] " if is_mora else ""

        ax.text(
            label_x + 0.3, label_y,
            f"{prefix}{name}  {val_str}",
            color=color,
            fontsize=7.5,
            fontweight="bold" if is_mora else "normal",
            va="center", ha="left",
            zorder=9,
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    # X-axis: decade ticks + final year
    first_decade = (min(all_years) // 10 + 1) * 10
    last_year    = max(all_years)
    x_ticks      = sorted(set(range(first_decade, last_year, 10)) | {last_year})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(y) for y in x_ticks], fontsize=9, color=PAL["subtext"])

    # Y-axis: millions formatter
    # R: scale_y_continuous(labels = scales::label_number(scale=1e-6, suffix="M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2_000_000))
    ax.tick_params(axis="y", labelsize=9, colors=PAL["subtext"])

    ax.grid(axis="y", color=PAL["grid"], linewidth=0.55, linestyle="--", zorder=0)
    ax.grid(axis="x", visible=False)

    ax.spines["left"].set_color(PAL["spine"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(PAL["spine"])
    ax.spines["bottom"].set_linewidth(0.8)

    ax.set_ylabel(
        "Annual recreation visitors (TRV)",
        color=PAL["subtext"], fontsize=9, labelpad=7,
    )

    # ── Title block ───────────────────────────────────────────────────────────
    fig.add_artist(
        plt.Line2D(
            [0.12, 0.17], [0.975, 0.975],
            transform=fig.transFigure,
            color=PAL["mora"], linewidth=4, solid_capstyle="butt",
        )
    )

    mora_note = (
        "Mount Rainier [R] added for context — outside the top 10."
        if mora_added else
        "Mount Rainier [R] is among the top 10."
    )

    fig.text(
        0.12, 0.955,
        "TOP 10 BACKPACKING PARKS — AND WHERE RAINIER SITS",
        fontsize=14.0, fontweight="bold",
        color=PAL["text"], ha="left", va="top",
        transform=fig.transFigure,
    )
    fig.text(
        0.12, 0.907,
        f"Annual recreation visitors (TRV), {min(all_years)}–{last_year}.  "
        f"Wilderness and backcountry parks only;\nparkways, seashores, and urban NRAs excluded.  "
        f"{mora_note}",
        fontsize=9.0, color=PAL["subtext"],
        ha="left", va="top", linespacing=1.45,
        transform=fig.transFigure,
    )

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

    # ── Save (Pillow → exactly 1080 × 1080 px) ───────────────────────────────
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=DPI,
        bbox_inches="tight",
        facecolor=PAL["bg"], edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)

    rendered = Image.open(buf)
    w, h     = rendered.size
    side     = max(w, h)

    bg_r = int(PAL["bg"][1:3], 16)
    bg_g = int(PAL["bg"][3:5], 16)
    bg_b = int(PAL["bg"][5:7], 16)

    square = Image.new("RGB", (side, side), (bg_r, bg_g, bg_b))
    square.paste(rendered.convert("RGB"), ((side - w) // 2, (side - h) // 2))
    final  = square.resize((PX, PX), Image.LANCZOS)
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
    print("  Mount Rainier — Top 10 Parks Comparison")
    print("=" * 60)

    series_dict, meta, mora_added = load_top_parks(n=10)
    create_chart(series_dict, meta, mora_added)


if __name__ == "__main__":
    main()
