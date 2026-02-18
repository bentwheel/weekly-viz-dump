"""
scripts/visualize.py
====================
Create a publication-quality, Instagram-ready (1080 × 1080 px) time series
chart of Mount Rainier National Park annual visitation, 1967–2024.

Designed to look like FiveThirtyEight / The Pudding — clean, editorial,
story-first — with a Pacific Northwest colour palette.

Usage:
    python scripts/visualize.py

Output:
    output/rainier_visitors.png  (1080 × 1080 px, 150 DPI)

─────────────────────────────────────────────────────────────────────────────
Key R → matplotlib translation notes
─────────────────────────────────────────────────────────────────────────────
  R / ggplot2                       matplotlib equivalent
  ──────────────────────────────    ──────────────────────────────────────────
  theme_set() / theme()             plt.rcParams.update({...})
  ggplot(df, aes(x, y))            fig, ax = plt.subplots()
  geom_area()                       ax.fill_between(x, 0, y, ...)
  geom_line()                       ax.plot(x, y, ...)
  geom_point()                      ax.scatter(x, y, ...)
  annotate("text", ...)             ax.text(x, y, label, ...)
  annotate("segment", ...)          ax.axvline(x, ...)
  scale_y_continuous(labels=...)    ax.yaxis.set_major_formatter(FuncFormatter)
  scale_x_continuous(breaks=...)    ax.set_xticks([...])
  labs(title=, subtitle=, ...)      fig.text(...) — more layout control
  theme(panel.grid.minor=element_blank()) ax.grid(axis="y")
  ggsave(filename, dpi=, width=)    fig.savefig(path, dpi=, bbox_inches=)

  Key conceptual difference: ggplot2 has a "grammar of graphics" — you
  declare *what* you want and it figures out *how*.  matplotlib is
  procedural — you explicitly call each drawing operation in order.
  The object-oriented matplotlib API (fig, ax = plt.subplots()) is
  strongly preferred over the stateful plt.* API for anything non-trivial.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# Use the non-interactive Agg backend — renders to a file without needing
# a display server.  In R, this is equivalent to calling png(filename)
# before your plot commands.
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_CSV  = PROJECT_ROOT / "data" / "cleaned" / "rainier_visitation.csv"
OUTPUT_DIR   = PROJECT_ROOT / "output"
OUTPUT_FILE  = OUTPUT_DIR / "rainier_visitors.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Design system — Pacific Northwest colour palette
# ─────────────────────────────────────────────────────────────────────────────
# In R you might do: colors <- c(bg="#F4F1EC", line="#2D6A4F", ...)
# Python dicts work the same way — keyed constants, no special package needed.

PAL = {
    # Backgrounds — warm morning-fog parchment, evokes alpine haze
    "bg":          "#F4F1EC",
    "panel":       "#EDE8E2",

    # Data ink — Rainier old-growth forest green
    "line":        "#2D6A4F",
    "fill":        "#52B788",   # spruce green for the area fill
    "fill_alpha":   0.16,

    # Structural elements
    "grid":        "#D5CFC8",   # warm granite gray
    "spine":       "#C5BFB8",   # axis border
    "text":        "#1E2832",   # Cascades near-black
    "subtext":     "#5E6E7E",   # Olympic slate blue-gray
    "handle":      "#2D6A4F",   # @handle — same forest green as data line

    # Event annotation colours — distinct but harmonious
    "ev_volcano":  "#9E3B2C",   # Mt St Helens: rust red (volcanic)
    "ev_early90s": "#4A5568",   # Early 90s: neutral charcoal
    "ev_covid":    "#6B48A0",   # COVID: muted purple
    "ev_peak":     "#1B4332",   # All-time high: dark forest
    "ev_pilot":    "#2C5282",   # Timed entry: deep slate blue

    # Backcountry callout — mountain-lake blue-gray
    "bc_fill":     "#B7C4CF",
    "bc_alpha":     0.22,
}

# ─────────────────────────────────────────────────────────────────────────────
# Global matplotlib style
# ─────────────────────────────────────────────────────────────────────────────
# plt.rcParams is a dict-like object that controls all default plot settings.
# R equivalent: theme_set(theme_minimal()) + theme(text = element_text(...))
plt.rcParams.update({
    "font.family":            "sans-serif",
    # Specify font fallback chain — matplotlib uses the first available font.
    # On macOS: Helvetica Neue; on Linux: Liberation Sans or DejaVu Sans.
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

# ─────────────────────────────────────────────────────────────────────────────
# Key historical events to annotate
# ─────────────────────────────────────────────────────────────────────────────
# Each dict is one annotation: year, display label, colour, and placement hints.
# y_frac is a fraction of the full y-axis height (0 = bottom, 1 = top) where
# the annotation label will sit — we stagger these manually to avoid overlap.

EVENTS = [
    {
        # The eruption of Mt. St. Helens (100 mi SW of Rainier) in May 1980 had
        # a minimal effect on MORA visitation — still ~2M — but it's a landmark
        # PNW event worth contextualising.
        "year":   1980,
        "label":  "Mt. St. Helens\nerupts",
        "color":  PAL["ev_volcano"],
        "y_frac": 0.72,   # above the 1980 data point (~0.65 of y_ceil)
        "ha":     "left",  # label to the RIGHT of the vertical line
    },
    {
        # 1992 is the actual peak of the early 90s surge — 2.36M visitors,
        # nearly matching the 1977 all-time high at the time (2.44M).
        "year":   1992,
        "label":  "Early-'90s\npeak",
        "color":  PAL["ev_early90s"],
        "y_frac": 0.92,
        "ha":     "left",
    },
    {
        # Park closed to vehicles for most of spring 2020; visitation dropped
        # ~28% from 2019 (2.25M → 1.62M). Clear valley in the data.
        "year":   2020,
        "label":  "COVID-19\nclosure",
        "color":  PAL["ev_covid"],
        "y_frac": 0.48,
        "ha":     "left",
    },
    {
        # 2023: new all-time record — 2.52M recreation visitors.
        # This is ABOVE the shaded "oversubscribed" band (≥ pre-COVID high).
        "year":   2023,
        "label":  "All-time high\n2.52M visitors",
        "color":  PAL["ev_peak"],
        "y_frac": 0.98,
        "ha":     "right",
    },
    {
        # NPS launched a timed-entry reservation system for the Nisqually
        # corridor in summer 2024 — visitation dipped slightly to 2.49M.
        "year":   2024,
        "label":  "Timed-entry\npilot begins",
        "color":  PAL["ev_pilot"],
        "y_frac": 0.73,
        "ha":     "left",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def millions_fmt(x: float, _pos) -> str:
    """
    Y-axis tick formatter: converts raw visitor counts to "1M", "1.5M" etc.

    R equivalent:
        scale_y_continuous(labels = scales::label_number(scale=1e-6, suffix="M"))

    The _pos argument is required by matplotlib's FuncFormatter signature
    but we don't use it — it's the tick index.
    """
    if x == 0:
        return "0"
    m = x / 1_000_000
    # Show decimal only when needed: 1M not 1.0M, but 1.5M not 1M
    return f"{m:.1f}M" if m != int(m) else f"{int(m)}M"


def draw_event(ax: plt.Axes, df: pd.DataFrame, event: dict, y_max_data: float) -> None:
    """
    Draw a single event annotation: a vertical marker line, a dot at the
    data point, and a floating text label.

    Using ax.axvline(ymax=y_frac) draws a line that extends from y=0
    up to the fraction y_frac of the axes height.  This is cleaner than
    computing data coordinates because we don't need to know the y-axis range.

    R / ggplot2 equivalent:
        annotate("segment", x=year, xend=year, y=0, yend=label_y) +
        annotate("point", x=year, y=visitors) +
        annotate("text", x=year, y=label_y, label=..., hjust=..., vjust=...)
    """
    year  = event["year"]
    color = event["color"]
    ha    = event["ha"]
    y_frac = event["y_frac"]

    # Look up the actual visitor count for this year
    # R: df$visitors[df$year == year]
    row = df[df["year"] == year]
    if row.empty:
        return  # year not in dataset — skip gracefully
    visitors = row["visitors"].iloc[0]  # .iloc[0] is like [[1]] in R (first element)

    # Full-height vertical dotted marker line.
    # Using full height (no ymin/ymax) is cleaner: the dot on the line shows
    # the actual data value; the label floats at a pre-set y position.
    ax.axvline(
        x=year,
        color=color, linewidth=0.9, linestyle=":", alpha=0.55, zorder=2,
    )

    # Dot at the actual data point on the line
    # edgecolors="white" + linewidths gives a halo effect for legibility
    ax.scatter(
        year, visitors,
        color=color, s=60, zorder=7,
        edgecolors="white", linewidths=1.2,
        clip_on=False,
    )

    # Text label — positioned at y_frac of the y-axis height (data coords)
    # We convert the axis fraction to a data coordinate for ax.text()
    y_min_data = 0
    label_y = y_min_data + y_frac * (y_max_data - y_min_data)

    # Horizontal offset: push text away from the marker line by ~1.5 years
    x_nudge = -1.8 if ha == "right" else 1.8

    ax.text(
        year + x_nudge, label_y,
        event["label"],
        color=color,
        fontsize=8.0,
        ha=ha, va="center",
        linespacing=1.35,
        zorder=8,
        # fontweight="medium",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main chart function
# ─────────────────────────────────────────────────────────────────────────────

def create_chart(df: pd.DataFrame) -> None:
    """
    Build and save the Instagram-ready 1080 × 1080 chart.

    Layout (in figure-normalised coordinates, 0 = bottom, 1 = top):
        0.88–0.98   Title text
        0.80–0.88   Subtitle text
        0.77–0.80   Decorative rule
        0.12–0.77   Plot axes (the actual chart)
        0.02–0.10   Footer (source + handle)
    """

    # ── Figure setup ─────────────────────────────────────────────────────────
    # 1080 ÷ 150 DPI = 7.2 inches.  This produces exactly 1080 × 1080 pixels.
    # R equivalent: png("output.png", width=1080, height=1080, res=150)
    DPI = 150
    PX  = 1080
    fig = plt.figure(
        figsize=(PX / DPI, PX / DPI),
        dpi=DPI,
        facecolor=PAL["bg"],
    )

    # Add the main plot axes at a specific position within the figure.
    # [left, bottom, width, height] as fractions of the figure (0–1).
    # matplotlib's fig.add_axes() gives us precise control — more explicit
    # than ggplot2's automatic layout, but more flexible for custom designs.
    ax = fig.add_axes([0.12, 0.12, 0.84, 0.63])
    ax.set_facecolor(PAL["bg"])

    # ── Data prep ────────────────────────────────────────────────────────────
    # Sort and extract arrays for plotting.
    # pandas note: .values returns a numpy array — equivalent to as.numeric(df$col) in R.
    df_sorted = df.sort_values("year").reset_index(drop=True)
    years     = df_sorted["year"].astype(int).values
    visitors  = df_sorted["visitors"].values

    # ── Y-axis range ─────────────────────────────────────────────────────────
    # Give 22% headroom above the max for annotation labels to float in.
    y_ceil = df_sorted["visitors"].max() * 1.22

    ax.set_xlim(1964, 2027)
    ax.set_ylim(0, y_ceil)

    # ── Area fill (geom_area equivalent) ─────────────────────────────────────
    # fill_between() fills the area between two y-curves.
    # Here: between y=0 and y=visitors — like ggplot2's geom_area().
    ax.fill_between(
        years, 0, visitors,
        color=PAL["fill"],
        alpha=PAL["fill_alpha"],
        zorder=1,
    )

    # Subtle gradient effect: a second, narrower fill near the line
    ax.fill_between(
        years, visitors * 0.92, visitors,
        color=PAL["fill"],
        alpha=PAL["fill_alpha"] * 0.8,
        zorder=1,
    )

    # ── Main line (geom_line equivalent) ─────────────────────────────────────
    # solid_capstyle/joinstyle make the line ends and corners round
    # (like lineend="round" in ggplot2's geom_line).
    ax.plot(
        years, visitors,
        color=PAL["line"],
        linewidth=2.4,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    # ── Backcountry context — shaded capacity band ────────────────────────────
    # A subtle horizontal shaded region contextualises the visitor numbers
    # against the park's extremely limited backcountry capacity.
    # The NPS issues ~60 backcountry permits per day in summer (≈ 120 days),
    # totalling ~7,200 summer backcountry nights for ~2–3M annual visitors.
    #
    # We shade from the 2019 pre-COVID level upward to the top of the chart
    # to visually flag the "oversubscribed" zone.
    pre_covid_ref = 2_248_518   # actual 2019 MORA recreation visitors (last pre-COVID year)
    ax.axhspan(
        pre_covid_ref, y_ceil,
        alpha=PAL["bc_alpha"] * 0.6,
        color=PAL["bc_fill"],
        zorder=0,
        linewidth=0,
    )
    # Horizontal reference line at the pre-COVID level
    ax.axhline(
        pre_covid_ref,
        color=PAL["subtext"],
        linewidth=0.7,
        linestyle=(0, (5, 4)),   # custom dash pattern — like lty in R
        alpha=0.5,
        zorder=2,
    )

    # Backcountry callout text box — upper-right corner of the axes
    # transform=ax.transAxes means (0,0)=bottom-left, (1,1)=top-right of the axes.
    # This is like placing text using "npc" coordinates in R's grid graphics.
    bbox_style = dict(
        boxstyle="round,pad=0.45",
        facecolor=PAL["bg"],
        edgecolor=PAL["bc_fill"],
        alpha=0.90,
        linewidth=1.0,
    )
    ax.text(
        0.985, 0.97,
        "Oversubscribed zone\n"
        "~2.5M visitors/yr pass through\n"
        "a park issuing ~60 backcountry\n"
        "permits per day in summer",
        transform=ax.transAxes,          # axes-fraction coordinates
        fontsize=7.4,
        color=PAL["subtext"],
        ha="right", va="top",
        linespacing=1.40,
        bbox=bbox_style,
        zorder=9,
    )

    # ── Event annotations ─────────────────────────────────────────────────────
    for event in EVENTS:
        draw_event(ax, df_sorted, event, y_max_data=y_ceil)

    # ── Axes formatting ───────────────────────────────────────────────────────

    # X-axis ticks at each decade + the final year
    # R: scale_x_continuous(breaks = c(seq(1970, 2020, 10), 2024))
    x_ticks      = list(range(1970, 2025, 10)) + [2024]
    x_tick_labels = [str(y) for y in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=9, color=PAL["subtext"])

    # Y-axis: one tick per 500K visitors, formatted as "1M", "1.5M", etc.
    # FuncFormatter takes a function (value, position) → string.
    # R: scale_y_continuous(labels = scales::label_number(scale=1e-6, suffix="M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(500_000))
    ax.tick_params(axis="y", labelsize=9, colors=PAL["subtext"])

    # Grid: horizontal lines only (FiveThirtyEight aesthetic)
    # R: theme(panel.grid.major.x = element_blank())
    ax.grid(axis="y", color=PAL["grid"], linewidth=0.6, linestyle="--", zorder=0)
    ax.grid(axis="x", visible=False)

    # Spine visibility
    ax.spines["left"].set_color(PAL["spine"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(PAL["spine"])
    ax.spines["bottom"].set_linewidth(0.8)

    # Y-axis label — sits to the left of the tick labels
    # R: labs(y = "Annual recreation visitors")
    ax.set_ylabel(
        "Annual recreation visitors",
        color=PAL["subtext"],
        fontsize=9,
        labelpad=7,
    )

    # ── All-time high callout (direct label on the line) ─────────────────────
    # Rather than a generic "last data point" callout, find the actual max
    # and annotate it — this emphasises the headline of the chart.
    peak_idx  = int(np.argmax(visitors))
    peak_year = int(years[peak_idx])
    peak_val  = visitors[peak_idx]
    # Only add this label if the 2023 EVENT annotation won't clash
    # (the EVENT already labels 2023; skip if it's the same year)
    ev_years = {ev["year"] for ev in EVENTS}
    if peak_year not in ev_years:
        ax.annotate(
            f"All-time high: {peak_val / 1e6:.2f}M ({peak_year})",
            xy=(peak_year, peak_val),
            xytext=(peak_year - 4, peak_val * 1.07),
            color=PAL["ev_peak"],
            fontsize=8.5,
            fontweight="bold",
            ha="right",
            arrowprops=dict(arrowstyle="-", color=PAL["ev_peak"], lw=0.8),
            zorder=10,
        )

    # ── Title area (above the axes) ───────────────────────────────────────────
    # We use fig.text() rather than ax.set_title() so we can position the
    # text precisely in figure coordinates (independent of the axes).
    # R equivalent: this level of layout control is easiest in cowplot or patchwork.

    # Decorative colour accent — a short thick rule ABOVE the title text.
    # This is a common editorial header pattern (NYT, FiveThirtyEight).
    # We place it at y=0.975 — clearly in the top margin, well above the chart.
    # The axes top is at y≈0.75 in figure-fraction coords, so y=0.975 is
    # unambiguously part of the title block, not the data area.
    fig.add_artist(
        plt.Line2D(
            [0.12, 0.12 + 0.05],   # x: short horizontal rule
            [0.975, 0.975],          # y: top margin, above the title text
            transform=fig.transFigure,
            color=PAL["line"],
            linewidth=4,
            solid_capstyle="butt",
        )
    )

    # Main title
    fig.text(
        0.12, 0.955,
        "MOUNT RAINIER IS MORE POPULAR THAN EVER",
        fontsize=15.5,
        fontweight="bold",
        color=PAL["text"],
        ha="left", va="top",
        linespacing=1.1,
        transform=fig.transFigure,
    )

    # Subtitle
    fig.text(
        0.12, 0.905,
        "Annual recreation visitors to Mount Rainier National Park, 1967–2025.\n"
        "The park draws millions — but backcountry access has barely changed.",
        fontsize=9.5,
        color=PAL["subtext"],
        ha="left", va="top",
        linespacing=1.45,
        transform=fig.transFigure,
    )

    # Thin horizontal rule between header and chart
    fig.add_artist(
        plt.Line2D(
            [0.12, 0.955],
            [0.808, 0.808],
            transform=fig.transFigure,
            color=PAL["spine"],
            linewidth=0.7,
        )
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    fig.text(
        0.12, 0.048,
        "Source: NPS Visitor Use Statistics  ·  irma.nps.gov",
        fontsize=7.8,
        color=PAL["subtext"],
        ha="left", va="bottom",
        transform=fig.transFigure,
    )
    fig.text(
        0.955, 0.048,
        "@readingatabar",
        fontsize=8.5,
        color=PAL["handle"],
        fontweight="bold",
        ha="right", va="bottom",
        transform=fig.transFigure,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    # We save to a temporary buffer first, then use Pillow to guarantee the
    # final output is exactly 1080 × 1080 pixels (Instagram native resolution).
    #
    # Why not just fig.savefig() directly?
    #   bbox_inches="tight" adjusts the bounding box to fit all text, which
    #   makes the final size unpredictable.  Without it, fig annotations
    #   near the edges can be clipped.  The Pillow pad-to-square approach
    #   gives us the best of both worlds: tight text fitting + exact pixel size.
    #
    # R equivalent: ggsave(file, plot, width=7.2, height=7.2, dpi=150)
    #   then magick::image_resize(img, "1080x1080!") if exact dims are needed.
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor=PAL["bg"],
        edgecolor="none",
    )
    plt.close(fig)   # Free memory — R equivalent: dev.off()
    buf.seek(0)

    # Open the rendered image and pad to a square with the background colour
    rendered = Image.open(buf)
    w, h     = rendered.size
    side     = max(w, h)

    # Convert background hex colour to RGB tuple for Pillow
    bg_r = int(PAL["bg"][1:3], 16)
    bg_g = int(PAL["bg"][3:5], 16)
    bg_b = int(PAL["bg"][5:7], 16)

    square = Image.new("RGB", (side, side), (bg_r, bg_g, bg_b))
    square.paste(rendered.convert("RGB"), ((side - w) // 2, (side - h) // 2))

    # Resize to exactly 1080 × 1080 using high-quality Lanczos resampling
    final = square.resize((PX, PX), Image.LANCZOS)
    final.save(OUTPUT_FILE, dpi=(DPI, DPI))

    print(f"[done]  Chart saved → {OUTPUT_FILE}")
    from PIL import Image as _I
    _img = _I.open(OUTPUT_FILE)
    print(f"        {OUTPUT_FILE.stat().st_size / 1024:.0f} KB  |  {DPI} DPI  |  {_img.size[0]} × {_img.size[1]} px")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 58)
    print("  Mount Rainier — Visualization")
    print("=" * 58)

    # ── Load cleaned data ─────────────────────────────────────────────────────
    if not CLEANED_CSV.exists():
        print(
            f"\n[error] Cleaned data not found at:\n"
            f"        {CLEANED_CSV}\n\n"
            f"        Run 'python scripts/fetch_data.py' first."
        )
        sys.exit(1)

    # R equivalent: df <- readr::read_csv(path)
    df = pd.read_csv(CLEANED_CSV)
    print(f"[data]  Loaded {len(df)} rows from {CLEANED_CSV.name}")

    # Quick data check
    # R: stopifnot("year" %in% names(df), "visitors" %in% names(df))
    for col in ("year", "visitors"):
        if col not in df.columns:
            print(f"[error] Required column '{col}' missing. Re-run fetch_data.py.")
            sys.exit(1)

    # Drop any remaining nulls in key columns
    df = df.dropna(subset=["year", "visitors"]).copy()
    df["year"] = df["year"].astype(int)

    print(f"[data]  Year range: {df['year'].min()}–{df['year'].max()}")
    print(f"[data]  Visitors — min: {df['visitors'].min():,.0f}  max: {df['visitors'].max():,.0f}")

    # ── Print a mini data story to the terminal ────────────────────────────────
    peak_row = df.loc[df["visitors"].idxmax()]
    print(f"\n[story] All-time high: {int(peak_row['visitors']):,} visitors in {int(peak_row['year'])}")

    covid_row = df[df["year"] == 2020]
    if not covid_row.empty:
        print(f"[story] COVID year 2020: {int(covid_row['visitors'].iloc[0]):,} visitors")

    # ── Generate chart ────────────────────────────────────────────────────────
    create_chart(df)


if __name__ == "__main__":
    main()
