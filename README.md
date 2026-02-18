# weekly-viz-dump
Creating one polished visualization every week.

---

## Week 1 — Mount Rainier Visitation (1967–2024)

> **"Mount Rainier is more popular than ever — but the park's backcountry capacity hasn't kept up."**

An Instagram-ready time series chart of annual recreation visitor counts at Mount Rainier National Park, with annotations for key historical moments: the Mt. St. Helens eruption, the COVID-19 closure, the post-COVID surge to all-time highs, and the 2024 timed-entry pilot.

This project is also a **learning exercise** for experienced R users picking up Python for data visualization. Every step includes R → Python translation notes in the source code comments.

### Output

`output/rainier_visitors.png` — 1080 × 1080 px, 150 DPI, Instagram-ready.

---

### Data source

**NPS Visitor Use Statistics**
- Primary (scraped): [Mount Rainier Annual Visitation page](https://www.nps.gov/mora/learn/management/annual-visitation.htm) — HTML table, 1967–2024.
- Fallback (download): Full NPS data package via [data.gov / IRMA](https://irma.nps.gov/DataStore/DownloadFile/753817?Reference=2316688) — all parks, 1979–2024, filtered to park code `MORA`.

Raw files are cached in `data/raw/` (gitignored). Cleaned output is committed at `data/cleaned/rainier_visitation.csv`.

---

### Project structure

```
weekly-viz-dump/
├── data/
│   ├── raw/                  ← gitignored; cached downloads
│   └── cleaned/
│       └── rainier_visitation.csv
├── scripts/
│   ├── fetch_data.py         ← data acquisition + cleaning
│   └── visualize.py          ← chart generation
├── output/                   ← gitignored; regenerable images
├── .gitignore
├── requirements.txt
└── README.md
```

---

### How to reproduce

**1. Set up a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv) (faster):

```bash
uv venv && uv pip install -r requirements.txt
source .venv/bin/activate
```

**2. Fetch and clean the data**

```bash
python scripts/fetch_data.py
```

This tries the NPS HTML scrape first, then falls back to the full data package download. Raw files are cached in `data/raw/`; the cleaned output is written to `data/cleaned/rainier_visitation.csv`.

To force a fresh download (ignore cache):

```bash
python scripts/fetch_data.py --refresh
```

**3. Generate the visualization**

```bash
python scripts/visualize.py
```

Output: `output/rainier_visitors.png`

---

### R → Python translation highlights

| Task | R | Python |
|---|---|---|
| Project-relative paths | `here::here("data", "raw")` | `Path(__file__).resolve().parent.parent / "data" / "raw"` |
| Read HTML table | `rvest::read_html(url) \|> html_table()` | `pd.read_html(resp.text)` |
| Clean column names | `janitor::clean_names()` | custom `clean_names()` using `.str` accessor on `pd.Index` |
| Parse numbers with commas | `readr::parse_number()` | custom `parse_number()` with `pd.to_numeric(errors="coerce")` |
| Filter rows | `dplyr::filter(df, col == val)` | `df[df["col"] == val]` |
| Group & summarise | `group_by(year) \|> summarise(...)` | `df.groupby("year").sum()` |
| Add columns | `dplyr::mutate(...)` | `df["new_col"] = expression` |
| Assert / test | `stopifnot(all(!is.na(df$year)))` | `assert df["year"].notna().all()` |
| Save CSV | `readr::write_csv(df, path)` | `df.to_csv(path, index=False)` |
| Plot area fill | `geom_area()` | `ax.fill_between(x, 0, y)` |
| Plot line | `geom_line()` | `ax.plot(x, y)` |
| Format y-axis | `scale_y_continuous(labels=label_number(...))` | `ax.yaxis.set_major_formatter(FuncFormatter(fn))` |
| Save figure | `ggsave(file, width=7.2, height=7.2, dpi=150)` | `fig.savefig(file, dpi=150, bbox_inches="tight")` |

---

### Design choices

- **Format**: Square 1080 × 1080 (native Instagram grid size)
- **Palette**: Pacific Northwest — forest greens (`#2D6A4F`), slate blues (`#5E6E7E`), warm mountain grays (`#F4F1EC`)
- **Aesthetic**: FiveThirtyEight / The Pudding — clean grid, direct labelling, story-first typography
- **Annotations**: Mt. St. Helens eruption (1980), early-'90s peak, COVID-19 closure (2020), post-COVID all-time high (2023), timed-entry pilot (2024)
- **Backcountry context**: Shaded band + callout box highlighting the gap between ~2.5M annual visitors and the park's ~60 permitted backcountry overnights per day in summer

---

*Data: NPS Visitor Use Statistics · Visualization: @bentwheel*
