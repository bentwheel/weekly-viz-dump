"""
scripts/fetch_data.py
=====================
Acquire and clean Mount Rainier National Park annual visitation data.

Two acquisition strategies (with automatic fallback):
  PRIMARY   — Scrape the NPS annual visitation HTML page for MORA.
  FALLBACK  — Download the full NPS data package CSV from data.gov,
              then filter to MORA (park code "MORA").

Both strategies cache their raw output in data/raw/ so subsequent runs
are instant. Use --refresh to force a re-download.

Usage:
    python scripts/fetch_data.py             # uses cache when available
    python scripts/fetch_data.py --refresh   # forces re-download

─────────────────────────────────────────────────────────────────────────────
R → Python translation guide (woven through comments below)
─────────────────────────────────────────────────────────────────────────────
  R                           Python / pandas equivalent
  ─────────────────────────── ────────────────────────────────────────────
  library(here)               pathlib.Path(__file__).resolve().parent
  dir.create(recursive=TRUE)  Path.mkdir(parents=True, exist_ok=True)
  read_html(url) / html_table pandas.read_html()
  readr::read_csv()           pandas.read_csv()
  dplyr::filter()             df[boolean_mask]  or  df.query()
  dplyr::select()             df[["col1", "col2"]]
  dplyr::mutate()             df["new_col"] = expression
  dplyr::group_by() |>        df.groupby(col).agg(...)
    summarise()
  janitor::clean_names()      custom clean_names() below
  readr::parse_number()       custom parse_number() below
  str(df) / glimpse(df)       df.info() / df.head() / df.dtypes
  summary(df)                 df.describe()
  stopifnot()                 assert condition, "message"
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import io
import zipfile
import argparse
from pathlib import Path

import requests
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
# Path(__file__) is this script's location.  .resolve() canonicalises it
# (resolves symlinks, makes absolute).  .parent gives the scripts/ directory;
# .parent.parent climbs up to the project root.
# R equivalent: here::here()  (or rprojroot::find_package_root_file())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR  = PROJECT_ROOT / "data" / "cleaned"

# mkdir(parents=True) ≈ R's dir.create(path, recursive = TRUE)
# exist_ok=True means no error if it already exists (≈ showWarnings = FALSE)
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NPS_HTML_URL    = "https://www.nps.gov/mora/learn/management/annual-visitation.htm"
NPS_PACKAGE_URL = "https://irma.nps.gov/DataStore/DownloadFile/753817?Reference=2316688"
MORA_CODE       = "MORA"

# Cached file paths
RAW_SCRAPED  = RAW_DIR / "mora_annual_scraped.csv"
RAW_PACKAGE  = RAW_DIR / "nps_visitation_all_parks.csv"
CLEANED_OUT  = CLEANED_DIR / "rainier_visitation.csv"

# Polite HTTP headers — always identify your scraper
HEADERS = {"User-Agent": "Mozilla/5.0 (research; rainier-viz-project/1.0; non-commercial)"}


# =============================================================================
# STRATEGY 1 — Scrape the NPS HTML page
# =============================================================================

def scrape_nps_html(force_refresh: bool = False) -> pd.DataFrame | None:
    """
    Fetch Mount Rainier's annual visitation HTML table from the NPS website.

    Python equivalent of:
        rvest::read_html(url) |> rvest::html_table() |> purrr::pluck(1)

    We use pandas.read_html(), which:
      1. Fetches the HTML (via requests under the hood, or we pass the text)
      2. Finds all <table> elements and parses each into a DataFrame
      3. Returns a Python *list* of DataFrames

    This is conceptually similar to rvest::html_nodes("table") |> html_table(),
    but in one call and without needing a separate rvest library.
    """
    # Cache check — avoids hammering the NPS server on every run
    # R equivalent: if (file.exists(path)) return(read_csv(path))
    if RAW_SCRAPED.exists() and not force_refresh:
        print(f"[cache] Using cached scrape data: {RAW_SCRAPED.name}")
        return pd.read_csv(RAW_SCRAPED)

    print(f"[fetch] Scraping: {NPS_HTML_URL}")
    try:
        # requests.get() ≈ httr2::request(url) |> httr2::req_perform()
        # timeout= prevents hanging forever on a slow connection
        resp = requests.get(NPS_HTML_URL, headers=HEADERS, timeout=30)
        # raise_for_status() throws an exception on 4xx/5xx HTTP errors
        # R equivalent: httr::stop_for_status(resp)
        resp.raise_for_status()
    except requests.RequestException as e:
        # Python uses try/except where R uses tryCatch()
        print(f"[warn]  Scrape request failed: {e}")
        return None

    try:
        # pandas.read_html() is the Pythonic way to parse HTML tables.
        # thousands="," tells it to handle American number formatting
        # (e.g., "1,234,567") automatically — like readr::locale(grouping_mark=",")
        # We pass io.StringIO(resp.text) to read from the string, not a URL,
        # since we already have the HTML from our requests call.
        tables = pd.read_html(io.StringIO(resp.text), thousands=",")
    except ValueError as e:
        print(f"[warn]  No <table> elements found on page: {e}")
        return None

    if not tables:
        print("[warn]  read_html() returned an empty list.")
        return None

    # The NPS visitation page has exactly one data table — we take index [0].
    # Python lists are zero-indexed (unlike R's 1-indexed lists).
    # In R: tables[[1]]
    df = tables[0]
    print(f"[info]  Scraped table — shape: {df.shape} | columns: {list(df.columns)}")

    # Save raw result before any cleaning (good practice: separate raw from clean)
    df.to_csv(RAW_SCRAPED, index=False)
    print(f"[cache] Saved raw scrape → {RAW_SCRAPED.name}")
    return df


# =============================================================================
# STRATEGY 2 — Download full NPS data package, filter to MORA
# =============================================================================

def download_nps_package(force_refresh: bool = False) -> pd.DataFrame | None:
    """
    Download the complete NPS visitation data package (all parks, 1979–2024)
    and filter it to Mount Rainier (park code MORA).

    Python equivalent of:
        df_all <- readr::read_csv(url)
        df     <- dplyr::filter(df_all, UnitCode == "MORA")

    The file may be 50–200 MB and could be either a plain CSV or a ZIP archive.
    We handle both cases.
    """
    if not RAW_PACKAGE.exists() or force_refresh:
        print(f"[fetch] Downloading NPS data package (may be large — please wait)…")
        try:
            # stream=True means we download in chunks instead of loading the
            # entire file into memory at once — important for large files.
            # R equivalent: download.file(url, destfile, method="libcurl")
            resp = requests.get(
                NPS_PACKAGE_URL, headers=HEADERS,
                timeout=120, stream=True
            )
            resp.raise_for_status()

            total_bytes = int(resp.headers.get("content-length", 0))
            downloaded  = 0

            with open(RAW_PACKAGE, "wb") as fh:
                # iter_content() yields chunks of raw bytes
                for chunk in resp.iter_content(chunk_size=65_536):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = downloaded / total_bytes * 100
                        # \r (carriage return) overwrites the current line for a progress bar
                        print(f"\r        {pct:5.1f}%  ({downloaded / 1e6:.1f} / {total_bytes / 1e6:.1f} MB)",
                              end="", flush=True)
            print()  # newline after progress bar
            print(f"[cache] Saved → {RAW_PACKAGE.name}")

        except requests.RequestException as e:
            print(f"[warn]  Download failed: {e}")
            return None

    # Read the downloaded file — handle both plain CSV and ZIP formats
    print(f"[read]  Parsing {RAW_PACKAGE.name}…")
    try:
        if zipfile.is_zipfile(RAW_PACKAGE):
            # The file is a ZIP archive — find the largest CSV inside
            with zipfile.ZipFile(RAW_PACKAGE) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    print("[warn]  ZIP contains no CSV files.")
                    return None
                # max() with key= ≈ R's which.max() — pick the file with largest size
                target = max(csv_names, key=lambda n: zf.getinfo(n).file_size)
                print(f"[info]  Reading '{target}' from ZIP archive")
                with zf.open(target) as f:
                    df_all = pd.read_csv(f, encoding="latin1", low_memory=False)
        else:
            # Plain CSV — try UTF-8 first, then fall back to Latin-1
            # Government datasets often use Latin-1 (ISO-8859-1) encoding
            try:
                df_all = pd.read_csv(RAW_PACKAGE, low_memory=False)
            except UnicodeDecodeError:
                df_all = pd.read_csv(RAW_PACKAGE, encoding="latin1", low_memory=False)

    except Exception as e:
        print(f"[warn]  Could not read downloaded file: {e}")
        return None

    print(f"[info]  Full dataset — shape: {df_all.shape}")

    # Find the park-code column — the name varies slightly across NPS releases
    code_col = _find_column(df_all.columns, ["unitcode", "unit_code", "parkcode", "park_code"])
    if code_col is None:
        print(f"[warn]  Could not find park-code column. Available: {list(df_all.columns)}")
        return None

    # Filter to MORA using boolean indexing
    # R equivalent: dplyr::filter(df_all, UnitCode == "MORA")
    # pandas note: .copy() prevents a SettingWithCopyWarning when we later
    # modify the filtered DataFrame — always copy after filtering.
    mora_df = df_all[df_all[code_col].astype(str).str.strip() == MORA_CODE].copy()
    print(f"[info]  MORA rows found: {len(mora_df)}")
    return mora_df


# =============================================================================
# Column-finding utility
# =============================================================================

def _find_column(columns: pd.Index, keywords: list[str]) -> str | None:
    """
    Case-insensitive column search by keyword list.

    R equivalent: dplyr::select(df, contains("unitcode"))
    Returns the original column name (with original casing) if found.
    """
    # Build a lookup: normalised_name → original_name
    normalised = {
        c.lower().replace(" ", "_").replace("-", "_"): c
        for c in columns
    }
    # Try exact keyword match first
    for kw in keywords:
        if kw in normalised:
            return normalised[kw]
    # Fall back to substring search
    for kw in keywords:
        for norm, orig in normalised.items():
            if kw in norm:
                return orig
    return None


# =============================================================================
# CLEANING utilities — shared by both strategies
# =============================================================================

def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise all column names to lowercase snake_case.

    R equivalent: janitor::clean_names()

    Python note: pd.Index.str is the vectorised string accessor for Index
    objects — like purrr::map_chr(names(df), str_to_lower) but faster.
    regex=True in str.replace() enables regular expression syntax.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)  # non-alphanumeric → underscore
        .str.strip("_")                                 # trim leading/trailing underscores
    )
    return df


def parse_number(series: pd.Series) -> pd.Series:
    """
    Strip numeric formatting characters (commas, %, +, whitespace) then
    coerce to float, turning unparseable values into NaN.

    R equivalent: readr::parse_number()

    Key Python points:
    - .str methods on a Series are vectorised (no for-loop needed)
    - pd.to_numeric(errors="coerce") silently converts failures to NaN,
      equivalent to suppressWarnings(as.numeric(x)) in R
    - .replace() here is pandas Series.replace(), not str.replace()
    """
    cleaned = (
        series
        .astype(str)
        .str.replace(r"[,+%\s]", "", regex=True)   # strip formatting
        .str.replace("—",  "",   regex=False)       # em-dash (used for missing)
        .str.replace("N/A", "",  regex=False)
        .str.replace("n/a", "",  regex=False)
        .replace("", float("nan"))                  # empty string → NaN
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_scraped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw HTML-scraped table into a tidy annual DataFrame.

    Expected raw columns (from the NPS page):
        Year | Number of Vehicles | Number of Visitors | Percent Change
    After clean_names():
        year | number_of_vehicles | number_of_visitors | percent_change

    The NPS page HTML often has no <thead> element, so pandas.read_html()
    assigns integer column headers (0, 1, 2, 3).  In that case, the first
    data row contains the real column names — we promote it to the header.
    R equivalent: this rarely happens in rvest, which handles <thead> implicitly.
    """
    # Detect headerless table: all columns are integers (or their string form)
    if all(isinstance(c, int) or str(c).isdigit() for c in df.columns):
        df = df.copy()
        df.columns = [str(v) for v in df.iloc[0]]   # promote row 0 → column names
        df = df.iloc[1:].reset_index(drop=True)      # drop the now-redundant row 0
        print("[clean] Promoted first row to column names (headerless table detected)")

    df = clean_names(df)
    print(f"[clean] Columns after clean_names(): {list(df.columns)}")

    # Flexible column detection handles minor NPS page redesigns
    year_col    = _find_column(df.columns, ["year"])
    visitor_col = _find_column(df.columns, ["number_of_visitors", "visitors", "recreationvisitors"])
    vehicle_col = _find_column(df.columns, ["number_of_vehicles", "vehicles"])
    pct_col     = _find_column(df.columns, ["percent_change", "percent", "change"])

    if not year_col:
        raise ValueError(f"Cannot find 'year' column. Got: {list(df.columns)}")
    if not visitor_col:
        raise ValueError(f"Cannot find 'visitors' column. Got: {list(df.columns)}")

    # Build the output DataFrame column-by-column
    # R equivalent: transmute(year = parse_number(year), visitors = parse_number(visitors), ...)
    out = pd.DataFrame()

    # Int64 (capital I) is pandas' nullable integer type.
    # Unlike int64, it can hold NA values — equivalent to R's integer NA.
    out["year"]     = parse_number(df[year_col]).astype("Int64")
    out["visitors"] = parse_number(df[visitor_col])

    if vehicle_col:
        out["vehicles"] = parse_number(df[vehicle_col])
    if pct_col:
        out["pct_change"] = parse_number(df[pct_col])

    # Drop non-data rows: summary rows like "Total" or "Average" will have
    # non-numeric (NaN) years after parse_number().
    # R equivalent: dplyr::filter(!is.na(year), year > 1900)
    out = out[out["year"].notna() & (out["year"] > 1900)].copy()

    return out.sort_values("year").reset_index(drop=True)


def clean_package(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the MORA-filtered package data into a tidy annual DataFrame.

    The NPS IRMA data package uses a LONG format where each row is one
    statistic for one park-year-month, with columns:
        UnitCode | Year | Month | Statistic | Value

    NPS statistic codes relevant here:
        TRV  = Total Recreation Visitors  ← this is what we want
        TNRV = Total Non-Recreation Visitors
        TV   = Total Visitors (TRV + TNRV)

    R equivalent for this long→wide pivot + filter:
        df |>
            filter(Statistic == "TRV") |>
            group_by(Year) |>
            summarise(visitors = sum(Value, na.rm = TRUE))

    If the data is already in wide format (visitor count as a column), we
    fall back to column-name-based detection.
    """
    df = clean_names(df)
    print(f"[clean] Package columns after clean_names(): {list(df.columns)}")

    year_col      = _find_column(df.columns, ["year"])
    statistic_col = _find_column(df.columns, ["statistic"])
    value_col     = _find_column(df.columns, ["value"])

    if not year_col:
        raise ValueError(f"Cannot find year column. Got: {list(df.columns)}")

    # ── Long format (Statistic + Value columns) ───────────────────────────────
    # The NPS IRMA download uses this tall format — one metric per row.
    if statistic_col and value_col:
        print(f"[info]  Long format detected ('{statistic_col}' + '{value_col}' columns)")
        # Show available codes to aid future debugging
        codes = df[statistic_col].unique().tolist()
        print(f"[info]  Available statistic codes: {codes}")

        # TRV = Total Recreation Visitors (the standard NPS code)
        # Filter: pandas boolean indexing ≈ dplyr::filter(Statistic == "TRV")
        trv = df[df[statistic_col].astype(str).str.upper() == "TRV"].copy()

        if trv.empty:
            # Fallback: try TV (Total Visitors) if TRV not found
            print("[warn]  TRV not found — trying TV (Total Visitors) instead")
            trv = df[df[statistic_col].astype(str).str.upper() == "TV"].copy()

        if trv.empty:
            raise ValueError(f"No TRV or TV statistic found. Codes present: {codes}")

        trv[year_col]  = parse_number(trv[year_col]).astype("Int64")
        trv[value_col] = parse_number(trv[value_col])

        print(f"[info]  TRV rows: {len(trv)}  |  Aggregating monthly → annual")
        # Sum monthly TRV values to get annual total
        # R: trv |> group_by(Year) |> summarise(visitors = sum(Value, na.rm = TRUE))
        annual = (
            trv
            .groupby(year_col, as_index=False)[value_col]
            .sum()
            .rename(columns={year_col: "year", value_col: "visitors"})
        )

    # ── Wide format (visitor count already a column) ──────────────────────────
    else:
        visitor_col = _find_column(df.columns, [
            "recreation_visitors", "recreationvisitors",
            "rec_visitors", "visitors"
        ])
        if not visitor_col:
            raise ValueError(f"Cannot find visitor column. Got: {list(df.columns)}")

        df = df.copy()
        df[year_col]    = parse_number(df[year_col]).astype("Int64")
        df[visitor_col] = parse_number(df[visitor_col])

        month_col = _find_column(df.columns, ["month"])
        if month_col:
            print(f"[info]  Monthly data in '{month_col}' — aggregating to annual")

        annual = (
            df
            .groupby(year_col, as_index=False)[visitor_col]
            .sum()
            .rename(columns={year_col: "year", visitor_col: "visitors"})
        )

    annual = annual[annual["year"] > 1900].copy()
    return annual.sort_values("year").reset_index(drop=True)


# =============================================================================
# Derived columns
# =============================================================================

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional metrics for richer analysis and visualisation.

    R equivalent:
        df |>
            mutate(
                visitors_per_vehicle = visitors / vehicles,
                yoy_pct_change       = (visitors / lag(visitors) - 1) * 100
            )
    """
    df = df.copy()

    # Visitors-per-vehicle: a proxy for average group/family size over time.
    # If vehicles has risen faster than visitors, groups are getting smaller.
    if "vehicles" in df.columns and df["vehicles"].notna().any():
        df["visitors_per_vehicle"] = df["visitors"] / df["vehicles"]

    # Year-over-year percent change (compute only if not already scraped)
    if "pct_change" not in df.columns:
        # DataFrame.pct_change() computes (x[i] - x[i-1]) / x[i-1].
        # Multiply by 100 to get percentage points.
        # R equivalent: (visitors / dplyr::lag(visitors) - 1) * 100
        df["pct_change"] = df["visitors"].pct_change() * 100

    return df


# =============================================================================
# Data validation
# =============================================================================

def validate(df: pd.DataFrame) -> None:
    """
    Run sanity checks on the cleaned DataFrame and print a summary report.

    R equivalent: stopifnot() + summary() + testthat::expect_*()

    Python's assert statement raises AssertionError (with an optional message)
    if the condition is False — equivalent to stopifnot(condition) in R.
    """
    print("\n[validate] ── Quality Checks ─────────────────────────────────────")
    print(f"  Shape:         {df.shape[0]} rows × {df.shape[1]} cols")  # R: dim(df)
    print(f"  Year range:    {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"  Visitor range: {df['visitors'].min():,.0f} – {df['visitors'].max():,.0f}")
    print(f"\n  Null counts per column:")
    # R equivalent: colSums(is.na(df))
    for col, n in df.isnull().sum().items():
        status = "✓" if n == 0 else f"⚠  {n} nulls"
        print(f"    {col:<25} {status}")

    # Assertions — any failure raises AssertionError and stops execution
    assert df["year"].notna().all(),      "Found null year values — check raw data!"
    assert df["visitors"].notna().all(),  "Found null visitor counts — check raw data!"
    assert (df["year"] >= 1960).all(),    "Found suspiciously old year (< 1960)"
    assert (df["visitors"] > 0).all(),    "Found zero or negative visitor count"
    assert df["year"].is_monotonic_increasing, "Years are not sorted ascending"

    print("\n[validate] All checks passed ✓")


# =============================================================================
# MAIN
# =============================================================================

def main(force_refresh: bool = False) -> None:
    print("=" * 62)
    print("  Mount Rainier National Park — Data Acquisition & Cleaning")
    print("=" * 62)

    df: pd.DataFrame | None = None

    # ── Strategy 1: Scrape NPS HTML page ─────────────────────────────────────
    print("\n▶ Strategy 1: Scrape NPS annual visitation HTML page")
    raw = scrape_nps_html(force_refresh=force_refresh)
    if raw is not None:
        try:
            df = clean_scraped(raw)
            print(f"[ok]    Scrape strategy succeeded — {len(df)} annual rows.")
        except Exception as e:
            print(f"[warn]  Cleaning after scrape failed: {e}")
            df = None

    # ── Strategy 2: Download full data package (fallback) ────────────────────
    if df is None:
        print("\n▶ Strategy 2: Download NPS data package from data.gov (fallback)")
        raw = download_nps_package(force_refresh=force_refresh)
        if raw is not None:
            try:
                df = clean_package(raw)
                print(f"[ok]    Package strategy succeeded — {len(df)} annual rows.")
            except Exception as e:
                print(f"[error] Cleaning after download failed: {e}")
                sys.exit(1)

    if df is None:
        print(
            "\n[error] Both acquisition strategies failed.\n"
            "        Check your internet connection, or inspect the raw data\n"
            "        in data/raw/ if files were partially downloaded."
        )
        sys.exit(1)

    # ── Enrich & inspect ─────────────────────────────────────────────────────
    df = add_derived_columns(df)

    # These are the Python equivalents of R's str() / glimpse() / summary():
    print("\n[info]  ── DataFrame Overview ──────────────────────────────────")
    print(f"  Shape:   {df.shape}")                          # R: dim(df)
    print(f"\n  Column types (dtypes):                        # R: str(df)")
    print(df.dtypes.to_string())
    print(f"\n  First 5 rows:                                 # R: head(df)")
    print(df.head().to_string(index=False))
    print(f"\n  Descriptive statistics:                       # R: summary(df)")
    print(df.describe().round(1).to_string())

    validate(df)

    # ── Save cleaned output ───────────────────────────────────────────────────
    # R equivalent: readr::write_csv(df, path)
    df.to_csv(CLEANED_OUT, index=False)
    print(f"\n[done]  Cleaned data saved → {CLEANED_OUT}")
    print(f"        Run 'python scripts/visualize.py' to generate the chart.")


if __name__ == "__main__":
    # argparse ≈ R's optparse or argparse packages
    parser = argparse.ArgumentParser(
        description="Fetch and clean Mount Rainier NPS visitation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download even if cached files exist in data/raw/.",
    )
    args = parser.parse_args()
    main(force_refresh=args.refresh)
