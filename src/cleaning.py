"""Reusable data-cleaning functions for the UCI "Online Retail" dataset.

Every cleaning decision lives in a small named function with a docstring that says WHY,
so the notebook stays a readable story and the same logic can be re-used or tested.

Typical use:
    raw = load_raw()
    result = clean_transactions(raw)
    result.clean      # positive merchandise/service sales, ready for analysis
    result.returns    # cancellation rows (kept separately for return analysis)
    result.report     # one row per cleaning step: how many rows it removed and why
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Stock codes that are services, fees or accounting adjustments, not merchandise.
# (Found by listing every StockCode that is not a normal 5-digit product code.)
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES", "AMAZONFEE", "CRUK", "B",
}
NON_PRODUCT_PREFIXES = ("GIFT_",)   # gift vouchers: money in, but not a product sold


# --------------------------------------------------------------------------- loading
def load_raw(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the raw dataset from data/ (xlsx or csv), else download it from UCI."""
    data_dir = Path(data_dir)
    xlsx, csv = data_dir / "Online Retail.xlsx", data_dir / "online_retail_raw.csv"
    if xlsx.exists():
        df = pd.read_excel(xlsx)
    elif csv.exists():
        df = pd.read_csv(csv, parse_dates=["InvoiceDate"])
    else:
        from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
        df = fetch_ucirepo(id=352).data.features.copy()
    return df


# ------------------------------------------------------------------- cleaning steps
def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Give every column a sensible, consistent dtype.

    * Identifiers (InvoiceNo, StockCode) are text: they are labels, not numbers to add up.
    * StockCode is upper-cased because the same product appears as e.g. '85123A' and '85123a'.
    * CustomerID arrives as float (17850.0) only because of NaNs; the nullable Int64 type
      keeps real IDs as integers while still allowing missing values.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Quantity"] = df["Quantity"].astype("int64")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["Country"] = df["Country"].astype(str).str.strip()
    return df


def flag_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags instead of deleting rows, so each decision stays visible."""
    df = df.copy()
    df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C")
    codes = df["StockCode"]
    df["IsNonProduct"] = codes.isin(NON_PRODUCT_CODES) | codes.str.startswith(NON_PRODUCT_PREFIXES)
    df["HasCustomer"] = df["CustomerID"].notna()
    return df


def find_reversed_orders(df: pd.DataFrame, max_gap: str = "7D") -> pd.Series:
    """Flag orders that were placed and then fully cancelled shortly afterwards.

    A cancellation row (invoice starting with 'C') only removes the *cancellation*; the original
    order is still in the data. If we keep that original order, sales are overstated. Example in
    this dataset: 80,995 paper-craft items ordered at 09:15 and cancelled at 09:27 the same day,
    worth GBP 168k, which would otherwise look like the best-selling "product" of the year.

    Pass 1: an order is matched to a cancellation when customer, product, quantity and unit price
            are identical and the cancellation comes within `max_gap` AFTER the order.
    Pass 2: a few cancellations are booked on a generic 'M' (Manual) line instead of the product
            (e.g. 60 picnic baskets = GBP 38,970 cancelled by one 'Manual' row), so those are
            matched on customer + exact line value instead.

    Only exact full reversals are caught; partial returns remain in the data (we analyse gross
    sales). Returns a boolean Series (indexed like `df`): True for the reversed original orders.
    """
    known = df["CustomerID"].notna()
    orders = df[known & ~df["IsCancellation"] & (df["Quantity"] > 0)]
    cancels = df[known & df["IsCancellation"] & (df["Quantity"] < 0)]

    def _match(left, right, keys):
        for part in (left, right):
            part["CustomerID"] = part["CustomerID"].astype("int64")
        left, right = left.sort_values("InvoiceDate"), right.sort_values("InvoiceDate")
        matched = pd.merge_asof(left, right, on="InvoiceDate", by=keys,
                                tolerance=pd.Timedelta(max_gap), direction="backward")
        return matched["order_idx"].dropna().astype("int64").unique()

    # pass 1: same customer, product, quantity and price
    keys1 = ["CustomerID", "StockCode", "UnitPrice", "AbsQty"]
    right1 = orders.assign(AbsQty=orders["Quantity"], order_idx=orders.index)[keys1 + ["InvoiceDate", "order_idx"]]
    left1 = cancels.assign(AbsQty=-cancels["Quantity"])[keys1 + ["InvoiceDate"]]
    reversed_idx = set(_match(left1, right1, keys1))

    # pass 2: 'Manual' cancellations matched on customer + line value
    keys2 = ["CustomerID", "LineValue"]
    manual = cancels[cancels["StockCode"] == "M"]
    candidates = orders[orders["StockCode"] != "M"]
    right2 = candidates.assign(LineValue=(candidates["Quantity"] * candidates["UnitPrice"]).round(2),
                               order_idx=candidates.index)[keys2 + ["InvoiceDate", "order_idx"]]
    left2 = manual.assign(LineValue=(-manual["Quantity"] * manual["UnitPrice"]).round(2))[keys2 + ["InvoiceDate"]]
    reversed_idx |= set(_match(left2, right2, keys2))

    return pd.Series(df.index.isin(list(reversed_idx)), index=df.index, name="IsReversedOrder")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features used throughout the analysis."""
    df = df.copy()
    df["TotalSales"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df


# ---------------------------------------------------------------------- the pipeline
@dataclass
class CleaningResult:
    clean: pd.DataFrame      # rows used for sales analysis
    returns: pd.DataFrame    # cancellation rows (kept for return analysis)
    report: pd.DataFrame     # audit trail: rows removed by each step and why


def clean_transactions(raw: pd.DataFrame) -> CleaningResult:
    """Run the full cleaning pipeline and keep an audit trail of every step."""
    steps = []

    def record(step, before, after, why):
        steps.append({"step": step, "rows_removed": before - after, "rows_remaining": after, "why": why})

    df = standardize_types(raw)
    n = len(df)
    steps.append({"step": "0. raw data", "rows_removed": 0, "rows_remaining": n, "why": "starting point"})

    df = df.drop_duplicates()
    record("1. exact duplicate rows", n, len(df), "identical in every column: almost certainly double-logged lines")
    n = len(df)

    df = flag_rows(df)
    df["IsReversedOrder"] = find_reversed_orders(df)

    returns = add_features(df[df["IsCancellation"]])
    df = df[~df["IsCancellation"]]
    record("2. cancellation rows", n, len(df), "invoice starts with 'C': money going back out, analysed separately")
    n = len(df)

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    record("3. non-positive quantity or price", n, len(df),
           "stock write-offs, free items, bad-debt adjustments: not real sales")
    n = len(df)

    df = df[~df["IsReversedOrder"]]
    record("4. orders fully cancelled soon after", n, len(df),
           "the original order of a cancelled pair (same product, or cancelled via a Manual line); keeping it would overstate sales")

    df = add_features(df.drop(columns=["IsCancellation"]))
    n_nonprod = int(df["IsNonProduct"].sum())
    steps.append({"step": "5. flagged (kept): non-product lines", "rows_removed": 0, "rows_remaining": len(df),
                  "why": f"{n_nonprod:,} postage/fee/adjustment lines are flagged IsNonProduct and excluded from product rankings"})
    return CleaningResult(df.reset_index(drop=True), returns.reset_index(drop=True), pd.DataFrame(steps))
