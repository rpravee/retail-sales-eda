"""Small, reusable aggregation helpers for the Online Retail analysis."""
from __future__ import annotations

import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Sales, orders and customers per month, flagging a month the data does not fully cover."""
    out = (df.groupby("Month")
             .agg(sales=("TotalSales", "sum"), orders=("InvoiceNo", "nunique"),
                  customers=("CustomerID", "nunique"))
             .reset_index())
    last_day_in_data = df["InvoiceDate"].max().normalize()
    is_last = out["Month"] == out["Month"].max()
    out["is_partial"] = is_last & (last_day_in_data < out["Month"].max().end_time.normalize())
    return out


def weekday_sales(df: pd.DataFrame) -> pd.Series:
    """Sales per weekday; days with no trading show as 0 rather than disappearing / NaN."""
    return df.groupby("DayOfWeek")["TotalSales"].sum().reindex(WEEKDAYS, fill_value=0.0)


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top merchandise by revenue. Groups by StockCode (stable id) and labels with its most common
    description, because one code can carry several slightly different descriptions."""
    merch = df[~df["IsNonProduct"]]
    label = (merch.dropna(subset=["Description"]).groupby("StockCode")["Description"]
             .agg(lambda s: s.mode().iat[0]))
    out = (merch.groupby("StockCode").agg(sales=("TotalSales", "sum"), units=("Quantity", "sum"),
                                          orders=("InvoiceNo", "nunique"))
           .sort_values("sales", ascending=False).head(n))
    out.insert(0, "Description", label.reindex(out.index))
    return out


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("Country").agg(sales=("TotalSales", "sum"), orders=("InvoiceNo", "nunique"))
             .sort_values("sales", ascending=False))
    out["share"] = out["sales"] / out["sales"].sum()
    out["avg_order_value"] = out["sales"] / out["orders"]
    return out


def order_values(df: pd.DataFrame) -> pd.Series:
    """Total value of each invoice (one number per order)."""
    return df.groupby("InvoiceNo")["TotalSales"].sum()


def customer_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Customers ranked by revenue with the cumulative share of total sales (identified customers only)."""
    cust = (df.dropna(subset=["CustomerID"]).groupby("CustomerID")["TotalSales"].sum()
              .sort_values(ascending=False).to_frame("sales"))
    cust["cum_share"] = cust["sales"].cumsum() / cust["sales"].sum()
    cust["customer_share"] = pd.RangeIndex(1, len(cust) + 1) / len(cust)
    return cust


def describe_with_skew(s: pd.Series) -> pd.Series:
    """Descriptive stats that make skew obvious: mean vs median, skewness, tail percentiles."""
    return pd.Series({"count": s.count(), "mean": s.mean(), "median": s.median(), "std": s.std(),
                      "skew": s.skew(), "p95": s.quantile(.95), "p99": s.quantile(.99), "max": s.max()})
