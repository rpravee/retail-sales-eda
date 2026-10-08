import pandas as pd
import pytest

from src.analysis import monthly_sales, top_products, weekday_sales
from src.cleaning import (DEFAULT_DATA_DIR, add_features, clean_transactions, find_reversed_orders,
                          flag_rows, load_raw, standardize_types)


def make(rows):
    """rows: (InvoiceNo, StockCode, Description, Quantity, 'YYYY-mm-dd HH:MM', UnitPrice, CustomerID, Country)"""
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                       "InvoiceDate", "UnitPrice", "CustomerID", "Country"])


def prepared(rows):
    return flag_rows(standardize_types(make(rows)))


def test_standardize_types_keeps_customer_ids_as_integers_and_uppercases_codes():
    df = standardize_types(make([("1", "85123a", "X", 1, "2011-01-03 10:00", 1.0, 17850.0, "UK"),
                                 ("2", "85123A", "X", 1, "2011-01-03 10:00", 1.0, None, "UK")]))
    assert str(df["CustomerID"].dtype) == "Int64"
    assert df["CustomerID"].iloc[0] == 17850 and pd.isna(df["CustomerID"].iloc[1])
    assert df["StockCode"].tolist() == ["85123A", "85123A"]


def test_flag_rows():
    df = prepared([("C100", "POST", "POSTAGE", -1, "2011-01-03 10:00", 5.0, None, "UK"),
                   ("101", "22423", "CAKESTAND", 2, "2011-01-03 10:00", 5.0, 1.0, "UK"),
                   ("102", "gift_0001_20", "Voucher", 1, "2011-01-03 10:00", 20.0, 1.0, "UK")])
    assert df["IsCancellation"].tolist() == [True, False, False]
    assert df["IsNonProduct"].tolist() == [True, False, True]
    assert df["HasCustomer"].tolist() == [False, True, True]


def test_find_reversed_orders_matches_only_exact_recent_reversals():
    rows = [
        ("1", "A", "x", 5, "2011-01-03 09:00", 2.0, 1.0, "UK"),    # reversed 10 minutes later -> flag
        ("C2", "A", "x", -5, "2011-01-03 09:10", 2.0, 1.0, "UK"),
        ("3", "B", "x", 5, "2011-01-03 09:00", 2.0, 1.0, "UK"),    # cancel has different quantity -> keep
        ("C4", "B", "x", -3, "2011-01-03 09:10", 2.0, 1.0, "UK"),
        ("5", "C", "x", 5, "2011-01-03 09:00", 2.0, 1.0, "UK"),    # cancelled 30 days later -> keep
        ("C6", "C", "x", -5, "2011-02-10 09:00", 2.0, 1.0, "UK"),
        ("C7", "D", "x", -5, "2011-01-03 09:00", 2.0, 1.0, "UK"),  # cancellation BEFORE the order -> keep
        ("8", "D", "x", 5, "2011-01-03 09:30", 2.0, 1.0, "UK"),
        ("9", "E", "x", 5, "2011-01-03 09:00", 2.0, None, "UK"),   # unknown customer -> cannot match, keep
        ("C10", "E", "x", -5, "2011-01-03 09:10", 2.0, None, "UK"),
    ]
    out = find_reversed_orders(prepared(rows))
    assert out.tolist() == [True, False, False, False, False, False, False, False, False, False]


def test_find_reversed_orders_catches_manual_line_cancellations():
    rows = [
        ("1", "22502", "BASKET", 60, "2011-06-01 09:00", 649.5, 15098.0, "UK"),     # GBP 38,970
        ("C2", "M", "Manual", -1, "2011-06-01 09:30", 38970.0, 15098.0, "UK"),       # cancelled via 'Manual'
        ("3", "22502", "BASKET", 60, "2011-06-01 09:00", 649.5, 99.0, "UK"),         # other customer: keep
    ]
    assert find_reversed_orders(prepared(rows)).tolist() == [True, False, False]


def test_clean_transactions_report_and_result():
    rows = [
        ("1", "22423", "CAKESTAND", 2, "2011-01-03 10:00", 5.0, 1.0, "UK"),
        ("1", "22423", "CAKESTAND", 2, "2011-01-03 10:00", 5.0, 1.0, "UK"),   # exact duplicate
        ("2", "22423", "CAKESTAND", 4, "2011-01-04 10:00", 5.0, 2.0, "UK"),
        ("C3", "22423", "CAKESTAND", -1, "2011-01-05 10:00", 5.0, 2.0, "UK"),  # partial return
        ("4", "20725", "FREEBIE", 1, "2011-01-05 10:00", 0.0, 3.0, "UK"),      # zero price
        ("5", "20726", "BAG", 10, "2011-01-06 09:00", 3.0, 4.0, "UK"),         # fully reversed
        ("C6", "20726", "BAG", -10, "2011-01-06 09:05", 3.0, 4.0, "UK"),
        ("7", "POST", "POSTAGE", 1, "2011-01-06 10:00", 18.0, 1.0, "UK"),
    ]
    res = clean_transactions(make(rows))
    rep = res.report.set_index("step")["rows_removed"]
    assert rep["1. exact duplicate rows"] == 1
    assert rep["2. cancellation rows"] == 2
    assert rep["3. non-positive quantity or price"] == 1
    assert rep["4. orders fully cancelled soon after"] == 1
    c = res.clean
    assert (c["Quantity"] > 0).all() and (c["UnitPrice"] > 0).all()
    assert len(c) == 3 and len(res.returns) == 2
    assert c["TotalSales"].sum() == pytest.approx(2 * 5 + 4 * 5 + 18)
    assert c["IsNonProduct"].sum() == 1


def test_monthly_sales_flags_partial_last_month():
    df = add_features(prepared([("1", "A", "x", 1, "2011-11-15 10:00", 10.0, 1.0, "UK"),
                                ("2", "A", "x", 1, "2011-12-05 10:00", 10.0, 1.0, "UK")]))
    out = monthly_sales(df)
    assert out["is_partial"].tolist() == [False, True]   # data stops on 5 Dec


def test_weekday_sales_shows_missing_day_as_zero():
    df = add_features(prepared([("1", "A", "x", 1, "2011-01-03 10:00", 10.0, 1.0, "UK")]))  # a Monday
    s = weekday_sales(df)
    assert s["Monday"] == 10 and s["Saturday"] == 0 and len(s) == 7


def test_top_products_excludes_non_product_lines():
    df = add_features(prepared([("1", "POST", "POSTAGE", 1, "2011-01-03 10:00", 999.0, 1.0, "UK"),
                                ("2", "22423", "CAKESTAND", 2, "2011-01-03 10:00", 5.0, 1.0, "UK")]))
    out = top_products(df, 5)
    assert out.index.tolist() == ["22423"]


@pytest.mark.skipif(not (DEFAULT_DATA_DIR / "Online Retail.xlsx").exists()
                    and not (DEFAULT_DATA_DIR / "online_retail_raw.csv").exists(),
                    reason="dataset not in data/")
def test_real_data_sanity():
    res = clean_transactions(load_raw())
    c = res.clean
    assert len(c) < 541_909 and c["TotalSales"].min() > 0
    assert top_products(c, 1).iloc[0]["Description"] != "DOTCOM POSTAGE"
    assert not c[c["Quantity"] >= 70_000].shape[0]        # the two cancelled mega-orders are gone
