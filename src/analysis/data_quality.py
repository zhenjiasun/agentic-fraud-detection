"""Data-quality checks over the transaction table: nulls, value ranges,
unexpected categorical levels, referential integrity, and row-count sanity.
Each check returns pass/warn/fail with detail.
"""
from __future__ import annotations

from src.data.schema import COUNTRIES, MCC_CODES


def data_quality_section(store) -> dict:
    checks = []

    t = store.table("transactions")
    n = len(t)

    null_amt = int(t["amount"].isna().sum())
    checks.append(_check("null_amounts", null_amt == 0,
                         f"{null_amt} null transaction amounts"))

    neg_amt = int((t["amount"] < 0).sum())
    checks.append(_check("negative_amounts", neg_amt == 0,
                         f"{neg_amt} negative amounts"))

    bad_mcc = int((~t["mcc"].isin(MCC_CODES.keys())).sum())
    checks.append(_check("unknown_mcc", bad_mcc == 0,
                         f"{bad_mcc} transactions with unknown MCC", warn_only=True))

    user_ids = set(store.table("users")["user_id"])
    orphans = int((~t["user_id"].isin(user_ids)).sum())
    checks.append(_check("orphan_transactions", orphans == 0,
                         f"{orphans} transactions referencing missing users"))

    checks.append(_check("row_count", n > 0, f"{n} transactions present"))

    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    status = "fail" if n_fail else ("warn" if n_warn else "pass")
    return {"status": status, "n_fail": n_fail, "n_warn": n_warn, "checks": checks}


def _check(name, ok, detail, warn_only=False):
    if ok:
        status = "pass"
    else:
        status = "warn" if warn_only else "fail"
    return {"name": name, "status": status, "detail": detail}
