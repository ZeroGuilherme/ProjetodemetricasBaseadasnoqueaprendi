from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_delta(current: float | int | None, previous: float | int | None) -> float | None:
    if current is None or previous is None:
        return None
    if pd.isna(previous) or previous == 0:
        return None
    return float((current - previous) / previous * 100)


def _base_metrics(df: pd.DataFrame) -> dict[str, Any]:
    revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0

    if "order_id" in df.columns:
        orders = int(df["order_id"].nunique(dropna=True))
    else:
        orders = int(len(df))

    ticket = float(revenue / orders) if orders > 0 else 0.0

    customers: int | None
    if "customer_id" in df.columns:
        customers = int(df["customer_id"].nunique(dropna=True))
    else:
        customers = None

    return {
        "revenue": revenue,
        "orders": orders,
        "ticket": ticket,
        "customers": customers,
    }


def compute_kpis(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute KPI values and percentage deltas against previous period."""
    current = _base_metrics(current_df)
    previous = _base_metrics(previous_df) if previous_df is not None else _base_metrics(pd.DataFrame())

    customers_delta = None
    if current["customers"] is not None and previous["customers"] is not None:
        customers_delta = _safe_delta(current["customers"], previous["customers"])

    return {
        "revenue": {
            "value": current["revenue"],
            "delta_pct": _safe_delta(current["revenue"], previous["revenue"]),
            "available": True,
        },
        "orders": {
            "value": current["orders"],
            "delta_pct": _safe_delta(current["orders"], previous["orders"]),
            "available": True,
        },
        "ticket": {
            "value": current["ticket"],
            "delta_pct": _safe_delta(current["ticket"], previous["ticket"]),
            "available": True,
        },
        "customers": {
            "value": current["customers"],
            "delta_pct": customers_delta,
            "available": current["customers"] is not None,
        },
    }
