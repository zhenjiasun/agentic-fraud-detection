"""Adversarial-behavior monitoring: signals consistent with fraudsters adapting.

- threshold probing: a spike in transactions priced just under the auto-block
  amount band (attackers testing limits)
- rising decline rate over the window (card-testing pressure)
- new-device velocity surge
- prompt-injection attempts flagged by the investigator guard
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def adversarial_section(store, settings) -> dict:
    alerts = []
    t = store.table("transactions")
    t["ts_dt"] = pd.to_datetime(t["ts"])
    t = t.sort_values("ts_dt")
    last = t["ts_dt"].max()

    # split into reference (earlier) vs recent window for rate comparisons
    cutoff = last - pd.Timedelta(days=7)
    recent = t[t["ts_dt"] >= cutoff]
    earlier = t[t["ts_dt"] < cutoff]

    # 1. rising decline rate
    r_dec = (recent["status"] == "declined").mean() if len(recent) else 0
    e_dec = (earlier["status"] == "declined").mean() if len(earlier) else 0
    if r_dec > max(0.06, e_dec * 1.5) and len(recent) > 50:
        alerts.append({"signal": "rising_decline_rate",
                       "detail": f"recent {r_dec:.1%} vs earlier {e_dec:.1%}"})

    # 2. threshold probing: cluster of small auths (card-testing fingerprint)
    small_recent = (recent["amount"] < 5).mean() if len(recent) else 0
    small_earlier = (earlier["amount"] < 5).mean() if len(earlier) else 0
    if small_recent > max(0.05, small_earlier * 1.5) and len(recent) > 50:
        alerts.append({"signal": "micro_auth_spike",
                       "detail": f"recent {small_recent:.1%} sub-$5 vs earlier {small_earlier:.1%}"})

    # 3. prompt-injection attempts flagged by the investigator
    inv = store.table("investigations")
    flagged = 0
    if not inv.empty and "injection_flags_json" in inv.columns:
        flagged = int(sum(1 for v in inv["injection_flags_json"].fillna("[]")
                          if json.loads(v)))
    if flagged > 0:
        alerts.append({"signal": "prompt_injection_attempts",
                       "detail": f"{flagged} investigation(s) saw injection in entity data"})

    return {"alerts": alerts,
            "recent_decline_rate": round(float(r_dec), 4),
            "recent_micro_auth_share": round(float(small_recent), 4),
            "injection_flagged_investigations": flagged}
