"""Read-only tool registry for the investigator agent.

CAPABILITY RESTRICTION IS THE PRIMARY SECURITY CONTROL: every tool here is a
pure read against the store. There is deliberately NO block / allow / refund /
update_case / move_money tool — the agent literally cannot emit a money-affecting
action because no such tool exists. Attacker-controlled free-text (merchant
names) is routed through guards.tag_untrusted before it enters the model context.

`submit_finding` is the structured-output channel (advertised separately by the
investigator); it records a recommendation and is NOT a write tool.
"""
from __future__ import annotations

import json

from src.agent import guards

# Names that must never appear as tools (asserted by the boundedness test).
FORBIDDEN_TOOL_NAMES = {"block", "allow", "refund", "approve", "update_case",
                        "resolve", "move_money", "transfer", "delete", "write"}


class ToolRegistry:
    """Binds read tools to a store + the case under investigation.

    Tools default to the case's subject when an id arg is omitted, so a model
    that calls a tool with no arguments still gets the relevant evidence.
    """

    def __init__(self, store, case: dict):
        self.store = store
        self.case = case
        self.subject_id = case.get("subject_id")
        self.injection_flags: list[str] = []

        self._tools = {
            "get_case": self._get_case,
            "get_account_profile": self._get_account_profile,
            "get_transaction": self._get_transaction,
            "get_recent_transactions": self._get_recent_transactions,
            "get_graph_neighborhood": self._get_graph_neighborhood,
            "get_model_explanation": self._get_model_explanation,
        }

    # --- discovery (provider-neutral tool specs) ---
    def specs(self) -> list[dict]:
        return [
            {"name": "get_case", "description": "Fetch the case under review "
             "(status, model score, fired rule codes, graph signals).",
             "input_schema": {"type": "object", "properties": {
                 "case_id": {"type": "string"}}, "required": []}},
            {"name": "get_account_profile", "description": "Account summary for a "
             "user: age, spend, device/IP/card counts, calibrated risk score, ring flag.",
             "input_schema": {"type": "object", "properties": {
                 "user_id": {"type": "string"}}, "required": []}},
            {"name": "get_transaction", "description": "Fetch one transaction by id.",
             "input_schema": {"type": "object", "properties": {
                 "txn_id": {"type": "string"}}, "required": ["txn_id"]}},
            {"name": "get_recent_transactions", "description": "Recent transactions "
             "for a user (most recent first).",
             "input_schema": {"type": "object", "properties": {
                 "user_id": {"type": "string"},
                 "n": {"type": "integer"}}, "required": []}},
            {"name": "get_graph_neighborhood", "description": "Shared-infrastructure "
             "neighborhood for a user: devices/IPs shared with other accounts, ring flag.",
             "input_schema": {"type": "object", "properties": {
                 "user_id": {"type": "string"}}, "required": []}},
            {"name": "get_model_explanation", "description": "Top model feature "
             "contributions / reason codes for the case score.",
             "input_schema": {"type": "object", "properties": {
                 "case_id": {"type": "string"}}, "required": []}},
        ]

    @property
    def tool_names(self) -> set[str]:
        return set(self._tools.keys())

    def execute(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return self._tools[name](args or {})
        except Exception as e:  # tools are read-only; surface errors as data
            return json.dumps({"error": str(e)})

    # --- tool implementations (all reads) ---
    def _get_case(self, args) -> str:
        case = self.store.get_case(args.get("case_id") or self.case["case_id"]) or {}
        graph_signals = json.loads(case.get("graph_signals_json") or "{}")
        return json.dumps({
            "case_id": case.get("case_id"), "subject_id": case.get("subject_id"),
            "status": case.get("status"), "model_score": case.get("model_score"),
            "rule_codes": json.loads(case.get("rule_codes_json") or "[]"),
            "ring_member": graph_signals.get("ring_member", 0),
            "graph_signals": graph_signals,
        })

    def _get_account_profile(self, args) -> str:
        uid = args.get("user_id") or self.subject_id
        user = self.store.get_row("users", "user_id", uid) or {}
        score_row = self.store.get_row("account_scores", "user_id", uid) or {}
        txns = self.store.recent_transactions(uid, 200)
        amounts = [t["amount"] for t in txns]
        ring = json.loads(
            (self.store.get_case(self.case["case_id"]) or {}).get("graph_signals_json") or "{}"
        ).get("ring_member", 0)
        return json.dumps({
            "user_id": uid,
            "country": user.get("country"), "segment": user.get("segment"),
            "account_created": user.get("created_at"),
            "credit_limit": user.get("credit_limit"),
            "n_transactions": len(txns),
            "total_spend": round(sum(amounts), 2),
            "max_amount": round(max(amounts), 2) if amounts else 0,
            "model_score": score_row.get("score", self.case.get("model_score", 0)),
            "ring_member": ring,
        })

    def _get_transaction(self, args) -> str:
        t = self.store.get_row("transactions", "txn_id", args.get("txn_id")) or {}
        if not t:
            return json.dumps({"error": "not found"})
        merchant = self.store.get_row("merchants", "merchant_id", t.get("merchant_id")) or {}
        # merchant name is attacker-controllable -> tag as untrusted, flag injection
        name_tagged, flags = guards.tag_untrusted(merchant.get("name", ""))
        self.injection_flags += flags
        return json.dumps({
            "txn_id": t.get("txn_id"), "ts": t.get("ts"), "amount": t.get("amount"),
            "mcc": t.get("mcc"), "status": t.get("status"),
            "merchant_name_untrusted": name_tagged,
            "injection_flags": flags,
        })

    def _get_recent_transactions(self, args) -> str:
        uid = args.get("user_id") or self.subject_id
        n = int(args.get("n", 15))
        txns = self.store.recent_transactions(uid, n)
        rows = []
        for t in txns:
            merchant = self.store.get_row("merchants", "merchant_id", t["merchant_id"]) or {}
            name_tagged, flags = guards.tag_untrusted(merchant.get("name", ""))
            self.injection_flags += flags
            rows.append({"txn_id": t["txn_id"], "ts": t["ts"], "amount": t["amount"],
                         "status": t["status"], "merchant_name_untrusted": name_tagged})
        return json.dumps({"user_id": uid, "transactions": rows})

    def _get_graph_neighborhood(self, args) -> str:
        uid = args.get("user_id") or self.subject_id
        links = self.store.query_df(
            "SELECT dst_type, dst_id FROM entity_links WHERE src_type='user' "
            "AND src_id=? AND link_type IN ('uses_device','uses_ip','shares_identity')",
            (uid,),
        )
        neighbors = 0
        for _, r in links.iterrows():
            co = self.store.query_df(
                "SELECT COUNT(DISTINCT src_id) n FROM entity_links "
                "WHERE dst_type=? AND dst_id=? AND src_type='user'",
                (r["dst_type"], r["dst_id"]),
            )
            neighbors = max(neighbors, int(co.iloc[0]["n"]) - 1)
        ring = json.loads(
            (self.store.get_case(self.case["case_id"]) or {}).get("graph_signals_json") or "{}"
        ).get("ring_member", 0)
        return json.dumps({
            "user_id": uid, "max_accounts_sharing_infrastructure": neighbors,
            "ring_member": ring,
        })

    def _get_model_explanation(self, args) -> str:
        case = self.store.get_case(args.get("case_id") or self.case["case_id"]) or {}
        return json.dumps({
            "case_id": case.get("case_id"),
            "model_score": case.get("model_score"),
            "reason_codes": json.loads(case.get("rule_codes_json") or "[]"),
        })
