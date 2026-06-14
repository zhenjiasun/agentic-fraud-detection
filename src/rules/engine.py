"""Safe rule evaluation via a restricted AST interpreter.

`when` expressions are parsed with ast.parse and walked by a hand-written
interpreter that supports ONLY: boolean ops, comparisons, arithmetic, numeric/
string/bool constants, namespaced names (account/graph/txn/model) and one level
of attribute access on those namespaces. No calls, no subscripts, no dunder, no
arbitrary attribute traversal. eval()/exec() are never used — so a malicious rule
string cannot execute code (covered by test_rules_engine.py).
"""
from __future__ import annotations

import ast
import operator
from typing import Any

from src.rules.loader import Rule

ROOTS = {"account", "graph", "txn", "model"}

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Mod: operator.mod}
_CMP_OPS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
            ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


class RuleSyntaxError(ValueError):
    pass


def _eval(node: ast.AST, ctx: dict[str, dict]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, ctx)
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, ctx) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _eval(node.operand, ctx)
        if isinstance(node.op, ast.USub):
            return -_eval(node.operand, ctx)
        raise RuleSyntaxError(f"unary op not allowed: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if not op:
            raise RuleSyntaxError(f"binary op not allowed: {type(node.op).__name__}")
        return op(_eval(node.left, ctx), _eval(node.right, ctx))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for op_node, comp in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if not op:
                raise RuleSyntaxError(f"comparison not allowed: {type(op_node).__name__}")
            right = _eval(comp, ctx)
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id not in ROOTS:
            raise RuleSyntaxError("attribute access only on account/graph/txn/model")
        if node.attr.startswith("_"):
            raise RuleSyntaxError("dunder/underscore attribute access forbidden")
        return ctx.get(node.value.id, {}).get(node.attr, 0)
    if isinstance(node, ast.Name):
        # bare names not allowed (must be namespaced) except booleans handled as Constant
        raise RuleSyntaxError(f"bare name not allowed: {node.id}")
    raise RuleSyntaxError(f"node type not allowed: {type(node).__name__}")


def safe_eval(expr: str, ctx: dict[str, dict]) -> bool:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RuleSyntaxError(str(e))
    return bool(_eval(tree, ctx))


class RulesEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules
        # fail fast on malformed expressions at construction time
        for r in rules:
            ast.parse(r.when, mode="eval")

    def evaluate(self, ctx: dict[str, dict]) -> list[dict]:
        """Return all fired rules for one subject's context."""
        fired = []
        for r in self.rules:
            try:
                if safe_eval(r.when, ctx):
                    fired.append({"id": r.id, "action": r.action,
                                  "reason_code": r.reason_code, "confidence": r.confidence})
            except RuleSyntaxError:
                raise
        return fired
