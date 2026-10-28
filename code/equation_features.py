"""Extract and check simple arithmetic equations from candidate traces."""

from __future__ import annotations

import ast
import math
import re

import numpy as np


EQUATION_FEATURE_NAMES = (
    "has_checked_equation",
    "log_checked_equations",
    "log_verified_equations",
    "log_invalid_equations",
    "verified_equation_fraction",
    "any_invalid_equation",
    "equation_coverage",
    "verified_minus_invalid",
)

_FRACTION_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_EQUATION_RUN_RE = re.compile(r"[0-9.+\-*/()%\s=^]+")
_SAFE_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}
_SAFE_UNARY = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def _normalise_math(text: str) -> str:
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("\\div", "/").replace("\\left", " ").replace("\\right", " ")
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
    text = text.replace("\\%", "%").replace("$", " ").replace(",", "")
    for _ in range(6):
        replaced = _FRACTION_RE.sub(r"((\1)/(\2))", text)
        if replaced == text:
            break
        text = replaced
    return text.replace("%", " ")


def _safe_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Expression):
        return _safe_number(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY:
        value = _safe_number(node.operand)
        return None if value is None else _SAFE_UNARY[type(node.op)](value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _safe_number(node.left)
        right = _safe_number(node.right)
        if left is None or right is None:
            return None
        try:
            return float(_SAFE_OPERATORS[type(node.op)](left, right))
        except (ArithmeticError, OverflowError, ValueError):
            return None
    return None


def _evaluate_expression(expression: str) -> float | None:
    expression = expression.strip()
    if not expression or not re.search(r"\d", expression):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_number(tree)
    except (SyntaxError, ValueError):
        return None
    if value is None or not math.isfinite(value):
        return None
    return value


def check_equations(text: str) -> tuple[int, int]:
    """Return (verified equation count, invalid equation count)."""
    normalized = _normalise_math(text)
    checked = 0
    verified = 0
    for line in normalized.splitlines():
        for match in _EQUATION_RUN_RE.finditer(line):
            run = match.group().strip()
            if "=" not in run:
                continue
            sides = [side.strip() for side in run.split("=")]
            if len(sides) < 2:
                continue
            values = [_evaluate_expression(side) for side in sides]
            if any(value is None for value in values):
                continue
            if len({(round(value, 10),) for value in values}) == 1:
                reference = values[0]
                tolerance = 1e-7 * max(1.0, abs(reference))
                valid = all(abs(value - reference) <= tolerance for value in values)
            else:
                valid = False
            checked += 1
            verified += int(valid)
    return verified, checked - verified


def build_equation_features(
    bank: list[dict], embeddings: np.ndarray
) -> np.ndarray:
    if embeddings.shape[:2] != (len(bank), 4):
        raise ValueError("Bank and embeddings are not aligned")
    result = np.zeros((len(bank), 4, len(EQUATION_FEATURE_NAMES)), dtype=np.float32)
    for q_index, row in enumerate(bank):
        counts = []
        for candidate in row["candidates"]:
            verified, invalid = check_equations(candidate["text"])
            counts.append((verified, invalid))
        max_checked = max((verified + invalid for verified, invalid in counts), default=0)
        for i, (verified, invalid) in enumerate(counts):
            checked = verified + invalid
            result[q_index, i] = (
                float(checked > 0),
                math.log1p(checked),
                math.log1p(verified),
                math.log1p(invalid),
                verified / checked if checked else 0.5,
                float(invalid > 0),
                checked / max(max_checked, 1),
                math.log1p(verified) - math.log1p(invalid),
            )
    return result
