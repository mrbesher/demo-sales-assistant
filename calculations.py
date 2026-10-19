import ast
import math
import operator
import re
from pathlib import Path

import pandas as pd


def calculate(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Provide an arithmetic expression.")
    if len(expression) > 300:
        raise ValueError("Expressions are limited to 300 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Use numbers, parentheses and arithmetic operators only.") from error
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    pending = [(tree.body, False, 0)]
    values = []
    visited = 0
    while pending:
        node, ready, depth = pending.pop()
        if not ready:
            visited += 1
            if visited > 100 or depth > 20:
                raise ValueError("The expression is too complex.")
        if isinstance(node, ast.Constant):
            number_type = type(node.value)
            if number_type not in (int, float):
                raise ValueError("Only real numbers are supported.")
            result = node.value
        elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
            operation_type = type(node.op)
            if operation_type not in operations:
                raise ValueError("Supported operators are +, -, *, / and **.")
            if not ready:
                pending.append((node, True, depth))
                if isinstance(node, ast.BinOp):
                    pending.append((node.right, False, depth + 1))
                    pending.append((node.left, False, depth + 1))
                else:
                    pending.append((node.operand, False, depth + 1))
                continue
            operation = operations[operation_type]
            right = values.pop()
            try:
                if isinstance(node, ast.BinOp):
                    left = values.pop()
                    if operation_type is ast.Pow and (right < -12 or right > 12):
                        raise ValueError("Powers are limited to exponents between -12 and 12.")
                    result = operation(left, right)
                else:
                    result = operation(right)
            except (ZeroDivisionError, OverflowError) as error:
                raise ValueError("The arithmetic result is undefined or too large.") from error
        else:
            raise ValueError("Use numbers, parentheses and arithmetic operators only.")
        result_type = type(result)
        if result_type not in (int, float):
            raise ValueError("Only real results are supported.")
        finite = math.isfinite(result)
        if not finite or result < -1e15 or result > 1e15:
            raise ValueError("Numbers and results must be finite and within ±1e15.")
        values.append(result)
    return {"expression": expression, "result": values[0]}


class ProcessData:
    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        self.data = pd.read_csv(self.path)
        if "date" not in self.data:
            raise ValueError("The process CSV must contain a date column.")
        self.data["date"] = pd.to_datetime(self.data["date"], errors="raise")
        numeric = self.data.select_dtypes(include="number")
        self.columns = list(numeric.columns)
        if not self.columns:
            raise ValueError("The process CSV has no numeric measurements.")
        self.data[self.columns] = numeric.replace([math.inf, -math.inf], math.nan)

    def analyze(
        self, measurements: dict[str, float], filters: dict[str, list[float]] | None = None
    ) -> dict:
        if not measurements or len(measurements) > 10:
            raise ValueError("Provide between one and ten measurements.")
        filters = filters or {}
        if len(filters) > 5:
            raise ValueError("Use at most five historical comparison filters.")
        population = self.data
        for column, bounds in filters.items():
            if column not in self.columns or len(bounds) != 2:
                raise ValueError("Each filter needs a known measurement and [minimum, maximum].")
            for value in bounds:
                value_type = type(value)
                if value_type not in (int, float):
                    raise ValueError("Filter bounds must be finite numbers.")
                finite = math.isfinite(value)
                if not finite:
                    raise ValueError("Filter bounds must be finite numbers.")
            lower, upper = bounds
            if lower > upper:
                raise ValueError("Filter minimum cannot exceed maximum.")
            selected = population[column].between(lower, upper)
            population = population[selected]
        total_rows = len(self.data)
        matched_rows = len(population)
        if not matched_rows:
            raise ValueError("No historical rows match these filters.")
        first_date = population["date"].min()
        last_date = population["date"].max()
        date_start = first_date.isoformat()
        date_end = last_date.isoformat()
        comparisons = {}
        for column, value in measurements.items():
            if column not in self.columns:
                raise ValueError(f"Unknown measurement: {column}.")
            value_type = type(value)
            if value_type not in (int, float):
                raise ValueError("Measurements must be finite numbers.")
            finite = math.isfinite(value)
            if not finite:
                raise ValueError("Measurements must be finite numbers.")
            observed = population[column].dropna()
            count = len(observed)
            units = "not stated in CSV"
            if column.startswith("%"):
                units = "%"
            elif column == "Ore Pulp pH":
                units = "pH"
            comparison = {
                "value": value,
                "unit": units,
                "observations": count,
                "missing": matched_rows - count,
            }
            if count:
                quantiles = observed.quantile([0, 0.05, 0.5, 0.95, 1])
                lower_count = observed.lt(value)
                lower_count = lower_count.sum()
                equal_count = observed.eq(value)
                equal_count = equal_count.sum()
                percentile = 100 * (lower_count + 0.5 * equal_count) / count
                comparison["percentile"] = float(percentile)
                names = ["minimum", "p05", "median", "p95", "maximum"]
                quantile_pairs = zip(names, quantiles, strict=True)
                for name, quantile in quantile_pairs:
                    comparison[name] = float(quantile)
            comparisons[column] = comparison
        return {
            "source": self.path.name,
            "total_rows": total_rows,
            "matched_rows": matched_rows,
            "date_start": date_start,
            "date_end": date_end,
            "filters": filters,
            "measurements": comparisons,
            "interpretation": "Percentile counts lower readings plus half of ties. Historical comparisons do not establish operating targets, causes or safe adjustments. Rows are hourly averages, not independent experiments.",
        }


def extract_report(text: str) -> dict:
    plain = text.replace("**", "")
    report = {
        "report_id": None,
        "date": None,
        "site": None,
        "country": None,
        "equipment": None,
        "work": None,
        "onsite_hours": None,
        "travel_hours": None,
    }
    patterns = {
        "report_id": r"\b(FSR-\d+)\b",
        "date": r"(?im)^(?:Visit Date|Date)\s*:\s*([^\n]+)",
        "site": r"(?im)^(?:Site Location|Site)\s*:\s*([^\n]+)",
        "equipment": r"(?im)^\s*-?\s*Equipment\s*:\s*([^\n]+)",
        "work": r"(?ims)^(?:###\s*)?(?:Work Completed|Work Performed|Tasks Performed)\s*:?\s*\n(.*?)(?=\n(?:---|###|Total)|\Z)",
        "onsite_hours": r"(?i)(?:total\s+)?(?:on[- ]site\s+(?:hours|work)|hours\s+on[- ]site)\s*:\s*(\d+(?:\.\d+)?)",
        "travel_hours": r"(?i)(?:total\s+)?travel(?:\s+(?:time|hours))?\s*:\s*(\d+(?:\.\d+)?)",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, plain)
        if match:
            value = match.group(1)
            value = value.strip()
            if field in ("onsite_hours", "travel_hours"):
                value = float(value)
            report[field] = value
    if report["site"] and "," in report["site"]:
        location_parts = report["site"].rsplit(",", 1)
        report["country"] = location_parts[1].strip()
    lines = plain.splitlines()
    report["cost_lines"] = []
    report["stated_totals"] = []
    report["hours_lines"] = []
    for line in lines:
        line = line.strip()
        money = re.search(
            r"(?i)\b(?:USD|EUR|AUD|CAD|CLP|ZAR|SEK|cost|invoice|charge|rate|total due)\b|[$€£]",
            line,
        )
        hours = re.search(r"(?i)\b(?:hours|hrs|on.site|travel time)\b", line)
        total = re.search(r"(?i)\btotal\b", line)
        amount = re.search(r"\d", line)
        if money:
            report["cost_lines"].append(line)
            if total and amount:
                report["stated_totals"].append(line)
        if hours:
            report["hours_lines"].append(line)
    currencies = re.findall(r"\b(?:USD|EUR|AUD|CAD|CLP|ZAR|GBP|SEK)\b", plain)
    if "€" in plain:
        currencies.append("EUR")
    if "£" in plain:
        currencies.append("GBP")
    currencies = set(currencies)
    report["currencies"] = sorted(currencies)
    report["notes"] = [
        "Extracted fields are incomplete when wording varies; consult the full original report. Preserve stated totals separately from any calculated sum."
    ]
    if "$" in plain and not currencies:
        report["notes"].append("Dollar currency is unspecified. Do not assume USD.")
    if not report["stated_totals"]:
        report["notes"].append(
            "No explicit monetary total was extracted; missing cost is not zero."
        )
    return report
