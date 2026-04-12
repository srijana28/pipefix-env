from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def dataset_sample(dataset: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    return deepcopy(dataset[:limit])


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _slash_to_iso(value: str) -> str:
    parsed = datetime.strptime(value, "%m/%d/%y")
    return parsed.strftime("%Y-%m-%d")


def apply_fix(
    dataset: List[Dict[str, Any]],
    schema: Dict[str, str],
    action_type: str,
    parameters: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Optional[str]]:
    data = deepcopy(dataset)
    new_schema = deepcopy(schema)
    error = None

    if action_type == "fix_transformation":
        mode = parameters.get("mode")
        if mode == "date_to_iso":
            for row in data:
                if isinstance(row.get("date"), str) and "/" in row["date"]:
                    row["date"] = _slash_to_iso(row["date"])
            new_schema["date"] = "date_iso"
        elif mode == "non_negative_age":
            for row in data:
                if isinstance(row.get("age"), int) and row["age"] < 0:
                    row["age"] = abs(row["age"])
        else:
            error = "Unsupported transformation mode"

    elif action_type == "fix_schema":
        column = parameters.get("column")
        target_type = parameters.get("target_type")
        if column not in new_schema:
            error = f"Column not found: {column}"
        else:
            for row in data:
                val = row.get(column)
                if val is None:
                    continue
                if target_type == "int" and not isinstance(val, int):
                    try:
                        row[column] = int(val)
                    except Exception:
                        error = f"TypeError: expected int but got {type(val).__name__}"
                        break
            new_schema[column] = target_type

    elif action_type == "fill_missing":
        column = parameters.get("column")
        strategy = parameters.get("strategy", "median")
        if column not in new_schema:
            error = f"Column not found: {column}"
        elif strategy == "median":
            vals = []
            for r in data:
                raw = r.get(column)
                if isinstance(raw, (int, float)):
                    vals.append(raw)
                elif isinstance(raw, str):
                    try:
                        vals.append(float(raw))
                    except ValueError:
                        pass
            fill = int(sum(vals) / len(vals)) if vals else 0
            for row in data:
                if row.get(column) is None:
                    row[column] = fill
        elif strategy == "forward_fill":
            last = None
            for row in data:
                if row.get(column) is None and last is not None:
                    row[column] = last
                if row.get(column) is not None:
                    last = row.get(column)
            for row in data:
                if row.get(column) is None:
                    row[column] = 0
        else:
            error = "Unsupported missing-value strategy"

    elif action_type == "drop_column":
        mode = parameters.get("mode")
        if mode == "deduplicate":
            seen = set()
            deduped = []
            for row in data:
                marker = tuple(sorted(row.items()))
                if marker not in seen:
                    seen.add(marker)
                    deduped.append(row)
            data = deduped
        else:
            column = parameters.get("column")
            if column not in new_schema:
                error = f"Column not found: {column}"
            else:
                new_schema.pop(column, None)
                for row in data:
                    row.pop(column, None)
    else:
        error = "Action does not modify dataset"

    return data, new_schema, error


def run_pipeline(dataset, schema):
    logs = ["[INFO] Pipeline execution started"]

    # CLEAN
    for row in dataset:
        if any(v is None for v in row.values()):
            logs.append("[WARN] Missing values detected in dataset")
            return "clean", "failed", "incomplete data", logs

    logs.append("[INFO] Cleaning stage passed")

    # TRANSFORM
    if schema.get("date") != "date_iso":
        logs.append("[WARN] Date format inconsistent")
        return "transform", "failed", "date format issue", logs

    logs.append("[INFO] Transform stage passed")

    # VALIDATE
    for row in dataset:
        if isinstance(row.get("age"), int) and row["age"] < 0:
            logs.append("[ERROR] Invalid numeric range detected")
            return "validate", "failed", "invalid age", logs

    for col, typ in schema.items():
        if typ == "int":
            for row in dataset:
                if not isinstance(row.get(col), int):
                    logs.append("[ERROR] Type inconsistency detected")
                    return "validate", "failed", "type mismatch", logs

    logs.append("[INFO] Validation stage passed")

    logs.append("[INFO] Output successfully generated")
    return "output", "success", None, logs