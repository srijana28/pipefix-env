from __future__ import annotations

from typing import Any, Dict, List


TASKS: Dict[str, Dict[str, Any]] = {
    "easy_single_failure": {
        "difficulty": "easy",
        "description": "Pipeline fails due to invalid date format in transform stage.",
        "dataset": [
            {"user_id": 1001, "date": "12/01/23", "age": 27, "country": "US"},
            {"user_id": 1002, "date": "12/02/23", "age": 31, "country": "CA"},
        ],
        "schema": {"user_id": "int", "date": "date_iso", "age": "int", "country": "str"},
        "required_fix_order": ["fix_transformation:date_to_iso"],
        "final_data": [
            {"user_id": 1001, "date": "2023-12-01", "age": 27, "country": "US"},
            {"user_id": 1002, "date": "2023-12-02", "age": 31, "country": "CA"},
        ],
        "optimal_steps": 4,
    },
    "medium_multi_issue": {
        "difficulty": "medium",
        "description": "Pipeline has missing values, duplicates, and wrong schema type for age.",
        "dataset": [
            {"user_id": 1, "date": "2024-01-02", "age": "29", "country": "US"},
            {"user_id": 1, "date": "2024-01-02", "age": "29", "country": "US"},
            {"user_id": 2, "date": "2024-01-03", "age": None, "country": "IN"},
        ],
        "schema": {"user_id": "int", "date": "date_iso", "age": "str", "country": "str"},
        "required_fix_order": [
            "fill_missing:age:median",
            "drop_column:deduplicate",
            "fix_schema:age:int",
        ],
        "final_data": [
            {"user_id": 1, "date": "2024-01-02", "age": 29, "country": "US"},
            {"user_id": 2, "date": "2024-01-03", "age": 29, "country": "IN"},
        ],
        "optimal_steps": 7,
    },
    "hard_cascading_failures": {
        "difficulty": "hard",
        "description": "Cascading errors: schema mismatch, invalid dates, negative ages, and hidden missing user_id.",
        "dataset": [
            {"user_id": "10", "date": "03/01/24", "age": -4, "country": "US"},
            {"user_id": None, "date": "03/02/24", "age": "34", "country": "FR"},
            {"user_id": "12", "date": "03/03/24", "age": "41", "country": "DE"},
        ],
        "schema": {"user_id": "str", "date": "date_slash", "age": "str", "country": "str"},
        "required_fix_order": [
            "fill_missing:user_id:forward_fill",
            "fix_transformation:date_to_iso",
            "fix_schema:user_id:int",
            "fix_schema:age:int",
            "fix_transformation:non_negative_age",
        ],
        "final_data": [
            {"user_id": 10, "date": "2024-03-01", "age": 4, "country": "US"},
            {"user_id": 10, "date": "2024-03-02", "age": 34, "country": "FR"},
            {"user_id": 12, "date": "2024-03-03", "age": 41, "country": "DE"},
        ],
        "optimal_steps": 10,
    },
}


def list_task_names() -> List[str]:
    return list(TASKS.keys())
