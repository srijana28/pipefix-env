from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Dict, List, Set, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

from openai import APIError, OpenAI

from env import Action, PipeFixEnv

ACTION_CHEATSHEET = (
    "Parameter rules: fix_transformation only supports parameters.mode = date_to_iso or non_negative_age. "
    "fix_schema requires parameters.column and parameters.target_type (e.g. int, str), not new_type. "
    "Do not use fix_schema to fix date strings; slash dates need fix_transformation mode date_to_iso. "
    "fill_missing uses parameters.column and parameters.strategy median or forward_fill. "
    "drop_column uses parameters.mode deduplicate OR parameters.column to drop. "
    "Use inspect_logs or inspect_data at most twice; then apply fixes or run_pipeline. "
    "Do not repeat the same action many times."
)

TASK_HINTS: Dict[str, str] = {
    "easy_single_failure": (
        "Task easy: dates are MM/DD/YY strings. Apply fix_transformation with "
        "parameters.mode=date_to_iso first; do not change schema date to str. "
        "Then run_pipeline until success; then finish."
    ),
    "medium_multi_issue": (
        "Task medium: apply fixes in order — fill_missing column age strategy median, "
        "then drop_column mode deduplicate, then fix_schema age target_type int. "
        "Do not fix_schema age to int before missing ages are filled."
    ),
    "hard_cascading_failures": (
        "Task hard: typical sequence — fill_missing user_id forward_fill, "
        "fix_transformation date_to_iso, fix_schema user_id int, fix_schema age int, "
        "fix_transformation non_negative_age, run_pipeline, finish."
    ),
}

DEFAULT_TASK_ORDER = [
    "easy_single_failure",
    "medium_multi_issue",
    "hard_cascading_failures",
]

FINISH_ACTION_STR = json.dumps(
    {"action_type": "finish", "parameters": {}},
    separators=(",", ":"),
)

DEFAULT_TASK_SEEDS = [42, 43, 44]
MAX_STEPS = 24
TEMPERATURE = 0.0
MAX_TOKENS = 512

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

ALLOWED_ACTIONS: Set[str] = {
    "inspect_data",
    "inspect_logs",
    "run_pipeline",
    "fix_schema",
    "fix_transformation",
    "fill_missing",
    "drop_column",
    "rollback",
    "finish",
}


def _safe_json_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _extract_json_object(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()

    d = _safe_json_parse(raw)
    if d:
        return d

    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        return _safe_json_parse(m.group(0))

    return {}


def _normalize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    at = raw.get("action_type")

    if at not in ALLOWED_ACTIONS:
        return {}

    params = raw.get("parameters", {})

    if not isinstance(params, dict):
        return {}

    return {
        "action_type": at,
        "parameters": params
    }


def _llm_action(
    client: OpenAI,
    task: str,
    obs: Dict[str, Any],
    step: int,
    recent_actions: List[str]
) -> Dict[str, Any]:

    system = (
        "You are a data pipeline debugging agent. "
        "Reply ONLY in JSON format."
    )

    user = json.dumps({
        "task_name": task,
        "step": step,
        "observation": obs,
        "allowed_actions": list(ALLOWED_ACTIONS),
        "recent_action_types": recent_actions[-5:]
    })

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = completion.choices[0].message.content or ""
    parsed = _extract_json_object(raw)

    return _normalize_action(parsed)


def run_episode(task: str, client: OpenAI, seed: int | None):

    env = PipeFixEnv(task_name=task)

    # FIXED RESET LOGIC
    try:
        reset_result = env.reset(seed=seed)
    except TypeError:
        try:
            reset_result = env.reset()
        except Exception as exc:
            raise RuntimeError(f"Environment reset failed: {exc}")

    obs = reset_result.model_dump()

    rewards = []
    recent_actions = []

    for step in range(1, MAX_STEPS + 1):

        action_dict = _llm_action(
            client,
            task,
            obs,
            step,
            recent_actions
        )

        if not action_dict:
            action_dict = {
                "action_type": "inspect_logs",
                "parameters": {}
            }

        action = Action(**action_dict)

        result = env.step(action)

        obs = result.observation.model_dump()
        rewards.append(result.reward.value)

        recent_actions.append(action.action_type)

        if result.done:
            break

    return result.info.score, False, ""


def main():

    if not API_KEY:
        raise RuntimeError("Missing API key")

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )

    for task, seed in zip(DEFAULT_TASK_ORDER, DEFAULT_TASK_SEEDS):
        run_episode(task, client, seed)


if __name__ == "__main__":
    main()
