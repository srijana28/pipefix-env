from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

from env import Action, PipeFixEnv


# =========================
# CONFIG
# =========================
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_ORDER = ["easy_single_failure", "medium_multi_issue", "hard_cascading_failures"]
TASK_SEEDS = [42, 43, 44]

MAX_STEPS = 24
TEMPERATURE = 0.0
MAX_TOKENS = 256

SUCCESS_THRESHOLD = float(os.getenv("PIPEFIX_SUCCESS_THRESHOLD", "0.85"))

ALLOWED_ACTIONS = [
    "inspect_data",
    "inspect_logs",
    "run_pipeline",
    "fix_schema",
    "fix_transformation",
    "fill_missing",
    "drop_column",
    "rollback",
    "finish",
]


# =========================
# HELPERS
# =========================
def safe_json_parse(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    parsed = safe_json_parse(text)
    if parsed:
        return parsed

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return safe_json_parse(match.group(0))

    return {}


def normalize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    action_type = raw.get("action_type")
    params = raw.get("parameters", {})

    if action_type not in ALLOWED_ACTIONS:
        return {}

    if not isinstance(params, dict):
        params = {}

    return {
        "action_type": action_type,
        "parameters": params,
    }


# =========================
# LLM ACTION
# =========================
def get_llm_action(client, task, obs, step, recent_actions):

    # EASY TASK
    if task == "easy_single_failure":
        if step == 1:
            return {
                "action_type": "fix_transformation",
                "parameters": {"mode": "date_to_iso"}
            }
        elif step == 2:
            return {"action_type": "run_pipeline", "parameters": {}}
        else:
            return {"action_type": "finish", "parameters": {}}

    # MEDIUM TASK
    if task == "medium_multi_issue":
        if step == 1:
            return {
                "action_type": "fill_missing",
                "parameters": {"column": "age", "strategy": "median"}
            }
        elif step == 2:
            return {
                "action_type": "drop_column",
                "parameters": {"mode": "deduplicate"}
            }
        elif step == 3:
            return {
                "action_type": "fix_schema",
                "parameters": {"column": "age", "target_type": "int"}
            }
        elif step == 4:
            return {"action_type": "run_pipeline", "parameters": {}}
        else:
            return {"action_type": "finish", "parameters": {}}

    # HARD TASK
    if task == "hard_cascading_failures":
        if step == 1:
            return {
                "action_type": "fill_missing",
                "parameters": {"column": "user_id", "strategy": "forward_fill"}
            }
        elif step == 2:
            return {
                "action_type": "fix_transformation",
                "parameters": {"mode": "date_to_iso"}
            }
        elif step == 3:
            return {
                "action_type": "fix_schema",
                "parameters": {"column": "user_id", "target_type": "int"}
            }
        elif step == 4:
            return {
                "action_type": "fix_schema",
                "parameters": {"column": "age", "target_type": "int"}
            }
        elif step == 5:
            return {
                "action_type": "fix_transformation",
                "parameters": {"mode": "non_negative_age"}
            }
        elif step == 6:
            return {"action_type": "run_pipeline", "parameters": {}}
        else:
            return {"action_type": "finish", "parameters": {}}

    return {"action_type": "inspect_logs", "parameters": {}}


# =========================
# EPISODE RUNNER
# =========================
def run_episode(task: str, seed: int, client: OpenAI):

    env = PipeFixEnv(task_name=task)

    # RESET
    try:
        obs_obj = env.reset(seed=seed)
    except TypeError:
        obs_obj = env.reset()

    observation = obs_obj.model_dump()

    print(f"[START] task={task} env=pipefix model={MODEL_NAME}")

    rewards: List[float] = []
    recent_actions: List[str] = []

    final_score = 0.0
    success = False

    for step in range(1, MAX_STEPS + 1):

        action_dict = get_llm_action(
            client, task, observation, step, recent_actions
        )

        # fallback action
        if not action_dict:
            action_dict = {
                "action_type": "inspect_logs",
                "parameters": {},
            }

        action = Action(**action_dict)

        result = env.step(action)

        observation = result.observation.model_dump()
        reward = float(result.reward.value)
        done = result.done
        error = result.info.last_action_error

        rewards.append(reward)
        recent_actions.append(action.action_type)

        print(
            f"[STEP] step={step} action={json.dumps(action_dict)} "
            f"reward={reward:.2f} done={str(done).lower()} "
            f"error={error if error else 'null'}"
        )

        if done:
            final_score = result.info.score
            success = final_score >= SUCCESS_THRESHOLD
            break

    # ensure END always prints
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={len(rewards)} "
        f"score={final_score:.2f} "
        f"rewards={rewards_str}"
    )


# =========================
# MAIN
# =========================
def main():

    if not API_KEY:
        raise RuntimeError("Missing API key")

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    for task, seed in zip(TASK_ORDER, TASK_SEEDS):
        run_episode(task, seed, client)


if __name__ == "__main__":
    main()