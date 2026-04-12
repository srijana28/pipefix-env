from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

from env import Action, PipeFixEnv


# =========================
# CONFIG (FIXED)
# =========================
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4.1-mini"

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
# ✅ LLM ACTION (FIXED)
# =========================
def get_llm_action(client, task, obs, step, recent_actions):

    prompt = f"""
You are an AI agent fixing a broken data pipeline.

Task: {task}
Step: {step}

Current Observation:
{json.dumps(obs)}

Recent Actions:
{recent_actions}

Choose the BEST next action.

Allowed actions:
{ALLOWED_ACTIONS}

Respond ONLY in JSON format:
{{
    "action_type": "...",
    "parameters": {{}}
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a pipeline debugging assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = response.choices[0].message.content
        parsed = extract_json(text)
        action = normalize_action(parsed)

        if not action:
            return {
                "action_type": "inspect_logs",
                "parameters": {}
            }

        return action

    except Exception as e:
        print("LLM ERROR:", str(e))
        return {
            "action_type": "inspect_logs",
            "parameters": {}
        }


# =========================
# EPISODE RUNNER
# =========================
def run_episode(task: str, seed: int, client: OpenAI):

    env = PipeFixEnv(task_name=task)

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

    if not API_KEY or not API_BASE_URL:
        raise RuntimeError("Missing API_KEY or API_BASE_URL")

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    for task, seed in zip(TASK_ORDER, TASK_SEEDS):
        run_episode(task, seed, client)


if __name__ == "__main__":
    main()
