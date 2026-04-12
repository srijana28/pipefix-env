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

DEFAULT_TASK_SEEDS = [42, 43, 44]

MAX_STEPS = 24
TEMPERATURE = 0.0
MAX_TOKENS = 512

FINISH_ACTION_STR = json.dumps(
    {"action_type": "finish", "parameters": {}},
    separators=(",", ":"),
)

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
)
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

SUCCESS_SCORE_THRESHOLD = float(
    os.getenv("PIPEFIX_SUCCESS_THRESHOLD", "0.85")
)

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


def _fmt_bool(v: bool) -> str:
    return "true" if v else "false"


def _fmt_reward(v: float) -> str:
    return f"{max(0.0, min(1.0, v)):.2f}"


def _print_start(task: str) -> None:
    print(f"[START] task={task} model={MODEL_NAME}", flush=True)


def _print_step(
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    error: str | None
) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={_fmt_reward(reward)} done={_fmt_bool(done)} error={err}",
        flush=True,
    )


def _print_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float]
) -> None:
    rewards_str = ",".join(_fmt_reward(r) for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={_fmt_bool(success)} "
        f"steps={steps} score={_fmt_reward(score)} rewards={rewards_str}",
        flush=True,
    )


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

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        d = _safe_json_parse(match.group(0))
        if d:
            return d

    return {}


def _normalize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    action_type = raw.get("action_type")

    if not isinstance(action_type, str):
        return {}

    if action_type not in ALLOWED_ACTIONS:
        return {}

    parameters = raw.get("parameters", {})

    if not isinstance(parameters, dict):
        return {}

    return {
        "action_type": action_type,
        "parameters": parameters,
    }


def _llm_action(
    client: OpenAI,
    task: str,
    obs: Dict[str, Any],
    step: int,
    recent_actions: List[str],
) -> Dict[str, Any]:

    system = (
        "You are a data pipeline debugging agent. "
        "Reply with exactly one JSON object: "
        "{\"action_type\": string, \"parameters\": object}. "
        "No markdown. No extra text. "
        + ACTION_CHEATSHEET
        + " "
        + TASK_HINTS.get(task, "")
    )

    user = json.dumps({
        "task_name": task,
        "step": step,
        "observation": obs,
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "recent_action_types": recent_actions[-10:],
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


def run_episode(
    task: str,
    client: OpenAI,
    seed: int | None
) -> Tuple[float, bool, str]:

    rewards: List[float] = []
    score = 0.0
    success = False
    recent_actions: List[str] = []

    _print_start(task)

    try:
        env = PipeFixEnv(task_name=task)

        # FIXED RESET BLOCK
        try:
            try:
                reset_result = env.reset(seed=seed)
            except TypeError:
                reset_result = env.reset()

            obs = reset_result.model_dump()

        except Exception as exc:
            print(f"[ERROR] reset failed: {exc}", flush=True)
            _print_end(False, 0, 0.0, [])
            return 0.0, False, str(exc)

        for step in range(1, MAX_STEPS + 1):

            try:
                action_dict = _llm_action(
                    client,
                    task,
                    obs,
                    step,
                    recent_actions
                )
            except APIError as exc:
                _print_step(
                    step,
                    FINISH_ACTION_STR,
                    0.0,
                    True,
                    str(exc)
                )
                break

            if not action_dict:
                action_dict = {
                    "action_type": "inspect_logs",
                    "parameters": {}
                }

            try:
                action = Action(**action_dict)
            except Exception:
                action = Action(
                    action_type="inspect_logs",
                    parameters={}
                )

            try:
                result = env.step(action)

                obs = result.observation.model_dump()
                reward = result.reward.value
                done = result.done
                error = result.info.last_action_error
                score = result.info.score

            except Exception as exc:
                reward = 0.0
                done = True
                error = f"StepRuntimeError: {exc}"

            recent_actions.append(action.action_type)
            rewards.append(reward)

            _print_step(
                step,
                json.dumps(action_dict, separators=(",", ":")),
                reward,
                done,
                error,
            )

            if done:
                success = (
                    score >= SUCCESS_SCORE_THRESHOLD
                    and error is None
                )
                break

        _print_end(success, step, score, rewards)

        return score, False, ""

    except Exception as exc:
        print(f"[FATAL ERROR] {exc}", flush=True)
        _print_end(False, 0, 0.0, [])
        return 0.0, False, str(exc)


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Set HF_TOKEN or OPENAI_API_KEY or API_KEY"
        )

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )

    for task, seed in zip(
        DEFAULT_TASK_ORDER,
        DEFAULT_TASK_SEEDS
    ):
        run_episode(task, client, seed)


if __name__ == "__main__":
    main()
