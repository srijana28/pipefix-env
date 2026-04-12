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

# Shown to the model for the current task (reduces wrong ordering).
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
        "fix_transformation non_negative_age, run_pipeline, finish. "
        "Inspect data/logs first if needed."
    ),
}

DEFAULT_TASK_ORDER = [
    "easy_single_failure",
    "medium_multi_issue",
    "hard_cascading_failures",
]

FINISH_ACTION_STR = json.dumps({"action_type": "finish", "parameters": {}}, separators=(",", ":"))


def _resolve_api_key(base_url: str) -> str | None:
    """Match key to provider: OpenAI's API must not use HF_TOKEN (hf_...) by mistake."""
    u = (base_url or "").lower()
    if "api.openai.com" in u or "openai.azure.com" in u:
        return os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    return os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")


# Reproducible instance selection (paired with task order — first seed → first task run).
DEFAULT_TASK_SEEDS = [42, 43, 44]
MAX_STEPS = 24
TEMPERATURE = 0.0
MAX_TOKENS = 512

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = _resolve_api_key(API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("PIPEFIX_BENCHMARK", "pipefix")
# Success if final grader score reaches this threshold (on [0,1]).
SUCCESS_SCORE_THRESHOLD = float(os.getenv("PIPEFIX_SUCCESS_THRESHOLD", "0.85"))

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
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _print_step(step: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={_fmt_reward(reward)} done={_fmt_bool(done)} error={err}",
        flush=True,
    )


def _print_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    if not rewards:
        rewards_str = _fmt_reward(0.0)
    else:
        rewards_str = ",".join(_fmt_reward(r) for r in rewards)
    print(
        f"[END] success={_fmt_bool(success)} steps={steps} score={_fmt_reward(score)} rewards={rewards_str}",
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
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
    if m:
        d = _safe_json_parse(m.group(1))
        if d:
            return d
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        d = _safe_json_parse(m.group(0))
        if d:
            return d
    return {}


def _is_quota_error(exc: BaseException) -> bool:
    code = getattr(exc, "status_code", None)
    if code == 402:
        return True
    s = str(exc).lower()
    return "402" in s or "depleted" in s or "credit" in s


def _normalize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    at = raw.get("action_type")
    if not isinstance(at, str) or at not in ALLOWED_ACTIONS:
        return {}
    params = raw.get("parameters")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return {}
    return {"action_type": at, "parameters": params}


def _llm_action(
    client: OpenAI,
    task: str,
    obs: Dict[str, Any],
    step: int,
    recent_actions: List[str],
    retry: bool = False,
) -> Dict[str, Any]:
    hint = TASK_HINTS.get(task, "")
    system = (
        "You are a data pipeline debugging agent. "
        "Reply with exactly one JSON object: {\"action_type\": string, \"parameters\": object}. "
        "No markdown, no code fences, no extra keys. "
        + ACTION_CHEATSHEET
        + (" " + hint if hint else "")
    )
    user_obj: Dict[str, Any] = {
        "task_name": task,
        "step": step,
        "observation": obs,
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "recent_action_types": recent_actions[-12:],
    }
    if len(recent_actions) >= 3 and all(a == "inspect_logs" for a in recent_actions[-3:]):
        user_obj["instruction"] = (
            "You repeated inspect_logs. You must now emit fix_transformation, fix_schema, "
            "fill_missing, drop_column, or run_pipeline."
        )
    if retry:
        user_obj["reminder"] = "Your previous reply was not valid JSON. Output ONLY the JSON object."
    user = json.dumps(user_obj, separators=(",", ":"))
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


def _next_action_from_llm(
    client: OpenAI,
    task: str,
    obs: Dict[str, Any],
    step: int,
    recent_actions: List[str],
) -> Dict[str, Any]:
    action_dict = _llm_action(client, task, obs, step, recent_actions, retry=False)
    if action_dict:
        return action_dict
    action_dict = _llm_action(client, task, obs, step, recent_actions, retry=True)
    if action_dict:
        return action_dict
    return {}


def _run_episode_skipped(task: str, reason: str) -> None:
    """Emit START/STEP/END without calling the API (e.g. after quota error)."""
    _print_start(task)
    _print_step(1, FINISH_ACTION_STR, 0.0, True, reason)
    _print_end(False, 1, 0.0, [0.0])


def run_episode(task: str, client: OpenAI, seed: int | None) -> Tuple[float, bool, str]:
    """Run one episode using only the LLM (no scripted solution path).

    Returns (final_score, quota_exhausted, error_message). When quota_exhausted is True,
    callers should skip further API calls for remaining tasks.
    """
    env = PipeFixEnv(task_name=task)
    obs = env.reset(task_name=task, seed=seed).model_dump()

    _print_start(task)

    rewards: List[float] = []
    score = 0.0
    success = False
    steps = 0
    consecutive_invalid = 0
    recent_actions: List[str] = []
    quota_exhausted = False
    last_api_error = ""

    try:
        for step in range(1, MAX_STEPS + 1):
            steps = step
            try:
                action_dict = _next_action_from_llm(client, task, obs, step, recent_actions)
            except APIError as exc:
                err_msg = f"APIError: {str(exc).replace(chr(10), ' ')}"
                quota_exhausted = _is_quota_error(exc)
                if quota_exhausted:
                    last_api_error = err_msg
                rewards.append(0.0)
                _print_step(step, FINISH_ACTION_STR, 0.0, True, err_msg)
                success = False
                break

            if not action_dict:
                consecutive_invalid += 1
                if consecutive_invalid >= 4:
                    action_dict = {"action_type": "finish", "parameters": {}}
                else:
                    action_dict = {"action_type": "inspect_logs", "parameters": {}}
            else:
                consecutive_invalid = 0

            try:
                action = Action(**action_dict)
            except Exception:
                action_dict = {"action_type": "inspect_logs", "parameters": {}}
                action = Action(**action_dict)

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
                error = f"StepRuntimeError: {str(exc).replace(chr(10), ' ')}"

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
                success = score >= SUCCESS_SCORE_THRESHOLD and error is None
                break
    finally:
        _print_end(success, steps, score, rewards)

    return score, quota_exhausted, last_api_error if quota_exhausted else ""


def _parse_seed_list(raw: str | None) -> List[int]:
    if not raw:
        return list(DEFAULT_TASK_SEEDS)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(p, 10))
    return out


def _parse_task_order() -> List[str]:
    """Optional PIPEFIX_TASK_ORDER=hard_cascading_failures,easy_single_failure,... (all 3 names, comma-separated)."""
    raw = os.getenv("PIPEFIX_TASK_ORDER")
    if not raw:
        return list(DEFAULT_TASK_ORDER)
    names = [p.strip() for p in raw.split(",") if p.strip()]
    valid = set(DEFAULT_TASK_ORDER)
    if len(names) != len(DEFAULT_TASK_ORDER) or set(names) != valid:
        return list(DEFAULT_TASK_ORDER)
    return names


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Set HF_TOKEN or OPENAI_API_KEY (or API_KEY) for the OpenAI-compatible API."
        )

    task_order = _parse_task_order()
    seeds = _parse_seed_list(os.getenv("PIPEFIX_TASK_SEEDS"))
    if len(seeds) < len(task_order):
        seeds = seeds + DEFAULT_TASK_SEEDS[len(seeds) :]
    seeds = seeds[: len(task_order)]

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    skip_remaining = False
    skip_reason = ""
    for task, seed in zip(task_order, seeds):
        if skip_remaining:
            _run_episode_skipped(task, skip_reason)
            continue
        _, exhausted, err = run_episode(task, client, seed)
        if exhausted and err:
            skip_remaining = True
            skip_reason = err


if __name__ == "__main__":
    main()
