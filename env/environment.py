from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from .grader import grade
from .models import Action, Observation, PipelineState, Reward, StepInfo, StepResult
from .pipeline import apply_fix, dataset_sample, run_pipeline
from .tasks import TASKS, list_task_names


class PipeFixEnv:
    def __init__(self, task_name: str = "easy_single_failure", max_steps: int = 24):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {list_task_names()}")
        self._task_name = task_name
        self._max_steps = max_steps
        self._state: Optional[PipelineState] = None
        self._snapshots: List[Tuple[List[dict], Dict[str, str], List[str]]] = []
        self.reset(task_name=task_name)

    def reset(self, task_name: Optional[str] = None) -> Observation:
        if task_name is not None:
            if task_name not in TASKS:
                raise ValueError(f"Unknown task: {task_name}. Available: {list_task_names()}")
            self._task_name = task_name

        task = TASKS[self._task_name]
        self._state = PipelineState(
            task_name=self._task_name,
            current_stage="clean",
            pipeline_status="failed",
            error_message="Pipeline has not been run yet",
            logs=["[INFO] Environment reset", f"[INFO] Task loaded: {self._task_name}"],
            dataset=deepcopy(task["dataset"]),
            schema=deepcopy(task["schema"]),
            output_correct=False,
            pipeline_runs=False,
            step_count=0,
            max_steps=self._max_steps,
            fixes_applied=[],
            action_history=[],
            done=False,
            optimal_steps=task["optimal_steps"],
        )
        self._snapshots = []
        return self._to_observation()

    def state(self) -> PipelineState:
        if self._state is None:
            raise RuntimeError("Environment not initialized")
        return deepcopy(self._state)

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment not initialized")

        state = self._state

        if state.done:
            return StepResult(
                observation=self._to_observation(),
                reward=Reward(value=0.0, components={"episode_done": 0.0}),
                done=True,
                info=StepInfo(task_name=state.task_name, last_action_error=None, raw_reward=0.0, score=self._score()),
            )

        prev_state = deepcopy(state)
        state.step_count += 1
        state.action_history.append(action.action_type)
        last_action_error: Optional[str] = None

        # ---------------- FIX ACTIONS ----------------
        if action.action_type in {"fix_schema", "fix_transformation", "fill_missing", "drop_column"}:
            self._snapshots.append((deepcopy(state.dataset), deepcopy(state.schema), deepcopy(state.fixes_applied)))

            new_data, new_schema, fix_error = apply_fix(
                state.dataset, state.schema, action.action_type, action.parameters
            )

            if fix_error:
                last_action_error = fix_error
                state.logs.append(f"[ERROR] {fix_error}")
                state.error_message = fix_error
            else:
                state.dataset = new_data
                state.schema = new_schema
                fix_marker = self._fix_marker(action)
                state.fixes_applied.append(fix_marker)
                state.logs.append(f"[INFO] Applied fix: {fix_marker}")

        elif action.action_type == "inspect_data":
            state.logs.append("[INFO] Data inspected")

        elif action.action_type == "inspect_logs":
            state.logs.append("[INFO] Logs inspected")

        elif action.action_type == "run_pipeline":
            stage, status, error, run_logs = run_pipeline(state.dataset, state.schema)
            state.current_stage = stage
            state.pipeline_status = status
            state.error_message = error
            state.logs.extend(run_logs[1:])
            state.pipeline_runs = True

            if status == "success":
                state.output_correct = self._is_output_correct()

        elif action.action_type == "rollback":
            if not self._snapshots:
                last_action_error = "No snapshot to rollback"
                state.logs.append(f"[ERROR] {last_action_error}")
                state.error_message = last_action_error
            else:
                data, schema, fixes = self._snapshots.pop()
                state.dataset = data
                state.schema = schema
                state.fixes_applied = fixes
                state.logs.append("[INFO] Rollback completed")

        elif action.action_type == "finish":
            state.done = True
            state.logs.append("[INFO] Agent finished episode")

        else:
            last_action_error = "Unknown action"
            state.logs.append("[ERROR] Unknown action")
            state.error_message = last_action_error

        # ---------------- DONE CONDITIONS ----------------
        if state.step_count >= state.max_steps:
            state.done = True

        if state.pipeline_status == "success" and self._is_output_correct():
            state.done = True
            state.current_stage = "done"

        # ---------------- REWARD ----------------
        raw_reward = 0.0
        if last_action_error:
            raw_reward -= 0.3
        elif state.pipeline_status == "success":
            raw_reward += 0.5

        obs = self._to_observation()
        score = self._score()

        return StepResult(
            observation=obs,
            reward=Reward(value=max(0.0, min(1.0, raw_reward)), components={}),
            done=state.done,
            info=StepInfo(task_name=state.task_name, last_action_error=last_action_error, raw_reward=raw_reward, score=score),
        )

    # ---------------- HELPERS ----------------

    def _to_observation(self) -> Observation:
        s = self._state
        return Observation(
            current_stage=s.current_stage,
            error_message=s.error_message,
            logs=s.logs[-8:],
            data_sample=dataset_sample(s.dataset),
            schema=deepcopy(s.schema),
            pipeline_status=s.pipeline_status,
            step_count=s.step_count,
        )

    def _fix_marker(self, action: Action) -> str:
        if action.action_type == "fix_transformation":
            return f"fix_transformation:{action.parameters.get('mode')}"
        if action.action_type == "fix_schema":
            return f"fix_schema:{action.parameters.get('column')}:{action.parameters.get('target_type')}"
        if action.action_type == "fill_missing":
            return f"fill_missing:{action.parameters.get('column')}:{action.parameters.get('strategy')}"
        if action.action_type == "drop_column":
            if action.parameters.get("mode") == "deduplicate":
                return "drop_column:deduplicate"
        return action.action_type

    def _is_output_correct(self) -> bool:
        assert self._state is not None
        state = self._state

        if state.pipeline_status != "success":
            return False

        if any(any(v is None for v in row.values()) for row in state.dataset):
            return False

        for row in state.dataset:
            if not isinstance(row.get("user_id"), int):
                return False
            if not isinstance(row.get("age"), int):
                return False
            if row.get("age", 0) < 0:
                return False

        return True

    def _score(self) -> float:
        assert self._state is not None
        return grade(self._state, TASKS[self._task_name]["final_data"])

    @staticmethod
    def available_tasks() -> List[str]:
        return list_task_names()