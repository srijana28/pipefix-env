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
            obs = self._to_observation()
            return StepResult(
                observation=obs,
                reward=Reward(value=0.0, components={"episode_done": 0.0}),
                done=True,
                info=StepInfo(task_name=state.task_name, last_action_error=None, raw_reward=0.0, score=self._score()),
            )

        prev_state = deepcopy(state)
        state.step_count += 1
        state.action_history.append(action.action_type)
        last_action_error: Optional[str] = None

        if action.action_type in {"fix_schema", "fix_transformation", "fill_missing", "drop_column"}:
            # Keep rollback snapshots for all mutating actions.
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
            state.logs.extend(run_logs[1:])  # skip duplicated run-start line
            state.pipeline_runs = True
            if status == "success":
                state.output_correct = self._is_output_correct()
        elif action.action_type == "rollback":
            if not self._snapshots:
                last_action_error = "No previous snapshot to rollback"
                state.logs.append(f"[ERROR] {last_action_error}")
                state.error_message = last_action_error
            else:
                old_data, old_schema, old_fixes = self._snapshots.pop()
                state.dataset = old_data
                state.schema = old_schema
                state.fixes_applied = old_fixes
                state.logs.append("[INFO] Rollback completed")
        elif action.action_type == "finish":
            state.done = True
            state.logs.append("[INFO] Agent finished episode")
        else:
            last_action_error = "Unknown action_type"
            state.logs.append("[ERROR] Unknown action_type")
            state.error_message = "Unknown action_type"

        if state.step_count >= state.max_steps:
            state.done = True
            state.logs.append("[WARN] Max steps reached")

        if state.pipeline_status == "success" and self._is_output_correct():
            state.done = True
            state.current_stage = "done"

        raw_reward, components = self._compute_reward(prev_state, state, action, last_action_error)
        score = self._score()
        clipped = max(0.0, min(1.0, raw_reward))
        obs = self._to_observation()
        info = StepInfo(task_name=state.task_name, last_action_error=last_action_error, raw_reward=raw_reward, score=score)
        return StepResult(
            observation=obs,
            reward=Reward(value=clipped, components=components),
            done=state.done,
            info=info,
        )

    def _to_observation(self) -> Observation:
        assert self._state is not None
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
            return f"fill_missing:{action.parameters.get('column')}:{action.parameters.get('strategy', 'median')}"
        if action.action_type == "drop_column":
            if action.parameters.get("mode") == "deduplicate":
                return "drop_column:deduplicate"
            return f"drop_column:{action.parameters.get('column')}"
        return action.action_type

    def _has_progressed(self, prev: PipelineState, new: PipelineState) -> bool:
        order = {"clean": 0, "transform": 1, "validate": 2, "output": 3, "done": 4}
        return order.get(new.current_stage, 0) >= order.get(prev.current_stage, 0) and (
            new.pipeline_status == "success" or new.pipeline_status != prev.pipeline_status
        )

    def _quality_score(self, state: PipelineState) -> float:
        # Higher quality means fewer hard failures in data+schema.
        score = 0.0
        if all(all(v is not None for v in row.values()) for row in state.dataset):
            score += 0.25
        if state.schema.get("date") == "date_iso":
            score += 0.25
        if state.schema.get("age") == "int" or all(isinstance(r.get("age"), int) for r in state.dataset):
            score += 0.25
        if state.schema.get("user_id") == "int" or all(isinstance(r.get("user_id"), int) for r in state.dataset if r.get("user_id") is not None):
            score += 0.25
        return score

    def _compute_reward(
        self,
        prev_state: PipelineState,
        new_state: PipelineState,
        action: Action,
        last_action_error: Optional[str],
    ) -> Tuple[float, Dict[str, float]]:
        reward = 0.0
        components: Dict[str, float] = {}

        error_resolved = prev_state.error_message is not None and new_state.error_message is None
        pipeline_progressed = self._has_progressed(prev_state, new_state)
        quality_delta = self._quality_score(new_state) - self._quality_score(prev_state)
        repeated_action = (
            len(new_state.action_history) >= 2
            and new_state.action_history[-1] == new_state.action_history[-2]
            and action.action_type not in {"run_pipeline", "finish"}
        )
        bad_action = last_action_error is not None

        if error_resolved:
            reward += 0.3
            components["error_resolved"] = 0.3
        if pipeline_progressed:
            reward += 0.2
            components["pipeline_progressed"] = 0.2
        if quality_delta > 0:
            increment = min(0.2, quality_delta)
            reward += increment
            components["data_quality_improved"] = increment
        if bad_action:
            reward -= 0.3
            components["bad_action"] = -0.3
        if repeated_action:
            reward -= 0.1
            components["repeated_action"] = -0.1

        if new_state.done and new_state.pipeline_status == "success" and self._is_output_correct():
            if new_state.step_count <= new_state.optimal_steps:
                reward += 0.2
                components["efficiency_bonus"] = 0.2
        if action.action_type in {"inspect_data", "inspect_logs"} and new_state.step_count > 2:
            reward -= 0.05
            components["unnecessary_action_penalty"] = -0.05

        return reward, components

    def _is_output_correct(self) -> bool:
        assert self._state is not None
        final_data = TASKS[self._task_name]["final_data"]
        required = TASKS[self._task_name]["required_fix_order"]
        prefix = self._state.fixes_applied[: len(required)]
        return self._state.dataset == final_data and prefix == required

    def _score(self) -> float:
        assert self._state is not None
        return grade(self._state, TASKS[self._task_name]["final_data"])

    @property
    def task_name(self) -> str:
        return self._task_name

    @staticmethod
    def available_tasks() -> List[str]:
        return list_task_names()
