from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ActionType = Literal[
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

PipelineStatus = Literal["running", "failed", "success"]
StageName = Literal["clean", "transform", "validate", "output", "done"]


class Action(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    current_stage: StageName
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    data_sample: List[Dict[str, Any]] = Field(default_factory=list)
    schema: Dict[str, str] = Field(default_factory=dict)
    pipeline_status: PipelineStatus = "failed"
    step_count: int = 0


class Reward(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    value: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)


class StepInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    last_action_error: Optional[str] = None
    task_name: str
    raw_reward: float = 0.0
    score: float = 0.0


class StepResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    observation: Observation
    reward: Reward
    done: bool
    info: StepInfo


class PipelineState(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    task_name: str
    current_stage: StageName = "clean"
    pipeline_status: PipelineStatus = "failed"
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    dataset: List[Dict[str, Any]] = Field(default_factory=list)
    schema: Dict[str, str] = Field(default_factory=dict)
    output_correct: bool = False
    pipeline_runs: bool = False
    step_count: int = 0
    max_steps: int = 24
    fixes_applied: List[str] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    done: bool = False
    optimal_steps: int = 8
