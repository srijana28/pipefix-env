from __future__ import annotations

from typing import List
from .models import PipelineState


def _normalized_efficiency(step_count: int, optimal_steps: int) -> float:
    if step_count <= optimal_steps:
        return 1.0
    overshoot = step_count - optimal_steps
    return max(0.0, 1.0 - (overshoot / max(optimal_steps, 1)))


def grade(state: PipelineState, final_data: List[dict]) -> float:
    score = 0.0

    # Pipeline success
    if state.pipeline_runs and state.pipeline_status == "success":
        score += 0.4

    # Output correctness (flexible, not exact match)
    if state.output_correct:
        score += 0.4

    # Efficiency
    score += 0.2 * _normalized_efficiency(
        state.step_count, state.optimal_steps
    )

    # Clamp strictly between (0,1)
    return max(0.01, min(0.99, score))