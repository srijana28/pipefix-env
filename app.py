from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from env import Action, PipeFixEnv


class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    seed: Optional[int] = None


app = FastAPI(title="PipeFix OpenEnv API", version="1.0.0")
DEFAULT_TASK = os.getenv("PIPEFIX_TASK", "easy_single_failure")
ENV = PipeFixEnv(task_name=DEFAULT_TASK)


@app.get("/")
def root() -> dict:
    return {"name": "PipeFix", "status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/reset")
def reset(payload: Optional[ResetRequest] = None) -> dict:
    if payload is None:
        obs = ENV.reset()
    else:
        obs = ENV.reset(task_name=payload.task_name, seed=payload.seed)
    return obs.model_dump()


@app.post("/step")
def step(action: Action) -> dict:
    result = ENV.step(action)
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    return ENV.state().model_dump()


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": PipeFixEnv.available_tasks()}
