---
license: mit
title: PipeFix Environment
emoji: 🐳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - fastapi
  - docker
  - reinforcement-learning
---

# PipeFix — Data Pipeline Debugging OpenEnv

PipeFix is a real-world environment where an agent debugs a broken analytics data pipeline by inspecting logs, fixing schema and transformations, then rerunning the pipeline until output is correct.

## Live deployment

| | Link |
|---|------|
| **Source repository** | [github.com/srijana28/pipefix-env](https://github.com/srijana28/pipefix-env) |
| **Hugging Face Space** | [huggingface.co/spaces/srijana28/pipefix-env](https://huggingface.co/spaces/srijana28/pipefix-env) |
| **Running app (API)** | [srijana28-pipefix-env.hf.space](https://srijana28-pipefix-env.hf.space) |

Docker Space, tag **`openenv`**, port **7860**. Health check: `GET /health` · root: `GET /`.

### Sync local changes to GitHub (and HF rebuild)

If this folder is a clone of the repo above, from the project root:

```bash
git status
git add -A
git commit -m "Update PipeFix environment"
git push origin main
```

Ensure **`.venv` is not committed** (see `.gitignore`). After `git push`, open the [Space](https://huggingface.co/spaces/srijana28/pipefix-env) — it should rebuild from `main` automatically when the Space is linked to this GitHub repository.

## Why this environment

Data teams routinely face production failures in ETL/ELT jobs. PipeFix models this workflow:

`Raw Data -> Clean -> Transform -> Validate -> Output`

Agents must perform sequential diagnosis and correction under step limits, with deterministic outcomes and reproducible grading.

## OpenEnv metadata

Project metadata for tooling and validation lives in **`openenv.yaml`** (environment name: **`pipefix`**, entrypoint class `PipeFixEnv` in `env.environment`). Validate locally:

```bash
pip install openenv-core
openenv validate
```

## Baseline integrity

The bundled **`inference.py`** is **LLM-only**: there is no hidden scripted trajectory that solves tasks when the model fails. Invalid JSON triggers a retry with a stricter prompt, then benign `inspect_logs` steps before the episode ends—never an oracle fix sequence.

## OpenEnv API

PipeFix implements the standard API:

- `reset(task_name?, seed?) -> Observation`
- `step(Action) -> {observation, reward, done, info}`
- `state() -> PipelineState`

HTTP endpoints (FastAPI):

- `POST /reset` — optional JSON body: `{"task_name": "...", "seed": 42}`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /health`

### Instance selection (`seed`)

Tasks load data from fixed **variants**. For `hard_cascading_failures`, two labeled instances (`batch_a`, `batch_b`) exist; `reset(seed=k)` chooses `variants[k % num_variants]`. Observations include **`instance_id`** (variant index).

## Observation space

`Observation` includes:

- `current_stage: clean | transform | validate | output | done`
- `error_message: Optional[str]`
- `logs: List[str]`
- `data_sample: List[Dict[str, Any]]`
- `schema: Dict[str, str]`
- `pipeline_status: running | failed | success`
- `step_count: int`
- `instance_id: int` — index of the active task variant

## Action space

`Action`:

- `action_type` in:
  - `inspect_data`
  - `inspect_logs`
  - `run_pipeline`
  - `fix_schema`
  - `fix_transformation`
  - `fill_missing`
  - `drop_column`
  - `rollback`
  - `finish`
- `parameters: Dict[str, Any]`

## Tasks and difficulty

1. **`easy_single_failure` (Easy)** — Single transform failure (invalid date format).
2. **`medium_multi_issue` (Medium)** — Missing values, duplicates, and schema mismatch for `age`.
3. **`hard_cascading_failures` (Hard)** — Cascading errors (schema, dates, negative ages, missing `user_id`). Two dataset instances; same fix *types* but different rows.

All tasks are deterministic; graders use canonical row sets per variant.

## Reward design

Step reward is clipped to **`[0.0, 1.0]`** with shaping:

- +0.3 if previous error is resolved
- +0.2 if pipeline progress increases
- +0.2 max for data quality improvement
- -0.3 for clearly invalid actions
- -0.1 for repeated non-productive action
- +0.2 efficiency bonus if solved within optimal steps
- -0.05 unnecessary inspect action after the initial phase

Episode ends on successful correct output, explicit `finish`, or max step budget.

## Grader

Final deterministic score in **`[0.0, 1.0]`** inclusive:

- +0.4 if the pipeline run completes with `success`
- +0.4 if output matches the canonical rows for the active variant **and** the required fix-prefix matches
- +0.2 weighted by efficiency vs `optimal_steps`

## Baseline scores (reference run)

Reproducible with default settings (`temperature=0` in `inference.py`), Hugging Face Inference Router, and seeds `42,43,44` for the three tasks in default order.

| Task | Model | Final score | Steps | Success (`[END]`) |
|------|--------|-------------|-------|-------------------|
| `easy_single_failure` | `Qwen/Qwen2.5-72B-Instruct` | 1.00 | 2 | true |
| `medium_multi_issue` | `Qwen/Qwen2.5-72B-Instruct` | 1.00 | 4 | true |
| `hard_cascading_failures` | `Qwen/Qwen2.5-72B-Instruct` | 1.00 | 7 | true |

**Environment:** `API_BASE_URL=https://router.huggingface.co/v1`, `HF_TOKEN` set, default `PIPEFIX_TASK_ORDER` and `PIPEFIX_TASK_SEEDS`. Your scores may vary slightly with a different model or provider.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t pipefix-env .
docker run --rm -p 7860:7860 pipefix-env
```

## Hugging Face Spaces

Use this repository as a **Docker** Space, add the **`openenv`** tag, and expose port **`7860`**.

## Baseline inference

`inference.py` is at the repository root and uses the **OpenAI** Python client against a compatible endpoint.

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | API key for Hugging Face router (default base URL) |
| `OPENAI_API_KEY` | Required when `API_BASE_URL` is `https://api.openai.com/v1` (`HF_TOKEN` is not used for OpenAI, so a leftover `hf_` token will not override your `sk-` key) |
| `API_BASE_URL` | Base URL (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Model id (default: `Qwen/Qwen2.5-72B-Instruct`) |
| `PIPEFIX_BENCHMARK` | Name in `[START]` line (default: `pipefix`) |
| `PIPEFIX_TASK_SEEDS` | Comma-separated seeds for each run task, e.g. `42,43,44` (order follows `PIPEFIX_TASK_ORDER`) |
| `PIPEFIX_TASK_ORDER` | Optional permutation of all three task ids, e.g. `hard_cascading_failures,medium_multi_issue,easy_single_failure` |
| `PIPEFIX_SUCCESS_THRESHOLD` | `[END] success=true` if final score ≥ this (default `0.85`) |

```bash
set HF_TOKEN=your_token
python inference.py
```

OpenAI example (real `sk-...` key; unset `HF_TOKEN` if it conflicts):

```bash
set API_BASE_URL=https://api.openai.com/v1
set OPENAI_API_KEY=sk-...
set MODEL_NAME=gpt-4o-mini
python inference.py
```

Stdout format (required for evaluation):

- `[START] task=... env=... model=...`
- One `[STEP]` per `step()`
- `[END] success=... steps=... score=... rewards=...`

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE) and the Space card metadata above.
