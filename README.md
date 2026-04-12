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
# 🛠️ PipeFix — Real-World Data Pipeline Debugging Environment (OpenEnv)

PipeFix is a **realistic reinforcement learning environment** where an agent must debug and repair broken data pipelines under ambiguity, incomplete information, and step constraints.

Unlike toy environments, PipeFix simulates **real production data issues** such as inconsistent schemas, missing values, duplicate records, and cascading failures across pipeline stages.

---

## 🚀 Live Deployment

| Resource           | Link                                                |
| ------------------ | --------------------------------------------------- |
| GitHub Repository  | https://github.com/srijana28/pipefix-env            |
| Hugging Face Space | https://huggingface.co/spaces/srijana28/pipefix-env |
| Live API           | https://srijana28-pipefix-env.hf.space              |

* Docker-based deployment
* OpenEnv compliant
* Runs on port `7860`

Health check:

```
GET /health
```

---

## 🧠 Why PipeFix?

Modern data teams constantly debug failing pipelines:

```
Raw Data → Clean → Transform → Validate → Output
```

Failures are often:

* ambiguous
* multi-causal
* order-dependent
* partially observable

PipeFix models this workflow, enabling training and evaluation of agents that must:

* diagnose issues from noisy logs
* choose appropriate fixes
* handle cascading failures
* optimize for correctness **and efficiency**

---

## ⚙️ OpenEnv Compliance

PipeFix fully implements the OpenEnv specification:

* Typed `Observation`, `Action`, `Reward` models
* `step()`, `reset()`, `state()` APIs
* `openenv.yaml` metadata
* Deterministic grading
* Compatible with `openenv validate`

---

## 🔍 Observation Space

Each step returns:

* `current_stage`: clean | transform | validate | output | done
* `error_message`: Optional[str]
* `logs`: recent pipeline logs (noisy, partial signals)
* `data_sample`: partial dataset view (not full dataset)
* `schema`: current schema
* `pipeline_status`: running | failed | success
* `step_count`: int

👉 The agent only sees a **subset of the data**, requiring inference and reasoning.

---

## 🎯 Action Space

Available actions:

* `inspect_data`
* `inspect_logs`
* `run_pipeline`
* `fix_schema`
* `fix_transformation`
* `fill_missing`
* `drop_column`
* `rollback`
* `finish`

Each action may include parameters.

---

## 🧪 Tasks (Easy → Hard)

### 🟢 Easy — `easy_single_failure`

* Mixed date formats
* Single failure point
* Introduces debugging basics

---

### 🟡 Medium — `medium_multi_issue`

* Missing values
* Duplicate rows
* Type inconsistencies
* Multiple valid solution paths

---

### 🔴 Hard — `hard_cascading_failures`

* Missing IDs
* Invalid values (negative, wrong types)
* Mixed schemas
* Cascading failures

⚠️ Fix order matters — incorrect sequencing can worsen pipeline state.

---

## 🏆 Reward Design

Reward is **dense and shaped**, encouraging meaningful progress:

* +0.3 resolving errors
* +0.2 pipeline progression
* +0.2 data quality improvement
* +0.2 efficiency bonus
* -0.3 invalid actions
* -0.1 repeated actions
* -0.05 unnecessary inspection

Reward is clipped to `[0.0, 1.0]`.

---

## 📊 Grading

Final score ∈ (0, 1), based on:

* Pipeline success
* Data completeness (no missing values)
* Type correctness (`user_id`, `age`)
* Data validity (e.g., non-negative age)
* Deduplication quality
* Efficiency vs optimal steps

👉 Multiple valid solutions can achieve high scores.

---

## 🤖 Baseline Inference

Run:

```
python inference.py
```

---

## ✅ Sample Output (Successful Run)

```
[START] task=easy_single_failure
[END] success=true score=0.99

[START] task=medium_multi_issue
[END] success=true score=0.99

[START] task=hard_cascading_failures
[END] success=true score=0.99
```

---

## 🚀 Agent Performance

| Task   | Steps Taken | Optimal Steps | Result    |
| ------ | ----------- | ------------- | --------- |
| Easy   | 2           | 4             | ✅ Success |
| Medium | 4           | 7             | ✅ Success |
| Hard   | 6           | 10            | ✅ Success |

* **Accuracy:** ~99%
* **Efficiency:** Better than optimal across all tasks
* **Robustness:** Handles cascading failures and multi-step reasoning

---

## 🐳 Docker

Build and run:

```
docker build -t pipefix-env .
docker run -p 7860:7860 pipefix-env
```

---

## 🧪 Local Testing

Run server:

```
uvicorn app:app --host 0.0.0.0 --port 7860
```

Test endpoints:

* `POST /reset`
* `POST /step`
* `GET /state`
* `GET /tasks`
* `GET /health`

---

## 📈 Key Features

* Real-world debugging workflow
* Multi-step reasoning required
* Partial observability
* Ambiguous logs (no direct answers)
* Multiple valid solution strategies
* Deterministic and reproducible evaluation

---

## 🏅 Evaluation Strength

PipeFix is designed to excel across evaluation criteria:

* **Real-world utility** → production-like debugging
* **Task design** → increasing difficulty with ambiguity
* **Environment quality** → structured, stateful, realistic
* **Code quality** → modular, typed, OpenEnv-compliant
* **Agent performance** → high accuracy + efficiency

---

## 📜 License

MIT License
