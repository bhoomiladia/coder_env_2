---
title: CodeDebtRefactor
emoji: 🔧
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# CodeDebtRefactor

**An OpenEnv-compliant reinforcement learning environment for testing and training AI agents on automated code reviews, syntax debugging, and security remediation.**

---

## Overview

The **CodeDebtRefactor** environment simulates a broken Python project directory, exposing an AI agent to varying levels of technical debt. Agents iteratively read files, run linters, test modifications, and patch code until the environment issues a `Reward: 1.0` — indicating a complete and secure fix.

### Key Features

- **OpenEnv Compliant** — Exposes standard REST endpoints (`/reset`, `/step`) with Pydantic Action/Observation schemas.
- **Virtual Filesystem** — In-memory file system lets agents safely read, write, and execute Python scripts.
- **Multi-Signal Rewards** — Patches are graded across 4 metrics: Syntax (0.15), Security (0.25), Logic (0.20), and Similarity (0.40).
- **Structured Logging** — Built-in `START` / `STEP` / `END` stdout markers for OpenEnv lifecycle tracing.

---

## Task Dataset

9 tasks are defined in `tasks.json` across 3 difficulty tiers:

| Difficulty | Category | Example Challenge |
|------------|----------|-------------------|
| Easy       | Syntax   | Fix a missing colon in a function definition |
| Medium     | Logic    | Refactor infinite loops and `IndexError` bugs |
| Hard       | Security | Migrate MD5 to SHA256, sanitize SQL injections |

---

## Environment API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST   | Initialize a task. Body: `{"task_id": 0}` |
| `/step`  | POST   | Submit an action. Body: `{"action": {...}, "timeout_s": 30}` |
| `/docs`  | GET    | OpenAPI documentation (Swagger UI) |

### Available Agent Commands

| Command | Description |
|---------|-------------|
| `READ_FILE` | Read contents of a file in the virtual directory |
| `WRITE_FILE` | Write/overwrite a file with patched content |
| `RUN_LINTER` | Run syntax and style checks on a file |
| `RUN_TESTS` | Execute test suite against the current state |
| `LIST_FILES` | List all files in the virtual project directory |

---

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Start the Server

```bash
server
```

The API boots at `http://localhost:7860`. Docs are available at `http://localhost:7860/docs`.

### 3. Configure Environment Variables

Create a `.env.local` file with:

```ini
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
HF_TOKEN=<your-api-token>
```

| Variable | Default | Required |
|----------|---------|----------|
| `API_BASE_URL` | `https://api.openai.com/v1` | No |
| `MODEL_NAME` | `gpt-4o-mini` | No |
| `HF_TOKEN` | — | **Yes** |

### 4. Run Inference

```bash
python inference.py
```

This runs the baseline agent across all 9 tasks automatically and outputs structured logs:

```
Connecting to environment at http://localhost:7860...

--- Evaluating Task 0 ---
START
STEP
Step 1: Agent suggested -> WRITE_FILE on app.py

Task Complete! Final Reward: 1.0
END
```

---

## Validation

Before submitting, validate your environment meets OpenEnv specifications:

```bash
bash validate-submission.sh "http://localhost:7860"
```

Or directly via the CLI:

```bash
openenv validate --url "http://localhost:7860"
```

---

## Reward Breakdown

Each step returns a detailed diagnostic payload:

```json
{
  "diagnostics": [
    {
      "file": "db.py",
      "line": "6",
      "severity": "high",
      "category": "security",
      "message": "SQL injection risk detected"
    }
  ],
  "reward_breakdown": {
    "syntax": 1.0,
    "security": 0.0,
    "logic": 0.5,
    "similarity": 0.3
  }
}
```

---

## Project Structure

```
coder_env_2/
├── server/
│   ├── app.py              # FastAPI entry point
│   └── environment.py      # Core RL environment logic
├── models.py               # Pydantic Action/Observation schemas
├── tasks.json              # 9 task definitions (easy/medium/hard)
├── inference.py            # Baseline agent with START/STEP/END logging
├── validate-submission.sh  # OpenEnv compliance checker
├── Dockerfile              # Container deployment config
├── pyproject.toml          # Package and dependency definitions
└── .env.local              # Local environment variables (not committed)
```

---

## Docker

```bash
docker build -t code-debt-refactor .
docker run -p 7860:7860 code-debt-refactor
```

---

## Links

- [OpenEnv Framework](https://github.com/deepfabric/openenv)
- [Hugging Face Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference)
