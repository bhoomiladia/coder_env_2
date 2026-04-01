---
title: CodeDebtRefactor
emoji: 🔧
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# CodeDebtRefactor — RL Environment for Technical Debt Remediation

An [OpenEnv](https://github.com/deepfabric/openenv)-compatible reinforcement learning environment that trains AI agents to automatically fix technical debt in Python codebases.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Start the server
server
# → http://localhost:7860/docs
```

## Environment Overview

| Feature | Details |
|---|---|
| **Actions** | `WRITE_FILE`, `READ_FILE`, `RUN_LINTER`, `RUN_TESTS`, `LIST_FILES` |
| **Reward** | Multi-signal: syntax (0.15) + security (0.25) + logic (0.20) + similarity (0.40) |
| **Tasks** | 9 tasks across 3 difficulty levels (easy / medium / hard) |
| **Categories** | Syntax errors, logic bugs, security vulnerabilities |

## Diagnostics

The environment returns structured diagnostics for every observation:

```json
{
  "diagnostics": [
    {"file": "db.py", "line": "6", "severity": "high", "category": "security", "message": "SQL injection risk..."}
  ],
  "reward_breakdown": {"syntax": 1.0, "security": 0.0, "logic": 0.5, "similarity": 0.3}
}
```

## Docker

```bash
docker build -t code-debt-refactor .
docker run -p 7860:7860 code-debt-refactor
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
