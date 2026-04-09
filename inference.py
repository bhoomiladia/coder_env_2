"""
OpenEnv Inference Script for CodeDebtRefactor environment.
Produces structured [START]/[STEP]/[END] output on stdout for evaluation.
"""

import os
import sys
import json
import textwrap
import traceback
from pathlib import Path

# Force unbuffered stdout to ensure the validator sees every print immediately
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# ── Safe optional imports ──
try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    load_dotenv(".env")
except ImportError:
    pass

# ── Configuration ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME  = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN    = os.getenv("HF_TOKEN")
ENV_URL     = os.getenv("ENV_URL", "https://bhoomiladia-coder-env.hf.space")

MAX_STEPS   = 10
TEMPERATURE = 0.2
MAX_TOKENS  = 1500

SYSTEM_PROMPT = textwrap.dedent("""
    You are a Senior DevOps Agent. Your goal is to modernize and secure a broken repository.
    You will receive diagnostic logs and code views.

    You must reply with exactly one JSON object representing your action.
    Do not output markdown code blocks (e.g., no ```json), just the raw JSON.

    Available commands in the JSON:
    {
      "command": "READ_FILE",
      "target_file": "db.py",
      "patch_content": "",
      "explanation": "Checking what is in the file."
    }

    Valid commands: "WRITE_FILE", "READ_FILE", "RUN_LINTER", "RUN_TESTS", "LIST_FILES".
    For WRITE_FILE, set `patch_content` to the full new file content.
""").strip()


# ── Helpers ──

def log(msg: str):
    """Print to stdout and flush immediately."""
    print(msg, flush=True)


def clamp_score(value) -> float:
    """Clamp a value to strictly within (0, 1) — never 0.0 or 1.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.01
    if v != v:  # NaN check
        return 0.01
    return max(0.01, min(0.99, v))


def load_tasks() -> list:
    """Load task metadata from tasks.json."""
    tasks_path = Path(__file__).parent / "tasks.json"
    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"Warning: Could not load tasks.json: {e}")
        return []


def parse_model_action(response_text: str) -> dict:
    try:
        text = response_text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception:
        return {
            "command": "LIST_FILES",
            "target_file": "",
            "patch_content": "",
            "explanation": "Fallback due to parse error",
        }


def create_llm_client():
    if OpenAI is None:
        return None
    try:
        api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY") or "no-key"
        return OpenAI(base_url=API_BASE_URL, api_key=api_key)
    except Exception as e:
        log(f"Warning: Failed to create OpenAI client: {e}")
        return None


def get_llm_action(client, state_data: dict, step_num: int) -> dict:
    fallback = {"command": "LIST_FILES", "target_file": "", "patch_content": "", "explanation": "Fallback"}
    if client is None:
        return fallback
    try:
        obs_text = json.dumps(state_data.get('observation', {}), indent=2)
        user_content = [{"type": "text", "text": f"Step {step_num} Observation:\n{obs_text}"}]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return parse_model_action(completion.choices[0].message.content or "")
    except Exception as e:
        log(f"Warning: LLM generation failed: {e}")
        return fallback


def env_reset(task_id: int) -> dict | None:
    if requests is None:
        return None
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def env_step(action: dict) -> dict | None:
    if requests is None:
        return None
    try:
        resp = requests.post(f"{ENV_URL}/step", json={"action": action, "timeout_s": 30}, timeout=30)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# ── Main ──

def main():
    log(f"Starting inference with ENV_URL={ENV_URL}")
    tasks = load_tasks()
    client = create_llm_client()

    # Determine tasks to evaluate
    if not tasks:
        # Fallback if tasks.json is completely missing
        task_list = [{"task_id": i, "name": f"task_{i}"} for i in range(9)]
    else:
        task_list = tasks
        
    for task_info in task_list:
        task_id = task_info.get("task_id", 0)
        # Handle cases where name might be missing (use task_id fallback)
        task_name = task_info.get("name", f"task_{task_id}")
        
        log(f"\n--- Evaluating {task_name} (ID: {task_id}) ---")

        # 1. Start execution logging block
        log(f"[START] task={task_name}")

        state_data = env_reset(task_id)
        if state_data is None:
            log(f"  Failed to reset environment. Marking task complete with floor score.")
            log(f"[END] task={task_name} score=0.01 steps=0")
            continue

        final_reward = 0.01
        steps_taken = 0

        # 2. Step execution logging block
        for step in range(1, MAX_STEPS + 1):
            action_dict = get_llm_action(client, state_data, step)
            log(f"  Step {step}: Agent action -> {action_dict.get('command')}")

            step_result = env_step(action_dict)
            if step_result is None:
                log("  Environment step failed. Aborting task.")
                steps_taken = step
                break

            state_data = step_result
            reward = clamp_score(state_data.get("reward"))
            final_reward = reward
            steps_taken = step

            log(f"[STEP] step={step} reward={reward}")

            if state_data.get("done"):
                log(f"  Task Complete! Final Reward: {reward}")
                break

        # 3. Final score logging block
        final_reward = clamp_score(final_reward)
        log(f"[END] task={task_name} score={final_reward} steps={steps_taken}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stdout)
        log(f"FATAL ERROR in main(): {exc}")
        
        # Absolute guarantee that structured output format is preserved even on crash
        fallback_tasks = load_tasks() or [{"task_id": i, "name": f"task_{i}"} for i in range(9)]
        for t in fallback_tasks:
            tname = t.get("name", f"task_{t.get('task_id', 0)}")
            log(f"[START] task={tname}")
            log(f"[STEP] step=1 reward=0.01")
            log(f"[END] task={tname} score=0.01 steps=0")
