import os
import json
import textwrap
from typing import Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env or .env.local
load_dotenv(".env.local")
load_dotenv(".env")

# LLM Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# OpenEnv Environment Configuration
ENV_URL = os.getenv("ENV_URL", "https://bhoomiladia-coder-env.hf.space")

MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 1500

SYSTEM_PROMPT = textwrap.dedent("""
    You are a Senior DevOps Agent. Your goal is to modernize and secure a broken repository.
    You will receive diagnostic logs and code views.

    You must reply with exactly one JSON object representing your action. Do not output markdown code blocks (e.g., no ```json), just the raw JSON.

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


def parse_model_action(response_text: str) -> dict:
    """Extract JSON action from model response."""
    try:
        # Strip potential markdown formatting
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse JSON. Falling back. Response was:\n{response_text}")
        return {"command": "LIST_FILES", "target_file": "", "patch_content": "", "explanation": "Fallback due to JSON parse error."}


def main():
    if not HF_TOKEN:
        print("Warning: HF_TOKEN is not set. Assuming local/mock LLM server or it will fail.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    print(f"Connecting to environment at {ENV_URL}...")
    
    # Evaluate all 9 tasks available in the environment
    for task_id in range(9):
        print(f"\n--- Evaluating Task {task_id} ---")
        # 1. Reset Environment
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        if reset_resp.status_code != 200:
            print(f"Failed to reset environment: {reset_resp.text}")
            continue
            
        state_data = reset_resp.json()
        obs_data = state_data.get('observation', {})
        print("START")

        for step in range(1, MAX_STEPS + 1):
            print("STEP")
            # Format the observation for the LLM
            obs_text = json.dumps(state_data, indent=2)
            user_content = [{"type": "text", "text": f"Step {step} Observation:\n{obs_text}"}]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            try:
                # 2. Get LLM Action
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                response_text = completion.choices[0].message.content or ""
                action_dict = parse_model_action(response_text)
                
                print(f"Step {step}: Agent suggested -> {action_dict.get('command')} on {action_dict.get('target_file')}")
                
                # 3. Take Step in Environment
                step_payload = {"action": action_dict, "timeout_s": 30}
                step_resp = requests.post(f"{ENV_URL}/step", json=step_payload)
                
                if step_resp.status_code != 200:
                    print(f"Failed step: {step_resp.text}")
                    break
                    
                state_data = step_resp.json()
                obs_data = state_data.get('observation', {})
                reward = state_data.get("reward", 0.0)
                
                if state_data.get("done"):
                    print(f"\nTask Complete! Final Reward: {reward}")
                    print("Final Output:", obs_data.get("terminal_output"))
                    break
                    
            except Exception as e:
                print(f"Error during inference: {e}")
                break

        print("END")

if __name__ == "__main__":
    main()
